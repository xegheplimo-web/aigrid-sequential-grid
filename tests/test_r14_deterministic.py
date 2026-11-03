"""R14 tests — deterministic plan builder (``aigrid/plans/deterministic.py``).

Card R14 §1 + D-093 + addendum v3/v4: a closed-candle -> REGIME check ->
deterministic plan chain with NO LLM on the path. Regime RANGE only;
UPTREND/DOWNTREND/HIGH_VOLATILITY (or a paused controller) -> ``None`` +
``PLAN_SKIPPED_REGIME``. Deterministic means deterministic: same input ->
byte-identical plan (``created_at``/``expires_at`` derive from ``as_of``,
the last closed candle — even ``plan_id`` is a stable input hash).

MinSpacing (D-099/D-100 v3): ``spacing = max(0.30% dry-run floor,
0.75 x ATR5m p70 measured, 3.0 x C_stress)`` with
``C_stress = maker-maker roundtrip fees + AS p85 reserve``; funding budget
is tracked separately, never inside spacing. If the level step implied by
the 20-candle reference range would clamp below the floor -> ``None``
(never narrow spacing to trade more).

All tests hermetic: synthetic OHLCV frames, a list-backed eventlog fake.
"""

from __future__ import annotations

import ast
import inspect
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from aigrid.plans.deterministic import (
    C_STRESS_K,
    DRY_RUN_SPACING_FLOOR,
    PLAN_TTL_SECONDS,
    atr_p70_pct,
    build_deterministic_plan,
    spacing_floor,
)
from aigrid.settings import AIGridSettings

ROOT = Path(__file__).resolve().parents[1]
PAIR = "ETH/USDT:USDT"
AS_OF = datetime(2026, 11, 3, 12, 0, 0, tzinfo=UTC)


class _Events:
    """EventLog duck: collects (kind, symbol, payload)."""

    def __init__(self):
        self.rows: list[tuple[str, str, dict]] = []

    def log(self, kind: str, symbol: str = "", payload: dict | None = None):
        self.rows.append((kind, symbol, payload or {}))

    def kinds(self) -> list[str]:
        return [k for k, _, _ in self.rows]


def _settings(**kw) -> AIGridSettings:
    base = {
        "enabled": True,
        "mode": "dryrun",
        "symbols": (PAIR,),
        "timeframe": "5m",
        "data_dir": ROOT / "user_data/data/binance/futures",
    }
    base.update(kw)
    return AIGridSettings(**base)


def _ohlcv(closes: np.ndarray, *, vol: float = 1e4, spread: float = 0.002) -> pd.DataFrame:
    """Closed 5m candles; high/low bracket each close by `spread`."""
    n = len(closes)
    idx = pd.date_range("2026-10-31", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * (1 + spread),
            "low": closes * (1 - spread),
            "close": closes,
            "volume": np.full(n, vol),
        },
        index=idx,
    )


def _range_df(n: int = 600, amp: float = 0.02) -> pd.DataFrame:
    """A RANGE-shaped series: closes oscillate ±`amp` around 100."""
    x = np.arange(n)
    closes = 100.0 * (1 + amp * np.sin(x / 7.0))
    return _ohlcv(closes)


def _flat_df(n: int = 600, amp: float = 0.0004) -> pd.DataFrame:
    """Nearly flat series — level step falls under the spacing floor."""
    closes = 100.0 * (1 + amp * np.sin(np.arange(n) / 5.0))
    return _ohlcv(closes, spread=0.0002)


def _build(df=None, **kw) -> object:
    args = {
        "regime": "RANGE",
        "settings": _settings(),
        "as_of": AS_OF,
        "ai_status": "UNAVAILABLE",
        "price_tick": 0.01,
    }
    args.update(kw)
    return build_deterministic_plan(PAIR, df if df is not None else _range_df(), **args)


# =============================================================== determinism
def test_deterministic_same_input_same_plan():
    """Same input -> byte-identical plan dict (plan_id is an input hash;
    created_at/expires_at derive from as_of, never from wall clock)."""
    df = _range_df()
    p1 = _build(df)
    p2 = _build(df)
    assert p1 is not None and p2 is not None
    d1 = p1.model_dump(mode="json")
    d2 = p2.model_dump(mode="json")
    assert d1 == d2
    assert p1.plan_id == p2.plan_id


def test_plan_id_is_a_stable_input_hash():
    """plan_id repeats for identical input and changes when the market
    window changes — the restart-duplicate key is meaningful."""
    p1 = _build(_range_df())
    p2 = _build(_range_df(amp=0.03))
    assert p1.plan_id.startswith("plan-") and len(p1.plan_id) >= 12
    assert p1.plan_id != p2.plan_id


# =============================================================== regime gate
@pytest.mark.parametrize("regime", ["UPTREND", "DOWNTREND", "HIGH_VOLATILITY"])
def test_regime_trending_yields_no_plan(regime):
    """TRENDING / HIGH volatility regimes never produce a plan."""
    ev = _Events()
    plan = _build(regime=regime, eventlog=ev)
    assert plan is None
    assert "PLAN_SKIPPED_REGIME" in ev.kinds()
    payload = ev.rows[-1][2]
    assert payload.get("regime") == regime


def test_paused_controller_yields_no_plan():
    """A paused controller (INITIAL_SAFE_HOLD, unconfirmed regime) fails
    closed — no plan even when the raw classification reads RANGE."""
    ev = _Events()
    plan = _build(paused=True, pause_reason="INITIAL_SAFE_HOLD", eventlog=ev)
    assert plan is None
    assert "PLAN_SKIPPED_REGIME" in ev.kinds()


def test_ai_down_statuses_still_plan():
    """ai_status OK / UNAVAILABLE / OFF / NO_ANALYSIS_YET all permit the
    deterministic path — AI may never veto it."""
    for status in ("OK", "UNAVAILABLE", "OFF", "NO_ANALYSIS_YET"):
        assert _build(ai_status=status) is not None, status


def test_ai_breaker_open_yields_no_plan():
    ev = _Events()
    assert _build(ai_status="BREAKER_OPEN", eventlog=ev) is None
    assert ev.kinds()


# =============================================================== MinSpacing (v3)
def test_spacing_floor_not_violated():
    """grid_spacing is the real step and respects the MinSpacing floor;
    C_stress = maker roundtrip (2x fee_rate_maker) + AS p85 reserve."""
    df = _range_df()
    plan = _build(df)
    assert plan is not None
    atr_p70 = atr_p70_pct(df)
    c_stress = 2 * 0.0002 + 0.00015  # maker 0.02% x2 + AS p85 0.015%
    expected = max(DRY_RUN_SPACING_FLOOR, 0.75 * atr_p70, C_STRESS_K * c_stress)
    real_step = (
        (plan.grid_upper - plan.grid_lower) / (plan.grid_levels - 1) / float(df["close"].iloc[-1])
    )
    assert plan.grid_spacing == pytest.approx(real_step, abs=1e-6)
    assert plan.grid_spacing >= expected
    assert plan.grid_spacing >= DRY_RUN_SPACING_FLOOR


def test_atr_term_dominates_when_vol_wide():
    """A wide-but-ranged market pushes 0.75 x ATR p70 above the 0.30%
    floor — the measured term wins (floor is a minimum, not a cap)."""
    df = _range_df(amp=0.04)
    atr_p70 = atr_p70_pct(df)
    assert 0.75 * atr_p70 > DRY_RUN_SPACING_FLOOR  # precondition: atr term wins
    plan = _build(df)
    assert plan is not None
    assert plan.grid_spacing >= 0.75 * atr_p70

    narrow = df.copy()
    narrow.iloc[-20:] = _flat_df().iloc[-20:].to_numpy()
    assert 0.75 * atr_p70_pct(narrow) > DRY_RUN_SPACING_FLOOR
    assert _build(narrow) is None


def test_spacing_floor_components_reported():
    """spacing_floor exposes c_stress / atr_p70 / funding budget separately
    — funding is tracked apart, never folded into spacing."""
    df = _range_df()
    info = spacing_floor(df, maker_roundtrip=0.0004, as_p85=0.00015)
    assert info["c_stress"] == pytest.approx(0.00055)
    assert info["atr_p70"] == pytest.approx(atr_p70_pct(df))
    assert info["spacing_min"] >= DRY_RUN_SPACING_FLOOR
    assert "funding_budget" in info  # tracked separately, not in spacing


def test_slop_clamp_to_none_when_below_floor():
    """A 20-candle range too tight for `grid_levels` would force the level
    step under the floor -> None. Spacing is NEVER narrowed to trade more."""
    plan = _build(_flat_df())
    assert plan is None


def test_grid_uses_last_20_candle_range():
    """grid_upper = max(high of last 20 closed), grid_lower = min(low)."""
    df = _range_df()
    plan = _build(df)
    ref = df.iloc[-20:]
    # bounds are the ref20 envelope, only expanded outwards to land on tick
    assert plan.grid_upper >= float(ref["high"].max()) - 1e-9
    assert plan.grid_lower <= float(ref["low"].min()) + 1e-9
    assert plan.grid_upper <= float(ref["high"].max()) + 0.2  # tiny tick slack
    assert plan.grid_lower >= float(ref["low"].min()) - 0.2


# =============================================================== plan shape
def test_ttl_is_ten_minutes():
    plan = _build()
    assert plan.ttl_seconds() == pytest.approx(PLAN_TTL_SECONDS)
    assert plan.ttl_seconds() == 600


def test_created_at_is_as_of_not_wall_clock():
    plan = _build()
    assert plan.created_at == AS_OF
    assert plan.expires_at == AS_OF + timedelta(seconds=600)


def test_stop_loss_three_pct_below_grid_lower():
    plan = _build()
    assert plan.stop_loss == pytest.approx(plan.grid_lower * 0.97, rel=1e-9)
    assert plan.stop_loss < plan.grid_lower


def test_levels_on_tick_when_tick_given():
    """Every published level price must be a tick multiple — the exchange
    rejects off-tick limits (validator LEVEL_PRICE_NOT_ON_TICK)."""
    plan = _build(price_tick=0.05)
    for price in plan.level_prices():
        ratio = price / 0.05
        assert abs(ratio - round(ratio)) < 1e-6, price


def test_capital_and_leverage_clamps():
    """max_leverage is pinned 1.0 (D-078); margin <= per-symbol cap and <=
    notional limit; level stakes fit inside max_capital."""
    settings = _settings()
    plan = _build(settings=settings)
    cap = settings.capital
    assert plan.max_leverage == 1.0
    assert plan.max_capital <= cap.per_symbol_margin_cap + 1e-9
    assert plan.max_capital <= settings.risk.max_position_notional_usdt + 1e-9
    assert plan.max_position_size == pytest.approx(plan.max_capital)  # lev 1
    assert plan.level_stake * plan.grid_levels <= plan.max_capital + 1e-6
    assert plan.level_stake >= cap.min_level_margin_usdt - 1e-9
    assert plan.level_notional == pytest.approx(plan.level_stake)


def test_grid_levels_from_settings_default_4():
    plan = _build()
    assert plan.grid_levels == 4
    assert len(plan.level_prices()) == 4


def test_market_regime_and_mode_are_range():
    plan = _build()
    assert plan.market_regime == "RANGE"
    assert str(plan.grid_mode) == "RANGE_GRID"


# =============================================================== boundaries
def test_no_llm_or_remote_imports_on_plan_path():
    """The deterministic module must not import the LLM adapter or any
    non-stdlib HTTP client — the publish critical path stays local."""
    import aigrid.plans.deterministic as det

    tree = ast.parse(inspect.getsource(det))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    banned = {"ta_adapter", "openrouter", "anthropic", "openai", "requests", "ccxt"}
    assert not (imported & banned), imported & banned


def test_plan_serializes_plain_dict_json():
    """Plan must be expressible as a plain JSON dict (schema contract)."""
    import json

    plan = _build()
    blob = json.dumps(plan.model_dump(mode="json"), sort_keys=True)
    assert isinstance(blob, str) and len(blob) > 50
