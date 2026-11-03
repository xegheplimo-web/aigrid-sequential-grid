"""R1 regression tests — P0-4 fail-safe stoploss + P0-3 stake unit (T-201/T-202).

The Freqtrade strategy module is imported with stubbed freqtrade/talib (those
packages only exist inside the dry-run container). Tests cover the pure
function `resolve_stoploss_pct`, the strategy hooks via dict payloads, and the
orchestrator's `_write_ft_integration` output — never the freqtrade runtime.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from aigrid.orchestrator import AIGridOrchestrator
from aigrid.plans.schema import AIGridPlan, GridMode, draft_plan_from_assessment
from aigrid.settings import AIGridSettings
from tests._attest import make_config, write_integration

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "user_data/data/binance/futures"
STRATEGY_PATH = ROOT / "user_data/strategies/AIGridStrategy.py"
PAIR = "ETH/USDT:USDT"
NOW = datetime.now(UTC)


# ------------------------------------------------------------------ strategy import helpers
def _real_stoploss_from_absolute(stop_rate, current_rate, is_short=False, leverage=1.0):
    """Mirror of freqtrade.strategy.stoploss_from_absolute (FT 2026.10):
    returns the POSITIVE risk distance relative to current_rate."""
    if current_rate == 0:
        return 1
    stoploss = 1 - (stop_rate / current_rate)
    if is_short:
        stoploss = -stoploss
    return max(stoploss, 0.0) * leverage


def _install_ft_stubs() -> None:
    """Minimal freqtrade/talib surface so AIGridStrategy.py imports."""
    if "freqtrade" in sys.modules:
        strat = sys.modules["freqtrade.strategy"]
        if not hasattr(strat, "stoploss_from_absolute"):
            strat.stoploss_from_absolute = _real_stoploss_from_absolute
        return
    freqtrade = types.ModuleType("freqtrade")
    persistence = types.ModuleType("freqtrade.persistence")
    strategy = types.ModuleType("freqtrade.strategy")

    class Trade: ...

    class IStrategy:
        def __init__(self, *a, **k): ...

    persistence.Trade = Trade
    strategy.IStrategy = IStrategy
    strategy.stoploss_from_absolute = _real_stoploss_from_absolute
    freqtrade.persistence = persistence
    freqtrade.strategy = strategy
    talib = types.ModuleType("talib")
    talib_abstract = types.ModuleType("talib.abstract")
    talib_abstract.EMA = lambda *a, **k: None
    talib.abstract = talib_abstract
    sys.modules.update(
        {
            "freqtrade": freqtrade,
            "freqtrade.persistence": persistence,
            "freqtrade.strategy": strategy,
            "talib": talib,
            "talib.abstract": talib_abstract,
        }
    )


@pytest.fixture(scope="module")
def strat_mod():
    _install_ft_stubs()
    spec = importlib.util.spec_from_file_location("aigrid_ft_strategy", STRATEGY_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _strategy(strat_mod):
    """Bare strategy instance — __init__ only sets attrs, no freqtrade needed."""
    s = object.__new__(strat_mod.AIGridStrategy)
    s.config = make_config()
    s._aigrid = {}
    s._plan = {}
    s._levels = []
    s._sl_cache = {}
    s._last_sl_pct = None
    return s


def _trade(open_rate=100.0, is_short=False):
    return types.SimpleNamespace(is_short=is_short, open_rate=open_rate)


def _write_integration(path: Path, plans: dict, enabled=True, verified=True) -> None:
    path.write_text(
        json.dumps(
            {
                "ts_ms": int(time.time() * 1000),
                "enabled": enabled,
                "dry_run_verified": verified,
                "mode": "dryrun",
                "plans": plans,
                "filled_levels": {},
                "filled_level_prices": {p: [] for p in plans},
            }
        ),
        encoding="utf-8",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    write_integration(path.parent, make_config(), **payload)


# ====================================================================== T-201: stoploss
def test_resolve_sl_valid_plan_long(strat_mod):
    pct = strat_mod.resolve_stoploss_pct({"stop_loss": 97.0}, None, is_short=False, open_rate=100.0)
    assert pct == pytest.approx(-0.03)


def test_resolve_sl_valid_plan_short(strat_mod):
    pct = strat_mod.resolve_stoploss_pct({"stop_loss": 103.0}, None, is_short=True, open_rate=100.0)
    assert pct == pytest.approx(-0.03)


def test_resolve_sl_no_plan_uses_cache(strat_mod):
    assert strat_mod.resolve_stoploss_pct({}, -0.023, False, 100.0) == pytest.approx(-0.023)
    assert strat_mod.resolve_stoploss_pct(
        {"stop_loss": None}, -0.045, True, 100.0
    ) == pytest.approx(-0.045)


def test_resolve_sl_no_plan_no_cache_returns_default(strat_mod):
    assert strat_mod.resolve_stoploss_pct({}, None, False, 100.0) == -0.10


def test_resolve_sl_never_returns_minus_99(strat_mod):
    cases = [
        ({}, None, False, 100.0),
        ({}, -0.05, False, 100.0),
        ({"stop_loss": 97.0}, None, False, 100.0),
        ({"stop_loss": 103.0}, None, True, 100.0),
        ({"stop_loss": 0.5}, None, False, 100.0),  # degenerate: >99% away
        ({"stop_loss": 1e9}, None, False, 100.0),  # SL above entry (long)
        ({"stop_loss": -5.0}, None, False, 100.0),  # invalid price
        ({}, -0.995, False, 100.0),  # degenerate cached value
    ]
    for plan, cache, short, rate in cases:
        pct = strat_mod.resolve_stoploss_pct(plan, cache, short, rate)
        assert pct is not None
        assert -0.99 < pct <= -0.01, (plan, cache, short, rate, pct)


def test_static_stoploss_is_hard_floor(strat_mod):
    assert strat_mod.AIGridStrategy.stoploss == -0.10
    assert strat_mod.SAFE_DEFAULT_SL_PCT == -0.10


def test_custom_stoploss_absent_file_returns_default(strat_mod, monkeypatch, tmp_path):
    monkeypatch.setattr(strat_mod, "INTEGRATION_FILE", tmp_path / "absent.json")
    monkeypatch.setattr(strat_mod, "SL_CACHE_FILE", tmp_path / "no_cache.json")
    s = _strategy(strat_mod)
    pct = s.custom_stoploss(PAIR, _trade(100.0), None, 99.0, -0.01, False)
    assert pct is not None and pct != -0.99
    # R7: the -10% fallback anchors at open*0.90 = 90 — at current=99 the
    # ratio is the distance from 99 to 90 (not a raw -0.10).
    assert pct == pytest.approx(-(1.0 - 90.0 / 99.0))


def test_custom_stoploss_caches_and_survives_dead_orchestrator(strat_mod, monkeypatch, tmp_path):
    integ = tmp_path / "ft_integration.json"
    cache = tmp_path / "strategy_sl_cache.json"
    monkeypatch.setattr(strat_mod, "INTEGRATION_FILE", integ)
    monkeypatch.setattr(strat_mod, "SL_CACHE_FILE", cache)
    _write_integration(
        integ,
        {
            PAIR: {
                "stop_loss": 97.0,
                "level_stake": 20.0,
                "grid_mode": "RANGE_GRID",
                "grid_levels_prices": [98.0, 99.0],
            }
        },
    )
    s = _strategy(strat_mod)
    pct = s.custom_stoploss(PAIR, _trade(100.0), None, 99.0, -0.01, False)
    # R7: stop anchored at the absolute plan price 97 — at current=99 the
    # ratio is the distance from 99 to 97 (not -0.03 from open_rate).
    assert pct == pytest.approx(-(1.0 - 97.0 / 99.0))
    # the plan SL was persisted per-pair as an ABSOLUTE price
    assert cache.exists()
    assert json.loads(cache.read_text())[PAIR] == pytest.approx(97.0)
    # orchestrator dies -> file vanishes -> a fresh strategy falls back to cache
    integ.unlink()
    s2 = _strategy(strat_mod)
    pct2 = s2.custom_stoploss(PAIR, _trade(100.0), None, 99.0, -0.01, False)
    assert pct2 == pytest.approx(-(1.0 - 97.0 / 99.0))
    assert pct2 != -0.99


def test_custom_stoploss_disabled_integration_uses_cache(strat_mod, monkeypatch, tmp_path):
    integ = tmp_path / "ft_integration.json"
    cache = tmp_path / "strategy_sl_cache.json"
    integ.write_text(json.dumps({"enabled": False, "dry_run_verified": False, "plans": {}}))
    # R7: the cache stores the last plan SL as an ABSOLUTE price
    cache.write_text(json.dumps({PAIR: 96.0}))
    monkeypatch.setattr(strat_mod, "INTEGRATION_FILE", integ)
    monkeypatch.setattr(strat_mod, "SL_CACHE_FILE", cache)
    s = _strategy(strat_mod)
    pct = s.custom_stoploss(PAIR, _trade(100.0), None, 99.0, -0.01, False)
    assert pct == pytest.approx(-(1.0 - 96.0 / 99.0))


# ====================================================================== T-202: stake unit
@pytest.fixture
def orch(tmp_path):
    settings = AIGridSettings(
        enabled=False,
        mode="shadow",
        symbols=(PAIR,),
        timeframe="15m",
        data_dir=DATA_DIR,
        state_dir=tmp_path / "state",
    )
    o = AIGridOrchestrator(settings)
    yield o
    o.close()


def _payload(orch) -> dict:
    target = orch.store.state_dir / "ft_integration.json"
    return json.loads(target.read_text(encoding="utf-8"))


def test_integration_level_stake_is_margin_not_notional(orch):
    plan = draft_plan_from_assessment(
        PAIR,
        "RANGE",
        0.8,
        3000.0,
        20.0,
        grid_mode="RANGE_GRID",
        ttl_minutes=45,
        margin_usdt=100.0,
        leverage=2.0,
        levels=5,
    )
    assert plan.level_stake == pytest.approx(20.0)  # 100/5 margin
    assert plan.level_notional == pytest.approx(40.0)  # 20 x 2 leverage
    orch.adopt_plan(plan)
    orch.persist()
    p = _payload(orch)["plans"][PAIR]
    assert p["level_stake"] == pytest.approx(plan.level_stake)
    assert p["level_stake"] != pytest.approx(plan.level_notional)
    assert p["level_stake"] * len(p["grid_levels_prices"]) <= plan.max_capital + 1e-6


@pytest.mark.parametrize("leverage", [1.0, 2.0, 3.0])
def test_level_notional_scales_with_leverage(leverage, orch):
    plan = draft_plan_from_assessment(
        PAIR,
        "RANGE",
        0.8,
        3000.0,
        20.0,
        grid_mode="RANGE_GRID",
        ttl_minutes=45,
        margin_usdt=120.0,
        leverage=leverage,
        levels=6,
    )
    assert plan.level_notional == pytest.approx(plan.level_stake * leverage)
    orch.adopt_plan(plan)
    orch.persist()
    p = _payload(orch)["plans"][PAIR]
    assert p["level_stake"] == pytest.approx(plan.level_stake)
    assert p["level_stake"] * len(p["grid_levels_prices"]) <= plan.max_capital + 1e-6


def test_integration_level_stake_clamped_to_max_capital(orch):
    """An inconsistent plan (level_stake * levels > max_capital) is clamped
    to max_capital / levels instead of silently over-allocating margin."""
    plan = AIGridPlan(
        plan_id="plan-r1clamp001",
        plan_version=1,
        symbol=PAIR,
        created_at=NOW,
        expires_at=NOW + timedelta(minutes=45),
        market_regime="RANGE",
        confidence=0.7,
        grid_mode=GridMode.RANGE_GRID,
        grid_lower=2900.0,
        grid_upper=3100.0,
        grid_levels=5,
        grid_spacing=0.01,
        max_capital=100.0,
        max_position_size=400.0,
        max_leverage=2.0,
        level_stake=40.0,
        level_notional=80.0,  # 40 x 5 = 200 > 100 cap
        stop_loss=2850.0,
        take_profit=None,
        invalidation_conditions=[],
        risk_level="MEDIUM",
        reasoning_summary="t",
    )
    orch.adopt_plan(plan)
    orch.persist()
    p = _payload(orch)["plans"][PAIR]
    assert p["level_stake"] == pytest.approx(20.0)  # clamped to 100/5
    assert p["level_stake"] * len(p["grid_levels_prices"]) <= plan.max_capital + 1e-6


def test_custom_stake_amount_returns_plan_margin(strat_mod, monkeypatch, tmp_path):
    integ = tmp_path / "ft_integration.json"
    monkeypatch.setattr(strat_mod, "INTEGRATION_FILE", integ)
    monkeypatch.setattr(strat_mod, "SL_CACHE_FILE", tmp_path / "cache.json")
    _write_integration(
        integ,
        {
            PAIR: {
                "level_stake": 20.0,
                "stop_loss": 97.0,
                "grid_mode": "RANGE_GRID",
                "grid_levels": 2,
                "grid_levels_prices": [98.0, 99.0],
                "max_capital": 1000.0,
                "max_position_size": 1000.0,
            }
        },
    )
    s = _strategy(strat_mod)
    stake = s.custom_stake_amount(PAIR, None, 99.0, 40.0, 10.0, 1000.0, 2.0, None, "long")
    assert stake == pytest.approx(20.0)


def test_custom_stake_amount_disabled_returns_none(strat_mod, monkeypatch, tmp_path):
    integ = tmp_path / "ft_integration.json"
    monkeypatch.setattr(strat_mod, "INTEGRATION_FILE", integ)
    integ.write_text(json.dumps({"enabled": False, "plans": {}}))
    s = _strategy(strat_mod)
    assert s.custom_stake_amount(PAIR, None, 99.0, 40.0, 10.0, 1000.0, 2.0, None, "long") is None
    # absent file -> also None
    integ.unlink()
    s2 = _strategy(strat_mod)
    assert s2.custom_stake_amount(PAIR, None, 99.0, 40.0, 10.0, 1000.0, 2.0, None, "long") is None
