"""R11 tests — engine<->strategy contract: index base, capital cap, pending.

Card R11 (P0-5b / P0-8 / P0-9):

- P0-5b: the engine numbers levels DESCENDING from ``grid_upper`` while the
  strategy sorts them ASCENDING, so ``filled_levels`` indices can never be
  compared across the wire. The orchestrator now publishes
  ``filled_level_prices: {pair: [price, ...]}`` (from Freqtrade order state
  — R12; previously ``engine.state.lots``);
  the strategy matches by PRICE (|delta| <= 1e-9) and fails CLOSED when the
  field is absent — an index is only a position, the price is the identity.
- P0-8: ``max_position_size`` (notional), ``max_capital`` (margin) and
  ``grid_levels`` (max_levels) are enforced before any stake is returned —
  in ``adjust_trade_position`` AND ``custom_stake_amount``. Missing cap
  fields fail closed; a breach returns None + rate-limited
  ``CAPITAL_CAP_BLOCKED``; the stake is never shrunk to fit.
- P0-9: FT 2026.10 gives the strategy exactly one supported way to drop an
  UNFILLED entry order on its own trade: ``check_entry_timeout() -> True``
  (checked each loop for every open entry order; ``unfilledtimeout`` config
  is evaluated first). Verified NOT usable: ``confirm_trade_entry`` (veto at
  placement only), ``adjust_trade_position`` (emits new orders, never
  cancels), ``dp.send_msg`` (notifications), ``self.wallets`` (balances).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import time
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from tests._attest import make_config, write_integration

ROOT = Path(__file__).resolve().parents[1]
STRATEGY_PATH = ROOT / "user_data/strategies/AIGridStrategy.py"
DATA_DIR = ROOT / "user_data/data/binance/futures"
PAIR = "ETH/USDT:USDT"


def _real_stoploss_from_absolute(stop_rate, current_rate, is_short=False, leverage=1.0):
    """Mirror of freqtrade.strategy.stoploss_from_absolute (FT 2026.10)."""
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
    spec = importlib.util.spec_from_file_location("aigrid_ft_strategy_r11", STRATEGY_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _strategy(strat_mod):
    """Bare strategy instance driven through the real bot_start() gate."""
    s = object.__new__(strat_mod.AIGridStrategy)
    s._aigrid = {}
    s._plan = {}
    s._levels = []
    s._sl_cache = {}
    s._last_sl_pct = None
    s._last_closed = {}
    s._plan_expires_ms = None
    s.config = make_config()
    s.bot_start()
    return s


def _trade(
    pair=PAIR,
    is_short=False,
    open_rate=100.0,
    amount=0.0,
    stake_amount=0.0,
    leverage=1.0,
):
    return types.SimpleNamespace(
        pair=pair,
        is_short=is_short,
        open_rate=open_rate,
        amount=amount,
        stake_amount=stake_amount,
        leverage=leverage,
        liquidation_price=None,
    )


def _write_integration(
    path: Path,
    plans: dict,
    filled: dict | None = None,
    filled_prices: dict | None = None,
    enabled=True,
    verified=True,
) -> None:
    """Write the integration payload. ``filled_level_prices`` defaults to the
    real publisher's shape (one key per plan, empty list) — tests that want
    the fail-closed path pass an explicit dict (e.g. ``{}``)."""
    path.write_text(
        json.dumps(
            {
                "ts_ms": int(time.time() * 1000),
                "enabled": enabled,
                "dry_run_verified": verified,
                "mode": "dryrun",
                "plans": plans,
                "filled_levels": filled or {},
                "filled_level_prices": (
                    filled_prices if filled_prices is not None else {p: [] for p in plans}
                ),
            }
        ),
        encoding="utf-8",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    write_integration(path.parent, make_config(), **payload)


def _entry_plan(**kw):
    """Plan with valid cap fields — tests overriding caps pass their own."""
    plan = {
        "plan_id": "plan-r11",
        "stop_loss": 90.0,
        "level_stake": 20.0,
        "grid_mode": "RANGE_GRID",
        "grid_upper": 110.0,
        "grid_lower": 90.0,
        "grid_levels": 4,
        "grid_levels_prices": [95.0, 96.0, 97.0, 98.0],
        "max_capital": 1000.0,
        "max_position_size": 1000.0,
        "expires_at": (datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
    }
    plan.update(kw)
    return plan


def _df(candles) -> pd.DataFrame:
    return pd.DataFrame(candles, columns=["open", "high", "low", "close"])


@pytest.fixture
def integ(strat_mod, monkeypatch, tmp_path):
    path = tmp_path / "ft_integration.json"
    monkeypatch.setattr(strat_mod, "INTEGRATION_FILE", path)
    monkeypatch.setattr(strat_mod, "SL_CACHE_FILE", tmp_path / "cache.json")
    return path


def _adjust(s, trade, current_rate=97.0):
    return s.adjust_trade_position(
        trade,
        None,
        current_rate,
        -0.03,
        10.0,
        1000.0,
        current_rate,
        current_rate,
        -0.03,
        -0.03,
    )


def _closed_candle(s, prev_close, o, h, low, c, pair=PAIR):
    """Feed the strategy a 2-candle df so _last_closed[pair] holds the last
    CLOSED candle (o/h/low/c) with prev_close as its crossing reference."""
    s.populate_indicators(
        _df(
            [
                (prev_close - 0.1, prev_close + 0.1, prev_close - 0.2, prev_close),
                (o, h, low, c),
            ]
        ),
        {"pair": pair},
    )


# ====================================================================== P0-5b: price-keyed filled levels
def test_filled_by_price_blocks_that_level(strat_mod, integ):
    """Engine fill at price 96 -> filled_level_prices=[96.0] -> a candle
    crossing 96 and 95 must NOT request 96 again; it requests the NEXT
    crossed level (95). Matching is by price, never by index."""
    _write_integration(integ, {PAIR: _entry_plan()}, filled_prices={PAIR: [96.0]})
    s = _strategy(strat_mod)
    trade = _trade()
    # prev_close 97 -> crossed: 96 (97>96>=94.5) and 95 (97>95>=94.5)
    _closed_candle(s, 97.0, 96.8, 97.0, 94.5, 95.0)
    assert _adjust(s, trade) == (20.0, "grid_level_0")  # 95.0, not 96.0
    # control: with an empty fill book the SAME candle requests 96 first
    _write_integration(integ, {PAIR: _entry_plan()})
    assert _adjust(s, trade) == (20.0, "grid_level_1")  # 96.0


def test_filled_by_price_blocks_single_crossed(strat_mod, integ):
    """Card-literal case: engine filled the level at price 95.0; a candle
    that only crossed 95 -> adjust_trade_position is None (it must not
    re-request the already-open level)."""
    _write_integration(integ, {PAIR: _entry_plan()}, filled_prices={PAIR: [95.0]})
    s = _strategy(strat_mod)
    # prev_close 96 -> only 95 crossed (96>95>=94.5; 96>96 is not >)
    _closed_candle(s, 96.0, 95.8, 96.0, 94.5, 95.0)
    assert _adjust(s, _trade()) is None


def test_missing_filled_prices_fails_closed(strat_mod, integ):
    """No filled_level_prices field at all -> no DCA (fail closed). The
    strategy must not fall back to the legacy index list."""
    _write_integration(integ, {PAIR: _entry_plan()}, filled={PAIR: [0]}, filled_prices={})
    s = _strategy(strat_mod)
    _closed_candle(s, 99.0, 98.8, 99.0, 94.5, 95.0)
    assert _adjust(s, _trade()) is None
    # legacy index present but price book absent for THIS pair -> still closed
    _write_integration(
        integ,
        {PAIR: _entry_plan()},
        filled_prices={"BTC/USDT:USDT": [96.0]},
    )
    assert _adjust(s, _trade()) is None


def test_filled_price_tolerance_1e9(strat_mod, integ):
    """Matching tolerance is 1e-9: within it the level is blocked, outside
    it is not."""
    s = _strategy(strat_mod)
    trade = _trade()
    _closed_candle(s, 97.0, 96.8, 97.0, 94.5, 95.0)
    _write_integration(integ, {PAIR: _entry_plan()}, filled_prices={PAIR: [96.0 + 5e-10]})
    assert _adjust(s, trade) == (20.0, "grid_level_0")  # 96 still blocked
    _write_integration(integ, {PAIR: _entry_plan()}, filled_prices={PAIR: [96.0 + 2e-9]})
    assert _adjust(s, trade) == (20.0, "grid_level_1")  # outside tolerance


def test_legacy_filled_levels_index_is_ignored(strat_mod, integ):
    """Regression for the index-base bug: engine's filled_levels=[1] meant
    'second level from grid_upper' (descending); the old code read it as
    ascending index 1. With the new contract the index field is left for
    old consumers but NEVER used by the strategy — an empty price book
    means nothing is blocked."""
    _write_integration(
        integ,
        {PAIR: _entry_plan()},
        filled={PAIR: [0, 1, 2, 3]},
        filled_prices={PAIR: []},
    )
    s = _strategy(strat_mod)
    _closed_candle(s, 99.0, 98.8, 99.0, 96.5, 97.0)
    # all 4 legacy indices "filled" but the price book is empty -> DCA runs
    assert _adjust(s, _trade()) == (20.0, "grid_level_3")


def test_index_base_round_trip(strat_mod, integ, tmp_path):
    """Real engine + real orchestrator serializer: engine fills the level at
    index 2 of the DESCENDING level list (price 106.0); the payload must
    carry filled_level_prices=[106.0] so the strategy blocks exactly that
    price (its own ascending index 3) and requests the next crossed level.

    The two cap fields (max_position_size/max_capital) are added to the
    serialized plan below because this card does not republish them — the
    P0-8 gate fails closed without them; everything else in the payload is
    verbatim from _write_ft_integration. R12: filled_level_prices comes
    from FT order state, so a fake reader reports a closed filled entry
    tagged for this plan at price 106.0."""
    from aigrid.orchestrator import AIGridOrchestrator
    from aigrid.plans.schema import AIGridPlan, GridMode
    from aigrid.settings import AIGridSettings

    class _StaticFillsReader:
        def fetch_open_trades(self):
            return [
                {
                    "pair": PAIR,
                    "orders": [
                        {
                            "ft_order_tag": "aigrid_plan-r11-roundtrip",
                            "ft_is_entry": True,
                            "status": "closed",
                            "filled": 0.5,
                            "price": 106.0,
                            "average": 106.0,
                        }
                    ],
                }
            ]

    settings = AIGridSettings(
        enabled=True,
        mode="shadow",
        symbols=(PAIR,),
        timeframe="5m",
        data_dir=DATA_DIR,
        state_dir=tmp_path / "state",
    )
    orch = AIGridOrchestrator(settings, ft_fills_reader=_StaticFillsReader())
    try:
        now = datetime.now(UTC)
        plan = AIGridPlan(
            plan_id="plan-r11-roundtrip",
            plan_version=1,
            symbol=PAIR,
            created_at=now,
            expires_at=now + timedelta(minutes=30),
            market_regime="RANGE",
            confidence=0.8,
            grid_mode=GridMode.RANGE_GRID,
            grid_lower=100.0,
            grid_upper=110.0,
            grid_levels=6,
            grid_spacing=0.02,
            max_capital=240.0,
            max_position_size=720.0,
            max_leverage=3.0,
            level_stake=20.0,
            level_notional=40.0,
            stop_loss=90.0,
        )
        # engine levels DESCEND from grid_upper: [110,108,106,104,102,100]
        assert plan.level_prices() == [110.0, 108.0, 106.0, 104.0, 102.0, 100.0]
        orch.adopt_plan(plan)
        engine = orch._engines[PAIR]
        # first candle: ref = open 107 -> crosses only 106 (level index 2)
        actions = engine.on_candle(1_700_000_000_000, 107.0, 107.0, 105.5, 106.0)
        assert [a.level for a in actions if a.kind == "BUY"] == [2]
        assert engine.state.filled_levels == {2}
        orch.persist()
        payload = json.loads(
            (settings.state_dir / "ft_integration.json").read_text(encoding="utf-8")
        )
        assert payload["filled_levels"][PAIR] == [2]  # engine index (kept)
        assert payload["filled_level_prices"][PAIR] == [106.0]  # price contract
    finally:
        orch.close()

    # hand the real serializer output to the strategy (+ the two cap fields
    # the publisher may not yet emit — P0-8 fails closed without them)
    payload["plans"][PAIR]["max_capital"] = 240.0
    payload["plans"][PAIR]["max_position_size"] = 720.0
    write_integration(integ.parent, make_config(), **payload)

    s = _strategy(strat_mod)
    # strategy levels ASCEND: [100,102,104,106,108,110]; blocked price 106.0
    # is its index 3 — an index compare would block ascending index 2 (104).
    _closed_candle(s, 107.0, 106.5, 107.0, 101.5, 104.0)
    # crossed: 106, 104, 102 -> 106 skipped by price -> returns 104.0
    res = _adjust(s, _trade(amount=0.05, stake_amount=5.2))
    assert res == (20.0, "grid_level_2")  # 104.0 — NOT 106.0


# ====================================================================== P0-8: capital caps
def test_capital_cap_blocks_adjust(strat_mod, integ, caplog):
    """current margin + proposed level_stake > max_capital -> None, and the
    block is logged once as CAPITAL_CAP_BLOCKED (rate-limited)."""
    _write_integration(
        integ,
        {PAIR: _entry_plan(max_capital=100.0, max_position_size=1000.0)},
    )
    s = _strategy(strat_mod)
    _closed_candle(s, 99.0, 98.8, 99.0, 96.5, 97.0)
    trade = _trade(amount=1.0, stake_amount=90.0)  # 90 + 20 > 100
    with caplog.at_level(logging.WARNING):
        assert _adjust(s, trade) is None
    assert any("CAPITAL_CAP_BLOCKED" in r.message for r in caplog.records)


def test_capital_cap_blocks_adjust_notional(strat_mod, integ):
    """Margin is under cap but current notional + proposed exceeds
    max_position_size -> None. Current notional = trade.amount *
    current_rate (mark-to-market, documented choice)."""
    _write_integration(
        integ,
        {PAIR: _entry_plan(max_capital=1000.0, max_position_size=200.0)},
    )
    s = _strategy(strat_mod)
    _closed_candle(s, 99.0, 98.8, 99.0, 96.5, 97.0)
    # amount*rate = 190; +20*1 = 210 > 200 while margin 50+20 <= 1000
    trade = _trade(amount=2.0, stake_amount=50.0, leverage=1.0)
    assert _adjust(s, trade, current_rate=95.0) is None


def test_capital_cap_allows_below_limit(strat_mod, integ):
    """Under every cap -> the crossed level's stake is returned."""
    _write_integration(
        integ,
        {PAIR: _entry_plan(max_capital=1000.0, max_position_size=1000.0)},
    )
    s = _strategy(strat_mod)
    _closed_candle(s, 99.0, 98.8, 99.0, 96.5, 97.0)
    trade = _trade(amount=1.0, stake_amount=50.0, leverage=1.0)
    assert _adjust(s, trade) == (20.0, "grid_level_3")


def test_missing_cap_fails_closed(strat_mod, integ):
    """Each missing cap field -> None in BOTH stake paths (the strategy
    never invents a limit the publisher did not send)."""
    base = _entry_plan()
    for field in ("max_position_size", "max_capital", "grid_levels"):
        plan = {k: v for k, v in base.items() if k != field}
        _write_integration(integ, {PAIR: plan})
        s = _strategy(strat_mod)
        _closed_candle(s, 99.0, 98.8, 99.0, 96.5, 97.0)
        assert _adjust(s, _trade(amount=1.0, stake_amount=50.0)) is None, field
        assert (
            s.custom_stake_amount(PAIR, None, 99.0, 40.0, 10.0, 1000.0, 1.0, None, "long") is None
        ), field


def test_max_levels_blocks_adjust(strat_mod, integ):
    """filled levels >= grid_levels -> None (max_levels cap)."""
    _write_integration(
        integ,
        {PAIR: _entry_plan(grid_levels=2)},
        filled_prices={PAIR: [95.0, 96.0]},
    )
    s = _strategy(strat_mod)
    _closed_candle(s, 99.0, 98.8, 99.0, 94.5, 95.0)
    assert _adjust(s, _trade()) is None


def test_capital_cap_blocks_custom_stake(strat_mod, integ):
    """First entry: level_stake itself may exceed max_capital, or its
    notional (stake x leverage) may exceed max_position_size -> None."""
    s = _strategy(strat_mod)
    _write_integration(integ, {PAIR: _entry_plan(max_capital=10.0, max_position_size=1000.0)})
    assert s.custom_stake_amount(PAIR, None, 99.0, 40.0, 10.0, 1000.0, 1.0, None, "long") is None
    _write_integration(integ, {PAIR: _entry_plan(max_capital=1000.0, max_position_size=30.0)})
    # 20 margin x leverage 2 = 40 notional > 30
    assert s.custom_stake_amount(PAIR, None, 99.0, 40.0, 10.0, 1000.0, 2.0, None, "long") is None
    # same caps, leverage 1 -> 20 <= 30 -> allowed
    assert s.custom_stake_amount(
        PAIR, None, 99.0, 40.0, 10.0, 1000.0, 1.0, None, "long"
    ) == pytest.approx(20.0)


def test_capital_cap_no_shrink_to_fit(strat_mod, integ):
    """Over the cap returns None — the stake is NEVER shrunk to fit."""
    _write_integration(integ, {PAIR: _entry_plan(max_capital=100.0, max_position_size=1000.0)})
    s = _strategy(strat_mod)
    _closed_candle(s, 99.0, 98.8, 99.0, 96.5, 97.0)
    trade = _trade(amount=1.0, stake_amount=95.0)  # 95+20 > 100, 5 would fit
    assert _adjust(s, trade) is None


# ====================================================================== P0-9: pending entry on expiry
def _order(tag="aigrid_plan-r11"):
    return types.SimpleNamespace(ft_order_tag=tag)


def test_expired_plan_cancels_pending_entry(strat_mod, integ):
    """check_entry_timeout is FT 2026.10's supported strategy-side cancel
    for unfilled entry orders on our own trade: an expired plan -> True.
    A still-valid plan -> False (the order lives on)."""
    plan = _entry_plan(expires_at=(datetime.now(UTC) - timedelta(minutes=5)).isoformat())
    _write_integration(integ, {PAIR: plan})
    s = _strategy(strat_mod)
    trade = _trade()
    assert s.check_entry_timeout(PAIR, trade, _order(), None) is True
    # a fresh plan keeps the pending entry alive
    _write_integration(integ, {PAIR: _entry_plan()})
    s2 = _strategy(strat_mod)
    assert s2.check_entry_timeout(PAIR, trade, _order(), None) is False


def test_disabled_or_missing_payload_cancels_pending(strat_mod, integ):
    """enabled=false, absent file, and halted entries all cancel the pending
    order — it must never become a NEW entry after the plan is dead."""
    _write_integration(integ, {PAIR: _entry_plan()}, enabled=False)
    s = _strategy(strat_mod)
    assert s.check_entry_timeout(PAIR, _trade(), _order(), None) is True
    integ.unlink()
    s2 = _strategy(strat_mod)
    assert s2.check_entry_timeout(PAIR, _trade(), _order(), None) is True
    _write_integration(integ, {PAIR: _entry_plan()})
    s3 = _strategy(strat_mod)
    s3._halted = True
    assert s3.check_entry_timeout(PAIR, _trade(), _order(), None) is True
    # SAFE_HOLD authorizes no entries — any pending entry is illegitimate
    _write_integration(integ, {PAIR: _entry_plan(grid_mode="SAFE_HOLD")})
    s4 = _strategy(strat_mod)
    assert s4.check_entry_timeout(PAIR, _trade(), _order(None), None) is True


def test_stale_plan_tag_cancels_pending(strat_mod, integ):
    """A pending order carrying a DIFFERENT plan's enter_tag belongs to a
    dead plan — cancelled; the current plan's own tag keeps the order."""
    _write_integration(integ, {PAIR: _entry_plan(plan_id="plan-r11")})
    s = _strategy(strat_mod)
    assert s.check_entry_timeout(PAIR, _trade(), _order("aigrid_plan-OLD"), None) is True
    assert s.check_entry_timeout(PAIR, _trade(), _order("aigrid_plan-r11"), None) is False
    # tag-less orders (DCA adds carry no enter_tag) can't be attributed ->
    # not cancelled on tag grounds; the enabled gate still rules
    assert s.check_entry_timeout(PAIR, _trade(), _order(None), None) is False


def test_pending_cancel_boundary_is_documented(strat_mod):
    """The checked-API boundary is explicit in code: PENDING_CANCEL_UNSUPPORTED
    names whether the strategy can cancel ANOTHER trade's order (False only
    for our own via check_entry_timeout; the constant documents the external
    case stays unsupported)."""
    assert hasattr(strat_mod, "PENDING_CANCEL_UNSUPPORTED")
    assert strat_mod.PENDING_CANCEL_UNSUPPORTED is False
    assert callable(getattr(strat_mod.AIGridStrategy, "check_entry_timeout"))  # noqa: B009
