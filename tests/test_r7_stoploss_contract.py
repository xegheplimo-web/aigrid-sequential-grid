"""R7 tests — T-223b: custom_stoploss absolute-price contract.

Verified FT 2026.10 semantics: custom_stoploss() returns the risk distance
measured from CURRENT_RATE (not open_rate); Trade.adjust_stop_loss converts
it back with ``current * (1 - |sl| / leverage)``. So the plan's absolute stop
price must go through ``stoploss_from_absolute`` with trade.leverage — a raw
``sl/open_rate - 1`` ratio is -5% price risk at 2x (should be -10%) and rides
up with price at 1x (unwanted trailing).

The plan stop is therefore ANCHORED at its absolute price: whatever the
returned ratio, FT must land the stop on the same absolute level. Tests run
the real ``stoploss_from_absolute`` semantics (positive return, sign ignored
by FT) — the strategy reports the conventional negative risk ratio.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import time
import types
from pathlib import Path

import pytest
from tests._attest import make_config, write_integration

ROOT = Path(__file__).resolve().parents[1]
STRATEGY_PATH = ROOT / "user_data/strategies/AIGridStrategy.py"
PAIR = "ETH/USDT:USDT"


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
    spec = importlib.util.spec_from_file_location("aigrid_ft_strategy_r7", STRATEGY_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _strategy(strat_mod):
    s = object.__new__(strat_mod.AIGridStrategy)
    s.config = make_config()
    s._aigrid = {}
    s._plan = {}
    s._levels = []
    s._sl_cache = {}
    s._last_sl_pct = None
    return s


def _trade(open_rate=100.0, is_short=False, leverage=1.0, liquidation_price=None):
    return types.SimpleNamespace(
        is_short=is_short,
        open_rate=open_rate,
        leverage=leverage,
        liquidation_price=liquidation_price,
    )


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
            }
        ),
        encoding="utf-8",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    write_integration(path.parent, make_config(), **payload)


def _plan(stop_loss, **kw):
    p = {
        "stop_loss": stop_loss,
        "level_stake": 20.0,
        "grid_mode": "RANGE_GRID",
        "grid_upper": 110.0,
        "grid_lower": 90.0,
        "grid_levels_prices": [98.0, 99.0],
    }
    p.update(kw)
    return p


@pytest.fixture
def wired(strat_mod, monkeypatch, tmp_path):
    """Strategy wired to a tmp integration + cache file."""
    integ = tmp_path / "ft_integration.json"
    cache = tmp_path / "strategy_sl_cache.json"
    monkeypatch.setattr(strat_mod, "INTEGRATION_FILE", integ)
    monkeypatch.setattr(strat_mod, "SL_CACHE_FILE", cache)
    return integ, cache


# ====================================================================== anchoring
def test_long_1x_stop_at_plan_price(strat_mod, wired):
    """1x, entry 100, plan SL 90, current 100 -> -0.10 (stop lands on 90)."""
    integ, _ = wired
    _write_integration(integ, {PAIR: _plan(90.0)})
    s = _strategy(strat_mod)
    pct = s.custom_stoploss(PAIR, _trade(100.0), None, 100.0, 0.0, False)
    assert pct == pytest.approx(-0.10)


def test_long_1x_stop_does_not_ride_up(strat_mod, wired):
    """Price rises to 105: stop must stay ANCHORED at 90 -> -0.142857 from
    current. The old open_rate ratio returned -0.10, pushing the stop to
    94.5 (unwanted trailing)."""
    integ, _ = wired
    _write_integration(integ, {PAIR: _plan(90.0)})
    s = _strategy(strat_mod)
    pct = s.custom_stoploss(PAIR, _trade(100.0), None, 105.0, 0.05, False)
    assert pct == pytest.approx(-0.142857, abs=1e-5)


@pytest.mark.parametrize("leverage,expected", [(2.0, -0.20), (3.0, -0.30)])
def test_long_leveraged_stop_is_leverage_adjusted(strat_mod, wired, leverage, expected):
    """2x/3x, SL 90, current 100: FT divides by leverage, so the risk ratio
    must be leverage-scaled (-0.20 / -0.30) to land the stop on 90."""
    integ, _ = wired
    _write_integration(integ, {PAIR: _plan(90.0)})
    s = _strategy(strat_mod)
    t = _trade(100.0, leverage=leverage, liquidation_price=50.0)
    pct = s.custom_stoploss(PAIR, t, None, 100.0, 0.0, False)
    assert pct == pytest.approx(expected)


@pytest.mark.parametrize("leverage,expected", [(1.0, -0.10), (2.0, -0.20)])
def test_short_stop_anchored_symmetric(strat_mod, wired, leverage, expected):
    """SHORT: stop above price stays absolute. entry 100, SL 110,
    current 100 -> -0.10 / -0.20."""
    integ, _ = wired
    _write_integration(integ, {PAIR: _plan(110.0)})
    s = _strategy(strat_mod)
    t = _trade(100.0, is_short=True, leverage=leverage, liquidation_price=180.0)
    pct = s.custom_stoploss(PAIR, t, None, 100.0, 0.0, False)
    assert pct == pytest.approx(expected)


def test_short_stop_does_not_slide_down(strat_mod, wired):
    """SHORT price falls to 95: stop stays at 110 -> -(110/95 - 1) = -0.1579."""
    integ, _ = wired
    _write_integration(integ, {PAIR: _plan(110.0)})
    s = _strategy(strat_mod)
    t = _trade(100.0, is_short=True, leverage=1.0)
    pct = s.custom_stoploss(PAIR, t, None, 95.0, 0.05, False)
    assert pct == pytest.approx(-(110.0 / 95.0 - 1.0))


# ====================================================================== fallback
@pytest.mark.parametrize("bad_sl", [None, 0, -5.0])
def test_missing_or_zero_stop_falls_back_to_safe_default(strat_mod, wired, bad_sl):
    """plan.stop_loss missing/0/negative -> fallback stop at open*0.90,
    still via the absolute-price path (not a raw ratio)."""
    integ, _ = wired
    _write_integration(integ, {PAIR: _plan(bad_sl)})
    s = _strategy(strat_mod)
    pct = s.custom_stoploss(PAIR, _trade(100.0), None, 100.0, 0.0, False)
    assert pct == pytest.approx(-0.10)


def test_fallback_stop_is_absolute_not_raw_ratio(strat_mod, wired):
    """The -10% fallback anchors at open*0.90: at current=95 the returned
    ratio is the distance from 95 to 90, NOT -0.10."""
    integ, _ = wired
    _write_integration(integ, {PAIR: _plan(None)})
    s = _strategy(strat_mod)
    pct = s.custom_stoploss(PAIR, _trade(100.0), None, 95.0, -0.05, False)
    assert pct == pytest.approx(-(1.0 - 90.0 / 95.0))


# ====================================================================== validation
def test_wrong_side_stop_returns_none_and_blocks(strat_mod, wired, caplog):
    """LONG plan SL 105 above current 100 -> None + SL_INVALID + pair
    blocked from new entries (no clamping a bad plan into validity)."""
    integ, _ = wired
    _write_integration(integ, {PAIR: _plan(105.0)})
    s = _strategy(strat_mod)
    with caplog.at_level(logging.ERROR):
        pct = s.custom_stoploss(PAIR, _trade(100.0), None, 100.0, 0.0, False)
    assert pct is None
    assert any("SL_INVALID" in r.message for r in caplog.records)
    assert s.confirm_trade_entry(PAIR, "limit", 1.0, 99.0, "GTC", None, None, "long") is False
    assert s.custom_stake_amount(PAIR, None, 99.0, 40.0, 10.0, 1000.0, 1.0, None, "long") is None


def test_short_wrong_side_returns_none(strat_mod, wired, caplog):
    """SHORT plan SL 95 below current 100 -> None + SL_INVALID."""
    integ, _ = wired
    _write_integration(integ, {PAIR: _plan(95.0)})
    s = _strategy(strat_mod)
    with caplog.at_level(logging.ERROR):
        pct = s.custom_stoploss(PAIR, _trade(100.0, is_short=True), None, 100.0, 0.0, False)
    assert pct is None
    assert any("SL_INVALID" in r.message for r in caplog.records)


def test_stop_too_close_to_liquidation_returns_none(strat_mod, wired, caplog):
    """LONG liq=88, plan SL 89: only ~1.1% above liquidation (< 3% buffer)
    -> None + SL_INVALID + pair blocked."""
    integ, _ = wired
    _write_integration(integ, {PAIR: _plan(89.0)})
    s = _strategy(strat_mod)
    t = _trade(100.0, leverage=5.0, liquidation_price=88.0)
    with caplog.at_level(logging.ERROR):
        pct = s.custom_stoploss(PAIR, t, None, 100.0, 0.0, False)
    assert pct is None
    assert any("SL_INVALID" in r.message for r in caplog.records)
    assert s.confirm_trade_entry(PAIR, "limit", 1.0, 99.0, "GTC", None, None, "long") is False


def test_blocked_pair_unblocks_on_fixed_plan(strat_mod, wired):
    """A NEW plan with a valid stop releases the block — the block targets
    the bad plan, not the symbol forever."""
    integ, _ = wired
    _write_integration(integ, {PAIR: _plan(105.0)})
    s = _strategy(strat_mod)
    assert s.custom_stoploss(PAIR, _trade(100.0), None, 100.0, 0.0, False) is None
    _write_integration(integ, {PAIR: _plan(90.0)})
    pct = s.custom_stoploss(PAIR, _trade(100.0), None, 100.0, 0.0, False)
    assert pct == pytest.approx(-0.10)
    assert s.confirm_trade_entry(PAIR, "limit", 1.0, 99.0, "GTC", None, None, "long") is True
