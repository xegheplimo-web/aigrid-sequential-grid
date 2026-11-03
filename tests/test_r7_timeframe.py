"""R7 tests — T-223c: timeframe pinned to 5m + startup fail-closed check.

The incident: the strategy shipped ``timeframe = "15m"`` while the approved
runtime runs 5m via config override — lose the override and the bot silently
trades 15m. Now the class attribute is 5m AND ``bot_start`` verifies the
effective timeframe: a mismatch logs an error and sets ``_halted`` so no new
orders are created (open positions keep being managed — no hard raise).
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

ROOT = Path(__file__).resolve().parents[1]
STRATEGY_PATH = ROOT / "user_data/strategies/AIGridStrategy.py"
PAIR = "ETH/USDT:USDT"


def _real_stoploss_from_absolute(stop_rate, current_rate, is_short=False, leverage=1.0):
    if current_rate == 0:
        return 1
    stoploss = 1 - (stop_rate / current_rate)
    if is_short:
        stoploss = -stoploss
    return max(stoploss, 0.0) * leverage


def _install_ft_stubs() -> None:
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
    spec = importlib.util.spec_from_file_location("aigrid_ft_strategy_r7tf", STRATEGY_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _strategy(strat_mod):
    s = object.__new__(strat_mod.AIGridStrategy)
    s._aigrid = {}
    s._plan = {}
    s._levels = []
    s._sl_cache = {}
    s._last_sl_pct = None
    return s


def _write_integration(path: Path, plans: dict) -> None:
    path.write_text(
        json.dumps(
            {
                "ts_ms": int(time.time() * 1000),
                "enabled": True,
                "dry_run_verified": True,
                "mode": "dryrun",
                "plans": plans,
                "filled_levels": {},
            }
        ),
        encoding="utf-8",
    )


def _entry_plan():
    return {
        "stop_loss": 90.0,
        "level_stake": 20.0,
        "grid_mode": "RANGE_GRID",
        "grid_upper": 110.0,
        "grid_lower": 90.0,
        "grid_levels_prices": [98.0, 99.0],
    }


@pytest.fixture
def wired(strat_mod, monkeypatch, tmp_path):
    integ = tmp_path / "ft_integration.json"
    monkeypatch.setattr(strat_mod, "INTEGRATION_FILE", integ)
    monkeypatch.setattr(strat_mod, "SL_CACHE_FILE", tmp_path / "cache.json")
    _write_integration(integ, {PAIR: _entry_plan()})
    return integ


# ====================================================================== attribute + gate
def test_class_timeframe_is_5m(strat_mod):
    assert strat_mod.AIGridStrategy.timeframe == "5m"


def test_bot_start_halts_on_wrong_effective_timeframe(strat_mod, wired, caplog):
    """Effective timeframe 15m (config override lost/changed) -> _halted,
    and both entry gates refuse. Bot itself does not raise."""
    s = _strategy(strat_mod)
    s.timeframe = "15m"
    with caplog.at_level(logging.ERROR):
        s.bot_start()
    assert s._halted is True
    assert any("timeframe" in r.message for r in caplog.records)
    assert s.confirm_trade_entry(PAIR, "limit", 1.0, 99.0, "GTC", None, None, "long") is False
    assert s.custom_stake_amount(PAIR, None, 99.0, 40.0, 10.0, 1000.0, 1.0, None, "long") is None


def test_bot_start_config_timeframe_overrides_attr(strat_mod, wired):
    """FT resolves config["timeframe"] over the class attribute — a stray
    15m override must still halt even when the attribute says 5m."""
    s = _strategy(strat_mod)
    s.timeframe = "5m"
    s.config = {"timeframe": "15m"}
    s.bot_start()
    assert s._halted is True


def test_bot_start_5m_runs_normally(strat_mod, wired):
    s = _strategy(strat_mod)
    s.timeframe = "5m"
    s.config = {"timeframe": "5m"}
    s.bot_start()
    assert s._halted is False
    assert s.confirm_trade_entry(PAIR, "limit", 1.0, 99.0, "GTC", None, None, "long") is True
    assert s.custom_stake_amount(
        PAIR, None, 99.0, 40.0, 10.0, 1000.0, 1.0, None, "long"
    ) == pytest.approx(20.0)


def test_halted_blocks_position_adds_but_not_protection(strat_mod, wired):
    """Halt = no NEW orders: entries AND position adds are refused, but
    custom_stoploss/custom_exit keep managing the open trade."""
    s = _strategy(strat_mod)
    s.timeframe = "15m"
    s.bot_start()
    trade = types.SimpleNamespace(
        pair=PAIR, is_short=False, open_rate=100.0, leverage=1.0, liquidation_price=None
    )
    assert (
        s.adjust_trade_position(trade, None, 97.0, -0.03, 10.0, 1000.0, 97.0, 97.0, -0.03, -0.03)
        is None
    )
    # protection still evaluated for the open position
    assert s.custom_stoploss(PAIR, trade, None, 100.0, 0.0, False) == pytest.approx(-0.10)
