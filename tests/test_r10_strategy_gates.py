"""R10 tests — strategy gates: plan TTL, self-contained DCA gate, crossing.

Card R10 (P0-2 / P0-6 / P0-7):

- P0-2: ``_refresh_integration`` parses ``expires_at`` (ISO-8601, naive ==
  UTC) into ``_plan_expires_ms``; ``_enabled()`` returns False once the plan
  expired, with a rate-limited ``PLAN_EXPIRED_BLOCKED`` log. Open-position
  protection (``custom_stoploss``) keeps working from the SL cache.
  ``MAX_FILE_AGE_S`` is now 2 x the 10-minute plan TTL.
- P0-7: FT never calls ``confirm_trade_entry`` for ``mode="pos_adjust"``,
  so ``adjust_trade_position`` runs the full gate itself: halted ->
  blocked pair -> enabled (incl. TTL) -> grid_mode != SAFE_HOLD ->
  level_stake > 0.
- P0-6: the DCA trigger uses the engine's crossing semantics —
  long ``prev_close > level >= low`` / short ``prev_close < level <= high``
  on the last CLOSED candle captured in ``populate_indicators``
  (``self._last_closed[pair]``). Missing candle state -> no DCA.
"""

from __future__ import annotations

import importlib.util
import json
import os
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
    spec = importlib.util.spec_from_file_location("aigrid_ft_strategy_r10", STRATEGY_PATH)
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


def _trade(pair=PAIR, is_short=False, open_rate=100.0):
    return types.SimpleNamespace(
        pair=pair,
        is_short=is_short,
        open_rate=open_rate,
        leverage=1.0,
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
    plan = {
        "plan_id": "plan-r10",
        "stop_loss": 90.0,
        "level_stake": 20.0,
        "grid_mode": "RANGE_GRID",
        "grid_upper": 110.0,
        "grid_lower": 90.0,
        "grid_levels": 4,
        "grid_levels_prices": [95.0, 96.0, 97.0, 98.0],
        "max_capital": 1000.0,
        "max_position_size": 1000.0,
    }
    plan.update(kw)
    return plan


def _short_plan(**kw):
    plan = _entry_plan(
        grid_mode="SHORT_GRID",
        stop_loss=110.0,
        grid_levels_prices=[101.0, 102.0, 103.0, 104.0],
    )
    plan.update(kw)
    return plan


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _df(candles) -> pd.DataFrame:
    """DataFrame from (open, high, low, close) rows; last row = the most
    recent CLOSED candle, second-to-last close = prev_close."""
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


# ====================================================================== TTL (P0-2)
def test_expired_plan_blocks_entry_and_adjust(strat_mod, integ):
    """Plan expired 90min ago -> every NEW-order path refuses; _enabled()
    itself reports False."""
    plan = _entry_plan(expires_at=_iso(_utcnow() - timedelta(minutes=90)))
    _write_integration(integ, {PAIR: plan})
    s = _strategy(strat_mod)
    s._refresh_integration(PAIR)
    assert s._enabled() is False
    assert s.confirm_trade_entry(PAIR, "limit", 1.0, 99.0, "GTC", None, None, "long") is False
    assert _adjust(s, _trade()) is None
    assert s.custom_stake_amount(PAIR, None, 99.0, 40.0, 10.0, 1000.0, 1.0, None, "long") is None


def test_plan_at_exact_expiry_is_blocked(strat_mod, integ):
    """Boundary: expires_at == now -> blocked (expiry is inclusive)."""
    plan = _entry_plan(expires_at=_iso(_utcnow()))
    _write_integration(integ, {PAIR: plan})
    s = _strategy(strat_mod)
    s._refresh_integration(PAIR)
    assert s._enabled() is False
    assert s.confirm_trade_entry(PAIR, "limit", 1.0, 99.0, "GTC", None, None, "long") is False


def test_expired_plan_still_allows_stoploss(strat_mod, integ):
    """An expired plan disables new orders but NEVER the open-position
    protection: custom_stoploss still resolves a valid stop — from the SL
    cache when the expired plan carries none, and from the expired plan's
    own stop_loss (re-remembered on refresh) when it does."""
    plan = _entry_plan(stop_loss=96.0, expires_at=_iso(_utcnow() + timedelta(minutes=5)))
    _write_integration(integ, {PAIR: plan})
    s = _strategy(strat_mod)
    # while the plan is valid its SL is remembered into the SL cache
    pct = s.custom_stoploss(PAIR, _trade(), None, 99.0, -0.01, False)
    assert pct == pytest.approx(-(1.0 - 96.0 / 99.0))
    # the plan then expires: new orders blocked, but the cached absolute
    # stop keeps protecting the open position
    plan["expires_at"] = _iso(_utcnow() - timedelta(minutes=90))
    _write_integration(integ, {PAIR: plan})
    s._refresh_integration(PAIR)
    assert s._enabled() is False
    pct2 = s.custom_stoploss(PAIR, _trade(), None, 99.0, -0.01, False)
    assert pct2 == pytest.approx(-(1.0 - 96.0 / 99.0))
    assert pct2 != -0.99
    # the expired plan's own stop_loss still protects too
    plan["stop_loss"] = 90.0
    _write_integration(integ, {PAIR: plan})
    pct3 = s.custom_stoploss(PAIR, _trade(), None, 99.0, -0.01, False)
    assert pct3 == pytest.approx(-(1.0 - 90.0 / 99.0))


def test_naive_expires_at_treated_as_utc(strat_mod, integ):
    """Naive ISO-8601 (no offset) is interpreted as UTC: a naive past is
    expired, a naive future still enables."""
    past_naive = (_utcnow() - timedelta(minutes=1)).replace(tzinfo=None)
    plan = _entry_plan(expires_at=past_naive.isoformat())
    _write_integration(integ, {PAIR: plan})
    s = _strategy(strat_mod)
    s._refresh_integration(PAIR)
    assert s._enabled() is False

    future_naive = (_utcnow() + timedelta(minutes=5)).replace(tzinfo=None)
    plan["expires_at"] = future_naive.isoformat()
    _write_integration(integ, {PAIR: plan})
    s._refresh_integration(PAIR)
    assert s._enabled() is True


def test_file_age_gate_is_two_ttl(strat_mod, integ):
    """MAX_FILE_AGE_S is 2 x the 10min plan TTL: a file 21min old is stale
    (fail-closed), a file 19min old still enables a valid plan."""
    plan = _entry_plan(expires_at=_iso(_utcnow() + timedelta(minutes=5)))
    _write_integration(integ, {PAIR: plan})
    s = _strategy(strat_mod)

    old = time.time() - 21 * 60
    os.utime(integ, (old, old))
    s._refresh_integration(PAIR)
    assert s._enabled() is False

    fresh = time.time() - 19 * 60
    os.utime(integ, (fresh, fresh))
    s._refresh_integration(PAIR)
    assert s._enabled() is True


# ====================================================================== DCA gate (P0-7)
def test_dca_blocked_when_halted(strat_mod, integ):
    """Halted entries also halt position adds — FT's confirm_trade_entry
    is never consulted for pos_adjust, so the gate must live here."""
    _write_integration(integ, {PAIR: _entry_plan()})
    s = _strategy(strat_mod)
    _closed_candle(s, 99.0, 99.0, 99.5, 96.5, 97.0)  # valid cross present
    s._halted = True
    assert _adjust(s, _trade()) is None


def test_dca_blocked_when_pair_blocked(strat_mod, integ):
    """A symbol blocked by SL_INVALID refuses DCA while the SAME bad plan
    stop keeps arriving."""
    _write_integration(integ, {PAIR: _entry_plan(stop_loss=90.0)})
    s = _strategy(strat_mod)
    s._block_pair_sl(PAIR, 90.0, "test: bad stop")
    _closed_candle(s, 99.0, 99.0, 99.5, 96.5, 97.0)
    assert _adjust(s, _trade()) is None


def test_dca_blocked_when_plan_expired(strat_mod, integ):
    plan = _entry_plan(expires_at=_iso(_utcnow() - timedelta(minutes=5)))
    _write_integration(integ, {PAIR: plan})
    s = _strategy(strat_mod)
    _closed_candle(s, 99.0, 99.0, 99.5, 96.5, 97.0)
    assert _adjust(s, _trade()) is None


def test_dca_blocked_in_safe_hold(strat_mod, integ):
    _write_integration(integ, {PAIR: _entry_plan(grid_mode="SAFE_HOLD")})
    s = _strategy(strat_mod)
    _closed_candle(s, 99.0, 99.0, 99.5, 96.5, 97.0)
    assert _adjust(s, _trade()) is None


def test_dca_blocked_when_level_stake_not_positive(strat_mod, integ):
    _write_integration(integ, {PAIR: _entry_plan(level_stake=0.0)})
    s = _strategy(strat_mod)
    _closed_candle(s, 99.0, 99.0, 99.5, 96.5, 97.0)
    assert _adjust(s, _trade()) is None


# ====================================================================== crossing (P0-6)
def test_dca_requires_crossing_not_price(strat_mod, integ):
    """Level below current_rate is NOT enough: prev_close must have been
    above the level and the last closed candle's low must reach it."""
    _write_integration(integ, {PAIR: _entry_plan()})
    s = _strategy(strat_mod)
    trade = _trade()
    # current_rate 95.5 sits above level 95 but the previous close (94.5)
    # was already BELOW it -> no cross -> no DCA (old code used to buy).
    _closed_candle(s, 94.5, 94.5, 95.2, 93.5, 94.8)
    assert _adjust(s, trade, current_rate=95.5) is None
    # prev_close above the level and the candle's low touched it -> DCA.
    _closed_candle(s, 99.0, 99.0, 99.2, 97.5, 98.0)
    assert _adjust(s, trade, current_rate=98.0) == (20.0, "grid_level_3")


def test_dca_skips_intermediate_level_only_when_crossed(strat_mod, integ):
    """A candle gapping through levels only requests the levels actually
    crossed: 98 then 97 — never 96/95, which sit below current_rate but
    were never crossed."""
    _write_integration(integ, {PAIR: _entry_plan()})
    s = _strategy(strat_mod)
    trade = _trade()
    _closed_candle(s, 99.0, 99.0, 99.5, 96.5, 97.0)
    assert _adjust(s, trade, current_rate=97.0) == (20.0, "grid_level_3")
    _write_integration(integ, {PAIR: _entry_plan()}, filled_prices={PAIR: [98.0]})
    assert _adjust(s, trade, current_rate=97.0) == (20.0, "grid_level_2")
    _write_integration(integ, {PAIR: _entry_plan()}, filled_prices={PAIR: [97.0, 98.0]})
    assert _adjust(s, trade, current_rate=97.0) is None


def test_dca_no_candle_state_fails_closed(strat_mod, integ):
    """No captured closed candle for the pair -> no DCA, and no fallback
    to current_rate."""
    _write_integration(integ, {PAIR: _entry_plan()})
    s = _strategy(strat_mod)
    assert _adjust(s, _trade(), current_rate=94.0) is None
    s2 = _strategy(strat_mod)
    s2._last_closed = {
        "BTC/USDT:USDT": {
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "prev_close": 1.0,
        }
    }
    assert _adjust(s2, _trade(), current_rate=94.0) is None


def test_short_dca_crossing(strat_mod, integ):
    """Short side mirrors: prev_close < level <= high on the last closed
    candle; the lowest crossed level is requested first."""
    _write_integration(integ, {PAIR: _short_plan()})
    s = _strategy(strat_mod)
    trade = _trade(is_short=True, open_rate=100.0)
    # no cross: prev_close already above every level
    _closed_candle(s, 105.0, 105.0, 105.5, 104.5, 105.0)
    assert _adjust(s, trade, current_rate=104.0) is None
    # crossed up through 101 and 102 -> lowest crossed level first
    _closed_candle(s, 100.5, 100.8, 102.5, 100.5, 102.0)
    assert _adjust(s, trade, current_rate=102.0) == (20.0, "grid_level_0")
    _write_integration(integ, {PAIR: _short_plan()}, filled_prices={PAIR: [101.0]})
    assert _adjust(s, trade, current_rate=102.0) == (20.0, "grid_level_1")
