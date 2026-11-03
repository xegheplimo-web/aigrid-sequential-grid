"""R13b tests — three P0 defects in per-level TP, found by the Docker probe
(``user_data/repro/repro_r13.py``, evidence table in card-R13b.md):

- **P0-11 (case D)**: ``_entry_level_index`` must attribute an ENTRY fill to
  a plan level ONLY via ``ft_order_tag == "grid_level_{i}"``. The old
  half-gap price fallback mapped an off-level fill (92.3, inside the 90
  half-gap) back to level 0 and emitted a reduce — removed: any tag that is
  not a well-formed in-range ``grid_level_`` tag -> ``unmapped`` -> the lot
  ledger stays unresolved -> ``None`` (fail-closed, never guessed).
- **P0-4b (case A)**: the lot ledger is built from per-order fills
  (``order.filled`` per closed ENTRY order), never the uniform
  ``trade.amount / n_live`` allocation. An exit fill decrements the qty of
  exactly its tagged level. ``sum(qty_remaining)`` must reconcile with
  ``trade.amount`` within ``max(1e-6, 0.5% * amount)`` else
  ``LEDGER_UNBALANCED`` + ``None``. The negative stake is computed from the
  lot's OWN qty and fill price (``order.average`` / ``order.price``),
  divided by leverage — not scaled via ``trade.stake_amount/amount``.
- **P0-12 (case C)**: when EVERY live lot is at TP, ``adjust`` must not
  emit a reduce — ``custom_exit`` performs the whole-trade close. The
  probe measured both paths firing on the same candle.

Case F (exit seq never reused) was verified CORRECT by the probe and is
kept green here as a regression guard.

All tests are hermetic: freqtrade/talib are stubbed, trades/orders are
SimpleNamespace fakes — no Freqtrade install, no exchange, no socket.
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

ROOT = Path(__file__).resolve().parents[1]
STRATEGY_PATH = ROOT / "user_data/strategies/AIGridStrategy.py"
PAIR = "SOL/USDT:USDT"
PLAN_ID = "r13b"
LEVELS = [90.0, 95.0, 100.0]  # ascending (strategy sorts published levels)
LEVEL_STAKE = 40.0


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
    spec = importlib.util.spec_from_file_location("aigrid_ft_strategy_r13b", STRATEGY_PATH)
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
    s.config = {"timeframe": "5m"}
    s.bot_start()
    return s


def _entry(level_idx, qty, *, price=None, average=None, tag=None, status="closed"):
    """A CLOSED, filled ENTRY order — the same shape the DCA add path
    produces: ``ft_order_tag = grid_level_{i}``, ``average`` = realized fill
    price (defaults to the level price), ``price`` = order price."""
    level = LEVELS[level_idx]
    avg = level if average is None else average
    order_price = level if price is None else price
    return types.SimpleNamespace(
        ft_order_tag=(f"grid_level_{level_idx}" if tag is None else tag),
        ft_is_entry=True,
        ft_order_side="buy",
        status=status,
        price=order_price,
        average=avg,
        amount=qty,
        filled=qty,
    )


def _exit(tag, *, status="open", filled=0.0, price=94.6):
    return types.SimpleNamespace(
        ft_order_tag=tag,
        ft_is_entry=False,
        ft_order_side="sell",
        status=status,
        price=price,
        average=price,
        amount=filled,
        filled=filled,
    )


def _trade(orders, *, amount, stake=120.0, is_short=False, has_open_orders=False):
    return types.SimpleNamespace(
        pair=PAIR,
        is_short=is_short,
        open_rate=95.0,
        amount=amount,
        stake_amount=stake,
        leverage=1.0,
        liquidation_price=None,
        has_open_orders=has_open_orders,
        orders=orders,
    )


def _write_integration(path: Path, plans: dict, filled_prices=None):
    path.write_text(
        json.dumps(
            {
                "ts_ms": int(time.time() * 1000),
                "enabled": True,
                "dry_run_verified": True,
                "mode": "dryrun",
                "plans": plans,
                "filled_levels": {},
                "filled_level_prices": (
                    filled_prices if filled_prices is not None else {p: [] for p in plans}
                ),
            }
        ),
        encoding="utf-8",
    )


def _tp_plan(**kw):
    plan = {
        "plan_id": PLAN_ID,
        "stop_loss": 80.0,
        "level_stake": LEVEL_STAKE,
        "grid_mode": "RANGE_GRID",
        "grid_upper": 100.0,
        "grid_lower": 85.0,
        "grid_levels": 3,
        "grid_spacing": 0.05,
        "grid_levels_prices": [100.0, 95.0, 90.0],  # published descending
        "max_capital": 1000.0,
        "max_position_size": 5000.0,
        "expires_at": (datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
    }
    plan.update(kw)
    return plan


@pytest.fixture
def integ(strat_mod, monkeypatch, tmp_path):
    path = tmp_path / "ft_integration.json"
    monkeypatch.setattr(strat_mod, "INTEGRATION_FILE", path)
    monkeypatch.setattr(strat_mod, "SL_CACHE_FILE", tmp_path / "cache.json")
    return path


def _adjust(s, trade, current_rate):
    return s.adjust_trade_position(
        trade,
        None,
        current_rate,
        0.0,
        10.0,
        1000.0,
        current_rate,
        current_rate,
        0.0,
        0.0,
    )


def _exit_reason(s, trade, current_rate, current_profit=0.2):
    return s.custom_exit(PAIR, trade, None, current_rate, current_profit)


def _book3():
    return {PAIR: [90.0, 95.0, 100.0]}


def _uniform_entries():
    """The probe's shape: 3 lots, each carrying exactly LEVEL_STAKE
    notional (qty = 40/level) at the 3 levels, grid_level_-tagged."""
    return [_entry(i, LEVEL_STAKE / lv) for i, lv in enumerate(LEVELS)]


def _offuniform_entries():
    """Card's off-uniform fills: qty 0.398 / 0.421 / 0.404 on 90/95/100."""
    return [_entry(i, q) for i, q in enumerate((0.398, 0.421, 0.404))]


# =============================================================== P0-11 (case D)
def test_d_offlevel_entry_fails_closed(strat_mod, integ):
    """Probe case D: an extra ENTRY fill at 92.3 — inside the 90 half-gap
    but tagged with a foreign order tag — must NOT be price-mapped back to
    level 0. The old code emitted ``(-100.0, exit:1)``; the corrected code
    counts it unmapped -> ledger None -> adjust returns None."""
    _write_integration(integ, {PAIR: _tp_plan()}, filled_prices=_book3())
    s = _strategy(strat_mod)
    orders = _uniform_entries()
    orders.append(_entry(0, 0.4, price=92.3, average=92.3, tag=f"{PLAN_ID}:grid:5:entry:6"))
    trade = _trade(orders, amount=sum(o.filled for o in orders))
    assert _adjust(s, trade, current_rate=99.76) is None


def test_offlevel_aigrid_tag_fails_closed(strat_mod, integ):
    """An entry fill whose tag is the generic plan tag ``aigrid_{plan_id}``
    (the tag populate_entry_trend stamps) is also NOT a level identity:
    at an off-level price it is unmapped -> None; even at an exact level
    price it carries no ``grid_level_`` identity -> unmapped -> None."""
    _write_integration(integ, {PAIR: _tp_plan()}, filled_prices=_book3())
    s = _strategy(strat_mod)
    orders = _uniform_entries()
    orders.append(_entry(0, 0.4, price=92.3, average=92.3, tag=f"aigrid_{PLAN_ID}"))
    trade = _trade(orders, amount=sum(o.filled for o in orders))
    assert _adjust(s, trade, current_rate=99.76) is None

    s2 = _strategy(strat_mod)
    orders2 = _uniform_entries()
    orders2.append(_entry(0, 0.4, tag=f"aigrid_{PLAN_ID}"))  # exact level price 90
    trade2 = _trade(orders2, amount=sum(o.filled for o in orders2))
    assert _adjust(s2, trade2, current_rate=99.76) is None


def test_bad_or_out_of_range_level_tag_fails_closed(strat_mod, integ):
    """grid_level_ tags that are malformed or out of range -> unmapped."""
    _write_integration(integ, {PAIR: _tp_plan()}, filled_prices=_book3())
    for tag in ("grid_level_9", "grid_level_-1", "grid_level_x", "grid_level_"):
        s = _strategy(strat_mod)
        orders = [*_uniform_entries(), _entry(0, 0.4, tag=tag)]
        trade = _trade(orders, amount=sum(o.filled for o in orders))
        assert _adjust(s, trade, current_rate=99.76) is None, tag


# =============================================================== P0-4b (case A)
def test_offuniform_stake_matches_own_lot_qty(strat_mod, integ):
    """Card's numbers: lots 0.398 (90) / 0.421 (95) / 0.404 (100). At
    94.6 only the 90-lot crossed its TP (94.5). The returned stake must be
    the lot's OWN qty x its OWN fill price: -(0.398 * 90) = -35.82, NOT
    the uniform share -(amount/3) * stake/amount = -40.0."""
    _write_integration(integ, {PAIR: _tp_plan()}, filled_prices=_book3())
    s = _strategy(strat_mod)
    entries = _offuniform_entries()
    trade = _trade(entries, amount=1.223)
    res = _adjust(s, trade, current_rate=94.6)
    assert res is not None
    stake, tag = res
    assert stake == pytest.approx(-0.398 * 90.0, abs=1e-4)
    assert stake != pytest.approx(-40.0)  # old uniform allocation
    assert tag == f"{PLAN_ID}:grid:0:exit:1"


def test_stake_uses_lot_average_fill_price(strat_mod, integ):
    """The fill price is the order's realized ``average`` (fallback
    ``price``), not the nominal level price: a grid_level_0 lot that
    filled at avg 91.5 stakes -(0.4 * 91.5) = -36.6."""
    _write_integration(integ, {PAIR: _tp_plan()}, filled_prices=_book3())
    s = _strategy(strat_mod)
    entries = [
        _entry(0, 0.4, average=91.5),
        _entry(1, 0.421),
        _entry(2, 0.404),
    ]
    trade = _trade(entries, amount=1.225)
    res = _adjust(s, trade, current_rate=94.6)
    assert res is not None
    assert res[0] == pytest.approx(-0.4 * 91.5, abs=1e-4)


def test_exit_fill_decrements_exact_lot(strat_mod, integ):
    """A confirmed exit fill decrements only its tagged level's qty. A
    partial exit fill (0.2 of the 0.398 lot) leaves 0.198 live at level 90;
    the next reduce drains the remainder with a fresh seq."""
    _write_integration(integ, {PAIR: _tp_plan()}, filled_prices=_book3())
    s = _strategy(strat_mod)
    orders = _offuniform_entries()
    trade = _trade(orders, amount=1.223)
    res = _adjust(s, trade, current_rate=94.6)
    assert res is not None and res[1] == f"{PLAN_ID}:grid:0:exit:1"

    orders.append(_exit(res[1], status="closed", filled=0.2))
    trade.amount = 1.223 - 0.2
    res2 = _adjust(s, trade, current_rate=94.6)
    assert res2 is not None
    assert res2[0] == pytest.approx(-0.198 * 90.0, abs=1e-4)
    assert res2[1] == f"{PLAN_ID}:grid:0:exit:2"

    # the rest of the lot drains too -> level leaves the ledger
    orders.append(_exit(res2[1], status="closed", filled=0.198))
    trade.amount = 1.223 - 0.398
    assert _adjust(s, trade, current_rate=94.6) is None


def test_ledger_unbalanced_fails_closed(strat_mod, integ, caplog):
    """sum(qty_remaining) must reconcile with trade.amount within
    max(1e-6, 0.5% * amount). A 5% drift -> LEDGER_UNBALANCED + None."""
    _write_integration(integ, {PAIR: _tp_plan()}, filled_prices=_book3())
    s = _strategy(strat_mod)
    trade = _trade(_offuniform_entries(), amount=1.223 * 1.05)
    with caplog.at_level("WARNING"):
        assert _adjust(s, trade, current_rate=94.6) is None
    assert "LEDGER_UNBALANCED" in caplog.text


def test_ledger_within_tolerance_emits(strat_mod, integ):
    """0.4% drift (< 0.5%) still reconciles -> the reduce is emitted."""
    _write_integration(integ, {PAIR: _tp_plan()}, filled_prices=_book3())
    s = _strategy(strat_mod)
    trade = _trade(_offuniform_entries(), amount=1.223 * 1.004)
    res = _adjust(s, trade, current_rate=94.6)
    assert res is not None and res[1] == f"{PLAN_ID}:grid:0:exit:1"


# =============================================================== P0-12 (case C)
def test_all_lots_at_tp_emits_no_reduce_custom_exit_closes(strat_mod, integ):
    """Probe case C: at 105.05 every live lot is at TP. adjust must NOT
    emit a reduce alongside the whole-trade close — it returns None, and
    custom_exit returns the whole-trade reason on the same candle."""
    _write_integration(integ, {PAIR: _tp_plan()}, filled_prices=_book3())
    s = _strategy(strat_mod)
    entries = _offuniform_entries()
    trade = _trade(entries, amount=1.223)
    rate = 105.05
    assert _adjust(s, trade, current_rate=rate) is None
    assert _exit_reason(s, trade, current_rate=rate) == "GRID_TP_ALL"


def test_partial_tp_still_emits(strat_mod, integ):
    """2 of 3 lots at TP is still per-level reduce business: exactly one
    reduce for the longest-realised lot, custom_exit stays silent."""
    _write_integration(integ, {PAIR: _tp_plan()}, filled_prices=_book3())
    s = _strategy(strat_mod)
    trade = _trade(_offuniform_entries(), amount=1.223)
    res = _adjust(s, trade, current_rate=99.76)
    assert res is not None and res[1] == f"{PLAN_ID}:grid:0:exit:1"
    assert _exit_reason(s, trade, current_rate=99.76) is None


# =============================================================== case F regression
def test_exit_seq_never_reused(strat_mod, integ):
    """After a confirmed exit + re-arm fill at the same level the next
    reduce uses seq+1; and a fresh strategy instance (restart) raises the
    seq past every observed :exit: tag — ids are never reused."""
    _write_integration(integ, {PAIR: _tp_plan()}, filled_prices=_book3())
    s = _strategy(strat_mod)
    orders = _offuniform_entries()
    trade = _trade(orders, amount=1.223)
    res = _adjust(s, trade, current_rate=94.6)
    assert res[1] == f"{PLAN_ID}:grid:0:exit:1"

    # confirm the exit, then a NEW lot fills the same level (re-arm)
    orders.append(_exit(res[1], status="closed", filled=0.398))
    trade.amount = 1.223 - 0.398
    orders.append(_entry(0, 0.398))
    trade.amount = 1.223
    res2 = _adjust(s, trade, current_rate=94.6)
    assert res2 is not None and res2[1] == f"{PLAN_ID}:grid:0:exit:2"

    # restart: a bare strategy sees the order book and continues the seq
    s2 = _strategy(strat_mod)
    orders.append(_exit(res2[1], status="closed", filled=0.398))
    trade.amount = 1.223 - 0.398
    orders.append(_entry(0, 0.398))
    trade.amount = 1.223
    res3 = _adjust(s2, trade, current_rate=94.6)
    assert res3 is not None and res3[1] == f"{PLAN_ID}:grid:0:exit:3"
