"""R13 tests — per-level take-profit via REDUCE orders (P0-4 / D-097).

Card R13: TP-per-level as negative stake inside ``adjust_trade_position`` —
the ONLY per-level close path in Freqtrade 2026.10. Contract under test:

- a live lot at level price P reaches TP when the CURRENT rate crosses
  ``P * (1 +- grid_spacing)`` (long: >= P*(1+s), short: <= P*(1-s));
- the callback returns ``stake = -(lot qty x lot fill price) / leverage``
  — the lot's own committed margin (R13b / P0-4b) — plus an id tag
  ``{plan_id}:grid:{level_idx}:exit:{seq}`` — at most ONE level per
  callback, the longest-realised first;
- ``trade.has_open_orders`` -> None (FT cancels a pending order when the
  returned order differs — never stack or replace an exit);
- the lot ledger derives from per-order fills: each closed ENTRY fill
  (``grid_level_{i}`` tag, ``order.filled`` qty) funds its own lot;
  unresolvable -> None (no guess);
- ``custom_exit`` may only return a reason when EVERY live lot is at TP —
  never the per-level close path;
- a level re-arms only after its exit order is ``closed`` with
  ``filled > 0`` — reconciled from ``trade.orders``, never from candles.

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

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
STRATEGY_PATH = ROOT / "user_data/strategies/AIGridStrategy.py"
PAIR = "ETH/USDT:USDT"
PLAN_ID = "plan-r13"


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
    spec = importlib.util.spec_from_file_location("aigrid_ft_strategy_r13", STRATEGY_PATH)
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


def _order(
    tag="aigrid_plan-r13",
    *,
    entry=True,
    status="closed",
    price=90.0,
    filled=1.0,
    is_short=False,
):
    side = ("sell" if is_short else "buy") if entry else ("buy" if is_short else "sell")
    return types.SimpleNamespace(
        ft_order_tag=tag,
        ft_is_entry=entry,
        ft_order_side=side,
        status=status,
        price=price,
        average=price,
        amount=filled,
        filled=filled,
    )


_LEVEL_IDX = {90.0: 0, 100.0: 1, 110.0: 2}
_LEVEL_STAKE = 20.0  # margin per lot — qty = level_stake / fill price
_LOTS3_QTY = sum(_LEVEL_STAKE / p for p in (90.0, 100.0, 110.0))


def _entry_order(price, plan_id=PLAN_ID, **kw):
    """A closed ENTRY fill as the DCA add path actually stamps it:
    ``ft_order_tag = grid_level_{i}`` (R13b / P0-11 — level identity comes
    ONLY from this tag) and ``filled`` = the lot's own qty."""
    kw.setdefault("filled", _LEVEL_STAKE / price)
    return _order(f"grid_level_{_LEVEL_IDX[price]}", price=price, **kw)


def _exit_order(tag, *, status="open", filled=0.0, price=94.6, is_short=False):
    return _order(tag, entry=False, status=status, price=price, filled=filled, is_short=is_short)


def _trade(
    orders=None,
    is_short=False,
    has_open_orders=False,
    amount=_LOTS3_QTY,
    stake=60.0,
):
    return types.SimpleNamespace(
        pair=PAIR,
        is_short=is_short,
        open_rate=100.0,
        amount=amount,
        stake_amount=stake,
        leverage=1.0,
        liquidation_price=None,
        has_open_orders=has_open_orders,
        # the SAME list object: tests append order-state changes between
        # callbacks, exactly as FT mutates trade.orders between loops
        orders=orders,
    )


def _three_lot_orders(is_short=False):
    """Three closed entry fills at the 3 plan levels — the FT-side truth
    the ledger reconciles against."""
    return [_entry_order(p, is_short=is_short) for p in (90.0, 100.0, 110.0)]


def _write_integration(path: Path, plans: dict, filled_prices=None, enabled=True, verified=True):
    path.write_text(
        json.dumps(
            {
                "ts_ms": int(time.time() * 1000),
                "enabled": enabled,
                "dry_run_verified": verified,
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
        "level_stake": 20.0,
        "grid_mode": "RANGE_GRID",
        "grid_upper": 110.0,
        "grid_lower": 85.0,
        "grid_levels": 3,
        "grid_spacing": 0.05,
        "grid_levels_prices": [90.0, 100.0, 110.0],
        "max_capital": 1000.0,
        "max_position_size": 5000.0,
        "expires_at": (datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
    }
    plan.update(kw)
    return plan


def _short_plan(**kw):
    plan = _tp_plan(grid_mode="SHORT_GRID", stop_loss=130.0)
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
    return {PAIR: [90.0, 100.0, 110.0]}


# =============================================================== card cases
def test_partial_tp_reduces_only_that_level(strat_mod, integ):
    """3 live lots (90/100/110), price 94.6: only the lot at 90 crossed its
    TP (94.5). The callback must return a NEGATIVE stake covering exactly
    that lot's own margin — qty = 20/90 = 0.2222 at fill price 90 ->
    stake = -(0.2222 * 90) = -20 — tagged ``plan-r13:grid:0:exit:1``.
    The 100/110 lots stay untouched."""
    _write_integration(integ, {PAIR: _tp_plan()}, filled_prices=_book3())
    s = _strategy(strat_mod)
    trade = _trade(orders=_three_lot_orders())
    res = _adjust(s, trade, current_rate=94.6)
    assert res is not None
    stake, tag = res
    assert stake == pytest.approx(-20.0)  # one lot of three, margin share
    assert tag == f"{PLAN_ID}:grid:0:exit:1"


def test_no_partial_when_open_exit_order(strat_mod, integ):
    """trade.has_open_orders -> None. FT cancels a pending order when the
    returned order differs — returning a reduce while an exit order is
    open would stack/replace it."""
    _write_integration(integ, {PAIR: _tp_plan()}, filled_prices=_book3())
    s = _strategy(strat_mod)
    trade = _trade(orders=_three_lot_orders(), has_open_orders=True)
    assert _adjust(s, trade, current_rate=94.6) is None


def test_full_tp_still_closes_whole_trade(strat_mod, integ):
    """custom_exit returns a whole-trade reason ONLY when every live lot
    is at TP. At 105.5 lots 90/100 are at TP but 110 is not -> None
    (custom_exit must never be the per-level close path). At 116.0 all
    three are at TP -> a reason is returned."""
    _write_integration(integ, {PAIR: _tp_plan()}, filled_prices=_book3())
    s = _strategy(strat_mod)
    trade = _trade(orders=_three_lot_orders())
    assert _exit_reason(s, trade, current_rate=105.5) is None
    assert _exit_reason(s, trade, current_rate=116.0) is not None


def test_unresolvable_ledger_fails_closed(strat_mod, integ):
    """The R13b lot ledger resolves from ``trade.orders`` alone — the
    published price book is no longer consulted. Order state that cannot
    produce a ledger (no orders at all, or fills that carry no
    ``grid_level_`` identity) -> None, never a guessed quantity."""
    _write_integration(integ, {PAIR: _tp_plan()}, filled_prices=_book3())
    s = _strategy(strat_mod)
    assert _adjust(s, _trade(orders=None), current_rate=116.0) is None
    assert _adjust(s, _trade(orders=[]), current_rate=116.0) is None
    # an entry fill with an out-of-range level tag -> unmapped -> None
    orders = [*_three_lot_orders(), _order("grid_level_7", price=80.0, filled=0.5)]
    assert _adjust(s, _trade(orders=orders), current_rate=116.0) is None
    # a malformed/absent price book does NOT veto an order-resolvable ledger
    _write_integration(integ, {PAIR: _tp_plan()}, filled_prices={})
    res = _adjust(s, _trade(orders=_three_lot_orders()), current_rate=94.6)
    assert res is not None and res[1] == f"{PLAN_ID}:grid:0:exit:1"


def test_short_partial_tp(strat_mod, integ):
    """Mirror for shorts: TP at level*(1 - spacing). Lot at 110 TPs at
    104.5; rate 104.4 reduces exactly that level (ascending index 2)."""
    _write_integration(integ, {PAIR: _short_plan()}, filled_prices=_book3())
    s = _strategy(strat_mod)
    trade = _trade(orders=_three_lot_orders(is_short=True), is_short=True)
    res = _adjust(s, trade, current_rate=104.4)
    assert res is not None
    stake, tag = res
    assert stake == pytest.approx(-20.0)
    assert tag == f"{PLAN_ID}:grid:2:exit:1"
    # 103.0 is below 110's TP too but ABOVE 100*0.95=95 -> still only 110.
    # 94.0 crosses 100's TP (95.0) as well -> longest-realised first is
    # still the extreme level (110 for shorts).
    res2 = _adjust(s, trade, current_rate=94.0)
    assert res2 is not None and res2[1] == f"{PLAN_ID}:grid:2:exit:2"


def test_rearm_only_after_fill_confirmed(strat_mod, integ):
    """An emitted reduce does not re-arm the level until FT reports the
    exit order closed + filled. While the exit order is open the level is
    occupied; once confirmed the level leaves the ledger — and a NEW lot
    filled at the same price (entries > exits in FT order state) re-arms
    it so the same level can TP again with a fresh seq."""
    _write_integration(integ, {PAIR: _tp_plan()}, filled_prices=_book3())
    s = _strategy(strat_mod)
    orders = _three_lot_orders()
    trade = _trade(orders=orders)
    tag1 = _adjust(s, trade, current_rate=94.6)[1]
    assert tag1 == f"{PLAN_ID}:grid:0:exit:1"

    # exit order open (not yet filled) -> no new order may be emitted
    orders.append(_exit_order(tag1, status="open", filled=0.0))
    trade.has_open_orders = True
    assert _adjust(s, trade, current_rate=94.6) is None
    # not re-armed while open: even after the flag clears, the level is
    # still blocked — the order exists but is not a confirmed fill
    trade.has_open_orders = False
    assert _adjust(s, trade, current_rate=94.6) is None

    # exit order closed + filled -> level 90's lot is gone (the exact
    # lot's qty leaves the ledger); trade.amount shrinks with it
    orders[-1].status = "closed"
    orders[-1].filled = _LEVEL_STAKE / 90.0
    trade.amount = _LEVEL_STAKE / 100.0 + _LEVEL_STAKE / 110.0
    assert _adjust(s, trade, current_rate=94.6) is None  # no double-sell

    # re-arm: a NEW closed entry fill at 90 makes the level live again ->
    # a fresh reduce with the next seq.
    orders.append(_entry_order(90.0))
    trade.amount = _LOTS3_QTY
    res = _adjust(s, trade, current_rate=94.6)
    assert res is not None
    assert res[1] == f"{PLAN_ID}:grid:0:exit:2"
    assert res[0] == pytest.approx(-20.0)


# --------------------------------------------------------------- contract extras
def test_one_level_per_callback_longest_first(strat_mod, integ):
    """Multiple lots at TP -> exactly ONE reduce per callback, the
    longest-realised level first. For a long grid the deepest (lowest)
    lot reached its TP earliest; each emission produces a new seq."""
    _write_integration(integ, {PAIR: _tp_plan()}, filled_prices=_book3())
    s = _strategy(strat_mod)
    orders = _three_lot_orders()
    trade = _trade(orders=orders)
    # rate 106: levels 90 (TP 94.5) and 100 (TP 105) are both at TP;
    # 110 (TP 115.5) is not.
    res = _adjust(s, trade, current_rate=106.0)
    assert res == (pytest.approx(-20.0), f"{PLAN_ID}:grid:0:exit:1")
    # confirm level 90's exit (its own qty leaves the position); next
    # callback must reduce level 100 — still one level, next seq, and the
    # stake is that lot's own margin (0.2 x 100 = -20).
    orders.append(_exit_order(res[1], status="closed", filled=_LEVEL_STAKE / 90.0))
    trade.amount = _LEVEL_STAKE / 100.0 + _LEVEL_STAKE / 110.0
    trade.stake_amount = 40.0
    res2 = _adjust(s, trade, current_rate=106.0)
    assert res2 == (pytest.approx(-20.0), f"{PLAN_ID}:grid:1:exit:2")


def test_reduce_survives_plan_expiry_but_not_unverified(strat_mod, integ):
    """A TP reduce is position management: like custom_stoploss it keeps
    working after the plan TTL expires (open-position protection), but a
    missing dry-run attestation blocks it (fail-closed)."""
    plan = _tp_plan(expires_at=(datetime.now(UTC) - timedelta(minutes=5)).isoformat())
    _write_integration(integ, {PAIR: plan}, filled_prices=_book3())
    s = _strategy(strat_mod)
    trade = _trade(orders=_three_lot_orders())
    res = _adjust(s, trade, current_rate=94.6)
    assert res is not None and res[0] == pytest.approx(-20.0)

    _write_integration(integ, {PAIR: _tp_plan()}, filled_prices=_book3(), verified=False)
    s2 = _strategy(strat_mod)
    assert _adjust(s2, _trade(orders=_three_lot_orders()), current_rate=94.6) is None


def test_reduce_blocks_during_add_pending(strat_mod, integ):
    """An open ENTRY order also blocks the reduce — FT would replace the
    pending order with ours, churning the sequential book."""
    _write_integration(integ, {PAIR: _tp_plan()}, filled_prices=_book3())
    s = _strategy(strat_mod)
    orders = [
        *_three_lot_orders(),
        _order("grid_level_0", entry=True, status="open", price=85.0, filled=0.0),
    ]
    trade = _trade(orders=orders, has_open_orders=True)
    assert _adjust(s, trade, current_rate=94.6) is None


def test_unmappable_order_fails_closed(strat_mod, integ):
    """A closed+filled order the strategy cannot place on the grid (a
    foreign exit, or an entry between levels) makes the ledger
    unresolvable -> None, never a guessed quantity."""
    _write_integration(integ, {PAIR: _tp_plan()}, filled_prices=_book3())
    s = _strategy(strat_mod)
    # an untagged closed exit fill that is not one of our :exit: orders
    orders = [
        *_three_lot_orders(),
        _order(None, entry=False, status="closed", price=94.6, filled=0.5),
    ]
    assert _adjust(s, _trade(orders=orders), current_rate=116.0) is None
    # an entry fill off the grid entirely (80.0 maps to no plan level)
    orders = [*_three_lot_orders(), _order("aigrid_plan-r13", price=80.0, filled=0.5)]
    assert _adjust(s, _trade(orders=orders), current_rate=116.0) is None


def test_custom_exit_unresolvable_fails_closed(strat_mod, integ):
    """custom_exit without a resolvable ledger must NOT close the whole
    trade on the old profit-threshold heuristic."""
    _write_integration(integ, {PAIR: _tp_plan()}, filled_prices=_book3())
    s = _strategy(strat_mod)
    trade = _trade(orders=None)
    assert _exit_reason(s, trade, current_rate=200.0, current_profit=5.0) is None
    # unmapped fills make the ledger unresolvable the same way
    orders = [*_three_lot_orders(), _order("grid_level_7", price=80.0, filled=0.5)]
    trade2 = _trade(orders=orders)
    assert _exit_reason(s, trade2, current_rate=200.0, current_profit=5.0) is None
