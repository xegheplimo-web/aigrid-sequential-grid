"""R14 tests — P0-5c-extended fill semantics in ``aigrid/state/ft_fills.py``.

Card R14 addendum v2 (D-100): confirmed entry fills are counted across
statuses — ``closed`` (already), ``canceled``/``expired`` with ``filled>0``
(the exchange killed the order but the partial fill still exists as real
exposure), and ``open`` orders mid partial-fill. REST status is a snapshot
of the same order book, so an order that keeps filling must be applied by
DELTA only: the ``OrderFillLedger`` keys on ``order_id`` and accumulates
``cumulative_filled_qty`` — a repeated REST payload never double-counts.

Reconciliation failure locks entry/DCA/rearm for the pair, NEVER the
protective exit / stop-loss path — the publisher gate test lives in
test_r14_publish.py.

All tests hermetic: trades are plain dicts, no REST, no sockets.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from aigrid.state.ft_fills import (
    OrderFillLedger,
    filled_prices_for_pair,
)

ROOT = Path(__file__).resolve().parents[1]
PAIR = "ETH/USDT:USDT"
PLAN_ID = "plan-r14"
LEVELS = [95.0, 90.0, 85.0]  # descending, like plan.level_prices()


def _order(**kw) -> dict:
    order = {
        "order_id": "o1",
        "pair": PAIR,
        "ft_pair": "ETHUSDT",
        "ft_order_side": "buy",
        "ft_is_entry": True,
        "ft_order_tag": f"aigrid_{PLAN_ID}",
        "status": "closed",
        "order_type": "limit",
        "price": 95.0,
        "average": 95.0,
        "amount": 1.0,
        "filled": 1.0,
        "remaining": 0.0,
    }
    order.update(kw)
    return order


def _trade(orders, pair: str = PAIR) -> dict:
    return {
        "trade_id": 1,
        "pair": pair,
        "strategy": "AIGridStrategy",
        "is_open": True,
        "orders": orders,
    }


# =============================================================== statuses (P0-5c ext)
def test_canceled_order_with_fill_counted():
    """P0-5c: an order CANCELED after a partial fill is real exposure —
    its level must be counted (the R12 code dropped it entirely)."""
    trades = [_trade([_order(status="canceled", filled=0.4, remaining=0.6)])]
    res = filled_prices_for_pair(trades, pair=PAIR, plan_id=PLAN_ID, level_prices=LEVELS)
    assert res.prices == [95.0]
    assert res.unmapped_orders == 0


def test_expired_order_with_fill_counted():
    """Same for FT ``expired`` status — partially filled before expiry."""
    trades = [_trade([_order(status="expired", filled=0.4)])]
    res = filled_prices_for_pair(trades, pair=PAIR, plan_id=PLAN_ID, level_prices=LEVELS)
    assert res.prices == [95.0]


def test_open_partial_fill_counted():
    """GPT round-4 gap: an OPEN order mid partial-fill is live exposure —
    count the filled part; the level is already spoken for."""
    trades = [_trade([_order(status="open", filled=0.5, remaining=0.5)])]
    res = filled_prices_for_pair(trades, pair=PAIR, plan_id=PLAN_ID, level_prices=LEVELS)
    assert res.prices == [95.0]
    assert res.partial_qty == pytest.approx(0.5)


def test_open_order_zero_fill_not_counted():
    """An open order with nothing filled yet is not a fill."""
    trades = [_trade([_order(status="open", filled=0.0, remaining=1.0)])]
    res = filled_prices_for_pair(trades, pair=PAIR, plan_id=PLAN_ID, level_prices=LEVELS)
    assert res.prices == []
    assert res.partial_qty == pytest.approx(0.0)


def test_canceled_zero_fill_not_counted():
    """Canceled with zero fill -> never traded -> not counted."""
    trades = [_trade([_order(status="canceled", filled=0.0)])]
    res = filled_prices_for_pair(trades, pair=PAIR, plan_id=PLAN_ID, level_prices=LEVELS)
    assert res.prices == []


def test_rejected_order_never_counted():
    """Statuses outside the confirmed set (rejected etc.) never count —
    even if a buggy upstream reports filled>0."""
    trades = [_trade([_order(status="rejected", filled=1.0)])]
    res = filled_prices_for_pair(trades, pair=PAIR, plan_id=PLAN_ID, level_prices=LEVELS)
    assert res.prices == []


# =============================================================== idempotent ledger
def test_ledger_apply_returns_only_the_delta():
    """cumulative_filled_qty per order_id: apply returns the INCREMENT.
    open 0.3 -> +0.3, same order at 0.6 -> +0.3 more (not +0.6)."""
    ledger = OrderFillLedger()
    assert ledger.apply("o1", 0.3) == pytest.approx(0.3)
    assert ledger.apply("o1", 0.6) == pytest.approx(0.3)
    assert ledger.apply("o1", 0.6) == pytest.approx(0.0)  # repeat payload
    assert ledger.apply("o2", 1.0) == pytest.approx(1.0)  # other order independent
    assert ledger.cumulative("o1") == pytest.approx(0.6)
    assert ledger.cumulative("o2") == pytest.approx(1.0)


def test_ledger_never_decrements_on_stale_payload():
    """A stale payload reporting less filled must not rewind the ledger."""
    ledger = OrderFillLedger()
    ledger.apply("o1", 0.9)
    assert ledger.apply("o1", 0.4) == pytest.approx(0.0)
    assert ledger.cumulative("o1") == pytest.approx(0.9)


def test_fills_result_order_qty_is_cumulative_per_order():
    """order_qty reflects cumulative filled per order_id across the call."""
    trades = [
        _trade(
            [
                _order(order_id="a", status="open", filled=0.3),
                _order(order_id="b", status="canceled", filled=0.4),
                _order(order_id="c", status="closed", filled=1.0),
            ]
        )
    ]
    res = filled_prices_for_pair(trades, pair=PAIR, plan_id=PLAN_ID, level_prices=LEVELS)
    assert res.order_qty == {"a": 0.3, "b": 0.4, "c": 1.0}
    # open partial contributes to partial_qty; closed/canceled do not
    assert res.partial_qty == pytest.approx(0.3)


def test_repeat_poll_never_double_counts_via_ledger():
    """REST returns the same order book twice: with a shared ledger the
    second pass applies ZERO new qty — the price set stays stable too."""
    ledger = OrderFillLedger()
    trades = [_trade([_order(order_id="a", status="open", filled=0.5)])]
    res1 = filled_prices_for_pair(
        trades, pair=PAIR, plan_id=PLAN_ID, level_prices=LEVELS, ledger=ledger
    )
    assert res1.order_qty == {"a": 0.5}
    res2 = filled_prices_for_pair(
        trades, pair=PAIR, plan_id=PLAN_ID, level_prices=LEVELS, ledger=ledger
    )
    assert res2.prices == [95.0]
    assert res2.order_qty == {"a": 0.5}  # cumulative, not 0.5+0.5
    assert ledger.cumulative("a") == pytest.approx(0.5)


def test_ledger_accumulates_partial_progression():
    """Same order reported open at 0.3, then 0.6, then closed at 1.0 —
    cumulative climbs monotonically, never exceeding realized fills."""
    ledger = OrderFillLedger()
    for status, filled in (("open", 0.3), ("open", 0.6), ("closed", 1.0)):
        res = filled_prices_for_pair(
            [_trade([_order(order_id="a", status=status, filled=filled)])],
            pair=PAIR,
            plan_id=PLAN_ID,
            level_prices=LEVELS,
            ledger=ledger,
        )
    assert res.order_qty == {"a": 1.0}
    assert ledger.cumulative("a") == pytest.approx(1.0)


# =============================================================== reconciliation semantics
def test_unmapped_partial_fill_is_flagged_not_silently_dropped():
    """A real partial fill that maps to NO level is reconciliation
    failure -> counted unmapped (the publisher must lock entries for the
    pair; protective exits are untouched — gated in publish.py)."""
    trades = [_trade([_order(status="open", filled=0.5, price=77.7, average=77.7)])]
    res = filled_prices_for_pair(trades, pair=PAIR, plan_id=PLAN_ID, level_prices=LEVELS)
    assert res.unmapped_orders == 1
    assert res.prices == []
