"""R14c: real Freqtrade initial-entry and DCA order tag contract."""

import pytest
from aigrid.state.ft_fills import FillsResult, OrderFillLedger, filled_prices_for_pair

PAIR = "SOL/USDT:USDT"


def _order(oid, tag, price=99.0, *, status="closed", filled=1.0, entry=True):
    return {
        "order_id": oid,
        "ft_order_tag": tag,
        "average": price,
        "status": status,
        "filled": filled,
        "ft_is_entry": entry,
    }


def _read(orders, *, trade_tag="aigrid_p", ledger=None):
    return filled_prices_for_pair(
        [{"pair": PAIR, "enter_tag": trade_tag, "orders": orders}],
        pair=PAIR,
        plan_id="p",
        level_prices=[100.0, 99.0, 98.0],
        price_tick=0.01,
        ledger=ledger,
    )


def test_a_initial_entry_and_two_dca_adds():
    result = _read(
        [
            _order("initial", "aigrid_p", 100.0),
            _order("add1", "grid_level_1", 99.0),
            _order("add2", "grid_level_2", 98.0),
        ]
    )
    assert result == FillsResult(
        prices=[98.0, 99.0, 100.0],
        order_qty={"initial": 1.0, "add1": 1.0, "add2": 1.0},
    )


@pytest.mark.parametrize("trade_tag", ["aigrid_other", None, "aigrid_p:other"])
def test_b_dca_requires_exact_trade_enter_tag(trade_tag):
    assert _read([_order("other", "grid_level_1")], trade_tag=trade_tag) == FillsResult()


@pytest.mark.parametrize(
    "tag",
    [
        "aigrid_other_plan",
        "aigrid_p_other",
        "grid_level_",
        "grid_level_-1",
        "grid_level_1extra",
        "grid_level_1\n",
        None,
        1,
    ],
)
def test_c_other_tags_excluded_from_our_trade(tag):
    assert _read([_order("other", tag)]) == FillsResult()


def test_d_dca_statuses_partial_qty_and_cumulative_ledger():
    orders = [
        _order("canceled", "grid_level_1", status="canceled", filled=0.5),
        _order("expired", "grid_level_2", 98.0, status="expired", filled=0.25),
        _order("open", "grid_level_2", 98.0, status="open", filled=0.3),
        _order("zero", "grid_level_1", filled=0.0),
        _order("exit", "grid_level_1", entry=False),
        _order("rejected", "grid_level_1", status="rejected"),
    ]
    ledger = OrderFillLedger()
    ledger.apply("canceled", 0.75)
    expected = FillsResult(
        prices=[98.0, 99.0],
        order_qty={"canceled": 0.75, "expired": 0.25, "open": 0.3},
        partial_qty=0.3,
    )
    assert _read(orders, ledger=ledger) == expected
    assert _read(orders, ledger=ledger) == expected
    assert ledger.apply("open", 0.3) == 0.0


@pytest.mark.parametrize("trade_tag", ["aigrid_p", "aigrid_other", None])
@pytest.mark.parametrize("tag", ["aigrid_p", "aigrid_p:grid_level_3"])
def test_e_plan_order_tags_accepted_independently_of_trade_tag(tag, trade_tag):
    assert _read([_order("ours", tag)], trade_tag=trade_tag) == FillsResult(
        prices=[99.0], order_qty={"ours": 1.0}
    )


def test_f_dca_outside_half_tick_is_unmapped():
    result = _read(
        [
            _order("near", "grid_level_1", 99.004),
            _order("far", "grid_level_2", 98.006),
        ]
    )
    assert result == FillsResult(
        prices=[99.0], unmapped_orders=1, order_qty={"near": 1.0, "far": 1.0}
    )
