"""R6 tests — T-213b: exit fee must match the order type.

TP exits are limit fills -> maker fee_rate_exit. Leg-SL, plan-SL and
CLOSE_ALL are market exits -> taker fee_rate_emergency. The legacy single
fee_rate kwarg still defaults all three rates (backward compatible).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from aigrid.grid.engine import ActionKind, GridLot, VirtualGridEngine
from aigrid.plans.schema import AIGridPlan, GridMode

NOW = datetime.now(UTC)


def _plan(**kw) -> AIGridPlan:
    base = dict(
        plan_id="plan-r6fee00001",
        plan_version=1,
        symbol="ETH/USDT:USDT",
        created_at=NOW,
        expires_at=NOW + timedelta(minutes=45),
        market_regime="RANGE",
        confidence=0.7,
        grid_mode=GridMode.RANGE_GRID,
        grid_lower=94.0,
        grid_upper=100.0,
        grid_levels=4,
        grid_spacing=0.01,
        # consistent: notional (200) = margin (100) x leverage (2)
        max_capital=100.0,
        max_position_size=200.0,
        max_leverage=2.0,
        level_stake=25.0,
        level_notional=50.0,
        stop_loss=90.0,
        take_profit=None,
        invalidation_conditions=[],
        risk_level="MEDIUM",
        reasoning_summary="t",
    )
    base.update(kw)
    return AIGridPlan(**base)


def _engine(plan=None, **kw) -> VirtualGridEngine:
    plan = plan or _plan()
    defaults = dict(
        fee_rate_entry=0.0002,
        fee_rate_exit=0.0002,
        fee_rate_emergency=0.0005,
        slippage_rate=0.0,
        leg_stop_loss_pct=0.03,
        max_inventory_notional=200.0,
    )
    defaults.update(kw)
    return VirtualGridEngine(plan, **defaults)


def _seed_lot(
    e: VirtualGridEngine, level: int, entry: float, qty: float, entry_fee: float = 0.0
) -> None:
    e.state.lots[level] = GridLot(
        level, entry_price=entry, qty=qty, entry_fee=entry_fee, entry_candle_ms=-1
    )


# ------------------------------------------------------------------ per-path fees
def test_tp_exit_fee_is_maker():
    """100 USDT notional leaving via TP pays maker: 100 * 0.0002."""
    e = _engine()  # spacing 0.01 -> tp = entry * 1.01
    _seed_lot(e, 0, entry=100.0 / 1.01, qty=1.0)
    actions = e.on_candle(0, 100.5, 100.5, 99.0, 100.0, permit_entry=False)
    sells = [a for a in actions if a.kind == ActionKind.SELL]
    assert sells[0].reason == "LEG_TAKE_PROFIT"
    assert e.state.total_fees == pytest.approx(0.02)  # 100 * 0.0002


def test_leg_sl_exit_fee_is_taker():
    """100 USDT notional leaving via leg-SL pays taker: 100 * 0.0005."""
    e = _engine()  # leg_sl 0.03 -> sl = entry * 0.97
    _seed_lot(e, 0, entry=100.0 / 0.97, qty=1.0)
    actions = e.on_candle(0, 102.0, 102.0, 99.5, 100.0, permit_entry=False)
    sells = [a for a in actions if a.kind == ActionKind.SELL]
    assert sells[0].reason == "LEG_STOP_LOSS"
    assert e.state.total_fees == pytest.approx(0.05)  # 100 * 0.0005


def test_close_all_uses_taker():
    """PLAN_STOP_LOSS_TRIGGERED closes everything at the emergency (taker) rate."""
    e = _engine()
    _seed_lot(e, 0, entry=97.0, qty=1.0)
    _seed_lot(e, 1, entry=95.0, qty=2.0)
    e.on_candle(0, 95.0, 96.0, 89.0, 91.0, permit_entry=False)
    assert e.state.halted and e.state.halt_reason == "PLAN_STOP_LOSS_TRIGGERED"
    # exits at close 91.0: gross = 91 * (1 + 2) = 273 -> fee = 273 * 0.0005
    assert e.state.total_fees == pytest.approx(273.0 * 0.0005)


def test_entry_fee_is_maker_entry_rate():
    e = _engine()
    e.on_candle(0, 97.5, 98.0, 95.5, 97.0)  # crosses 96 only -> qty = 50/96
    # entry exec = 96.0 (slippage 0); gross = 50.0 -> fee = 50 * 0.0002
    assert e.state.total_fees == pytest.approx(0.01)


# ------------------------------------------------------------------ economics
def test_tp_cycle_profitable_at_018pct_spacing():
    """spacing 0.18% with a maker round-trip stays net-positive (audit example)."""
    plan = _plan(grid_spacing=0.0018)
    e = _engine(plan)
    qty = 50.0 / 96.0
    entry_fee = qty * 96.0 * 0.0002
    _seed_lot(e, 2, entry=96.0, qty=qty, entry_fee=entry_fee)
    e.state.total_fees += entry_fee
    e.on_candle(0, 96.5, 96.4, 96.0, 96.2, permit_entry=False)  # tp = 96.1728
    assert e.state.realized_pnl == pytest.approx(
        (96.1728 - 96.0) * qty - 96.1728 * qty * 0.0002 - entry_fee
    )
    assert e.state.realized_pnl > 0


def test_sl_exit_costs_exactly_003pct_more_than_maker():
    """Same exit at the same price: taker instead of maker lowers net pnl by
    exactly (0.0005 - 0.0002) x notional = 0.03%."""
    plan = _plan()
    qty, entry, sl = 50.0 / 96.0, 96.0, 96.0 * 0.97  # sl = 93.12
    e_taker = _engine(plan)  # emergency = 0.0005
    e_maker = VirtualGridEngine(  # all rates = maker
        plan,
        fee_rate=0.0002,
        slippage_rate=0.0,
        leg_stop_loss_pct=0.03,
        max_inventory_notional=200.0,
    )
    entry_fee = qty * entry * 0.0002
    for e in (e_taker, e_maker):
        _seed_lot(e, 0, entry=entry, qty=qty, entry_fee=entry_fee)
        e.state.total_fees += entry_fee
        e.on_candle(0, 95.0, 95.5, 92.5, 93.5, permit_entry=False)
    notional = qty * sl
    assert e_maker.state.realized_pnl - e_taker.state.realized_pnl == pytest.approx(
        0.0003 * notional
    )
    assert e_taker.state.realized_pnl < e_maker.state.realized_pnl


# ------------------------------------------------------------------ backward compat
def test_legacy_fee_rate_kwarg_defaults_all_three_rates():
    plan = _plan()
    e = VirtualGridEngine(
        plan,
        fee_rate=0.0002,
        slippage_rate=0.0005,
        leg_stop_loss_pct=0.03,
        max_inventory_notional=200.0,
    )
    assert e.fee_rate == 0.0002
    assert e.fee_rate_entry == 0.0002
    assert e.fee_rate_exit == 0.0002
    assert e.fee_rate_emergency == 0.0002
