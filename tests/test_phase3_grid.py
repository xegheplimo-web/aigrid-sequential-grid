"""PHASE 3 tests: VirtualGridEngine + executors."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from aigrid.grid.engine import ActionKind, VirtualGridEngine
from aigrid.grid.executor import ShadowExecutor
from aigrid.plans.schema import AIGridPlan, GridMode, draft_plan_from_assessment

NOW = datetime.now(UTC)


def _range_plan(**kw) -> AIGridPlan:
    base = dict(
        plan_id="plan-test000001",
        plan_version=1,
        symbol="ETH/USDT:USDT",
        created_at=NOW,
        expires_at=NOW + timedelta(minutes=45),
        market_regime="RANGE",
        confidence=0.7,
        grid_mode=GridMode.RANGE_GRID,
        grid_lower=2950.0,
        grid_upper=3050.0,
        grid_levels=6,
        grid_spacing=0.01,
        # consistent: notional (240) = margin (120) x leverage (2)
        max_capital=120.0,
        max_position_size=240.0,
        max_leverage=2.0,
        stop_loss=2900.0,
        take_profit=None,
        invalidation_conditions=["PRICE_BREAKS_GRID_BOUNDARY"],
        risk_level="MEDIUM",
        reasoning_summary="t",
    )
    base.update(kw)
    return AIGridPlan(**base)


def _engine(plan=None, **kw) -> VirtualGridEngine:
    plan = plan or _range_plan()
    defaults = dict(
        fee_rate=0.0002, slippage_rate=0.0005, leg_stop_loss_pct=0.03, max_inventory_notional=240.0
    )
    defaults.update(kw)
    return VirtualGridEngine(plan, **defaults)


# ------------------------------------------------------------------ levels
def test_levels_ordered_and_counted():
    e = _engine()
    assert e.state.levels == [3050.0, 3030.0, 3010.0, 2990.0, 2970.0, 2950.0]


def test_safe_hold_emits_no_actions():
    plan = draft_plan_from_assessment(
        "ETH/USDT:USDT",
        "HIGH_VOLATILITY",
        0.4,
        3000.0,
        20.0,
        grid_mode="SAFE_HOLD",
        ttl_minutes=45,
        margin_usdt=0.0,
        leverage=1.0,
        levels=2,
    )
    e = _engine(plan)
    assert e.on_candle(0, 3000.0, 3100, 2900, 3000) == []
    assert e.state.halted is False


# ------------------------------------------------------------------ entries
def test_entry_on_level_touch_and_budget_cap():
    e = _engine()
    # candle opens 3010, dips to 2989 -> crosses DOWN through 2990 only
    actions = e.on_candle(0, 3010.0, 3010, 2989, 2995)
    buys = [a for a in actions if a.kind == ActionKind.BUY]
    assert len(buys) == 1
    assert buys[0].price == 2990.0
    assert buys[0].client_order_id.endswith(":ENTRY:3")
    # dip to 2900 touches 2970, 2950 AND plan SL 2900 -> close_all fires first
    actions2 = e.on_candle(1, 2995.0, 2995, 2895, 2920)
    kinds = {a.kind for a in actions2}
    assert kinds == {ActionKind.SELL}  # close-all via plan SL
    assert e.state.halted and e.state.halt_reason == "PLAN_STOP_LOSS_TRIGGERED"


def test_no_duplicate_entry_same_level():
    e = _engine()
    e.on_candle(0, 3010.0, 3010, 2989, 2995)  # fill level 3 (2990)
    actions = e.on_candle(1, 2995.0, 2995, 2985, 2990)  # touches 2990 again
    buys = [a for a in actions if a.kind == ActionKind.BUY and a.level == 3]
    assert buys == []  # duplicate prevented


def test_level_rearms_after_tp():
    """Classic grid: after a level's lot is TP'd, the level can fill again."""
    e = _engine()
    e.on_candle(0, 3010.0, 3010, 2989, 2995)  # fill 2990
    lot = e.state.lots[3]
    tp = lot.entry_price * 1.01
    e.on_candle(1, tp + 1, tp + 1, 2995, tp)  # TP out
    assert 3 not in e.state.lots
    actions = e.on_candle(2, tp, tp, 2989, 2990)  # price falls back through 2990
    buys = [a for a in actions if a.kind == ActionKind.BUY and a.level == 3]
    assert len(buys) == 1  # re-armed


def test_gap_through_multiple_levels_buys_all_within_budget():
    e = _engine()
    # gap: open 3000 -> low 2940 crosses 2990, 2970, 2950 (3 levels x 40 = 120 <= 240)
    actions = e.on_candle(0, 3000.0, 3000, 2940, 2975)
    buys = [a for a in actions if a.kind == ActionKind.BUY]
    assert len(buys) == 3
    assert {b.price for b in buys} == {2990.0, 2970.0, 2950.0}


def test_gap_beyond_budget_skips_extra_levels():
    plan = _range_plan(max_position_size=120.0)
    e = VirtualGridEngine(
        plan,
        fee_rate=0.0002,
        slippage_rate=0.0005,
        leg_stop_loss_pct=0.03,
        max_inventory_notional=120.0,
    )
    # low=2940 crosses 2990,2970,2950; budget=120 -> exactly 3 x 40 = 120 OK
    actions = e.on_candle(0, 3000.0, 3000, 2940, 2975)
    assert len([a for a in actions if a.kind == ActionKind.BUY]) == 3
    # next candle dips further: no budget left, no new entries
    actions2 = e.on_candle(1, 2975.0, 2975, 2935, 2945)
    assert [a for a in actions2 if a.kind == ActionKind.BUY] == []


# ------------------------------------------------------------------ exits
def test_tp_per_level():
    e = _engine()
    e.on_candle(0, 3010.0, 3010, 2989, 2995)  # fill 2990 (+slippage)
    lot = e.state.lots[3]
    assert lot.entry_price == pytest.approx(2990 * 1.0005)
    # rally on a LATER candle (low stays above the level: no re-cross)
    tp = lot.entry_price * 1.01
    actions = e.on_candle(1, tp + 1, tp + 1, 2996, tp)
    sells = [a for a in actions if a.kind == ActionKind.SELL]
    assert len(sells) == 1 and sells[0].reason == "LEG_TAKE_PROFIT"
    assert 3 not in e.state.lots
    assert e.state.realized_pnl > 0


def test_no_same_candle_fill_and_tp():
    """Pessimistic: a candle that fills a level cannot also TP it."""
    e = _engine()
    # huge candle: dips through 2990 (fill) AND rallies to 3050 (would-be TP)
    actions = e.on_candle(0, 2995.0, 3050, 2989, 3050)
    buys = [a for a in actions if a.kind == ActionKind.BUY]
    sells = [a for a in actions if a.kind == ActionKind.SELL]
    assert len(buys) >= 1
    assert all(s.reason != "LEG_TAKE_PROFIT" for s in sells)


def test_leg_stop_loss_trumps_tp():
    e = _engine(leg_stop_loss_pct=0.03)
    e.on_candle(0, 3010.0, 3010, 2989, 2995)
    lot = e.state.lots[3]
    # candle where BOTH tp and leg-sl are hit: high >= tp, low <= entry*(1-0.03)
    tp = lot.entry_price * 1.01
    sl = lot.entry_price * 0.97
    actions = e.on_candle(1, tp, tp, sl, sl)  # close at sl
    sells = [a for a in actions if a.kind == ActionKind.SELL]
    assert sells[0].reason == "LEG_STOP_LOSS"  # SL wins
    assert e.state.realized_pnl < 0


# ------------------------------------------------------------------ short grid
def test_short_grid_mirrored():
    plan = _range_plan(
        grid_mode=GridMode.SHORT_GRID, grid_lower=2950.0, grid_upper=3050.0, stop_loss=3100.0
    )
    e = _engine(plan)
    assert e.is_short
    # rally: open 3040 crosses UP through 3050
    actions = e.on_candle(0, 3040.0, 3051, 2990, 3040)
    sells = [a for a in actions if a.kind == ActionKind.SELL]
    assert len(sells) == 1 and sells[0].price == 3050.0
    # drop to tp -> cover (BUY) at entry*(1-spacing)
    lot = e.state.lots[0]
    tp = lot.entry_price * (1 - plan.grid_spacing)
    actions2 = e.on_candle(1, 3040.0, 3040, tp - 1, tp)
    covers = [a for a in actions2 if a.kind == ActionKind.BUY]
    assert len(covers) == 1 and covers[0].reason == "LEG_TAKE_PROFIT"
    assert e.state.realized_pnl > 0


def test_short_grid_plan_sl_above_grid():
    plan = _range_plan(grid_mode=GridMode.SHORT_GRID, stop_loss=3100.0)
    e = _engine(plan)
    actions = e.on_candle(0, 3000.0, 3105, 3000, 3080)  # high >= 3100 -> close all
    assert e.state.halted and e.state.halt_reason == "PLAN_STOP_LOSS_TRIGGERED"


# ------------------------------------------------------------------ invalidation
def test_close_below_grid_halts_grid():
    e = _engine()
    actions = e.on_candle(0, 3000.0, 3050, 2960, 2940)  # close < grid_lower
    assert e.state.halted
    assert e.state.halt_reason == "PRICE_BROKE_GRID_BOUNDARY"
    cancels = [a for a in actions if a.kind == ActionKind.CANCEL_PENDING]
    assert len(cancels) == 1


def test_halted_engine_emits_nothing():
    e = _engine()
    e.on_candle(0, 3000.0, 3050, 2960, 2940)
    assert e.on_candle(1, 2940.0, 3050, 2940, 2950) == []


# ------------------------------------------------------------------ idempotency
def test_deterministic_order_ids():
    e1, e2 = _engine(), _engine()
    a1 = e1.on_candle(0, 3010.0, 3010, 2989, 2995)
    a2 = e2.on_candle(0, 3010.0, 3010, 2989, 2995)
    assert [a.client_order_id for a in a1] == [a.client_order_id for a in a2]


def test_apply_external_fill_idempotent():
    e = _engine()
    e.on_candle(0, 3010.0, 3010, 2989, 2995)
    oid = e.order_id(3, "ENTRY")
    before = dict(e.state.lots)
    assert e.apply_external_fill(oid, 2990.0, e.state.lots[3].qty) is True
    assert e.state.lots == before  # duplicate feedback: no change
    assert e.apply_external_fill("plan-x:grid:ENTRY:99", 2990.0, 0.01) is True


# ------------------------------------------------------------------ shadow executor
def test_shadow_executor_records_fills_no_orders():
    e = _engine()
    actions = e.on_candle(0, 3010.0, 3010, 2989, 2995)
    ex = ShadowExecutor(e)
    fills = ex.execute(actions, candle_ms=123)
    assert len(fills) == 1
    assert ex.stats.orders_submitted == 0  # NEVER submits real orders
    assert fills[0].side == "BUY"


def test_engine_summary_shape():
    e = _engine()
    e.on_candle(0, 3010.0, 3010, 2989, 2995)
    d = e.to_dict(2995.0)
    assert d["plan_id"] == "plan-test000001"
    assert "open_lots" in d and "realized_pnl" in d and "total_fees" in d
