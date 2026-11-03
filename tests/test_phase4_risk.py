"""PHASE 4 tests: independent Risk Engine + Symbol Ownership Registry."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from aigrid.plans.schema import AIGridPlan, GridMode
from aigrid.risk.engine import AccountState, MarketFreshness, RiskEngine
from aigrid.risk.events import RiskEvent
from aigrid.risk.ownership import OwnershipRegistry, OwnerState
from aigrid.settings import RiskLimits

NOW = datetime.now(UTC)


def _plan(**kw) -> AIGridPlan:
    base = dict(
        plan_id="plan-risktest0001",
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
        # consistent: notional (200) = margin (100) x leverage (2)
        max_capital=100.0,
        max_position_size=200.0,
        max_leverage=2.0,
        stop_loss=2900.0,
        take_profit=None,
        invalidation_conditions=[],
        risk_level="MEDIUM",
        reasoning_summary="t",
    )
    base.update(kw)
    return AIGridPlan(**base)


def _account(**kw) -> AccountState:
    base = dict(
        equity_usdt=1000.0,
        starting_equity_usdt=1000.0,
        peak_equity_usdt=1000.0,
        realized_pnl_today_usdt=0.0,
        total_exposure_usdt=0.0,
        free_margin_usdt=800.0,
    )
    base.update(kw)
    return AccountState(**base)


@pytest.fixture
def limits():
    return RiskLimits()


@pytest.fixture
def engine(limits):
    return RiskEngine(limits)


# ------------------------------------------------------------------ entry veto
def test_entry_allowed_when_all_healthy(engine):
    d = engine.evaluate_entry(_plan(), _account())
    assert d.allowed and d.events == ()


def test_entry_vetoed_when_plan_expired(engine):
    p = _plan(created_at=NOW - timedelta(hours=2), expires_at=NOW - timedelta(minutes=30))
    d = engine.evaluate_entry(p, _account())
    assert not d.allowed
    assert str(RiskEvent.AI_PLAN_EXPIRED) in d.events


def test_entry_vetoed_on_exposure(engine):
    d = engine.evaluate_entry(_plan(max_position_size=400.0), _account(total_exposure_usdt=200.0))
    assert not d.allowed
    assert str(RiskEvent.EXPOSURE_LIMIT) in d.events


def test_entry_vetoed_on_leverage(engine):
    d = engine.evaluate_entry(_plan(max_leverage=8.0), _account())
    assert not d.allowed
    assert str(RiskEvent.LEVERAGE_LIMIT) in d.events


def test_entry_vetoed_on_daily_loss(engine):
    d = engine.evaluate_entry(_plan(), _account(realized_pnl_today_usdt=-30.0))  # -3% > 2%
    assert not d.allowed
    assert str(RiskEvent.DAILY_LOSS_LIMIT) in d.events


def test_entry_vetoed_on_drawdown(engine):
    d = engine.evaluate_entry(_plan(), _account(equity_usdt=900.0, peak_equity_usdt=1000.0))
    assert not d.allowed
    assert str(RiskEvent.DRAWDOWN_LIMIT) in d.events


def test_entry_vetoed_on_stale_data(engine):
    fr = MarketFreshness(last_candle_close_ms=0, now_ms=10**12, timeframe_seconds=900)
    d = engine.evaluate_entry(_plan(), _account(), freshness=fr)
    assert not d.allowed
    assert str(RiskEvent.DATA_STALE) in d.events


def test_entry_vetoed_on_funding_cost(engine):
    d = engine.evaluate_entry(_plan(), _account(funding_cost_today_usdt=6.0))
    assert not d.allowed
    assert str(RiskEvent.FUNDING_LIMIT_EXCEEDED) in d.events


def test_entry_vetoed_on_low_margin(engine):
    d = engine.evaluate_entry(_plan(), _account(free_margin_usdt=100.0))
    assert not d.allowed
    assert str(RiskEvent.MARGIN_LOW) in d.events


def test_entry_vetoed_on_duplicate_plan_order(engine):
    d = engine.evaluate_entry(_plan(), _account(), existing_open_orders=("plan-risktest0001",))
    assert not d.allowed
    assert str(RiskEvent.DUP_ORDER_BLOCKED) in d.events


def test_risk_engine_works_with_ai_dead(engine):
    """The engine's account checks never consult the AI — AI down changes nothing."""
    d = engine.evaluate_account(_account(realized_pnl_today_usdt=-25.0))
    assert not d.allowed
    assert d.emergency_action == "CLOSE_ALL"
    assert str(RiskEvent.MAX_LOSS_REACHED) in d.events


def test_emergency_reduce_on_drawdown(engine):
    d = engine.evaluate_account(_account(equity_usdt=910.0, peak_equity_usdt=1000.0))
    assert d.emergency_action == "REDUCE"
    assert str(RiskEvent.DRAWDOWN_LIMIT) in d.events


def test_position_mismatch_detection(engine):
    d = engine.check_position_mismatch(
        expected_pairs=set(), actual_pairs={"ETHUSDT"}, symbol="ETHUSDT"
    )
    assert not d.allowed
    assert str(RiskEvent.POSITION_MISMATCH) in d.events
    assert d.emergency_action == "CLOSE_ALL"


def test_fresh_data_no_veto(engine):
    import time

    now_ms = int(time.time() * 1000)
    fr = MarketFreshness(last_candle_close_ms=now_ms - 60_000, now_ms=now_ms, timeframe_seconds=900)
    d = engine.evaluate_entry(_plan(), _account(), freshness=fr)
    assert d.allowed


# ------------------------------------------------------------------ ownership
def test_ownership_acquire_and_conflict():
    reg = OwnershipRegistry()
    ok, _ = reg.acquire("ETH/USDT:USDT", "grid", "grid:plan-1")
    assert ok and reg.state_of("ETH/USDT:USDT") == OwnerState.GRID_OWNED
    ok2, reason = reg.acquire("ETH/USDT:USDT", "trend", "trend:V19")
    assert not ok2 and reason == str(RiskEvent.OWNERSHIP_CONFLICT)
    ok3, _ = reg.can_open("ETH/USDT:USDT", "trend")
    assert not ok3
    ok4, _ = reg.can_open("ETH/USDT:USDT", "grid")
    assert ok4  # same family may manage


def test_ownership_release_and_reacquire():
    reg = OwnershipRegistry()
    reg.acquire("ETH/USDT:USDT", "grid", "grid:plan-1")
    ok, _ = reg.release("ETH/USDT:USDT", "grid:plan-1")
    assert ok and reg.state_of("ETH/USDT:USDT") == OwnerState.FREE
    ok2, _ = reg.acquire("ETH/USDT:USDT", "trend", "trend:V19")
    assert ok2 and reg.state_of("ETH/USDT:USDT") == OwnerState.TREND_OWNED


def test_ownership_transition_flow():
    reg = OwnershipRegistry()
    reg.acquire("ETH/USDT:USDT", "grid", "grid:plan-1")
    ok, _ = reg.begin_transition("ETH/USDT:USDT")
    assert ok and reg.state_of("ETH/USDT:USDT") == OwnerState.TRANSITIONING
    ok2, _ = reg.complete_transition("ETH/USDT:USDT", "trend", "trend:V19")
    assert ok2 and reg.state_of("ETH/USDT:USDT") == OwnerState.TREND_OWNED


def test_ownership_risk_lock_blocks_everything():
    reg = OwnershipRegistry()
    reg.acquire("ETH/USDT:USDT", "grid", "grid:plan-1")
    ok, _ = reg.risk_lock("ETH/USDT:USDT", "MAX_LOSS_REACHED")
    assert ok and reg.state_of("ETH/USDT:USDT") == OwnerState.RISK_LOCKED
    ok2, reason = reg.release("ETH/USDT:USDT", "grid:plan-1")
    assert not ok2  # only risk engine unlocks
    ok3, _ = reg.can_open("ETH/USDT:USDT", "grid")
    assert not ok3
    ok4, _ = reg.risk_unlock("ETH/USDT:USDT")
    assert ok4 and reg.state_of("ETH/USDT:USDT") == OwnerState.FREE


def test_ownership_illegal_transition_rejected():
    reg = OwnershipRegistry()
    ok, _ = reg.begin_transition("ETH/USDT:USDT")  # from FREE -> illegal
    assert not ok


def test_risk_engine_locks_symbol_via_ownership(limits):
    engine = RiskEngine(limits)
    engine.lock_symbol("ETH/USDT:USDT", "TEST_LOCK")
    ok, _ = engine.ownership.can_open("ETH/USDT:USDT", "grid")
    assert not ok
    engine.unlock_symbol("ETH/USDT:USDT")
    ok2, _ = engine.ownership.can_open("ETH/USDT:USDT", "grid")
    assert ok2
