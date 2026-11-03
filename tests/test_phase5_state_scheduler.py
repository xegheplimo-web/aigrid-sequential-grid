"""PHASE 5 tests: state store (atomic + idempotency), restart recovery, scheduler."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from aigrid.plans.schema import AIGridPlan, GridMode
from aigrid.scheduler import AnalysisRequest, AnalysisScheduler, TriggerKind
from aigrid.settings import SchedulerSettings
from aigrid.state.reconcile import RestartReconciler
from aigrid.state.store import EventLog, StateStore

NOW = datetime.now(UTC)


@pytest.fixture
def state_dir(tmp_path):
    return tmp_path / "state"


# ------------------------------------------------------------------ state store
def test_atomic_save_and_load(state_dir):
    store = StateStore(state_dir)
    store.save_state("grid_state", {"open_lots": {"3": {"entry": 2990.0}}, "halted": False})
    loaded = store.load_state("grid_state")
    assert loaded["open_lots"]["3"]["entry"] == 2990.0


def test_load_missing_returns_none(state_dir):
    assert StateStore(state_dir).load_state("nope") is None


def test_load_corrupt_returns_none_fail_closed(state_dir):
    store = StateStore(state_dir)
    (state_dir / "grid_state.json").write_text("{corrupt json", encoding="utf-8")
    assert store.load_state("grid_state") is None


def test_no_temp_files_left_behind(state_dir):
    store = StateStore(state_dir)
    store.save_state("x", {"a": 1})
    leftovers = [p for p in state_dir.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


# ------------------------------------------------------------------ event log + idempotency
def test_event_log_append_and_query(state_dir):
    log = EventLog(state_dir)
    log.log("GRID_ACTION", "ETH/USDT:USDT", {"level": 3})
    log.log("RISK_EVENT", "ETH/USDT:USDT", {"event": "MAX_LOSS_REACHED"})
    rows = log.recent(limit=10)
    assert len(rows) == 2
    assert rows[0]["kind"] == "RISK_EVENT"  # newest first
    assert rows[1]["payload"]["level"] == 3
    log.close()


def test_idempotency_mark_and_seen(state_dir):
    log = EventLog(state_dir)
    assert log.mark_seen("plan-x:grid:ENTRY:3") is True
    assert log.mark_seen("plan-x:grid:ENTRY:3") is False  # duplicate dropped
    assert log.seen("plan-x:grid:ENTRY:3") is True
    assert log.seen("plan-x:grid:ENTRY:4") is False
    log.close()


# ------------------------------------------------------------------ reconciler
def _plan(**kw) -> AIGridPlan:
    base = dict(
        plan_id="plan-recon000001",
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


def test_reconcile_fresh_start_no_state(state_dir):
    rec = RestartReconciler(StateStore(state_dir))
    r = rec.recover(actual_open_pairs=set(), actual_open_order_ids=set(), now_ms=0)
    assert r.status == "FRESH_START"


def test_reconcile_unexpected_position_fails_closed(state_dir):
    rec = RestartReconciler(StateStore(state_dir))
    r = rec.recover(actual_open_pairs={"ETHUSDT"}, actual_open_order_ids=set(), now_ms=0)
    assert r.status == "SAFE_HOLD"
    assert "POSITION_MISMATCH" in r.reasons


def test_reconcile_restores_valid_plan_and_lots(state_dir):
    store = StateStore(state_dir)
    plan = _plan()
    store.save_state(
        "current_plan", {"plan": plan.model_dump(mode="json"), "symbol": "ETH/USDT:USDT"}
    )
    store.save_state("grid_state", {"open_lots": {"3": {"entry": 2990.0, "qty": 0.013}}})
    rec = RestartReconciler(store)
    r = rec.recover(actual_open_pairs={"ETHUSDT"}, actual_open_order_ids=set(), now_ms=0)
    assert r.status == "RESTORED"
    assert r.plan.plan_id == "plan-recon000001"
    assert r.restored_lots["3"]["entry"] == 2990.0


def test_reconcile_expired_plan_with_positions_goes_safe_hold(state_dir):
    store = StateStore(state_dir)
    plan = _plan(created_at=NOW - timedelta(hours=2), expires_at=NOW - timedelta(minutes=30))
    store.save_state(
        "current_plan", {"plan": plan.model_dump(mode="json"), "symbol": "ETH/USDT:USDT"}
    )
    store.save_state("grid_state", {"open_lots": {"3": {"entry": 2990.0, "qty": 0.013}}})
    rec = RestartReconciler(store)
    r = rec.recover(actual_open_pairs={"ETHUSDT"}, actual_open_order_ids=set(), now_ms=0)
    assert r.status == "SAFE_HOLD"
    assert "AI_PLAN_EXPIRED" in r.reasons


def test_reconcile_lots_without_ft_position_mismatch(state_dir):
    store = StateStore(state_dir)
    plan = _plan()
    store.save_state(
        "current_plan", {"plan": plan.model_dump(mode="json"), "symbol": "ETH/USDT:USDT"}
    )
    store.save_state("grid_state", {"open_lots": {"3": {"entry": 2990.0, "qty": 0.013}}})
    rec = RestartReconciler(store)
    r = rec.recover(actual_open_pairs=set(), actual_open_order_ids=set(), now_ms=0)
    assert r.status == "SAFE_HOLD"
    assert "POSITION_MISMATCH" in r.reasons


# ------------------------------------------------------------------ scheduler
def _req(trigger=TriggerKind.PERIODIC, digest="d1", ts_ms=1_000_000_000_000):
    return AnalysisRequest("ETH/USDT:USDT", trigger, ts_ms, digest)


def test_scheduler_periodic_cooldown():
    s = AnalysisScheduler(SchedulerSettings(periodic_minutes=30))
    ok, _ = s.should_analyze(_req())
    assert ok
    s.record_dispatch(_req())
    ok2, reason = s.should_analyze(_req(ts_ms=1_000_000_000_000 + 10 * 60_000))
    assert not ok2 and reason == "PERIODIC_COOLDOWN"
    ok3, _ = s.should_analyze(_req(ts_ms=1_000_000_000_000 + 31 * 60_000))
    assert ok3


def test_scheduler_event_cooldown_and_dedup():
    s = AnalysisScheduler(SchedulerSettings(event_cooldown_minutes=10))
    req = _req(TriggerKind.PRICE_BREAKOUT)
    ok, _ = s.should_analyze(req)
    assert ok
    s.record_dispatch(req)
    ok2, reason = s.should_analyze(_req(TriggerKind.PRICE_BREAKOUT, ts_ms=req.ts_ms + 5 * 60_000))
    assert not ok2 and reason == "EVENT_COOLDOWN"
    # different trigger after cooldown OK
    ok3, _ = s.should_analyze(_req(TriggerKind.VOLATILITY_SPIKE, ts_ms=req.ts_ms + 11 * 60_000))
    assert ok3
    # same trigger+digest after cooldown -> deduped
    s.record_dispatch(_req(TriggerKind.VOLATILITY_SPIKE, ts_ms=req.ts_ms + 11 * 60_000))
    ok4, reason = s.should_analyze(
        _req(TriggerKind.VOLATILITY_SPIKE, ts_ms=req.ts_ms + 21 * 60_000)
    )
    assert not ok4 and reason == "DUPLICATE_REQUEST"


def test_scheduler_rate_limit():
    s = AnalysisScheduler(SchedulerSettings(max_llm_calls_per_hour=3, event_cooldown_minutes=0))
    base = 1_000_000_000_000
    for i in range(3):
        req = _req(TriggerKind.PRICE_BREAKOUT, digest=f"d{i}", ts_ms=base + i * 60_000)
        ok, _ = s.should_analyze(req)
        assert ok
        s.record_dispatch(req)
        s.record_result(req, ok=True, latency_ms=100)
    req4 = _req(TriggerKind.REVERSAL_SIGNAL, digest="d9", ts_ms=base + 4 * 60_000)
    ok, reason = s.should_analyze(req4)
    assert not ok and reason == "RATE_LIMITED"


def test_scheduler_circuit_breaker():
    s = AnalysisScheduler(
        SchedulerSettings(circuit_breaker_failures=3, circuit_breaker_cooldown_minutes=60)
    )
    base = 1_000_000_000_000
    for i in range(3):
        req = _req(TriggerKind.PRICE_BREAKOUT, digest=f"f{i}", ts_ms=base + i * 60_000)
        s.record_dispatch(req)
        s.record_result(req, ok=False, latency_ms=10, error="boom")
    assert s.breaker_open(base + 4 * 60_000)
    ok, reason = s.should_analyze(_req(ts_ms=base + 4 * 60_000))
    assert not ok and reason == "CIRCUIT_BREAKER_OPEN"
    # after cooldown (opens at last failure base+2m, closes base+62m) -> closed again
    assert not s.breaker_open(base + 63 * 60_000)
    ok2, _ = s.should_analyze(_req(ts_ms=base + 63 * 60_000))
    assert ok2


def test_scheduler_success_resets_failure_streak():
    s = AnalysisScheduler(SchedulerSettings(circuit_breaker_failures=3))
    base = 1_000_000_000_000
    for i in range(2):
        req = _req(TriggerKind.PRICE_BREAKOUT, digest=f"f{i}", ts_ms=base + i * 60_000)
        s.record_dispatch(req)
        s.record_result(req, ok=False, latency_ms=10, error="boom")
    req_ok = _req(TriggerKind.VOLATILITY_SPIKE, digest="ok", ts_ms=base + 3 * 60_000)
    s.record_dispatch(req_ok)
    s.record_result(req_ok, ok=True, latency_ms=10)
    assert s.state.consecutive_failures == 0
    assert not s.breaker_open()


def test_scheduler_cost_summary():
    s = AnalysisScheduler(SchedulerSettings())
    req = _req()
    s.record_dispatch(req)
    s.record_result(req, ok=True, latency_ms=1500)
    summary = s.cost_summary()
    assert summary["total_calls"] == 1
    assert summary["avg_latency_ms"] == 1500.0
