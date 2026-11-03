"""R8 — T-207 (orchestrator level): SLOW AI runs async off the candle path.

run_analysis() dispatches the 3-agent pipeline to a single-worker executor
and returns immediately; results land via state + events. All fake LLMs are
bounded/interruptible — no network, whole file < 5s.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import aigrid.orchestrator as orch_mod
import pandas as pd
import pytest
from aigrid.orchestrator import AIGridOrchestrator
from aigrid.plans.schema import AIGridPlan, GridMode, draft_plan_from_assessment
from aigrid.scheduler import TriggerKind
from aigrid.selection import ReferenceVerdict
from aigrid.settings import AIGridSettings, SchedulerSettings
from aigrid.ta_adapter import (
    RiskCritique,
    StrategyProposal,
    TechnicalReport,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "user_data/data/binance/futures"
PAIR = "ETH/USDT:USDT"
NOW = datetime.now(UTC)


# --------------------------------------------------------------------- fake LLM
class _FakeStructured:
    def __init__(self, llm, schema):
        self.llm, self.schema = llm, schema

    def invoke(self, messages):
        return self.llm._invoke(self.schema, messages)


class FakeLLM:
    def __init__(self, handler):
        self._handler = handler
        self.calls: list[str] = []

    def with_structured_output(self, schema):
        return _FakeStructured(self, schema)

    def _invoke(self, schema, messages):
        self.calls.append(schema.__name__)
        return self._handler(schema, messages)


def _responses():
    return {
        TechnicalReport: TechnicalReport(
            trend="SIDEWAYS",
            momentum="WEAK",
            volatility="LOW",
            support=98.0,
            resistance=102.0,
            summary="flat",
        ),
        RiskCritique: RiskCritique(
            main_risks=[],
            trend_break_risk="LOW",
            volatility_assessment="CALM",
            funding_concern=False,
            conditions_ok_for_grid=True,
            objections=[],
            summary="ok",
        ),
        StrategyProposal: StrategyProposal(
            regime="RANGE",
            confidence=0.7,
            grid_mode="RANGE_GRID",
            grid_lower=98.0,
            grid_upper=102.0,
            levels=4,
            max_capital_suggestion=100.0,
            reasoning_summary="range grid",
        ),
    }


# --------------------------------------------------------------------- helpers
def _sched(**kw) -> SchedulerSettings:
    base = dict(llm_timeout_seconds=0.3, llm_total_deadline_seconds=0.8, llm_max_retries=1)
    base.update(kw)
    try:
        return SchedulerSettings(**base)
    except TypeError:  # old schema: deadline knob missing
        base.pop("llm_total_deadline_seconds", None)
        return SchedulerSettings(**base)


def _make_orch(tmp_path: Path, **sched_kw) -> AIGridOrchestrator:
    settings = AIGridSettings(
        enabled=True,
        mode="shadow",
        symbols=(PAIR,),
        timeframe="5m",
        data_dir=DATA_DIR,
        state_dir=tmp_path / "state",
        scheduler=_sched(**sched_kw),
    )
    o = AIGridOrchestrator(settings)
    _stub_market(o)
    return o


def _stub_market(o: AIGridOrchestrator, last_close: float = 100.0) -> None:
    """Deterministic market snapshot + reference verdict — no feather reads."""
    snap = SimpleNamespace(
        as_of_ms=int(time.time() * 1000),
        ohlcv=pd.DataFrame({"close": [last_close]}),
        indicators={"atr": 1.0},
        to_ai_context=lambda: {"pair": PAIR, "last_close": last_close},
    )
    o.adapter.snapshot = lambda *a, **k: snap
    o.check_reference = lambda as_of=None: ReferenceVerdict(
        alt_long_ok=True, alt_short_ok=True, reasons=()
    )


def _set_fake(orch: AIGridOrchestrator, handler) -> FakeLLM:
    fake = FakeLLM(handler)
    orch.ta._llm = fake  # inject — no network
    return fake


def _candle(open_ms, o, h, low, c):
    return {"open_ms": open_ms, "open": o, "high": h, "low": low, "close": c, "volume": 1.0}


def _grid_plan() -> AIGridPlan:
    """RANGE grid on 100: levels [102, 100.67, 99.33, 98], stop_loss ~97.02."""
    return draft_plan_from_assessment(
        PAIR,
        "RANGE",
        0.8,
        100.0,
        2.0,
        grid_mode="RANGE_GRID",
        ttl_minutes=45,
        margin_usdt=100.0,
        leverage=2.0,
        levels=4,
    )


def _expired_plan() -> AIGridPlan:
    return AIGridPlan(
        plan_id="plan-r8stale00001",
        plan_version=1,
        symbol=PAIR,
        created_at=NOW - timedelta(hours=2),
        expires_at=NOW - timedelta(hours=1),
        market_regime="RANGE",
        confidence=0.7,
        grid_mode=GridMode.RANGE_GRID,
        grid_lower=94.0,
        grid_upper=100.0,
        grid_levels=4,
        grid_spacing=0.05,
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


def _kinds(orch: AIGridOrchestrator) -> list[str]:
    return [e["kind"] for e in orch.eventlog.recent()]


@pytest.fixture
def orch(tmp_path):
    o = _make_orch(tmp_path)
    yield o
    o.close()


# ------------------------------------------------------------------ spec #1: non-blocking dispatch
def test_run_analysis_returns_immediately(orch):
    """A fake LLM that hangs inside the call must NOT block run_analysis."""
    release = threading.Event()

    def handler(schema, _m):
        release.wait(5.0)  # bounded hang (200s analog)
        return _responses()[schema]

    _set_fake(orch, handler)
    started = time.monotonic()
    res = orch.run_analysis(PAIR, TriggerKind.PERIODIC)
    elapsed = time.monotonic() - started
    release.set()
    assert elapsed < 1.0
    assert res["dispatched"] is True and res["pending"] is True
    assert orch.wait_for_analysis(5.0) is not None


def test_completed_analysis_adopts_plan_and_logs(orch):
    _set_fake(orch, lambda s, m: _responses()[s])
    res = orch.run_analysis(PAIR, TriggerKind.PERIODIC)
    assert res["pending"] is True
    out = orch.wait_for_analysis(5.0)
    assert out["ok"] is True and out["plan_id"]
    assert "AI_ANALYSIS_COMPLETED" in _kinds(orch)
    assert PAIR in orch._engines
    assert orch._ai_down is False


# ------------------------------------------------------------------ spec #3b: single in-flight
def test_second_cycle_skipped_while_in_flight(orch):
    release = threading.Event()

    def handler(schema, _m):
        release.wait(5.0)
        return _responses()[schema]

    _set_fake(orch, handler)
    res1 = orch.run_analysis(PAIR, TriggerKind.PERIODIC)
    res2 = orch.run_analysis(PAIR, TriggerKind.PRICE_BREAKOUT)
    release.set()
    assert res1["pending"] is True
    assert res2["dispatched"] is False
    assert res2["reason"] == "AI_IN_FLIGHT"
    orch.wait_for_analysis(5.0)


# ------------------------------------------------------------------ spec #2b: timeout event + metrics
def test_analysis_timeout_event_and_metrics(orch):
    def handler(schema, _m):
        time.sleep(3.0)
        return _responses()[schema]

    _set_fake(orch, handler)
    orch.run_analysis(PAIR, TriggerKind.PERIODIC)
    out = orch.wait_for_analysis(5.0)
    assert out["timed_out"] is True
    assert "AI_ANALYSIS_TIMEOUT" in _kinds(orch)
    c = orch.ta.cost.as_dict()
    assert c["ai_calls"] >= 1 and c["ai_timeouts"] >= 1
    assert c["ai_seconds_total"] > 0
    assert orch._ai_down is True


def test_analysis_failure_event(orch):
    class BoomError(Exception):
        pass

    def boom(schema, _m):
        raise BoomError("provider exploded")

    _set_fake(orch, boom)
    orch.run_analysis(PAIR, TriggerKind.PERIODIC)
    out = orch.wait_for_analysis(5.0)
    assert out["ok"] is False and not out.get("timed_out")
    assert "AI_ANALYSIS_FAILED" in _kinds(orch)
    assert orch._ai_down is True


# ------------------------------------------------------------------ spec #6: expired TTL rejected
def test_expired_plan_rejected_not_adopted(orch, monkeypatch):
    _set_fake(orch, lambda s, m: _responses()[s])
    stale = _expired_plan()
    monkeypatch.setattr(orch_mod, "draft_plan_from_assessment", lambda *a, **k: stale)
    orch.run_analysis(PAIR, TriggerKind.PERIODIC)
    out = orch.wait_for_analysis(5.0)
    assert out["ok"] is False and out["error"] == "PLAN_STALE_REJECTED"
    assert "PLAN_STALE_REJECTED" in _kinds(orch)
    assert PAIR not in orch._engines


# ------------------------------------------------------------------ spec #7: plan after deadline rejected
def test_late_analysis_result_rejected(tmp_path, monkeypatch):
    """A plan that only materialises after the total deadline is refused."""
    o = _make_orch(tmp_path, llm_total_deadline_seconds=0.15)
    try:
        _set_fake(o, lambda s, m: _responses()[s])
        real_assess = o.ta.assess

        def slow_assess(pair, ctx, as_of_ms):
            time.sleep(0.4)  # arrive after deadline
            return real_assess(pair, ctx, as_of_ms)

        monkeypatch.setattr(o.ta, "assess", slow_assess)
        o.run_analysis(PAIR, TriggerKind.PERIODIC)
        out = o.wait_for_analysis(5.0)
        assert out["ok"] is False and out["error"] == "PLAN_STALE_REJECTED"
        assert "PLAN_STALE_REJECTED" in _kinds(o)
        assert PAIR not in o._engines
    finally:
        o.close()


# ------------------------------------------------------------------ spec #8: candle path not blocked
def test_process_candle_runs_while_ai_in_flight(orch):
    release = threading.Event()

    def handler(schema, _m):
        release.wait(5.0)
        return _responses()[schema]

    _set_fake(orch, handler)
    orch.adopt_plan(_grid_plan())
    res = orch.run_analysis(PAIR, TriggerKind.PERIODIC)
    assert res["pending"] is True  # AI busy in background
    started = time.monotonic()
    snap = orch.process_candle(PAIR, _candle(1_000_000, 103.0, 103.5, 101.0, 101.5))
    elapsed = time.monotonic() - started
    release.set()
    assert elapsed < 1.0
    assert snap.pair == PAIR
    # candle was really processed: level 102 crossed down -> 1 lot filled
    assert len(orch._engines[PAIR].state.lots) == 1
    orch.wait_for_analysis(5.0)


# ------------------------------------------------------------------ spec #9: AI down -> entries off, protection on
def test_ai_unavailable_blocks_entries_but_exits_run(orch):
    class BoomError(Exception):
        pass

    _set_fake(orch, lambda s, m: (_ for _ in ()).throw(BoomError("down")))
    orch.adopt_plan(_grid_plan())
    # candle fills the 102 level normally (AI still up)
    orch.process_candle(PAIR, _candle(1_000_000, 103.0, 103.5, 101.0, 101.5))
    assert len(orch._engines[PAIR].state.lots) == 1

    orch.run_analysis(PAIR, TriggerKind.PERIODIC)
    out = orch.wait_for_analysis(5.0)
    assert out["ok"] is False
    assert "AI_ANALYSIS_FAILED" in _kinds(orch)
    assert orch._ai_down is True

    # price crosses fresh levels 100.67/99.33 — new entries must be vetoed
    orch.process_candle(PAIR, _candle(1_300_000, 101.5, 101.6, 99.0, 99.5))
    assert len(orch._engines[PAIR].state.lots) == 1

    # price crashes through the plan stop -> protection still fires
    orch.process_candle(PAIR, _candle(1_600_000, 99.5, 99.6, 96.5, 97.0))
    assert len(orch._engines[PAIR].state.lots) == 0
    assert "GRID_ACTIONS" in _kinds(orch)


# ------------------------------------------------------------------ spec #3c: bounded shutdown
def test_close_joins_worker_with_timeout(tmp_path):
    """SIGTERM path: close() must not hang on a still-running analysis."""
    release = threading.Event()
    o = _make_orch(tmp_path, llm_timeout_seconds=30.0, llm_total_deadline_seconds=60.0)
    _set_fake(o, lambda s, m: (release.wait(30.0), _responses()[s])[1])
    o.run_analysis(PAIR, TriggerKind.PERIODIC)
    started = time.monotonic()
    o.close(ai_join_timeout=0.3)
    elapsed = time.monotonic() - started
    release.set()
    assert elapsed < 2.0
