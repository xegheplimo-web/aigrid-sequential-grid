"""R8 — T-207 (adapter level): hard LLM deadline + bounded transient retry.

Production defaults are 40s per call / 120s total / 1 retry; tests scale the
same knobs down (per-call ~0.3s, total ~0.8s) so everything runs < 5s with
fake LLMs — NO network calls anywhere.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pytest
from aigrid.settings import AIGridSettings, SchedulerSettings
from aigrid.ta_adapter import (
    RiskCritique,
    StrategyProposal,
    TechnicalReport,
    TradingAgentsAdapter,
)

ROOT = Path(__file__).resolve().parents[1]
PAIR = "ETH/USDT:USDT"


# --------------------------------------------------------------------- fake LLM
class _FakeStructured:
    def __init__(self, llm, schema):
        self.llm, self.schema = llm, schema

    def invoke(self, messages):
        return self.llm._invoke(self.schema, messages)


class FakeLLM:
    """Same LangChain surface the adapter uses; handler decides per schema."""

    def __init__(self, handler):
        self._handler = handler
        self.calls: list[str] = []

    def with_structured_output(self, schema):
        return _FakeStructured(self, schema)

    def _invoke(self, schema, messages):
        self.calls.append(schema.__name__)
        return self._handler(schema, messages)


def _tech():
    return TechnicalReport(
        trend="SIDEWAYS",
        momentum="WEAK",
        volatility="LOW",
        support=2490.0,
        resistance=2515.0,
        summary="flat",
    )


def _crit():
    return RiskCritique(
        main_risks=[],
        trend_break_risk="LOW",
        volatility_assessment="CALM",
        funding_concern=False,
        conditions_ok_for_grid=True,
        objections=[],
        summary="ok",
    )


def _prop():
    return StrategyProposal(
        regime="RANGE",
        confidence=0.6,
        grid_mode="RANGE_GRID",
        grid_lower=2490.0,
        grid_upper=2515.0,
        levels=5,
        max_capital_suggestion=200.0,
        reasoning_summary="range grid",
    )


_RESPONSES = {TechnicalReport: _tech, RiskCritique: _crit, StrategyProposal: _prop}


class Transient503(Exception):
    """Simulates a provider 5xx — transient, retryable."""

    status_code = 503


def _adapter(fake: FakeLLM, **sched_kw) -> TradingAgentsAdapter:
    base = dict(llm_timeout_seconds=0.3, llm_total_deadline_seconds=0.8, llm_max_retries=1)
    base.update(sched_kw)
    try:
        sched = SchedulerSettings(**base)
    except TypeError:  # old schema: deadline knob missing
        base.pop("llm_total_deadline_seconds", None)
        sched = SchedulerSettings(**base)
    settings = replace(AIGridSettings(), enabled=True, mode="shadow", scheduler=sched)
    ta = TradingAgentsAdapter(settings)
    ta._llm = fake  # inject — no network
    return ta


# ------------------------------------------------------------------ spec #2: total deadline
def test_total_deadline_caps_slow_pipeline():
    """3 agents each sleeping past the per-call cap -> structured TIMEOUT and
    the pipeline aborts at llm_total_deadline_seconds, not after all sleeps."""
    fake = FakeLLM(lambda schema, _m: (time.sleep(2.0), _RESPONSES[schema]())[1])
    ta = _adapter(fake)  # per-call .3 / total .8
    started = time.monotonic()
    a = ta.assess(PAIR, {"last_close": 2500.0}, 0)
    elapsed = time.monotonic() - started
    assert a.timed_out is True
    assert not a.ok and a.regime == "AI_TIMEOUT"
    assert elapsed <= 0.8 + 0.45  # deadline + tolerance
    assert ta.cost.ai_timeouts >= 1
    assert ta.cost.ai_calls >= 1
    assert ta.cost.ai_seconds_total >= elapsed - 0.05


# ------------------------------------------------------------------ spec #3: per-call timeout
def test_per_call_timeout_isolates_one_hanging_agent():
    """One hanging agent is cut at llm_timeout_seconds; the pipeline stops and
    reports TIMEOUT instead of waiting on it."""

    def handler(schema, _m):
        if schema is TechnicalReport:
            time.sleep(3.0)
        return _RESPONSES[schema]()

    fake = FakeLLM(handler)
    ta = _adapter(fake, llm_max_retries=0, llm_total_deadline_seconds=5.0)
    started = time.monotonic()
    a = ta.assess(PAIR, {}, 0)
    elapsed = time.monotonic() - started
    assert a.timed_out is True
    assert elapsed <= 0.3 + 0.45  # per-call cap + tolerance
    assert fake.calls == ["TechnicalReport"]  # never reached agent 2/3


# ------------------------------------------------------------------ spec #4: transient retry
def test_transient_5xx_retried_once_then_ok():
    """A 503 on the first attempt is retried once; success on attempt 2."""
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise Transient503("service unavailable")
        return _tech()

    ta = _adapter(FakeLLM(lambda s, m: _RESPONSES[s]()))
    out = ta._invoke_agent("agent1", flaky, time.monotonic() + 5.0)
    assert out.trend == "SIDEWAYS"
    assert attempts["n"] == 2
    assert ta.cost.ai_calls == 2


def test_timeout_error_is_transient_and_retried():
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TimeoutError("read timed out")
        return _crit()

    ta = _adapter(FakeLLM(lambda s, m: _RESPONSES[s]()))
    out = ta._invoke_agent("agent2", flaky, time.monotonic() + 5.0)
    assert out.conditions_ok_for_grid is True
    assert ta.cost.ai_calls == 2


# ------------------------------------------------------------------ spec #5: no retry on schema/logic errors
def test_schema_error_not_retried():
    """Validation/logic errors are NOT transient: fail fast, no retry."""
    attempts = {"n": 0}

    def bad():
        attempts["n"] += 1
        raise ValueError("schema mismatch: field 'trend' missing")

    ta = _adapter(FakeLLM(lambda s, m: _RESPONSES[s]()))
    with pytest.raises(ValueError):
        ta._invoke_agent("agent1", bad, time.monotonic() + 5.0)
    assert attempts["n"] == 1
    assert ta.cost.ai_calls == 1
    assert ta.cost.ai_failures == 1


# ------------------------------------------------------------------ spec #5b: cost ledger on success
def test_cost_ledger_records_calls_and_seconds():
    def handler(s, m):
        time.sleep(0.02)  # coarse monotonic ticks
        return _RESPONSES[s]()

    ta = _adapter(FakeLLM(handler))
    a = ta.assess(PAIR, {}, 0)
    assert a.ok and a.llm_calls == 3
    c = ta.cost.as_dict()
    assert c["ai_calls"] == 3
    assert c["ai_timeouts"] == 0 and c["ai_failures"] == 0
    assert c["ai_seconds_total"] > 0
    assert c["ai_tokens_in"] >= 0 and c["ai_tokens_out"] >= 0
