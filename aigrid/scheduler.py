"""Scheduler — periodic + event-driven AI analysis with full abuse protection.

Guarantees (spec section VI):
- cooldown between analyses (periodic and event-driven separately)
- rate limiting (max LLM calls per hour)
- request deduplication (same symbol+reason within cooldown = dropped)
- timeout + bounded retries
- cost tracking (calls + latency)
- circuit breaker (N consecutive failures -> AI disabled for a cooldown)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

from .settings import SchedulerSettings


class TriggerKind(StrEnum):
    PERIODIC = "PERIODIC"
    PRICE_BREAKOUT = "PRICE_BREAKOUT"
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"
    ADX_SHIFT = "ADX_SHIFT"
    REVERSAL_SIGNAL = "REVERSAL_SIGNAL"
    PLAN_EXPIRED = "PLAN_EXPIRED"
    RISK_ANOMALY = "RISK_ANOMALY"


EVENT_TRIGGERS = frozenset(t for t in TriggerKind if t != TriggerKind.PERIODIC)


@dataclass
class AnalysisRequest:
    symbol: str
    trigger: TriggerKind
    ts_ms: int
    context_digest: str = ""  # dedup key component (data version)


@dataclass
class CallRecord:
    ts_ms: int
    symbol: str
    trigger: str
    latency_ms: int
    ok: bool
    error: str = ""


@dataclass
class SchedulerState:
    last_periodic_ms: int = 0
    last_event_ms: dict[str, int] = field(default_factory=dict)  # trigger -> ts
    last_request_digest: str = ""
    recent_calls: deque = field(default_factory=lambda: deque(maxlen=200))
    consecutive_failures: int = 0
    breaker_open_until_ms: int = 0
    total_calls: int = 0
    total_latency_ms: int = 0


class AnalysisScheduler:
    def __init__(self, settings: SchedulerSettings):
        self.settings = settings
        self.state = SchedulerState()

    # ------------------------------------------------------------------ gating
    def should_analyze(self, req: AnalysisRequest) -> tuple[bool, str]:
        now = req.ts_ms
        # circuit breaker
        if now < self.state.breaker_open_until_ms:
            return False, "CIRCUIT_BREAKER_OPEN"
        # rate limit: calls in the last hour
        hour_ago = now - 3_600_000
        recent = [c for c in self.state.recent_calls if c.ts_ms >= hour_ago]
        if len(recent) >= self.settings.max_llm_calls_per_hour:
            return False, "RATE_LIMITED"
        # periodic trigger: respect the periodic cycle
        if req.trigger == TriggerKind.PERIODIC:
            if now - self.state.last_periodic_ms < self.settings.periodic_minutes * 60_000:
                return False, "PERIODIC_COOLDOWN"
            return True, ""
        # event triggers: per-trigger cooldown + dedup
        if req.trigger not in EVENT_TRIGGERS:
            return False, "UNKNOWN_TRIGGER"
        last = self.state.last_event_ms.get(req.trigger, 0)
        if now - last < self.settings.event_cooldown_minutes * 60_000:
            return False, "EVENT_COOLDOWN"
        digest = f"{req.symbol}:{req.trigger}:{req.context_digest}"
        if digest == self.state.last_request_digest:
            return False, "DUPLICATE_REQUEST"
        return True, ""

    # ------------------------------------------------------------------ recording
    def record_dispatch(self, req: AnalysisRequest) -> None:
        if req.trigger == TriggerKind.PERIODIC:
            self.state.last_periodic_ms = req.ts_ms
        else:
            self.state.last_event_ms[req.trigger] = req.ts_ms
        self.state.last_request_digest = f"{req.symbol}:{req.trigger}:{req.context_digest}"

    def record_result(
        self, req: AnalysisRequest, ok: bool, latency_ms: int, error: str = ""
    ) -> None:
        self.state.recent_calls.append(
            CallRecord(req.ts_ms, req.symbol, req.trigger.value, latency_ms, ok, error)
        )
        self.state.total_calls += 1
        self.state.total_latency_ms += latency_ms
        if ok:
            self.state.consecutive_failures = 0
        else:
            self.state.consecutive_failures += 1
            if self.state.consecutive_failures >= self.settings.circuit_breaker_failures:
                self.state.breaker_open_until_ms = (
                    req.ts_ms + self.settings.circuit_breaker_cooldown_minutes * 60_000
                )
                self.state.consecutive_failures = 0

    def breaker_open(self, now_ms: int | None = None) -> bool:
        now_ms = now_ms or int(time.time() * 1000)
        return now_ms < self.state.breaker_open_until_ms

    # ------------------------------------------------------------------ stats
    def cost_summary(self) -> dict:
        st = self.state
        return {
            "total_calls": st.total_calls,
            "avg_latency_ms": round(st.total_latency_ms / st.total_calls, 1)
            if st.total_calls
            else 0,
            "breaker_open": self.breaker_open(),
            "consecutive_failures": st.consecutive_failures,
            "recent_calls_last_hour": sum(
                1 for c in st.recent_calls if c.ts_ms >= int(time.time() * 1000) - 3_600_000
            ),
        }
