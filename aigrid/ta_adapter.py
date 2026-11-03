"""TradingAgents adapter — 3-Agent pipeline trên một model (Z.ai GLM 5.3 Flash
qua OpenRouter), theo chỉ đạo Sếp 2026-11-03:

    Agent 1 — TECHNICAL/FUTURES ANALYST: phân tích kỹ thuật + dữ liệu Futures.
    Agent 2 — RISK CRITIC: phản biện rủi ro + điều kiện thị trường.
    Agent 3 — STRATEGY SYNTHESIZER: tổng hợp, đề xuất chiến lược + vùng lưới
              + vốn tối đa (chỉ mang tính tham khảo — CapitalManager quyết).

Sau đó Risk Engine ĐỘC LẬP kiểm tra trước khi Freqtrade vào lệnh.

Thiết kế:
- Dùng upstream TradingAgents llm_clients (create_llm_client) — không fork.
- Mỗi phân tích = đúng 3 lần gọi LLM (rẻ, nhanh, đủ sâu).
- Mọi output đi qua Pydantic schema; lỗi → AI_UNAVAILABLE (fail-closed).
- LLM KHÔNG BAO GIỜ đặt lệnh; chỉ trả nhận định + đề xuất có cấu trúc.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from .settings import AIGridSettings

logger = logging.getLogger(__name__)

REGIMES = ("RANGE", "UPTREND", "DOWNTREND", "HIGH_VOLATILITY")
GRID_MODES = ("RANGE_GRID", "LONG_GRID", "SHORT_GRID", "SAFE_HOLD")

_API_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "glm": "ZHIPU_API_KEY",
    "glm-cn": "ZHIPU_CN_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


# --------------------------------------------------------------------- schemas
class TechnicalReport(BaseModel):
    """Agent 1 — phân tích kỹ thuật trên dữ liệu Binance futures."""

    trend: str = Field(pattern=r"^(UP|DOWN|SIDEWAYS)$")
    momentum: str = Field(pattern=r"^(STRONG|MODERATE|WEAK)$")
    volatility: str = Field(pattern=r"^(LOW|NORMAL|HIGH|EXTREME)$")
    support: float = Field(gt=0, description="mức hỗ trợ gần nhất (USDT)")
    resistance: float = Field(gt=0, description="mức kháng cự gần nhất (USDT)")
    liquidity_note: str = Field(default="", max_length=300)
    summary: str = Field(max_length=1500)


class RiskCritique(BaseModel):
    """Agent 2 — phản biện rủi ro và điều kiện thị trường."""

    main_risks: list[str] = Field(default_factory=list, max_length=8)
    trend_break_risk: str = Field(pattern=r"^(LOW|MEDIUM|HIGH)$")
    volatility_assessment: str = Field(pattern=r"^(CALM|NORMAL|ELEVATED|DANGEROUS)$")
    funding_concern: bool = False
    conditions_ok_for_grid: bool
    objections: list[str] = Field(default_factory=list, max_length=8)
    summary: str = Field(max_length=1000)


class StrategyProposal(BaseModel):
    """Agent 3 — tổng hợp + đề xuất (tham khảo; validator + risk quyết)."""

    regime: str = Field(pattern=r"^(RANGE|UPTREND|DOWNTREND|HIGH_VOLATILITY)$")
    confidence: float = Field(ge=0.0, le=1.0)
    grid_mode: str = Field(pattern=r"^(RANGE_GRID|LONG_GRID|SHORT_GRID|SAFE_HOLD)$")
    grid_lower: float = Field(gt=0)
    grid_upper: float = Field(gt=0)
    levels: int = Field(ge=3, le=8)
    max_capital_suggestion: float = Field(ge=0, description="USDT margin đề xuất (tham khảo)")
    reasoning_summary: str = Field(max_length=1500)


@dataclass(frozen=True)
class MarketAssessment:
    pair: str
    as_of_ms: int
    regime: str  # one of REGIMES or 'AI_UNAVAILABLE'/'AI_TIMEOUT'
    confidence: float  # 0..1
    reasoning_summary: str
    proposal: StrategyProposal | None = None
    technical: TechnicalReport | None = None
    critique: RiskCritique | None = None
    llm_latency_ms: int = 0
    llm_calls: int = 0
    ai_available: bool = False
    timed_out: bool = False  # deadline exceeded -> structured TIMEOUT

    @property
    def ok(self) -> bool:
        return self.ai_available and self.regime in REGIMES


@dataclass
class AICostLedger:
    """Cumulative LLM cost counters (ai_calls counts every attempt, including
    retries). Updated from the AI worker thread; reads are GIL-atomic."""

    ai_calls: int = 0
    ai_timeouts: int = 0
    ai_failures: int = 0
    ai_tokens_in: int = 0
    ai_tokens_out: int = 0
    ai_seconds_total: float = 0.0

    def as_dict(self) -> dict:
        return {
            "ai_calls": self.ai_calls,
            "ai_timeouts": self.ai_timeouts,
            "ai_failures": self.ai_failures,
            "ai_tokens_in": self.ai_tokens_in,
            "ai_tokens_out": self.ai_tokens_out,
            "ai_seconds_total": round(self.ai_seconds_total, 3),
        }


class _PipelineTimeout(Exception):
    """Internal marker: total pipeline deadline or per-call timeout hit."""


# Errors worth ONE retry: timeouts, connection faults, rate limits, 5xx.
# Schema/logic errors (ValidationError, ValueError, ...) are never retried.
_TRANSIENT_STATUS = {408, 425, 429} | set(range(500, 600))
_TRANSIENT_MARKERS = (
    "timeout",
    "timedout",
    "rate_limit",
    "ratelimit",
    "overloaded",
    "connection",
    "unavailable",
    "internalserver",
    "temporar",
)


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int) and status in _TRANSIENT_STATUS:
        return True
    name = type(exc).__name__.lower()
    return any(m in name for m in _TRANSIENT_MARKERS)


def _run_isolated(fn: Callable[[], Any], timeout_s: float):
    """Run fn() in a daemon thread; returns (value, exc, timed_out).

    A timed-out call keeps running in the background (bounded: at most the
    configured attempts per analysis and one analysis in flight), but the
    pipeline never waits on it again.
    """
    box: dict = {}

    def _wrap() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:
            box["exc"] = exc

    t = threading.Thread(target=_wrap, daemon=True, name="aigrid-llm-call")
    t.start()
    t.join(max(0.0, timeout_s))
    if t.is_alive():
        return None, None, True
    return box.get("value"), box.get("exc"), False


# --------------------------------------------------------------------- prompts
AGENT1_SYSTEM = """You are Agent 1 (Technical/Futures Analyst) in a crypto futures
trading system. You analyze Binance USDT-M PERPETUAL FUTURES data (24/7 market,
NOT a stock market — no sessions, no gaps except funding). You receive a
deterministic market snapshot (OHLCV summary + indicators + funding) computed
from exchange data. Use ONLY the numbers provided; do not invent prices.
Be concise and factual."""

AGENT2_SYSTEM = """You are Agent 2 (Risk Critic). You review the technical report
and market snapshot to challenge the trading idea: identify the main risks
(trend break, volatility regime, funding cost, liquidity, range breakouts),
assess market conditions, and state clearly whether conditions are suitable
for a grid trading strategy right now. Be skeptical and specific."""

AGENT3_SYSTEM = """You are Agent 3 (Strategy Synthesizer). You combine the
technical report and the risk critique into ONE final proposal:
- regime: RANGE / UPTREND / DOWNTREND / HIGH_VOLATILITY
- grid_mode: RANGE_GRID (sideways), LONG_GRID (uptrend, buy dips),
  SHORT_GRID (downtrend, sell rips), or SAFE_HOLD (dangerous/uncertain)
- grid_lower/grid_upper: the price zone for the grid (must bracket the last
  close sensibly; width proportional to volatility)
- levels: number of grid levels (3-8)
- max_capital_suggestion: ADVISORY ONLY margin suggestion in USDT (the Capital
  Manager decides the real budget)
If the risk critic says conditions are NOT ok for a grid, choose SAFE_HOLD.
Never propose orders — only this structured proposal."""


# --------------------------------------------------------------------- adapter
class TradingAgentsAdapter:
    """3-agent pipeline over one shared model. Lazy init; fail-closed.

    Cost accounting lives in self.cost (AICostLedger): ai_calls counts every
    invoke attempt (retries included), ai_timeouts every timed-out call,
    ai_failures every non-timeout failed call, tokens best-effort from the
    provider's usage metadata.
    """

    def __init__(self, settings: AIGridSettings):
        self.settings = settings
        self._llm: Any | None = None
        self._init_error: str | None = None
        self.cost = AICostLedger()
        self._cost_lock = threading.Lock()

    # ------------------------------------------------------------------ setup
    def _ensure_llm(self) -> Any:
        if self._llm is not None or self._init_error is not None:
            return self._llm
        provider = self.settings.llm.provider.lower()
        key_env = _API_KEY_ENV.get(provider)
        if key_env and not os.environ.get(key_env):
            # best-effort: load the workspace .env (gitignored) for the key
            try:
                from pathlib import Path

                from dotenv import load_dotenv

                for root in (Path.cwd(), Path(__file__).resolve().parents[1]):
                    env_file = root / ".env"
                    if env_file.exists():
                        load_dotenv(env_file, override=False)
                        break
            except Exception:
                pass
        if key_env and not os.environ.get(key_env):
            self._init_error = f"MISSING_API_KEY:{key_env}"
            logger.warning("AI disabled: %s not set", key_env)
            return None
        try:
            from tradingagents.llm_clients.factory import create_llm_client

            client = create_llm_client(
                provider, self.settings.llm.model, base_url=self.settings.llm.backend_url or None
            )
            self._llm = client.get_llm()
        except Exception as exc:
            self._init_error = f"INIT_FAILED:{exc}"
            logger.exception("LLM client init failed")
            return None
        return self._llm

    @property
    def unavailable_reason(self) -> str | None:
        return self._init_error

    # ------------------------------------------------------------------ cost helpers
    def _bump(self, name: str, n: int = 1) -> None:
        with self._cost_lock:
            setattr(self.cost, name, getattr(self.cost, name) + n)

    def _book_seconds(self, started: float) -> None:
        with self._cost_lock:
            self.cost.ai_seconds_total += time.monotonic() - started

    def _record_tokens(self, value: Any) -> None:
        """Best-effort token accounting — only when the provider returns usage."""
        usage: Any = getattr(value, "usage_metadata", None)
        meta = getattr(value, "response_metadata", None)
        if usage is None and isinstance(meta, dict):
            usage = meta.get("token_usage") or meta.get("usage")
        if not isinstance(usage, dict):
            return
        tin = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        tout = usage.get("completion_tokens") or usage.get("output_tokens") or 0
        with self._cost_lock:
            self.cost.ai_tokens_in += int(tin or 0)
            self.cost.ai_tokens_out += int(tout or 0)

    # ------------------------------------------------------------------ analyze
    def assess(self, pair: str, market_context: dict, as_of_ms: int) -> MarketAssessment:
        """Run the 3-agent pipeline under a hard deadline (never raises).

        Each call gets min(llm_timeout_seconds, remaining deadline); a
        transient failure may retry up to llm_max_retries with short backoff —
        always inside llm_total_deadline_seconds. Structured outcomes only:
        ok / AI_TIMEOUT / AI_UNAVAILABLE / AI_RUN_FAILED.
        """
        if not self.settings.enabled:
            return MarketAssessment(
                pair, as_of_ms, "AI_UNAVAILABLE", 0.0, "AI disabled by settings (enabled=false)"
            )
        llm = self._ensure_llm()
        if llm is None:
            return MarketAssessment(
                pair, as_of_ms, "AI_UNAVAILABLE", 0.0, self._init_error or "AI unavailable"
            )
        started = time.monotonic()
        deadline = started + float(self.settings.scheduler.llm_total_deadline_seconds)
        calls_before = self.cost.ai_calls
        timeouts_before = self.cost.ai_timeouts
        try:
            tech = self._invoke_agent(
                "agent1_technical", lambda: self._run_agent1(llm, market_context), deadline
            )
            critique = self._invoke_agent(
                "agent2_risk", lambda: self._run_agent2(llm, market_context, tech), deadline
            )
            proposal = self._invoke_agent(
                "agent3_synth",
                lambda: self._run_agent3(llm, market_context, tech, critique),
                deadline,
            )
        except _PipelineTimeout as exc:
            latency = int((time.monotonic() - started) * 1000)
            self._book_seconds(started)
            if self.cost.ai_timeouts == timeouts_before:
                self._bump("ai_timeouts")  # deadline hit outside any call
            logger.warning("AI pipeline TIMEOUT for %s: %s", pair, exc)
            return MarketAssessment(
                pair,
                as_of_ms,
                "AI_TIMEOUT",
                0.0,
                f"AI_TIMEOUT:{exc}"[:300],
                None,
                None,
                None,
                latency,
                self.cost.ai_calls - calls_before,
                False,
                timed_out=True,
            )
        except Exception as exc:
            latency = int((time.monotonic() - started) * 1000)
            self._book_seconds(started)
            logger.warning("AI pipeline failed for %s: %s", pair, exc)
            return MarketAssessment(
                pair,
                as_of_ms,
                "AI_UNAVAILABLE",
                0.0,
                f"AI_RUN_FAILED:{type(exc).__name__}:{exc}"[:300],
                None,
                None,
                None,
                latency,
                self.cost.ai_calls - calls_before,
                False,
            )
        latency = int((time.monotonic() - started) * 1000)
        self._book_seconds(started)
        return MarketAssessment(
            pair,
            as_of_ms,
            proposal.regime,
            proposal.confidence,
            proposal.reasoning_summary,
            proposal,
            tech,
            critique,
            latency,
            self.cost.ai_calls - calls_before,
            True,
        )

    # ------------------------------------------------------------------ call wrapper
    def _invoke_agent(self, name: str, invoke: Callable[[], Any], deadline: float) -> Any:
        """One agent call: per-call timeout + bounded transient retries.

        invoke() is a zero-arg callable producing the structured output.
        Returns the parsed object; raises _PipelineTimeout on deadline
        exhaustion or the original exception on a permanent failure.
        """
        per_call = float(self.settings.scheduler.llm_timeout_seconds)
        max_attempts = 1 + int(self.settings.scheduler.llm_max_retries)
        last_exc: BaseException | None = None
        for attempt in range(1, max_attempts + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _PipelineTimeout(f"{name}: total deadline exceeded")
            cap = min(per_call, remaining)
            self._bump("ai_calls")
            value, exc, timed_out = _run_isolated(invoke, cap)
            if timed_out:
                self._bump("ai_timeouts")
                last_exc = _PipelineTimeout(f"{name}: call timed out (>{cap:.2f}s)")
            elif exc is not None:
                self._bump("ai_failures")
                if not _is_transient(exc):
                    raise exc  # schema/logic: never retry
                last_exc = exc
            else:
                self._record_tokens(value)
                return value
            # transient failure — bounded backoff, still inside the deadline
            if attempt < max_attempts:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.5 * attempt, remaining))
        if isinstance(last_exc, (_PipelineTimeout, TimeoutError)):
            raise _PipelineTimeout(str(last_exc))
        raise last_exc if last_exc is not None else _PipelineTimeout(name)

    # ------------------------------------------------------------------ agents
    def _run_agent1(self, llm, ctx: dict) -> TechnicalReport:
        from .agents_prompts import render_agent1_prompt

        structured = llm.with_structured_output(TechnicalReport)
        out = structured.invoke(
            [
                {"role": "system", "content": AGENT1_SYSTEM},
                {"role": "user", "content": render_agent1_prompt(ctx)},
            ]
        )
        return out

    def _run_agent2(self, llm, ctx: dict, tech: TechnicalReport) -> RiskCritique:
        from .agents_prompts import render_agent2_prompt

        structured = llm.with_structured_output(RiskCritique)
        out = structured.invoke(
            [
                {"role": "system", "content": AGENT2_SYSTEM},
                {"role": "user", "content": render_agent2_prompt(ctx, tech)},
            ]
        )
        return out

    def _run_agent3(
        self, llm, ctx: dict, tech: TechnicalReport, critique: RiskCritique
    ) -> StrategyProposal:
        from .agents_prompts import render_agent3_prompt

        structured = llm.with_structured_output(StrategyProposal)
        out = structured.invoke(
            [
                {"role": "system", "content": AGENT3_SYSTEM},
                {"role": "user", "content": render_agent3_prompt(ctx, tech, critique)},
            ]
        )
        # hard guard: risk critic veto -> SAFE_HOLD
        if not critique.conditions_ok_for_grid and out.grid_mode != "SAFE_HOLD":
            out = out.model_copy(update={"grid_mode": "SAFE_HOLD"})
        return out
