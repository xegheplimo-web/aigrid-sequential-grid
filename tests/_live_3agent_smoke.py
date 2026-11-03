# ruff: noqa: E402 -- standalone scripts bootstrap the repository before imports.
"""Live smoke test: 3-agent pipeline over OpenRouter z-ai/glm-5.3-flash.
Run manually (needs OPENROUTER_API_KEY in .env). NOT part of pytest suite.
Counts LLM calls + token usage + latency; prints assessment summary only.
"""

import os
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

assert os.environ.get("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY missing"

from aigrid.data.binance_adapter import BinanceFuturesDataAdapter
from aigrid.settings import AIGridSettings
from aigrid.ta_adapter import TradingAgentsAdapter

# ---- token usage recorder: patch the underlying chat model _generate
usage_log: list[dict] = []
_orig_generate = None


def install_usage_recorder():
    global _orig_generate
    from tradingagents.llm_clients.openai_client import NormalizedChatOpenAI

    _orig_generate = NormalizedChatOpenAI._generate

    def wrapped(self, messages, stop=None, run_manager=None, **kwargs):
        result = _orig_generate(self, messages, stop=stop, run_manager=run_manager, **kwargs)
        usage = (result.llm_output or {}).get("token_usage") or {}
        usage_log.append({"model": self.model_name, **usage})
        return result

    NormalizedChatOpenAI._generate = wrapped


def main():
    install_usage_recorder()
    settings = replace(AIGridSettings(), enabled=True, mode="shadow")
    adapter = TradingAgentsAdapter(settings)
    data = BinanceFuturesDataAdapter(ROOT / "user_data/data/binance/futures")
    snap = data.snapshot("ETH/USDT:USDT", "15m")
    ctx = snap.to_ai_context()
    print(
        f"[context] pair={ctx['pair']} close={ctx['last_close']} "
        f"candles={ctx['candles_used']} funding={ctx['funding_rate']}"
    )
    print(f"[indicators] {ctx['indicators']}")

    t0 = time.monotonic()
    a = adapter.assess("ETH/USDT:USDT", ctx, snap.as_of_ms)
    wall = time.monotonic() - t0

    print("\n================ RESULT ================")
    print(f"ai_available={a.ai_available} regime={a.regime} confidence={a.confidence}")
    print(f"llm_calls={a.llm_calls} llm_latency_ms={a.llm_latency_ms} wall_s={wall:.1f}")
    if a.technical:
        print(
            f"\n[Agent1] trend={a.technical.trend} mom={a.technical.momentum} "
            f"vol={a.technical.volatility} S={a.technical.support} "
            f"R={a.technical.resistance}"
        )
        print(f"  summary: {a.technical.summary[:300]}")
    if a.critique:
        print(
            f"\n[Agent2] ok_for_grid={a.critique.conditions_ok_for_grid} "
            f"break_risk={a.critique.trend_break_risk} "
            f"vol={a.critique.volatility_assessment} "
            f"funding_concern={a.critique.funding_concern}"
        )
        print(f"  risks: {a.critique.main_risks}")
        print(f"  summary: {a.critique.summary[:300]}")
    if a.proposal:
        print(
            f"\n[Agent3] regime={a.proposal.regime} mode={a.proposal.grid_mode} "
            f"zone=[{a.proposal.grid_lower},{a.proposal.grid_upper}] "
            f"levels={a.proposal.levels} cap_sugg={a.proposal.max_capital_suggestion}"
        )
        print(f"  reasoning: {a.proposal.reasoning_summary[:400]}")
    if not a.ok:
        print(f"\n[FAIL] {a.reasoning_summary}")

    print("\n================ TOKEN USAGE ================")
    total_in = sum(u.get("prompt_tokens", 0) for u in usage_log)
    total_out = sum(u.get("completion_tokens", 0) for u in usage_log)
    for i, u in enumerate(usage_log, 1):
        print(
            f"  call {i}: model={u['model']} in={u.get('prompt_tokens')} "
            f"out={u.get('completion_tokens')}"
        )
    print(f"  TOTAL: calls={len(usage_log)} in={total_in} out={total_out}")
    # rough OpenRouter pricing placeholder — glm-5.3-flash is cheap; report raw tokens
    return 0 if a.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
