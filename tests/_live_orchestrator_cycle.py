# ruff: noqa: E402 -- standalone scripts bootstrap the repository before imports.
"""Live verification: ONE full orchestrator AI analysis cycle on real data.
snapshot -> 3 agents (real API) -> CapitalManager -> draft -> validate -> risk -> adopt.
NOT part of pytest. Prints only non-secret outputs.
"""

import os
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from aigrid.exchange_info import ExchangeInfo
from aigrid.orchestrator import AIGridOrchestrator
from aigrid.scheduler import TriggerKind
from aigrid.settings import AIGridSettings

usage_log: list[dict] = []


def install_usage_recorder():
    from tradingagents.llm_clients.openai_client import NormalizedChatOpenAI

    orig = NormalizedChatOpenAI._generate

    def wrapped(self, messages, stop=None, run_manager=None, **kwargs):
        r = orig(self, messages, stop=stop, run_manager=run_manager, **kwargs)
        usage_log.append((r.llm_output or {}).get("token_usage") or {})
        return r

    NormalizedChatOpenAI._generate = wrapped


def main():
    assert os.environ.get("OPENROUTER_API_KEY")
    install_usage_recorder()
    with tempfile.TemporaryDirectory() as tmp:
        settings = replace(
            AIGridSettings(),
            enabled=True,
            mode="shadow",
            symbols=("ETH/USDT:USDT",),
            data_dir=ROOT / "user_data/data/binance/futures",
            state_dir=Path(tmp),
        )
        settings.validate()
        ei = ExchangeInfo(ROOT / ".orchestrator/fapi_exchangeinfo.json")
        orch = AIGridOrchestrator(settings, exchange_info=ei)
        # fresh-start recovery (no positions expected)
        rec = orch.recover(actual_open_pairs=set(), actual_open_order_ids=set())
        print(f"[recover] {rec['status']}")

        t0 = time.monotonic()
        res = orch.run_analysis("ETH/USDT:USDT", TriggerKind.PERIODIC)
        wall = time.monotonic() - t0
        print(f"\n[run_analysis] {res}")
        print(f"[wall_s] {wall:.1f}")

        plan = orch._engine.plan if orch._engine else None
        if plan is not None:
            print(
                f"\n[ADOPTED PLAN] mode={plan.grid_mode} "
                f"zone=[{plan.grid_lower},{plan.grid_upper}] "
                f"levels={plan.grid_levels} spacing={plan.grid_spacing}"
            )
            print(
                f"  margin={plan.max_capital} notional={plan.max_position_size} "
                f"lev={plan.max_leverage} level_margin={plan.level_stake} "
                f"level_notional={plan.level_notional}"
            )
            print(f"  stop_loss={plan.stop_loss} expires={plan.expires_at}")
            assert plan.max_position_size == plan.max_capital * plan.max_leverage
            assert plan.max_capital <= 100.0 + 1e-9  # per-symbol cap
            assert plan.max_leverage <= 3.0 + 1e-9  # risk cap
        else:
            print("[NO PLAN ADOPTED]")

        print(f"\n[scheduler] {orch.scheduler.cost_summary()}")
        tin = sum(u.get("prompt_tokens", 0) for u in usage_log)
        tout = sum(u.get("completion_tokens", 0) for u in usage_log)
        print(f"[tokens] calls={len(usage_log)} in={tin} out={tout}")
        # event log tail
        ev = Path(tmp) / "events.jsonl"
        if ev.exists():
            lines = ev.read_text().strip().splitlines()
            print(f"\n[eventlog] {len(lines)} entries; last:")
            for ln in lines[-4:]:
                print("  " + ln[:220])
        orch.close()


if __name__ == "__main__":
    main()
