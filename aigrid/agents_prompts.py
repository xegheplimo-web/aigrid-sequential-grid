"""Prompt renderers for the 3-agent pipeline (kept separate for testability)."""

from __future__ import annotations

import json


def _ctx_block(ctx: dict) -> str:
    return json.dumps(ctx, ensure_ascii=False, default=str, indent=1)


def render_agent1_prompt(ctx: dict) -> str:
    return (
        "Market snapshot (deterministic, from Binance USDT-M futures data, "
        "closed candles only):\n"
        f"{_ctx_block(ctx)}\n\n"
        "Tasks:\n"
        "1. Identify the current trend (UP/DOWN/SIDEWAYS) and its strength.\n"
        "2. Assess momentum and volatility from the indicators.\n"
        "3. Give the nearest support and resistance levels based on the data.\n"
        "4. Note anything about liquidity (quote volume) and funding.\n"
        "Respond with the structured TechnicalReport only."
    )


def render_agent2_prompt(ctx: dict, tech) -> str:
    return (
        f"Market snapshot:\n{_ctx_block(ctx)}\n\n"
        f"Agent 1 technical report:\n{tech.model_dump_json(indent=1)}\n\n"
        "Tasks:\n"
        "1. List the main risks of trading a grid RIGHT NOW on this symbol.\n"
        "2. Assess trend-break risk (LOW/MEDIUM/HIGH) and volatility.\n"
        "3. Check funding-rate concern (cost of holding positions).\n"
        "4. Decide: are conditions OK for a grid strategy? (conditions_ok_for_grid)\n"
        "5. State concrete objections if any.\n"
        "Respond with the structured RiskCritique only."
    )


def render_agent3_prompt(ctx: dict, tech, critique) -> str:
    return (
        f"Market snapshot:\n{_ctx_block(ctx)}\n\n"
        f"Agent 1 technical report:\n{tech.model_dump_json(indent=1)}\n\n"
        f"Agent 2 risk critique:\n{critique.model_dump_json(indent=1)}\n\n"
        "Tasks:\n"
        "1. Final regime + confidence.\n"
        "2. Final grid_mode (SAFE_HOLD if the critic objects or volatility is dangerous).\n"
        "3. Grid zone (grid_lower/grid_upper) bracketing the last close, width "
        "proportional to volatility (suggest roughly 2-6x the ATR% total width).\n"
        "4. levels: 3-8 grid levels inside the zone.\n"
        "5. max_capital_suggestion: advisory margin in USDT (a separate Capital "
        "Manager enforces the real budget).\n"
        "6. reasoning_summary: 3-6 sentences.\n"
        "Respond with the structured StrategyProposal only."
    )
