"""Deterministic plan builder (R14 / D-093) — the no-LLM plan path.

A Docker dry-run must be able to run
``closed-candle -> REGIME check -> deterministic plan -> guarded publish``
without any LLM call on the critical path. This module derives an
:class:`~aigrid.plans.schema.AIGridPlan` purely from closed 5m candles,
the confirmed regime from ``regime/controller.py`` and the capital caps
from ``settings.py``.

Rules (card R14 §1 + addendum v3):

- Regime ``RANGE`` only (side/low-volatility). UPTREND / DOWNTREND /
  HIGH_VOLATILITY or a paused controller -> ``None`` + event
  ``PLAN_SKIPPED_REGIME``. ``ai_status`` OK / UNAVAILABLE / OFF /
  NO_ANALYSIS_YET all permit the path — AI may not veto it.
- ``grid_upper = max(high)``, ``grid_lower = min(low)`` over the last 20
  closed candles, only expanded OUTWARDS so every level lands on the
  exchange tick (validator ``LEVEL_PRICE_NOT_ON_TICK``).
- MinSpacing (D-099/D-100): ``spacing = max(0.30% dry-run floor,
  0.75 x ATR5m p70 measured, 3.0 x C_stress)`` with
  ``C_stress = maker-maker roundtrip + AS p85 reserve``. Funding is a
  holding-cost budget tracked SEPARATELY — never inside spacing.
- If the level step implied by the ref-20 range would clamp below the
  floor -> ``None``. Spacing is never narrowed to trade more.
- ``stop_loss = grid_lower x (1 - 0.03)`` — always past the 3%
  liquidation buffer.
- Capital: ``max_leverage`` pinned 1.0 (D-078); ``max_capital`` clamped to
  ``min(per_symbol_margin_cap, max_position_notional_usdt)``;
  ``grid_levels`` default 4 clamped by the per-level margin floor.
- TTL: ``expires_at = as_of + 600s`` — ``as_of`` is the last closed
  candle's close time, so the same input yields a byte-identical plan
  (``plan_id`` is a stable hash of the inputs).
"""

from __future__ import annotations

import hashlib
import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from ..regime.engine import compute_features
from ..settings import AIGridSettings
from .schema import AIGridPlan, GridMode, RiskLevel

PLAN_TTL_SECONDS = 600  # 10 minutes — never hardcode elsewhere
DRY_RUN_SPACING_FLOOR = 0.0030  # 0.30% dry-run floor (D-099)
ATR_SPACING_COEF = 0.75  # m: ATR5m p70 coefficient (v3)
C_STRESS_K = 3.0  # k policy: 3 x C_stress (D-100)
ADVERSE_SELECTION_P85 = 0.00015  # AS p85 reserve (0.015%)
REF_WINDOW = 20  # ref-20 candle envelope
DEFAULT_GRID_LEVELS = 4  # card default; clamped by settings
STOP_LOSS_BUFFER = 0.03  # >= 3% liquidation buffer
ATR_P70_Q = 0.70  # p70 of ATR(14)/close

# Regimes that may produce a deterministic plan (side market only).
TRADEABLE_REGIME = "RANGE"
# ai_status values that permit the deterministic path — AI has no veto.
AI_STATUS_ALLOWED = frozenset({"OK", "UNAVAILABLE", "OFF", "NO_ANALYSIS_YET"})


def atr_p70_pct(ohlcv: pd.DataFrame) -> float:
    """70th percentile of causal ATR(14)/close over the given closed candles.

    Uses the regime engine's causal feature series (past rows only), so the
    same frame always yields the same value.
    """
    feats = compute_features(ohlcv)
    atr = feats["atr_pct"].dropna()
    if not len(atr):
        return 0.0
    return float(atr.quantile(ATR_P70_Q))


def spacing_floor(
    ohlcv: pd.DataFrame,
    *,
    maker_roundtrip: float,
    as_p85: float = ADVERSE_SELECTION_P85,
    funding_rate_avg: float | None = None,
    expected_settlements: int = 3,
) -> dict[str, float]:
    """MinSpacing decomposition (v3): each term reported separately.

    ``spacing_min = max(0.30% floor, 0.75 x ATR5m p70, 3.0 x C_stress)``
    with ``C_stress = maker_roundtrip + as_p85``. ``funding_budget`` is the
    holding-cost reserve (mean funding x expected 8h settlements) — tracked
    alongside, deliberately NOT part of spacing.
    """
    atr_p70 = atr_p70_pct(ohlcv)
    c_stress = maker_roundtrip + as_p85
    funding_budget = abs(funding_rate_avg or 0.0) * expected_settlements
    return {
        "spacing_min": max(
            DRY_RUN_SPACING_FLOOR, ATR_SPACING_COEF * atr_p70, C_STRESS_K * c_stress
        ),
        "floor": DRY_RUN_SPACING_FLOOR,
        "atr_p70": atr_p70,
        "atr_term": ATR_SPACING_COEF * atr_p70,
        "c_stress": c_stress,
        "c_stress_term": C_STRESS_K * c_stress,
        "funding_budget": funding_budget,
    }


def build_deterministic_plan(
    pair: str,
    ohlcv: pd.DataFrame,
    *,
    regime: str,
    settings: AIGridSettings,
    as_of: datetime,
    ai_status: str = "UNAVAILABLE",
    paused: bool = False,
    pause_reason: str = "",
    confidence: float = 0.5,
    price_tick: float | None = None,
    funding_rate_avg: float | None = None,
    eventlog: Any = None,
) -> AIGridPlan | None:
    """Build a deterministic AIGridPlan from closed candles, or None.

    ``as_of`` is the close time of the last closed candle — the plan is
    valid for PLAN_TTL_SECONDS from that point (a plan built on stale data
    is already decayed). Same input -> byte-identical plan.
    """
    # ---- gates -----------------------------------------------------------
    if paused or regime != TRADEABLE_REGIME:
        _log(
            eventlog,
            "PLAN_SKIPPED_REGIME",
            pair,
            {"regime": regime, "paused": bool(paused), "pause_reason": pause_reason},
        )
        return None
    if ai_status not in AI_STATUS_ALLOWED:
        _log(eventlog, "PLAN_SKIPPED_AI_STATUS", pair, {"ai_status": ai_status})
        return None
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=UTC)

    closes = ohlcv
    if len(closes) < REF_WINDOW:
        _log(
            eventlog,
            "PLAN_SKIPPED_DATA",
            pair,
            {"reason": "INSUFFICIENT_REF_WINDOW", "candles": len(closes)},
        )
        return None
    ref = closes.iloc[-REF_WINDOW:]
    last_close = float(closes["close"].iloc[-1])
    if not (math.isfinite(last_close) and last_close > 0):
        _log(eventlog, "PLAN_SKIPPED_DATA", pair, {"reason": "INVALID_LAST_CLOSE"})
        return None

    # ---- MinSpacing floor (v3): never narrow spacing to trade more --------
    info = spacing_floor(
        closes,
        maker_roundtrip=2.0 * float(settings.risk.fee_rate_maker),
        funding_rate_avg=funding_rate_avg,
    )
    spacing_min = info["spacing_min"]

    # ---- capital caps (settings, never the market) ------------------------
    max_capital = min(
        float(settings.capital.per_symbol_margin_cap),
        float(settings.risk.max_position_notional_usdt),
    )
    levels = min(
        DEFAULT_GRID_LEVELS,
        int(settings.capital.max_levels),
        int(max_capital // max(1e-9, float(settings.capital.min_level_margin_usdt))),
    )
    if levels < 2 or max_capital <= 0:
        _log(
            eventlog,
            "PLAN_SKIPPED_CAPITAL",
            pair,
            {"levels": levels, "max_capital": max_capital},
        )
        return None

    # ---- grid bounds: ref-20 envelope, expanded outwards onto the tick ----
    lower, upper = _align_bounds(
        float(ref["low"].min()), float(ref["high"].max()), levels, price_tick
    )
    if not upper > lower:
        _log(
            eventlog,
            "PLAN_SKIPPED_GEOMETRY",
            pair,
            {"grid_lower": lower, "grid_upper": upper},
        )
        return None
    step_frac = (upper - lower) / (levels - 1) / last_close
    if step_frac < spacing_min:
        _log(
            eventlog,
            "PLAN_SKIPPED_SPACING",
            pair,
            {"step_frac": step_frac, "spacing_min": spacing_min},
        )
        return None  # clamp would violate the floor -> NO PLAN

    level_stake = math.floor(max_capital / levels * 100) / 100
    created_ms = int(as_of.timestamp() * 1000)
    canon = f"{pair}|{lower:.8f}|{upper:.8f}|{levels}|{spacing_min:.8f}|{created_ms}"
    plan_id = f"plan-{hashlib.sha256(canon.encode()).hexdigest()[:12]}"

    return AIGridPlan(
        plan_id=plan_id,
        plan_version=1,
        symbol=pair,
        created_at=as_of,
        expires_at=as_of + timedelta(seconds=PLAN_TTL_SECONDS),
        market_regime=TRADEABLE_REGIME,
        confidence=max(0.3, min(0.95, float(confidence))),
        grid_mode=GridMode.RANGE_GRID,
        grid_lower=round(lower, 8),
        grid_upper=round(upper, 8),
        grid_levels=levels,
        grid_spacing=round(step_frac, 6),
        max_capital=round(max_capital, 2),
        max_position_size=round(max_capital, 2),  # leverage pinned 1.0
        max_leverage=1.0,
        level_stake=level_stake,
        level_notional=level_stake,
        stop_loss=round(lower * (1.0 - STOP_LOSS_BUFFER), 8),
        take_profit=None,
        invalidation_conditions=[
            "PRICE_BREAKS_GRID_BOUNDARY",
            "REGIME_CHANGES_AWAY_FROM_PLAN",
            "STOP_LOSS_TRIGGERED",
        ],
        risk_level=RiskLevel.LOW,
        reasoning_summary=(
            f"deterministic RANGE plan: step={step_frac:.6f} "
            f"(floor={spacing_min:.6f} atr_p70={info['atr_p70']:.5f} "
            f"c_stress={info['c_stress']:.5f}) "
            f"funding_budget={info['funding_budget']:.5f} (separate)"
        ),
    )


def _align_bounds(
    lower: float, upper: float, levels: int, tick: float | None
) -> tuple[float, float]:
    """Snap bounds to tick by EXPANSION only (lower down, upper up), then
    grow upper until the level step is an exact tick multiple — every
    published level lands on the exchange grid. No-op without a tick."""
    if not tick or tick <= 0 or levels < 2:
        return lower, upper
    lo = math.floor(lower / tick + 1e-9)
    hi = math.ceil(upper / tick - 1e-9)
    if hi <= lo:
        hi = lo + (levels - 1)
    while (hi - lo) % (levels - 1) != 0:
        hi += 1
    return round(lo * tick, 8), round(hi * tick, 8)


def _log(eventlog: Any, kind: str, symbol: str, payload: dict) -> None:
    if eventlog is not None:
        eventlog.log(kind, symbol, payload)
