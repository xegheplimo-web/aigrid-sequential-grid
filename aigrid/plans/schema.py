"""AIGridPlan — the structured contract between AI and execution.

The LLM proposes; Pydantic validates the shape; the Plan Validator + Risk
Engine validate the CONTENT. A plan that fails any check is rejected — its
parameters are never trusted directly.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class GridMode(StrEnum):
    RANGE_GRID = "RANGE_GRID"
    LONG_GRID = "LONG_GRID"
    SHORT_GRID = "SHORT_GRID"
    SAFE_HOLD = "SAFE_HOLD"


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


def new_plan_id() -> str:
    return f"plan-{uuid.uuid4().hex[:12]}"


class AIGridPlan(BaseModel):
    """Required fields per task spec section V. All prices in USDT (quote)."""

    plan_id: str = Field(min_length=8, max_length=64)
    plan_version: int = Field(ge=1, le=1000)
    symbol: str = Field(pattern=r"^[A-Z0-9]+/USDT:USDT$")
    created_at: datetime
    expires_at: datetime

    market_regime: str = Field(pattern=r"^(RANGE|UPTREND|DOWNTREND|HIGH_VOLATILITY)$")
    confidence: float = Field(ge=0.0, le=1.0)

    grid_mode: GridMode
    grid_lower: float = Field(gt=0)
    grid_upper: float = Field(gt=0)
    grid_levels: int = Field(ge=2, le=50)
    grid_spacing: float = Field(ge=0)  # fraction, e.g. 0.007 = 0.7% (0 for SAFE_HOLD)

    max_capital: float = Field(ge=0)  # MARGIN cấp cho plan (Capital Manager)
    max_position_size: float = Field(ge=0)  # NOTIONAL tối đa = margin × leverage
    max_leverage: float = Field(ge=1.0, le=10.0)
    level_stake: float = Field(ge=0, default=0.0)  # margin mỗi tầng
    level_notional: float = Field(ge=0, default=0.0)  # margin mỗi tầng × leverage

    stop_loss: float | None = Field(default=None, gt=0)
    take_profit: float | None = Field(default=None, gt=0)

    invalidation_conditions: list[str] = Field(default_factory=list, max_length=20)
    risk_level: RiskLevel = RiskLevel.MEDIUM
    reasoning_summary: str = Field(default="", max_length=2000)

    # ------------------------------------------------------------------ validators
    @field_validator("grid_upper")
    @classmethod
    def upper_above_lower(cls, v, info):
        lower = info.data.get("grid_lower")
        mode = info.data.get("grid_mode")
        if lower is not None and v <= lower and mode != GridMode.SAFE_HOLD:
            raise ValueError("grid_upper must be > grid_lower (except SAFE_HOLD)")
        return v

    @field_validator("expires_at")
    @classmethod
    def expiry_after_creation(cls, v, info):
        created = info.data.get("created_at")
        if created is not None and v <= created:
            raise ValueError("expires_at must be after created_at")
        return v

    # ------------------------------------------------------------------ helpers
    def is_expired(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(UTC)
        exp = self.expires_at if self.expires_at.tzinfo else self.expires_at.replace(tzinfo=UTC)
        return now >= exp

    def ttl_seconds(self) -> float:
        created = self.created_at if self.created_at.tzinfo else self.created_at.replace(tzinfo=UTC)
        exp = self.expires_at if self.expires_at.tzinfo else self.expires_at.replace(tzinfo=UTC)
        return (exp - created).total_seconds()

    def level_prices(self) -> list[float]:
        """Buy-level prices from grid_upper down to grid_lower (grid_levels entries)."""
        if self.grid_levels < 2:
            return [self.grid_upper]
        step = (self.grid_upper - self.grid_lower) / (self.grid_levels - 1)
        return [round(self.grid_upper - i * step, 8) for i in range(self.grid_levels)]


def draft_plan_from_assessment(
    pair: str,
    regime: str,
    confidence: float,
    last_close: float,
    atr: float,
    *,
    grid_mode: str,
    ttl_minutes: int,
    margin_usdt: float,  # từ CapitalManager (không phải LLM)
    leverage: float,
    levels: int,
    level_margin_usdt: float | None = None,
    reasoning: str = "",
    spacing_atr_mult: float = 0.5,
    grid_lower: float | None = None,  # Agent 3 đề xuất (validator vẫn kiểm)
    grid_upper: float | None = None,
    risk_level: str = "MEDIUM",
) -> AIGridPlan:
    """Deterministic plan DRAFT: LLM đề xuất regime/zone (tham khảo); vốn đến từ
    CapitalManager; geometry bị chặn trong biên an toàn; validator kiểm cuối."""
    now = datetime.now(UTC)
    if grid_mode == "SAFE_HOLD":
        # Degenerate plan: no grid; validator accepts it as a hold instruction.
        return AIGridPlan(
            plan_id=new_plan_id(),
            plan_version=1,
            symbol=pair,
            created_at=now,
            expires_at=now + timedelta(minutes=ttl_minutes),
            market_regime=regime,
            confidence=confidence,
            grid_mode=GridMode.SAFE_HOLD,
            grid_lower=last_close,
            grid_upper=last_close,
            grid_levels=2,
            grid_spacing=0.0,
            max_capital=0.0,
            max_position_size=0.0,
            max_leverage=1.0,
            level_stake=0.0,
            level_notional=0.0,
            stop_loss=None,
            take_profit=None,
            invalidation_conditions=["SAFE_HOLD_NO_NEW_POSITIONS"],
            risk_level=RiskLevel.LOW,
            reasoning_summary=reasoning or "Safe hold: no new positions.",
        )
    spacing = max(0.0015, min(0.05, (atr / last_close) * spacing_atr_mult))
    half = spacing * levels * 0.5
    if grid_lower is not None and grid_upper is not None and grid_upper > grid_lower:
        # Agent 3 đề xuất zone — clamp vào biên an toàn quanh giá (±25%)
        lo_bound, hi_bound = last_close * 0.75, last_close * 1.25
        lower = max(lo_bound, min(hi_bound, grid_lower))
        upper = max(lo_bound, min(hi_bound, grid_upper))
        if upper <= lower:
            lower, upper = last_close * (1 - half), last_close * (1 + half)
    elif grid_mode == "LONG_GRID":
        lower, upper = last_close * (1 - 2 * half), last_close * (1 - 0.1 * spacing)
    elif grid_mode == "SHORT_GRID":
        lower, upper = last_close * (1 + 0.1 * spacing), last_close * (1 + 2 * half)
    else:  # RANGE_GRID centered on price
        lower, upper = last_close * (1 - half), last_close * (1 + half)
    stop = lower * (1 - spacing) if grid_mode != "SHORT_GRID" else upper * (1 + spacing)
    level_margin = (
        level_margin_usdt if level_margin_usdt is not None else margin_usdt / max(1, levels)
    )
    return AIGridPlan(
        plan_id=new_plan_id(),
        plan_version=1,
        symbol=pair,
        created_at=now,
        expires_at=now + timedelta(minutes=ttl_minutes),
        market_regime=regime,
        confidence=confidence,
        grid_mode=GridMode(grid_mode),
        grid_lower=round(lower, 8),
        grid_upper=round(upper, 8),
        grid_levels=max(2, min(50, levels)),
        grid_spacing=round(spacing, 6),
        max_capital=round(margin_usdt, 2),
        max_position_size=round(margin_usdt * leverage, 2),
        max_leverage=leverage,
        level_stake=round(level_margin, 2),
        level_notional=round(level_margin * leverage, 2),
        stop_loss=round(stop, 8),
        take_profit=None,
        invalidation_conditions=[
            "PRICE_BREAKS_GRID_BOUNDARY",
            "REGIME_CHANGES_AWAY_FROM_PLAN",
            "STOP_LOSS_TRIGGERED",
        ],
        risk_level=RiskLevel(risk_level),
        reasoning_summary=reasoning[:2000],
    )
