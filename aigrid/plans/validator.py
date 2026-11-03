"""AI Grid Plan Validator — schema + content + exchange-rule + risk-bound checks.

A plan is APPROVED only if every check passes. Any failure -> REJECTED with a
machine-readable reason. Rejected plans are never executed and never retried
silently (they surface in the dashboard and state log).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from ..exchange_info import ExchangeInfo
from ..settings import RiskLimits
from .schema import AIGridPlan, GridMode


@dataclass(frozen=True)
class ValidationVerdict:
    approved: bool
    reasons: tuple[str, ...]  # empty when approved
    plan: AIGridPlan | None

    def as_dict(self) -> dict:
        return {
            "approved": self.approved,
            "reasons": list(self.reasons),
            "plan_id": self.plan.plan_id if self.plan else None,
        }


class PlanValidator:
    def __init__(self, exchange_info: ExchangeInfo, limits: RiskLimits, capital=None):
        self.ei = exchange_info
        self.limits = limits
        self.capital = capital  # CapitalManager (optional, for budget caps)

    def validate(
        self,
        plan: AIGridPlan,
        *,
        last_close: float,
        now: datetime | None = None,
        total_exposure_after: float = 0.0,
    ) -> ValidationVerdict:
        reasons: list[str] = []
        now = now or datetime.now(UTC)

        # ---- 1. TTL / expiry
        try:
            ttl = plan.ttl_seconds()
        except Exception:
            return ValidationVerdict(False, ("PLAN_TIME_FIELDS_INVALID",), None)
        if ttl <= 0:
            reasons.append("PLAN_TTL_NON_POSITIVE")
        if ttl > self.limits.plan_ttl_minutes * 60 * 4:
            reasons.append("PLAN_TTL_EXCEEDS_LIMIT")
        if plan.is_expired(now):
            reasons.append("PLAN_ALREADY_EXPIRED")

        # ---- 2. symbol known to exchange
        try:
            rules = self.ei.rules_for(plan.symbol)
        except KeyError:
            return ValidationVerdict(False, ("SYMBOL_NOT_IN_EXCHANGE_INFO",), None)

        # ---- 3. SAFE_HOLD plans skip grid geometry checks (no orders will be placed)
        if plan.grid_mode == GridMode.SAFE_HOLD:
            if plan.max_capital != 0.0 or plan.max_position_size != 0.0:
                reasons.append("SAFE_HOLD_WITH_NONZERO_CAPITAL")
            return self._verdict(plan, reasons)

        # ---- 3b. capital consistency: notional = margin × leverage
        if plan.max_position_size + 1e-6 < plan.max_capital * plan.max_leverage:
            reasons.append("NOTIONAL_LESS_THAN_MARGIN_TIMES_LEVERAGE")
        if plan.max_capital > 0:
            if self.capital is not None:
                if plan.max_capital > self.capital.settings.per_symbol_margin_cap + 1e-6:
                    reasons.append("MARGIN_EXCEEDS_PER_SYMBOL_CAP")
                if (
                    plan.max_capital
                    > self.capital.total_budget(self.capital.settings.per_symbol_margin_cap * 10)
                    + 1e-6
                ):
                    pass  # total budget checked by Risk Engine with real equity
            if (
                plan.level_stake > 0
                and abs(plan.level_stake * plan.grid_levels - plan.max_capital)
                > 0.05 * plan.max_capital
            ):
                reasons.append("LEVEL_STAKE_INCONSISTENT_WITH_MARGIN")

        # ---- 4. grid geometry
        if not (0 < plan.grid_lower < plan.grid_upper):
            reasons.append("GRID_BOUNDS_INVALID")
        else:
            span_pct = (plan.grid_upper - plan.grid_lower) / last_close
            if span_pct < 0.002:
                reasons.append("GRID_SPAN_TOO_NARROW")
            if span_pct > 0.5:
                reasons.append("GRID_SPAN_TOO_WIDE")
            # price sanity: grid must bracket the current price region (±40%)
            if not (plan.grid_lower * 0.6 <= last_close <= plan.grid_upper * 1.4):
                reasons.append("GRID_FAR_FROM_MARKET_PRICE")
            # price within exchange bounds
            if plan.grid_upper > rules.max_price or plan.grid_lower < rules.min_price:
                reasons.append("GRID_OUTSIDE_EXCHANGE_PRICE_LIMITS")

        if not 2 <= plan.grid_levels <= 50:
            reasons.append("GRID_LEVELS_OUT_OF_BOUNDS")
        if not 0.0005 <= plan.grid_spacing <= 0.05:
            reasons.append("GRID_SPACING_OUT_OF_BOUNDS")

        # ---- 5. per-level exchange rules (tick / lot / minNotional / fees payable)
        if not reasons or all(r not in ("GRID_BOUNDS_INVALID",) for r in reasons):
            reasons.extend(self._check_levels(plan, rules, last_close))

        # ---- 6. capital / leverage / exposure (NOTIONAL = margin × leverage)
        if plan.max_position_size > self.limits.max_position_notional_usdt:
            reasons.append("PLAN_CAPITAL_EXCEEDS_POSITION_LIMIT")
        if total_exposure_after + plan.max_position_size > self.limits.max_total_exposure_usdt:
            reasons.append("PLAN_CAPITAL_EXCEEDS_TOTAL_EXPOSURE")
        if plan.max_leverage > self.limits.max_leverage:
            reasons.append("PLAN_LEVERAGE_EXCEEDS_LIMIT")

        # ---- 7. stop loss sanity (must sit outside the grid on the risk side)
        if plan.stop_loss is not None:
            if plan.grid_mode in (GridMode.RANGE_GRID, GridMode.LONG_GRID):
                if plan.stop_loss >= plan.grid_lower:
                    reasons.append("STOP_LOSS_INSIDE_OR_ABOVE_GRID")
            elif plan.grid_mode == GridMode.SHORT_GRID:
                if plan.stop_loss <= plan.grid_upper:
                    reasons.append("STOP_LOSS_INSIDE_OR_BELOW_GRID")
            # SL must be within ±25% of last close (protects against absurd levels)
            if not 0.75 * last_close <= plan.stop_loss <= 1.25 * last_close:
                reasons.append("STOP_LOSS_FAR_FROM_MARKET")

        # ---- 8. confidence gate
        if plan.confidence < 0.3:
            reasons.append("CONFIDENCE_TOO_LOW")

        return self._verdict(plan, reasons)

    # ------------------------------------------------------------------ levels
    def _check_levels(self, plan: AIGridPlan, rules, last_close: float) -> list[str]:
        reasons: list[str] = []
        tick = rules.tick_size
        step = rules.step_size
        if tick <= 0 or step <= 0:
            return ["EXCHANGE_RULES_MISSING"]
        # per-level NOTIONAL = level margin × leverage (margin-based sizing)
        level_notional = (
            plan.level_notional
            if plan.level_notional > 0
            else plan.max_position_size / max(1, plan.grid_levels)
        )
        for price in plan.level_prices():
            # tick size: price must be a multiple of tick (within float epsilon)
            if tick > 0:
                ratio = price / tick
                if abs(ratio - round(ratio)) > 1e-6:
                    reasons.append("LEVEL_PRICE_NOT_ON_TICK")
                    break
            # min notional incl. fee buffer: level must be payable
            if level_notional < rules.min_notional * (1 + self.limits.fee_rate_taker) * 1.05:
                reasons.append("LEVEL_NOTIONAL_BELOW_MIN_NOTIONAL")
                break
            qty = level_notional / price
            if qty < rules.min_qty:
                reasons.append("LEVEL_QTY_BELOW_MIN_QTY")
                break
            # step size: qty must be a multiple of step
            if step > 0:
                qratio = qty / step
                if abs(qratio - round(qratio)) > 1e-6 and qty < rules.max_qty:
                    # rounding to step must not drop below min notional
                    rounded = round(qratio) * step
                    if rounded * price < rules.min_notional:
                        reasons.append("LEVEL_QTY_STEP_ROUNDING_BREAKS_MIN_NOTIONAL")
                        break
        return reasons

    @staticmethod
    def _verdict(plan: AIGridPlan, reasons: list[str]) -> ValidationVerdict:
        return ValidationVerdict(
            approved=not reasons, reasons=tuple(reasons), plan=None if reasons else plan
        )
