"""Independent Risk Engine — vetoes AI, Grid Engine, and entry signals.

Runs WITHOUT any AI dependency: when the LLM is down/expired/wrong, this engine
still enforces exposure, loss, drawdown, leverage, liquidation buffer, funding
cost, margin, data freshness, duplicate-order and position-mismatch rules, and
may trigger emergency exits (PROTECT) on its own authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..plans.schema import AIGridPlan, GridMode
from ..settings import RiskLimits
from .events import RiskEvent
from .ownership import OwnershipRegistry


@dataclass
class AccountState:
    """Snapshot of the trading account (from Freqtrade API in dry-run)."""

    equity_usdt: float
    starting_equity_usdt: float  # day start / initial
    peak_equity_usdt: float
    realized_pnl_today_usdt: float
    total_exposure_usdt: float  # sum |notional| across positions
    free_margin_usdt: float
    funding_cost_today_usdt: float = 0.0
    allocated_margins: dict[str, float] = field(default_factory=dict)  # symbol -> margin
    positions: dict[str, dict] = field(default_factory=dict)  # pair -> {side, notional, ...}


@dataclass
class MarketFreshness:
    last_candle_close_ms: int
    now_ms: int
    timeframe_seconds: int

    @property
    def stale_seconds(self) -> int:
        return max(0, (self.now_ms - self.last_candle_close_ms) // 1000)


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    events: tuple[str, ...]
    emergency_action: str | None = None  # None | "REDUCE" | "CLOSE_ALL"
    detail: dict = field(default_factory=dict)


class RiskEngine:
    def __init__(self, limits: RiskLimits, ownership: OwnershipRegistry | None = None):
        self.limits = limits
        self.ownership = ownership or OwnershipRegistry()
        self._halted_symbols: dict[str, str] = {}

    # ------------------------------------------------------------------ entry veto
    def evaluate_entry(
        self,
        plan: AIGridPlan,
        account: AccountState,
        freshness: MarketFreshness | None = None,
        existing_open_orders: tuple[str, ...] = (),
        capital_manager=None,
        now: datetime | None = None,
    ) -> RiskDecision:
        """Veto/allow opening (or adding to) positions under this plan."""
        events: list[str] = []
        now = now or datetime.now(UTC)

        # AI plan validity (independent checks — engine does not trust the AI)
        if plan.is_expired(now):
            events.append(str(RiskEvent.AI_PLAN_EXPIRED))
        if plan.grid_mode != GridMode.SAFE_HOLD:
            # ownership: one strategy family per symbol; RISK_LOCKED vetoes all
            ok, reason = self.ownership.can_open(plan.symbol, "grid")
            if not ok:
                events.append(reason or str(RiskEvent.OWNERSHIP_CONFLICT))
            # NOTIONAL exposure = margin × leverage
            exposure_after = account.total_exposure_usdt + plan.max_position_size
            if exposure_after > self.limits.max_total_exposure_usdt:
                events.append(str(RiskEvent.EXPOSURE_LIMIT))
            if plan.max_position_size > self.limits.max_position_notional_usdt:
                events.append(str(RiskEvent.EXPOSURE_LIMIT))
            if plan.max_leverage > self.limits.max_leverage:
                events.append(str(RiskEvent.LEVERAGE_LIMIT))
            if account.free_margin_usdt < self.limits.min_free_margin_usdt:
                events.append(str(RiskEvent.MARGIN_LOW))
            # capital budget: total allocated margin must stay within budget
            if capital_manager is not None:
                margins = dict(account.allocated_margins)
                margins[plan.symbol] = margins.get(plan.symbol, 0.0) + plan.max_capital
                budget = capital_manager.total_budget(account.equity_usdt)
                if sum(margins.values()) > budget + 1e-6:
                    events.append(str(RiskEvent.EXPOSURE_LIMIT))
                if plan.max_capital > capital_manager.settings.per_symbol_margin_cap + 1e-6:
                    events.append(str(RiskEvent.EXPOSURE_LIMIT))
            # liquidation buffer (approx: notional/leverage vs equity)
            if plan.max_leverage > 1.0:
                buffer_pct = (
                    100.0
                    * (account.equity_usdt - account.total_exposure_usdt / plan.max_leverage)
                    / max(account.equity_usdt, 1e-9)
                )
                if buffer_pct < self.limits.min_liquidation_buffer_pct:
                    events.append(str(RiskEvent.LIQUIDATION_BUFFER_LOW))
        if account.funding_cost_today_usdt > self.limits.max_funding_cost_daily_usdt:
            events.append(str(RiskEvent.FUNDING_LIMIT_EXCEEDED))
        if (
            freshness is not None
            and freshness.stale_seconds
            > self.limits.data_max_staleness_factor * freshness.timeframe_seconds
        ):
            events.append(str(RiskEvent.DATA_STALE))
        # duplicate order prevention
        if plan.plan_id in existing_open_orders:
            events.append(str(RiskEvent.DUP_ORDER_BLOCKED))

        # daily loss / drawdown (independent of AI)
        day_loss_pct = (
            -100.0 * account.realized_pnl_today_usdt / max(account.starting_equity_usdt, 1e-9)
        )
        if day_loss_pct >= self.limits.max_daily_loss_pct:
            events.append(str(RiskEvent.DAILY_LOSS_LIMIT))
        dd_pct = (
            100.0
            * (account.peak_equity_usdt - account.equity_usdt)
            / max(account.peak_equity_usdt, 1e-9)
        )
        if dd_pct >= self.limits.max_drawdown_pct:
            events.append(str(RiskEvent.DRAWDOWN_LIMIT))

        allowed = not events
        return RiskDecision(allowed=allowed, events=tuple(events))

    # ------------------------------------------------------------------ continuous monitoring
    def evaluate_account(self, account: AccountState) -> RiskDecision:
        """Periodic account-level check; may trigger EMERGENCY (PROTECT)."""
        events: list[str] = []
        emergency: str | None = None
        day_loss_pct = (
            -100.0 * account.realized_pnl_today_usdt / max(account.starting_equity_usdt, 1e-9)
        )
        if day_loss_pct >= self.limits.max_daily_loss_pct:
            events.append(str(RiskEvent.MAX_LOSS_REACHED))
            emergency = "CLOSE_ALL"
        dd_pct = (
            100.0
            * (account.peak_equity_usdt - account.equity_usdt)
            / max(account.peak_equity_usdt, 1e-9)
        )
        if dd_pct >= self.limits.max_drawdown_pct:
            events.append(str(RiskEvent.DRAWDOWN_LIMIT))
            emergency = emergency or "REDUCE"
        if account.funding_cost_today_usdt > self.limits.max_funding_cost_daily_usdt:
            events.append(str(RiskEvent.FUNDING_LIMIT_EXCEEDED))
            emergency = emergency or "REDUCE"
        if account.free_margin_usdt < self.limits.min_free_margin_usdt:
            events.append(str(RiskEvent.MARGIN_LOW))
            emergency = emergency or "REDUCE"
        if account.total_exposure_usdt > self.limits.max_total_exposure_usdt:
            events.append(str(RiskEvent.EXPOSURE_LIMIT))
            emergency = emergency or "REDUCE"
        return RiskDecision(allowed=not events, events=tuple(events), emergency_action=emergency)

    # ------------------------------------------------------------------ position reconciliation
    def check_position_mismatch(
        self, expected_pairs: set[str], actual_pairs: set[str], symbol: str
    ) -> RiskDecision:
        """Detect positions nobody manages (unexpected) or missing positions."""
        unexpected = actual_pairs - expected_pairs
        if symbol in unexpected:
            return RiskDecision(
                False,
                (str(RiskEvent.POSITION_MISMATCH),),
                emergency_action="CLOSE_ALL",
                detail={"unexpected": sorted(unexpected)},
            )
        return RiskDecision(True, (), detail={"unexpected": sorted(unexpected)})

    # ------------------------------------------------------------------ AI loss handling
    def on_ai_unavailable(self) -> RiskDecision:
        """AI down: no new positions from expired plans; protections continue."""
        return RiskDecision(False, (str(RiskEvent.AI_UNAVAILABLE),))

    # ------------------------------------------------------------------ ownership helpers
    def lock_symbol(self, symbol: str, reason: str) -> None:
        self.ownership.risk_lock(symbol, reason)
        self._halted_symbols[symbol] = reason

    def unlock_symbol(self, symbol: str) -> None:
        self.ownership.risk_unlock(symbol)
        self._halted_symbols.pop(symbol, None)

    def is_halted(self, symbol: str) -> bool:
        return symbol in self._halted_symbols
