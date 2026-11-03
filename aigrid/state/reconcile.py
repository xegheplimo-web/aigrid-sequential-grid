"""Restart recovery — reconcile persisted grid state against Freqtrade reality.

On startup the system MUST NOT blindly re-place the grid: it reconciles the
persisted plan/lots against actual FT positions/orders. Any mismatch it cannot
resolve -> SAFE_HOLD (fail-closed) + POSITION_MISMATCH event.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..plans.schema import AIGridPlan
from ..risk.events import RiskEvent
from .store import StateStore


@dataclass
class ReconciliationResult:
    status: str  # RESTORED | SAFE_HOLD | FRESH_START
    reasons: tuple[str, ...] = ()
    plan: AIGridPlan | None = None
    restored_lots: dict = field(default_factory=dict)
    unexpected_positions: tuple[str, ...] = ()
    filled_levels: tuple[int, ...] = ()


class RestartReconciler:
    def __init__(self, store: StateStore):
        self.store = store

    def recover(
        self,
        actual_open_pairs: set[str],
        actual_open_order_ids: set[str],
        now_ms: int,
        *,
        state_key: str = "grid_state",
        plan_key: str = "current_plan",
        actual_open_lots: dict[str, int] | None = None,
    ) -> ReconciliationResult:
        """Compare persisted state with live FT reality.

        actual_open_pairs: pairs with open FT positions (normalized, e.g. ETHUSDT)
        actual_open_order_ids: client order ids FT currently has pending
        state_key/plan_key: per-pair snapshot names (multi-pair mode).
        actual_open_lots: normalized symbol -> number of grid lots FT really
            holds; when given, a mismatch with persisted lots is a one-time
            entry-gate reconciliation failure -> SAFE_HOLD (no auto-fix).
        """
        snap = self.store.load_state(state_key)
        plan_data = self.store.load_state(plan_key)
        reasons: list[str] = []

        # ---- no persisted state: fresh start (nothing to restore)
        if snap is None and plan_data is None:
            unexpected = tuple(sorted(p for p in actual_open_pairs))
            if unexpected:
                return ReconciliationResult(
                    "SAFE_HOLD", (str(RiskEvent.POSITION_MISMATCH),), None, {}, unexpected
                )
            return ReconciliationResult("FRESH_START")

        # ---- restore plan
        plan: AIGridPlan | None = None
        if plan_data is not None:
            try:
                plan = AIGridPlan.model_validate(plan_data["plan"])
            except Exception as exc:
                reasons.append(f"PLAN_UNREADABLE:{exc}")
                plan = None

        if plan is not None and plan.is_expired():
            reasons.append(str(RiskEvent.AI_PLAN_EXPIRED))
            # expired plan with open positions -> keep protections, no new grid
            if actual_open_pairs:
                return ReconciliationResult("SAFE_HOLD", tuple(reasons), plan, {}, ())
            return ReconciliationResult("SAFE_HOLD", tuple(reasons), None, {}, ())

        # ---- restore lots
        lots = (snap or {}).get("open_lots", {})
        try:
            filled_levels = tuple(int(x) for x in (snap or {}).get("filled_levels", []) or ())
        except (TypeError, ValueError):
            filled_levels = ()
        symbol = (plan_data or {}).get("symbol", "")
        symbol_short = symbol.split(":", 1)[0].replace("/", "") if symbol else ""

        # positions FT holds that our state does not know about -> mismatch
        unexpected = tuple(sorted(p for p in actual_open_pairs if p != symbol_short))
        if unexpected:
            reasons.append(str(RiskEvent.POSITION_MISMATCH))

        if plan is None or not symbol_short:
            if unexpected or actual_open_pairs:
                return ReconciliationResult("SAFE_HOLD", tuple(reasons), None, {}, unexpected)
            return ReconciliationResult("FRESH_START", tuple(reasons))

        if lots and symbol_short not in actual_open_pairs:
            # we think we hold lots but FT has no position on the symbol
            reasons.append(str(RiskEvent.POSITION_MISMATCH))
            return ReconciliationResult("SAFE_HOLD", tuple(reasons), plan, {}, unexpected)

        # ---- one-time entry gate: persisted virtual lots must match the real
        # position size; any divergence -> SAFE_HOLD (never auto-fix state)
        if actual_open_lots is not None:
            actual_n = int(actual_open_lots.get(symbol_short, 0))
            if len(lots) != actual_n:
                if str(RiskEvent.POSITION_MISMATCH) not in reasons:
                    reasons.append(str(RiskEvent.POSITION_MISMATCH))
                return ReconciliationResult("SAFE_HOLD", tuple(reasons), plan, {}, unexpected)

        if unexpected:
            reasons.append(str(RiskEvent.POSITION_MISMATCH))
            return ReconciliationResult("SAFE_HOLD", tuple(reasons), plan, {}, unexpected)

        return ReconciliationResult(
            "RESTORED", tuple(reasons), plan, lots, unexpected, filled_levels
        )
