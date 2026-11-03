"""Symbol Ownership Registry — one owner per symbol at a time.

States: FREE / TREND_OWNED / GRID_OWNED / TRANSITIONING / RISK_LOCKED.
Transitions are validated; conflicting strategies can never control the same
symbol simultaneously (spec section X).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .events import RiskEvent


class OwnerState(StrEnum):
    FREE = "FREE"
    TREND_OWNED = "TREND_OWNED"
    GRID_OWNED = "GRID_OWNED"
    TRANSITIONING = "TRANSITIONING"
    RISK_LOCKED = "RISK_LOCKED"


# Legal transitions: from -> {to: guard}
# TRANSITIONING must pass through an explicit complete/abort step.
LEGAL_TRANSITIONS: dict[OwnerState, set[OwnerState]] = {
    OwnerState.FREE: {OwnerState.TREND_OWNED, OwnerState.GRID_OWNED, OwnerState.RISK_LOCKED},
    OwnerState.TREND_OWNED: {OwnerState.TRANSITIONING, OwnerState.RISK_LOCKED, OwnerState.FREE},
    OwnerState.GRID_OWNED: {OwnerState.TRANSITIONING, OwnerState.RISK_LOCKED, OwnerState.FREE},
    OwnerState.TRANSITIONING: {
        OwnerState.TREND_OWNED,
        OwnerState.GRID_OWNED,
        OwnerState.FREE,
        OwnerState.RISK_LOCKED,
    },
    OwnerState.RISK_LOCKED: {OwnerState.FREE},  # only the Risk Engine unlocks
}


@dataclass
class SymbolLease:
    symbol: str
    state: OwnerState = OwnerState.FREE
    owner_id: str = ""  # e.g. "trend:ZaratustraV19" / "grid:plan-xxx"
    reason: str = ""
    events: list[tuple[str, str, str]] = field(default_factory=list)  # (ts, from, to)


class OwnershipRegistry:
    """In-memory registry + optional JSON persistence hook (Phase 5 wires it)."""

    def __init__(self):
        self._leases: dict[str, SymbolLease] = {}

    # ------------------------------------------------------------------ queries
    def state_of(self, symbol: str) -> OwnerState:
        return self._lease(symbol).state

    def owner_of(self, symbol: str) -> str:
        return self._lease(symbol).owner_id

    def can_open(self, symbol: str, strategy: str) -> tuple[bool, str]:
        """Can `strategy` ('trend'|'grid') open a NEW position on symbol?"""
        lease = self._lease(symbol)
        if lease.state == OwnerState.RISK_LOCKED:
            return False, str(RiskEvent.OWNER_STOP)
        if lease.state == OwnerState.FREE:
            return True, ""
        owner_kind = lease.owner_id.split(":", 1)[0]
        if owner_kind == strategy:
            return True, ""  # same strategy family may manage its own
        return False, str(RiskEvent.OWNERSHIP_CONFLICT)

    # ------------------------------------------------------------------ transitions
    def acquire(self, symbol: str, strategy: str, owner_id: str, ts: str = "") -> tuple[bool, str]:
        lease = self._lease(symbol)
        if lease.state not in (OwnerState.FREE,):
            if lease.owner_id == owner_id:
                return True, ""  # re-acquire by same owner is a no-op
            return False, str(RiskEvent.OWNERSHIP_CONFLICT)
        target = OwnerState.TREND_OWNED if strategy == "trend" else OwnerState.GRID_OWNED
        return self._transition(symbol, target, owner_id, ts)

    def begin_transition(self, symbol: str, ts: str = "") -> tuple[bool, str]:
        lease = self._lease(symbol)
        if lease.state not in (OwnerState.TREND_OWNED, OwnerState.GRID_OWNED):
            return False, f"TRANSITION_NOT_ALLOWED_FROM_{lease.state}"
        return self._transition(symbol, OwnerState.TRANSITIONING, lease.owner_id, ts)

    def complete_transition(
        self, symbol: str, strategy: str, owner_id: str, ts: str = ""
    ) -> tuple[bool, str]:
        lease = self._lease(symbol)
        if lease.state != OwnerState.TRANSITIONING:
            return False, "NOT_TRANSITIONING"
        target = OwnerState.TREND_OWNED if strategy == "trend" else OwnerState.GRID_OWNED
        return self._transition(symbol, target, owner_id, ts)

    def release(self, symbol: str, owner_id: str, ts: str = "") -> tuple[bool, str]:
        lease = self._lease(symbol)
        if lease.state == OwnerState.FREE:
            return True, ""
        if lease.state == OwnerState.RISK_LOCKED:
            return False, str(RiskEvent.OWNER_STOP)
        if lease.owner_id != owner_id:
            return False, str(RiskEvent.OWNERSHIP_CONFLICT)
        return self._transition(symbol, OwnerState.FREE, owner_id, ts)

    def risk_lock(self, symbol: str, reason: str, ts: str = "") -> tuple[bool, str]:
        """ONLY the Risk Engine may call this (emergency)."""
        return self._transition(symbol, OwnerState.RISK_LOCKED, "risk-engine", ts, reason=reason)

    def risk_unlock(self, symbol: str, ts: str = "") -> tuple[bool, str]:
        lease = self._lease(symbol)
        if lease.state != OwnerState.RISK_LOCKED:
            return True, ""
        return self._transition(symbol, OwnerState.FREE, "risk-engine", ts)

    # ------------------------------------------------------------------ internals
    def _transition(
        self, symbol: str, target: OwnerState, owner_id: str, ts: str, reason: str = ""
    ) -> tuple[bool, str]:
        lease = self._lease(symbol)
        if target not in LEGAL_TRANSITIONS[lease.state]:
            return False, f"ILLEGAL_TRANSITION_{lease.state}_TO_{target}"
        old = lease.state
        lease.state = target
        lease.owner_id = owner_id if target != OwnerState.FREE else ""
        lease.reason = reason
        lease.events.append((ts or "", str(old), str(target)))
        return True, ""

    def _lease(self, symbol: str) -> SymbolLease:
        lease = self._leases.get(symbol)
        if lease is None:
            lease = SymbolLease(symbol)
            self._leases[symbol] = lease
        return lease

    # ------------------------------------------------------------------ export
    def to_dict(self) -> dict:
        return {
            sym: {"state": str(l.state), "owner": l.owner_id, "reason": l.reason}
            for sym, l in self._leases.items()
        }

    def load_dict(self, data: dict | None) -> None:
        """Restore leases persisted via to_dict(); unknown states fail closed."""
        self._leases = {}
        for sym, d in (data or {}).items():
            if not isinstance(d, dict):
                d = {}
            try:
                state = OwnerState(d.get("state", "FREE"))
            except (ValueError, AttributeError):
                state = OwnerState.RISK_LOCKED  # unknown state -> fail-closed
            if state == OwnerState.FREE:
                continue
            lease = self._lease(sym)
            lease.state = state
            lease.owner_id = str(d.get("owner", "") or "")
            lease.reason = str(d.get("reason", "") or "")
