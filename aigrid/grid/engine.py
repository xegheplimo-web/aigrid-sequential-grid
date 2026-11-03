"""Adaptive Grid Engine — virtual grid state machine over a validated plan.

Design (see ARCHITECTURE.md §5): Freqtrade's Strategy API cannot hold N
simultaneous resting limit orders per side, so the grid book lives HERE
(virtual) and a downstream Executor realizes actions SEQUENTIALLY (at most one
pending order per side). This module is pure state — no I/O, no orders.

Actions are emitted deterministically; client_order_ids are derived from
(plan_id, level, side, seq) so a restart can reconcile without duplicating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ..plans.schema import AIGridPlan, GridMode


class ActionKind(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    CLOSE_ALL = "CLOSE_ALL"
    CANCEL_PENDING = "CANCEL_PENDING"


@dataclass(frozen=True)
class GridAction:
    kind: ActionKind
    client_order_id: str
    level: int | None
    price: float
    qty: float
    reason: str


@dataclass
class GridLot:
    level: int
    entry_price: float
    qty: float
    entry_fee: float
    entry_candle_ms: int
    exit_fee_rate: float | None = None  # set by the exit path before _fill_exit


@dataclass
class GridState:
    plan_id: str
    grid_mode: GridMode
    levels: list[float]  # buy (or sell) level prices, ordered
    lots: dict[int, GridLot] = field(default_factory=dict)  # level -> lot
    filled_levels: set[int] = field(default_factory=set)  # levels ever filled (this plan)
    realized_pnl: float = 0.0
    total_fees: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    seq: int = 0  # monotonic action sequence (idempotency)
    level_seq: dict[int, int] = field(default_factory=dict)  # level -> cycles armed


class VirtualGridEngine:
    """Long-biased grid for RANGE_GRID/LONG_GRID; mirrored short for SHORT_GRID.

    SAFE_HOLD plans produce no actions (engine stays idle).
    """

    def __init__(
        self,
        plan: AIGridPlan,
        *,
        fee_rate: float | None = None,
        fee_rate_entry: float | None = None,
        fee_rate_exit: float | None = None,
        fee_rate_emergency: float | None = None,
        slippage_rate: float,
        leg_stop_loss_pct: float,
        max_inventory_notional: float,
    ):
        self.plan = plan
        base = 0.0 if fee_rate is None else fee_rate
        self.fee_rate = base  # legacy single-rate view (executor, backtest)
        self.fee_rate_entry = base if fee_rate_entry is None else fee_rate_entry
        self.fee_rate_exit = base if fee_rate_exit is None else fee_rate_exit
        self.fee_rate_emergency = base if fee_rate_emergency is None else fee_rate_emergency
        self.slippage_rate = slippage_rate
        self.leg_stop_loss_pct = leg_stop_loss_pct
        self.max_inventory_notional = max_inventory_notional
        self.state = self._init_state(plan)
        self._prev_close: float | None = None

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _init_state(plan: AIGridPlan) -> GridState:
        if plan.grid_mode == GridMode.SAFE_HOLD:
            levels: list[float] = []
        else:
            levels = plan.level_prices()
        return GridState(plan_id=plan.plan_id, grid_mode=plan.grid_mode, levels=levels)

    # ------------------------------------------------------------------ helpers
    @property
    def is_short(self) -> bool:
        return self.state.grid_mode == GridMode.SHORT_GRID

    def order_id(self, level: int, side: str, kind: str = "grid") -> str:
        """Deterministic id: same (plan, level, side, cycle) -> same id.

        level_seq[level] counts entry cycles armed for that level. The first
        cycle renders in the legacy 4-part form (parses as seq=1); later
        cycles append :{seq}, so a re-armed level never reuses an id and a
        retry of the same order always reproduces the identical id.
        """
        seq = self.state.level_seq.get(level, 0)
        if seq <= 1:
            return f"{self.plan.plan_id}:{kind}:{side}:{level}"
        return f"{self.plan.plan_id}:{kind}:{side}:{level}:{seq}"

    def inventory_notional(self, price: float) -> float:
        return sum(lot.qty * price for lot in self.state.lots.values())

    def unrealized_pnl(self, price: float) -> float:
        if self.is_short:
            return sum((lot.entry_price - price) * lot.qty for lot in self.state.lots.values())
        return sum((price - lot.entry_price) * lot.qty for lot in self.state.lots.values())

    def equity_delta(self, price: float) -> float:
        return self.state.realized_pnl + self.unrealized_pnl(price)

    # ------------------------------------------------------------------ main step
    def on_candle(
        self,
        candle_ms: int,
        open_: float,
        high: float,
        low: float,
        close: float,
        permit_entry: bool = True,
    ) -> list[GridAction]:
        """Process one CLOSED candle; returns the actions to realize (in order).

        Crossing semantics (no lookahead, single source of truth = this candle):
          long : level L is bought when price crosses DOWN through L
                 (prev_close > L >= low; first candle uses open as reference)
          short: level L is sold when price crosses UP through L
                 (prev_close < L <= high)
        Exits: TP needs the level to have been held since an EARLIER candle
        (pessimistic — no same-candle fill+TP); leg-SL may fire same-candle
        (protection trumps pessimism). Plan-SL closes everything immediately.
        """
        if self.state.halted or self.state.grid_mode == GridMode.SAFE_HOLD:
            return []
        prev_close = self._prev_close
        self._prev_close = close
        actions: list[GridAction] = []

        # 1) plan-level stop loss -> close everything (PROTECT path)
        sl = self.plan.stop_loss
        if sl is not None:
            if (not self.is_short and low <= sl) or (self.is_short and high >= sl):
                actions.extend(self._close_all(close, "PLAN_STOP_LOSS_TRIGGERED"))
                return actions

        # 2) exits first (TP per level, then leg SL)
        actions.extend(self._exits(candle_ms, high, low, close))

        # 3) entries (only if not paused and inventory budget allows)
        if permit_entry:
            actions.extend(self._entries(candle_ms, open_, high, low, close, prev_close))

        # 4) invalidation: price broke the grid boundary -> stop the grid
        lower, upper = self.plan.grid_lower, self.plan.grid_upper
        if (not self.is_short and close < lower) or (self.is_short and close > upper):
            actions.append(
                GridAction(
                    ActionKind.CANCEL_PENDING,
                    self.order_id(0, "CANCEL", "cancel"),
                    None,
                    close,
                    0.0,
                    "PRICE_BROKE_GRID_BOUNDARY",
                )
            )
            self.state.halted = True
            self.state.halt_reason = "PRICE_BROKE_GRID_BOUNDARY"
        return actions

    # ------------------------------------------------------------------ exits
    def _exits(self, candle_ms: int, high: float, low: float, close: float) -> list[GridAction]:
        actions = []
        for level, lot in list(self.state.lots.items()):
            fresh = lot.entry_candle_ms == candle_ms
            if self.is_short:
                tp_price = lot.entry_price * (1 - self.plan.grid_spacing)
                sl_price = lot.entry_price * (1 + self.leg_stop_loss_pct)
                hit_tp = low <= tp_price  # price fell to the cover target
                hit_sl = high >= sl_price  # price rose to the stop
                exit_price = tp_price if hit_tp else (sl_price if hit_sl else None)
            else:
                tp_price = lot.entry_price * (1 + self.plan.grid_spacing)
                sl_price = lot.entry_price * (1 - self.leg_stop_loss_pct)
                hit_tp = high >= tp_price  # price rose to the profit target
                hit_sl = low <= sl_price  # price fell to the stop
                exit_price = tp_price if hit_tp else (sl_price if hit_sl else None)
            if exit_price is None:
                continue
            if hit_sl:
                # pessimistic: when both TP and SL are inside one candle, SL wins
                exit_price = sl_price
                hit_tp = False
            elif hit_tp and fresh:
                continue  # pessimistic: no fill+TP inside the same candle
            reason = "LEG_TAKE_PROFIT" if hit_tp else "LEG_STOP_LOSS"
            # closing side: SELL closes a long, BUY covers a short
            kind = ActionKind.SELL if not self.is_short else ActionKind.BUY
            actions.append(
                GridAction(kind, self.order_id(level, "EXIT"), level, exit_price, lot.qty, reason)
            )
            lot.exit_fee_rate = self.fee_rate_exit if hit_tp else self.fee_rate_emergency
            self._fill_exit(level, exit_price, lot)
        return actions

    def _fill_exit(self, level: int, price: float, lot: GridLot) -> None:
        # lot.exit_fee_rate is set by the caller (TP -> maker, emergency exits
        # -> taker); signature stays stable for RecordingGridEngine (backtest)
        exec_price = (
            price * (1 - self.slippage_rate)
            if not self.is_short
            else price * (1 + self.slippage_rate)
        )
        gross = lot.qty * exec_price
        rate = self.fee_rate_exit if lot.exit_fee_rate is None else lot.exit_fee_rate
        fee = gross * rate
        self.state.total_fees += fee
        if self.is_short:
            pnl = (lot.entry_price - exec_price) * lot.qty - fee - lot.entry_fee
        else:
            pnl = (exec_price - lot.entry_price) * lot.qty - fee - lot.entry_fee
        self.state.realized_pnl += pnl
        del self.state.lots[level]

    # ------------------------------------------------------------------ entries
    def _entries(
        self,
        candle_ms: int,
        open_: float,
        high: float,
        low: float,
        close: float,
        prev_close: float | None,
    ) -> list[GridAction]:
        actions = []
        # per-level NOTIONAL: ưu tiên plan.level_notional (từ CapitalManager:
        # level margin × leverage); fallback chia đều max_position_size.
        level_notional = (
            self.plan.level_notional
            if self.plan.level_notional > 0
            else self.plan.max_position_size / max(1, len(self.state.levels))
        )
        ref = prev_close if prev_close is not None else open_
        for level, level_price in enumerate(self.state.levels):
            if level in self.state.lots:
                continue  # duplicate prevention: one open lot per level
            if self.is_short:
                touched = ref < level_price <= high  # crossed UP through level
            else:
                touched = ref > level_price >= low  # crossed DOWN through level
            if not touched:
                continue
            projected = self.inventory_notional(level_price) + level_notional
            if projected > self.plan.max_position_size + 1e-9:
                continue  # budget cap: never exceed plan position size
            qty = round(level_notional / level_price, 8)
            if qty <= 0:
                continue
            self.state.level_seq[level] = self.state.level_seq.get(level, 0) + 1
            actions.append(
                GridAction(
                    ActionKind.BUY if not self.is_short else ActionKind.SELL,
                    self.order_id(level, "ENTRY"),
                    level,
                    level_price,
                    qty,
                    "GRID_LEVEL_CROSSED",
                )
            )
            self._fill_entry(level, level_price, qty, candle_ms)
        return actions

    def _fill_entry(self, level: int, price: float, qty: float, candle_ms: int) -> None:
        exec_price = (
            price * (1 + self.slippage_rate)
            if not self.is_short
            else price * (1 - self.slippage_rate)
        )
        gross = qty * exec_price
        fee = gross * self.fee_rate_entry
        self.state.total_fees += fee
        self.state.lots[level] = GridLot(level, exec_price, qty, fee, candle_ms)
        self.state.filled_levels.add(level)

    # ------------------------------------------------------------------ close all
    def _close_all(self, price: float, reason: str) -> list[GridAction]:
        actions = []
        for level, lot in list(self.state.lots.items()):
            actions.append(
                GridAction(
                    ActionKind.SELL, self.order_id(level, "CLOSEALL"), level, price, lot.qty, reason
                )
            )
            lot.exit_fee_rate = self.fee_rate_emergency
            self._fill_exit(level, price, lot)
        self.state.halted = True
        self.state.halt_reason = reason
        return actions

    # ------------------------------------------------------------------ reconciliation
    def apply_external_fill(
        self, client_order_id: str, price: float, qty: float, candle_ms: int = 0
    ) -> bool:
        """Record an externally-confirmed fill (executor feedback).

        Returns True if the id was known (idempotent replay OK), False if unknown.
        """
        parts = client_order_id.split(":")
        if len(parts) == 4:  # legacy id -> seq = 1
            _, kind, side, level_s = parts
            seq = 1
        elif len(parts) == 5:
            _, kind, side, level_s, seq_s = parts
            try:
                seq = int(seq_s)
            except ValueError:
                return False
        else:
            return False
        try:
            level = int(level_s)
        except ValueError:
            return False
        if kind == "grid" and side == "ENTRY" and level not in self.state.lots:
            self.state.level_seq[level] = max(self.state.level_seq.get(level, 0), seq)
            self._fill_entry(level, price, qty, candle_ms)
            return True
        if (
            kind in ("grid", "closeall")
            and side in ("EXIT", "CLOSEALL")
            and level in self.state.lots
        ):
            lot = self.state.lots[level]
            lot.exit_fee_rate = self.fee_rate_exit if side == "EXIT" else self.fee_rate_emergency
            self._fill_exit(level, price, lot)
            return True
        return True  # known id, already applied (duplicate feedback)

    def to_dict(self, price: float) -> dict:
        st = self.state
        return {
            "plan_id": st.plan_id,
            "grid_mode": str(st.grid_mode),
            "levels": st.levels,
            "open_lots": {
                str(k): {"entry": v.entry_price, "qty": v.qty} for k, v in st.lots.items()
            },
            "filled_levels": sorted(st.filled_levels),
            "level_seq": {str(k): v for k, v in sorted(st.level_seq.items())},
            "realized_pnl": round(st.realized_pnl, 6),
            "total_fees": round(st.total_fees, 6),
            "unrealized_pnl": round(self.unrealized_pnl(price), 6),
            "halted": st.halted,
            "halt_reason": st.halt_reason,
        }
