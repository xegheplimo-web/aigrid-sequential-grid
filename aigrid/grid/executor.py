"""Grid executors — realize VirtualGridEngine actions.

- ShadowExecutor: records virtual fills (shadow mode + backtesting). No orders.
- FreqtradeCommandExecutor: writes deterministic command files that the
  Freqtrade integration strategy consumes in dry-run (Phase 6/7). No live path.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .engine import ActionKind, GridAction, VirtualGridEngine


@dataclass
class FillRecord:
    client_order_id: str
    side: str
    price: float
    qty: float
    fee: float
    timestamp_ms: int
    reason: str


@dataclass
class ExecutionStats:
    fills: list[FillRecord] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    orders_submitted: int = 0  # real orders sent to any venue: ALWAYS 0 in shadow


class ShadowExecutor:
    """Simulates maker fills for emitted actions at the action price.

    Pessimistic: entry fills pay slippage+fee; exit fills pay fee. Records
    everything for the dashboard/backtest. orders_submitted stays 0 by design.
    """

    def __init__(self, engine: VirtualGridEngine):
        self.engine = engine
        self.stats = ExecutionStats()

    def execute(self, actions: list[GridAction], candle_ms: int) -> list[FillRecord]:
        fills = []
        for action in actions:
            if action.kind in (ActionKind.CLOSE_ALL, ActionKind.CANCEL_PENDING):
                continue
            fee = action.qty * action.price * self.engine.fee_rate
            fill = FillRecord(
                action.client_order_id,
                action.kind.value,
                action.price,
                action.qty,
                fee,
                candle_ms,
                action.reason,
            )
            fills.append(fill)
            self.stats.fills.append(fill)
        # engine state was already updated inside the engine (single source of truth)
        return fills


class FreqtradeCommandExecutor:
    """Dry-run integration: emits command records for the FT strategy to consume.

    Commands are appended to a JSONL file (atomic append + fsync). The strategy
    side polls the file and executes via its own callbacks. Idempotency: the
    strategy tracks consumed client_order_ids; duplicates are ignored.

    NO live-order code path exists in this class: it never talks to an exchange.
    """

    def __init__(self, engine: VirtualGridEngine, command_dir: str | Path):
        self.engine = engine
        self.command_dir = Path(command_dir)
        self.command_dir.mkdir(parents=True, exist_ok=True)
        self.commands_path = self.command_dir / "grid_commands.jsonl"
        self.stats = ExecutionStats()

    def execute(self, actions: list[GridAction], candle_ms: int) -> int:
        written = 0
        for action in actions:
            record = {
                "ts_ms": int(time.time() * 1000),
                "candle_ms": candle_ms,
                "client_order_id": action.client_order_id,
                "kind": action.kind.value,
                "level": action.level,
                "price": action.price,
                "qty": action.qty,
                "reason": action.reason,
                "plan_id": self.engine.plan.plan_id,
                "symbol": self.engine.plan.symbol,
                "dry_run_only": True,
            }
            self._append_jsonl(record)
            written += 1
        self.stats.orders_submitted += written
        return written

    def _append_jsonl(self, record: dict) -> None:
        line = json.dumps(record, separators=(",", ":")) + "\n"
        fd, tmp = tempfile.mkstemp(dir=str(self.command_dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            with open(self.commands_path, "a", encoding="utf-8") as out:
                out.write(line)
        finally:
            Path(tmp).unlink(missing_ok=True)
