"""Persistent state — atomic JSON snapshots + SQLite event log + idempotency.

Design:
- `StateStore.save_state()` writes JSON via temp-file + os.replace (atomic on
  Windows/POSIX). A crash mid-write leaves the previous snapshot intact.
- `EventLog` appends to SQLite (WAL) — crash-safe, queryable.
- Idempotency: `seen(client_order_id)` backed by a SQLite table; duplicate
  execution feedback is dropped.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path


class StateStore:
    def __init__(self, state_dir: str | Path):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ atomic JSON
    def save_state(self, name: str, payload: dict) -> None:
        target = self.state_dir / f"{name}.json"
        fd, tmp = tempfile.mkstemp(dir=str(self.state_dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=1, default=str)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)  # atomic on same volume
        finally:
            Path(tmp).unlink(missing_ok=True)

    def load_state(self, name: str) -> dict | None:
        target = self.state_dir / f"{name}.json"
        if not target.exists():
            return None
        try:
            with target.open(encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None  # corrupt snapshot -> treated as absent (fail-closed)

    def clear_state(self, name: str) -> None:
        (self.state_dir / f"{name}.json").unlink(missing_ok=True)


class EventLog:
    """Append-only SQLite event log (WAL mode)."""

    def __init__(self, state_dir: str | Path, db_name: str = "aigrid_events.sqlite"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.state_dir / db_name, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_ms INTEGER NOT NULL,
                kind TEXT NOT NULL,
                symbol TEXT,
                payload TEXT
            )""")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS idempotency (
                client_order_id TEXT PRIMARY KEY,
                first_seen_ms INTEGER NOT NULL,
                payload TEXT
            )""")
        self.conn.commit()

    def log(self, kind: str, symbol: str = "", payload: dict | None = None) -> None:
        self.conn.execute(
            "INSERT INTO events (ts_ms, kind, symbol, payload) VALUES (?,?,?,?)",
            (int(time.time() * 1000), kind, symbol, json.dumps(payload or {}, default=str)),
        )
        self.conn.commit()

    def recent(self, kind: str | None = None, limit: int = 100) -> list[dict]:
        if kind:
            rows = self.conn.execute(
                "SELECT ts_ms, kind, symbol, payload FROM events WHERE kind=? "
                "ORDER BY id DESC LIMIT ?",
                (kind, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT ts_ms, kind, symbol, payload FROM events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {
                "ts_ms": r[0],
                "kind": r[1],
                "symbol": r[2],
                "payload": json.loads(r[3]) if r[3] else {},
            }
            for r in rows
        ]

    # ------------------------------------------------------------------ idempotency
    def mark_seen(self, client_order_id: str, payload: dict | None = None) -> bool:
        """Returns True if newly seen; False if it was already recorded."""
        try:
            self.conn.execute(
                "INSERT INTO idempotency (client_order_id, first_seen_ms, payload) VALUES (?,?,?)",
                (client_order_id, int(time.time() * 1000), json.dumps(payload or {}, default=str)),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def seen(self, client_order_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM idempotency WHERE client_order_id=?", (client_order_id,)
        ).fetchone()
        return row is not None

    def close(self) -> None:
        self.conn.close()
