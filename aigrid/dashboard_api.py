"""AIGrid dashboard API — read-only panel for AI RECOMMENDATION vs ACTUAL EXECUTION.

Reads ONLY the orchestrator's state dir (snapshot JSON + SQLite event log).
No write endpoints, no exchange calls, no Freqtrade credentials.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/aigrid", tags=["aigrid"])

_STATE_DIR: Path | None = None


def init_dashboard_api(state_dir: str | Path) -> None:
    global _STATE_DIR
    _STATE_DIR = Path(state_dir)


def _require_state_dir() -> Path:
    if _STATE_DIR is None:
        raise HTTPException(status_code=503, detail="aigrid state dir not configured")
    return _STATE_DIR


def _read_json(name: str) -> dict | None:
    p = _require_state_dir() / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


@router.get("/status")
def status() -> dict:
    """Full AI vs execution panel payload."""
    snap = _read_json("aigrid_snapshot.json")
    plan = _read_json("current_plan.json")
    grid = _read_json("grid_state.json")
    return {
        "ts_ms": int(time.time() * 1000),
        "snapshot": snap,
        "ai_recommendation": {
            "plan": plan,
            "last_analysis": (snap or {}).get("ai_reasoning", ""),
            "ai_status": (snap or {}).get("ai_status", "OFF"),
        },
        "actual_execution": {
            "grid_state": grid,
            "grid_mode": (snap or {}).get("grid_mode", "UNKNOWN"),
            "paused": (snap or {}).get("paused", True),
            "pause_reason": (snap or {}).get("pause_reason", ""),
        },
        "safety": {"live_orders_allowed": False, "mode": (snap or {}).get("mode", "off")},
    }


@router.get("/events")
def events(limit: int = 50) -> dict:
    limit = max(1, min(limit, 500))
    db = _require_state_dir() / "aigrid_events.sqlite"
    if not db.exists():
        return {"events": []}
    import sqlite3

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)  # read-only
    try:
        rows = conn.execute(
            "SELECT ts_ms, kind, symbol, payload FROM events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    finally:
        conn.close()
    return {
        "events": [
            {
                "ts_ms": r[0],
                "kind": r[1],
                "symbol": r[2],
                "payload": json.loads(r[3]) if r[3] else {},
            }
            for r in rows
        ]
    }


@router.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "ts_ms": int(time.time() * 1000)}
