"""AIGrid shadow service loop (T-206a).

A long-running supervisor around the orchestrator:
- on startup, ONE preflight when the caller hands over both configs —
  drift OR unverifiable blocks the loop entirely (publish must never run
  on an unverified baseline) and run() returns 2
- every tick writes an atomic heartbeat (tmp + os.replace via StateStore)
  to <state_dir>/heartbeat.json: ts_ms, cycle, pid, preflight_ok, stopped
- soft stop via stop() / SIGINT / SIGTERM: a final heartbeat with
  stopped=true is written, then the process exits cleanly — no
  sys.exit() mid-cycle
- restart keeps state: aigrid_state/ is never wiped; the previous
  heartbeat is read back and resumed_from_cycle is logged

This round is heartbeat-only by design: the loop NEVER writes
ft_integration.json — it does not call orchestrator.persist() or
_write_ft_integration(), and it never dispatches entries or publishes
plans. With settings.enabled=false (the aigrid.toml default) it runs in
SHADOW_MODE: loop + heartbeat, no entry, no publish.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .orchestrator import AIGridOrchestrator
from .preflight import (
    APPROVED_CONFIG_ENV,
    DEFAULT_API_URL,
    FT_API_PASS_ENV,
    FT_API_URL_ENV,
    FT_API_USER_ENV,
    fetch_runtime_config,
)
from .settings import load_settings
from .state.store import StateStore

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "configs_aigrid/aigrid.toml"
DEFAULT_STATE_DIR = "user_data/aigrid_state"


class AIGridService:
    """Heartbeat service wrapping an AIGridOrchestrator (or a duck-typed
    stand-in exposing .settings.state_dir, .eventlog and .run_preflight).

    clock: callable returning seconds (time.time-compatible); injected in
    tests. on_cycle: optional callback invoked after each heartbeat tick.
    """

    def __init__(
        self,
        orchestrator: Any,
        interval_s: float = 30.0,
        *,
        clock: Callable[[], float] | None = None,
        on_cycle: Callable[[int], None] | None = None,
    ):
        self.orchestrator = orchestrator
        self.interval_s = float(interval_s)
        self._clock = clock or time.time
        self.on_cycle = on_cycle
        self._stop = threading.Event()
        state_dir = (
            getattr(getattr(orchestrator, "settings", None), "state_dir", None) or DEFAULT_STATE_DIR
        )
        self._store = StateStore(state_dir)

    # ------------------------------------------------------------------ control
    def stop(self) -> None:
        """Request a soft stop: the current tick finishes, a final
        stopped=true heartbeat is written, run() returns 0."""
        self._stop.set()

    # ------------------------------------------------------------------ internals
    def _log(self, kind: str, payload: dict) -> None:
        eventlog = getattr(self.orchestrator, "eventlog", None)
        if eventlog is not None:
            eventlog.log(kind, "", payload)
        logger.info("%s %s", kind, payload)

    def _write_heartbeat(self, cycle: int, *, stopped: bool, preflight_ok: bool) -> None:
        self._store.save_state(
            "heartbeat",
            {
                "ts_ms": int(self._clock() * 1000),
                "cycle": cycle,
                "pid": os.getpid(),
                "preflight_ok": bool(preflight_ok),
                "stopped": bool(stopped),
            },
        )

    # ------------------------------------------------------------------ main loop
    def run(self, runtime_cfg: dict | None = None, approved_cfg: dict | None = None) -> int:
        """0 = clean stop; 2 = preflight blocked (drift or unverifiable)."""
        old = self._store.load_state("heartbeat") or {}
        cycle = int(old.get("cycle") or 0)
        if cycle:
            self._log("SERVICE_RESUMED", {"resumed_from_cycle": cycle})
        settings = getattr(self.orchestrator, "settings", None)
        if not getattr(settings, "enabled", False):
            self._log(
                "SHADOW_MODE",
                {"note": "enabled=false — heartbeat loop only; no entries, no publish"},
            )
        preflight_ok = False
        if runtime_cfg is not None and approved_cfg is not None:
            check = self.orchestrator.run_preflight(runtime_cfg, approved_cfg)
            preflight_ok = check.ok
            if check.blocked:
                self._log(
                    "SERVICE_PREFLIGHT_BLOCKED",
                    {"drift": list(check.drift), "unverifiable": list(check.unverifiable)},
                )
                self._write_heartbeat(cycle, stopped=True, preflight_ok=False)
                return 2
        while not self._stop.is_set():
            cycle += 1
            self._write_heartbeat(cycle, stopped=False, preflight_ok=preflight_ok)
            if self.on_cycle is not None:
                self.on_cycle(cycle)
            self._stop.wait(self.interval_s)  # interruptible — stop() < 2s
        self._write_heartbeat(cycle, stopped=True, preflight_ok=preflight_ok)
        return 0


# ------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aigrid.service",
        description="AIGrid shadow service: startup preflight + heartbeat "
        "loop. Never publishes ft_integration.json.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"AIGrid TOML path (default {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--interval", type=float, default=30.0, help="seconds between heartbeat ticks (default 30)"
    )
    parser.add_argument(
        "--approved",
        default=os.environ.get(APPROVED_CONFIG_ENV),
        help=f"approved config JSON for startup preflight "
        f"(or env {APPROVED_CONFIG_ENV}); omitted -> no preflight",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get(FT_API_URL_ENV, DEFAULT_API_URL),
        help=f"Freqtrade REST base URL (env {FT_API_URL_ENV}, default {DEFAULT_API_URL})",
    )
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    orch = AIGridOrchestrator(settings)
    svc = AIGridService(orch, interval_s=args.interval)

    def _handle_signal(signum, frame) -> None:
        svc.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    runtime_cfg = approved_cfg = None
    if args.approved:
        try:
            runtime_cfg = fetch_runtime_config(
                args.api_url,
                username=os.environ.get(FT_API_USER_ENV),
                password=os.environ.get(FT_API_PASS_ENV),
            )
            approved_cfg = json.loads(Path(args.approved).read_text(encoding="utf-8"))
            if not isinstance(approved_cfg, dict):
                raise ValueError("approved config is not a JSON object")
        except Exception as exc:
            print(f"aigrid.service: cannot load preflight configs: {exc}", file=sys.stderr)
            orch.eventlog.close()
            return 2
    try:
        return svc.run(runtime_cfg, approved_cfg)
    finally:
        # eventlog only — never orch.close(): persist() would write
        # ft_integration.json, which this round must not produce.
        orch.eventlog.close()


if __name__ == "__main__":
    raise SystemExit(main())
