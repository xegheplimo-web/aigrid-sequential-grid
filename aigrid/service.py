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

In non-dryrun modes the loop is heartbeat-only by design: it NEVER
writes ft_integration.json itself. When ``settings.mode == "dryrun"``
and a DeterministicCycle is injected, each tick additionally runs the
R14 deterministic chain — closed-candle detect -> process_candle ->
REGIME check -> deterministic plan -> guarded atomic publish (tmp +
os.replace). That path has NO LLM call and no non-local network call
beyond the read-only Freqtrade REST GETs; a blocked guard writes
nothing and emits PUBLISH_BLOCKED. With settings.enabled=false (the
aigrid.toml default) it runs in SHADOW_MODE regardless: loop +
heartbeat, no entry, no publish.
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

from .config import load_settings
from .data.binance_adapter import TF_MS
from .exchange_info import ExchangeInfo
from .orchestrator import AIGridOrchestrator
from .preflight import (
    APPROVED_CONFIG_ENV,
    DEFAULT_API_URL,
    FT_API_PASS_ENV,
    FT_API_URL_ENV,
    FT_API_USER_ENV,
    fetch_runtime_config,
)
from .service_lock import ServiceLock
from .state.store import StateStore

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "configs_aigrid/aigrid.toml"
DEFAULT_STATE_DIR = "user_data/aigrid_state"


class AIGridService:
    """Heartbeat service wrapping an AIGridOrchestrator (or a duck-typed
    stand-in exposing .settings.state_dir, .eventlog and .run_preflight).

    clock: callable returning seconds (time.time-compatible); injected in
    tests. on_cycle: optional callback invoked after each heartbeat tick.
    cycle: optional DeterministicCycle — driven once per tick ONLY when
    settings.mode == "dryrun"; ignored in every other mode.
    """

    def __init__(
        self,
        orchestrator: Any,
        interval_s: float = 30.0,
        *,
        clock: Callable[[], float] | None = None,
        on_cycle: Callable[[int], None] | None = None,
        cycle: Any = None,
    ):
        self.orchestrator = orchestrator
        self.interval_s = float(interval_s)
        self._clock = clock or time.time
        self.on_cycle = on_cycle
        self._cycle = cycle
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
        processed = (
            dict(getattr(self._cycle, "processed_ts", self._cycle.seen_ts))
            if self._cycle is not None
            else {}
        )
        self._store.save_state(
            "heartbeat",
            {
                "ts_ms": int(self._clock() * 1000),
                "cycle": cycle,
                "pid": os.getpid(),
                "preflight_ok": bool(preflight_ok),
                "stopped": bool(stopped),
                # newest closed-candle open_ms per pair loaded this pass
                "candle_ts_ms": processed,
                "watchlist": list(processed),
            },
        )

    # Controller unlock needs >=3 closed candles in the same regime; run the
    # service continuously for >=15 minutes before the first publish.
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
                    {
                        "drift": list(check.drift),
                        "unverifiable": list(check.unverifiable),
                    },
                )
                self._write_heartbeat(cycle, stopped=True, preflight_ok=False)
                return 2
        # R14: the deterministic publish cycle runs ONLY in dryrun mode.
        dryrun_cycle = self._cycle if getattr(settings, "mode", "off") == "dryrun" else None
        if dryrun_cycle is not None:
            # D-102 (1): reconcile persisted state before ANY publish.
            try:
                dryrun_cycle.startup_reconcile()
            except Exception as exc:
                self._log(
                    "STARTUP_RECONCILIATION_ERROR",
                    {"error": f"{type(exc).__name__}:{exc}"},
                )
        while not self._stop.is_set():
            cycle += 1
            if dryrun_cycle is not None:
                try:
                    dryrun_cycle.tick(cycle)
                except Exception as exc:
                    self._log("CYCLE_TICK_ERROR", {"error": f"{type(exc).__name__}:{exc}"})
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
        description="AIGrid service: startup preflight + heartbeat loop; "
        "in dryrun mode also runs the deterministic "
        "closed-candle -> plan -> guarded atomic publish cycle.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"AIGrid TOML path (default {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="seconds between heartbeat ticks (default 30)",
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
    parser.add_argument("--lock-file", help="override the state directory service pidfile")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    lock = ServiceLock(
        args.lock_file or Path(settings.state_dir or DEFAULT_STATE_DIR) / "aigrid-service.lock"
    )
    result = lock.acquire()
    if result:
        return result
    try:
        return _run_main(args, settings, lock)
    finally:
        try:
            lock.release()
        finally:
            # Never orch.close(): persist() would write ft_integration.json.
            if lock.eventlog is not None:
                lock.eventlog.close()


def _run_main(args, settings, lock: ServiceLock) -> int:
    exchange_info = None
    if settings.exchange_info_path:
        try:
            exchange_info = ExchangeInfo(settings.exchange_info_path)
            if not exchange_info._rules:
                raise ValueError("snapshot has no tradable USDT perpetual symbols")
            for pair in settings.watchlist:
                rules = exchange_info.rules_for(pair)
                if rules.tick_size <= 0 or rules.step_size <= 0:
                    raise ValueError(f"invalid exchange filters for {pair}")
        except Exception as exc:
            logger.warning("EXCHANGE_INFO_UNAVAILABLE: %s", exc)
            exchange_info = None
    orch = AIGridOrchestrator(settings, exchange_info=exchange_info)
    lock.eventlog = getattr(orch, "eventlog", None)
    lock.emit("SERVICE_LOCK_ACQUIRED")
    cycle = None
    if settings.mode == "dryrun":
        # R14 deterministic publish chain — closed candle -> regime ->
        # plan -> guarded atomic publish. No LLM; the only network is the
        # read-only local Freqtrade REST (facts for the guards).
        from .publish import DeterministicCycle, FtRestClient, PlanPublisher

        client = FtRestClient(
            args.api_url,
            username=os.environ.get(FT_API_USER_ENV),
            password=os.environ.get(FT_API_PASS_ENV),
        )
        logger.info("FT REST base URL: %s", args.api_url)
        runtime = client.show_config()
        if not runtime:
            print(
                "aigrid.service: FT REST show_config unavailable (auth/URL?) — refusing to start",
                file=sys.stderr,
            )
            return 2
        watchlist = list(settings.watchlist or settings.symbols)
        runtime_watchlist = runtime.get("pair_whitelist")
        if not isinstance(runtime_watchlist, (list, tuple)):
            exchange = runtime.get("exchange")
            runtime_watchlist = (
                exchange.get("pair_whitelist") if isinstance(exchange, dict) else None
            )
        if not isinstance(runtime_watchlist, (list, tuple)):
            runtime_watchlist = client.whitelist()
        missing = (
            sorted(set(watchlist) - set(runtime_watchlist))
            if isinstance(runtime_watchlist, (list, tuple))
            and all(isinstance(pair, str) for pair in runtime_watchlist)
            else []
        )
        if missing:
            orch.eventlog.log(
                "PREFLIGHT_WARNING",
                "",
                {
                    "reason": "WATCHLIST_MISMATCH",
                    "settings": watchlist,
                    "runtime": list(runtime_watchlist),
                    "missing": missing,
                },
            )
        publisher = PlanPublisher(
            orch.store,
            orch.eventlog,
            validator=orch.validator,
            ownership=orch.ownership,
            settings=settings,
            price_tick_of=orch._price_tick,
        )
        timeframe_ms = TF_MS[settings.timeframe]

        def candle_loader(pair: str):
            # pair_candles already parses REST JSON through _candles_to_df.
            # An explicit loader must fail closed: never read adapter files.
            try:
                frame = client.pair_candles(
                    pair, settings.timeframe, publisher.min_closed_candles + 30
                )
                if frame is None:
                    raise ValueError("REST candles unavailable or invalid")
                now_ms = int(publisher._clock() * 1000)
                open_ms = frame.index.as_unit("ns").asi8 // 1_000_000
                frame = frame.loc[open_ms + timeframe_ms <= now_ms]
                if len(frame) < publisher.min_closed_candles:
                    raise ValueError(f"insufficient closed candles: {len(frame)}")
            except Exception as exc:
                logger.debug("REST candle loader failed for %s: %s", pair, exc)
                # The cycle returns immediately on None; record the failure here.
                orch.eventlog.log("DATA_UNAVAILABLE", pair, {"error": str(exc)})
                return None
            logger.debug(
                "REST_CANDLES pair=%s candle_ts_ms=%d closed_candles=%d",
                pair,
                int(frame.index[-1].value // 1_000_000),
                len(frame),
            )
            return frame

        cycle = DeterministicCycle(
            orch,
            publisher,
            candle_loader=candle_loader,
            facts_provider=client.facts,
            timeframe_s=timeframe_ms // 1000,
        )
    svc = AIGridService(orch, interval_s=args.interval, cycle=cycle)

    def _handle_signal(signum, frame) -> None:
        del signum, frame
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
                raise TypeError("approved config is not a JSON object")
        except Exception as exc:
            print(f"aigrid.service: cannot load preflight configs: {exc}", file=sys.stderr)
            return 2
    return svc.run(runtime_cfg, approved_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
