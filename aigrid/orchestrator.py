"""AIGrid Orchestrator — multi-pair loop tying every component together.

Architecture (user direction 2026-11-03):
    watchlist (8 pairs)
        --[PairSelector: deterministic technical scan, every exec candle, no LLM]
        --> top candidates (<= top_candidates)
        --[TradingAgents 3-agent analysis, <= max_ai_analyses_per_cycle]
        --> draft plan (capital from CapitalManager, never from the LLM)
        --> PlanValidator + RiskEngine + ReferenceMarketGuard
        --> engines (<= max_active_pairs) run 5m candles; exits always managed

Per closed candle per pair:
  1. classify regime -> controller (hysteresis/cooldown)
  2. active grid engine processes the candle -> actions
  3. event triggers (gated) may schedule an AI reanalysis
  4. plan adoption is DRAIN-THEN-SWITCH: an engine with open lots keeps
     managing exits until drained; new-plan entries start afterwards.

SAFE_HOLD / paused / AI-unavailable all keep stoploss + monitoring running.
AI NEVER places orders — the Freqtrade dry-run strategy realizes the plan.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .capital import CapitalManager
from .data.binance_adapter import BinanceFuturesDataAdapter
from .grid.engine import GridAction, GridLot, VirtualGridEngine
from .plans.schema import AIGridPlan, GridMode, draft_plan_from_assessment
from .plans.validator import PlanValidator
from .preflight import ConfigCheck, check_telegram_and_safety, compare_configs
from .regime.controller import RegimeController
from .regime.engine import classify
from .risk.engine import AccountState, RiskEngine
from .risk.events import RiskEvent
from .risk.ownership import OwnershipRegistry, OwnerState
from .scheduler import AnalysisRequest, AnalysisScheduler, TriggerKind
from .selection import (
    PairSelector,
    ReferenceMarketGuard,
    ReferenceVerdict,
    SelectionResult,
)
from .settings import AIGridSettings
from .state.ft_fills import FtFillsReader, filled_prices_for_pair
from .state.reconcile import RestartReconciler
from .state.store import EventLog, StateStore
from .ta_adapter import MarketAssessment, TradingAgentsAdapter

logger = logging.getLogger(__name__)


def _stem(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")


# Strategy name Freqtrade reports on our own open trades (sync uses it to
# tell AIGrid fills apart from foreign bots — T-204c).
OWN_STRATEGY_NAME = "AIGridStrategy"


@dataclass
class OrchestratorSnapshot:
    ts_ms: int
    pair: str
    regime: str
    regime_confidence: float
    paused: bool
    pause_reason: str
    grid_mode: str
    plan_id: str | None
    plan_expires_at: str | None
    ai_status: str  # OFF | UNAVAILABLE | OK | BREAKER_OPEN
    ai_last_analysis_ms: int | None
    ai_reasoning: str
    engine: dict
    risk_events: tuple[str, ...] = ()
    pending_plan_id: str | None = None
    selection: dict = field(default_factory=dict)  # latest scan result
    reference: dict = field(default_factory=dict)  # BTC/ETH guard verdict
    ai_metrics: dict = field(default_factory=dict)  # cumulative AI cost ledger

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["risk_events"] = list(self.risk_events)
        return d


class AIGridOrchestrator:
    def __init__(
        self,
        settings: AIGridSettings,
        *,
        exchange_info=None,
        ft_fills_reader=None,
        ft_fills_deadline_s: float = 0.5,
    ):
        self.settings = settings
        self.adapter = BinanceFuturesDataAdapter(settings.data_dir)
        self.ta = TradingAgentsAdapter(settings)
        self.scheduler = AnalysisScheduler(settings.scheduler)
        self.store = StateStore(settings.state_dir)
        self.eventlog = EventLog(settings.state_dir)
        self.reconciler = RestartReconciler(self.store)
        self.ownership = OwnershipRegistry()  # single registry, persisted
        self._load_ownership()
        self.risk = RiskEngine(settings.risk, ownership=self.ownership)
        self.capital = CapitalManager(settings.capital, max_leverage=settings.risk.max_leverage)
        self.selector = PairSelector(
            self.adapter,
            execution_tf=settings.selection.execution_timeframe,
            trend_tf=settings.selection.trend_timeframes[0],
            top_n=settings.selection.top_candidates,
            min_quote_volume_5m=settings.selection.min_quote_volume_5m,
            min_fee_coverage=settings.selection.min_fee_coverage,
        )
        self.ref_guard = ReferenceMarketGuard(
            self.adapter,
            reference_pairs=settings.selection.reference_pairs,
            trend_tf=settings.selection.trend_timeframes[0],
            drop_threshold_pct=settings.selection.ref_dump_threshold_pct,
        )
        self.validator = (
            PlanValidator(exchange_info, settings.risk, capital=self.capital)
            if exchange_info
            else None
        )
        self._exchange_info = exchange_info
        # R12: filled_level_prices is sourced ONLY from Freqtrade order state
        # (GET /api/v1/status). Injectable for hermetic tests; production uses
        # the same env credentials preflight does. The read runs on a single
        # worker with a bounded join — the candle path never waits on a dead
        # or slow REST endpoint (same pattern as the AI worker below).
        self._ft_fills_reader = (
            ft_fills_reader if ft_fills_reader is not None else FtFillsReader.from_env()
        )
        self._ft_fills_deadline_s = float(ft_fills_deadline_s)
        self._fills_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aigrid-fills")
        self._fills_future: Future | None = None
        # ---- multi-pair state
        self._engines: dict[str, VirtualGridEngine] = {}
        self._controllers: dict[str, RegimeController] = {}
        self._pending: dict[str, AIGridPlan] = {}
        self._last_assessment: MarketAssessment | None = None
        self._last_analysis_ms: int | None = None
        self._last_selection: SelectionResult | None = None
        self._last_verdict: ReferenceVerdict | None = None
        self._candle_index = 0
        self._primary_pair = settings.symbols[0]
        self._realized_pnl = 0.0
        self._fees = 0.0
        # ---- SLOW AI (D-092): exactly one analysis runs in a single-worker
        # executor so the 5m candle path never waits on the LLM. _ai_down is
        # the fail-closed flag set by on_ai_unavailable outcomes (timeout or
        # failure): new entries are vetoed while protection keeps running.
        self._ai_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aigrid-ai")
        self._ai_future: Future | None = None
        self._ai_down = False
        self._sched_lock = threading.Lock()  # scheduler state touched by AI worker
        # preflight (T-205b): False until run_preflight() verifies the live
        # config against the approved one; drift sets _halted -> no entries.
        self.preflight_ok = False
        self._halted = False

    # ------------------------------------------------------------------ compat (single-pair tests)
    @property
    def _engine(self) -> VirtualGridEngine | None:
        return self._engines.get(self._primary_pair)

    @_engine.setter
    def _engine(self, v: VirtualGridEngine | None) -> None:
        if v is None:
            self._engines.pop(self._primary_pair, None)
        else:
            self._engines[self._primary_pair] = v

    @property
    def _pending_plan(self) -> AIGridPlan | None:
        return self._pending.get(self._primary_pair)

    @_pending_plan.setter
    def _pending_plan(self, v: AIGridPlan | None) -> None:
        if v is None:
            self._pending.pop(self._primary_pair, None)
        else:
            self._pending[self._primary_pair] = v

    @property
    def controller(self) -> RegimeController:
        return self._controller_for(self._primary_pair)

    def _controller_for(self, pair: str) -> RegimeController:
        if pair not in self._controllers:
            self._controllers[pair] = RegimeController(confirm_candles=3, switch_cooldown_candles=6)
        return self._controllers[pair]

    # ================================================================== lifecycle
    def recover(
        self,
        actual_open_pairs: set[str],
        actual_open_order_ids: set[str],
        actual_open_lots: dict[str, int] | None = None,
    ) -> dict:
        """Restart recovery for ALL persisted pair plans; SAFE_HOLD on mismatch.

        actual_open_lots: normalized symbol -> number of grid lots FT really
        holds. When given, each restored engine's virtual lots are reconciled
        once against the real position before any new entry is allowed.
        """
        results: dict[str, dict] = {}
        restored_symbols: set[str] = set()
        now_ms = int(time.time() * 1000)
        for plan_file in sorted(self.store.state_dir.glob("current_plan_*.json")):
            stem = plan_file.stem.removeprefix("current_plan_")
            r = self.reconciler.recover(
                actual_open_pairs,
                actual_open_order_ids,
                now_ms,
                state_key=f"grid_state_{stem}",
                plan_key=f"current_plan_{stem}",
                actual_open_lots=actual_open_lots,
            )
            pair = None
            if r.plan is not None:
                pair = r.plan.symbol
                restored_symbols.add(pair.split(":", 1)[0].replace("/", ""))
            results[stem] = {
                "status": r.status,
                "reasons": list(r.reasons),
                "plan": r.plan,
                "plan_id": r.plan.plan_id if r.plan else None,
                "restored_lots": r.restored_lots,
                "filled_levels": list(r.filled_levels),
            }
            if r.status == "RESTORED" and r.plan is not None:
                self._engines[r.plan.symbol] = self._build_engine(
                    r.plan, restored_lots=r.restored_lots, filled_levels=r.filled_levels
                )
                self._controller_for(r.plan.symbol).resume()
                self._acquire_ownership(r.plan.symbol, r.plan)
            elif r.plan is not None:
                # untrusted/expired state -> SAFE_HOLD for that pair
                self._controller_for(r.plan.symbol).pause(
                    r.reasons[0] if r.reasons else "RECOVERY_SAFE_HOLD"
                )
            self.eventlog.log("RECOVERY", stem, {"status": r.status, "reasons": list(r.reasons)})
        # no per-pair snapshots -> run the global/legacy recovery path
        if not results:
            r = self.reconciler.recover(
                actual_open_pairs,
                actual_open_order_ids,
                now_ms,
                actual_open_lots=actual_open_lots,
            )
            if r.status == "RESTORED" and r.plan is not None:
                self._engines[r.plan.symbol] = self._build_engine(
                    r.plan, restored_lots=r.restored_lots, filled_levels=r.filled_levels
                )
                self._controller_for(r.plan.symbol).resume()
                self._acquire_ownership(r.plan.symbol, r.plan)
                restored_symbols.add(r.plan.symbol.split(":", 1)[0].replace("/", ""))
            elif r.plan is not None:
                self._controller_for(r.plan.symbol).pause(
                    r.reasons[0] if r.reasons else "RECOVERY_SAFE_HOLD"
                )
            results["_global"] = {
                "status": r.status,
                "reasons": list(r.reasons),
                "plan": r.plan,
                "plan_id": r.plan.plan_id if r.plan else None,
                "restored_lots": r.restored_lots,
                "filled_levels": list(r.filled_levels),
            }
            self.eventlog.log(
                "RECOVERY", "_global", {"status": r.status, "reasons": list(r.reasons)}
            )
        # positions on symbols we could not restore -> POSITION_MISMATCH
        unexpected = actual_open_pairs - restored_symbols
        if unexpected:
            self.eventlog.log("POSITION_MISMATCH", "", {"unexpected": sorted(unexpected)})
        # report the primary pair's result (or the global one) at top level
        primary = (
            results.get(_stem(self._primary_pair))
            or results.get("_global")
            or next(iter(results.values()))
        )
        reasons = list(primary["reasons"])
        if unexpected and str(RiskEvent.POSITION_MISMATCH) not in reasons:
            reasons.append(str(RiskEvent.POSITION_MISMATCH))
        return {
            "status": primary["status"],
            "reasons": reasons,
            "plan": primary["plan"],
            "plan_id": primary["plan_id"],
            "restored_lots": primary["restored_lots"],
            "filled_levels": primary["filled_levels"],
            "pairs": results,
            "unexpected_positions": sorted(unexpected),
        }

    def _build_engine(
        self, plan: AIGridPlan, *, restored_lots: dict | None = None, filled_levels=None
    ) -> VirtualGridEngine:
        """New virtual engine for `plan`; restored lots/filled_levels are loaded
        into engine.state BEFORE the engine may serve (P0-5)."""
        engine = VirtualGridEngine(
            plan,
            fee_rate=self.settings.risk.fee_rate_maker,
            fee_rate_entry=self.settings.risk.fee_rate_maker,
            fee_rate_exit=self.settings.risk.fee_rate_exit,
            fee_rate_emergency=self.settings.risk.fee_rate_emergency,
            slippage_rate=self.settings.risk.slippage_rate,
            leg_stop_loss_pct=0.03,
            max_inventory_notional=plan.max_position_size,
        )
        if isinstance(restored_lots, dict):
            for level_s, lot in restored_lots.items():
                try:
                    level, qty, entry = (
                        int(level_s),
                        float(lot["qty"]),
                        float(lot["entry"]),
                    )
                except (TypeError, ValueError, KeyError):
                    continue
                engine.state.lots[level] = GridLot(
                    level,
                    entry,
                    qty,
                    entry_fee=qty * entry * self.settings.risk.fee_rate_maker,
                    entry_candle_ms=0,
                )  # pre-restart fill -> TP allowed
        try:
            engine.state.filled_levels.update(int(x) for x in (filled_levels or ()))
        except (TypeError, ValueError):
            pass
        # per-level cycle counters (T-213a): reload from the snapshot so ids
        # emitted after restart never collide with ids used before it
        if restored_lots is not None or filled_levels is not None:
            snap = (
                self.store.load_state(f"grid_state_{_stem(plan.symbol)}")
                or self.store.load_state("grid_state")
                or {}
            )
            level_seq = snap.get("level_seq")
            if isinstance(level_seq, dict):
                try:
                    engine.state.level_seq.update({int(k): int(v) for k, v in level_seq.items()})
                except (TypeError, ValueError):
                    pass
        for lv in engine.state.filled_levels:
            # legacy snapshots have no level_seq: a level ever filled counts
            # as at least one armed cycle
            engine.state.level_seq.setdefault(lv, 1)
        return engine

    # ================================================================== ownership
    def _acquire_ownership(self, pair: str, plan: AIGridPlan) -> bool:
        """Lock the symbol for the grid when a plan will open positions."""
        if plan.grid_mode == GridMode.SAFE_HOLD:
            return True
        ok, reason = self.ownership.acquire(
            pair,
            "grid",
            f"grid:{plan.plan_id}",
            ts=datetime.now(UTC).isoformat(),
        )
        if not ok:
            self.eventlog.log(
                "OWNERSHIP_CONFLICT", pair, {"plan_id": plan.plan_id, "reason": reason}
            )
            self._controller_for(pair).pause("OWNERSHIP_CONFLICT")
        return ok

    def _release_ownership(self, pair: str, plan: AIGridPlan) -> None:
        owner = f"grid:{plan.plan_id}"
        if self.ownership.owner_of(pair) != owner:
            return  # lease not ours -> nothing to release
        ok, _ = self.ownership.release(pair, owner, ts=datetime.now(UTC).isoformat())
        if ok:
            self.eventlog.log("OWNERSHIP_RELEASED", pair, {"plan_id": plan.plan_id})

    def _load_ownership(self) -> None:
        data = self.store.load_state("ownership")
        if data:
            self.ownership.load_dict(data)

    def sync_external_positions(self, open_trades: list[dict]) -> None:
        """Reconcile the registry with the exchange's real open trades (T-204b).

        The caller passes every currently-open trade as a dict with "pair",
        "strategy", "is_open", "enter_tag"; this method does no I/O. Call it
        after recover() at startup and every cycle BEFORE evaluating entries.

        A trade belongs to another bot when its strategy is not ours AND its
        enter_tag does not reference one of our plan_ids: the pair takes a
        TREND_OWNED lease owned by "external:<strategy>", so evaluate_entry
        vetoes grid entries on it. Trades whose strategy is OWN_STRATEGY_NAME
        are ours even when enter_tag is None (AIGridStrategy tags entries via
        plan_id, but older/DCA-tagged trades may carry something else) —
        without this guard AIGrid self-locks its own pair (T-204c).
        External leases for pairs no longer reported open are released.
        Leases owned by "grid:*" (our own plans) are never taken or released.
        """
        my_plan_ids = {e.plan.plan_id for e in list(self._engines.values())}
        my_plan_ids.update(p.plan_id for p in list(self._pending.values()))
        external_pairs: set[str] = set()
        ts = datetime.now(UTC).isoformat()
        for trade in open_trades or ():
            if not isinstance(trade, dict) or not trade.get("is_open", True):
                continue
            pair = str(trade.get("pair") or "")
            if not pair:
                continue
            tag = str(trade.get("enter_tag") or "")
            if str(trade.get("strategy") or "") == OWN_STRATEGY_NAME or any(
                pid and pid in tag for pid in my_plan_ids
            ):
                continue  # our own trade, not external
            owner = f"external:{trade.get('strategy') or 'unknown'}"
            external_pairs.add(pair)
            self._own_external(pair, owner, ts)
        # external trades that closed -> release only their external:* leases
        for sym, info in self.ownership.to_dict().items():
            owner = str(info.get("owner") or "")
            if owner.startswith("external:") and sym not in external_pairs:
                ok, _ = self.ownership.release(sym, owner, ts=ts)
                if ok:
                    self.eventlog.log("EXTERNAL_POSITION_RELEASED", sym, {"owner": owner})

    def _own_external(self, pair: str, owner: str, ts: str) -> None:
        """Acquire a TREND_OWNED lease for a foreign bot's open trade."""
        current = self.ownership.owner_of(pair)
        if current == owner:
            return
        if current.startswith("external:"):
            self.ownership.release(pair, current, ts=ts)  # strategy changed
        elif self.ownership.state_of(pair) != OwnerState.FREE:
            return  # grid:/trend:/risk-engine lease — sync must not take it
        ok, reason = self.ownership.acquire(pair, "trend", owner, ts=ts)
        if ok:
            self.eventlog.log("EXTERNAL_POSITION_OWNED", pair, {"owner": owner})
        else:
            self.eventlog.log(
                "EXTERNAL_POSITION_CONFLICT", pair, {"owner": owner, "reason": reason}
            )

    # ================================================================== preflight
    def run_preflight(self, runtime_cfg: dict, approved_cfg: dict) -> ConfigCheck:
        """Compare the live Freqtrade runtime config against the approved
        config (T-205b/c). The caller fetches both dicts — no network I/O
        here.

        Returns a ConfigCheck with three separate groups: drift (exit 1),
        unverifiable (approved baseline missing a checked key, exit 3) and
        match. BOTH drift and unverifiable block publish: either one sets
        self._halted (blocks new entries via run_analysis / adopt_plan /
        process_candle while SL and exits keep running). Safety warnings
        are advisory — logged, never a halt.
        """
        check = compare_configs(runtime_cfg, approved_cfg)
        warnings = check_telegram_and_safety(runtime_cfg)
        self.preflight_ok = check.ok
        if check.drift:
            self._halted = True
            self.eventlog.log("PREFLIGHT_DRIFT", "", {"diffs": check.drift})
        if check.unverifiable:
            self._halted = True
            self.eventlog.log("PREFLIGHT_UNVERIFIABLE", "", {"unverifiable": check.unverifiable})
        if warnings:
            self.eventlog.log("PREFLIGHT_WARNING", "", {"warnings": warnings})
        return check

    # ================================================================== selection
    def scan_watchlist(self, as_of: datetime | None = None) -> SelectionResult:
        """Deterministic technical scan over the watchlist (NO LLM)."""
        as_of = as_of or datetime.now(UTC)
        result = self.selector.scan(self.settings.watchlist, as_of)
        self._last_selection = result
        self.eventlog.log(
            "PAIR_SCAN",
            "",
            {
                "candidates": result.top_pairs,
                "rejected": {m.pair: list(m.rejected) for m in result.rejected},
            },
        )
        return result

    def check_reference(self, as_of: datetime | None = None) -> ReferenceVerdict:
        as_of = as_of or datetime.now(UTC)
        verdict = self.ref_guard.check(as_of)
        self._last_verdict = verdict
        return verdict

    def active_pairs(self) -> list[str]:
        """Pairs currently allowed to open new positions (active grid plans)."""
        return [p for p, e in list(self._engines.items()) if e.plan.grid_mode != GridMode.SAFE_HOLD]

    def analyze_candidates(self, trigger: TriggerKind, as_of: datetime | None = None) -> list[dict]:
        """One coordination cycle: scan -> analyze at most max_ai candidates."""
        scan = self.scan_watchlist(as_of)
        verdict = self.check_reference(as_of)
        results = []
        budget = self.settings.selection.max_ai_analyses_per_cycle
        for cand in scan.candidates:
            if budget <= 0:
                break
            pair = cand.pair
            # skip pairs whose plan is fresh already
            eng = self._engines.get(pair)
            if eng is not None and not eng.plan.is_expired():
                continue
            res = self.run_analysis(pair, trigger, as_of=as_of)
            results.append(res)
            if res.get("dispatched"):
                budget -= 1
        self.eventlog.log(
            "ANALYSIS_CYCLE",
            "",
            {
                "candidates": scan.top_pairs,
                "analyzed": len(results),
                "reference": verdict.as_dict(),
            },
        )
        return results

    # ================================================================== per-candle
    def process_candle(
        self, pair: str, candle: dict, *, history_df=None, force_no_entry: bool = False
    ) -> OrchestratorSnapshot:
        """candle: {open_ms, open, high, low, close, volume} CLOSED candle."""
        self._candle_index += 1
        self._primary_pair = (
            pair if pair in self._engines or pair in self.settings.symbols else self._primary_pair
        )
        now_ms = int(candle["open_ms"]) + 1
        controller = self._controller_for(pair)
        engine = self._engines.get(pair)

        # 1) regime (uses provided history df — caller guarantees no lookahead)
        if history_df is not None and len(history_df) >= 60:
            rr = classify(history_df)
            controller.update(rr.regime, self._candle_index)

        # 2) engine step
        actions: list[GridAction] = []
        if engine is not None:
            permit = (
                not controller.paused
                and not force_no_entry
                and not self._halted
                and not self._ai_down
                and self._pending.get(pair) is None
            )
            actions = engine.on_candle(
                candle["open_ms"],
                candle["open"],
                candle["high"],
                candle["low"],
                candle["close"],
                permit_entry=permit,
            )
            self._realized_pnl = engine.state.realized_pnl
            self._fees = engine.state.total_fees
            if actions:
                self.eventlog.log(
                    "GRID_ACTIONS",
                    pair,
                    {"n": len(actions), "kinds": [a.kind.value for a in actions]},
                )
            # drain-then-switch: engine fully drained -> adopt pending plan
            pending = self._pending.get(pair)
            if pending is not None and not engine.state.lots:
                self._swap_engine(pair, pending)
                self._pending.pop(pair, None)
                engine = self._engines.get(pair)
            # ownership: engine flat AND finished (halted/plan expired) -> the
            # position is closed, so the symbol lease is released
            if (
                engine is not None
                and not engine.state.lots
                and (engine.state.halted or engine.plan.is_expired())
            ):
                self._release_ownership(pair, engine.plan)

        # 3) plan expiry check
        plan = engine.plan if engine else None
        risk_events: list[str] = []
        if plan is not None and plan.is_expired():
            risk_events.append(str(RiskEvent.AI_PLAN_EXPIRED))
            if engine is not None and engine.state.lots:
                force_no_entry = True

        # 4) event-driven triggers (only when enabled)
        if self.settings.enabled:
            self._maybe_trigger_events(pair, candle, history_df)

        snap = self.snapshot(pair, now_ms, risk_events, candle)
        self.persist(snap)
        return snap

    # ================================================================== AI cycle
    def run_analysis(
        self,
        pair: str,
        trigger: TriggerKind,
        as_of: datetime | None = None,
        history_df=None,
    ) -> dict:
        """Dispatch one AI analysis cycle to the SLOW-AI worker thread.

        Returns immediately ({"dispatched": True, "pending": True}) — the
        candle path never waits on the LLM. At most ONE analysis is in
        flight; a busy worker makes this call a no-op (no unbounded queue).
        The outcome lands via _ai_pipeline -> state + AI_ANALYSIS_* events.
        """
        if self._halted:
            return {"dispatched": False, "reason": "PREFLIGHT_DRIFT"}
        if self._ai_future is not None and not self._ai_future.done():
            return {"dispatched": False, "reason": "AI_IN_FLIGHT"}
        req = _mk_request(pair, trigger, history_df)
        with self._sched_lock:
            ok, reason = self.scheduler.should_analyze(req)
            if ok:
                self.scheduler.record_dispatch(req)
        if not ok:
            return {"dispatched": False, "reason": reason}

        if not self.settings.enabled:
            return {"dispatched": False, "reason": "AI_DISABLED"}

        self._ai_future = self._ai_executor.submit(self._ai_pipeline, pair, req, as_of)
        return {"dispatched": True, "pending": True, "pair": pair}

    def wait_for_analysis(self, timeout_s: float = 10.0) -> dict | None:
        """Result dict of the in-flight/finished analysis, or None on timeout.
        For tests and supervisors — the candle path never calls this."""
        fut = self._ai_future
        if fut is None:
            return None
        try:
            return fut.result(timeout=timeout_s)
        except FutureTimeout:
            return None

    def _ai_pipeline(self, pair: str, req: AnalysisRequest, as_of: datetime | None) -> dict:
        """Worker body (runs in _ai_executor): snapshot -> assess -> capital ->
        draft -> guard -> validate -> risk -> adopt. Never raises; the outcome
        is a structured dict mirrored into events + state."""
        started = time.monotonic()
        deadline_s = float(self.settings.scheduler.llm_total_deadline_seconds)
        try:
            try:
                snap = self.adapter.snapshot(pair, self.settings.timeframe, as_of=as_of)
            except Exception as exc:
                with self._sched_lock:
                    self.scheduler.record_result(req, False, _elapsed(started), f"DATA:{exc}")
                self.eventlog.log("AI_ANALYSIS_FAILED", pair, {"error": f"DATA:{exc}"})
                self._ai_down = True
                return {"ok": False, "error": f"DATA:{exc}"}

            assessment = self.ta.assess(pair, snap.to_ai_context(), snap.as_of_ms)
            latency = _elapsed(started)
            with self._sched_lock:
                self.scheduler.record_result(
                    req,
                    assessment.ok,
                    latency,
                    "" if assessment.ok else assessment.reasoning_summary,
                )
            self._last_assessment = assessment
            self._last_analysis_ms = int(time.time() * 1000)
            if not assessment.ok:
                decision = self.risk.on_ai_unavailable()
                self._ai_down = True
                kind = "AI_ANALYSIS_TIMEOUT" if assessment.timed_out else "AI_ANALYSIS_FAILED"
                self.eventlog.log(
                    kind,
                    pair,
                    {
                        "reason": assessment.reasoning_summary,
                        "latency_ms": latency,
                        "events": list(decision.events),
                    },
                )
                return {
                    "ok": False,
                    "timed_out": assessment.timed_out,
                    "error": assessment.reasoning_summary,
                    "latency_ms": latency,
                }
            self._ai_down = False

            # --- D-092: an analysis that only finished AFTER the total
            # deadline is stale — refuse to act on it.
            if time.monotonic() - started > deadline_s:
                self.eventlog.log(
                    "PLAN_STALE_REJECTED",
                    pair,
                    {
                        "reason": "ANALYSIS_OVER_DEADLINE",
                        "elapsed_ms": _elapsed(started),
                    },
                )
                return {
                    "ok": False,
                    "error": "PLAN_STALE_REJECTED",
                    "latency_ms": latency,
                }

            # --- reference guard: BTC/ETH dump vetoes alt LONG/RANGE grids
            verdict = self.check_reference(as_of)
            controller = self._controller_for(pair)
            grid_mode = _mode_for(assessment.regime, controller.paused)
            is_reference = pair in self.settings.selection.reference_pairs
            if not is_reference:
                if grid_mode in ("LONG_GRID", "RANGE_GRID") and not verdict.alt_long_ok:
                    self.eventlog.log(
                        "REFERENCE_VETO",
                        pair,
                        {"mode": grid_mode, "reasons": list(verdict.reasons)},
                    )
                    return {
                        "ok": False,
                        "error": f"REFERENCE_VETO:{list(verdict.reasons)}",
                    }
                if grid_mode == "SHORT_GRID" and not verdict.alt_short_ok:
                    self.eventlog.log(
                        "REFERENCE_VETO",
                        pair,
                        {"mode": grid_mode, "reasons": list(verdict.reasons)},
                    )
                    return {
                        "ok": False,
                        "error": f"REFERENCE_VETO:{list(verdict.reasons)}",
                    }

            # --- active-pair cap (grid plans only; SAFE_HOLD always allowed)
            if (
                grid_mode != "SAFE_HOLD"
                and pair not in self.active_pairs()
                and len(self.active_pairs()) >= self.settings.selection.max_active_pairs
            ):
                self.eventlog.log(
                    "MAX_ACTIVE_PAIRS",
                    pair,
                    {"cap": self.settings.selection.max_active_pairs},
                )
                return {"ok": False, "error": "MAX_ACTIVE_PAIRS_REACHED"}

            # --- capital allocation (CapitalManager — never the LLM)
            account = self._account_state()
            allocation = self.capital.allocate(
                pair,
                account.equity_usdt,
                account.allocated_margins,
                levels=(assessment.proposal.levels if assessment.proposal else 5),
            )
            if allocation is None:
                self.eventlog.log("NO_BUDGET", pair, {})
                return {"ok": False, "error": "NO_BUDGET_AVAILABLE"}
            prop = assessment.proposal
            plan = draft_plan_from_assessment(
                pair,
                assessment.regime,
                assessment.confidence,
                float(snap.ohlcv["close"].iloc[-1]),
                snap.indicators["atr"] or 0.0,
                grid_mode=grid_mode,
                ttl_minutes=self.settings.risk.plan_ttl_minutes,
                margin_usdt=allocation.margin_usdt,
                leverage=allocation.leverage,
                levels=allocation.levels,
                level_margin_usdt=allocation.level_margin_usdt,
                grid_lower=(prop.grid_lower if prop else None),
                grid_upper=(prop.grid_upper if prop else None),
                reasoning=assessment.reasoning_summary,
            )

            # --- TTL: an already-expired plan is stale — never adopted
            if plan.is_expired():
                self.eventlog.log(
                    "PLAN_STALE_REJECTED",
                    pair,
                    {
                        "reason": "PLAN_TTL_EXPIRED",
                        "plan_id": plan.plan_id,
                        "elapsed_ms": _elapsed(started),
                    },
                )
                return {
                    "ok": False,
                    "error": "PLAN_STALE_REJECTED",
                    "plan_id": plan.plan_id,
                    "latency_ms": latency,
                }

            # --- validate
            if self.validator is not None:
                verdict_v = self.validator.validate(
                    plan, last_close=float(snap.ohlcv["close"].iloc[-1])
                )
                if not verdict_v.approved:
                    self.eventlog.log(
                        "PLAN_REJECTED",
                        pair,
                        {"reasons": list(verdict_v.reasons), "plan_id": plan.plan_id},
                    )
                    return {
                        "ok": False,
                        "error": f"PLAN_REJECTED:{list(verdict_v.reasons)}",
                    }

            # --- risk entry check (independent)
            decision = self.risk.evaluate_entry(plan, account, capital_manager=self.capital)
            if not decision.allowed:
                self.eventlog.log(
                    "PLAN_VETOED",
                    pair,
                    {"events": list(decision.events), "plan_id": plan.plan_id},
                )
                return {"ok": False, "error": f"RISK_VETO:{list(decision.events)}"}

            # --- adopt (drain-then-switch)
            self.adopt_plan(plan)
            self.eventlog.log(
                "PLAN_ADOPTED",
                pair,
                {
                    "plan_id": plan.plan_id,
                    "mode": str(plan.grid_mode),
                    "regime": plan.market_regime,
                    "confidence": plan.confidence,
                },
            )
            self.eventlog.log(
                "AI_ANALYSIS_COMPLETED",
                pair,
                {
                    "plan_id": plan.plan_id,
                    "mode": str(plan.grid_mode),
                    "regime": plan.market_regime,
                    "confidence": plan.confidence,
                    "latency_ms": latency,
                },
            )
            return {
                "ok": True,
                "plan_id": plan.plan_id,
                "grid_mode": str(plan.grid_mode),
                "latency_ms": latency,
            }
        except Exception as exc:
            logger.exception("AI pipeline crashed for %s", pair)
            self._ai_down = True
            try:
                self.eventlog.log(
                    "AI_ANALYSIS_FAILED", pair, {"error": f"{type(exc).__name__}:{exc}"}
                )
            except Exception:
                pass  # eventlog may already be closed
            return {"ok": False, "error": f"{type(exc).__name__}:{exc}"}

    def adopt_plan(self, plan: AIGridPlan) -> None:
        """Drain-then-switch adoption (never closes existing lots)."""
        if self._halted:
            # a new plan would be published to the Freqtrade strategy and
            # could open entries — refuse while preflight-halted (T-205b)
            self.eventlog.log(
                "PLAN_VETOED",
                plan.symbol,
                {"plan_id": plan.plan_id, "reason": "PREFLIGHT_DRIFT"},
            )
            return
        pair = plan.symbol
        eng = self._engines.get(pair)
        if eng is None or not eng.state.lots:
            self._swap_engine(pair, plan)
        else:
            self._pending[pair] = plan
            self.eventlog.log(
                "PLAN_QUEUED",
                pair,
                {"plan_id": plan.plan_id, "waiting_drain": len(eng.state.lots)},
            )

    def _swap_engine(self, pair: str, plan: AIGridPlan) -> None:
        old = self._engines.get(pair)
        if old is not None:
            self._release_ownership(pair, old.plan)
        self._engines[pair] = self._build_engine(plan)
        self._acquire_ownership(pair, plan)
        controller = self._controller_for(pair)
        if plan.grid_mode == GridMode.SAFE_HOLD:
            controller.pause("PLAN_SAFE_HOLD")
        else:
            pr = controller.pause_reason
            if (
                pr in ("INITIAL_SAFE_HOLD", "")
                or pr.startswith("PLAN_")
                or plan.market_regime == controller.confirmed
            ):
                controller.resume()

    # ================================================================== events
    def _maybe_trigger_events(self, pair: str, candle: dict, history_df) -> None:
        engine = self._engines.get(pair)
        plan = engine.plan if engine else None
        close = candle["close"]
        triggers: list[TriggerKind] = []
        if plan is not None and plan.grid_mode != GridMode.SAFE_HOLD:
            if close < plan.grid_lower or close > plan.grid_upper:
                triggers.append(TriggerKind.PRICE_BREAKOUT)
        if plan is not None and plan.is_expired():
            triggers.append(TriggerKind.PLAN_EXPIRED)
        if history_df is not None and len(history_df) >= 60:
            rr = classify(history_df)
            if rr.regime == "HIGH_VOLATILITY" and rr.reason == "VOLATILITY_EXTREME":
                triggers.append(TriggerKind.VOLATILITY_SPIKE)
        for t in triggers:
            req = _mk_request(pair, t, history_df)
            with self._sched_lock:
                ok, _reason = self.scheduler.should_analyze(req)
                if ok:
                    self.scheduler.record_dispatch(req)
            if ok:
                self.eventlog.log("EVENT_TRIGGER", pair, {"trigger": t.value})

    # ================================================================== snapshot
    def _account_state(self, prices: dict[str, float] | None = None) -> AccountState:
        prices = prices or {}
        equity = 1000.0
        inventory = sum(
            e.inventory_notional(prices.get(p, e.plan.grid_upper))
            for p, e in list(self._engines.items())
        )
        allocated: dict[str, float] = {}
        for p, e in list(self._engines.items()):
            if e.plan.grid_mode != GridMode.SAFE_HOLD:
                allocated[p] = allocated.get(p, 0.0) + e.plan.max_capital
        for p, pp in list(self._pending.items()):
            if pp.grid_mode != GridMode.SAFE_HOLD:
                allocated[p] = allocated.get(p, 0.0) + pp.max_capital
        return AccountState(
            equity_usdt=equity,
            starting_equity_usdt=equity,
            peak_equity_usdt=max(equity, equity + self._realized_pnl),
            realized_pnl_today_usdt=self._realized_pnl,
            total_exposure_usdt=inventory,
            free_margin_usdt=equity - inventory,
            allocated_margins=allocated,
        )

    def snapshot(
        self, pair: str, now_ms: int, risk_events: list[str], candle: dict
    ) -> OrchestratorSnapshot:
        engine = self._engines.get(pair)
        controller = self._controller_for(pair)
        plan = engine.plan if engine else None
        assessment = self._last_assessment
        ai_status = "OFF"
        if self.settings.enabled:
            if self.scheduler.breaker_open(now_ms):
                ai_status = "BREAKER_OPEN"
            elif assessment is None:
                ai_status = "NO_ANALYSIS_YET"
            elif assessment.ok:
                ai_status = "OK"
            else:
                ai_status = "UNAVAILABLE"
        engine_dict = engine.to_dict(candle["close"]) if engine else {}
        sel = self._last_selection
        return OrchestratorSnapshot(
            ts_ms=now_ms,
            pair=pair,
            regime=controller.confirmed,
            regime_confidence=0.0,
            paused=controller.paused,
            pause_reason=controller.pause_reason,
            grid_mode=controller.grid_mode_for(),
            plan_id=plan.plan_id if plan else None,
            plan_expires_at=plan.expires_at.isoformat() if plan else None,
            ai_status=ai_status,
            ai_last_analysis_ms=self._last_analysis_ms,
            ai_reasoning=assessment.reasoning_summary if assessment else "",
            engine=engine_dict,
            risk_events=tuple(risk_events),
            pending_plan_id=(self._pending.get(pair).plan_id if self._pending.get(pair) else None),
            selection={
                "candidates": sel.top_pairs if sel else [],
                "as_of_ms": sel.as_of_ms if sel else 0,
            },
            reference=self._last_verdict.as_dict() if self._last_verdict else {},
            ai_metrics=self.ta.cost.as_dict(),
        )

    # ================================================================== persistence
    def persist(self, snapshot: OrchestratorSnapshot | None = None) -> None:
        active = set()
        for pair, engine in list(self._engines.items()):
            stem = _stem(pair)
            active.add(stem)
            self.store.save_state(f"grid_state_{stem}", engine.to_dict(engine.plan.grid_upper))
            self.store.save_state(
                f"current_plan_{stem}",
                {
                    "plan": engine.plan.model_dump(mode="json"),
                    "symbol": engine.plan.symbol,
                },
            )
        # clear stale per-pair snapshots whose engines were removed
        for f in self.store.state_dir.glob("grid_state_*.json"):
            stem = f.stem.removeprefix("grid_state_")
            if stem not in active:
                self.store.clear_state(f"grid_state_{stem}")
                self.store.clear_state(f"current_plan_{stem}")
        self.store.save_state("ownership", self.ownership.to_dict())
        self._write_ft_integration()
        if snapshot is not None:
            self.store.save_state("aigrid_snapshot", snapshot.to_dict())

    def _write_ft_integration(self) -> None:
        """Integration file for the Freqtrade strategy (per-pair plans).

        dry_run_verified=True only in shadow/dryrun modes.
        """
        verified = self.settings.mode in ("shadow", "dryrun")
        payload: dict[str, Any] = {
            "ts_ms": int(time.time() * 1000),
            "enabled": bool(self._engines) and self.settings.enabled,
            "dry_run_verified": verified,
            "mode": self.settings.mode,
            "plans": {},
            "filled_levels": {},
            # filled_levels is the VIRTUAL candle-driven index book — kept
            # for older consumers, never trusted as execution truth (R12).
            "filled_levels_source": "virtual",
            "reconciliation_state": "UNVERIFIED",
        }
        # R12: filled_level_prices is published ONLY from Freqtrade order
        # state (one GET /api/v1/status per persist). REST down, unparseable
        # or zero trades -> the key stays ABSENT and the strategy fails
        # closed. engine.state.lots is never consulted as a fallback.
        trades = self._fetch_ft_trades()
        if trades:
            payload["reconciliation_state"] = "VERIFIED"
            payload["filled_level_prices"] = {}
            payload["unmapped_orders"] = {}
        for pair, engine in list(self._engines.items()):
            plan = engine.plan
            levels = plan.level_prices()
            n_levels = max(1, len(levels))
            # adjust_trade_position expects stake = MARGIN (level_stake),
            # never notional (= margin x leverage) — P0-3.
            level_stake = float(plan.level_stake or 0.0)
            if level_stake <= 0:
                level_stake = plan.max_capital / n_levels
            if level_stake * len(levels) > plan.max_capital + 1e-6:
                level_stake = plan.max_capital / n_levels
                logger.warning(
                    "level_stake over max_capital for %s; clamped to %.4f",
                    pair,
                    level_stake,
                )
                self.eventlog.log(
                    "LEVEL_STAKE_CLAMPED",
                    pair,
                    {"max_capital": plan.max_capital, "levels": len(levels)},
                )
            level_stake = round(level_stake, 2)
            if level_stake * len(levels) > plan.max_capital + 1e-6:
                level_stake = math.floor(level_stake * 100) / 100
            payload["plans"][pair] = {
                "plan_id": plan.plan_id,
                "symbol": plan.symbol,
                "grid_mode": str(plan.grid_mode),
                "grid_lower": plan.grid_lower,
                "grid_upper": plan.grid_upper,
                "grid_levels": plan.grid_levels,
                "grid_levels_prices": levels,
                "grid_spacing": plan.grid_spacing,
                "stop_loss": plan.stop_loss,
                "level_stake": level_stake,
                "expires_at": plan.expires_at.isoformat(),
            }
            payload["filled_levels"][pair] = sorted(engine.state.filled_levels)
            if trades:
                res = filled_prices_for_pair(
                    trades,
                    pair=pair,
                    plan_id=plan.plan_id,
                    level_prices=levels,
                    price_tick=self._price_tick(pair),
                )
                payload["filled_level_prices"][pair] = res.prices
                payload["unmapped_orders"][pair] = res.unmapped_orders
        self.store.save_state("ft_integration", payload)

    def _fetch_ft_trades(self) -> list[dict] | None:
        """GET /api/v1/status through the injected reader; None on ANY
        failure — the caller omits filled_level_prices (fail-closed).

        The read runs on a dedicated worker; this method joins it for at
        most ``_ft_fills_deadline_s`` seconds (a dead local REST port can
        cost seconds before the OS reports the refusal). A read still in
        flight is left running — the next persist consumes it whenever it
        lands, so at most one request is ever outstanding.
        """
        reader = self._ft_fills_reader
        if reader is None or not self._engines:
            return None
        fut = self._fills_future
        if fut is None or fut.done():
            fut = self._fills_executor.submit(reader.fetch_open_trades)
            self._fills_future = fut
        try:
            trades = fut.result(timeout=self._ft_fills_deadline_s)
        except Exception as exc:
            if not isinstance(exc, FutureTimeout):
                logger.warning("FT fills read failed: %s", exc)
            return None
        return trades if isinstance(trades, list) else None

    def _price_tick(self, pair: str) -> float | None:
        """tickSize from the fapi_exchangeinfo snapshot; None -> the mapper
        tightens to an exact (1e-9) match against the plan's level prices."""
        ei = self._exchange_info
        if ei is None:
            return None
        try:
            tick = float(ei.rules_for(pair).tick_size)
        except Exception:
            return None
        return tick if tick > 0 else None

    def close(self, ai_join_timeout: float = 5.0) -> None:
        """Stop: join the AI worker for at most ai_join_timeout seconds —
        shutdown must never hang on a stuck LLM call (SIGTERM path)."""
        self._shutdown_ai(ai_join_timeout)
        self.persist()
        self._fills_executor.shutdown(wait=False, cancel_futures=True)
        self.eventlog.close()

    def _shutdown_ai(self, timeout_s: float) -> None:
        self._ai_executor.shutdown(wait=False, cancel_futures=True)
        fut = self._ai_future
        if fut is not None and not fut.done():
            try:
                fut.result(timeout=timeout_s)
            except Exception:
                pass  # bounded join; never hang


def _mk_request(pair: str, trigger: TriggerKind, history_df) -> AnalysisRequest:
    digest = ""
    if history_df is not None and len(history_df):
        digest = f"{len(history_df)}:{history_df.index[-1]}"
    return AnalysisRequest(pair, trigger, int(time.time() * 1000), digest)


def _mode_for(regime: str, paused: bool) -> str:
    if paused:
        return "SAFE_HOLD"
    return {
        "RANGE": "RANGE_GRID",
        "UPTREND": "LONG_GRID",
        "DOWNTREND": "SHORT_GRID",
        "HIGH_VOLATILITY": "SAFE_HOLD",
    }.get(regime, "SAFE_HOLD")


def _elapsed(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
