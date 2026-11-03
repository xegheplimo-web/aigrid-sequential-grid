"""Guarded atomic plan publisher (R14 / D-093) + dry-run publish cycle.

Two layers:

- :class:`PlanPublisher` — takes a deterministic plan plus runtime facts
  and either writes ``ft_integration.json`` ATOMICALLY (tmp + os.replace,
  the same mechanism ``StateStore.save_state`` uses) or writes NOTHING and
  emits ``PUBLISH_BLOCKED`` with machine-readable reasons. Guards are the
  card-R14 §2 set (a–k) extended by addendum v2 (D-100:
  OPEN_PARTIAL_FILL_ACCOUNTED, NO_CONFLICTING_EXIT_ORDERS) and addendum v4
  (D-102: STARTUP_RECONCILIATION_PASS, >=576 closed-candle freshness,
  max_open_trades, CLOCK_SKEW < 5s, pending-not-beyond-TTL,
  restart-duplicate-check on (plan_id, pair)).

- :class:`DeterministicCycle` — the per-tick chain the service dry-run
  branch drives: closed-candle detect -> ``process_candle`` -> regime ->
  deterministic plan -> guarded publish -> restore the integration file if
  an engine persist() clobbered it. NO LLM and no non-local network call
  on this path: the only I/O is the local candle file / the read-only
  Freqtrade REST (GET /status, /show_config, /health, /balance, /pair_candles).

``LIVE_REFUSED`` is the harshest guard: the publish path writes no file
of any kind for it — not even a control or kill-switch artifact. The only
side effect permitted is the PUBLISH_BLOCKED event in the SQLite log.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import os
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from .plans.deterministic import PLAN_TTL_SECONDS, build_deterministic_plan
from .plans.schema import AIGridPlan, GridMode
from .preflight import (
    DEFAULT_API_URL,
    FT_API_PASS_ENV,
    FT_API_URL_ENV,
    FT_API_USER_ENV,
    write_attestation,
)
from .regime.engine import classify
from .risk.ownership import OwnerState
from .settings import AIGridSettings
from .state.ft_fills import OrderFillLedger, filled_prices_for_pair
from .state.store import StateStore

logger = logging.getLogger(__name__)

DEFAULT_THROTTLE_S = 240.0  # don't republish over a <240s-fresh file
MIN_CLOSED_CANDLES = 576  # 48h of 5m candles (D-102 freshness)
MAX_CLOCK_SKEW_MS = 5_000  # D-102: server UTC vs local < 5s
PENDING_TTL_MS = PLAN_TTL_SECONDS * 1_000

FACTS_UNAVAILABLE = "FACTS_UNAVAILABLE"

STATUS_PUBLISHED = "PUBLISHED"
STATUS_BLOCKED = "BLOCKED"
STATUS_SKIPPED = "SKIPPED"


@dataclass(frozen=True)
class RuntimeFacts:
    """What the publish path was able to READ from the runtime. ``None``
    means "not fetched". The aggregate capital guard fails closed when
    exposure or equity is unavailable; older optional guards may skip."""

    runtime_config: dict = field(default_factory=dict)
    dry_run: bool | None = None
    trading_mode: str | None = None
    runmode: str | None = None
    pair_whitelist: Sequence[str] | None = None
    max_open_trades: int | None = None
    open_trades: list[dict] | None = None
    equity_usdt: float | None = None
    exchange_healthy: bool | None = None
    server_time_ms: int | None = None
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class PublishOutcome:
    status: str  # PUBLISHED | BLOCKED | SKIPPED
    reasons: tuple[str, ...] = ()
    event_id: str | None = None


class PlanPublisher:
    """Guarded atomic publisher of deterministic plans.

    Persistence (restart-duplicate + crowded-plan + crash restore) lives
    in ``<state_dir>/published_plans.json`` written via StateStore (atomic
    itself). A ``(plan_id, pair)`` key is published at most once — ever.
    """

    def __init__(
        self,
        store: StateStore,
        eventlog: Any = None,
        *,
        validator: Any = None,
        ownership: Any = None,
        settings: AIGridSettings | None = None,
        throttle_s: float = DEFAULT_THROTTLE_S,
        min_closed_candles: int = MIN_CLOSED_CANDLES,
        max_clock_skew_ms: int = MAX_CLOCK_SKEW_MS,
        pending_ttl_ms: int = PENDING_TTL_MS,
        timeframe_s: int = 300,
        price_tick_of: Callable[[str], float | None] | None = None,
        clock: Callable[[], float] = time.time,
        integration_name: str = "ft_integration",
        ledger_name: str = "published_plans",
    ):
        self._store = store
        self._eventlog = eventlog
        self._validator = validator
        self._ownership = ownership
        self._settings = settings
        self.throttle_s = float(throttle_s)
        self.min_closed_candles = int(min_closed_candles)
        self.max_clock_skew_ms = int(max_clock_skew_ms)
        self.pending_ttl_ms = int(pending_ttl_ms)
        factor = (
            float(getattr(settings.risk, "data_max_staleness_factor", 3.0))
            if settings is not None
            else 3.0
        )
        self.max_staleness_ms = factor * timeframe_s * 1000.0
        self._price_tick_of = price_tick_of
        self._clock = clock
        self._integration = integration_name
        self._ledger_name = ledger_name
        self._ledger: dict[str, Any] = self._load_ledger()
        self._runtime_facts: RuntimeFacts | None = None
        self._sign_failed = False

    # ------------------------------------------------------------------ public
    def publish(
        self,
        plan: AIGridPlan,
        *,
        facts: RuntimeFacts | None = None,
        candles_count: int = 0,
        candle_close_ms: int | None = None,
        last_close: float | None = None,
        startup_reconciled: bool = False,
        preflight_ok: bool | None = None,
        fill_ledger: OrderFillLedger | None = None,
    ) -> PublishOutcome:
        """Run every guard; on pass write the integration file atomically."""
        if facts is None or facts.unavailable_reason:
            self._log(
                "PUBLISH_BLOCKED",
                plan.symbol,
                {
                    "plan_id": plan.plan_id,
                    "reasons": [FACTS_UNAVAILABLE],
                },
            )
            return PublishOutcome(STATUS_BLOCKED, (FACTS_UNAVAILABLE,))
        now_ms = int(self._clock() * 1000)
        now_dt = datetime.fromtimestamp(now_ms / 1000.0, tz=UTC)
        capital_detail = self._capital_guard(plan, facts, now_ms)
        reasons, fills = self._guard_reasons(
            plan,
            facts=facts,
            candles_count=candles_count,
            candle_close_ms=candle_close_ms,
            last_close=last_close,
            startup_reconciled=startup_reconciled,
            preflight_ok=preflight_ok,
            fill_ledger=fill_ledger,
            now_ms=now_ms,
            now_dt=now_dt,
        )
        if capital_detail is not None:
            reasons.append("CAPITAL_CAP_BLOCKED")
        if reasons:
            self._log(
                "PUBLISH_BLOCKED",
                plan.symbol,
                {
                    "plan_id": plan.plan_id,
                    "reasons": list(reasons),
                    "live_refused": "LIVE_REFUSED" in reasons,
                    **({"detail": capital_detail} if capital_detail is not None else {}),
                },
            )
            return PublishOutcome(STATUS_BLOCKED, reasons)

        target = self._store.state_dir / f"{self._integration}.json"
        if target.exists() and self.throttle_s > 0:
            age = self._clock() - target.stat().st_mtime
            if age < self.throttle_s:
                self._log(
                    "PUBLISH_THROTTLED",
                    plan.symbol,
                    {"plan_id": plan.plan_id, "age_s": round(age, 1)},
                )
                return PublishOutcome(STATUS_SKIPPED, ("PUBLISH_THROTTLED",))

        payload = self._payload(plan, now_ms, fills, facts)
        self._runtime_facts = facts
        self._write_signed(payload, facts)
        event_id = self._event_id(plan, now_ms)
        self._record_publish(plan, payload, event_id, now_ms)
        self._log(
            "PLAN_PUBLISHED",
            plan.symbol,
            {"plan_id": plan.plan_id, "event_id": event_id},
        )
        return PublishOutcome(STATUS_PUBLISHED, (), event_id)

    def _capital_guard(
        self, plan: AIGridPlan, facts: RuntimeFacts | None, now_ms: int
    ) -> dict | str | None:
        """Use one cycle snapshot and reserve earlier publications in the ledger.

        Required margin is notional / leverage; reserve is minimum free margin.
        Missing, negative or nonfinite inputs cannot establish safe exposure.
        """

        def number(value, *, positive=False):
            if isinstance(value, bool):
                raise TypeError("boolean capital input")
            result = float(value)
            if not math.isfinite(result) or result < 0 or (positive and result == 0):
                raise ValueError("invalid capital input")
            return result

        def quote(pair):
            return str(pair).split("/", 1)[1].split(":", 1)[0]

        try:
            if facts is None or facts.open_trades is None or self._settings is None:
                return "UNKNOWN_EXPOSURE"
            equity = number(facts.equity_usdt, positive=True)
            target_quote = quote(plan.symbol)
            marked = pending = pending_margin = 0.0
            for trade in facts.open_trades:
                if not isinstance(trade, dict):
                    return "UNKNOWN_EXPOSURE"
                if not trade.get("is_open", True):
                    continue
                if quote(trade["pair"]) != target_quote:
                    continue
                marked += number(trade["amount"]) * number(trade["current_rate"], positive=True)
                orders = trade.get("orders", [])
                if not isinstance(orders, list):
                    return "UNKNOWN_EXPOSURE"
                for order in orders:
                    if not isinstance(order, dict):
                        return "UNKNOWN_EXPOSURE"
                    if not order.get("ft_is_entry") or str(order.get("status")).lower() not in (
                        "open",
                        "new",
                        "partially_filled",
                    ):
                        continue
                    remaining = number(number(order["amount"]) - number(order.get("filled", 0)))
                    notional = remaining * number(order["price"], positive=True)
                    pending += notional
                    pending_margin += notional / number(trade["leverage"], positive=True)
            allocated = 0.0
            # Read without pruning/saving the ledger: even a refused publish
            # (especially LIVE_REFUSED) must have no filesystem side effects.
            for symbol, record in self._ledger["active"].items():
                if now_ms >= number(record["expires_at_ms"]):
                    continue
                active = record["plan"]
                if quote(symbol) == target_quote:
                    allocated += number(active["max_position_size"]) / number(
                        active["max_leverage"], positive=True
                    )
            required = number(plan.max_position_size) / number(plan.max_leverage, positive=True)
            reserve = number(self._settings.risk.min_free_margin_usdt)
            projected_n = marked + pending + plan.max_position_size
            projected_c = allocated + pending_margin + required + reserve
            exposure_cap = number(self._settings.risk.max_total_exposure_usdt, positive=True)
            budget = equity * number(self._settings.capital.total_budget_pct, positive=True)
            detail = {
                "N_mark": marked,
                "N_pending": pending,
                "N_projected": projected_n,
                "C_allocated": allocated,
                "C_pending": pending_margin,
                "required_margin": required,
                "C_reserve": reserve,
                "C_projected": projected_c,
                "equity": equity,
                "exposure_cap": exposure_cap,
                "capital_budget": budget,
            }
            if projected_n > exposure_cap or projected_c > budget:
                return detail
            return None
        except (KeyError, IndexError, TypeError, ValueError, OverflowError):
            return "UNKNOWN_EXPOSURE"

    def _write_signed(self, payload: dict, facts: RuntimeFacts | None) -> bool:
        if facts is not None and facts.unavailable_reason:
            self._log(FACTS_UNAVAILABLE, "", {"reason": facts.unavailable_reason})
            return False
        if facts is None or not facts.runtime_config:
            self._log("ATTESTATION_RUNTIME_MISSING", "", {})
            return False
        self._store.save_state(self._integration, payload)
        if self._sign_failed:
            return False
        try:
            write_attestation(
                facts.runtime_config, self._store.state_dir / f"{self._integration}.json"
            )
        except Exception as exc:
            self._sign_failed = True
            self._log("ATTESTATION_SIGN_FAILED", "", {"reason": str(exc)})
            return False
        return True

    def sync_active_file(self, facts: RuntimeFacts | None = None) -> bool:
        """Re-write the integration file when the ledger's active plans are
        not what the file currently shows (e.g. an engine persist() just
        overwrote it). Never creates the file when nothing is active."""
        facts = facts or self._runtime_facts
        now_ms = int(self._clock() * 1000)
        desired = self._desired_plans(now_ms)
        current = self._store.load_state(self._integration)
        if current is not None and current.get("plans") == desired and current.get("attestation"):
            return False
        if not desired and current is None:
            return False
        if not desired:
            payload = self._base_payload(now_ms)
        else:
            payload = self._payload(None, now_ms, None, facts, plans=desired)
        return self._write_signed(payload, facts)

    # ------------------------------------------------------------------ guards
    def _guard_reasons(
        self,
        plan: AIGridPlan,
        *,
        facts: RuntimeFacts | None,
        candles_count: int,
        candle_close_ms: int | None,
        last_close: float | None,
        startup_reconciled: bool,
        preflight_ok: bool | None,
        fill_ledger: OrderFillLedger | None,
        now_ms: int,
        now_dt: datetime,
    ) -> tuple[list[str], Any]:
        """Every guard, evaluated in order; all failures collected.

        Returns (reasons, fills_result) — the fill map is computed once
        here and reused for the payload (P0-5c accounting)."""
        r: list[str] = []
        if facts is None or not facts.runtime_config:
            r.append("ATTESTATION_RUNTIME_MISSING")
            self._log("ATTESTATION_RUNTIME_MISSING", plan.symbol, {})
        trades = facts.open_trades if facts is not None else None

        # --- LIVE_REFUSED (D-093 a/b): refuse any live-looking runtime ------
        if facts is not None and (
            facts.dry_run is False
            or str(facts.trading_mode).lower() == "live"
            or str(facts.runmode).lower() == "live"
        ):
            r.append("LIVE_REFUSED")

        # --- D-102 (1): startup reconciliation must have PASSED -------------
        if not startup_reconciled:
            r.append("STARTUP_RECONCILIATION_REQUIRED")

        # --- D-102 (2): data freshness — >=576 closed candles + fresh tail --
        if candles_count < self.min_closed_candles:
            r.append("INSUFFICIENT_CLOSED_CANDLES")
        if candle_close_ms is not None and (now_ms - candle_close_ms > self.max_staleness_ms):
            r.append("DATA_STALE")

        # --- D-102 (4): clock skew vs exchange server UTC < 5s ---------------
        if (
            facts is not None
            and facts.server_time_ms is not None
            and abs(facts.server_time_ms - now_ms) >= self.max_clock_skew_ms
        ):
            r.append("CLOCK_SKEW")

        # --- D-093 (c): pair must be whitelisted ------------------------------
        if (
            facts is not None
            and facts.pair_whitelist is not None
            and plan.symbol not in set(map(str, facts.pair_whitelist))
        ):
            r.append("PAIR_NOT_IN_WHITELIST")

        # --- D-093 (g): plan TTL still valid ----------------------------------
        if plan.is_expired(now_dt):
            r.append("PLAN_EXPIRED")

        # --- D-102 (2b): regime PASS only — deterministic plans are RANGE ------
        if plan.market_regime != "RANGE" or plan.grid_mode != GridMode.RANGE_GRID:
            r.append("REGIME_NOT_RANGE")

        # --- D-093 (k): SL blueprint must exist --------------------------------
        if not (plan.stop_loss and plan.stop_loss > 0):
            r.append("SL_BLUEPRINT_MISSING")

        # --- D-093 (d): pair not locked / owned --------------------------------
        if self._ownership is not None and self._ownership.state_of(plan.symbol) != OwnerState.FREE:
            r.append("PAIR_LOCKED_OR_OWNED")

        # --- trade-derived guards ----------------------------------------------
        pair_trades = [
            t for t in trades or () if isinstance(t, dict) and t.get("pair") == plan.symbol
        ]
        if trades is not None and pair_trades:
            r.append("PAIR_HAS_OPEN_TRADE")  # owned incl. AIGridStrategy
        if trades is not None:
            pending = 0
            stale_pending = 0
            open_exits = 0
            for t in pair_trades:
                orders = t.get("orders")
                if not isinstance(orders, list):
                    continue
                for o in orders:
                    if not isinstance(o, dict):
                        continue
                    if str(o.get("status")).lower() != "open":
                        continue
                    if o.get("ft_is_entry"):
                        pending += 1
                        if self._order_age_ms(o, now_ms) > self.pending_ttl_ms:
                            stale_pending += 1  # pending beyond TTL (D-102-5)
                    else:
                        open_exits += 1
            if pending:
                r.append("PENDING_ENTRY_EXISTS")
            if stale_pending:
                r.append("PENDING_ORDER_BEYOND_TTL")
            if open_exits:
                r.append("CONFLICTING_EXIT_ORDERS")  # D-100

        # --- D-102 (3): max_open_trades ----------------------------------------
        if (
            facts is not None
            and facts.max_open_trades is not None
            and facts.max_open_trades >= 0
            and trades is not None
            and len(trades) >= facts.max_open_trades
        ):
            r.append("MAX_OPEN_TRADES_REACHED")

        # --- P0-5c-ext / D-100: fill accounting must reconcile ------------------
        fills = None
        if trades is not None:
            tick = self._price_tick_of(plan.symbol) if self._price_tick_of else None
            fills = filled_prices_for_pair(
                trades,
                pair=plan.symbol,
                plan_id=plan.plan_id,
                level_prices=plan.level_prices(),
                price_tick=tick,
                ledger=fill_ledger,
            )
            if fills.unmapped_orders:
                # locks entry/DCA/rearm for the pair; exits/SL untouched
                r.append("PARTIAL_FILL_UNMAPPED")

        # --- D-093 (e): plan must pass the content validator (fail-closed) -----
        if self._validator is None:
            r.append("VALIDATOR_UNAVAILABLE")
        else:
            verdict = self._validator.validate(
                plan,
                last_close=(
                    last_close if last_close else (plan.grid_lower + plan.grid_upper) / 2.0
                ),
                now=now_dt,
            )
            if not verdict.approved:
                r.extend(f"PLAN_REJECTED:{x}" for x in verdict.reasons)

        # --- D-093 (j): exchange must be alive ----------------------------------
        if facts is not None and facts.exchange_healthy is False:
            r.append("EXCHANGE_UNHEALTHY")

        # --- D-093 (h): preflight baseline verified -----------------------------
        if preflight_ok is False:
            r.append("PREFLIGHT_NOT_OK")

        # --- D-102 (6): (plan_id, pair) publishes exactly once -------------------
        if f"{plan.plan_id}|{plan.symbol}" in self._ledger["seen"]:
            r.append("DUPLICATE_PLAN")

        # --- D-093 (f): one active plan per pair (crowded-plan skip) -------------
        if self._active_plan(plan.symbol, now_ms) is not None:
            r.append("ACTIVE_PLAN_EXISTS")

        return r, fills

    # ------------------------------------------------------------------ payload
    def _payload(
        self,
        plan: AIGridPlan | None,
        now_ms: int,
        fills: Any,
        facts: RuntimeFacts | None,
        *,
        plans: dict[str, dict] | None = None,
    ) -> dict:
        """Integration payload — same shape the orchestrator emits so the
        strategy sees one contract (plus the cap fields P0-8 requires)."""
        settings = self._settings
        enabled = bool(settings is not None and settings.enabled and settings.mode == "dryrun")
        payload: dict[str, Any] = self._base_payload(now_ms)
        payload["enabled"] = enabled
        payload["dry_run_verified"] = bool(facts is not None and facts.dry_run is True)
        # merge: every ledger-active plan rides along (one file, all pairs)
        payload["plans"] = dict(plans) if plans is not None else self._desired_plans(now_ms)
        payload["filled_levels"] = {p: [] for p in payload["plans"]}
        if plans is not None:
            self._apply_fills(payload)
            return payload
        assert plan is not None
        pair = plan.symbol
        payload["plans"][pair] = self._plan_dict(plan)
        payload["filled_levels"][pair] = []
        if fills is not None:
            self._active_fills()[pair] = {
                "prices": list(fills.prices),
                "unmapped": int(fills.unmapped_orders),
                "partial_qty": float(fills.partial_qty),
            }
        else:
            self._active_fills()[pair] = None
        self._apply_fills(payload)
        return payload

    def _base_payload(self, now_ms: int) -> dict:
        settings = self._settings
        return {
            "ts_ms": now_ms,
            "enabled": False,
            "dry_run_verified": False,
            "mode": settings.mode if settings is not None else "off",
            "publisher": "deterministic",
            "plan_ttl_s": PLAN_TTL_SECONDS,
            "plans": {},
            "filled_levels": {},
            "filled_levels_source": "virtual",
            "reconciliation_state": "UNVERIFIED",
        }

    def _plan_dict(self, plan: AIGridPlan) -> dict:
        return {
            "plan_id": plan.plan_id,
            "symbol": plan.symbol,
            "grid_mode": str(plan.grid_mode),
            "grid_lower": plan.grid_lower,
            "grid_upper": plan.grid_upper,
            "grid_levels": plan.grid_levels,
            "grid_levels_prices": plan.level_prices(),  # descending
            "grid_spacing": plan.grid_spacing,
            "stop_loss": plan.stop_loss,
            "level_stake": plan.level_stake,
            "max_capital": plan.max_capital,
            "max_position_size": plan.max_position_size,
            "max_leverage": plan.max_leverage,
            "market_regime": plan.market_regime,
            "confidence": plan.confidence,
            "expires_at": plan.expires_at.isoformat(),
            "publisher": "deterministic",
        }

    def _apply_fills(self, payload: dict) -> None:
        """Attach the price-keyed fill book per active pair from the ledger
        records; a pair without verified fills gets NO key (fail-closed)."""
        fills_map = self._active_fills()
        verified_pairs = {p for p in payload["plans"] if isinstance(fills_map.get(p), dict)}
        if verified_pairs:
            payload["filled_level_prices"] = {p: fills_map[p]["prices"] for p in verified_pairs}
            payload["unmapped_orders"] = {p: fills_map[p]["unmapped"] for p in verified_pairs}
            payload["partial_filled_qty"] = {p: fills_map[p]["partial_qty"] for p in verified_pairs}
        if verified_pairs == set(payload["plans"]):
            payload["reconciliation_state"] = "VERIFIED" if payload["plans"] else "UNVERIFIED"
        else:
            payload["reconciliation_state"] = "UNVERIFIED"

    # ------------------------------------------------------------------ ledger
    def _load_ledger(self) -> dict:
        data = self._store.load_state(self._ledger_name) or {}
        return {
            "seen": dict(data.get("seen") or {}),
            "active": dict(data.get("active") or {}),
            "fills": dict(data.get("fills") or {}),
        }

    def _save_ledger(self) -> None:
        self._store.save_state(self._ledger_name, self._ledger)

    def _record_publish(self, plan: AIGridPlan, payload: dict, event_id: str, now_ms: int) -> None:
        key = f"{plan.plan_id}|{plan.symbol}"
        self._ledger["seen"][key] = {"published_ms": now_ms, "event_id": event_id}
        self._ledger["active"][plan.symbol] = {
            "plan_id": plan.plan_id,
            "expires_at_ms": int(plan.expires_at.timestamp() * 1000),
            "plan": payload["plans"][plan.symbol],
        }
        self._save_ledger()

    def _active_fills(self) -> dict:
        return self._ledger["fills"]

    def _active_plan(self, pair: str, now_ms: int) -> dict | None:
        rec = self._ledger["active"].get(pair)
        if rec is None:
            return None
        if now_ms >= int(rec.get("expires_at_ms") or 0):
            del self._ledger["active"][pair]
            self._ledger["fills"].pop(pair, None)
            self._save_ledger()
            return None
        return rec

    def _desired_plans(self, now_ms: int) -> dict:
        out = {}
        for pair in list(self._ledger["active"]):
            rec = self._active_plan(pair, now_ms)
            if rec is not None:
                out[pair] = rec["plan"]
        return out

    # ------------------------------------------------------------------ misc
    def _order_age_ms(self, order: dict, now_ms: int) -> float:
        for key in ("order_timestamp", "filled_timestamp"):
            v = order.get(key)
            if isinstance(v, (int, float)) and v > 0:
                return max(0.0, now_ms - (v * 1000 if v < 1e12 else v))
        for key in ("order_date", "created_at", "order_filled_date"):
            v = order.get(key)
            if isinstance(v, str) and v.strip():
                dt = _parse_iso(v)
                if dt is not None:
                    return max(0.0, now_ms - dt.timestamp() * 1000)
        return 0.0  # unknown age -> treated as fresh pending (still blocks)

    def _event_id(self, plan: AIGridPlan, now_ms: int) -> str:
        raw = f"{plan.plan_id}|{plan.symbol}|{now_ms}"
        return f"pub-{hashlib.sha256(raw.encode()).hexdigest()[:10]}"

    def _log(self, kind: str, symbol: str, payload: dict) -> None:
        if self._eventlog is not None:
            self._eventlog.log(kind, symbol, payload)
        logger.info("%s %s %s", kind, symbol, payload)


def _parse_iso(value: str) -> datetime | None:
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


# ====================================================================== REST
class FtRestClient:
    """Read-only Freqtrade REST client — GETs only, never a write call.

    The ONLY network permitted on the deterministic publish path is the
    local Freqtrade API (show_config / status / health / balance / pair_candles).
    Basic-auth from the same env vars preflight uses.
    """

    def __init__(
        self,
        base_url: str,
        *,
        username: str | None = None,
        password: str | None = None,
        timeout_s: float = 5.0,
        opener=None,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self._username = username
        self._password = password
        self.timeout_s = float(timeout_s)
        self._opener = (
            opener
            if opener is not None
            else urllib.request.build_opener(urllib.request.ProxyHandler({}))
        )

    @classmethod
    def from_env(cls, timeout_s: float = 5.0) -> FtRestClient:
        return cls(
            os.environ.get(FT_API_URL_ENV, DEFAULT_API_URL),
            username=os.environ.get(FT_API_USER_ENV),
            password=os.environ.get(FT_API_PASS_ENV),
            timeout_s=timeout_s,
        )

    def _get(self, path: str, params: dict | None = None) -> Any:
        url = f"{self.base_url}{path}"
        if params:
            from urllib.parse import urlencode

            url = f"{url}?{urlencode(params)}"
        req = urllib.request.Request(url)
        if self._username:
            token = base64.b64encode(f"{self._username}:{self._password or ''}".encode()).decode()
            req.add_header("Authorization", f"Basic {token}")
        with self._opener.open(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # ------------------------------------------------------------- endpoints
    def show_config(self) -> dict | None:
        try:
            cfg = self._get("/api/v1/show_config")
            return cfg if isinstance(cfg, dict) else None
        except Exception as exc:
            logger.warning("FT show_config failed: %s", exc)
            return None

    def whitelist(self) -> list[str] | None:
        """Return the runtime whitelist, or None when unavailable or malformed."""
        try:
            data = self._get("/api/v1/whitelist")
        except Exception:
            return None
        pairs = data.get("whitelist") if isinstance(data, dict) else None
        return (
            pairs
            if isinstance(pairs, list) and all(isinstance(pair, str) for pair in pairs)
            else None
        )

    def status(self) -> list[dict] | None:
        try:
            data = self._get("/api/v1/status")
            return data if isinstance(data, list) else None
        except Exception as exc:
            logger.warning("FT /status failed: %s", exc)
            return None

    def health(self) -> dict | None:
        try:
            data = self._get("/api/v1/health")
            return data if isinstance(data, dict) else None
        except Exception as exc:
            logger.warning("FT /health failed: %s", exc)
            return None

    def server_time_ms(self) -> int | None:
        """Exchange-side UTC from /health (last_process_ts / last_process);
        None when unparseable — the skew gate skips on absent data."""
        health = self.health()
        if not isinstance(health, dict):
            return None
        ts = health.get("last_process_ts")
        if isinstance(ts, (int, float)) and ts > 0:
            return int(ts * 1000 if ts < 1e12 else ts)
        dt = _parse_iso(str(health.get("last_process") or ""))
        return int(dt.timestamp() * 1000) if dt else None

    def pair_candles(
        self, pair: str, timeframe: str = "5m", limit: int = 600
    ) -> pd.DataFrame | None:
        """GET /api/v1/pair_candles -> closed candles frame, or None when
        the endpoint is absent (older FT) — the caller fails closed."""
        try:
            data = self._get(
                "/api/v1/pair_candles",
                {"pair": pair, "timeframe": timeframe, "limit": limit},
            )
        except Exception as exc:
            logger.warning("FT pair_candles failed: %s", exc)
            return None
        return _candles_to_df(data)

    # ------------------------------------------------------------- facts
    def facts(self) -> RuntimeFacts:
        """One bounded read of the runtime — the only facts the guards use."""
        cfg = dict(self.show_config() or {})
        trades = self.status()
        equity = None
        try:
            balance = self._get("/api/v1/balance")
            equity = balance.get("total") if isinstance(balance, dict) else None
        except Exception as exc:
            logger.warning("FT /balance failed: %s", exc)
        health = self.health()
        whitelist = cfg.get("pair_whitelist")
        if whitelist is None and isinstance(cfg.get("exchange"), dict):
            whitelist = cfg["exchange"].get("pair_whitelist")
        if not isinstance(whitelist, list) or not all(isinstance(p, str) for p in whitelist):
            whitelist = self.whitelist()
            if whitelist is None:
                return RuntimeFacts(
                    runtime_config=cfg, unavailable_reason="RUNTIME_WHITELIST_UNAVAILABLE"
                )
            cfg["exchange"] = {"pair_whitelist": list(whitelist)}
        elif "pair_whitelist" not in cfg:
            cfg["exchange"] = {**cfg["exchange"], "pair_whitelist": list(whitelist)}
        return RuntimeFacts(
            runtime_config=cfg,
            dry_run=cfg.get("dry_run"),
            trading_mode=cfg.get("trading_mode"),
            runmode=cfg.get("runmode"),
            pair_whitelist=(list(whitelist) if isinstance(whitelist, (list, tuple)) else None),
            max_open_trades=cfg.get("max_open_trades"),
            open_trades=trades,
            equity_usdt=equity,
            exchange_healthy=(health is not None),
            server_time_ms=self._health_time(health),
        )

    def _health_time(self, health: dict | None) -> int | None:
        if not isinstance(health, dict):
            return None
        ts = health.get("last_process_ts")
        if isinstance(ts, (int, float)) and ts > 0:
            return int(ts * 1000 if ts < 1e12 else ts)
        dt = _parse_iso(str(health.get("last_process") or ""))
        return int(dt.timestamp() * 1000) if dt else None


def _candles_to_df(data: Any) -> pd.DataFrame | None:
    """FT pair_candles {columns, data} -> OHLCV frame indexed by open time
    (UTC). None on any shape deviation — fail closed."""
    if not isinstance(data, dict):
        return None
    cols = data.get("columns")
    rows = data.get("data")
    if not isinstance(cols, list) or not isinstance(rows, list) or not rows:
        return None
    try:
        df = pd.DataFrame(rows, columns=cols)
        date_col = "date" if "date" in df.columns else cols[0]
        df["date"] = pd.to_datetime(df[date_col], utc=True)
        df = df.set_index("date").sort_index()
        if not {"open", "high", "low", "close", "volume"}.issubset(df.columns):
            return None
        return df[["open", "high", "low", "close", "volume"]]
    except (ValueError, KeyError, TypeError):
        return None


# ====================================================================== cycle
class DeterministicCycle:
    """The R14 dry-run chain, driven once per service tick (<=30s budget):

    for each whitelisted pair with a NEW closed candle:
        process_candle -> confirmed regime -> deterministic plan ->
        guarded atomic publish. After the pair pass, the integration file
        is re-synced so a persist() clobber can't strand an active plan.
    """

    def __init__(
        self,
        orchestrator: Any,
        publisher: PlanPublisher,
        *,
        candle_loader: Callable[[str], pd.DataFrame | None] | None = None,
        facts_provider: Callable[[], RuntimeFacts] | None = None,
        store: StateStore | None = None,
        eventlog: Any = None,
        whitelist: Sequence[str] | None = None,
        timeframe_s: int = 300,
        ai_status: str = "UNAVAILABLE",
        cycle_state: str = "publish_cycle",
    ):
        self.orch = orchestrator
        self.publisher = publisher
        self._loader = candle_loader
        self._facts_provider = facts_provider
        self._store = store or getattr(orchestrator, "store", None)
        self._eventlog = (
            eventlog if eventlog is not None else getattr(orchestrator, "eventlog", None)
        )
        settings = getattr(orchestrator, "settings", None)
        self._settings = settings
        self._wl = tuple(
            whitelist or getattr(settings, "watchlist", None) or getattr(settings, "symbols", ())
        )
        self._timeframe_s = timeframe_s
        self._ai_status = ai_status
        self._cycle_state = cycle_state
        self._fill_ledger = OrderFillLedger()
        self._seen: dict[str, int] = {}
        self._processed: dict[str, int] = {}
        self._reconciled = False
        self._load_state()

    # --------------------------------------------------------------- startup
    def startup_reconcile(
        self,
        *,
        actual_open_pairs: set[str] | None = None,
        actual_open_order_ids: set[str] | None = None,
        actual_open_lots: dict[str, int] | None = None,
    ) -> dict:
        """D-102 (1): reconcile persisted state BEFORE any publish is
        allowed. PASS (RESTORED/FRESH_START) arms the cycle; SAFE_HOLD or
        an unreadable state leaves the gate closed."""
        if actual_open_pairs is None:
            actual_open_pairs, actual_open_order_ids, actual_open_lots = self._actuals_from_facts()
        result: dict
        recover = getattr(self.orch, "recover", None)
        if callable(recover):
            result = recover(
                actual_open_pairs or set(),
                actual_open_order_ids or set(),
                actual_open_lots,
            )
        else:
            result = {"status": "FRESH_START", "reasons": []}
        self._reconciled = result.get("status") in ("RESTORED", "FRESH_START")
        self._log(
            "STARTUP_RECONCILIATION_PASS" if self._reconciled else "STARTUP_RECONCILIATION_FAILED",
            "",
            {
                "status": result.get("status"),
                "reasons": list(result.get("reasons") or []),
            },
        )
        self._save_state()
        return result

    @property
    def reconciled(self) -> bool:
        return self._reconciled

    @property
    def seen_ts(self) -> dict[str, int]:
        return dict(self._seen)

    @property
    def processed_ts(self) -> dict[str, int]:
        """Newest closed candles observed this pass, excluding persisted history."""
        return dict(self._processed)

    # --------------------------------------------------------------- tick
    def tick(self, cycle_no: int = 0) -> list[PublishOutcome]:
        outcomes: list[PublishOutcome] = []
        self.publisher._sign_failed = False
        self._processed = {}
        try:
            facts = self._facts_provider() if self._facts_provider else None
        except Exception:
            facts = None
        if facts is None or not facts.runtime_config or facts.unavailable_reason:
            self._log(
                FACTS_UNAVAILABLE,
                "",
                {
                    "reason": facts.unavailable_reason
                    if facts is not None
                    else "RUNTIME_FACTS_UNAVAILABLE",
                },
            )
            # process_candle may persist the handoff; without signing evidence
            # leave both the prior handoff and candle markers untouched.
            if self._facts_provider is not None:
                return outcomes
        for pair in self._wl:
            df = self._load_candles(pair)
            if df is None or len(df) == 0:
                continue
            now_ms = int(self.publisher._clock() * 1000)
            df = df.loc[
                (df.index.as_unit("ns").asi8 // 1_000_000 + self._timeframe_s * 1000) <= now_ms
            ]
            if df.empty:
                continue
            self._processed[pair] = int(df.index[-1].value // 1_000_000)
            last_seen = self._seen.get(pair, -1)
            new_idx = [i for i, ts in enumerate(df.index) if int(ts.value // 1_000_000) > last_seen]
            if not new_idx:
                continue
            # process_candle once per NEW closed candle, oldest first; the
            # history slice is causal (no lookahead past that candle).
            ok = True
            for i in new_idx:
                open_ms = int(df.index[i].value // 1_000_000)
                row = df.iloc[i]
                candle = {
                    "open_ms": open_ms,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
                if not self._process(pair, candle, df.iloc[: i + 1]):
                    ok = False
                    break
                self._seen[pair] = open_ms
                self._save_state()
            if not ok:
                continue  # fail-closed: no plan on a half-processed backlog
            last = df.iloc[-1]
            open_ms = int(df.index[-1].value // 1_000_000)
            regime, paused, confidence = self._regime(pair, df)
            close_dt = datetime.fromtimestamp((open_ms + self._timeframe_s * 1000) / 1000.0, tz=UTC)
            plan = build_deterministic_plan(
                pair,
                df,
                regime=regime,
                paused=paused,
                ai_status=self._ai_status,
                confidence=confidence,
                settings=self._settings,
                as_of=close_dt,
                price_tick=self._price_tick(pair),
                eventlog=self._eventlog,
            )
            if plan is None:
                continue
            outcomes.append(
                self.publisher.publish(
                    plan,
                    facts=facts,
                    candles_count=len(df),
                    candle_close_ms=open_ms + self._timeframe_s * 1000,
                    last_close=float(last["close"]),
                    startup_reconciled=self._reconciled,
                    preflight_ok=getattr(self.orch, "preflight_ok", None),
                    fill_ledger=self._fill_ledger,
                )
            )
        # a persist() inside process_candle may have clobbered the file —
        # restore the ledger's active plans so the file is never stale.
        if facts is not None and facts.runtime_config:
            self.publisher.sync_active_file(facts)
        return outcomes

    # --------------------------------------------------------------- internals
    def _load_candles(self, pair: str) -> pd.DataFrame | None:
        try:
            if self._loader is not None:
                return self._loader(pair)
            adapter = getattr(self.orch, "adapter", None)
            if adapter is not None:
                return adapter.get_ohlcv(
                    pair,
                    getattr(self._settings, "timeframe", "5m"),
                    as_of=datetime.now(UTC),
                    lookback=self.publisher.min_closed_candles + 30,
                )
        except Exception as exc:
            self._log("DATA_UNAVAILABLE", pair, {"error": f"{type(exc).__name__}:{exc}"})
            return None
        self._log("DATA_UNAVAILABLE", pair, {"reason": "no_candle_source"})
        return None

    def _process(self, pair: str, candle: dict, df: pd.DataFrame) -> bool:
        fn = getattr(self.orch, "process_candle", None)
        if not callable(fn):
            return True
        try:
            fn(pair, candle, history_df=df)
            return True
        except Exception as exc:
            self._log(
                "CANDLE_PROCESS_FAILED",
                pair,
                {"error": f"{type(exc).__name__}:{exc}"},
            )
            return False

    def _regime(self, pair: str, df: pd.DataFrame) -> tuple[str, bool, float]:
        controller_for = getattr(self.orch, "_controller_for", None)
        rr = classify(df)
        if callable(controller_for):
            try:
                ctrl = controller_for(pair)
                return str(ctrl.confirmed), bool(ctrl.paused), rr.confidence
            except Exception as exc:
                # must never un-pause the gate: fall back fail-CLOSED.
                logger.warning("controller read failed for %s: %s", pair, exc)
                return rr.regime, True, rr.confidence
        return rr.regime, False, rr.confidence

    def _price_tick(self, pair: str) -> float | None:
        fn = getattr(self.orch, "_price_tick", None)
        if callable(fn):
            try:
                tick = fn(pair)
                return float(tick) if tick else None
            except Exception:
                return None
        return None

    def _actuals_from_facts(self):
        facts = self._facts_provider() if self._facts_provider else None
        pairs, order_ids, lots = set(), set(), {}
        for t in facts.open_trades or ():
            if not isinstance(t, dict) or not t.get("is_open", True):
                continue
            short = str(t.get("pair") or "").split(":", 1)[0].replace("/", "")
            if short:
                pairs.add(short)
            orders = t.get("orders")
            if isinstance(orders, list):
                n_entry = 0
                for o in orders:
                    if not isinstance(o, dict):
                        continue
                    oid = o.get("order_id") or o.get("id")
                    if oid:
                        order_ids.add(str(oid))
                    if o.get("ft_is_entry") and str(o.get("status")).lower() in (
                        "open",
                        "closed",
                    ):
                        n_entry += 1
                if short and n_entry:
                    lots[short] = n_entry
        return pairs, order_ids, lots

    # --------------------------------------------------------------- state
    def _load_state(self) -> None:
        if self._store is None:
            return
        data = self._store.load_state(self._cycle_state) or {}
        self._seen = {str(k): int(v) for k, v in (data.get("seen") or {}).items()}
        self._reconciled = bool(data.get("reconciled", False))

    def _save_state(self) -> None:
        if self._store is None:
            return
        self._store.save_state(
            self._cycle_state,
            {"seen": self._seen, "reconciled": self._reconciled},
        )

    def _log(self, kind: str, symbol: str, payload: dict) -> None:
        if self._eventlog is not None:
            self._eventlog.log(kind, symbol, payload)
        logger.info("%s %s %s", kind, symbol, payload)
