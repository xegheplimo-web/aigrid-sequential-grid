"""AIGridStrategy — Freqtrade integration layer for the Adaptive AI Grid.

Runs INSIDE Freqtrade (dry-run only). It is the ONLY execution path: the
orchestrator never talks to an exchange; it writes an integration file that
this strategy consumes.

Design (Sequential Grid via DCA — see ARCHITECTURE.md §5):
- Freqtrade manages ONE trade per pair (its DCA model). The virtual grid book
  lives in the orchestrator; this strategy realizes its actions sequentially:
    * entry signal  -> when price crosses the next grid level (from the plan)
    * custom_entry_price -> the exact grid level price
    * adjust_trade_position -> adds the next level when price crosses it,
      and returns a NEGATIVE stake to take profit on a single level (R13)
    * custom_exit -> whole-trade exit once EVERY live lot reached its TP
    * custom_stoploss -> plan stop loss
- ALL behavior is OFF unless the integration file says enabled=true AND
  dry-run is verified. With the file absent/corrupt -> pass-through no-op.

Safety:
- No live-order code. No exchange API calls. Reads one JSON file per loop.
- Stale integration file (> 2x plan TTL) -> trading disabled (fail-closed).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, stoploss_from_absolute
from pandas import DataFrame

try:  # Repo checkout: prefer the single source of truth.
    from aigrid.preflight import canonical_json as _canonical_json
    from aigrid.preflight import config_hash as _config_hash
except ImportError:  # Dry-run container mounts only user_data/.
    _canonical_json = None
    _config_hash = None


def _local_canonical_json(value: dict) -> bytes:
    """Keep byte-identical to aigrid.preflight.canonical_json."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _local_config_hash(config: dict) -> str:
    """Mirror preflight's six fields, normalization and sorted whitelist."""
    fields = {
        key: config[key]
        for key in (
            "strategy",
            "dry_run",
            "trading_mode",
            "stake_amount",
            "max_open_trades",
        )
    }
    for key in ("stake_amount", "max_open_trades"):
        try:
            fields[key] = float(fields[key])
        except (ValueError, TypeError):
            pass
    exchange = config.get("exchange")
    pairs = exchange.get("pair_whitelist") if isinstance(exchange, dict) else None
    if pairs is None:
        pairs = config["pair_whitelist"]
    if not isinstance(pairs, list) or not all(isinstance(p, str) for p in pairs):
        raise ValueError("invalid runtime whitelist")
    fields["exchange"] = {"pair_whitelist": sorted(pairs)}
    return hashlib.sha256(_local_canonical_json(fields)).hexdigest()


def _canonical(value: dict) -> bytes:
    return (_canonical_json or _local_canonical_json)(value)


def _hash(config: dict) -> str:
    return (_config_hash or _local_config_hash)(config)


logger = logging.getLogger(__name__)

INTEGRATION_FILE = Path(
    os.environ.get("AIGRID_INTEGRATION_FILE", "user_data/aigrid_state/ft_integration.json")
)
SL_CACHE_FILE = Path(
    os.environ.get("AIGRID_SL_CACHE_FILE", "user_data/aigrid_state/strategy_sl_cache.json")
)
MAX_FILE_AGE_S = 2 * 600  # integration file must be fresher than 2x the 10m plan TTL
PLAN_EXPIRED_LOG_S = 60.0  # rate-limit for PLAN_EXPIRED_BLOCKED logging
CAPITAL_CAP_LOG_S = 60.0  # rate-limit for CAPITAL_CAP_BLOCKED logging
SAFE_DEFAULT_SL_PCT = -0.10  # hard floor when neither plan nor cache is available
REQUIRED_TIMEFRAME = "5m"  # approved runtime timeframe — see bot_start()
LIQ_BUFFER_PCT = 0.03  # plan SL must sit >= 3% of liq price off liquidation
# P0-9 boundary (verified against Freqtrade 2026.10 strategy API): the only
# supported way for a strategy to cancel an UNFILLED entry order is
# check_entry_timeout()/adjust_order_price — and only for orders on its OWN
# trades. APIs checked that can NOT cancel: confirm_trade_entry (veto at
# placement time only, mode="initial" per freqtradebot.execute_entry),
# adjust_trade_position (returns stake for new orders, never cancels),
# dp.send_msg (notifications only), self.wallets (balances only). Cancelling
# an order belonging to ANOTHER bot's trade is still impossible strategy-side
# — that cleanup belongs to the orchestrator via the REST API (later card).
PENDING_CANCEL_UNSUPPORTED = False


def _parse_expires_ms(value) -> int | None:
    """ISO-8601 `expires_at` -> epoch ms; a naive datetime is read as UTC
    (same convention as AIGridPlan.is_expired). None when absent or
    unparseable — never raises."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _load_integration() -> dict:
    """Read the integration file; {} when absent/corrupt/stale (fail-closed)."""
    try:
        if not INTEGRATION_FILE.exists():
            return {}
        age = time.time() - INTEGRATION_FILE.stat().st_mtime
        if age > MAX_FILE_AGE_S:
            return {"enabled": False, "reason": "INTEGRATION_FILE_STALE"}
        data = json.loads(INTEGRATION_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return data
    except (OSError, json.JSONDecodeError):
        return {"enabled": False, "reason": "INTEGRATION_FILE_UNREADABLE"}


def _load_sl_cache() -> dict:
    """Persisted per-pair plan SL PRICES (absolute); {} on any failure
    (fail-closed). Non-positive values are legacy open_rate ratios — dropped:
    an absolute stop must never be re-derived from a moved open_rate."""
    try:
        if not SL_CACHE_FILE.exists():
            return {}
        data = json.loads(SL_CACHE_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return {
            str(k): float(v)
            for k, v in data.items()
            if isinstance(v, (int, float)) and math.isfinite(v) and v > 0
        }
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def _atomic_write_json(path: Path, payload: dict) -> None:
    """tmp-file + os.replace + fsync — a crash must not corrupt the cache."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=1)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            Path(tmp).unlink(missing_ok=True)
    except OSError:
        logger.warning("AIGrid: could not persist SL cache to %s", path)


def resolve_stoploss_pct(
    plan: dict, cached_pct: float | None, is_short: bool, open_rate: float
) -> float:
    """SL as a price ratio in (-0.99, -0.01]. Never None, never -0.99.

    Priority: a valid plan `stop_loss` -> distance from `open_rate`; else the
    last cached plan SL; else SAFE_DEFAULT_SL_PCT (-10% hard floor). A
    degenerate plan SL (>= 99% away) is treated as absent — falling back to
    the cache/default is strictly more protective.
    """
    sl = plan.get("stop_loss") if isinstance(plan, dict) else None
    if isinstance(sl, (int, float)) and math.isfinite(sl) and sl > 0 and open_rate > 0:
        sl_pct = (1.0 - sl / open_rate) if is_short else (sl / open_rate - 1.0)
        if sl_pct > -0.99:
            return min(-0.01, sl_pct)
    if (
        isinstance(cached_pct, (int, float))
        and math.isfinite(cached_pct)
        and -0.99 < cached_pct < 0
    ):
        return min(-0.01, float(cached_pct))
    return SAFE_DEFAULT_SL_PCT


def protective_stop_error(
    stop_rate: float, current_rate: float, is_short: bool, liq_price
) -> str | None:
    """None if the absolute stop is a valid protective stop, else the reason.

    Long:  liq*(1+buffer) <= stop < current.   Short: mirrored above.
    A stop on the wrong side of price, non-finite, or sitting inside the
    liquidation buffer must be REJECTED (SL_INVALID), never clamped into a
    bad-but-valid plan (D-092 §e).
    """
    if not (isinstance(stop_rate, (int, float)) and math.isfinite(stop_rate) and stop_rate > 0):
        return f"invalid stop_rate {stop_rate!r}"
    if not (math.isfinite(current_rate) and current_rate > 0):
        return f"invalid current_rate {current_rate!r}"
    if is_short:
        if stop_rate <= current_rate:
            return f"short stop {stop_rate} not above current {current_rate}"
    elif stop_rate >= current_rate:
        return f"long stop {stop_rate} not below current {current_rate}"
    if isinstance(liq_price, (int, float)) and math.isfinite(liq_price) and liq_price > 0:
        if is_short and stop_rate > liq_price * (1.0 - LIQ_BUFFER_PCT):
            return f"short stop {stop_rate} within {LIQ_BUFFER_PCT:.0%} of liquidation {liq_price}"
        if not is_short and stop_rate < liq_price * (1.0 + LIQ_BUFFER_PCT):
            return f"long stop {stop_rate} within {LIQ_BUFFER_PCT:.0%} of liquidation {liq_price}"
    return None


class AIGridStrategy(IStrategy):
    """Grid execution layer driven by a validated AIGridPlan (dry-run only)."""

    INTERFACE_VERSION = 3
    timeframe = REQUIRED_TIMEFRAME  # 5m — bot_start() halts entries on drift
    can_short = True
    process_only_new_candles = True
    startup_candle_count = 2016  # Seven days of 5m candles for ATR percentiles.
    stoploss = -0.10  # hard floor; plan SL enforced via custom_stoploss
    use_exit_signal = True
    exit_profit_only = False
    position_adjustment_enable = True
    use_custom_stoploss = True  # plan SL enforced via custom_stoploss below

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aigrid: dict = {}
        self._plan: dict = {}
        self._levels: list[float] = []
        self._next_entry_level: int | None = None
        self._sl_cache: dict = _load_sl_cache()
        self._last_sl_pct: float | None = None
        # T-223c: set by bot_start() when the effective timeframe != 5m —
        # blocks every NEW order path while exits/SL keep running.
        self._halted: bool = False
        # T-223b: pair -> rejected absolute stop_rate; blocks new plans for
        # the symbol until a DIFFERENT plan stop arrives (no silent clamp).
        self._sl_blocked: dict[str, float] = {}
        self._sl_invalid_logged: set[tuple[str, float]] = set()
        # R10: plan TTL — parsed from plan.expires_at in _refresh_integration;
        # an expired plan disables _enabled() (new orders), never the SL path.
        self._plan_expires_ms: int | None = None
        self._expired_log_ts: float = 0.0
        # R11: rate-limit for CAPITAL_CAP_BLOCKED logging (P0-8)
        self._cap_log_ts: float = 0.0
        # R10: pair -> last CLOSED candle {open,high,low,close,prev_close},
        # captured in populate_indicators for engine crossing semantics.
        self._last_closed: dict[str, dict] = {}
        # R13: per-level TP reduce bookkeeping (all lazily re-created for
        # bare instances too — tests build via object.__new__):
        #   _exit_orders: pair -> {level_price: {"tag","confirmed"}}
        #   _exit_seq:    plan_id -> last used :exit: sequence number
        #   _tp_since:    pair -> {level_price: first-at-TP monotonic ts}
        self._exit_orders: dict = {}
        self._exit_seq: dict = {}
        self._tp_since: dict = {}

    # ------------------------------------------------------------------ bot start
    def bot_start(self, **kwargs) -> None:
        """Fail-closed timeframe check (T-223c) + trailing warning (T-223b).

        FT resolves config["timeframe"] over the class attribute; losing the
        5m override once made the bot silently trade 15m. On mismatch we log
        and set _halted — new orders are refused but open positions keep
        being managed (no raise; the bot must not die)."""
        cfg = getattr(self, "config", None)
        effective = self.timeframe
        if isinstance(cfg, dict):
            effective = cfg.get("timeframe", effective)
        if str(effective) != REQUIRED_TIMEFRAME:
            self._halted = True
            logger.error(
                "AIGrid: effective timeframe %r != required %r — HALTED: no "
                "new orders will be created (open positions still managed)",
                effective,
                REQUIRED_TIMEFRAME,
            )
        else:
            self._halted = False
        # FT only ever TIGHTENS a custom_stoploss — a configured trailing
        # stop or a tighter initial stop cannot be relaxed back to the plan.
        ts = getattr(self, "trailing_stop", False)
        tsp = getattr(self, "trailing_stop_positive", None)
        if isinstance(cfg, dict):
            ts = cfg.get("trailing_stop", ts)
            tsp = cfg.get("trailing_stop_positive", tsp)
        if ts or tsp is not None:
            logger.warning(
                "AIGrid: trailing_stop=%s trailing_stop_positive=%s — FT "
                "only tightens custom_stoploss; a wider plan SL will NOT "
                "relax an already-tight stop",
                ts,
                tsp,
            )

    # ------------------------------------------------------------------ helpers
    def _entries_halted(self) -> bool:
        return bool(getattr(self, "_halted", False))

    def _blocked_pairs(self) -> dict:
        return self.__dict__.setdefault("_sl_blocked", {})

    def _block_pair_sl(self, pair: str, stop_rate: float, reason: str) -> None:
        """Record SL_INVALID and block new plans/entries for the symbol."""
        self._blocked_pairs()[pair] = stop_rate
        logged = self.__dict__.setdefault("_sl_invalid_logged", set())
        if (pair, stop_rate) not in logged:
            logged.add((pair, stop_rate))
            logger.error(
                "SL_INVALID pair=%s stop_rate=%s reason=%s — new plans/"
                "entries for this symbol are blocked until a valid plan "
                "arrives",
                pair,
                stop_rate,
                reason,
            )

    def _resolve_stop_price(self, plan: dict, pair: str, trade: Trade) -> float | None:
        """Absolute stop price: plan.stop_loss -> last cached plan SL price
        -> open_rate anchored SAFE_DEFAULT_SL_PCT fallback. None if no
        price can be derived at all (static -10% floor still applies)."""
        sl = plan.get("stop_loss") if isinstance(plan, dict) else None
        if isinstance(sl, (int, float)) and math.isfinite(sl) and sl > 0:
            return float(sl)
        cached = self._sl_cache.get(pair)
        if isinstance(cached, (int, float)) and math.isfinite(cached) and cached > 0:
            return float(cached)
        open_rate = float(getattr(trade, "open_rate", 0.0) or 0.0)
        if open_rate > 0:
            if getattr(trade, "is_short", False):
                return open_rate * (1.0 - SAFE_DEFAULT_SL_PCT)
            return open_rate * (1.0 + SAFE_DEFAULT_SL_PCT)
        return None

    def _refresh_integration(self, pair: str | None = None, *, trade: Trade | None = None) -> None:
        self._aigrid = _load_integration()
        # per-pair plans: {"plans": {"ETH/USDT:USDT": {...}, ...}};
        # legacy single "plan" key still accepted
        plans = self._aigrid.get("plans")
        if isinstance(plans, dict) and pair is not None:
            plan = plans.get(pair) or {}
        else:
            plan = self._aigrid.get("plan") or {}
        self._plan = plan if self._aigrid.get("enabled") and plan else {}
        self._plan_expires_ms = _parse_expires_ms(self._plan.get("expires_at"))
        levels = self._plan.get("grid_levels_prices") or []
        self._levels = sorted(float(x) for x in levels) if levels else []
        blocked = self._blocked_pairs()
        if pair in blocked:
            # a DIFFERENT plan stop gets a fresh evaluation; the same bad
            # value stays blocked (no silent re-admission).
            new_sl = self._plan.get("stop_loss")
            if not (
                isinstance(new_sl, (int, float))
                and math.isfinite(new_sl)
                and float(new_sl) == blocked[pair]
            ):
                blocked.pop(pair)
        if self._plan:
            self._remember_sl(pair, trade)
        elif not self._sl_cache:
            self._sl_cache = _load_sl_cache()

    def _remember_sl(self, pair: str | None, trade: Trade | None) -> None:
        """Persist the plan's ABSOLUTE SL price per pair (in-memory + disk)
        so a dead or stalled orchestrator cannot reopen the -99% stoploss
        hole. Ratios are not stored: FT anchors at current_rate, so only the
        absolute price survives a moving open_rate."""
        if pair is None or trade is None:
            return
        sl = self._plan.get("stop_loss")
        if not (isinstance(sl, (int, float)) and math.isfinite(sl) and sl > 0):
            return
        self._last_sl_pct = float(sl)
        if self._sl_cache.get(pair) != float(sl):
            self._sl_cache[pair] = float(sl)
            _atomic_write_json(SL_CACHE_FILE, self._sl_cache)

    def _runtime_dryrun_checked(self) -> tuple[bool, str]:
        """Layer 1: consult the strategy's own effective runtime, never a file."""
        cfg = getattr(self, "config", None)
        # Include presence/type: True and 1 must never share a cached result.
        key = repr(
            (
                type(cfg),
                [
                    (name, type(cfg.get(name)), cfg.get(name))
                    for name in ("dry_run", "trading_mode", "runmode")
                ]
                if isinstance(cfg, dict)
                else None,
            )
        )
        cached = getattr(self, "_runtime_dryrun_cache", None)
        now = time.monotonic()
        if cached and cached[0] == key and now - cached[1] < 60:
            return cached[2]
        result = (True, "")
        if not isinstance(cfg, dict):
            result = (False, "NO_CONFIG")
        elif cfg.get("dry_run") is not True:
            result = (False, "RUNTIME_NOT_DRY")
        elif str(cfg.get("trading_mode", "")).lower() == "live":
            result = (False, "RUNTIME_LIVE_MODE")
        elif str(cfg.get("runmode", "")).lower() == "live":
            result = (False, "RUNTIME_LIVE_RUNMODE")
        self._runtime_dryrun_cache = (key, now, result)
        return result

    def _attestation_checked(self, data: dict) -> tuple[bool, str]:
        """Layer 2: fresh config-bound evidence signed with a local secret."""
        block = data.get("attestation")
        required = {"ts_ms", "nonce", "container_id", "config_hash", "sig"}
        if not isinstance(block, dict) or not required.issubset(block):
            return False, "ATTESTATION_MISSING"
        ts = block["ts_ms"]
        if type(ts) is not int or not -60_000 <= time.time() * 1000 - ts <= 1_800_000:
            return False, "ATTESTATION_STALE"
        try:
            digest = _hash(self.config)
        except (KeyError, TypeError, ValueError, OverflowError):
            return False, "ATTESTATION_CONFIG_MISMATCH"
        if block["config_hash"] != digest:
            return False, "ATTESTATION_CONFIG_MISMATCH"
        try:
            secret = (INTEGRATION_FILE.parent / ".attest_secret").read_bytes()
            if len(secret) != 32:
                return False, "ATTESTATION_NO_SECRET"
        except OSError:
            return False, "ATTESTATION_NO_SECRET"
        fields = {key: block[key] for key in required - {"sig"}}
        try:
            expected = hmac.new(secret, _canonical(fields), hashlib.sha256).hexdigest()
            valid = isinstance(block["sig"], str) and hmac.compare_digest(block["sig"], expected)
        except (TypeError, ValueError):
            valid = False
        return (True, "") if valid else (False, "ATTESTATION_BAD_SIG")

    def _dryrun_proven(self) -> bool:
        ok, reason = self._runtime_dryrun_checked()
        if ok:
            # Read the file each time: deleting/replacing evidence revokes it
            # immediately; only Layer 1's runtime result is cached.
            try:
                data = _load_integration()
            except (OSError, ValueError, TypeError):
                data = {}
            ok, reason = self._attestation_checked(data)
        self._dryrun_reason = reason
        if not ok:
            now = time.monotonic()
            last = getattr(self, "_dryrun_log_ts", None)
            if last is None or now - last >= 60:
                self._dryrun_log_ts = now
                logger.warning("DRYRUN_UNPROVEN reason=%s", reason)
        return ok

    def _enabled(self) -> bool:
        if not self._dryrun_proven() or not bool(self._plan):
            return False
        exp_ms = getattr(self, "_plan_expires_ms", None)
        if exp_ms is not None and time.time() * 1000 >= exp_ms:
            last = getattr(self, "_expired_log_ts", 0.0)
            now = time.monotonic()
            if now - last >= PLAN_EXPIRED_LOG_S:
                self._expired_log_ts = now
                logger.warning(
                    "PLAN_EXPIRED_BLOCKED pair=%s plan_id=%s expires_ms=%s — "
                    "entries/DCA blocked (open-position protection still "
                    "runs)",
                    self._plan.get("symbol"),
                    self._plan.get("plan_id"),
                    exp_ms,
                )
            return False
        return True

    # ------------------------------------------------------------------ R11 contract
    def _filled_level_prices(self, pair: str) -> list[float] | None:
        """Price-keyed fill book for `pair`, or None when absent/malformed.

        P0-5b: the engine numbers levels DESCENDING from grid_upper while
        ``self._levels`` is ASCENDING — an index is only a position in a
        list, never an identity; the PRICE is the level's identity. The
        legacy ``filled_levels`` index list stays in the payload for older
        consumers but is NEVER consulted here: callers must fail closed on
        None instead of guessing an index base.
        """
        raw = self._aigrid.get("filled_level_prices")
        if not isinstance(raw, dict):
            return None
        prices = raw.get(pair)
        if not isinstance(prices, list) or not all(
            isinstance(p, (int, float)) and math.isfinite(p) for p in prices
        ):
            return None
        return [float(p) for p in prices]

    def _capital_cap_blocked(
        self,
        pair: str,
        *,
        filled_count: int,
        current_notional: float,
        current_margin: float,
        proposed_margin: float,
        leverage: float,
    ) -> str | None:
        """P0-8: plan capital caps enforced BEFORE any stake is returned.

        Missing cap fields fail CLOSED — the strategy never invents a limit
        the publisher did not send. A breach blocks the whole add (the
        stake is never shrunk to fit). Returns the reason or None.
        """
        max_levels = self._plan.get("grid_levels")
        max_pos = self._plan.get("max_position_size")
        max_cap = self._plan.get("max_capital")
        if not (
            isinstance(max_levels, (int, float))
            and math.isfinite(max_levels)
            and max_levels > 0
            and isinstance(max_pos, (int, float))
            and math.isfinite(max_pos)
            and isinstance(max_cap, (int, float))
            and math.isfinite(max_cap)
        ):
            return "CAP_FIELDS_MISSING"
        if filled_count >= int(max_levels):
            return f"MAX_LEVELS {filled_count}>={int(max_levels)}"
        if current_margin + proposed_margin > max_cap + 1e-9:
            return f"MAX_CAPITAL {current_margin + proposed_margin:.4f}>{max_cap:.4f}"
        proposed_notional = proposed_margin * leverage
        if current_notional + proposed_notional > max_pos + 1e-9:
            return f"MAX_POSITION_SIZE {current_notional + proposed_notional:.4f}>{max_pos:.4f}"
        return None

    def _log_capital_cap_blocked(self, pair: str, reason: str) -> None:
        now = time.monotonic()
        if now - getattr(self, "_cap_log_ts", 0.0) < CAPITAL_CAP_LOG_S:
            return
        self._cap_log_ts = now
        logger.warning(
            "CAPITAL_CAP_BLOCKED pair=%s plan_id=%s reason=%s",
            pair,
            self._plan.get("plan_id"),
            reason,
        )

    # ------------------------------------------------------------------ R13 lot ledger
    def _manage_enabled(self) -> bool:
        """Position-MANAGEMENT gate (reduces / whole-trade TP): needs a live
        plan + dry-run attestation but NOT the TTL — like custom_stoploss,
        protection of an open position survives plan expiry."""
        return bool(self._plan) and bool(self._aigrid.get("dry_run_verified"))

    def _level_index(self, price: float) -> int | None:
        """Ascending plan-level index for an exact (1e-9) price, else None."""
        for i, lv in enumerate(self._levels):
            if abs(lv - price) <= 1e-9:
                return i
        return None

    @staticmethod
    def _parse_exit_tag(tag) -> tuple[str, int, int] | None:
        """``{plan_id}:grid:{idx}:exit:{seq}`` -> (plan_id, idx, seq) else None."""
        if not isinstance(tag, str):
            return None
        parts = tag.split(":")
        if len(parts) < 5 or parts[-4] != "grid" or parts[-2] != "exit":
            return None
        try:
            idx = int(parts[-3])
            seq = int(parts[-1])
        except ValueError:
            return None
        return (":".join(parts[:-4]), idx, seq)

    def _entry_level_index(self, order) -> int | None:
        """Attribute a filled ENTRY order to a plan level by IDENTITY, never
        by price (P0-11): only the ``grid_level_{i}`` tag the DCA add path
        stamps counts. A malformed tag or an out-of-range index -> None —
        the fill is counted unmapped. The old half-gap price fallback is
        gone: it mapped an off-level fill (92.3, inside the 90 half-gap)
        back to level 0 and emitted a reduce on a guessed attribution."""
        tag = getattr(order, "ft_order_tag", None)
        if not (isinstance(tag, str) and tag.startswith("grid_level_")):
            return None
        suffix = tag[len("grid_level_") :]
        if not suffix.isdigit():
            return None
        idx = int(suffix)
        return idx if 0 <= idx < len(self._levels) else None

    @staticmethod
    def _entry_fill_price(order) -> float | None:
        """The fill's own price: realized ``average`` first, ``price`` as
        fallback. None when neither is a positive finite number — a lot
        whose fill price cannot be read cannot be staked (never guessed)."""
        price = getattr(order, "average", None)
        if isinstance(price, (int, float)) and math.isfinite(price) and price > 0:
            return float(price)
        price = getattr(order, "price", None)
        if isinstance(price, (int, float)) and math.isfinite(price) and price > 0:
            return float(price)
        return None

    def _order_is_entry(self, order, is_short: bool) -> bool:
        """ft_is_entry when present; else infer from the order side."""
        flag = getattr(order, "ft_is_entry", None)
        if flag is not None:
            return bool(flag)
        side = getattr(order, "ft_order_side", getattr(order, "side", None))
        return side == ("sell" if is_short else "buy")

    def _level_order_stats(self, trade: Trade) -> dict | None:
        """Per-level fill/exit truth from ``trade.orders`` (FT-side state);
        None when the order list is unreachable — never inferred from
        candles. ``entries`` accumulates each level's closed ENTRY fill
        QUANTITY (``order.filled``, P0-4b) and its notional at the fills'
        own prices; ``exits`` accumulates the exited qty per level;
        ``unmapped`` counts closed fills that cannot be attributed to the
        current plan's grid (foreign exit, off-tag entry, exit tag from
        another plan, unreadable fill price)."""
        orders = getattr(trade, "orders", None)
        if not isinstance(orders, (list, tuple)):
            return None
        is_short = bool(getattr(trade, "is_short", False))
        plan_id = str(self._plan.get("plan_id") or "")
        stats = {"entries": {}, "exits": {}, "open_exit_levels": set(), "unmapped": 0}
        for order in orders:
            status = str(getattr(order, "status", "") or "").lower()
            filled = getattr(order, "filled", 0.0)
            filled = float(filled) if isinstance(filled, (int, float)) else 0.0
            if self._order_is_entry(order, is_short):
                if status != "closed" or filled <= 0:
                    continue
                idx = self._entry_level_index(order)
                price = self._entry_fill_price(order)
                if idx is None or price is None:
                    stats["unmapped"] += 1
                else:
                    ent = stats["entries"].setdefault(idx, {"qty": 0.0, "notional": 0.0})
                    ent["qty"] += filled
                    ent["notional"] += filled * price
                continue
            parsed = self._parse_exit_tag(getattr(order, "ft_order_tag", None))
            ours = parsed is not None and parsed[0] == plan_id
            idx = parsed[1] if ours and 0 <= parsed[1] < len(self._levels) else None
            if status == "closed" and filled > 0:
                if idx is None:
                    stats["unmapped"] += 1  # a real exit fill we cannot place
                else:
                    stats["exits"][idx] = stats["exits"].get(idx, 0.0) + filled
            elif status == "open" and idx is not None:
                stats["open_exit_levels"].add(idx)
        return stats

    def _reconcile_level_exits(self, trade: Trade, stats: dict | None) -> None:
        """Sync emitted-exit bookkeeping with FT order state. An exit order
        ``closed`` with ``filled > 0`` CONFIRMS the reduce (the level's lot
        is gone); open keeps it pending; dead/absent releases it for a
        retry. ``_exit_seq`` is raised past every observed ``:exit:`` tag so
        ids are never reused — even across a strategy restart. Nothing
        here reads candles."""
        if stats is None:
            return
        recs = self.__dict__.setdefault("_exit_orders", {}).setdefault(trade.pair, {})
        seqs = self.__dict__.setdefault("_exit_seq", {})
        plan_id = str(self._plan.get("plan_id") or "")
        by_tag = {}
        for order in trade.orders:
            tag = getattr(order, "ft_order_tag", None)
            if isinstance(tag, str):
                by_tag[tag] = order
            parsed = self._parse_exit_tag(tag)
            if parsed is not None and parsed[0] == plan_id:
                seqs[plan_id] = max(seqs.get(plan_id, 0), parsed[2])
        for price, rec in list(recs.items()):
            order = by_tag.get(rec["tag"])
            if order is None:
                if not rec.get("confirmed"):
                    del recs[price]  # never materialised -> may retry
                continue
            status = str(getattr(order, "status", "") or "").lower()
            filled = getattr(order, "filled", 0.0)
            filled = float(filled) if isinstance(filled, (int, float)) else 0.0
            if status == "closed" and filled > 0:
                rec["confirmed"] = True
            elif status != "open":
                del recs[price]  # cancelled/expired/rejected/empty -> retry
        for price, rec in list(recs.items()):
            if not rec.get("confirmed"):
                continue
            idx = self._level_index(price)
            if idx is None:
                continue
            ent = stats["entries"].get(idx)
            net = (ent["qty"] if ent else 0.0) - stats["exits"].get(idx, 0.0)
            if net > 0:
                del recs[price]  # re-armed: a NEW lot filled at this level

    def _live_lot_ledger(self, trade: Trade, stats: dict | None) -> dict | None:
        """``{level_price: {"qty", "fill", "pending"}}`` for live lots, or
        None when the ledger cannot be reconciled (fail closed — never
        guessed).

        P0-4b / D-099(3): each closed ENTRY fill funds its own lot —
        ``qty = order.filled``, fill price = ``order.average`` /
        ``order.price``. A confirmed exit fill decrements the qty of the
        level its ``:exit:`` tag names (the exact lot that was reduced —
        partial exits leave a live remainder). No uniform ``amount/n_live``
        allocation, no scaling through ``trade.stake_amount``.

        Reconciliation: the position's qty must equal what the order book
        explains — ``abs(sum(qty_remaining) - trade.amount)`` within
        ``max(1e-6, 0.5% * amount)``, else LEDGER_UNBALANCED + None."""
        if stats is None or stats["unmapped"]:
            return None
        amount = float(getattr(trade, "amount", 0.0) or 0.0)
        if not (math.isfinite(amount) and amount > 0):
            return None
        lots = {}
        for idx, ent in stats["entries"].items():
            qty = ent["qty"] - stats["exits"].get(idx, 0.0)
            if qty <= 0:
                continue
            lots[self._levels[idx]] = {
                "qty": qty,
                "fill": ent["notional"] / ent["qty"],
                "pending": idx in stats["open_exit_levels"],
            }
        total = sum(lot["qty"] for lot in lots.values())
        if abs(total - amount) > max(1e-6, 0.005 * amount):
            logger.warning(
                "LEDGER_UNBALANCED pair=%s plan_id=%s qty_remaining=%.8f "
                "trade.amount=%.8f — the order book cannot explain the "
                "position; refusing to guess lot sizes",
                trade.pair,
                self._plan.get("plan_id"),
                total,
                amount,
            )
            return None
        return lots

    def _reduce_level_tp(self, trade: Trade, current_rate: float):
        """Per-level take-profit: ``(negative_stake, client_order_id)`` for
        the single longest-realised level at TP, else None.

        stake = ``-(lot's own qty x its own fill price) / leverage`` — the
        margin that exact lot committed (P0-4b), never the uniform
        ``trade.stake_amount / trade.amount`` share. The id scheme
        ``{plan_id}:grid:{level_idx}:exit:{seq}`` is monotonic per plan and
        never reused. P0-12: when EVERY live lot is at TP this returns
        None — the whole-trade close belongs to ``custom_exit``, and
        emitting a reduce on the same candle would stack two close paths."""
        if not self._manage_enabled():
            return None
        plan_id = self._plan.get("plan_id")
        spacing = self._plan.get("grid_spacing")
        if not (
            isinstance(plan_id, str)
            and plan_id
            and isinstance(spacing, (int, float))
            and math.isfinite(spacing)
            and spacing > 0
        ):
            return None
        rate = float(current_rate or 0.0)
        if not (math.isfinite(rate) and rate > 0):
            return None
        stats = self._level_order_stats(trade)
        self._reconcile_level_exits(trade, stats)
        ledger = self._live_lot_ledger(trade, stats)
        if not ledger:
            return None
        is_short = bool(getattr(trade, "is_short", False))

        def _reached(price: float) -> bool:
            if is_short:
                return rate <= price * (1.0 - spacing)
            return rate >= price * (1.0 + spacing)

        if all(_reached(p) for p in ledger):
            return None  # every live lot at TP -> custom_exit closes it all
        reached = [p for p, lot in ledger.items() if not lot["pending"] and _reached(p)]
        if not reached:
            return None
        since = self.__dict__.setdefault("_tp_since", {}).setdefault(trade.pair, {})
        for price in list(since):
            if price not in reached:
                del since[price]
        now = time.monotonic()
        for price in reached:
            since.setdefault(price, now)
        # longest-realised first; a tie breaks towards the extreme lot
        # (lowest for longs, highest for shorts) whose TP was crossed first.
        target = min(reached, key=lambda p: (since[p], -p if is_short else p))
        lot = ledger[target]
        leverage = float(getattr(trade, "leverage", None) or 1.0)
        if not (math.isfinite(leverage) and leverage > 0):
            leverage = 1.0
        stake = -lot["qty"] * lot["fill"] / leverage
        level_idx = self._level_index(target)
        if level_idx is None:
            return None
        seqs = self.__dict__.setdefault("_exit_seq", {})
        seqs[plan_id] = seqs.get(plan_id, 0) + 1
        tag = f"{plan_id}:grid:{level_idx}:exit:{seqs[plan_id]}"
        self.__dict__.setdefault("_exit_orders", {}).setdefault(trade.pair, {})[target] = {
            "tag": tag,
            "confirmed": False,
        }
        return stake, tag

    # ------------------------------------------------------------------ indicators
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata.get("pair")
        self._refresh_integration(pair)
        if pair is not None and len(dataframe):
            # The last row is the most recent CLOSED candle; its
            # predecessor's close is the crossing reference the engine
            # uses (first candle falls back to its own open).
            last = dataframe.iloc[-1]
            prev = (
                float(dataframe["close"].iloc[-2]) if len(dataframe) >= 2 else float(last["open"])
            )
            self.__dict__.setdefault("_last_closed", {})[pair] = {
                "open": float(last["open"]),
                "high": float(last["high"]),
                "low": float(last["low"]),
                "close": float(last["close"]),
                "prev_close": prev,
            }
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        if (
            self._entries_halted()
            or metadata.get("pair") in self._blocked_pairs()
            or not self._enabled()
        ):
            return dataframe
        # Freqtrade copies the row's enter_tag onto the Trade — the only
        # supported way for a strategy to tag entries. The tag embeds the
        # plan_id so the orchestrator's ownership sync recognizes the fill
        # as ours even if the strategy name guard were bypassed (T-204c).
        plan_id = str(self._plan.get("plan_id") or "")
        if plan_id:
            dataframe["enter_tag"] = f"aigrid_{plan_id}"
        mode = self._plan.get("grid_mode", "SAFE_HOLD")
        if mode in ("RANGE_GRID", "LONG_GRID"):
            # enter when price crosses down into the grid's top region
            dataframe.loc[
                (dataframe["close"] <= self._plan["grid_upper"])
                & (dataframe["close"] > self._plan["grid_lower"]),
                "enter_long",
            ] = 1
        elif mode == "SHORT_GRID":
            dataframe.loc[
                (dataframe["close"] >= self._plan["grid_lower"])
                & (dataframe["close"] < self._plan["grid_upper"]),
                "enter_short",
            ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe  # exits are level-based via custom_exit/adjust

    # ------------------------------------------------------------------ execution mapping
    def custom_entry_price(
        self,
        pair: str,
        trade,
        current_time,
        proposed_rate: float,
        entry_tag,
        side: str,
        **kwargs,
    ) -> float:
        """Enter at the nearest grid level (limit-style).

        P0-9: when the plan is dead this returns proposed_rate — it never
        invents a fresh price for a stale order. Unfilled orders are
        bounded by check_entry_timeout; adjust_order_price is not
        overridden, so FT keeps the order's original price (no repricing
        after expiry)."""
        self._refresh_integration(pair)
        if self._entries_halted() or pair in self._blocked_pairs() or not self._enabled():
            return proposed_rate
        if side == "long":
            candidates = [p for p in self._levels if p <= proposed_rate]
            return max(candidates) if candidates else proposed_rate
        candidates = [p for p in self._levels if p >= proposed_rate]
        return min(candidates) if candidates else proposed_rate

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time,
        entry_tag,
        side: str,
        **kwargs,
    ) -> bool:
        self._refresh_integration(pair)
        if self._entries_halted() or pair in self._blocked_pairs() or not self._enabled():
            return False
        # ownership + risk re-check happens in the orchestrator; here: mode gate
        mode = self._plan.get("grid_mode", "SAFE_HOLD")
        if mode == "SAFE_HOLD":
            return False
        if side == "long" and mode not in ("RANGE_GRID", "LONG_GRID"):
            return False
        return not (side == "short" and mode != "SHORT_GRID")

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time,
        current_rate: float,
        current_profit: float,
        min_stake,
        max_stake,
        current_entry_rate,
        current_exit_rate,
        current_entry_profit,
        current_exit_profit,
        **kwargs,
    ):
        """Two paths, one order per callback, reduce first:

        1. R13 PER-LEVEL TAKE-PROFIT (position management): a live lot
           whose level price is a full grid_spacing past current_rate is
           reduced via a NEGATIVE stake — the only per-level close path FT
           has (``-(lot qty x lot fill price) / leverage`` releases exactly
           that lot's committed margin). Runs past plan TTL/halt like
           custom_stoploss; needs a live plan + dry-run attestation.
        2. Sequential DCA ADD (new exposure): the full entry gate —
           halted -> blocked pair -> enabled (incl. plan TTL) ->
           grid_mode != SAFE_HOLD -> level_stake > 0 -> closed candle ->
           price-keyed fill book -> capital caps.

        ``trade.has_open_orders`` returns None for BOTH paths: FT cancels a
        pending order whenever the returned order differs, so emitting
        anything while an order is open would stack exits or churn the
        sequential book. P0-5b: fills are matched by PRICE (|delta| <=
        1e-9) — an index is only a position, the price is the identity.
        P0-8: notional is marked at current_rate; proposed stake counts as
        level_stake margin / level_stake * leverage notional vs the caps."""
        self._refresh_integration(trade.pair)
        if getattr(trade, "has_open_orders", False):
            return None
        reduced = self._reduce_level_tp(trade, current_rate)
        if reduced is not None:
            return reduced
        if self._entries_halted():
            return None
        if trade.pair in self._blocked_pairs():
            return None
        if not self._enabled():
            return None
        if self._plan.get("grid_mode") == "SAFE_HOLD":
            return None
        level_stake = float(self._plan.get("level_stake", 0.0))
        if level_stake <= 0:
            return None
        candle = self.__dict__.setdefault("_last_closed", {}).get(trade.pair)
        if candle is None:
            return None  # no closed candle captured -> never DCA (fail-closed)
        filled_prices = self._filled_level_prices(trade.pair)
        if filled_prices is None:
            return None  # no price-keyed fill book -> never DCA (fail-closed)
        cap_reason = self._capital_cap_blocked(
            trade.pair,
            filled_count=len(filled_prices),
            current_notional=float(getattr(trade, "amount", 0.0) or 0.0)
            * float(current_rate or 0.0),
            current_margin=float(getattr(trade, "stake_amount", 0.0) or 0.0),
            proposed_margin=level_stake,
            leverage=float(getattr(trade, "leverage", None) or 1.0),
        )
        if cap_reason is not None:
            self._log_capital_cap_blocked(trade.pair, cap_reason)
            return None
        levels = self._levels
        is_long = not trade.is_short
        ref = candle["prev_close"]
        # next crossed level below (long) / above (short) not yet filled
        for price in sorted(levels, reverse=True) if is_long else sorted(levels):
            level_idx = levels.index(price)
            if any(abs(price - fp) <= 1e-9 for fp in filled_prices):
                continue
            if is_long and ref > price >= candle["low"]:
                return level_stake, f"grid_level_{level_idx}"
            if (not is_long) and ref < price <= candle["high"]:
                # positive stake ADDS to the position for both directions;
                # a negative return would REDUCE it (per-level TP path)
                return level_stake, f"grid_level_{level_idx}"
        return None

    def check_entry_timeout(
        self, pair: str, trade: Trade, order, current_time, *args, **kwargs
    ) -> bool:
        """Cancel an UNFILLED entry order once the plan can no longer enter.

        P0-9: FT 2026.10 calls this every loop for each open entry-side
        order of our own trades (initial entries AND DCA adds; the
        ``unfilledtimeout`` config is evaluated first) — returning True
        cancels the order. It is the ONLY strategy-side cancel API
        (PENDING_CANCEL_UNSUPPORTED documents the boundary). Remaining
        limits, honestly: the check is loop-granular, so an order can still
        fill between two iterations, and only our own trades' orders are
        reachable — orders of another bot's trade are not. A pending order
        whose enter_tag belongs to a different (dead) plan is cancelled
        too; tag-less orders can't be attributed and are left to the
        enabled-gate alone."""
        self._refresh_integration(pair)
        if (
            self._entries_halted()
            or pair in self._blocked_pairs()
            or not self._enabled()
            or self._plan.get("grid_mode") == "SAFE_HOLD"
        ):
            return True
        tag = getattr(order, "ft_order_tag", None)
        plan_id = str(self._plan.get("plan_id") or "")
        if plan_id and isinstance(tag, str) and tag:
            return tag != f"aigrid_{plan_id}"
        return False

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        """Whole-trade exit ONLY: every LIVE lot has reached its level TP.

        custom_exit carries no amount — it can only close the whole trade,
        so it must never be the per-level close path (R13 / H2): per-level
        TP lives in ``adjust_trade_position`` as a negative-stake reduce.
        Stop-loss and emergencies go through ``custom_stoploss``. An
        unresolvable lot ledger returns None — the old aggregate-profit
        heuristic is gone; it closed a 4-lot trade on the FIRST level TP."""
        self._refresh_integration(pair)
        if not self._manage_enabled():
            return None
        spacing = self._plan.get("grid_spacing")
        if not (isinstance(spacing, (int, float)) and math.isfinite(spacing) and spacing > 0):
            return None
        rate = float(current_rate or 0.0)
        if not (math.isfinite(rate) and rate > 0):
            return None
        stats = self._level_order_stats(trade)
        self._reconcile_level_exits(trade, stats)
        ledger = self._live_lot_ledger(trade, stats)
        if not ledger:
            return None
        is_short = bool(getattr(trade, "is_short", False))
        if all(
            ((rate <= p * (1.0 - spacing)) if is_short else (rate >= p * (1.0 + spacing)))
            for p in ledger
        ):
            return "GRID_TP_ALL"
        return None

    def custom_stake_amount(
        self,
        pair: str,
        current_time,
        current_rate: float,
        proposed_stake: float,
        min_stake,
        max_stake,
        leverage: float,
        entry_tag,
        side: str,
        **kwargs,
    ):
        """Size the first entry by the plan's per-level MARGIN (level_stake).

        P0-8: the same capital caps as adjust_trade_position gate the first
        entry — a first trade has no booked notional/margin yet, so the
        proposed level_stake alone is checked against max_capital and, as
        level_stake * leverage, against max_position_size."""
        self._refresh_integration(pair)
        if self._entries_halted() or pair in self._blocked_pairs() or not self._enabled():
            return None
        level_stake = float(self._plan.get("level_stake") or 0.0)
        if level_stake <= 0:
            return None
        filled_prices = self._filled_level_prices(pair)
        if filled_prices is None:
            return None  # no price-keyed fill book -> fail closed
        cap_reason = self._capital_cap_blocked(
            pair,
            filled_count=len(filled_prices),
            current_notional=0.0,
            current_margin=0.0,
            proposed_margin=level_stake,
            leverage=float(leverage or 1.0),
        )
        if cap_reason is not None:
            self._log_capital_cap_blocked(pair, cap_reason)
            return None
        return float(level_stake)

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float:
        """Plan SL anchored at its ABSOLUTE price (T-223b / D-092 §e).

        FT anchors the returned ratio at CURRENT_RATE and divides by
        leverage (adjust_stop_loss: new_loss = current*(1 - |sl|/lev)), so
        the plan's absolute stop goes through the official
        ``stoploss_from_absolute`` helper — a raw ``sl/open_rate - 1``
        ratio would run at -5% price risk under 2x (should be -10%) and
        ride the price up at 1x (unwanted trailing). We return the
        conventional negative risk ratio; FT uses abs() either way.

        Invalid stops (wrong side of price / inside the liquidation
        buffer) return None — FT keeps the previous stop — and the pair is
        blocked from new plans (SL_INVALID), never clamped into validity.
        Order: plan.stop_loss -> cached plan SL -> open_rate-anchored -10%
        fallback. A dead/stale orchestrator must not leave positions
        unprotected."""
        self._refresh_integration(pair, trade=trade)
        plan = self._plan if self._enabled() else {}
        stop_rate = self._resolve_stop_price(plan, pair, trade)
        if stop_rate is None:
            return None
        reason = protective_stop_error(
            stop_rate,
            float(current_rate or 0.0),
            bool(trade.is_short),
            getattr(trade, "liquidation_price", None),
        )
        if reason is not None:
            self._block_pair_sl(pair, stop_rate, reason)
            return None
        self._blocked_pairs().pop(pair, None)
        leverage = float(getattr(trade, "leverage", None) or 1.0)
        return -1.0 * stoploss_from_absolute(
            stop_rate,
            current_rate=current_rate,
            is_short=trade.is_short,
            leverage=leverage,
        )
