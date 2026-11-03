"""AIGridStrategy — Freqtrade integration layer for the Adaptive AI Grid.

Runs INSIDE Freqtrade (dry-run only). It is the ONLY execution path: the
orchestrator never talks to an exchange; it writes an integration file that
this strategy consumes.

Design (Sequential Grid via DCA — see ARCHITECTURE.md §5):
- Freqtrade manages ONE trade per pair (its DCA model). The virtual grid book
  lives in the orchestrator; this strategy realizes its actions sequentially:
    * entry signal  -> when price crosses the next grid level (from the plan)
    * custom_entry_price -> the exact grid level price
    * adjust_trade_position -> adds the next level when price crosses it
    * custom_exit / adjust(negative) -> per-level take-profit
    * custom_stoploss -> plan stop loss
- ALL behavior is OFF unless the integration file says enabled=true AND
  dry-run is verified. With the file absent/corrupt -> pass-through no-op.

Safety:
- No live-order code. No exchange API calls. Reads one JSON file per loop.
- Stale integration file (> 2x plan TTL) -> trading disabled (fail-closed).
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from pathlib import Path

import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, stoploss_from_absolute
from pandas import DataFrame

logger = logging.getLogger(__name__)

INTEGRATION_FILE = Path(
    os.environ.get("AIGRID_INTEGRATION_FILE", "user_data/aigrid_state/ft_integration.json")
)
SL_CACHE_FILE = Path(
    os.environ.get("AIGRID_SL_CACHE_FILE", "user_data/aigrid_state/strategy_sl_cache.json")
)
MAX_FILE_AGE_S = 2 * 3600  # integration file must be fresher than 2h
SAFE_DEFAULT_SL_PCT = -0.10  # hard floor when neither plan nor cache is available
REQUIRED_TIMEFRAME = "5m"  # approved runtime timeframe — see bot_start()
LIQ_BUFFER_PCT = 0.03  # plan SL must sit >= 3% of liq price off liquidation


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
    startup_candle_count = 60
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

    def _enabled(self) -> bool:
        return bool(self._plan) and bool(self._aigrid.get("dry_run_verified"))

    # ------------------------------------------------------------------ indicators
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self._refresh_integration(metadata.get("pair"))
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
        self, pair: str, trade, current_time, proposed_rate: float, entry_tag, side: str, **kwargs
    ) -> float:
        """Enter at the nearest grid level (limit-style)."""
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
        if side == "short" and mode != "SHORT_GRID":
            return False
        return True

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
        """Add the next grid level when price crosses it (sequential DCA)."""
        self._refresh_integration(trade.pair)
        if self._entries_halted() or trade.pair in self._blocked_pairs() or not self._enabled():
            return None
        mode = self._plan.get("grid_mode")
        if mode == "SAFE_HOLD":
            return None
        levels = self._levels
        # filled_levels is per-pair in the multi-pair payload
        raw_filled = self._aigrid.get("filled_levels", {})
        filled = (
            raw_filled.get(trade.pair, []) if isinstance(raw_filled, dict) else list(raw_filled)
        )
        level_stake = float(self._plan.get("level_stake", 0.0))
        if level_stake <= 0:
            return None
        is_long = not trade.is_short
        # next level below (long) / above (short) not yet filled
        for price in sorted(levels, reverse=True) if is_long else sorted(levels):
            level_idx = levels.index(price)
            if level_idx in filled:
                continue
            if is_long and price < current_rate:
                return level_stake, f"grid_level_{level_idx}"
            if (not is_long) and price > current_rate:
                # positive stake ADDS to the position for both directions;
                # a negative return would REDUCE it (per-level TP path)
                return level_stake, f"grid_level_{level_idx}"
        return None

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        """Per-level take-profit: exit when profit >= one grid spacing."""
        self._refresh_integration(pair)
        if not self._enabled():
            return None
        spacing = float(self._plan.get("grid_spacing", 0.0))
        if spacing > 0 and current_profit >= spacing * 0.9:
            return "GRID_LEVEL_TP"
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
        """Size the first entry by the plan's per-level MARGIN (level_stake)."""
        self._refresh_integration(pair)
        if self._entries_halted() or pair in self._blocked_pairs() or not self._enabled():
            return None
        level_stake = float(self._plan.get("level_stake") or 0.0)
        if level_stake <= 0:
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
            stop_rate, current_rate=current_rate, is_short=trade.is_short, leverage=leverage
        )
