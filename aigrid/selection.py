"""Pair selection — deterministic technical scan over the watchlist (NO LLM).

Pipeline (per user direction, 2026-11-03):
    8 watched pairs --[technical scan every 5m candle]--> top-N candidates
    --> TradingAgents analyzes at most `max_ai_per_cycle` of them
    --> Freqtrade trades at most `max_active_pairs` concurrently.

Filters (a pair is tradable for grids only if ALL pass):
  - liquidity   : mean quote volume per candle >= min_quote_volume_5m
  - volatility  : fee_coverage = atr_pct / round_trip_cost >= min_fee_coverage
                  (levels can't pay for fees+slippage otherwise)
  - freshness   : last closed candle within max_staleness of the scan time
  - regime      : HIGH_VOLATILITY on the trend timeframe -> excluded (fail-closed)

Score (deterministic, higher = better):
  score = fee_coverage * log10(quote_vol / min_quote_vol + 1)
  + small bonus when the trend-TF regime is RANGE (best grid habitat).

BTC/ETH are NOT in the candidate pool — they feed ReferenceMarketGuard only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from .data.binance_adapter import TF_MS, BinanceFuturesDataAdapter
from .regime.engine import (
    DOWNTREND,
    HIGH_VOLATILITY,
    RANGE,
    UPTREND,
    RegimeThresholds,
    compute_features,
    decide_from_features,
)


@dataclass(frozen=True)
class CandidateMetrics:
    pair: str
    quote_vol_5m: float  # mean quote volume per execution candle
    atr_pct: float  # ATR% on execution TF (per-candle move scale)
    fee_coverage: float  # atr_pct / round_trip_cost
    trend_regime: str  # regime on the trend timeframe
    trend_confidence: float
    last_close: float
    funding_rate: float | None
    candles: int
    score: float = 0.0
    rejected: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return self.__dict__ | {"rejected": list(self.rejected)}


@dataclass(frozen=True)
class SelectionResult:
    as_of_ms: int
    candidates: tuple[CandidateMetrics, ...]  # accepted, best first
    rejected: tuple[CandidateMetrics, ...]  # filtered out, with reasons
    scanned: int

    @property
    def top_pairs(self) -> list[str]:
        return [c.pair for c in self.candidates]


class PairSelector:
    """Ranks the watchlist every execution candle. Deterministic; no LLM."""

    def __init__(
        self,
        adapter: BinanceFuturesDataAdapter,
        *,
        execution_tf: str = "5m",
        trend_tf: str = "15m",
        top_n: int = 3,
        min_quote_volume_5m: float = 300_000.0,  # USDT per 5m candle (mean 24h)
        min_fee_coverage: float = 1.5,  # atr_pct must cover 1.5x costs
        round_trip_cost: float = 0.002,  # 2x(taker 5bps + slip 5bps)
        lookback: int = 300,
        liquidity_window: int = 288,  # 24h of 5m candles
    ):
        self.adapter = adapter
        self.execution_tf = execution_tf
        self.trend_tf = trend_tf
        self.top_n = top_n
        self.min_quote_volume = min_quote_volume_5m
        self.min_coverage = min_fee_coverage
        self.round_trip = round_trip_cost
        self.lookback = lookback
        self.liquidity_window = liquidity_window
        self._th = RegimeThresholds()

    # ------------------------------------------------------------------ scan
    def scan(self, watchlist: tuple[str, ...] | list[str], as_of: datetime) -> SelectionResult:
        """Rank the watchlist. Closed candles only; deterministic."""
        as_of_ms = int(pd.Timestamp(as_of).timestamp() * 1000)
        accepted: list[CandidateMetrics] = []
        rejected: list[CandidateMetrics] = []
        for pair in watchlist:
            m = self._metrics(pair, as_of)
            (accepted if not m.rejected else rejected).append(m)
        accepted.sort(key=lambda m: m.score, reverse=True)
        return SelectionResult(
            as_of_ms,
            tuple(accepted[: self.top_n]),
            tuple(accepted[self.top_n :]) + tuple(rejected),
            scanned=len(watchlist),
        )

    # ------------------------------------------------------------------ metrics
    def _metrics(self, pair: str, as_of: datetime) -> CandidateMetrics:
        rej: list[str] = []
        try:
            df = self.adapter.get_ohlcv(
                pair, self.execution_tf, as_of=as_of, lookback=self.lookback
            )
        except (FileNotFoundError, ValueError):
            return self._reject(pair, ("NO_DATA",))
        if len(df) < 60:
            rej.append("INSUFFICIENT_HISTORY")
        # freshness: last candle close must be within 2 execution TFs of as_of
        last_close_ms = int(df.index[-1].value // 1_000_000) + TF_MS[self.execution_tf]
        if pd.Timestamp(as_of).tzinfo is None:
            as_of = as_of.replace(tzinfo=UTC)
        stale = (int(pd.Timestamp(as_of).timestamp() * 1000) - last_close_ms) > 2 * TF_MS[
            self.execution_tf
        ]
        if stale:
            rej.append("DATA_STALE")

        feats = compute_features(df)
        v = {
            k: float(feats.iloc[-1][k])
            for k in (
                "adx",
                "plus_di",
                "minus_di",
                "atr_pct",
                "bbw",
                "slope",
                "qv",
                "ema20_gt_ema50",
            )
        }
        atr_pct = v["atr_pct"]
        exec_regime = decide_from_features(
            v, self._th, timeframe_seconds=TF_MS[self.execution_tf] // 1000
        )

        # trend regime on the trend TF (separate from execution-timeframe state)
        try:
            tdf = self.adapter.get_ohlcv(pair, self.trend_tf, as_of=as_of, lookback=self.lookback)
            tfeats = compute_features(tdf)
            tv = {
                k: float(tfeats.iloc[-1][k])
                for k in (
                    "adx",
                    "plus_di",
                    "minus_di",
                    "atr_pct",
                    "bbw",
                    "slope",
                    "qv",
                    "ema20_gt_ema50",
                )
            }
            trend_res = decide_from_features(
                tv, self._th, timeframe_seconds=TF_MS[self.trend_tf] // 1000
            )
        except (FileNotFoundError, ValueError, IndexError):
            trend_res = exec_regime  # missing trend data -> use exec regime
            rej.append("TREND_DATA_MISSING")

        tail = df.tail(min(self.liquidity_window, len(df)))
        quote_vol = float((tail["volume"] * tail["close"]).mean())
        funding = self.adapter.get_funding(pair, as_of=as_of)
        frate = float(funding.iloc[-1]["funding_rate"]) if len(funding) else None
        coverage = atr_pct / self.round_trip if self.round_trip > 0 else 0.0
        last_close = float(df.iloc[-1]["close"])

        if quote_vol < self.min_quote_volume:
            rej.append("LOW_LIQUIDITY")
        if coverage < self.min_coverage:
            rej.append("VOLATILITY_BELOW_FEE_FLOOR")
        if exec_regime.regime == HIGH_VOLATILITY or trend_res.regime == HIGH_VOLATILITY:
            rej.append("REGIME_HIGH_VOLATILITY")

        # deterministic score: coverage x liquidity + RANGE bonus
        score = 0.0
        if not rej:
            score = coverage * math.log10(quote_vol / self.min_quote_volume + 1.0)
            if trend_res.regime == RANGE:
                score *= 1.25  # ranges are the grid's best habitat
            elif trend_res.regime in (UPTREND, DOWNTREND):
                score *= 0.9  # directional grids allowed but slightly discounted
        return CandidateMetrics(
            pair=pair,
            quote_vol_5m=round(quote_vol, 2),
            atr_pct=round(atr_pct, 6),
            fee_coverage=round(coverage, 3),
            trend_regime=trend_res.regime,
            trend_confidence=round(trend_res.confidence, 3),
            last_close=last_close,
            funding_rate=frate,
            candles=len(df),
            score=round(score, 4),
            rejected=tuple(rej),
        )

    @staticmethod
    def _reject(pair: str, reasons: tuple[str, ...]) -> CandidateMetrics:
        return CandidateMetrics(
            pair=pair,
            quote_vol_5m=0.0,
            atr_pct=0.0,
            fee_coverage=0.0,
            trend_regime="",
            trend_confidence=0.0,
            last_close=0.0,
            funding_rate=None,
            candles=0,
            score=0.0,
            rejected=reasons,
        )


# ------------------------------------------------------------------ BTC/ETH guard
@dataclass(frozen=True)
class ReferenceVerdict:
    """Reference-market state. long_grids_ok=False vetoes LONG/RANGE entries
    on alt pairs; SHORT grids remain allowed (aligned with a dump)."""

    alt_long_ok: bool
    alt_short_ok: bool
    reasons: tuple[str, ...]
    btc_regime: str = ""
    eth_regime: str = ""
    btc_drop_pct_1h: float = 0.0
    eth_drop_pct_1h: float = 0.0

    def as_dict(self) -> dict:
        return self.__dict__ | {"reasons": list(self.reasons)}


class ReferenceMarketGuard:
    """BTC (and ETH) as market compass: a BTC dump re-evaluates alt longs."""

    def __init__(
        self,
        adapter: BinanceFuturesDataAdapter,
        *,
        reference_pairs=("BTC/USDT:USDT", "ETH/USDT:USDT"),
        trend_tf: str = "15m",
        drop_window_candles: int = 12,  # 12 x 5m = 1h
        drop_threshold_pct: float = 0.015,
    ):  # -1.5% in 1h = dump
        self.adapter = adapter
        self.reference_pairs = tuple(reference_pairs)
        self.trend_tf = trend_tf
        self.drop_window = drop_window_candles
        self.drop_threshold = drop_threshold_pct
        self._th = RegimeThresholds()

    def check(self, as_of: datetime) -> ReferenceVerdict:
        reasons: list[str] = []
        regimes: dict[str, str] = {}
        drops: dict[str, float] = {}
        for ref in self.reference_pairs:
            base = ref.split("/")[0]
            try:
                tdf = self.adapter.get_ohlcv(ref, self.trend_tf, as_of=as_of, lookback=300)
                feats = compute_features(tdf)
                v = {
                    k: float(feats.iloc[-1][k])
                    for k in (
                        "adx",
                        "plus_di",
                        "minus_di",
                        "atr_pct",
                        "bbw",
                        "slope",
                        "qv",
                        "ema20_gt_ema50",
                    )
                }
                rr = decide_from_features(
                    v, self._th, timeframe_seconds=TF_MS[self.trend_tf] // 1000
                )
                regimes[base] = rr.regime
            except (FileNotFoundError, ValueError, IndexError):
                regimes[base] = HIGH_VOLATILITY
                reasons.append(f"{base}_REFERENCE_DATA_MISSING")
            try:
                df5 = self.adapter.get_ohlcv(ref, "5m", as_of=as_of, lookback=self.drop_window + 2)
                win = df5.tail(self.drop_window)
                if len(win) >= 2:
                    drops[base] = float(win.iloc[-1]["close"] / win["high"].max() - 1.0)
                else:
                    drops[base] = 0.0
            except (FileNotFoundError, ValueError):
                drops[base] = 0.0

        long_ok, short_ok = True, True
        for base, regime in regimes.items():
            drop = drops.get(base, 0.0)
            if regime in (DOWNTREND, HIGH_VOLATILITY):
                reasons.append(f"{base}_REGIME_{regime}")
                long_ok = False
            if regime == UPTREND:
                short_ok = False  # BTC ripping up -> don't short alts
            if drop <= -self.drop_threshold:
                reasons.append(f"{base}_DUMP_{drop * 100:.1f}%_1H")
                long_ok = False
            if drop >= self.drop_threshold:
                reasons.append(f"{base}_PUMP_+{drop * 100:.1f}%_1H")
                short_ok = False
        return ReferenceVerdict(
            alt_long_ok=long_ok,
            alt_short_ok=short_ok,
            reasons=tuple(reasons),
            btc_regime=regimes.get("BTC", ""),
            eth_regime=regimes.get("ETH", ""),
            btc_drop_pct_1h=round(drops.get("BTC", 0.0) * 100, 3),
            eth_drop_pct_1h=round(drops.get("ETH", 0.0) * 100, 3),
        )
