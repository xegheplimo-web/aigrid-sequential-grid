"""Market Regime Engine — 4 states: RANGE / UPTREND / DOWNTREND / HIGH_VOLATILITY.

Extends the proven grid_companion regime core (Wilder ADX/ATR, BBW, closed
candles only, conservative UNKNOWN handling) with trend separation and
liquidity/uncertainty gates. Deterministic: same candles -> same regime.

Fail-closed: insufficient data or malformed input -> HIGH_VOLATILITY (the
"UNCERTAIN" bucket: no new positions allowed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

RANGE = "RANGE"
UPTREND = "UPTREND"
DOWNTREND = "DOWNTREND"
HIGH_VOLATILITY = "HIGH_VOLATILITY"  # includes UNCERTAIN bucket
REGIMES = (RANGE, UPTREND, DOWNTREND, HIGH_VOLATILITY)


@dataclass(frozen=True)
class RegimeThresholds:
    adx_max: float = 22.0  # above -> trending
    adx_trend_min: float = 20.0  # must exceed for a trend call
    bbw_max: float = 0.065  # Bollinger bandwidth ceiling for RANGE
    bbw_hvol_min: float = 0.12  # above -> high volatility outright
    atr_pct_max: float = 0.019  # ATR/close ceiling for RANGE
    atr_pct_hvol_min: float = 0.035  # above -> high volatility outright
    ema_slope_max_pct: float = 0.001  # |ema20 slope per candle / price| ceiling for RANGE
    min_quote_volume_per_min: float = 20_000.0  # liquidity gate (quote volume per minute)
    min_candles: int = 60


@dataclass(frozen=True)
class RegimeResult:
    regime: str
    confidence: float  # 0..1
    adx: float
    plus_di: float
    minus_di: float
    atr_pct: float
    bb_width: float
    ema20_slope_pct: float
    quote_volume_usdt: float
    reason: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def classify(
    ohlcv: pd.DataFrame, th: RegimeThresholds | None = None, timeframe_seconds: int = 900
) -> RegimeResult:
    """Classify the regime from CLOSED candles (index = open time, tz-aware UTC).

    The caller is responsible for the no-lookahead cut; this function only uses
    the rows it is given.
    """
    th = th or RegimeThresholds()
    min_qv = th.min_quote_volume_per_min * max(1, timeframe_seconds // 60)
    try:
        if len(ohlcv) < th.min_candles:
            return _fail(HIGH_VOLATILITY, "INSUFFICIENT_HISTORY")
        _validate_frame(ohlcv)
        feats = compute_features(ohlcv).iloc[-1]
        v = {k: float(feats[k]) for k in feats.index if k != "close"}
        if not all(math.isfinite(x) for x in v.values()):
            return _fail(HIGH_VOLATILITY, "NON_FINITE_FEATURE")
        return _decide(v, th, min_qv)
    except (ValueError, ZeroDivisionError, OverflowError, KeyError):
        return _fail(HIGH_VOLATILITY, "INVALID_CANDLES")


def compute_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Causal feature series (ADX/DI/ATR%/BBW/EMA slope/quote volume).

    All transforms are causal (ewm/rolling use past rows only), so computing
    them ONCE over a long series and slicing at index i is equivalent to
    computing on the trailing window -- and O(n) instead of O(n*window).
    """
    close = ohlcv["close"]
    high, low, volume = ohlcv["high"], ohlcv["low"], ohlcv["volume"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1
    )
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA)
    adx = dx.fillna(0.0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    ma = close.rolling(20).mean()
    sd = close.rolling(20).std(ddof=0)
    bbw = 4 * sd / ma.replace(0, pd.NA)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    slope = (ema20.diff(5) / 5) / close
    qv = (close * volume).rolling(20).mean()
    return pd.DataFrame(
        {
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "atr_pct": atr / close,
            "bbw": bbw,
            "slope": slope,
            "qv": qv,
            "ema20_gt_ema50": (ema20 > ema50).astype(float),
        }
    )


def decide_from_features(
    v: dict, th: RegimeThresholds | None = None, timeframe_seconds: int = 900
) -> RegimeResult:
    """Classify from one precomputed feature row (fast path for backtests)."""
    th = th or RegimeThresholds()
    min_qv = th.min_quote_volume_per_min * max(1, timeframe_seconds // 60)
    try:
        if not all(math.isfinite(float(x)) for x in v.values()):
            return _fail(HIGH_VOLATILITY, "NON_FINITE_FEATURE")
        return _decide(v, th, min_qv)
    except (ValueError, ZeroDivisionError, OverflowError, KeyError):
        return _fail(HIGH_VOLATILITY, "INVALID_CANDLES")


def _decide(v: dict, th: RegimeThresholds, min_quote_volume: float) -> RegimeResult:
    # --- liquidity gate: thin market -> no reliable signal
    if v["qv"] < min_quote_volume:
        return _fail(HIGH_VOLATILITY, "LOW_LIQUIDITY")

    # --- outright high volatility
    if v["atr_pct"] >= th.atr_pct_hvol_min or v["bbw"] >= th.bbw_hvol_min:
        return _full(v, HIGH_VOLATILITY, _conf(v, th), "VOLATILITY_EXTREME")

    # --- trend calls
    trending = v["adx"] >= th.adx_trend_min
    up_flow = v["plus_di"] > v["minus_di"]
    if trending and up_flow and v["slope"] > 0 and v["ema20_gt_ema50"] > 0:
        return _full(v, UPTREND, _conf(v, th), "ADX_DI_EMA_ALIGNED_UP")
    if trending and (not up_flow) and v["slope"] < 0 and v["ema20_gt_ema50"] == 0:
        return _full(v, DOWNTREND, _conf(v, th), "ADX_DI_EMA_ALIGNED_DOWN")
    if trending:
        # ADX strong but flows/EMA disagree -> uncertain direction
        return _full(v, HIGH_VOLATILITY, _conf(v, th), "TREND_DIRECTION_CONFLICT")

    # --- range call (all quiet thresholds must pass)
    if (
        v["adx"] <= th.adx_max
        and v["bbw"] <= th.bbw_max
        and v["atr_pct"] <= th.atr_pct_max
        and abs(v["slope"]) <= th.ema_slope_max_pct
    ):
        return _full(v, RANGE, _conf(v, th), "ALL_RANGE_THRESHOLDS_PASS")

    return _full(v, HIGH_VOLATILITY, _conf(v, th), "MIXED_SIGNALS_UNCERTAIN")


def _full(v: dict, regime: str, confidence: float, reason: str) -> RegimeResult:
    return RegimeResult(
        regime,
        round(max(0.0, min(1.0, confidence)), 3),
        round(v["adx"], 3),
        round(v["plus_di"], 3),
        round(v["minus_di"], 3),
        round(v["atr_pct"], 6),
        round(v["bbw"], 6),
        round(v["slope"], 8),
        round(v["qv"], 2),
        reason,
    )


def _fail(regime: str, reason: str) -> RegimeResult:
    return RegimeResult(regime, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, reason)


def _validate_frame(df: pd.DataFrame) -> None:
    cols = {"open", "high", "low", "close", "volume"}
    if not cols.issubset({c.lower() for c in df.columns}):
        raise ValueError("OHLCV columns missing")
    if df.index.has_duplicates:
        raise ValueError("duplicate candle timestamps")
    if not df.index.is_monotonic_increasing:
        raise ValueError("non-monotonic candle timestamps")
    if bool((df["high"] < df[["open", "close"]].max(axis=1)).any()):
        raise ValueError("high < max(open, close)")
    if bool((df["low"] > df[["open", "close"]].min(axis=1)).any()):
        raise ValueError("low > min(open, close)")


def _conf(v: dict, th: RegimeThresholds) -> float:
    """Confidence = distance from decision boundaries (heuristic, 0.3..0.95)."""
    c = 0.5
    if v["adx"] > th.adx_max:
        c += min(0.25, (v["adx"] - th.adx_max) / 100)
    else:
        c += min(0.2, (th.adx_max - v["adx"]) / 100)
    if v["atr_pct"] > th.atr_pct_max:
        c -= min(0.2, (v["atr_pct"] - th.atr_pct_max) * 5)
    if v["bbw"] > th.bbw_max:
        c -= min(0.15, (v["bbw"] - th.bbw_max) * 2)
    return round(max(0.3, min(0.95, c)), 3)
