"""Binance USDT-M Futures data adapter.

Reads the EXISTING feather dataset under `user_data/data/binance/futures/`
(read-only). Never downloads historical data. Realtime updates for dry-run
use the public REST kline endpoint only (no keys, no account data).

Pair-name normalization:
  ETH/USDT:USDT  <->  ETH_USDT_USDT (feather filename)  <->  ETHUSDT (REST)

No-lookahead contract: every getter returns only CLOSED candles whose close
time is strictly <= `as_of` (UTC). `as_of=None` means "now" (live mode).
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

TF_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}
SUPPORTED_TF = tuple(TF_MS)
PUBLIC_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"


def pair_to_filestem(pair: str) -> str:
    """ETH/USDT:USDT -> ETH_USDT_USDT"""
    base_quote = pair.split(":", 1)[0]  # ETH/USDT
    return base_quote.replace("/", "_") + "_" + base_quote.split("/", 1)[1]


def filestem_to_pair(stem: str) -> str:
    """ETH_USDT_USDT -> ETH/USDT:USDT.

    Freqtrade futures filenames replace both '/' and ':' with '_' from the pair
    string 'ETH/USDT:USDT', so the trailing '_USDT' (margin suffix) and the
    preceding '_USDT' (quote) are stripped in turn.
    """
    if not stem.endswith("_USDT"):
        raise ValueError(f"Unexpected futures filestem: {stem!r}")
    inner = stem[: -len("_USDT")]  # drop ':USDT' margin suffix -> ETH_USDT
    if not inner.endswith("_USDT"):
        raise ValueError(f"Missing quote suffix in futures filestem: {stem!r}")
    inner = inner[: -len("_USDT")]  # drop '/USDT' quote -> ETH
    if not inner:
        raise ValueError(f"Empty base asset in filestem: {stem!r}")
    return inner.replace("_", "/") + "/USDT:USDT"


def pair_to_rest_symbol(pair: str) -> str:
    """ETH/USDT:USDT -> ETHUSDT"""
    return pair.split(":", 1)[0].replace("/", "")


def validate_pair(pair: str) -> str:
    if not re.fullmatch(r"[A-Z0-9]+/USDT:USDT", pair):
        raise ValueError(f"Unsupported futures pair: {pair!r}")
    return pair


@dataclass(frozen=True)
class MarketSnapshot:
    """Point-in-time market state (no lookahead)."""

    pair: str
    timeframe: str
    as_of_ms: int
    ohlcv: pd.DataFrame  # closed candles only, index = close time (UTC)
    funding_rate: float | None  # latest funding rate known at as_of
    next_funding_ms: int | None
    mark_close: float | None
    indicators: dict  # atr, atr_pct, adx, rsi, bb_upper, bb_lower, bb_width, ema20, ema50

    def to_ai_context(self) -> dict:
        """Compact JSON-safe context handed to the LLM (numbers rounded)."""
        last = self.ohlcv.iloc[-1]
        return {
            "pair": self.pair,
            "timeframe": self.timeframe,
            "as_of_utc": datetime.fromtimestamp(self.as_of_ms / 1000, tz=UTC).isoformat(),
            "last_close": round(float(last["close"]), 8),
            "indicators": {
                k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.indicators.items()
            },
            "funding_rate": self.funding_rate,
            "candles_used": len(self.ohlcv),
        }


class BinanceFuturesDataAdapter:
    """Read-only adapter over the existing feather dataset + optional public REST tail."""

    def __init__(self, data_dir: str | Path, realtime: bool = False, rest_timeout: float = 7.0):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data dir not found: {self.data_dir}")
        self.realtime = realtime
        self.rest_timeout = rest_timeout
        self._cache: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ files
    def _path(self, pair: str, tf: str, kind: str) -> Path:
        validate_pair(pair)
        if tf not in SUPPORTED_TF:
            raise ValueError(f"Unsupported timeframe: {tf}")
        return self.data_dir / f"{pair_to_filestem(pair)}-{tf}-{kind}.feather"

    def available_pairs(self) -> list[str]:
        pairs = set()
        for f in self.data_dir.glob("*-futures.feather"):
            stem = f.name.split("-")[0]
            try:
                pairs.add(filestem_to_pair(stem))
            except ValueError:
                continue
        return sorted(pairs)

    def data_range(self, pair: str, tf: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        path = self._path(pair, tf, "futures")
        if not path.exists():
            return None
        df = self._load(path)
        return df.index[0], df.index[-1]

    # ------------------------------------------------------------------ load
    def _load(self, path: Path) -> pd.DataFrame:
        df = self._cache.get(str(path))
        if df is None:
            df = pd.read_feather(path)
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df = df.set_index("date").sort_index()
            if df.index.has_duplicates:
                df = df[~df.index.duplicated(keep="last")]
            self._cache[str(path)] = df
        return df

    # ------------------------------------------------------------------ ohlcv
    def get_ohlcv(
        self,
        pair: str,
        tf: str,
        as_of: datetime | None = None,
        lookback: int = 300,
        include_partial_tail: bool = False,
    ) -> pd.DataFrame:
        """Closed candles with close time <= as_of. File data only (no network)."""
        validate_pair(pair)
        if tf not in SUPPORTED_TF:
            raise ValueError(f"Unsupported timeframe: {tf}")
        path = self._path(pair, tf, "futures")
        if not path.exists():
            raise FileNotFoundError(f"No futures OHLCV data for {pair} {tf}: {path.name}")
        df = self._load(path)
        if as_of is not None:
            if as_of.tzinfo is None:
                as_of = as_of.replace(tzinfo=UTC)
            # A candle with open time t closes at t + TF; it is "known" at close time.
            cutoff = pd.Timestamp(as_of) - pd.Timedelta(milliseconds=TF_MS[tf])
            df = df[df.index <= cutoff]
        if not include_partial_tail and len(df) > lookback:
            df = df.iloc[-lookback:]
        return df

    # ------------------------------------------------------------- funding/mark
    def get_funding(self, pair: str, as_of: datetime | None = None) -> pd.DataFrame:
        path = self._path(pair, "1h", "funding_rate")
        if not path.exists():
            return pd.DataFrame(columns=["funding_rate"])
        df = self._load(path)
        if as_of is not None:
            ts = pd.Timestamp(as_of, tz=UTC) if as_of.tzinfo is None else pd.Timestamp(as_of)
            df = df[df.index <= ts]
        return df

    def get_mark(self, pair: str, tf: str = "1h", as_of: datetime | None = None) -> pd.DataFrame:
        path = self._path(pair, tf, "mark")
        if not path.exists():
            return pd.DataFrame()
        df = self._load(path)
        if as_of is not None:
            ts = pd.Timestamp(as_of, tz=UTC) if as_of.tzinfo is None else pd.Timestamp(as_of)
            df = df[df.index <= ts]
        return df

    # ------------------------------------------------------------ snapshot
    def snapshot(
        self, pair: str, tf: str, as_of: datetime | None = None, lookback: int = 300
    ) -> MarketSnapshot:
        """Full point-in-time snapshot with indicators (used by AI + regime + risk)."""
        ohlcv = self.get_ohlcv(pair, tf, as_of=as_of, lookback=max(lookback, 210))
        if len(ohlcv) < 60:
            raise ValueError(f"Insufficient history for {pair} {tf}: {len(ohlcv)} candles")
        as_of_ms = self._as_of_ms(as_of)
        funding = self.get_funding(pair, as_of=as_of)
        funding_rate: float | None = None
        next_funding_ms: int | None = None
        if len(funding):
            funding_rate = float(funding["funding_rate"].iloc[-1])
            last_ms = int(funding.index[-1].value // 1_000_000)
            next_funding_ms = last_ms + 8 * 3_600_000
        mark = self.get_mark(pair, "1h", as_of=as_of)
        mark_close = float(mark["close"].iloc[-1]) if len(mark) else None
        indicators = compute_indicators(ohlcv)
        return MarketSnapshot(
            pair=pair,
            timeframe=tf,
            as_of_ms=as_of_ms,
            ohlcv=ohlcv,
            funding_rate=funding_rate,
            next_funding_ms=next_funding_ms,
            mark_close=mark_close,
            indicators=indicators,
        )

    @staticmethod
    def _as_of_ms(as_of: datetime | None) -> int:
        if as_of is None:
            return int(time.time() * 1000)
        ts = pd.Timestamp(as_of, tz=UTC) if as_of.tzinfo is None else pd.Timestamp(as_of)
        return int(ts.value // 1_000_000)

    # ------------------------------------------------------------ realtime (dry-run only)
    def fetch_recent_public_klines(self, pair: str, tf: str, limit: int = 100) -> list[dict]:
        """PUBLIC endpoint only. For realtime dry-run tails; never for history backfill."""
        validate_pair(pair)
        if tf not in ("5m", "15m", "1h"):
            raise ValueError("Realtime tail supported for 5m/15m/1h only")
        if not 1 <= limit <= 1000:
            raise ValueError("limit out of bounds")
        url = f"{PUBLIC_KLINES_URL}?symbol={pair_to_rest_symbol(pair)}&interval={tf}&limit={limit}"
        req = urllib.request.Request(url, headers={"User-Agent": "AIGrid-Research/0.1"})
        # The URL is built from the pinned HTTPS public-klines constant above.
        with urllib.request.urlopen(  # nosec B310
            req, timeout=self.rest_timeout
        ) as resp:
            rows = json.load(resp)
        now_ms = int(time.time() * 1000)
        out = []
        for r in rows:
            open_ms, o, h, l, c, vol, close_ms = int(r[0]), r[1], r[2], r[3], r[4], r[5], int(r[6])
            if close_ms >= now_ms:
                continue  # still-forming candle: never returned (no lookahead)
            out.append(
                {
                    "open_time_ms": open_ms,
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "volume": float(vol),
                    "close_time_ms": close_ms,
                }
            )
        return out


# --------------------------------------------------------------------- indicators
def _wilder(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def compute_indicators(ohlcv: pd.DataFrame) -> dict:
    """ATR(14), ADX(14), RSI(14), BB(20,2), EMA20/50 — computed on closed candles only."""
    high, low, close = ohlcv["high"], ohlcv["low"], ohlcv["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = _wilder(tr, 14)
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr_s = _wilder(tr, 14)
    plus_di = 100 * _wilder(plus_dm, 14) / atr_s
    minus_di = 100 * _wilder(minus_dm, 14) / atr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA)
    adx = _wilder(dx.fillna(0.0), 14)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    rs = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / loss.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean().replace(0, pd.NA)
    rsi = 100 - 100 / (1 + rs)
    ma = close.rolling(20).mean()
    sd = close.rolling(20).std(ddof=0)
    bb_upper = ma + 2 * sd
    bb_lower = ma - 2 * sd
    bb_width = (bb_upper - bb_lower) / ma.replace(0, pd.NA)
    out = {
        "atr": float(atr.iloc[-1]),
        "atr_pct": float(atr.iloc[-1] / close.iloc[-1]),
        "adx": float(adx.iloc[-1]),
        "plus_di": float(plus_di.iloc[-1]),
        "minus_di": float(minus_di.iloc[-1]),
        "rsi": float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None,
        "bb_upper": float(bb_upper.iloc[-1]),
        "bb_lower": float(bb_lower.iloc[-1]),
        "bb_width": float(bb_width.iloc[-1]) if pd.notna(bb_width.iloc[-1]) else None,
        "ema20": float(close.ewm(span=20, adjust=False).mean().iloc[-1]),
        "ema50": float(close.ewm(span=50, adjust=False).mean().iloc[-1]),
    }
    return {
        k: (round(v, 8) if isinstance(v, float) and pd.notna(v) else None) for k, v in out.items()
    }
