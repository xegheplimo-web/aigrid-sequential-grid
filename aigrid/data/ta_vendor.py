"""Binance Futures vendor for TradingAgents (runtime registration, no fork).

Upstream TradingAgents routes market data through `VENDOR_METHODS` (public
registry) and `data_vendors` config. This module registers a `binance_futures`
vendor for the two categories the market analyst uses:

  - get_stock_data   (core_stock_apis)      -> OHLCV from existing feather files
  - get_indicators   (technical_indicators) -> stockstats indicators on feather data

It also points the deterministic "verified market snapshot" at the same data by
rebinding `snapshot.load_ohlcv` in the upstream module namespace (contained,
reversible runtime patch — upstream source files are NOT modified).

Rationale (task constraint): the market analyst must analyze BINANCE FUTURES
data, not Yahoo stock/spot substitutes. All exact price claims are therefore
grounded in the same feather dataset used for backtesting.

No-lookahead: every function filters rows with close time <= requested date.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .binance_adapter import TF_MS, BinanceFuturesDataAdapter, validate_pair

_REGISTER_LOCK = threading.Lock()
_REGISTERED = False

# The timeframe served to the analyst for the current run (set per analysis).
_CURRENT_TF: str = "15m"
_ADAPTERS: dict[str, BinanceFuturesDataAdapter] = {}
_TF_LOCK = threading.Lock()


def set_served_timeframe(tf: str) -> None:
    global _CURRENT_TF
    if tf not in TF_MS:
        raise ValueError(f"Unsupported timeframe: {tf}")
    with _TF_LOCK:
        _CURRENT_TF = tf


def get_served_timeframe() -> str:
    with _TF_LOCK:
        return _CURRENT_TF


def _adapter_for(data_dir: Path) -> BinanceFuturesDataAdapter:
    """One adapter per data_dir, cached on the module (read-only, cheap)."""
    key = str(data_dir)
    adapter = _ADAPTERS.get(key)
    if adapter is None:
        adapter = BinanceFuturesDataAdapter(data_dir)
        _ADAPTERS[key] = adapter
    return adapter


def _symbol_to_pair(symbol: str) -> str:
    """ETHUSDT / ETHUSDT:USDT / ETH/USDT:USDT -> ETH/USDT:USDT"""
    s = symbol.strip().upper()
    if s.endswith(":USDT"):
        s = s[:-5]
    if "/" in s:
        validate_pair(s if s.endswith(":USDT") else s + ":USDT")
        return s + ":USDT"
    if s.endswith("USDT") and len(s) > 4:
        return s[:-4] + "/USDT:USDT"
    raise ValueError(f"Unsupported symbol for Binance futures vendor: {symbol!r}")


def _as_of_ts(as_of_date: str) -> pd.Timestamp:
    dt = datetime.strptime(as_of_date, "%Y-%m-%d").replace(tzinfo=UTC)
    return pd.Timestamp(dt) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)


def _ohlcv_frame(symbol: str, as_of_date: str, data_dir: Path) -> pd.DataFrame:
    """Closed candles up to end of as_of_date, standardized to stockstats shape."""
    pair = _symbol_to_pair(symbol)
    adapter = _adapter_for(data_dir)
    tf = get_served_timeframe()
    end_ts = _as_of_ts(as_of_date)
    df = adapter.get_ohlcv(pair, tf, as_of=end_ts.to_pydatetime(), lookback=10_000)
    out = df.reset_index()
    out = out.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    out["Date"] = out["Date"].dt.tz_localize(None)
    return out[["Date", "Open", "High", "Low", "Close", "Volume"]]


# ---------------------------------------------------------------------- vendors
def binance_get_stock_data(
    symbol: str, start_date: str, end_date: str, data_dir: Path | None = None
) -> str:
    """Upstream contract: header + CSV of OHLCV rows between start and end dates."""
    from tradingagents.dataflows.errors import NoMarketDataError

    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")
    if data_dir is None:
        data_dir = _default_data_dir()
    df = _ohlcv_frame(symbol, end_date, data_dir)
    start_ts = pd.Timestamp(datetime.strptime(start_date, "%Y-%m-%d"))
    df = df[df["Date"] >= start_ts]
    if df.empty:
        raise NoMarketDataError(
            symbol, symbol, f"no futures rows between {start_date} and {end_date}"
        )
    csv_string = df.to_csv(index=False)
    header = (
        f"# Binance USDT-M futures data for {symbol} ({get_served_timeframe()} candles, "
        f"UTC) from {start_date} to {end_date}\n"
        f"# Source: local feather dataset (no lookahead; closed candles only)\n"
        f"# Total records: {len(df)}\n\n"
    )
    return header + csv_string


def binance_get_indicators(
    symbol: str, indicator: str, as_of_date: str, look_back_days: int, data_dir: Path | None = None
) -> str:
    """Upstream contract: '## {indicator} values ...' + per-date values + description."""
    from datetime import datetime as _dt

    from stockstats import wrap

    _dt.strptime(as_of_date, "%Y-%m-%d")
    if data_dir is None:
        data_dir = _default_data_dir()
    look_back_days = max(1, min(int(look_back_days), 90))
    df = _ohlcv_frame(symbol, as_of_date, data_dir)
    if df.empty:
        from tradingagents.dataflows.errors import NoMarketDataError

        raise NoMarketDataError(symbol, symbol, f"no futures rows on or before {as_of_date}")
    stock_df = wrap(df.copy())
    try:
        stock_df[indicator]
    except Exception as exc:
        return f"Indicator {indicator} not available on Binance futures data: {exc}"
    as_of_dt = pd.Timestamp(_dt.strptime(as_of_date, "%Y-%m-%d"))
    before = as_of_dt - pd.Timedelta(days=look_back_days)
    mask = (stock_df["Date"] >= before) & (stock_df["Date"] <= as_of_dt + pd.Timedelta(days=1))
    window = stock_df[mask]
    if window.empty:
        return f"No indicator rows for {indicator} between {before.date()} and {as_of_date}"
    # One value per calendar day (last candle of that day) to match upstream cadence.
    per_day = window.assign(day=window["Date"].dt.date).groupby("day", as_index=False).last()
    lines = []
    for _, row in per_day.iterrows():
        value = row[indicator]
        lines.append(f"{row['day']}: {_fmt_num(value)}")
    desc = _INDICATOR_DOCS.get(indicator, "No description available.")
    return (
        f"## {indicator} values from {before.date()} to {as_of_date} "
        f"(Binance futures {get_served_timeframe()}, UTC):\n\n" + "\n".join(lines) + "\n\n" + desc
    )


def binance_load_ohlcv(
    symbol: str, as_of_date: str, fill_gaps: bool = False, data_dir: Path | None = None
) -> pd.DataFrame:
    """Replacement for yahoo.snapshot.load_ohlcv — same shape (Date/Open/High/Low/Close/Volume)."""
    from tradingagents.dataflows.errors import NoMarketDataError

    if data_dir is None:
        data_dir = _default_data_dir()
    df = _ohlcv_frame(symbol, as_of_date, data_dir)
    if df.empty:
        raise NoMarketDataError(symbol, symbol, f"no futures rows on or before {as_of_date}")
    return df


def _fmt_num(value) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if f != f:  # NaN
        return "N/A"
    return f"{f:.8g}"


_INDICATOR_DOCS = {
    "close_50_sma": "50 SMA: medium-term trend indicator.",
    "close_200_sma": "200 SMA: long-term trend benchmark.",
    "close_10_ema": "10 EMA: responsive short-term average.",
    "macd": "MACD: momentum via EMA differences.",
    "macds": "MACD signal line.",
    "macdh": "MACD histogram.",
    "rsi": "RSI: momentum, overbought/oversold.",
    "boll": "Bollinger middle band (20 SMA).",
    "boll_ub": "Bollinger upper band.",
    "boll_lb": "Bollinger lower band.",
    "atr": "ATR: average true range (volatility).",
    "vwma": "Volume-weighted moving average.",
}


def _default_data_dir() -> Path:
    explicit = globals().get("_DEFAULT_DATA_DIR")
    if explicit is not None:
        return Path(explicit)
    try:
        from ..settings import AIGridSettings

        return AIGridSettings().data_dir
    except Exception:
        return Path("user_data/data/binance/futures")


# ------------------------------------------------------------------ registration
def register_binance_vendor(data_dir: str | Path | None = None) -> None:
    """Register `binance_futures` into the upstream registry + point config at it.

    Idempotent. Modifies ONLY runtime registries (upstream source untouched):
      - VENDOR_METHODS['get_stock_data' | 'get_indicators']['binance_futures']
      - snapshot.load_ohlcv rebinding (verified snapshot uses Binance futures)
    """
    global _REGISTERED
    with _REGISTER_LOCK:
        if data_dir is not None:
            globals()["_DEFAULT_DATA_DIR"] = Path(data_dir)
        from tradingagents.dataflows import router as _router
        from tradingagents.dataflows.vendors.yahoo import snapshot as _snapshot

        _router.VENDOR_METHODS["get_stock_data"]["binance_futures"] = binance_get_stock_data
        _router.VENDOR_METHODS["get_indicators"]["binance_futures"] = binance_get_indicators
        _snapshot.load_ohlcv = binance_load_ohlcv
        _REGISTERED = True


def configure_vendor_config(cfg: dict, data_dir: str | Path | None = None) -> dict:
    """Set data_vendors so the market analyst uses the binance_futures vendor."""
    register_binance_vendor(data_dir)
    vendors = dict(cfg.get("data_vendors", {}))
    vendors["core_stock_apis"] = "binance_futures"
    vendors["technical_indicators"] = "binance_futures"
    cfg["data_vendors"] = vendors
    return cfg
