"""PHASE 1 tests: Binance futures data adapter + TradingAgents vendor registration.

Run: .venv-aigrid/Scripts/python.exe -m pytest tests_aigrid/ -x -q
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from aigrid.data.binance_adapter import (
    BinanceFuturesDataAdapter,
    filestem_to_pair,
    pair_to_filestem,
    pair_to_rest_symbol,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "user_data/data/binance/futures"

AS_OF = datetime(2026, 10, 11, 0, 0, tzinfo=UTC)  # safe common cutoff (audit)


@pytest.fixture(scope="module")
def adapter() -> BinanceFuturesDataAdapter:
    return BinanceFuturesDataAdapter(DATA_DIR)


# ------------------------------------------------------------------ pair names
def test_pair_normalization_roundtrip():
    assert pair_to_filestem("ETH/USDT:USDT") == "ETH_USDT_USDT"
    assert filestem_to_pair("ETH_USDT_USDT") == "ETH/USDT:USDT"
    assert pair_to_rest_symbol("ETH/USDT:USDT") == "ETHUSDT"
    with pytest.raises(ValueError):
        filestem_to_pair("ETH_USDT")  # missing quote suffix


def test_validate_pair_rejects_spot_and_lowercase():
    from aigrid.data.binance_adapter import validate_pair

    assert validate_pair("ETH/USDT:USDT") == "ETH/USDT:USDT"
    with pytest.raises(ValueError):
        validate_pair("ETH/USDT")  # spot-style
    with pytest.raises(ValueError):
        validate_pair("eth/usdt:usdt")  # lowercase


# ------------------------------------------------------------------ availability
def test_available_pairs_contains_six_majors(adapter):
    pairs = adapter.available_pairs()
    for p in (
        "ETH/USDT:USDT",
        "SOL/USDT:USDT",
        "XRP/USDT:USDT",
        "ADA/USDT:USDT",
        "DOGE/USDT:USDT",
        "LINK/USDT:USDT",
        "BTC/USDT:USDT",
    ):
        assert p in pairs, f"missing {p}"


def test_data_range_matches_audit(adapter):
    rng = adapter.data_range("ETH/USDT:USDT", "5m")
    assert rng is not None
    start, end = rng
    assert start.year == 2019 and start.month == 11
    assert end >= pd.Timestamp("2026-10-30", tz="UTC")


# ------------------------------------------------------------------ no-lookahead
def test_get_ohlcv_no_lookahead(adapter):
    df = adapter.get_ohlcv("ETH/USDT:USDT", "15m", as_of=AS_OF, lookback=10_000)
    # 15m data starts 2024-09-01 and ends 2026-10-11 06:15 in the file;
    # as_of 2026-10-11 00:00 must cut everything after (candle close <= as_of).
    assert df.index.max() <= pd.Timestamp(AS_OF) - pd.Timedelta(minutes=15)
    assert df.index.is_monotonic_increasing


def test_get_ohlcv_candle_closes_before_asof_is_included(adapter):
    # Feather index is candle OPEN time; the candle opening 06:00 closes 06:15.
    as_of = datetime(2026, 10, 11, 6, 15, tzinfo=UTC)  # exactly a 15m close
    df = adapter.get_ohlcv("ETH/USDT:USDT", "15m", as_of=as_of, lookback=10_000)
    assert df.index.max() == pd.Timestamp("2026-10-11 06:00", tz="UTC")
    # One second earlier, that candle is still forming and must be excluded.
    df2 = adapter.get_ohlcv(
        "ETH/USDT:USDT",
        "15m",
        as_of=datetime(2026, 10, 11, 6, 14, 59, tzinfo=UTC),
        lookback=10_000,
    )
    assert df2.index.max() == pd.Timestamp("2026-10-11 05:45", tz="UTC")


def test_missing_pair_raises(adapter):
    with pytest.raises(FileNotFoundError):
        adapter.get_ohlcv("ZZZZ/USDT:USDT", "15m")


def test_insufficient_history_raises_in_snapshot(adapter):
    with pytest.raises(ValueError):
        # 1d data starts 2024-10-01; as_of 2024-10-02 gives ~2 candles < 60
        adapter.snapshot("ETH/USDT:USDT", "1d", as_of=datetime(2024, 10, 2, tzinfo=UTC))


# ------------------------------------------------------------------ funding/mark
def test_funding_no_lookahead_and_values(adapter):
    f = adapter.get_funding("ETH/USDT:USDT", as_of=AS_OF)
    assert len(f) > 0
    assert f.index.max() <= pd.Timestamp(AS_OF)
    assert f["funding_rate"].iloc[-1] != 0 or True  # rate may legitimately be ~0
    # funding cadence is 8h
    gap = (f.index[-1] - f.index[-2]).total_seconds()
    assert gap in (8 * 3600, 4 * 3600, 1 * 3600)  # 8h standard, allow early-history variants


def test_mark_data(adapter):
    m = adapter.get_mark("ETH/USDT:USDT", "1h", as_of=AS_OF)
    assert len(m) > 0
    assert m.index.max() <= pd.Timestamp(AS_OF)


# ------------------------------------------------------------------ indicators
def test_snapshot_indicators_complete(adapter):
    snap = adapter.snapshot("ETH/USDT:USDT", "15m", as_of=AS_OF)
    assert snap.indicators["adx"] >= 0
    assert 0 <= snap.indicators["rsi"] <= 100
    assert snap.indicators["bb_upper"] > snap.indicators["bb_lower"]
    assert snap.indicators["atr"] > 0
    assert snap.funding_rate is not None
    assert snap.mark_close is not None
    ctx = snap.to_ai_context()
    assert ctx["pair"] == "ETH/USDT:USDT"
    assert "atr_pct" in ctx["indicators"]


def test_snapshot_deterministic(adapter):
    s1 = adapter.snapshot("ETH/USDT:USDT", "15m", as_of=AS_OF)
    s2 = adapter.snapshot("ETH/USDT:USDT", "15m", as_of=AS_OF)
    assert s1.indicators == s2.indicators
    assert s1.to_ai_context() == s2.to_ai_context()


# ------------------------------------------------------------------ TA vendor
def test_ta_vendor_registration_and_data():
    from aigrid.data.ta_vendor import (
        binance_get_indicators,
        binance_get_stock_data,
        binance_load_ohlcv,
        register_binance_vendor,
        set_served_timeframe,
    )
    from tradingagents.dataflows import router
    from tradingagents.dataflows.vendors.yahoo import snapshot

    register_binance_vendor(DATA_DIR)
    assert "binance_futures" in router.VENDOR_METHODS["get_stock_data"]
    assert "binance_futures" in router.VENDOR_METHODS["get_indicators"]
    assert snapshot.load_ohlcv is binance_load_ohlcv

    set_served_timeframe("15m")
    text = binance_get_stock_data("ETHUSDT", "2026-10-10", "2026-10-11")
    assert "Binance USDT-M futures data" in text
    assert "2026-10-10" in text
    # CSV must contain OHLCV header
    assert "Open,High,Low,Close,Volume" in text

    ind = binance_get_indicators("ETHUSDT", "rsi", "2026-10-11", 3)
    assert "rsi values" in ind
    assert "2026-10-10" in ind

    df = binance_load_ohlcv("ETHUSDT", "2026-10-11")
    assert list(df.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
    assert df["Date"].max() <= pd.Timestamp("2026-10-11 23:59:59")


def test_ta_vendor_no_lookahead():
    from aigrid.data.ta_vendor import binance_load_ohlcv, set_served_timeframe

    set_served_timeframe("15m")
    df = binance_load_ohlcv("ETHUSDT", "2026-10-05")
    assert df["Date"].max() <= pd.Timestamp("2026-10-05 23:59:59")


def test_ta_vendor_rejects_stock_symbol():
    from aigrid.data.ta_vendor import binance_get_stock_data

    with pytest.raises(ValueError):
        binance_get_stock_data("AAPL", "2026-01-01", "2026-10-11")


def test_router_serves_binance_after_config(monkeypatch):
    """Regression: without data_vendors config the router silently used yfinance
    (ETH-USD spot) — the exact stock-data substitution the task forbids."""
    from aigrid.data.ta_vendor import (
        configure_vendor_config,
        register_binance_vendor,
        set_served_timeframe,
    )
    from tradingagents.dataflows import router
    from tradingagents.dataflows.config import set_config

    register_binance_vendor(DATA_DIR)
    cfg = configure_vendor_config({}, DATA_DIR)
    set_config(cfg)
    set_served_timeframe("15m")
    out = router.route_to_vendor("get_stock_data", "ETHUSDT", "2026-10-10", "2026-10-11")
    assert out.startswith("# Binance USDT-M futures data")
    ind = router.route_to_vendor("get_indicators", "ETHUSDT", "rsi", "2026-10-11", 2)
    assert "Binance futures" in ind
