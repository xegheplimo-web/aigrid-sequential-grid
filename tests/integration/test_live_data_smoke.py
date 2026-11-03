"""Live-data smoke test: public Binance endpoints -> adapter -> indicators.

This is the only test that touches the network. It is skipped unless
``AIGRID_LIVE_DATA=1`` so the default suite stays hermetic and deterministic.

What it proves that the unit suite cannot:
  * the filestem <-> pair mapping matches real Freqtrade futures filenames,
  * a freshly downloaded frame loads with a tz-aware UTC index,
  * ``as_of`` trimming never leaks a still-open candle (no lookahead),
  * the indicator stack runs on real values without NaN explosions.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
from aigrid.data.binance_adapter import (
    BinanceFuturesDataAdapter,
    compute_indicators,
    pair_to_filestem,
)

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("AIGRID_LIVE_DATA") != "1",
        reason="network test: set AIGRID_LIVE_DATA=1 to enable",
    ),
]

SYMBOL = "SOL/USDT:USDT"
TIMEFRAME = "5m"


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("live_fixtures")

    from scripts.fetch_fixtures import fetch, pair_stem

    df = fetch("SOLUSDT", TIMEFRAME, 400)
    assert not df.empty, "Binance returned no candles"
    df.to_feather(out / f"{pair_stem('SOLUSDT')}-{TIMEFRAME}-futures.feather")
    return out


def test_filestem_roundtrip() -> None:
    assert pair_to_filestem(SYMBOL) == "SOL_USDT_USDT"


def test_adapter_reads_live_frame(data_dir: Path) -> None:
    adapter = BinanceFuturesDataAdapter(data_dir=data_dir)
    df = adapter.get_ohlcv(SYMBOL, TIMEFRAME, lookback=100)
    assert len(df) == 100
    assert list(df.columns)[:5] == ["open", "high", "low", "close", "volume"]
    assert df.index.tz is not None
    assert df.index.is_monotonic_increasing
    assert (df["high"] >= df["low"]).all()


def test_as_of_never_returns_open_candle(data_dir: Path) -> None:
    adapter = BinanceFuturesDataAdapter(data_dir=data_dir)
    full = adapter.get_ohlcv(SYMBOL, TIMEFRAME, lookback=200)
    probe = full.index[-1] + pd.Timedelta(minutes=1)  # one minute into the last candle
    trimmed = adapter.get_ohlcv(SYMBOL, TIMEFRAME, as_of=probe.to_pydatetime(), lookback=200)
    assert trimmed.index[-1] < full.index[-1], "an unfinished candle leaked into the snapshot"


def test_indicators_are_finite_on_real_data(data_dir: Path) -> None:
    adapter = BinanceFuturesDataAdapter(data_dir=data_dir)
    df = adapter.get_ohlcv(SYMBOL, TIMEFRAME, lookback=300)
    ind = compute_indicators(df)
    assert ind, "no indicators computed"
    for key, value in ind.items():
        if isinstance(value, float):
            assert value == value, f"{key} is NaN"  # NaN != NaN
