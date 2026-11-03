"""Contract tests for the file-backed market-data adapter.

These run on a **synthetic** feather dataset written into a tmp dir, so they are
hermetic: no network, no multi-gigabyte recording. They pin the properties the
rest of the engine depends on:

* filename <-> pair <-> REST symbol round-tripping,
* no lookahead: a candle is only visible once it has closed,
* deterministic trimming (`as_of`, `lookback`, duplicate rows),
* funding/mark reads degrade to empty rather than raising,
* `snapshot()` refuses to invent history it does not have,
* `validate_pair` accepts only futures pairs.

The recorded-data assertions live in the (skipped-by-default) replay tier; this
module exists so the adapter is still exercised on a clean clone.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from aigrid.data.binance_adapter import (
    TF_MS,
    BinanceFuturesDataAdapter,
    filestem_to_pair,
    pair_to_filestem,
    pair_to_rest_symbol,
    validate_pair,
)

PAIR = "SOL/USDT:USDT"
TF = "1h"
START = datetime(2026, 1, 1, tzinfo=UTC)
N = 300


def _write_frame(path, frame: pd.DataFrame) -> None:
    frame.rename_axis("date").reset_index().to_feather(path)


def _ohlcv_frame(start: datetime = START, n: int = N, tf: str = TF) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq=pd.Timedelta(milliseconds=TF_MS[tf]), tz="UTC")
    close = pd.Series(range(n), index=idx, dtype="float64") * 0.5 + 100.0
    return pd.DataFrame(
        {
            "open": close - 0.25,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1_000.0,
        },
        index=idx,
    )


@pytest.fixture()
def data_dir(tmp_path):
    d = tmp_path / "futures"
    d.mkdir()
    stem = pair_to_filestem(PAIR)
    _write_frame(d / f"{stem}-{TF}-futures.feather", _ohlcv_frame())
    _write_frame(d / f"{stem}-{TF}-mark.feather", _ohlcv_frame())
    funding_idx = pd.date_range(START, periods=40, freq="8h", tz="UTC")
    _write_frame(
        d / f"{stem}-1h-funding_rate.feather",
        pd.DataFrame({"funding_rate": 0.0001}, index=funding_idx),
    )
    return d


# --------------------------------------------------------------- naming contract
def test_pair_naming_roundtrip():
    assert pair_to_filestem("ETH/USDT:USDT") == "ETH_USDT_USDT"
    assert filestem_to_pair("ETH_USDT_USDT") == "ETH/USDT:USDT"
    assert pair_to_rest_symbol("ETH/USDT:USDT") == "ETHUSDT"
    assert pair_to_filestem("1000PEPE/USDT:USDT") == "1000PEPE_USDT_USDT"
    assert filestem_to_pair("1000PEPE_USDT_USDT") == "1000PEPE/USDT:USDT"


@pytest.mark.parametrize("bad", ["eth/usdt", "ETH/USDT", "ETHUSDT", "", "ETH/USDT:BUSD"])
def test_validate_pair_rejects(bad):
    with pytest.raises(ValueError):
        validate_pair(bad)


def test_validate_pair_accepts_futures_pair():
    assert validate_pair("ETH/USDT:USDT") == "ETH/USDT:USDT"


def test_available_pairs_reads_filenames(data_dir):
    adapter = BinanceFuturesDataAdapter(data_dir)
    assert adapter.available_pairs() == [PAIR]
    # a mark/funding-only file must not be reported as an OHLCV pair
    assert adapter.data_range(PAIR, TF) is not None


def test_missing_data_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        BinanceFuturesDataAdapter(tmp_path / "nope")


# --------------------------------------------------------------- no-lookahead
def test_partial_candle_is_hidden_until_it_closes(data_dir):
    adapter = BinanceFuturesDataAdapter(data_dir)
    last_open = pd.Timestamp(START) + pd.Timedelta(milliseconds=TF_MS[TF]) * (N - 1)

    # 1 ms after the last candle opens: it is still forming -> must be excluded
    just_before_close = last_open + pd.Timedelta(milliseconds=1)
    df = adapter.get_ohlcv(PAIR, TF, as_of=just_before_close.to_pydatetime(), lookback=N)
    assert df.index.max() == last_open - pd.Timedelta(milliseconds=TF_MS[TF])

    # exactly at close time it becomes known
    df2 = adapter.get_ohlcv(PAIR, TF, as_of=(last_open + pd.Timedelta(hours=1)).to_pydatetime())
    assert df2.index.max() == last_open


def test_naive_as_of_is_treated_as_utc(data_dir):
    adapter = BinanceFuturesDataAdapter(data_dir)
    naive = datetime(2026, 1, 5, 0, 0)
    aware = naive.replace(tzinfo=UTC)
    a = adapter.get_ohlcv(PAIR, TF, as_of=naive, lookback=N)
    b = adapter.get_ohlcv(PAIR, TF, as_of=aware, lookback=N)
    assert a.index.equals(b.index)


def test_lookback_trims_the_tail_only(data_dir):
    adapter = BinanceFuturesDataAdapter(data_dir)
    df = adapter.get_ohlcv(PAIR, TF, lookback=50)
    assert len(df) == 50
    assert df.index.is_monotonic_increasing
    assert adapter.get_ohlcv(PAIR, TF, lookback=10_000).index[0] == pd.Timestamp(START)


def test_duplicate_rows_keep_last(data_dir):
    stem = pair_to_filestem(PAIR)
    frame = _ohlcv_frame(n=10)
    dup = pd.concat([frame, frame.iloc[[-1]].assign(close=999.0)])
    _write_frame(data_dir / f"{stem}-{TF}-futures.feather", dup)
    df = BinanceFuturesDataAdapter(data_dir).get_ohlcv(PAIR, TF, lookback=100)
    assert not df.index.has_duplicates
    assert df["close"].iloc[-1] == 999.0


def test_missing_pair_file_raises(data_dir):
    adapter = BinanceFuturesDataAdapter(data_dir)
    with pytest.raises(FileNotFoundError):
        adapter.get_ohlcv("DOGE/USDT:USDT", TF)


def test_unsupported_timeframe_rejected(data_dir):
    adapter = BinanceFuturesDataAdapter(data_dir)
    with pytest.raises(ValueError):
        adapter.get_ohlcv(PAIR, "7m")
    assert "7m" not in TF_MS


# --------------------------------------------------------------- funding / mark
def test_funding_and_mark_load_and_respect_as_of(data_dir):
    adapter = BinanceFuturesDataAdapter(data_dir)
    funding = adapter.get_funding(PAIR)
    mark = adapter.get_mark(PAIR)
    assert len(funding) == 40
    assert len(mark) == N
    assert {"funding_rate"} <= set(funding.columns)
    assert {"close"} <= set(mark.columns)

    cutoff = datetime(2026, 1, 3, tzinfo=UTC)
    assert adapter.get_funding(PAIR, as_of=cutoff).index.max() <= pd.Timestamp(cutoff)
    assert adapter.get_mark(PAIR, as_of=cutoff).index.max() <= pd.Timestamp(cutoff)


def test_funding_and_mark_absent_degrade_to_empty(tmp_path):
    d = tmp_path / "futures"
    d.mkdir()
    _write_frame(d / f"{pair_to_filestem(PAIR)}-{TF}-futures.feather", _ohlcv_frame())
    adapter = BinanceFuturesDataAdapter(d)
    assert adapter.get_funding(PAIR).empty
    assert adapter.get_mark(PAIR).empty


# --------------------------------------------------------------- snapshot
def test_snapshot_refuses_insufficient_history(tmp_path):
    d = tmp_path / "futures"
    d.mkdir()
    _write_frame(d / f"{pair_to_filestem(PAIR)}-{TF}-futures.feather", _ohlcv_frame(n=30))
    adapter = BinanceFuturesDataAdapter(d)
    with pytest.raises(ValueError, match="Insufficient history"):
        adapter.snapshot(PAIR, TF)


def test_snapshot_wires_funding_mark_and_indicators(data_dir):
    adapter = BinanceFuturesDataAdapter(data_dir)
    snap = adapter.snapshot(PAIR, TF)
    assert snap.pair == PAIR and snap.timeframe == TF
    assert snap.funding_rate == pytest.approx(0.0001)
    assert snap.next_funding_ms is not None
    assert snap.mark_close is not None
    for key in ("atr", "adx", "rsi", "bb_upper", "bb_lower", "ema20", "ema50"):
        assert key in snap.indicators, key
    assert snap.as_of_ms > 0
    # the snapshot must not contain a still-forming candle
    assert snap.ohlcv.index.max() <= pd.Timestamp.now(tz="UTC")


def test_snapshot_as_of_is_deterministic(data_dir):
    adapter = BinanceFuturesDataAdapter(data_dir)
    as_of = datetime(2026, 1, 5, tzinfo=UTC)
    a = adapter.snapshot(PAIR, TF, as_of=as_of)
    b = adapter.snapshot(PAIR, TF, as_of=as_of)
    assert a.as_of_ms == b.as_of_ms == int(pd.Timestamp(as_of).value // 1_000_000)
    assert a.ohlcv.equals(b.ohlcv)
    assert a.indicators == b.indicators
