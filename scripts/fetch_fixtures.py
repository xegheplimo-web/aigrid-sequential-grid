"""Download a small slice of public Binance USDT-M futures candles.

The unit suite is deterministic and needs no market data, but a subset of the
tests replays recorded candles from ``user_data/data/binance/futures``. Those
recordings are multi-gigabyte files that are never committed, so this script
fetches a representative slice instead.

Usage
-----
    python scripts/fetch_fixtures.py                     # 4 pairs, 1500 candles
    python scripts/fetch_fixtures.py --pairs 8 --candles 3000
    python scripts/fetch_fixtures.py --symbols ETHUSDT,SOLUSDT

Only public endpoints are used (no API key, no account access).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

KLINE_URL = "https://fapi.binance.com/fapi/v1/klines"
MAX_LIMIT = 1500
DEFAULT_SYMBOLS = [
    "ETHUSDT",
    "SOLUSDT",
    "DOGEUSDT",
    "XRPUSDT",
    "LINKUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "SUIUSDT",
]
DEFAULT_TIMEFRAMES = ["5m", "15m", "1h"]
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "user_data" / "data" / "binance" / "futures"
COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def pair_stem(symbol: str) -> str:
    """ETHUSDT -> ETH_USDT_USDT (Freqtrade futures filestem)."""
    if not symbol.endswith("USDT"):
        raise ValueError(f"Unsupported symbol: {symbol!r}")
    base = symbol[: -len("USDT")]
    return f"{base}_USDT_USDT"


def _request(symbol: str, interval: str, limit: int, end_time: int | None) -> list[list]:
    params = f"symbol={symbol}&interval={interval}&limit={limit}"
    if end_time is not None:
        params += f"&endTime={end_time}"
    req = urllib.request.Request(
        f"{KLINE_URL}?{params}", headers={"User-Agent": "aigrid-fixtures/1.0"}
    )
    # KLINE_URL is a pinned HTTPS constant; no caller-supplied scheme reaches here.
    with urllib.request.urlopen(  # nosec B310
        req, timeout=30
    ) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected klines payload for {symbol}: {payload!r}")
    return payload


def fetch(symbol: str, interval: str, candles: int) -> pd.DataFrame:
    rows: list[list] = []
    end_time: int | None = None
    while len(rows) < candles:
        want = min(MAX_LIMIT, candles - len(rows))
        batch = _request(symbol, interval, want, end_time)
        if not batch:
            break
        rows = batch + rows
        end_time = int(batch[0][0]) - 1
        if len(batch) < want:
            break
        time.sleep(0.25)  # stay well inside the public rate limit

    df = pd.DataFrame(
        [
            {
                "date": pd.to_datetime(r[0], unit="ms", utc=True),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]),
            }
            for r in rows
        ]
    )
    if df.empty:
        return df
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    return df[COLUMNS]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=4, help="how many symbols to fetch")
    parser.add_argument("--candles", type=int, default=1500, help="candles per symbol/timeframe")
    parser.add_argument("--symbols", default="", help="comma-separated Binance symbols")
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args(argv)

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else DEFAULT_SYMBOLS[: max(1, args.pairs)]
    )
    timeframes = [tf.strip() for tf in args.timeframes.split(",") if tf.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    failures = 0
    for symbol in symbols:
        for tf in timeframes:
            try:
                df = fetch(symbol, tf, args.candles)
            except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
                print(f"  !! {symbol} {tf}: {exc}", file=sys.stderr)
                failures += 1
                continue
            if df.empty:
                print(f"  !! {symbol} {tf}: empty payload", file=sys.stderr)
                failures += 1
                continue
            path = out / f"{pair_stem(symbol)}-{tf}-futures.feather"
            df.to_feather(path)
            print(
                f"  ok {path.name}: {len(df)} candles {df['date'].iloc[0]} -> {df['date'].iloc[-1]}"
            )

    print(f"\n{len(symbols) * len(timeframes) - failures} file(s) written to {out}")
    return 1 if failures and not any(out.iterdir()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
