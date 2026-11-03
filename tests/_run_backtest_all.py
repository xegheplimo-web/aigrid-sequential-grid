# ruff: noqa: E402 -- standalone scripts bootstrap the repository before imports.
"""Batch backtest run — config C (no AI) on all pairs, real saved data."""

import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigrid.backtest import GridBacktester
from aigrid.settings import AIGridSettings

DATA_DIR = ROOT / "user_data/data/binance/futures"
PAIRS = [
    "ETH/USDT:USDT",
    "BTC/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "ADA/USDT:USDT",
    "DOGE/USDT:USDT",
    "LINK/USDT:USDT",
]
START = datetime(2026, 9, 1, tzinfo=UTC)
END = datetime(2026, 10, 11, tzinfo=UTC)


def main():
    settings = AIGridSettings(
        enabled=False, mode="backtest", symbols=tuple(PAIRS), timeframe="15m", data_dir=DATA_DIR
    )
    rows = []
    for pair in PAIRS:
        try:
            bt = GridBacktester(settings, config="C")
            t0 = time.monotonic()
            r = bt.run(pair, START, END, initial_usdt=1000.0)
            rows.append(r.summary() | {"elapsed": round(time.monotonic() - t0, 1)})
        except Exception as exc:
            rows.append({"pair": pair, "error": f"{type(exc).__name__}: {exc}"})
    hdr = (
        f"{'pair':<16} {'net%':>8} {'DD%':>7} {'trades':>6} {'WR':>6} "
        f"{'PF':>6} {'fees':>7} {'fund':>7} {'plans':>5} {'trans':>5} {'sec':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['pair']:<16} ERROR {r['error']}")
            continue
        print(
            f"{r['pair']:<16} {r['net_profit_pct']:>8.2f} "
            f"{r['max_drawdown_pct']:>7.2f} {r['trade_count']:>6} "
            f"{r['win_rate']:>6.2f} "
            f"{r['profit_factor'] if r['profit_factor'] is not None else float('nan'):>6.2f} "
            f"{r['total_fees_usdt']:>7.2f} {r['total_funding_usdt']:>7.2f} "
            f"{r['plans_adopted']:>5} {r['regime_transitions']:>5} {r['elapsed']:>5.1f}"
        )
    import json

    out = ROOT / "reports" / "backtest_C_capital_model.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, default=str))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
