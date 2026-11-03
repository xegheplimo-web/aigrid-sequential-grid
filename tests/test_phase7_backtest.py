"""PHASE 7 tests: backtest harness on real data (config C — no AI)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from aigrid.backtest import GridBacktester
from aigrid.settings import AIGridSettings

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "user_data/data/binance/futures"

START = datetime(2026, 9, 1, tzinfo=UTC)
END = datetime(2026, 10, 11, tzinfo=UTC)


@pytest.fixture(scope="module")
def settings():
    return AIGridSettings(
        enabled=False,
        mode="backtest",
        symbols=("ETH/USDT:USDT",),
        timeframe="15m",
        data_dir=DATA_DIR,
    )


def test_backtest_runs_on_real_data(settings):
    bt = GridBacktester(settings, config="C")
    res = bt.run("ETH/USDT:USDT", START, END, initial_usdt=1000.0)
    s = res.summary()
    assert s["trade_count"] >= 0
    assert s["final_equity_usdt"] > 0
    assert 0 <= s["max_drawdown_pct"] <= 100
    assert s["runtime_s"] < 120  # 40 days of 15m must be fast
    assert s["regime_transitions"] >= 0


def test_backtest_deterministic(settings):
    bt = GridBacktester(settings, config="C")
    r1 = bt.run("ETH/USDT:USDT", START, END)
    bt2 = GridBacktester(settings, config="C")
    r2 = bt2.run("ETH/USDT:USDT", START, END)
    assert r1.final_equity_usdt == pytest.approx(r2.final_equity_usdt, abs=1e-6)
    assert r1.trade_count == r2.trade_count


def test_backtest_no_lookahead_equity_monotonic_timestamps(settings):
    bt = GridBacktester(settings, config="C")
    res = bt.run("ETH/USDT:USDT", START, END)
    ts = [t for t, _ in res.equity_curve]
    assert ts == sorted(ts)


def test_backtest_insufficient_data_raises(settings):
    bt = GridBacktester(settings, config="C")
    with pytest.raises(ValueError):
        bt.run(
            "ETH/USDT:USDT", datetime(2026, 10, 10, tzinfo=UTC), datetime(2026, 10, 11, tzinfo=UTC)
        )


def test_backtest_config_d_degrades_without_key(settings):
    """Config D without an API key must still run (degrades to C behavior)."""
    bt = GridBacktester(settings, config="D")
    res = bt.run("ETH/USDT:USDT", START, END)
    assert res.summary()["trade_count"] >= 0
