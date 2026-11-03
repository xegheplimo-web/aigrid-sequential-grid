"""PHASE 5b/6 integration tests: Orchestrator end-to-end (AI disabled paths).

These are real-component integration tests (no core mocks): data adapter,
regime engine, controller, grid engine, risk engine, state store all live.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from aigrid.orchestrator import AIGridOrchestrator
from aigrid.plans.schema import draft_plan_from_assessment
from aigrid.settings import AIGridSettings

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "user_data/data/binance/futures"
AS_OF = datetime(2026, 10, 11, tzinfo=UTC)


@pytest.fixture
def settings(tmp_path):
    return AIGridSettings(
        enabled=False,
        mode="shadow",
        symbols=("ETH/USDT:USDT",),
        timeframe="15m",
        data_dir=DATA_DIR,
        state_dir=tmp_path / "state",
    )


@pytest.fixture
def orch(settings):
    o = AIGridOrchestrator(settings)
    yield o
    o.close()


def _candles(adapter, pair="ETH/USDT:USDT", tf="15m", n=120, as_of=AS_OF):
    df = adapter.get_ohlcv(pair, tf, as_of=as_of, lookback=n)
    rows = []
    for idx, row in df.iterrows():
        rows.append(
            {
                "open_ms": int(idx.value // 1_000_000),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        )
    return rows


def test_orchestrator_ai_disabled_stays_safe_hold(orch):
    adapter = orch.adapter
    candles = _candles(adapter)
    for i, c in enumerate(candles):
        hist = adapter.get_ohlcv(
            "ETH/USDT:USDT",
            "15m",
            as_of=datetime.fromtimestamp((c["open_ms"] + 900_000) / 1000, tz=UTC),
            lookback=300,
        )
        snap = orch.process_candle("ETH/USDT:USDT", c, history_df=hist)
    assert snap.grid_mode == "SAFE_HOLD"  # AI off -> never trades
    assert snap.ai_status == "OFF"
    assert snap.engine == {}  # no engine adopted


def test_orchestrator_adopted_plan_executes(orch):
    adapter = orch.adapter
    candles = _candles(adapter)
    # adopt a manual (validated) plan — simulates an approved AI cycle
    last_close = candles[-1]["close"]
    plan = draft_plan_from_assessment(
        "ETH/USDT:USDT",
        "RANGE",
        0.8,
        last_close,
        20.0,
        grid_mode="RANGE_GRID",
        ttl_minutes=600,
        margin_usdt=240.0,
        leverage=3.0,
        levels=6,
    )
    orch.adopt_plan(plan)
    assert orch._engine is not None
    # feed candles after adoption
    snaps = []
    for c in candles[-40:]:
        hist = adapter.get_ohlcv(
            "ETH/USDT:USDT",
            "15m",
            as_of=datetime.fromtimestamp((c["open_ms"] + 900_000) / 1000, tz=UTC),
            lookback=300,
        )
        snaps.append(orch.process_candle("ETH/USDT:USDT", c, history_df=hist))
    last = snaps[-1]
    assert last.plan_id == plan.plan_id
    assert last.engine["plan_id"] == plan.plan_id
    # If the regime controller confirmed a regime away from the plan's, it pauses
    # new entries (fail-closed) while the engine keeps managing exits — never
    # force-closing positions just because the regime moved.
    if last.paused:
        assert last.grid_mode == "SAFE_HOLD"
        assert last.pause_reason.startswith(("REGIME_", "PLAN_"))
    else:
        assert last.grid_mode == "RANGE_GRID"


def test_orchestrator_drain_then_switch(orch):
    """New plan while lots open: old engine drains exits; new entries wait."""
    adapter = orch.adapter
    candles = _candles(adapter)
    plan1 = draft_plan_from_assessment(
        "ETH/USDT:USDT",
        "RANGE",
        0.8,
        candles[-1]["close"],
        20.0,
        grid_mode="RANGE_GRID",
        ttl_minutes=600,
        margin_usdt=240.0,
        leverage=3.0,
        levels=6,
    )
    orch.adopt_plan(plan1)
    # feed one candle to (maybe) fill a level
    c = candles[-1]
    hist = adapter.get_ohlcv(
        "ETH/USDT:USDT",
        "15m",
        as_of=datetime.fromtimestamp((c["open_ms"] + 900_000) / 1000, tz=UTC),
        lookback=300,
    )
    orch.process_candle("ETH/USDT:USDT", c, history_df=hist)
    # queue a second plan while lots may be open
    plan2 = draft_plan_from_assessment(
        "ETH/USDT:USDT",
        "RANGE",
        0.8,
        candles[-1]["close"],
        20.0,
        grid_mode="RANGE_GRID",
        ttl_minutes=600,
        margin_usdt=240.0,
        leverage=3.0,
        levels=6,
    )
    orch.adopt_plan(plan2)
    if orch._engine.state.lots:
        assert orch._pending_plan is not None
        assert orch._pending_plan.plan_id == plan2.plan_id
        assert orch._engine.plan.plan_id == plan1.plan_id  # old engine still owns
    else:
        assert orch._engine.plan.plan_id == plan2.plan_id  # swapped immediately


def test_orchestrator_recovery_safe_hold_on_mismatch(settings, tmp_path):
    orch = AIGridOrchestrator(settings)
    result = orch.recover(actual_open_pairs={"SOLUSDT"}, actual_open_order_ids=set())
    assert result["status"] == "SAFE_HOLD"
    assert "POSITION_MISMATCH" in result["reasons"]
    orch.close()


def test_orchestrator_recovery_fresh_start(settings, tmp_path):
    orch = AIGridOrchestrator(settings)
    result = orch.recover(actual_open_pairs=set(), actual_open_order_ids=set())
    assert result["status"] == "FRESH_START"
    orch.close()


def test_orchestrator_persists_and_recovers_plan(settings, tmp_path):
    orch = AIGridOrchestrator(settings)
    candles = _candles(orch.adapter)
    plan = draft_plan_from_assessment(
        "ETH/USDT:USDT",
        "RANGE",
        0.8,
        candles[-1]["close"],
        20.0,
        grid_mode="RANGE_GRID",
        ttl_minutes=600,
        margin_usdt=240.0,
        leverage=3.0,
        levels=6,
    )
    orch.adopt_plan(plan)
    orch.persist()
    orch.close()

    orch2 = AIGridOrchestrator(settings)
    result = orch2.recover(actual_open_pairs={"ETHUSDT"}, actual_open_order_ids=set())
    assert result["status"] == "RESTORED"
    assert result["plan"].plan_id == plan.plan_id
    orch2.close()


def test_orchestrator_snapshot_separates_ai_vs_execution(orch):
    """Dashboard contract: AI recommendation and actual execution are separate."""
    adapter = orch.adapter
    candles = _candles(adapter, n=100)
    c = candles[-1]
    snap = orch.process_candle("ETH/USDT:USDT", c)
    d = snap.to_dict()
    assert "ai_status" in d and "grid_mode" in d
    assert "engine" in d and "ai_reasoning" in d
    assert d["ai_status"] == "OFF"  # AI recommendation side
    assert d["grid_mode"] == "SAFE_HOLD"  # actual execution side
