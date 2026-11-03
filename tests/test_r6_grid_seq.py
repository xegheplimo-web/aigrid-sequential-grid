"""R6 tests — T-213a: client_order_id carries a per-level cycle counter.

Audit S1-a: a level re-armed after TP re-emitted the identical
client_order_id, so the 2nd grid cycle was blocked as a duplicate. Every
entry arm now bumps a per-level seq; the first cycle keeps the legacy
4-part id (equivalent to seq=1), later cycles append :{seq}. seq is
persisted in to_dict() and reloaded on restart so ids stay unique.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from aigrid.grid.engine import ActionKind, VirtualGridEngine
from aigrid.orchestrator import AIGridOrchestrator
from aigrid.plans.schema import AIGridPlan, GridMode
from aigrid.settings import AIGridSettings

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "user_data/data/binance/futures"
NOW = datetime.now(UTC)
PAIR = "ETH/USDT:USDT"
PAIR_SHORT = "ETHUSDT"


@pytest.fixture
def settings(tmp_path):
    return AIGridSettings(
        enabled=False,
        mode="shadow",
        symbols=(PAIR,),
        timeframe="15m",
        data_dir=DATA_DIR,
        state_dir=tmp_path / "state",
    )


def _plan(**kw) -> AIGridPlan:
    base = dict(
        plan_id="plan-r6seq00001",
        plan_version=1,
        symbol=PAIR,
        created_at=NOW,
        expires_at=NOW + timedelta(minutes=45),
        market_regime="RANGE",
        confidence=0.7,
        grid_mode=GridMode.RANGE_GRID,
        grid_lower=94.0,
        grid_upper=100.0,
        grid_levels=4,
        grid_spacing=0.05,
        # consistent: notional (200) = margin (100) x leverage (2)
        max_capital=100.0,
        max_position_size=200.0,
        max_leverage=2.0,
        level_stake=25.0,
        level_notional=50.0,
        stop_loss=90.0,
        take_profit=None,
        invalidation_conditions=[],
        risk_level="MEDIUM",
        reasoning_summary="t",
    )
    base.update(kw)
    return AIGridPlan(**base)


def _engine(plan=None, **kw) -> VirtualGridEngine:
    plan = plan or _plan()
    defaults = dict(
        fee_rate=0.0002, slippage_rate=0.0005, leg_stop_loss_pct=0.03, max_inventory_notional=200.0
    )
    defaults.update(kw)
    return VirtualGridEngine(plan, **defaults)


def _candle(open_ms, o, h, low, c):
    return {"open_ms": open_ms, "open": o, "high": h, "low": low, "close": c, "volume": 1.0}


def _fill_level2(e: VirtualGridEngine, candle_ms: int):
    """One candle crossing down through level 2 (96.0) only."""
    return e.on_candle(candle_ms, 97.5, 98.0, 95.5, 97.0)


def _tp_level2(e: VirtualGridEngine, candle_ms: int):
    """One candle rallying past level 2's TP (entry*1.05 ~ 100.85).
    Closes at 97.5 so the next candle can re-cross 96.0 only."""
    return e.on_candle(candle_ms, 98.0, 101.2, 97.0, 97.5)


def _recross_level2(e: VirtualGridEngine, candle_ms: int):
    """One candle crossing back down through 96.0 only (prev close 97.5)."""
    return e.on_candle(candle_ms, 97.5, 98.0, 95.9, 97.0)


def _buys(actions, level: int | None = None):
    buys = [a for a in actions if a.kind == ActionKind.BUY]
    return [b for b in buys if b.level == level] if level is not None else buys


# ------------------------------------------------------------------ cycle ids
def test_rearm_after_tp_emits_new_client_order_id():
    """The 2nd grid cycle at the same level must not reuse the 1st cycle's id."""
    e = _engine()
    a1 = _fill_level2(e, 0)
    id1 = _buys(a1, level=2)[0].client_order_id
    _tp_level2(e, 1)
    assert 2 not in e.state.lots
    a3 = _recross_level2(e, 2)
    id2 = _buys(a3, level=2)[0].client_order_id
    assert id1 != id2  # identical ids before R6 (S1-a)
    assert e.state.level_seq[2] == 2


def test_first_cycle_id_is_legacy_4part_and_stable_for_retries():
    """seq=1 renders as the legacy 4-part id; retrying the same order must
    reproduce the identical id (idempotent resubmission)."""
    e = _engine()
    a1 = _fill_level2(e, 0)
    emitted = _buys(a1, level=2)[0].client_order_id
    assert emitted == "plan-r6seq00001:grid:ENTRY:2"
    assert e.order_id(2, "ENTRY") == emitted  # retry -> same id


def test_second_cycle_id_appends_seq():
    e = _engine()
    _fill_level2(e, 0)
    _tp_level2(e, 1)
    a3 = _recross_level2(e, 2)
    assert _buys(a3, level=2)[0].client_order_id == "plan-r6seq00001:grid:ENTRY:2:2"


def test_exit_id_tracks_the_entry_cycle():
    """The exit of a re-armed lot must not reuse the previous cycle's exit id."""
    e = _engine()
    _fill_level2(e, 0)
    s1 = _tp_level2(e, 1)
    exit_id1 = [a for a in s1 if a.kind == ActionKind.SELL and a.level == 2][0].client_order_id
    _recross_level2(e, 2)
    s2 = _tp_level2(e, 3)
    exit_id2 = [a for a in s2 if a.kind == ActionKind.SELL and a.level == 2][0].client_order_id
    assert exit_id1 != exit_id2
    assert exit_id2 == "plan-r6seq00001:grid:EXIT:2:2"


# --------------------------------------------------------- idempotent feedback
def test_apply_external_fill_replay_creates_no_second_lot():
    e = _engine()
    _fill_level2(e, 0)
    qty = e.state.lots[2].qty
    assert e.apply_external_fill("plan-r6seq00001:grid:ENTRY:2", 96.0, qty) is True
    assert set(e.state.lots) == {2}  # no duplicate lot
    _tp_level2(e, 1)
    _recross_level2(e, 2)
    assert e.apply_external_fill("plan-r6seq00001:grid:ENTRY:2:2", 96.0, qty) is True
    assert set(e.state.lots) == {2}


def test_apply_external_fill_parses_legacy_4part():
    e = _engine()
    assert e.apply_external_fill("plan-legacy:grid:ENTRY:7", 96.0, 0.4) is True
    assert 7 in e.state.lots


def test_apply_external_fill_parses_5part_and_bumps_level_seq():
    """A 5-part external ENTRY fill records the lot and advances the level's
    seq so a later engine arm of that level never reuses the external id."""
    e = _engine()
    assert e.apply_external_fill("plan-x:grid:ENTRY:7:2", 96.0, 0.4) is True
    assert 7 in e.state.lots
    assert e.state.level_seq[7] == 2


# ------------------------------------------------------------------ persistence
def test_level_seq_survives_restart_rearm_never_reuses_id(settings):
    """level_seq is persisted and reloaded: a level that already cycled twice
    before restart must arm cycle 3 after restart, never reusing an old id."""
    orch1 = AIGridOrchestrator(settings)
    plan = _plan()
    orch1.adopt_plan(plan)
    orch1.process_candle(PAIR, _candle(1_000_000, 97.5, 98.0, 95.5, 97.0))
    eng1 = orch1._engines[PAIR]
    assert set(eng1.state.lots) == {2}
    orch1.process_candle(PAIR, _candle(1_100_000, 98.0, 101.2, 97.0, 97.5))
    assert 2 not in eng1.state.lots
    orch1.process_candle(PAIR, _candle(1_200_000, 97.5, 98.0, 95.9, 97.0))
    assert set(eng1.state.lots) == {2}
    assert eng1.state.level_seq[2] == 2
    assert eng1.to_dict(97.0)["level_seq"]["2"] == 2
    ids_before = {f"{plan.plan_id}:grid:ENTRY:2", f"{plan.plan_id}:grid:ENTRY:2:2"}
    orch1.close()

    orch2 = AIGridOrchestrator(settings)
    res = orch2.recover(
        actual_open_pairs={PAIR_SHORT},
        actual_open_order_ids=set(),
        actual_open_lots={PAIR_SHORT: 1},
    )
    assert res["status"] == "RESTORED"
    eng2 = orch2._engines[PAIR]
    assert eng2.state.level_seq[2] == 2  # restored, not reset to 0

    emitted = []
    orig = eng2.on_candle

    def spy(*a, **k):
        acts = orig(*a, **k)
        emitted.extend(acts)
        return acts

    eng2.on_candle = spy

    orch2.process_candle(PAIR, _candle(1_300_000, 98.0, 101.2, 97.0, 97.5))
    orch2.process_candle(PAIR, _candle(1_400_000, 97.5, 98.0, 95.9, 97.0))
    buys = [a for a in emitted if a.kind == ActionKind.BUY]
    assert buys and all(b.client_order_id not in ids_before for b in buys)
    assert buys[-1].client_order_id == f"{plan.plan_id}:grid:ENTRY:2:3"
    orch2.close()
