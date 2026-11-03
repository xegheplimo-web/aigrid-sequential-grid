"""R2 tests — T-203 recovery reloads lots/filled_levels; T-204 ownership enforced.

T-203: RestartReconciler.recover() must return restored_lots + filled_levels and
the orchestrator must LOAD them into the rebuilt GridEngine — otherwise the
virtual grid thinks no orders exist and re-buys levels Freqtrade already holds.
Before any new entry on a pair, virtual lots are reconciled once against the
real position size; a mismatch -> SAFE_HOLD for that pair (no auto-fix).

T-204: The orchestrator owns ONE OwnershipRegistry persisted to state;
RiskEngine.evaluate_entry() must call can_open(); AIGrid open/close acquires /
releases the symbol lease; RISK_LOCKED blocks every entry.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from aigrid.orchestrator import AIGridOrchestrator
from aigrid.plans.schema import AIGridPlan, GridMode
from aigrid.risk.engine import AccountState
from aigrid.risk.events import RiskEvent
from aigrid.risk.ownership import OwnerState
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


@pytest.fixture
def account():
    return AccountState(
        equity_usdt=1000.0,
        starting_equity_usdt=1000.0,
        peak_equity_usdt=1000.0,
        realized_pnl_today_usdt=0.0,
        total_exposure_usdt=0.0,
        free_margin_usdt=800.0,
    )


def _plan(**kw) -> AIGridPlan:
    base = dict(
        plan_id="plan-r2test000001",
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


def _candle(open_ms, o, h, low, c):
    return {"open_ms": open_ms, "open": o, "high": h, "low": low, "close": c, "volume": 1.0}


# ------------------------------------------------------------------ T-203 recovery
def test_recovery_restores_lots_and_filled_levels(settings):
    """persist 3 lots + 3 filled_levels -> restart -> engine has exactly those
    lots and the next candle does NOT re-add the filled levels."""
    orch1 = AIGridOrchestrator(settings)
    plan = _plan()
    orch1.adopt_plan(plan)
    # crosses down through levels 100/98/96 -> fills lots at levels 0,1,2
    orch1.process_candle(PAIR, _candle(1_000_000, 101.0, 102.0, 95.5, 98.5))
    eng1 = orch1._engines[PAIR]
    assert set(eng1.state.lots) == {0, 1, 2}
    assert eng1.state.filled_levels == {0, 1, 2}
    orch1.close()

    # ---- simulated restart: fresh orchestrator on the same state dir
    orch2 = AIGridOrchestrator(settings)
    res = orch2.recover(
        actual_open_pairs={PAIR_SHORT},
        actual_open_order_ids=set(),
        actual_open_lots={PAIR_SHORT: 3},
    )
    assert res["status"] == "RESTORED"
    eng2 = orch2._engines[PAIR]
    assert set(eng2.state.lots) == {0, 1, 2}  # lots reloaded, not empty
    assert eng2.state.filled_levels == {0, 1, 2}

    # next candle dips back through filled level 98 -> must not duplicate the order
    orch2.process_candle(PAIR, _candle(1_300_000, 98.5, 99.5, 97.9, 98.3))
    assert set(eng2.state.lots) == {0, 1, 2}
    assert eng2.state.filled_levels == {0, 1, 2}
    orch2.close()


def test_recovery_lot_count_mismatch_goes_safe_hold(settings):
    """state says 2 lots but the real position is 1 lot -> SAFE_HOLD, no entry."""
    orch1 = AIGridOrchestrator(settings)
    orch1.adopt_plan(_plan())
    # crosses down through levels 100/98 only -> fills 2 lots
    orch1.process_candle(PAIR, _candle(1_000_000, 101.0, 102.0, 97.5, 98.5))
    assert len(orch1._engines[PAIR].state.lots) == 2
    orch1.close()

    orch2 = AIGridOrchestrator(settings)
    res = orch2.recover(
        actual_open_pairs={PAIR_SHORT},
        actual_open_order_ids=set(),
        actual_open_lots={PAIR_SHORT: 1},
    )  # FT really holds 1 lot
    assert res["status"] == "SAFE_HOLD"
    assert str(RiskEvent.POSITION_MISMATCH) in res["reasons"]
    assert PAIR not in orch2._engines  # untrusted state not used
    assert orch2._controller_for(PAIR).paused  # SAFE_HOLD for that pair

    # a candle crossing grid levels must not produce any engine / entry
    snap = orch2.process_candle(PAIR, _candle(1_300_000, 101.0, 102.0, 95.0, 98.0))
    assert snap.engine == {}
    orch2.close()


def test_no_state_fresh_start_and_normal_entries(settings):
    """No persisted state -> old behavior (FRESH_START, entries work normally)."""
    orch = AIGridOrchestrator(settings)
    res = orch.recover(actual_open_pairs=set(), actual_open_order_ids=set())
    assert res["status"] == "FRESH_START"
    orch.adopt_plan(_plan())
    orch.process_candle(PAIR, _candle(1_000_000, 101.0, 102.0, 95.5, 98.5))
    assert len(orch._engines[PAIR].state.lots) == 3
    orch.close()


# ------------------------------------------------------------------ T-204 ownership
def test_orchestrator_owns_single_shared_registry(settings):
    orch = AIGridOrchestrator(settings)
    assert orch.risk.ownership is orch.ownership
    orch.close()


def test_grid_entry_vetoed_when_trend_owns_symbol(settings, account):
    """Trend bot holds the pair -> grid entry vetoed OWNERSHIP_CONFLICT (no raise)."""
    orch = AIGridOrchestrator(settings)
    ok, _ = orch.ownership.acquire(PAIR, "trend", "trend:ZaratustraV19")
    assert ok
    d = orch.risk.evaluate_entry(_plan(), account)
    assert not d.allowed
    assert str(RiskEvent.OWNERSHIP_CONFLICT) in d.events
    orch.close()


def test_ownership_registry_persists_across_restart(settings, account):
    """Registry persisted to state -> restart -> the lock is still effective."""
    orch1 = AIGridOrchestrator(settings)
    orch1.ownership.acquire(PAIR, "trend", "trend:ZaratustraV19")
    orch1.persist()
    orch1.close()

    orch2 = AIGridOrchestrator(settings)
    assert orch2.ownership.state_of(PAIR) == OwnerState.TREND_OWNED
    assert orch2.ownership.owner_of(PAIR) == "trend:ZaratustraV19"
    d = orch2.risk.evaluate_entry(_plan(), account)
    assert not d.allowed
    assert str(RiskEvent.OWNERSHIP_CONFLICT) in d.events
    orch2.close()


def test_risk_locked_blocks_entries_even_with_empty_ownership(settings, account):
    """RISK_LOCKED (halt) blocks every entry, incl. same-strategy, even when no
    other strategy owns anything."""
    orch = AIGridOrchestrator(settings)
    orch.risk.lock_symbol(PAIR, "MAX_LOSS_REACHED")
    d = orch.risk.evaluate_entry(_plan(), account)
    assert not d.allowed
    assert str(RiskEvent.OWNER_STOP) in d.events
    orch.close()


def test_release_symbol_allows_entry_again(settings, account):
    orch = AIGridOrchestrator(settings)
    orch.ownership.acquire(PAIR, "trend", "trend:ZaratustraV19")
    d1 = orch.risk.evaluate_entry(_plan(), account)
    assert not d1.allowed
    assert str(RiskEvent.OWNERSHIP_CONFLICT) in d1.events
    ok, _ = orch.ownership.release(PAIR, "trend:ZaratustraV19")
    assert ok
    d2 = orch.risk.evaluate_entry(_plan(), account)
    assert d2.allowed
    orch.close()


def test_aigrid_open_acquires_grid_ownership(settings):
    """When AIGrid adopts a plan that will open positions it locks the symbol."""
    orch = AIGridOrchestrator(settings)
    plan = _plan()
    orch.adopt_plan(plan)
    assert orch.ownership.state_of(PAIR) == OwnerState.GRID_OWNED
    assert orch.ownership.owner_of(PAIR) == f"grid:{plan.plan_id}"
    orch.close()


def test_aigrid_flat_and_finished_releases_ownership(settings):
    """Engine flat + halted (position closed/done) -> symbol lease released."""
    orch = AIGridOrchestrator(settings)
    orch.adopt_plan(_plan())
    # candle opens below the grid and closes under grid_lower -> boundary halt,
    # nothing filled -> engine finished with no open lots
    orch.process_candle(PAIR, _candle(1_000_000, 93.5, 94.0, 92.0, 93.0))
    eng = orch._engines[PAIR]
    assert eng.state.halted and not eng.state.lots
    assert orch.ownership.state_of(PAIR) == OwnerState.FREE
    orch.close()
