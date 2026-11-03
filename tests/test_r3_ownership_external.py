"""R3 tests — T-204b: ownership must see EXTERNAL bots' open positions.

OwnershipRegistry is in-process state; trend bots running other Freqtrade
strategies never write to it, so "one pair, one strategy" previously held
only between AIGrid instances. sync_external_positions() reconciles the
registry with the exchange's real open trades (passed in by the caller —
no I/O inside): a trade that does not belong to AIGrid takes a
TREND_OWNED lease owned by "external:<strategy>" so grid entries on that
pair are vetoed; the external lease is released once the trade is gone.
grid:* leases are never taken or released by the sync.
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
PAIR = "SOL/USDT:USDT"
OTHER = "DOGE/USDT:USDT"


@pytest.fixture
def settings(tmp_path):
    return AIGridSettings(
        enabled=False,
        mode="shadow",
        symbols=(PAIR,),
        timeframe="5m",
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


def _plan(symbol: str = PAIR, **kw) -> AIGridPlan:
    base = dict(
        plan_id="plan-r3test000001",
        plan_version=1,
        symbol=symbol,
        created_at=NOW,
        expires_at=NOW + timedelta(minutes=45),
        market_regime="RANGE",
        confidence=0.7,
        grid_mode=GridMode.RANGE_GRID,
        grid_lower=94.0,
        grid_upper=100.0,
        grid_levels=4,
        grid_spacing=0.05,
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


def _ext(pair: str = PAIR, strategy: str = "ZaratustraV19", **kw) -> dict:
    trade = {"pair": pair, "strategy": strategy, "is_open": True, "enter_tag": ""}
    trade.update(kw)
    return trade


# ------------------------------------------------------------------ T-204b
def test_external_trade_on_pair_vetoes_grid_entry(settings, account):
    """Foreign bot holds SOL -> grid entry on SOL vetoed OWNERSHIP_CONFLICT."""
    orch = AIGridOrchestrator(settings)
    orch.sync_external_positions([_ext()])
    assert orch.ownership.state_of(PAIR) == OwnerState.TREND_OWNED
    assert orch.ownership.owner_of(PAIR) == "external:ZaratustraV19"
    d = orch.risk.evaluate_entry(_plan(), account)
    assert not d.allowed
    assert str(RiskEvent.OWNERSHIP_CONFLICT) in d.events
    orch.close()


def test_external_trade_closed_releases_lease_and_entry_allowed(settings, account):
    """The external trade disappears from the open list -> lease released."""
    orch = AIGridOrchestrator(settings)
    orch.sync_external_positions([_ext()])
    d1 = orch.risk.evaluate_entry(_plan(), account)
    assert not d1.allowed
    assert str(RiskEvent.OWNERSHIP_CONFLICT) in d1.events

    orch.sync_external_positions([])  # position closed on the exchange
    assert orch.ownership.state_of(PAIR) == OwnerState.FREE
    d2 = orch.risk.evaluate_entry(_plan(), account)
    assert d2.allowed
    orch.close()


def test_sync_never_touches_grid_lease(settings):
    """AIGrid's own grid lease survives every sync — never released/stolen."""
    orch = AIGridOrchestrator(settings)
    plan = _plan()
    orch.adopt_plan(plan)
    grid_owner = f"grid:{plan.plan_id}"
    assert orch.ownership.state_of(PAIR) == OwnerState.GRID_OWNED

    # empty sync keeps the grid lease
    orch.sync_external_positions([])
    assert orch.ownership.owner_of(PAIR) == grid_owner

    # a foreign trade on the SAME pair must not steal the grid lease either
    orch.sync_external_positions([_ext()])
    assert orch.ownership.state_of(PAIR) == OwnerState.GRID_OWNED
    assert orch.ownership.owner_of(PAIR) == grid_owner

    # external lease on a DIFFERENT pair still works and releases on its own
    orch.sync_external_positions([_ext(), _ext(pair=OTHER)])
    assert orch.ownership.state_of(OTHER) == OwnerState.TREND_OWNED
    orch.sync_external_positions([])
    assert orch.ownership.state_of(OTHER) == OwnerState.FREE
    assert orch.ownership.state_of(PAIR) == OwnerState.GRID_OWNED  # untouched
    orch.close()


def test_own_aigrid_trade_is_not_external(settings):
    """A trade whose enter_tag references our plan_id is ours — never leased
    to external:*."""
    orch = AIGridOrchestrator(settings)
    plan = _plan()
    orch.adopt_plan(plan)
    orch.sync_external_positions(
        [
            _ext(strategy="AIGridFT", enter_tag=f"grid entry {plan.plan_id}"),
        ]
    )
    assert orch.ownership.state_of(PAIR) == OwnerState.GRID_OWNED
    assert orch.ownership.owner_of(PAIR) == f"grid:{plan.plan_id}"
    orch.close()


def test_closed_trade_flag_is_ignored(settings):
    """Trades reported with is_open=False create no lease."""
    orch = AIGridOrchestrator(settings)
    orch.sync_external_positions([_ext(is_open=False)])
    assert orch.ownership.state_of(PAIR) == OwnerState.FREE
    orch.close()
