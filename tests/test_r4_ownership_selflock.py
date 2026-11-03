"""R4 tests — T-204c: AIGrid must never lease-lock its OWN trades.

AIGridStrategy does not set trade.enter_tag, so trades it opened reach
sync_external_positions() with enter_tag=None. Before the fix, only an
enter_tag containing a known plan_id marked a trade as ours — the
strategy's own fills were classified "external", took a TREND_OWNED
lease owned by "external:AIGridStrategy" on a FREE pair, and AIGrid
vetoed its own grid entries until the trade closed (self-lock).

Own trades are now detected by strategy name (OWN_STRATEGY_NAME) or by
an enter_tag containing a known plan_id. Veto behavior for foreign
strategies is unchanged.
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
        plan_id="plan-r4test000001",
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


def _trade(
    pair: str = PAIR, strategy: str = "AIGridStrategy", enter_tag=None, is_open: bool = True
) -> dict:
    return {"pair": pair, "strategy": strategy, "is_open": is_open, "enter_tag": enter_tag}


# ------------------------------------------------------------------ T-204c
def test_own_strategy_trade_without_enter_tag_is_not_external(settings, account):
    """THE bug: our own trade (enter_tag=None, exactly what AIGridStrategy
    produces) on a FREE pair must not take an external lease — the pair
    stays FREE and can_open(grid) stays True."""
    orch = AIGridOrchestrator(settings)
    assert orch.ownership.state_of(PAIR) == OwnerState.FREE
    orch.sync_external_positions([_trade()])
    ok, reason = orch.ownership.can_open(PAIR, "grid")
    assert ok, f"self-lock: can_open vetoed by {reason}"
    assert orch.ownership.state_of(PAIR) == OwnerState.FREE
    d = orch.risk.evaluate_entry(_plan(), account)
    assert d.allowed
    orch.close()


def test_own_strategy_trade_with_non_plan_tag_still_ours(settings):
    """Strategy name alone is sufficient: an enter_tag matching no plan_id
    (e.g. a DCA tag like 'grid_level_2') still belongs to us when
    strategy == AIGridStrategy."""
    orch = AIGridOrchestrator(settings)
    orch.sync_external_positions([_trade(enter_tag="grid_level_2")])
    assert orch.ownership.state_of(PAIR) == OwnerState.FREE
    ok, _ = orch.ownership.can_open(PAIR, "grid")
    assert ok
    orch.close()


def test_foreign_strategy_trade_still_vetoes_grid_entry(settings, account):
    """No behavior change for other bots: ZaratustraV19 on SOL still takes
    external:ZaratustraV19 and vetoes grid entries."""
    orch = AIGridOrchestrator(settings)
    orch.sync_external_positions([_trade(strategy="ZaratustraV19")])
    assert orch.ownership.state_of(PAIR) == OwnerState.TREND_OWNED
    assert orch.ownership.owner_of(PAIR) == "external:ZaratustraV19"
    ok, reason = orch.ownership.can_open(PAIR, "grid")
    assert not ok
    assert reason == str(RiskEvent.OWNERSHIP_CONFLICT)
    d = orch.risk.evaluate_entry(_plan(), account)
    assert not d.allowed
    assert str(RiskEvent.OWNERSHIP_CONFLICT) in d.events
    orch.close()


def test_own_trade_with_plan_id_enter_tag_not_external(settings):
    """Regression: an enter_tag containing a known plan_id is never
    external, and the grid lease stays untouched."""
    orch = AIGridOrchestrator(settings)
    plan = _plan()
    orch.adopt_plan(plan)
    orch.sync_external_positions(
        [
            _trade(enter_tag=f"aigrid_{plan.plan_id}"),
        ]
    )
    assert orch.ownership.state_of(PAIR) == OwnerState.GRID_OWNED
    assert orch.ownership.owner_of(PAIR) == f"grid:{plan.plan_id}"
    orch.close()
