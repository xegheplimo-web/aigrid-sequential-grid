"""Capital Manager + 3-Agent TA adapter + config-D backtest + capital risk checks.

Covers the user's capital architecture:
  AI proposes -> CapitalManager budgets -> Grid Engine splits per level ->
  Risk Engine vetoes. notional = margin x leverage, always.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from aigrid.capital import CapitalManager, CapitalSettings
from aigrid.exchange_info import ExchangeInfo
from aigrid.plans.schema import AIGridPlan, GridMode
from aigrid.plans.validator import PlanValidator
from aigrid.risk.engine import AccountState, RiskEngine
from aigrid.risk.events import RiskEvent
from aigrid.settings import AIGridSettings, RiskLimits
from aigrid.ta_adapter import (
    MarketAssessment,
    RiskCritique,
    StrategyProposal,
    TechnicalReport,
    TradingAgentsAdapter,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "user_data/data/binance/futures"
EI_PATH = ROOT / ".orchestrator/fapi_exchangeinfo.json"

NOW = datetime.now(UTC)


# ------------------------------------------------------------------ CapitalManager
@pytest.fixture
def cm():
    return CapitalManager(CapitalSettings(), max_leverage=3.0)


def test_capital_spec_example(cm):
    """User's example: 1000 USDT wallet -> 300 budget; ETH <=100 margin;
    5 levels -> 20 margin/level; 2x -> 40 notional/level, 200 max notional."""
    a = cm.allocate("ETH/USDT:USDT", equity_usdt=1000.0, levels=5)
    assert a is not None
    assert cm.total_budget(1000.0) == 300.0
    assert a.margin_usdt == 100.0
    assert a.levels == 5
    assert a.level_margin_usdt == pytest.approx(20.0)
    assert a.leverage == 2.0
    assert a.level_notional_usdt == pytest.approx(40.0)
    assert a.notional_usdt == pytest.approx(200.0)
    assert a.budget_remaining_usdt == pytest.approx(200.0)


def test_capital_global_budget_shared(cm):
    """Second symbol gets what remains of the 30% budget — not a fresh 100."""
    a1 = cm.allocate("ETH/USDT:USDT", 1000.0, levels=5)
    used = {a1.symbol: a1.margin_usdt}
    a2 = cm.allocate("BTC/USDT:USDT", 1000.0, allocated_margins=used, levels=5)
    assert a1.margin_usdt + a2.margin_usdt <= cm.total_budget(1000.0) + 1e-9
    assert a2.margin_usdt == pytest.approx(100.0)  # 300 - 100 used = 200 -> cap 100
    a3 = cm.allocate(
        "SOL/USDT:USDT",
        1000.0,
        allocated_margins={a1.symbol: a1.margin_usdt, a2.symbol: a2.margin_usdt},
        levels=5,
    )
    assert a3.margin_usdt == pytest.approx(100.0)  # budget leftover = 100


def test_capital_no_trade_when_budget_exhausted(cm):
    a = cm.allocate("ETH/USDT:USDT", 1000.0, allocated_margins={"X/USDT:USDT": 295.0}, levels=5)
    assert a is None or a.margin_usdt <= 5.0 + 1e-9
    assert a is None  # 5 USDT left < min_level_margin*levels -> NO TRADE


def test_capital_reduces_levels_to_keep_min_notional(cm):
    """Margin 60 -> 5 levels x12 margin x2x = 24 notional OK; margin 35 ->
    must cut levels so each stays >= ~21 notional."""
    a = cm.allocate("ETH/USDT:USDT", 1000.0, allocated_margins={"X/USDT:USDT": 265.0}, levels=5)
    assert a is not None
    assert a.levels <= 5
    assert a.level_notional_usdt >= 20.0 * 1.05 - 1e-9
    assert a.levels * a.level_margin_usdt == pytest.approx(a.margin_usdt, abs=0.05)


def test_capital_leverage_clamped_by_risk(cm):
    a = cm.allocate("ETH/USDT:USDT", 1000.0, levels=5, leverage=9.0)
    assert a.leverage == 3.0  # risk max wins
    assert a.notional_usdt == pytest.approx(a.margin_usdt * 3.0)


def test_capital_levels_bounded():
    cm2 = CapitalManager(CapitalSettings(min_levels=3, max_levels=8))
    a = cm2.allocate("ETH/USDT:USDT", 1000.0, levels=50)
    assert a.levels <= 8
    a = cm2.allocate("ETH/USDT:USDT", 1000.0, levels=1)
    assert a.levels >= 3


def test_capital_settings_validate_bounds():
    with pytest.raises(ValueError):
        CapitalSettings(total_budget_pct=3.0).validate()
    with pytest.raises(ValueError):
        CapitalSettings(default_leverage=50.0).validate()


# ------------------------------------------------------------------ 3-agent adapter
class _FakeStructured:
    def __init__(self, llm, schema):
        self.llm, self.schema = llm, schema

    def invoke(self, messages):
        self.llm.calls.append((self.schema.__name__, messages[0]["content"][:30]))
        return self.llm.responses[self.schema]


class FakeLLM:
    """Mimics the LangChain chat model surface the adapter uses."""

    def __init__(self, responses):
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    def with_structured_output(self, schema):
        return _FakeStructured(self, schema)


def _responses(ok_for_grid=True):
    return {
        TechnicalReport: TechnicalReport(
            trend="SIDEWAYS",
            momentum="WEAK",
            volatility="LOW",
            support=2490.0,
            resistance=2515.0,
            summary="flat",
        ),
        RiskCritique: RiskCritique(
            main_risks=["bb squeeze"],
            trend_break_risk="LOW",
            volatility_assessment="CALM",
            funding_concern=False,
            conditions_ok_for_grid=ok_for_grid,
            objections=[],
            summary="ok",
        ),
        StrategyProposal: StrategyProposal(
            regime="RANGE",
            confidence=0.6,
            grid_mode="RANGE_GRID",
            grid_lower=2490.0,
            grid_upper=2515.0,
            levels=5,
            max_capital_suggestion=200.0,
            reasoning_summary="range grid",
        ),
    }


def _enabled_settings() -> AIGridSettings:
    return replace(AIGridSettings(), enabled=True, mode="shadow")


def test_three_agent_sequence_and_count():
    ta = TradingAgentsAdapter(_enabled_settings())
    fake = FakeLLM(_responses())
    ta._llm = fake  # inject — no network in tests
    a = ta.assess("ETH/USDT:USDT", {"last_close": 2502.0}, 0)
    assert a.ai_available and a.ok
    assert a.llm_calls == 3
    assert [c[0] for c in fake.calls] == ["TechnicalReport", "RiskCritique", "StrategyProposal"]
    assert a.regime == "RANGE" and a.proposal.grid_mode == "RANGE_GRID"
    assert a.technical.trend == "SIDEWAYS"
    assert a.critique.conditions_ok_for_grid is True


def test_agent2_veto_forces_safe_hold():
    ta = TradingAgentsAdapter(_enabled_settings())
    ta._llm = FakeLLM(_responses(ok_for_grid=False))
    a = ta.assess("ETH/USDT:USDT", {}, 0)
    # even if the synthesizer tried an active grid, the veto wins
    assert a.proposal.grid_mode == "SAFE_HOLD"


def test_agent_failure_fails_closed():
    class BoomLLM:
        def with_structured_output(self, schema):
            raise RuntimeError("network down")

    ta = TradingAgentsAdapter(_enabled_settings())
    ta._llm = BoomLLM()
    a = ta.assess("ETH/USDT:USDT", {}, 0)
    assert not a.ai_available and not a.ok
    assert a.regime == "AI_UNAVAILABLE"
    assert "AI_RUN_FAILED" in a.reasoning_summary


def test_adapter_disabled_by_master_switch():
    ta = TradingAgentsAdapter(AIGridSettings())  # enabled=False default
    a = ta.assess("ETH/USDT:USDT", {}, 0)
    assert not a.ok and a.regime == "AI_UNAVAILABLE"


def test_adapter_missing_key_fails_closed(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    s = replace(_enabled_settings(), llm=replace(AIGridSettings().llm, provider="deepseek"))
    ta = TradingAgentsAdapter(s)
    a = ta.assess("ETH/USDT:USDT", {}, 0)
    assert not a.ok
    assert "MISSING_API_KEY" in (ta.unavailable_reason or "")


# ------------------------------------------------------------------ validator + risk capital checks
def _plan(**kw) -> AIGridPlan:
    base = dict(
        plan_id="plan-cap00000001",
        plan_version=1,
        symbol="ETH/USDT:USDT",
        created_at=NOW,
        expires_at=NOW.replace(minute=NOW.minute + 45)
        if NOW.minute < 15
        else NOW.replace(hour=NOW.hour + 1),
        market_regime="RANGE",
        confidence=0.7,
        grid_mode=GridMode.RANGE_GRID,
        grid_lower=2950.0,
        grid_upper=3050.0,
        grid_levels=6,
        grid_spacing=0.01,
        max_capital=100.0,
        max_position_size=200.0,
        max_leverage=2.0,
        stop_loss=2900.0,
        take_profit=None,
        invalidation_conditions=[],
        risk_level="MEDIUM",
        reasoning_summary="t",
    )
    base.update(kw)
    return AIGridPlan(**base)


@pytest.fixture(scope="module")
def ei():
    return ExchangeInfo(EI_PATH)


def test_validator_rejects_margin_over_symbol_cap(ei):
    v = PlanValidator(
        ei, RiskLimits(), capital=CapitalManager(CapitalSettings(per_symbol_margin_cap=100.0))
    )
    p = _plan(max_capital=150.0, max_position_size=300.0)  # 150 > 100 cap
    verdict = v.validate(p, last_close=3000.0)
    assert not verdict.approved
    assert "MARGIN_EXCEEDS_PER_SYMBOL_CAP" in verdict.reasons


def test_validator_rejects_inconsistent_notional(ei):
    v = PlanValidator(ei, RiskLimits())
    p = _plan(max_capital=200.0, max_position_size=200.0)  # 200 < 200x2
    verdict = v.validate(p, last_close=3000.0)
    assert not verdict.approved
    assert "NOTIONAL_LESS_THAN_MARGIN_TIMES_LEVERAGE" in verdict.reasons


def test_risk_vetoes_over_total_budget():
    engine = RiskEngine(RiskLimits())
    cm = CapitalManager(CapitalSettings())
    acc = AccountState(
        equity_usdt=1000.0,
        starting_equity_usdt=1000.0,
        peak_equity_usdt=1000.0,
        realized_pnl_today_usdt=0.0,
        total_exposure_usdt=0.0,
        free_margin_usdt=800.0,
        allocated_margins={"BTC/USDT:USDT": 250.0},
    )  # 250 used of 300 budget
    p = _plan(max_capital=100.0)  # +100 -> 350 > 300
    d = engine.evaluate_entry(p, acc, capital_manager=cm)
    assert not d.allowed
    assert str(RiskEvent.EXPOSURE_LIMIT) in d.events


def test_risk_allows_within_budget():
    engine = RiskEngine(RiskLimits())
    cm = CapitalManager(CapitalSettings())
    acc = AccountState(
        equity_usdt=1000.0,
        starting_equity_usdt=1000.0,
        peak_equity_usdt=1000.0,
        realized_pnl_today_usdt=0.0,
        total_exposure_usdt=0.0,
        free_margin_usdt=800.0,
        allocated_margins={"BTC/USDT:USDT": 100.0},
    )
    d = engine.evaluate_entry(_plan(), acc, capital_manager=cm)
    assert d.allowed, d.events


# ------------------------------------------------------------------ config D backtest
START = datetime(2026, 9, 25, tzinfo=UTC)
END = datetime(2026, 10, 11, tzinfo=UTC)


@pytest.fixture(scope="module")
def bt_settings():
    return AIGridSettings(
        enabled=False,
        mode="backtest",
        symbols=("ETH/USDT:USDT",),
        timeframe="15m",
        data_dir=DATA_DIR,
    )


def _assessment(regime="RANGE", mode="RANGE_GRID", lo=2400.0, hi=2600.0):
    return MarketAssessment(
        pair="ETH/USDT:USDT",
        as_of_ms=0,
        regime=regime,
        confidence=0.7,
        reasoning_summary="stub",
        ai_available=True,
        proposal=StrategyProposal(
            regime=regime,
            confidence=0.7,
            grid_mode=mode,
            grid_lower=lo,
            grid_upper=hi,
            levels=5,
            max_capital_suggestion=100.0,
            reasoning_summary="stub",
        ),
    )


def test_config_d_calls_ai_and_records(bt_settings):
    from aigrid.backtest import GridBacktester

    calls = []

    def assess(pair, ctx, as_of_ms):
        calls.append(as_of_ms)
        return _assessment()

    bt = GridBacktester(bt_settings, config="D", assess_fn=assess)
    res = bt.run("ETH/USDT:USDT", START, END, initial_usdt=1000.0)
    assert res.ai_calls == len(calls) and res.ai_calls > 0
    assert res.ai_unavailable_cycles == 0


def test_config_d_degrades_on_ai_failure(bt_settings):
    from aigrid.backtest import GridBacktester

    bt = GridBacktester(bt_settings, config="D", assess_fn=lambda *a: None)
    res = bt.run("ETH/USDT:USDT", START, END, initial_usdt=1000.0)
    assert res.ai_calls > 0
    assert res.ai_unavailable_cycles == res.ai_calls
    assert res.final_equity_usdt > 0  # still ran (C fallback)


def test_config_d_ai_can_veto_via_safe_hold(bt_settings):
    from aigrid.backtest import GridBacktester

    bt = GridBacktester(
        bt_settings,
        config="D",
        assess_fn=lambda *a: _assessment(regime="HIGH_VOLATILITY", mode="SAFE_HOLD"),
    )
    res = bt.run("ETH/USDT:USDT", START, END, initial_usdt=1000.0)
    assert res.trade_count == 0  # AI vetoed every adoption
    assert res.plans_adopted == 0
