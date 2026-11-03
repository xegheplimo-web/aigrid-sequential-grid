"""PHASE 2 tests: regime engine (4 states), controller (PAUSE/CONFIRM), plan schema + validator."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from aigrid.exchange_info import ExchangeInfo
from aigrid.plans.schema import AIGridPlan, GridMode, draft_plan_from_assessment
from aigrid.plans.validator import PlanValidator
from aigrid.regime.controller import RegimeController
from aigrid.regime.engine import (
    DOWNTREND,
    HIGH_VOLATILITY,
    RANGE,
    UPTREND,
    classify,
)
from aigrid.settings import RiskLimits

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "user_data/data/binance/futures"
EI_PATH = ROOT / ".orchestrator/fapi_exchangeinfo.json"
AS_OF = datetime(2026, 10, 11, tzinfo=UTC)


@pytest.fixture(scope="module")
def adapter():
    from aigrid.data.binance_adapter import BinanceFuturesDataAdapter

    return BinanceFuturesDataAdapter(DATA_DIR)


@pytest.fixture(scope="module")
def ei():
    return ExchangeInfo(EI_PATH)


@pytest.fixture(scope="module")
def limits():
    return RiskLimits()


# ------------------------------------------------------------------ regime engine
def _ohlc(close: np.ndarray, spread=0.001, vol=1e6) -> pd.DataFrame:
    n = len(close)
    high = close * (1 + spread)
    low = close * (1 - spread)
    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, vol),
        },
        index=pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC"),
    )


def test_regime_flat_market_is_range():
    rng = np.random.default_rng(7)
    base = 100.0
    close = base + rng.normal(0, 0.02, 400).cumsum() * 0.0 + rng.normal(0, 0.03, 400)
    res = classify(_ohlc(close))
    assert res.regime == RANGE, res


def test_regime_uptrend_detected():
    close = np.concatenate(
        [
            np.full(120, 100.0) + np.random.default_rng(1).normal(0, 0.05, 120),
            100.0 + np.linspace(0, 30, 280) + np.random.default_rng(2).normal(0, 0.08, 280),
        ]
    )
    res = classify(_ohlc(close))
    assert res.regime == UPTREND, res


def test_regime_downtrend_detected():
    close = np.concatenate(
        [
            np.full(120, 130.0) + np.random.default_rng(3).normal(0, 0.05, 120),
            130.0 - np.linspace(0, 35, 280) + np.random.default_rng(4).normal(0, 0.08, 280),
        ]
    )
    res = classify(_ohlc(close))
    assert res.regime == DOWNTREND, res


def test_regime_flash_crash_is_high_volatility():
    close = np.concatenate(
        [
            np.full(380, 100.0) + np.random.default_rng(5).normal(0, 0.03, 380),
            [100.0, 95.0, 88.0, 80.0, 82.0, 85.0, 84.0, 86.0],
            np.full(12, 85.0),
        ]
    )
    res = classify(_ohlc(close))
    assert res.regime == HIGH_VOLATILITY, res


def test_regime_insufficient_history_fails_closed():
    res = classify(_ohlc(np.full(30, 100.0)))
    assert res.regime == HIGH_VOLATILITY
    assert res.reason == "INSUFFICIENT_HISTORY"
    assert res.confidence == 0.0


def test_regime_low_liquidity_fails_closed():
    res = classify(_ohlc(np.full(300, 100.0), vol=10.0))  # tiny volume
    assert res.regime == HIGH_VOLATILITY
    assert res.reason == "LOW_LIQUIDITY"


def test_regime_on_real_data_eth_range_or_trend(adapter):
    df = adapter.get_ohlcv("ETH/USDT:USDT", "15m", as_of=AS_OF, lookback=400)
    res = classify(df)
    assert res.regime in (RANGE, UPTREND, DOWNTREND, HIGH_VOLATILITY)
    assert 0 <= res.confidence <= 1
    assert res.reason


def test_regime_deterministic(adapter):
    df = adapter.get_ohlcv("ETH/USDT:USDT", "15m", as_of=AS_OF, lookback=400)
    r1, r2 = classify(df), classify(df)
    assert r1 == r2


# ------------------------------------------------------------------ controller
def test_controller_requires_confirmation_candles():
    c = RegimeController(confirm_candles=3, switch_cooldown_candles=0, initial=HIGH_VOLATILITY)
    assert c.confirmed == HIGH_VOLATILITY and c.paused
    for _ in range(2):
        assert c.update(RANGE, 0) == HIGH_VOLATILITY  # not yet confirmed
    assert c.update(RANGE, 1) == RANGE  # 3rd agreeing candle confirms
    assert not c.paused  # resume after HV -> trading


def test_controller_hysteresis_blocks_flapping():
    c = RegimeController(confirm_candles=3, switch_cooldown_candles=6, initial=RANGE)
    c.resume()
    # single contrary candle must NOT flip the regime
    assert c.update(HIGH_VOLATILITY, 1) == RANGE
    assert c.update(UPTREND, 2) == RANGE
    assert c.update(DOWNTREND, 3) == RANGE
    assert c.update(UPTREND, 4) == RANGE
    assert c.update(UPTREND, 5) == RANGE
    # 3 consecutive UPTREND (candles 4,5,6) + cooldown 6 satisfied at candle 6 -> switch
    assert c.update(UPTREND, 6) == UPTREND


def test_controller_never_closes_positions_on_regime_change():
    """Regime change pauses new entries but exposes no 'close all' action."""
    c = RegimeController(confirm_candles=1, switch_cooldown_candles=0, initial=RANGE)
    c.resume()
    c.update(HIGH_VOLATILITY, 1)
    assert c.paused and c.pause_reason == "REGIME_HIGH_VOLATILITY"
    assert c.grid_mode_for() == "SAFE_HOLD"
    # controller API has no close-positions method by design
    assert not any(
        "close" in m.lower() for m in dir(c) if m.startswith("on_") or m.startswith("do_")
    )


def test_controller_grid_mode_mapping():
    c = RegimeController(initial=RANGE)
    c.resume()
    assert c.grid_mode_for() == "RANGE_GRID"
    c.pause("TEST")
    assert c.grid_mode_for() == "SAFE_HOLD"


# ------------------------------------------------------------------ plan schema
def _plan(**overrides) -> AIGridPlan:
    now = datetime.now(UTC)
    base = dict(
        plan_id="plan-abcdef123456",
        plan_version=1,
        symbol="ETH/USDT:USDT",
        created_at=now,
        expires_at=now + timedelta(minutes=45),
        market_regime=RANGE,
        confidence=0.7,
        grid_mode=GridMode.RANGE_GRID,
        grid_lower=2900.0,
        grid_upper=3100.0,
        grid_levels=6,
        grid_spacing=0.007,
        # consistent: notional (200) = margin (100) x leverage (2)
        max_capital=100.0,
        max_position_size=200.0,
        max_leverage=2.0,
        stop_loss=2850.0,
        take_profit=None,
        invalidation_conditions=["PRICE_BREAKS_GRID_BOUNDARY"],
        risk_level="MEDIUM",
        reasoning_summary="test",
    )
    base.update(overrides)
    return AIGridPlan(**base)


def test_plan_schema_accepts_valid():
    p = _plan()
    assert p.grid_mode == GridMode.RANGE_GRID
    assert len(p.level_prices()) == 6
    assert p.level_prices()[0] == pytest.approx(3100.0)
    assert p.level_prices()[-1] == pytest.approx(2900.0)


def test_plan_schema_rejects_bad_bounds():
    with pytest.raises(Exception):
        _plan(grid_upper=2800.0)  # upper <= lower
    with pytest.raises(Exception):
        _plan(expires_at=datetime.now(UTC) - timedelta(minutes=1))
    with pytest.raises(Exception):
        _plan(symbol="ETH/USDT")  # spot-style symbol
    with pytest.raises(Exception):
        _plan(confidence=1.5)
    with pytest.raises(Exception):
        _plan(max_leverage=25)


def test_plan_expiry_check():
    now = datetime.now(UTC)
    p = _plan(created_at=now, expires_at=now + timedelta(seconds=1))
    assert not p.is_expired()
    p2 = _plan(created_at=now - timedelta(hours=1), expires_at=now - timedelta(seconds=1))
    assert p2.is_expired()
    with pytest.raises(Exception):
        _plan(expires_at=now - timedelta(seconds=1))  # expiry before creation rejected


def test_draft_plan_safe_hold():
    p = draft_plan_from_assessment(
        "ETH/USDT:USDT",
        HIGH_VOLATILITY,
        0.4,
        3000.0,
        20.0,
        grid_mode="SAFE_HOLD",
        ttl_minutes=45,
        margin_usdt=0.0,
        leverage=1.0,
        levels=2,
    )
    assert p.grid_mode == GridMode.SAFE_HOLD
    assert p.max_capital == 0.0


def test_draft_plan_long_grid_geometry():
    p = draft_plan_from_assessment(
        "ETH/USDT:USDT",
        UPTREND,
        0.8,
        3000.0,
        15.0,
        grid_mode="LONG_GRID",
        ttl_minutes=45,
        margin_usdt=200.0,
        leverage=3.0,
        levels=6,
    )
    assert p.grid_mode == GridMode.LONG_GRID
    # LONG_GRID: all buy levels sit at/below the current price (buy the dips)
    assert p.grid_upper <= 3000.0
    assert p.grid_lower < 3000.0
    assert p.stop_loss < p.grid_lower
    assert p.ttl_seconds() == pytest.approx(45 * 60, rel=0.01)


# ------------------------------------------------------------------ validator
def test_validator_approves_reasonable_plan(ei, limits):
    v = PlanValidator(ei, limits)
    p = _plan()
    verdict = v.validate(p, last_close=3000.0)
    assert verdict.approved, verdict.reasons


def test_validator_rejects_capital_over_limit(ei, limits):
    v = PlanValidator(ei, limits)
    verdict = v.validate(
        _plan(max_capital=10_000.0, max_position_size=20_000.0), last_close=3000.0
    )  # 10k margin x2 = 20k notional > 250 limit
    assert not verdict.approved
    assert "PLAN_CAPITAL_EXCEEDS_POSITION_LIMIT" in verdict.reasons


def test_validator_rejects_leverage_over_limit(ei, limits):
    v = PlanValidator(ei, limits)
    verdict = v.validate(
        _plan(max_capital=25.0, max_position_size=250.0, max_leverage=10.0), last_close=3000.0
    )
    assert not verdict.approved
    assert "PLAN_LEVERAGE_EXCEEDS_LIMIT" in verdict.reasons


def test_validator_rejects_grid_far_from_price(ei, limits):
    v = PlanValidator(ei, limits)
    verdict = v.validate(
        _plan(grid_lower=100.0, grid_upper=110.0, stop_loss=95.0), last_close=3000.0
    )
    assert not verdict.approved
    assert "GRID_FAR_FROM_MARKET_PRICE" in verdict.reasons


def test_validator_rejects_stop_inside_grid(ei, limits):
    v = PlanValidator(ei, limits)
    verdict = v.validate(_plan(stop_loss=2950.0), last_close=3000.0)
    assert not verdict.approved
    assert "STOP_LOSS_INSIDE_OR_ABOVE_GRID" in verdict.reasons


def test_validator_rejects_unknown_symbol(ei, limits):
    v = PlanValidator(ei, limits)
    verdict = v.validate(_plan(symbol="FAKE/USDT:USDT"), last_close=3000.0)
    assert not verdict.approved
    assert "SYMBOL_NOT_IN_EXCHANGE_INFO" in verdict.reasons


def test_validator_rejects_low_confidence(ei, limits):
    v = PlanValidator(ei, limits)
    verdict = v.validate(_plan(confidence=0.1), last_close=3000.0)
    assert not verdict.approved
    assert "CONFIDENCE_TOO_LOW" in verdict.reasons


def test_validator_rejects_expired_plan(ei, limits):
    v = PlanValidator(ei, limits)
    p = _plan(
        created_at=datetime.now(UTC) - timedelta(hours=2),
        expires_at=datetime.now(UTC) - timedelta(minutes=30),
    )
    verdict = v.validate(p, last_close=3000.0)
    assert not verdict.approved
    assert "PLAN_ALREADY_EXPIRED" in verdict.reasons


def test_validator_min_notional_respected(ei, limits):
    """Level notional below Binance minNotional (20 USDT) must be rejected."""
    v = PlanValidator(ei, limits)
    verdict = v.validate(
        _plan(max_capital=60.0, max_position_size=60.0, max_leverage=1.0, grid_levels=6),
        last_close=3000.0,
    )  # 10 USDT/level < 20 minNotional
    assert not verdict.approved
    assert "LEVEL_NOTIONAL_BELOW_MIN_NOTIONAL" in verdict.reasons


def test_validator_total_exposure_check(ei, limits):
    v = PlanValidator(ei, limits)
    verdict = v.validate(
        _plan(max_capital=125.0, max_position_size=250.0),
        last_close=3000.0,
        total_exposure_after=300.0,
    )  # 300+250 > 500 total
    assert not verdict.approved
    assert "PLAN_CAPITAL_EXCEEDS_TOTAL_EXPOSURE" in verdict.reasons


def test_validator_safe_hold_plan(ei, limits):
    v = PlanValidator(ei, limits)
    p = draft_plan_from_assessment(
        "ETH/USDT:USDT",
        HIGH_VOLATILITY,
        0.4,
        3000.0,
        20.0,
        grid_mode="SAFE_HOLD",
        ttl_minutes=45,
        margin_usdt=0.0,
        leverage=1.0,
        levels=2,
    )
    verdict = v.validate(p, last_close=3000.0)
    assert verdict.approved, verdict.reasons
