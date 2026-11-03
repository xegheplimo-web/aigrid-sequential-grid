# Replay helpers require .orchestrator/fapi_exchangeinfo.json (unpublished).
"""R14b: spacing describes the tick-aligned grid geometry."""

from itertools import pairwise

import pytest

from test_r14_deterministic import (
    AS_OF,
    PAIR,
    _build,
    _Events,
    _flat_df,
    _range_df,
)
from test_r14_publish import _integration, _publish, _publisher


def _real_step(plan, df):
    return (
        (plan.grid_upper - plan.grid_lower) / (plan.grid_levels - 1) / float(df["close"].iloc[-1])
    )


def test_spacing_equals_real_tick_aligned_step():
    df = _range_df()
    plan = _build(df)
    assert plan is not None
    step = _real_step(plan, df)
    assert step > 0.003 * 2
    assert plan.grid_spacing == round(step, 6)
    assert f"step={step:.6f}" in plan.reasoning_summary
    assert "floor=" in plan.reasoning_summary


def test_narrow_envelope_refused_with_spacing_event():
    events = _Events()
    assert _build(_flat_df(), eventlog=events) is None
    assert events.kinds() == ["PLAN_SKIPPED_SPACING"]
    _, symbol, payload = events.rows[0]
    assert symbol == PAIR
    assert payload["step_frac"] < payload["spacing_min"]


def test_grid_levels_prices_descend_by_real_step():
    df = _range_df()
    plan = _build(df)
    assert plan is not None
    prices = plan.level_prices()
    assert len(prices) == plan.grid_levels
    assert prices[0] == plan.grid_upper
    assert prices[-1] == plan.grid_lower
    ref_close = float(df["close"].iloc[-1])
    for upper, lower in pairwise(prices):
        assert upper > lower
        step_frac = (upper - lower) / ref_close
        assert step_frac == pytest.approx(plan.grid_spacing, abs=0.5e-6)
        assert upper / 0.01 == pytest.approx(round(upper / 0.01))


def test_publish_payload_carries_real_spacing(tmp_path):
    df = _range_df()
    plan = _build(df)
    assert plan is not None
    pub = _publisher(tmp_path)
    out = _publish(
        pub,
        plan,
        last_close=float(df["close"].iloc[-1]),
        candle_close_ms=int(AS_OF.timestamp() * 1000),
    )
    assert out.status == "PUBLISHED", out
    payload = _integration(tmp_path / "state")["plans"][PAIR]
    assert payload["grid_spacing"] == round(_real_step(plan, df), 6)
    assert payload["grid_spacing"] == plan.grid_spacing
    assert payload["grid_levels_prices"] == plan.level_prices()
