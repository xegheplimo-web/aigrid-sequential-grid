# Replay helpers require .orchestrator/fapi_exchangeinfo.json (unpublished).
"""Hermetic R17 stop protection and portfolio capital regressions."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from tests.test_r1_stoploss_stake import (
    PAIR,
    _strategy,
    _trade,
    _write_integration,
)
from tests.test_r1_stoploss_stake import (
    strat_mod as strategy_module_fixture,
)
from tests.test_r13_level_tp import _order
from tests.test_r13_level_tp import _trade as lot_trade
from tests.test_r14_publish import (
    NOW_MS,
    PAIR_B,
    _Events,
    _facts,
    _plan,
    _publish,
    _publisher,
)

strat_mod = strategy_module_fixture


@pytest.fixture
def protected(strat_mod, monkeypatch, tmp_path):
    integration = tmp_path / "ft_integration.json"
    monkeypatch.setattr(strat_mod, "INTEGRATION_FILE", integration)
    monkeypatch.setattr(strat_mod, "SL_CACHE_FILE", tmp_path / "sl.json")
    _write_integration(
        integration,
        {
            PAIR: {
                "stop_loss": 97.0,
                "grid_mode": "RANGE_GRID",
                "expires_at": (datetime.now(UTC) + timedelta(minutes=10)).isoformat(),
                "grid_spacing": 0.01,
                "grid_levels_prices": [99.0, 98.0],
            }
        },
    )
    strategy = _strategy(strat_mod)
    trade = _trade()
    trade.leverage = 2.0
    return strategy, trade, integration


def stop(strategy, trade, rate):
    return strategy.custom_stoploss(PAIR, trade, datetime.now(UTC), rate, -0.01, False)


def test_absolute_sl_survives_plan_expiry(protected):
    strategy, trade, integration = protected
    assert stop(strategy, trade, 100) == pytest.approx(-0.06)
    _write_integration(
        integration,
        {
            PAIR: {
                "stop_loss": 97.0,
                "expires_at": (datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
            }
        },
    )
    assert stop(strategy, trade, 99) == pytest.approx(-2 * (1 - 97 / 99))


def test_absolute_sl_survives_missing_integration_file(protected, strat_mod):
    strategy, trade, integration = protected
    assert stop(strategy, trade, 100) == pytest.approx(-0.06)
    integration.unlink()
    # A fresh instance must also recover the durable absolute SL cache.
    assert stop(_strategy(strat_mod), trade, 99) == pytest.approx(-2 * (1 - 97 / 99))


def test_sl_never_widens(protected):
    strategy, trade, _ = protected
    previous = 97.0
    for rate in (100, 105, 99, 110, 98):
        ratio = stop(strategy, trade, rate)
        absolute = rate * (1 - abs(ratio) / trade.leverage)
        assert absolute >= previous - 1e-10
        assert absolute == pytest.approx(97)
        previous = absolute


def test_custom_exit_returns_none_when_no_level_tp_reached(protected):
    strategy, _, _ = protected
    trade = lot_trade(
        orders=[_order("grid_level_0", price=99.0, filled=1.0)], amount=1.0, stake=99.0
    )
    assert strategy.custom_exit(PAIR, trade, datetime.now(UTC), 98.0, -0.02) is None
    assert strategy._live_lot_ledger(trade, strategy._level_order_stats(trade))


def test_emergency_is_handled_by_stoploss_not_custom_exit(protected):
    strategy, _, _ = protected
    trade = lot_trade(
        orders=[_order("grid_level_0", price=99.0, filled=1.0)], amount=1.0, stake=99.0
    )
    # Adverse price remains above the protective stop, which FT enforces.
    ratio = stop(strategy, trade, 98.0)
    assert ratio is not None and ratio < 0
    assert 98.0 * (1 - abs(ratio) / trade.leverage) == pytest.approx(97.0)
    assert strategy.custom_exit(PAIR, trade, datetime.now(UTC), 98.0, -0.02) is None


def test_aggregate_capital_blocked(tmp_path):
    events = _Events()
    publisher = _publisher(tmp_path, events=events)
    facts = replace(
        _facts(),
        equity_usdt=2000,
        open_trades=[
            {
                "pair": PAIR_B,
                "amount": 4.0,
                "current_rate": 100.0,
                "leverage": 2,
                "orders": [
                    {"ft_is_entry": True, "status": "open", "amount": 2, "filled": 1, "price": 100}
                ],
            }
        ],
    )
    result = _publish(publisher, facts=facts)
    assert "CAPITAL_CAP_BLOCKED" in result.reasons
    detail = events.rows[-1][2]["detail"]
    assert detail["N_mark"] == 400
    assert detail["N_pending"] == 100
    assert detail["N_projected"] == 600
    assert detail["C_pending"] == 50
    assert not (publisher._store.state_dir / "ft_integration.json").exists()


@pytest.mark.parametrize(
    "facts",
    [
        _facts(runtime_config={}),
        _facts(open_trades=None),
        replace(
            _facts(),
            equity_usdt=2000,
            open_trades=[
                {
                    "pair": PAIR_B,
                    "amount": 1,
                    "orders": [],
                }
            ],
        ),
        _facts(),
    ],
)
def test_aggregate_capital_unknown_fails_closed(tmp_path, facts):
    events = _Events()
    publisher = _publisher(tmp_path, events=events)
    assert "CAPITAL_CAP_BLOCKED" in _publish(publisher, facts=facts).reasons
    assert events.rows[-1][2]["detail"] == "UNKNOWN_EXPOSURE"


def test_active_plan_margin_and_pending_margin_count(tmp_path):
    publisher = _publisher(tmp_path)
    publisher._ledger["active"][PAIR_B] = {
        "expires_at_ms": NOW_MS + 600000,
        "plan": publisher._plan_dict(_plan(PAIR_B)),
    }
    detail = publisher._capital_guard(_plan(), replace(_facts(), equity_usdt=1000), NOW_MS)
    assert detail["C_allocated"] == 100
    assert detail["C_projected"] == 400
    assert detail["capital_budget"] == 300


def test_capital_boundary_passes(tmp_path):
    publisher = _publisher(tmp_path)
    assert publisher._capital_guard(_plan(), replace(_facts(), equity_usdt=1000), NOW_MS) is None


def test_cycle_publications_reserve_margin(tmp_path):
    publisher = _publisher(tmp_path)
    facts = replace(_facts(), equity_usdt=1000)
    assert _publish(publisher, facts=facts).status == "PUBLISHED"
    # Reuse the identical runtime snapshot. The first publication has reserved
    # its margin even though no exchange trade has appeared in that snapshot.
    result = _publish(publisher, _plan(PAIR_B), facts=facts)
    assert "CAPITAL_CAP_BLOCKED" in result.reasons


@pytest.mark.parametrize("rate", [None, 0, -1, float("nan"), float("inf")])
def test_invalid_mark_fails_closed(tmp_path, rate):
    publisher = _publisher(tmp_path)
    facts = replace(
        _facts(),
        equity_usdt=2000,
        open_trades=[
            {
                "pair": PAIR_B,
                "amount": 1,
                "current_rate": rate,
            }
        ],
    )
    assert publisher._capital_guard(_plan(), facts, NOW_MS) == "UNKNOWN_EXPOSURE"
