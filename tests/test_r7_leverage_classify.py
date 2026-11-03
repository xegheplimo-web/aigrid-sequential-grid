"""R7 tests — T-205e: leverage classification in preflight compare.

FT ``show_config`` does NOT expose ``leverage`` (it is a per-trade property,
not a runtime config value). Before R7 a missing runtime key was classified
as DRIFT (``runtime=<absent> approved=1.0``) — wrong: the value is
unverifiable through that source, not drifting.

New contract:
- approved=1.0 + runtime absent + strategy has NO leverage() callback ->
  MATCH with note "1x by policy; verify trade.leverage == 1.0 at first fill"
  (FT defaults to 1x without the callback — the declared value is the
  only possible value)
- strategy HAS a leverage() callback -> UNVERIFIABLE until a real trade
  fills (the runtime could be running anything)
- approved declares != 1.0 with NO callback -> DRIFT (the declared value
  provably cannot be running: no callback means FT always uses 1x)
- other safety keys (stoploss, timeframe, max_entry_position_adjustment)
  keep verifying from the real source — a real mismatch stays DRIFT
"""

from __future__ import annotations

from pathlib import Path

from aigrid.preflight import compare_configs

ROOT = Path(__file__).resolve().parents[1]

PAIRS_8 = [
    "SOL/USDT:USDT",
    "DOGE/USDT:USDT",
    "XRP/USDT:USDT",
    "AVAX/USDT:USDT",
    "SUI/USDT:USDT",
    "LINK/USDT:USDT",
    "ADA/USDT:USDT",
    "NEAR/USDT:USDT",
]


def _approved(**overrides) -> dict:
    cfg = {
        "timeframe": "5m",
        "strategy": "AIGridStrategy",
        "dry_run": True,
        "trading_mode": "futures",
        "max_open_trades": 3,
        "stake_amount": 40,
        "max_entry_position_adjustment": -1,
        "leverage": 1.0,
        "stoploss": -0.10,
        "exchange": {"name": "binance", "pair_whitelist": list(PAIRS_8)},
    }
    cfg.update(overrides)
    return cfg


def _runtime(**overrides) -> dict:
    """What fetch_runtime_config produces: show_config + whitelist. NOTE:
    no "leverage" key — FT does not expose it there."""
    rt = {
        "timeframe": "5m",
        "strategy": "AIGridStrategy",
        "dry_run": True,
        "trading_mode": "futures",
        "max_open_trades": 3.0,
        "stake_amount": "40",
        "max_entry_position_adjustment": -1,
        "stoploss": -0.10,
        "pair_whitelist": list(PAIRS_8),
    }
    rt.update(overrides)
    return rt


# ------------------------------------------------------------------ policy match
def test_leverage_1x_no_callback_is_policy_match():
    """approved leverage=1.0, runtime silent, no leverage() callback ->
    match with the first-fill verification note, exit 0."""
    check = compare_configs(_runtime(), _approved())
    assert check.ok
    assert check.exit_code == 0
    assert check.drift == []
    assert check.unverifiable == []
    lev = [m for m in check.match if m.startswith("leverage")]
    assert lev, "leverage must land in match, not drift/unverifiable"
    assert "1x by policy" in lev[0]


def test_leverage_absent_runtime_is_not_drift():
    """The R7 bug in one assert: missing-from-show_config must NEVER be
    classified as drift."""
    check = compare_configs(_runtime(), _approved())
    assert not any(d.startswith("leverage") for d in check.drift)


# ------------------------------------------------------------------ callback present
def test_leverage_with_callback_is_unverifiable():
    """A strategy-side leverage() can return anything per trade — absent
    runtime proof it stays unverifiable (exit 3) until first fill."""
    check = compare_configs(_runtime(), _approved(), has_leverage_callback=True)
    assert check.exit_code == 3
    assert not check.ok
    lev = [u for u in check.unverifiable if u.startswith("leverage")]
    assert lev
    assert "not exposed by FT show_config" in lev[0]


def test_leverage_declared_above_1x_no_callback_drifts():
    """approved=2.0 + no callback: FT can only run 1x, so the declared
    value provably is not live -> honest drift, not unverifiable."""
    check = compare_configs(_runtime(), _approved(leverage=2.0))
    assert check.exit_code == 1
    assert any(d.startswith("leverage") for d in check.drift)


# ------------------------------------------------------------------ no loosening elsewhere
def test_stoploss_drift_still_drifts_with_leverage_absent():
    """Leverage being unexposable must not loosen the other keys: a real
    stoploss mismatch remains drift / exit 1."""
    check = compare_configs(_runtime(stoploss=-0.20), _approved())
    assert check.exit_code == 1
    assert any(d.startswith("stoploss") for d in check.drift)
    # leverage itself still classified as policy-match, not drift
    assert not any(d.startswith("leverage") for d in check.drift)


def test_timeframe_drift_still_drifts_with_leverage_absent():
    check = compare_configs(_runtime(timeframe="15m"), _approved())
    assert check.exit_code == 1
    assert "timeframe: runtime=15m approved=5m" in check.drift


def test_runtime_reporting_leverage_still_verified():
    """If the source ever DOES report leverage, it compares normally."""
    check = compare_configs(_runtime(leverage=1.0), _approved())
    assert check.ok
    check2 = compare_configs(_runtime(leverage=3.0), _approved())
    assert check2.exit_code == 1
    assert any(d.startswith("leverage") for d in check2.drift)
