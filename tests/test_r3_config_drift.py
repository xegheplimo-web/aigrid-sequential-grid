"""R3 tests — T-205: preflight config-drift detection.

Real incident: the bot ran 15m/3-pair while the approved config on disk
said 5m/8 pairs and nothing warned. These pure functions compare the
*runtime* config against the *approved* one and flag dangerous safety
settings; they do no file/network I/O themselves.
"""

from __future__ import annotations

from aigrid.preflight import check_config_drift, check_telegram_and_safety

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


def _approved() -> dict:
    # Complete approved baseline: all REQUIRED_SAFETY_KEYS declared
    # (T-205c) — an incomplete baseline is UNVERIFIABLE, not a match.
    return {
        "timeframe": "5m",
        "strategy": "AIGridFT",
        "dry_run": True,
        "trading_mode": "futures",
        "max_open_trades": 3,
        "stake_amount": 140.0,
        "pair_whitelist": list(PAIRS_8),
        "max_entry_position_adjustment": 2,
        "leverage": 2,
        "stoploss": -0.10,
    }


# ------------------------------------------------------------------ drift
def test_matching_configs_report_no_drift():
    assert check_config_drift(_approved(), _approved()) == []


def test_timeframe_drift_is_reported():
    runtime = {**_approved(), "timeframe": "15m"}
    diffs = check_config_drift(runtime, _approved())
    assert diffs == ["timeframe: runtime=15m approved=5m"]


def test_whitelist_same_set_different_order_no_drift():
    runtime = {**_approved(), "pair_whitelist": list(reversed(PAIRS_8))}
    assert check_config_drift(runtime, _approved()) == []


def test_whitelist_missing_pair_flagged():
    runtime = {**_approved(), "pair_whitelist": PAIRS_8[:3]}  # 3 pairs vs 8
    diffs = check_config_drift(runtime, _approved())
    assert len(diffs) == 1
    assert diffs[0].startswith("pair_whitelist:")
    assert "NEAR/USDT:USDT" in diffs[0]  # the missing pair is visible


def test_multiple_fields_drift_each_reported():
    runtime = {**_approved(), "dry_run": False, "max_open_trades": 8, "stake_amount": "unlimited"}
    diffs = check_config_drift(runtime, _approved())
    assert len(diffs) == 3
    keys = [d.split(":", 1)[0] for d in diffs]
    assert keys == ["dry_run", "max_open_trades", "stake_amount"]


def test_key_missing_on_one_side_is_drift():
    runtime = _approved()
    del runtime["strategy"]
    diffs = check_config_drift(runtime, _approved())
    assert len(diffs) == 1
    assert diffs[0].startswith("strategy:")


# ------------------------------------------------------------------ safety
def test_dry_run_false_warns():
    warnings = check_telegram_and_safety({"dry_run": False})
    assert len(warnings) == 1
    assert "dry_run" in warnings[0]


def test_static_stoploss_disabled_warns():
    for stoploss in (-0.99, -1.0):
        warnings = check_telegram_and_safety({"stoploss": stoploss})
        assert len(warnings) == 1
        assert "stoploss" in warnings[0]


def test_unlimited_position_adjustment_warns():
    warnings = check_telegram_and_safety({"max_entry_position_adjustment": -1})
    assert len(warnings) == 1
    assert "max_entry_position_adjustment" in warnings[0]


def test_safe_runtime_no_warnings():
    warnings = check_telegram_and_safety(
        {"dry_run": True, "stoploss": -0.05, "max_entry_position_adjustment": 3}
    )
    assert warnings == []
