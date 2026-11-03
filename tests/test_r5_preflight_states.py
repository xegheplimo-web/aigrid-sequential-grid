"""R5 tests — T-205c: preflight 3-state compare (match / drift / unverifiable).

Real incident behind this round: `python -m aigrid.preflight` against the
live FT API reported `stake_amount: runtime=40 approved=40` as a drift —
show_config serializes stake_amount as str("40") while the approved JSON
has int 40 — and flagged `pair_whitelist` / `max_entry_position_adjustment`
as drift because the approved file stores the whitelist nested under
`exchange.pair_whitelist` and does not declare keys FT defaults.

New contract (card-R5 / D-092):
- numeric keys compare via float() coercion when both sides coerce
  ("40" == 40), otherwise raw equality ("unlimited" != 140)
- pair_whitelist resolves approved["exchange"]["pair_whitelist"] and
  compares as a set
- required safety keys absent in approved -> UNVERIFIABLE (never match)
- drift -> exit 1, unverifiable -> exit 3, match -> exit 0; the publish
  caller treats both 1 and 3 as blocking
"""

from __future__ import annotations

import json
from pathlib import Path

import aigrid.preflight as preflight
import pytest
from aigrid.orchestrator import AIGridOrchestrator
from aigrid.preflight import (
    REQUIRED_SAFETY_KEYS,
    check_config_drift,
    compare_configs,
)
from aigrid.settings import AIGridSettings

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "user_data/data/binance/futures"

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
    """A COMPLETE approved baseline — every required safety key declared."""
    return {
        "timeframe": "5m",
        "strategy": "AIGridStrategy",
        "dry_run": True,
        "trading_mode": "futures",
        "max_open_trades": 3,
        "stake_amount": 40,
        "max_entry_position_adjustment": -1,
        "leverage": 1,
        "stoploss": -0.10,
        "exchange": {"name": "binance", "pair_whitelist": list(PAIRS_8)},
    }


def _runtime(**overrides) -> dict:
    """What fetch_runtime_config() hands the checker: FT serializes
    stake_amount as a string and reports the whitelist top-level."""
    rt = {
        "timeframe": "5m",
        "strategy": "AIGridStrategy",
        "dry_run": True,
        "trading_mode": "futures",
        "max_open_trades": 3.0,
        "stake_amount": "40",  # str — real show_config
        "max_entry_position_adjustment": -1,
        "leverage": 1,
        "stoploss": -0.10,
        "pair_whitelist": list(PAIRS_8),
    }
    rt.update(overrides)
    return rt


# ------------------------------------------------------------------ normalization
def test_stake_amount_string_vs_int_is_match_not_drift():
    """The exact false alarm from the live API: '40' vs 40 must MATCH."""
    check = compare_configs(_runtime(), _approved())
    assert check.drift == []
    assert check.unverifiable == []
    assert check.ok
    assert check_config_drift(_runtime(), _approved()) == []


def test_numeric_normalization_max_open_trades_float_vs_int():
    approved = {**_approved(), "max_open_trades": 3}
    runtime = _runtime(max_open_trades=3.0)
    assert compare_configs(runtime, approved).ok


def test_numeric_normalization_leverage():
    runtime = _runtime(leverage="2")
    approved = {**_approved(), "leverage": 2}
    assert compare_configs(runtime, approved).ok


def test_non_numeric_stake_amount_still_drifts():
    runtime = _runtime(stake_amount="unlimited")
    check = compare_configs(runtime, _approved())
    assert len(check.drift) == 1
    assert check.drift[0].startswith("stake_amount:")


def test_numeric_drift_detected_after_normalization():
    runtime = _runtime(stake_amount="140")
    check = compare_configs(runtime, _approved())
    assert len(check.drift) == 1
    assert check.drift[0].startswith("stake_amount:")


# ------------------------------------------------------------------ whitelist fallback
def test_whitelist_read_from_exchange_section_matches_same_set():
    runtime = _runtime(pair_whitelist=list(reversed(PAIRS_8)))
    check = compare_configs(runtime, _approved())
    assert check.ok  # same set, different order, nested path


def test_whitelist_set_difference_still_drifts():
    runtime = _runtime(pair_whitelist=PAIRS_8[:4])
    check = compare_configs(runtime, _approved())
    assert len(check.drift) == 1
    assert check.drift[0].startswith("pair_whitelist:")


# ------------------------------------------------------------------ required safety keys
def test_missing_required_key_is_unverifiable_never_drift():
    approved = _approved()
    del approved["leverage"]
    check = compare_configs(_runtime(), approved)
    assert check.drift == []
    assert [u.split(":", 1)[0] for u in check.unverifiable] == ["leverage"]
    assert not check.ok
    assert check.exit_code == 3


def test_every_required_key_absent_in_approved_is_unverifiable():
    """All 9 mandatory keys: each one missing in approved -> unverifiable."""
    for key in REQUIRED_SAFETY_KEYS:
        approved = _approved()
        if key == "pair_whitelist":
            approved["exchange"] = {"name": "binance"}
        else:
            approved.pop(key, None)
        check = compare_configs(_runtime(), approved)
        assert key in [u.split(":", 1)[0] for u in check.unverifiable], (
            f"{key} missing in approved must be unverifiable"
        )
        assert not check.ok


def test_required_key_absent_on_both_sides_is_unverifiable():
    """The baseline must DECLARE safety keys — absence in approved is
    unverifiable even when runtime does not report the key either."""
    approved = _approved()
    del approved["stoploss"]
    runtime = _runtime()
    del runtime["stoploss"]
    check = compare_configs(runtime, approved)
    assert "stoploss" in [u.split(":", 1)[0] for u in check.unverifiable]
    assert check.exit_code == 3


def test_non_required_key_missing_in_approved_only_unverifiable():
    """strategy is not safety-critical, but an approved baseline that omits
    it cannot verify the live value -> unverifiable, not silent match."""
    approved = _approved()
    del approved["strategy"]
    check = compare_configs(_runtime(), approved)
    assert "strategy" in [u.split(":", 1)[0] for u in check.unverifiable]


def test_missing_value_on_runtime_side_is_drift():
    """Approved declares a value the runtime does not report -> real drift."""
    runtime = _runtime()
    del runtime["strategy"]
    check = compare_configs(runtime, _approved())
    assert len(check.drift) == 1
    assert check.drift[0].startswith("strategy:")


# ------------------------------------------------------------------ group separation
def test_drift_and_unverifiable_are_not_merged():
    runtime = _runtime(timeframe="15m")  # real drift
    approved = _approved()
    del approved["max_entry_position_adjustment"]  # and a missing key
    check = compare_configs(runtime, approved)
    assert [d.split(":", 1)[0] for d in check.drift] == ["timeframe"]
    assert [u.split(":", 1)[0] for u in check.unverifiable] == ["max_entry_position_adjustment"]
    assert check.exit_code == 1  # drift wins over 3


def test_timeframe_mismatch_is_real_drift():
    check = compare_configs(_runtime(timeframe="15m"), _approved())
    assert check.drift == ["timeframe: runtime=15m approved=5m"]
    assert check.exit_code == 1


# ------------------------------------------------------------------ compat wrapper
def test_check_config_drift_fail_closed_on_unverifiable():
    """Legacy callers must never see an unverifiable config as a match."""
    approved = _approved()
    del approved["stoploss"]
    diffs = check_config_drift(_runtime(), approved)
    assert diffs and any("stoploss" in d for d in diffs)


# ------------------------------------------------------------------ CLI main() exit codes
def _write_approved(tmp_path, data=None) -> Path:
    f = tmp_path / "approved.json"
    f.write_text(json.dumps(data or _approved()), encoding="utf-8")
    return f


def test_main_exit_1_on_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(
        preflight, "fetch_runtime_config", lambda *a, **k: _runtime(timeframe="15m")
    )
    rc = preflight.main(
        ["--approved", str(_write_approved(tmp_path)), "--api-url", "http://127.0.0.1:1"]
    )
    assert rc == 1


def test_main_exit_3_on_unverifiable(tmp_path, monkeypatch, capsys):
    approved = _approved()
    del approved["leverage"]
    monkeypatch.setattr(preflight, "fetch_runtime_config", lambda *a, **k: _runtime())
    rc = preflight.main(
        ["--approved", str(_write_approved(tmp_path, approved)), "--api-url", "http://127.0.0.1:1"]
    )
    assert rc == 3
    out = capsys.readouterr().out
    assert "UNVERIFIABLE" in out
    assert "leverage" in out


def test_main_exit_0_on_match(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "fetch_runtime_config", lambda *a, **k: _runtime())
    rc = preflight.main(
        ["--approved", str(_write_approved(tmp_path)), "--api-url", "http://127.0.0.1:1"]
    )
    assert rc == 0


# ------------------------------------------------------------------ orchestrator wiring
@pytest.fixture
def settings(tmp_path):
    return AIGridSettings(
        enabled=False,
        mode="shadow",
        symbols=("ETH/USDT:USDT",),
        timeframe="5m",
        data_dir=DATA_DIR,
        state_dir=tmp_path / "state",
    )


def test_run_preflight_unverifiable_halts_like_drift(settings):
    """Unverifiable (approved missing safety key) blocks publish too: the
    orchestrator halts and logs PREFLIGHT_UNVERIFIABLE distinctly."""
    orch = AIGridOrchestrator(settings)
    approved = _approved()
    del approved["stoploss"]
    check = orch.run_preflight(_runtime(), approved)
    assert check.unverifiable
    assert check.drift == []
    assert orch.preflight_ok is False
    assert orch._halted is True
    kinds = [e["kind"] for e in orch.eventlog.recent()]
    assert "PREFLIGHT_UNVERIFIABLE" in kinds
    orch.eventlog.close()


def test_run_preflight_clean_match_leaves_orchestrator_running(settings):
    orch = AIGridOrchestrator(settings)
    check = orch.run_preflight(_runtime(), _approved())
    assert check.ok
    assert orch.preflight_ok is True
    assert orch._halted is False
    orch.eventlog.close()
