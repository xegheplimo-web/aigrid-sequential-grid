"""R4 tests — T-205b: preflight drift check is wired into the runtime.

preflight.main() reads the live Freqtrade REST config (show_config +
whitelist) and the approved JSON file, prints the drift list, and
returns 1 on drift / 0 on match. AIGridOrchestrator.run_preflight()
takes the same two dicts from the caller — no network I/O inside the
orchestrator — and on drift logs PREFLIGHT_DRIFT + sets self._halted,
which blocks new entries (analysis dispatch, plan adoption, grid fills)
while protection on existing positions keeps running.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aigrid.preflight as preflight
import pytest
from aigrid.orchestrator import AIGridOrchestrator
from aigrid.plans.schema import AIGridPlan, GridMode
from aigrid.scheduler import TriggerKind
from aigrid.settings import AIGridSettings

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "user_data/data/binance/futures"
NOW = datetime.now(UTC)
PAIR = "ETH/USDT:USDT"

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


def _approved() -> dict:
    # Complete approved baseline: all REQUIRED_SAFETY_KEYS declared
    # (T-205c) — an incomplete baseline is UNVERIFIABLE, not a match.
    return {
        "timeframe": "5m",
        "strategy": "AIGridStrategy",
        "dry_run": True,
        "trading_mode": "futures",
        "max_open_trades": 3,
        "stake_amount": 140.0,
        "pair_whitelist": list(PAIRS_8),
        "max_entry_position_adjustment": 2,
        "leverage": 2,
        "stoploss": -0.10,
    }


def _plan(**kw) -> AIGridPlan:
    base = dict(
        plan_id="plan-r4pf00000001",
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


# ------------------------------------------------------------------ orchestrator wiring
def test_run_preflight_drift_halts_orchestrator(settings):
    """Runtime timeframe drift -> non-empty diff list, eventlog entry, and
    the whole entry path blocked (analysis dispatch, plan adoption, fills)."""
    orch = AIGridOrchestrator(settings)
    runtime = {**_approved(), "timeframe": "15m"}
    check = orch.run_preflight(runtime, _approved())
    assert check.drift
    assert orch.preflight_ok is False
    assert orch._halted is True

    # analysis path refuses to dispatch
    res = orch.run_analysis(PAIR, TriggerKind.PERIODIC)
    assert res["dispatched"] is False

    # plan adoption refused while halted -> nothing may enter
    orch.adopt_plan(_plan())
    assert PAIR not in orch._engines

    # drift was written to the eventlog
    kinds = [e["kind"] for e in orch.eventlog.recent()]
    assert "PREFLIGHT_DRIFT" in kinds
    orch.close()


def test_run_preflight_match_does_not_halt(settings):
    """Matching configs -> empty diff list, no halt, entries work."""
    orch = AIGridOrchestrator(settings)
    check = orch.run_preflight(_approved(), _approved())
    assert check.ok
    assert orch.preflight_ok is True
    assert orch._halted is False

    orch.adopt_plan(_plan())
    # crosses down through levels 100/98/96 -> fills 3 lots normally
    orch.process_candle(PAIR, _candle(1_000_000, 101.0, 102.0, 95.5, 98.5))
    assert len(orch._engines[PAIR].state.lots) == 3
    orch.close()


# ------------------------------------------------------------------ CLI main()
def _write_approved(tmp_path) -> Path:
    f = tmp_path / "approved.json"
    f.write_text(json.dumps(_approved()), encoding="utf-8")
    return f


def test_main_returns_1_on_drift(tmp_path, monkeypatch, capsys):
    """Runtime fetched from the REST API drifts -> exit 1 + drift printed."""
    approved_file = _write_approved(tmp_path)
    runtime = {**_approved(), "max_open_trades": 8}
    monkeypatch.setattr(preflight, "fetch_runtime_config", lambda *a, **k: runtime)
    rc = preflight.main(["--approved", str(approved_file), "--api-url", "http://127.0.0.1:1"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "max_open_trades" in out


def test_main_returns_0_on_match(tmp_path, monkeypatch):
    """Runtime matches approved -> exit 0."""
    approved_file = _write_approved(tmp_path)
    monkeypatch.setattr(preflight, "fetch_runtime_config", lambda *a, **k: _approved())
    rc = preflight.main(["--approved", str(approved_file), "--api-url", "http://127.0.0.1:1"])
    assert rc == 0


def test_main_fails_closed_when_approved_unreadable(tmp_path, monkeypatch):
    """Approved config missing/unreadable -> non-zero (never silent pass)."""
    monkeypatch.setattr(preflight, "fetch_runtime_config", lambda *a, **k: _approved())
    rc = preflight.main(
        ["--approved", str(tmp_path / "missing.json"), "--api-url", "http://127.0.0.1:1"]
    )
    assert rc != 0
