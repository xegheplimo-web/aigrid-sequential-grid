"""R5 tests — T-206a: AIGridService shadow loop.

The service is a long-running supervisor loop: one preflight at startup,
then heartbeat-only ticks (NO entries, NO publish, NEVER writes
ft_integration.json in this round). Heartbeats are atomic (tmp +
os.replace) into <state_dir>/heartbeat.json so a monitor always sees a
complete file. SIGINT/SIGTERM -> stopped=true -> clean exit.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from aigrid.orchestrator import AIGridOrchestrator
from aigrid.preflight import ConfigCheck
from aigrid.service import AIGridService
from aigrid.settings import AIGridSettings
from aigrid.state.store import EventLog

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "user_data/data/binance/futures"
PAIR = "ETH/USDT:USDT"


class _Clock:
    """Deterministic clock: each call advances 30s."""

    def __init__(self, start: float = 1_700_000_000.0):
        self.t = start

    def __call__(self) -> float:
        self.t += 30.0
        return self.t


class _FakeOrch:
    """Minimal orchestrator stand-in: preflight result is injected, no I/O
    beyond the real EventLog the service writes into."""

    def __init__(self, state_dir: Path, check: ConfigCheck | None = None, enabled: bool = False):
        self.settings = SimpleNamespace(state_dir=state_dir, enabled=enabled)
        self.eventlog = EventLog(state_dir)
        self.preflight_ok = False
        self._check = check or ConfigCheck(drift=[], unverifiable=[], match=[])
        self.preflight_calls = 0

    def run_preflight(self, runtime_cfg: dict, approved_cfg: dict) -> ConfigCheck:
        self.preflight_calls += 1
        self.preflight_ok = self._check.ok
        return self._check


def _read_heartbeat(state_dir: Path) -> dict:
    return json.loads((state_dir / "heartbeat.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ heartbeat loop
def test_three_ticks_atomic_heartbeat_cycle_and_ts(tmp_path):
    """3 fake-clock ticks -> heartbeat cycle=3, ts_ms strictly increasing,
    no leftover .tmp files (atomic write via tmp + os.replace)."""
    state_dir = tmp_path / "state"
    orch = _FakeOrch(state_dir)
    svc = AIGridService(orch, interval_s=0, clock=_Clock())
    seen: list[dict] = []

    def on_cycle(cycle: int) -> None:
        seen.append(_read_heartbeat(state_dir))
        if cycle >= 3:
            svc.stop()

    svc.on_cycle = on_cycle
    rc = svc.run()
    assert rc == 0
    assert len(seen) == 3
    final = _read_heartbeat(state_dir)
    assert final["cycle"] == 3
    assert final["stopped"] is True
    assert final["pid"] > 0
    ts = [h["ts_ms"] for h in seen] + [final["ts_ms"]]
    assert ts == sorted(ts) and len(set(ts)) == len(ts)  # strictly increasing
    assert not list(state_dir.glob("*.tmp"))  # atomic: no temps


def test_on_cycle_constructor_arg(tmp_path):
    """on_cycle is a constructor injection point (spec signature)."""
    state_dir = tmp_path / "state"
    svc = AIGridService(
        _FakeOrch(state_dir), interval_s=0, clock=_Clock(), on_cycle=lambda c: svc.stop()
    )
    assert svc.run() == 0


def test_stop_marks_stopped_and_exits_fast(tmp_path):
    """stop() mid-loop -> heartbeat stopped=true, loop exits well under 2s
    even with a long interval (the wait is interruptible)."""
    state_dir = tmp_path / "state"
    svc = AIGridService(_FakeOrch(state_dir), interval_s=30)
    holder: dict[str, int] = {}

    th = threading.Thread(target=lambda: holder.__setitem__("rc", svc.run()), daemon=True)
    started = time.monotonic()
    th.start()
    time.sleep(0.2)
    svc.stop()
    th.join(timeout=2.0)
    elapsed = time.monotonic() - started
    assert not th.is_alive(), "service did not exit within 2s of stop()"
    assert elapsed < 2.0
    assert holder["rc"] == 0
    assert _read_heartbeat(state_dir)["stopped"] is True


# ------------------------------------------------------------------ preflight gate
def test_preflight_drift_blocks_loop(tmp_path):
    """run_preflight reports drift -> service never enters the tick loop,
    returns error code 2 (card-R5), heartbeat shows stopped."""
    state_dir = tmp_path / "state"
    orch = _FakeOrch(
        state_dir,
        check=ConfigCheck(drift=["timeframe: runtime=15m approved=5m"], unverifiable=[], match=[]),
    )
    svc = AIGridService(orch, interval_s=0, clock=_Clock())
    ticks = []
    svc.on_cycle = ticks.append
    rc = svc.run({"a": 1}, {"b": 2})
    assert rc == 2
    assert ticks == []
    assert orch.preflight_calls == 1  # exactly once, at startup
    hb = _read_heartbeat(state_dir)
    assert hb["stopped"] is True
    assert hb["preflight_ok"] is False
    kinds = [e["kind"] for e in orch.eventlog.recent()]
    assert "SERVICE_PREFLIGHT_BLOCKED" in kinds


def test_preflight_unverifiable_blocks_loop(tmp_path):
    """Unverifiable also blocks (exit 1 AND 3 block publish -> the loop
    never starts without a verified baseline)."""
    state_dir = tmp_path / "state"
    orch = _FakeOrch(
        state_dir,
        check=ConfigCheck(
            drift=[], unverifiable=["leverage: runtime=1 approved=<absent>"], match=[]
        ),
    )
    svc = AIGridService(orch, interval_s=0, clock=_Clock())
    ticks = []
    svc.on_cycle = ticks.append
    rc = svc.run({"a": 1}, {"b": 2})
    assert rc == 2
    assert ticks == []


def test_preflight_called_once_then_loop_runs(tmp_path):
    """Clean preflight -> exactly one call at startup, then ticks proceed."""
    state_dir = tmp_path / "state"
    orch = _FakeOrch(state_dir)
    svc = AIGridService(
        orch, interval_s=0, clock=_Clock(), on_cycle=lambda c: c >= 2 and svc.stop()
    )
    rc = svc.run({"a": 1}, {"b": 2})
    assert rc == 0
    assert orch.preflight_calls == 1
    assert _read_heartbeat(state_dir)["preflight_ok"] is True


def test_no_configs_means_no_preflight(tmp_path):
    """Service can run with no approved config handed over — loop +
    heartbeat only, preflight_ok stays false (honest: not verified)."""
    state_dir = tmp_path / "state"
    orch = _FakeOrch(state_dir)
    svc = AIGridService(
        orch, interval_s=0, clock=_Clock(), on_cycle=lambda c: c >= 2 and svc.stop()
    )
    rc = svc.run()
    assert rc == 0
    assert orch.preflight_calls == 0
    assert _read_heartbeat(state_dir)["preflight_ok"] is False


# ------------------------------------------------------------------ restart keeps state
def test_restart_resumes_cycle_and_logs(tmp_path):
    """An existing heartbeat.json survives restart: the old cycle is read
    back, logged as resumed_from_cycle, and the counter continues."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "heartbeat.json").write_text(
        json.dumps({"ts_ms": 1, "cycle": 7, "pid": 1, "preflight_ok": True, "stopped": True}),
        encoding="utf-8",
    )
    orch = _FakeOrch(state_dir)
    svc = AIGridService(orch, interval_s=0, clock=_Clock(), on_cycle=lambda c: svc.stop())
    rc = svc.run()
    assert rc == 0
    hb = _read_heartbeat(state_dir)
    assert hb["cycle"] == 8
    resumed = [e for e in orch.eventlog.recent() if e["kind"] == "SERVICE_RESUMED"]
    assert resumed and resumed[0]["payload"]["resumed_from_cycle"] == 7


# ------------------------------------------------------------------ shadow mode
def test_enabled_false_shadow_mode_no_ft_integration(tmp_path):
    """enabled=false -> SHADOW_MODE in the eventlog; the loop must NEVER
    produce ft_integration.json (T-206a is heartbeat-only, no publish)."""
    settings = AIGridSettings(
        enabled=False,
        mode="shadow",
        symbols=(PAIR,),
        timeframe="5m",
        data_dir=DATA_DIR,
        state_dir=tmp_path / "state",
    )
    orch = AIGridOrchestrator(settings)
    svc = AIGridService(
        orch, interval_s=0, clock=_Clock(), on_cycle=lambda c: c >= 2 and svc.stop()
    )
    rc = svc.run()
    assert rc == 0
    kinds = [e["kind"] for e in orch.eventlog.recent()]
    assert "SHADOW_MODE" in kinds
    assert not (tmp_path / "state" / "ft_integration.json").exists()
    orch.eventlog.close()  # NOT orch.close(): persist() would publish
