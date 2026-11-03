"""R21 single-instance startup and ownership regression tests."""

import json
import os
from types import SimpleNamespace

import pytest
from aigrid import service

_REAL_RUN = service.AIGridService.run


@pytest.fixture
def startup(tmp_path, monkeypatch):
    settings = SimpleNamespace(
        state_dir=str(tmp_path), mode="off", enabled=False, exchange_info_path=None
    )
    events = []
    sink = SimpleNamespace(log=lambda kind, *args: events.append(kind), close=lambda: None)
    monkeypatch.setattr(service, "load_settings", lambda _: settings)
    monkeypatch.setattr(
        service,
        "AIGridOrchestrator",
        lambda *a, **k: SimpleNamespace(settings=settings, eventlog=sink),
    )
    monkeypatch.setattr(service.signal, "signal", lambda *a: None)
    monkeypatch.setattr(service.AIGridService, "run", lambda *a: 0)
    return tmp_path / "aigrid-service.lock", events


def test_second_main_refused_while_first_holds_lock(startup, monkeypatch):
    path, events = startup

    def run(*args):
        assert path.exists()
        assert service.main([]) == 3
        return 0

    monkeypatch.setattr(service.AIGridService, "run", run)
    assert service.main([]) == 0
    assert not path.exists()
    assert events.count("SERVICE_LOCK_ACQUIRED") == 1
    assert events.count("SERVICE_LOCK_RELEASED") == 1


@pytest.mark.parametrize("pid", [2147483647, os.getpid(), None])
def test_stale_lock_replaced(startup, monkeypatch, pid):
    path, _ = startup
    path.write_text(json.dumps({"pid": pid}) if pid else "broken")

    def run(*args):
        record = json.loads(path.read_text())
        assert record["pid"] == os.getpid()
        assert isinstance(record["ts_ms"], int)
        assert isinstance(record["argv"], str)
        return 0

    monkeypatch.setattr(service.AIGridService, "run", run)
    assert service.main([]) == 0
    assert not path.exists()


def test_shutdown_preserves_replacement_owner(startup, monkeypatch):
    path, events = startup
    monkeypatch.setattr(os, "kill", lambda *args: None)

    def run(*args):
        assert path.exists()
        path.write_text(json.dumps({"pid": os.getpid() + 1}))
        return 0

    monkeypatch.setattr(service.AIGridService, "run", run)
    assert service.main([]) == 0
    assert json.loads(path.read_text())["pid"] == os.getpid() + 1
    assert "SERVICE_LOCK_RELEASED" not in events


@pytest.mark.parametrize("permission_denied", [False, True])
def test_live_foreign_owner_refused(startup, monkeypatch, caplog, permission_denied):
    path, events = startup
    owner = os.getpid() + 1
    path.write_text(json.dumps({"pid": owner}))

    def probe(pid, sig):
        assert (pid, sig) == (owner, 0)
        if permission_denied:
            raise PermissionError("live but protected")

    monkeypatch.setattr(os, "kill", probe)
    # Keep this Windows process-check test portable on Linux as well.
    monkeypatch.setattr("aigrid.service_lock.sys.platform", "win32")
    assert service.main([]) == 3
    assert json.loads(path.read_text())["pid"] == owner
    assert events == []
    assert "SERVICE_LOCKED" in caplog.text


def test_stale_recovery_retries_only_once(startup, monkeypatch):
    path, _ = startup
    path.write_text("broken")
    calls = []

    def raced(*args, **kwargs):
        calls.append(args)
        raise FileExistsError("another creator won")

    monkeypatch.setattr(os, "open", raced)
    assert service.main([]) == 3
    assert len(calls) == 2


def test_unwritable_state_exits_two(startup, monkeypatch, caplog):
    def denied(*args, **kwargs):
        raise PermissionError("state directory is unwritable")

    monkeypatch.setattr(os, "open", denied)
    assert service.main([]) == 2
    assert "unwritable" in caplog.text.lower()


def test_lock_file_override(startup, monkeypatch):
    path, _ = startup
    override = path.with_name("custom.lock")

    def run(*args):
        assert override.exists()
        assert not path.exists()
        return 0

    monkeypatch.setattr(service.AIGridService, "run", run)
    assert service.main(["--lock-file", str(override)]) == 0
    assert not override.exists()


@pytest.mark.parametrize("signum", [service.signal.SIGINT, service.signal.SIGTERM])
def test_signal_stop_releases_lock(startup, monkeypatch, signum):
    path, events = startup
    handlers = {}
    monkeypatch.setattr(service.signal, "signal", lambda sig, fn: handlers.update({sig: fn}))

    def run(self, *args):
        self.on_cycle = lambda _: handlers[signum](signum, None)
        return _REAL_RUN(self, *args)

    monkeypatch.setattr(service.AIGridService, "run", run)
    assert service.main([]) == 0
    assert json.loads(path.with_name("heartbeat.json").read_text())["stopped"] is True
    assert not path.exists()
    assert events.count("SERVICE_LOCK_RELEASED") == 1
