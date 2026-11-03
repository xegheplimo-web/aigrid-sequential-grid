"""R19: authenticate REST before constructing a cycle."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from aigrid import publish, service
from aigrid.config import AIGridSettings
from aigrid.state.store import StateStore
from tests._attest import make_config


@pytest.mark.parametrize("config", [None, {}, make_config(), make_config(pair_whitelist=[])])
def test_startup_probe(tmp_path, monkeypatch, capsys, config):
    settings = AIGridSettings(
        mode="dryrun",
        symbols=("ETH/USDT:USDT",),
        watchlist=("ETH/USDT:USDT",),
        state_dir=tmp_path,
    )
    orch = SimpleNamespace(
        settings=settings,
        store=StateStore(tmp_path),
        eventlog=Mock(),
        validator=None,
        ownership=None,
        _price_tick=Mock(),
    )
    client = Mock()
    client.show_config.return_value = config
    construct = Mock()
    run = Mock(return_value=0)
    monkeypatch.setattr(service, "load_settings", lambda _: settings)
    monkeypatch.setattr(service, "AIGridOrchestrator", lambda *a, **kw: orch)
    monkeypatch.setattr(publish, "FtRestClient", Mock(return_value=client))
    monkeypatch.setattr(publish, "DeterministicCycle", construct)
    monkeypatch.setattr(service.AIGridService, "run", run)
    monkeypatch.setattr(service.signal, "signal", Mock())
    monkeypatch.delenv(service.APPROVED_CONFIG_ENV, raising=False)
    assert service.main([]) == (0 if config else 2)
    client.show_config.assert_called_once_with()
    orch.eventlog.close.assert_called_once_with()
    if config:
        construct.assert_called_once()
        run.assert_called_once()
        runtime = config.get("pair_whitelist", config["exchange"]["pair_whitelist"])
        warnings = [
            call for call in orch.eventlog.log.call_args_list if call.args[0] == "PREFLIGHT_WARNING"
        ]
        if not runtime:
            assert len(warnings) == 1
            assert warnings[0].args == (
                "PREFLIGHT_WARNING",
                "",
                {
                    "reason": "WATCHLIST_MISMATCH",
                    "settings": ["ETH/USDT:USDT"],
                    "runtime": [],
                    "missing": ["ETH/USDT:USDT"],
                },
            )
        else:
            assert warnings == []
    else:
        assert "show_config unavailable" in capsys.readouterr().err
        construct.assert_not_called()
        run.assert_not_called()


@pytest.mark.parametrize(
    "whitelist",
    [
        pytest.param(["ETH/USDT:USDT"], id="real-show-config"),
        pytest.param(None, id="missing-pair-whitelist"),
        pytest.param("ETH/USDT:USDT", id="non-list-pair-whitelist"),
    ],
)
def test_startup_probe_real_show_config(tmp_path, monkeypatch, whitelist):
    settings = AIGridSettings(
        mode="dryrun",
        symbols=("ETH/USDT:USDT",),
        watchlist=("ETH/USDT:USDT",),
        state_dir=tmp_path,
    )
    orch = SimpleNamespace(
        settings=settings,
        store=StateStore(tmp_path),
        eventlog=Mock(),
        validator=None,
        ownership=None,
        _price_tick=Mock(),
    )
    config = {
        "exchange": "binance",
        "dry_run": True,
        "trading_mode": "futures",
        "runmode": "dry_run",
    }
    if whitelist is not None:
        config["pair_whitelist"] = whitelist
    client = Mock()
    client.show_config.return_value = config
    client.whitelist.return_value = None
    construct = Mock()
    run = Mock(return_value=0)
    monkeypatch.setattr(service, "load_settings", lambda _: settings)
    monkeypatch.setattr(service, "AIGridOrchestrator", lambda *a, **kw: orch)
    monkeypatch.setattr(publish, "FtRestClient", Mock(return_value=client))
    monkeypatch.setattr(publish, "DeterministicCycle", construct)
    monkeypatch.setattr(service.AIGridService, "run", run)
    monkeypatch.setattr(service.signal, "signal", Mock())
    monkeypatch.delenv(service.APPROVED_CONFIG_ENV, raising=False)

    assert service.main([]) == 0
    client.show_config.assert_called_once_with()
    construct.assert_called_once()
    run.assert_called_once()
    orch.eventlog.close.assert_called_once_with()
    warnings = [
        call.args
        for call in orch.eventlog.log.call_args_list
        if call.args[0] == "PREFLIGHT_WARNING"
    ]
    assert warnings == []


@pytest.mark.parametrize(
    "endpoint, missing",
    [
        (["ETH/USDT:USDT", "BTC/USDT:USDT", "SOL/USDT:USDT"], []),
        (["ETH/USDT:USDT", "BTC/USDT:USDT"], ["SOL/USDT:USDT"]),
        (None, []),
        ({"whitelist": "SOL"}, []),
    ],
)
def test_startup_whitelist_endpoint(tmp_path, monkeypatch, endpoint, missing):
    pairs = ["ETH/USDT:USDT", "BTC/USDT:USDT", "SOL/USDT:USDT"]
    settings = AIGridSettings(
        mode="dryrun",
        symbols=tuple(pairs),
        watchlist=tuple(pairs),
        state_dir=tmp_path,
    )
    orch = SimpleNamespace(
        settings=settings,
        store=StateStore(tmp_path),
        eventlog=Mock(),
        validator=None,
        ownership=None,
        _price_tick=Mock(),
    )
    client = Mock()
    client.show_config.return_value = {"exchange": "binance", "dry_run": True}
    client.whitelist.return_value = endpoint
    monkeypatch.setattr(service, "load_settings", lambda _: settings)
    monkeypatch.setattr(service, "AIGridOrchestrator", lambda *a, **kw: orch)
    monkeypatch.setattr(publish, "FtRestClient", Mock(return_value=client))
    monkeypatch.setattr(publish, "DeterministicCycle", Mock())
    monkeypatch.setattr(service.AIGridService, "run", Mock(return_value=0))
    monkeypatch.setattr(service.signal, "signal", Mock())
    monkeypatch.delenv(service.APPROVED_CONFIG_ENV, raising=False)

    assert service.main([]) == 0
    client.whitelist.assert_called_once_with()
    warnings = [
        call.args
        for call in orch.eventlog.log.call_args_list
        if call.args[0] == "PREFLIGHT_WARNING"
    ]
    assert warnings == (
        [
            (
                "PREFLIGHT_WARNING",
                "",
                {
                    "reason": "WATCHLIST_MISMATCH",
                    "settings": pairs,
                    "runtime": endpoint,
                    "missing": missing,
                },
            )
        ]
        if missing
        else []
    )


def test_rest_client_whitelist():
    client = publish.FtRestClient("http://localhost:8080")
    client._get = Mock(
        side_effect=[
            {"whitelist": ["ETH/USDT:USDT"]},
            {"detail": "Not Found"},
            {"whitelist": "SOL"},
            {"whitelist": [1]},
            OSError("absent"),
        ]
    )
    assert client.whitelist() == ["ETH/USDT:USDT"]
    for _ in range(4):
        assert client.whitelist() is None
    assert client._get.call_count == 5
    client._get.assert_called_with("/api/v1/whitelist")
