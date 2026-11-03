# Replay helpers require .orchestrator/fapi_exchangeinfo.json (unpublished).
"""R19e: runtime whitelist evidence and resilient attestation signing."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from aigrid import preflight, publish
from tests._attest import make_config
from tests.test_r10_strategy_gates import _install_ft_stubs
from tests.test_r14_cycle import _cycle, _integration

CFG_PAIRS = [f"{p}/USDT:USDT" for p in ["SOL", "DOGE", "XRP", "AVAX", "SUI", "LINK", "ADA", "NEAR"]]
API_PAIRS = [f"{p}/USDT:USDT" for p in ["SOL", "XRP", "NEAR", "DOGE", "SUI", "LINK", "ADA", "AVAX"]]


@pytest.fixture
def strategy(tmp_path, monkeypatch):
    _install_ft_stubs()
    spec = importlib.util.spec_from_file_location(
        "r19e_strategy", Path("user_data/strategies/AIGridStrategy.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "INTEGRATION_FILE", tmp_path / "ft_integration.json")
    return mod


def _client(whitelist=API_PAIRS):
    cfg = make_config(exchange="binance", stake_amount="40")
    client = publish.FtRestClient("http://localhost:8080")
    replies = {
        "/api/v1/show_config": cfg,
        "/api/v1/status": [],
        "/api/v1/balance": {"total": 1000},
        "/api/v1/health": {"last_process_ts": 1_793_707_200},
        "/api/v1/whitelist": {"whitelist": whitelist},
    }
    client._get = Mock(side_effect=lambda endpoint: replies[endpoint])
    return client, cfg


def test_hash_whitelist_order():
    cfg = make_config(exchange={"pair_whitelist": CFG_PAIRS})
    api = make_config(exchange={"pair_whitelist": API_PAIRS})
    assert preflight.config_hash(cfg) == preflight.config_hash(api)
    assert cfg["exchange"]["pair_whitelist"] == CFG_PAIRS


def test_strategy_hash_matches_both_orders(strategy, monkeypatch):
    expected = preflight.config_hash(make_config(exchange={"pair_whitelist": CFG_PAIRS}))
    for pairs in (CFG_PAIRS, API_PAIRS):
        cfg = make_config(exchange={"pair_whitelist": pairs})
        assert strategy._hash(cfg) == expected
        assert strategy._local_config_hash(cfg) == expected
    monkeypatch.setattr(strategy, "_config_hash", None)
    assert strategy._hash(cfg) == expected


def test_runtime_facts_inject_whitelist(strategy):
    client, source = _client()
    facts = client.facts()
    assert facts.runtime_config["exchange"]["pair_whitelist"] == API_PAIRS
    assert source["exchange"] == "binance"
    assert facts.pair_whitelist == API_PAIRS
    cfg = make_config(exchange={"pair_whitelist": CFG_PAIRS})
    assert preflight.config_hash(facts.runtime_config) == strategy._hash(cfg)
    # Top-level REST evidence must also remain usable without the endpoint.
    source["pair_whitelist"] = CFG_PAIRS
    client.whitelist = Mock(side_effect=AssertionError("unnecessary endpoint"))
    assert client.facts().runtime_config["pair_whitelist"] == CFG_PAIRS


def test_runtime_attestation_roundtrip(strategy, tmp_path):
    client, _ = _client()
    path = tmp_path / "ft_integration.json"
    preflight.write_attestation(client.facts().runtime_config, path)
    instance = object.__new__(strategy.AIGridStrategy)
    instance.config = make_config(exchange={"pair_whitelist": CFG_PAIRS})
    assert instance._attestation_checked(json.loads(path.read_text())) == (True, "")


def test_missing_whitelist_fails_closed(tmp_path):
    client, _ = _client(None)
    cycle, _, _, events = _cycle(tmp_path)
    cycle._facts_provider = client.facts
    assert cycle.tick() == []
    assert _integration(tmp_path / "state") is None
    rows = [payload for kind, _, payload in events.rows if kind == "FACTS_UNAVAILABLE"]
    assert len(rows) == 1
    assert rows[0]["reason"] == "RUNTIME_WHITELIST_UNAVAILABLE"


def test_sign_failure_cycle_continues(tmp_path, monkeypatch):
    cycle, _, _, events = _cycle(tmp_path)
    cycle._reconciled = True
    signer = Mock(side_effect=ValueError("signing unavailable"))
    monkeypatch.setattr(publish, "write_attestation", signer)
    assert cycle.tick()
    assert cycle.seen_ts
    assert not _integration(tmp_path / "state").get("attestation")
    rows = [payload for kind, _, payload in events.rows if kind == "ATTESTATION_SIGN_FAILED"]
    assert rows == [{"reason": "signing unavailable"}]
    assert signer.call_count == 1
    # A later tick retries signing and refreshes the unsigned handoff.
    cycle.tick()
    assert signer.call_count == 2
