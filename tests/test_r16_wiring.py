# Replay helpers require .orchestrator/fapi_exchangeinfo.json (unpublished).
"""Production wiring regression coverage for R16."""

import hashlib
import hmac
import json
from dataclasses import replace
from types import SimpleNamespace

from aigrid.config import load_settings
from aigrid.preflight import canonical_json
from aigrid.publish import DeterministicCycle, FtRestClient
from tests.test_r14_cycle import _df
from tests.test_r14_publish import _facts, _plan, _publish, _publisher
from tests.test_r15_attestation import proof  # noqa: F401


def test_config_snapshot_path(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[aigrid]\nexchange_info_path="snapshot.json"\n')
    settings = load_settings(path)
    assert settings.exchange_info_path == "snapshot.json"
    assert settings.enabled is False


def test_rest_preserves_raw_config(monkeypatch):
    client = FtRestClient("http://unused")
    raw = {"dry_run": True, "strategy": "AIGridStrategy", "stake_amount": 40}
    monkeypatch.setattr(client, "show_config", lambda: raw)
    monkeypatch.setattr(client, "status", list)
    monkeypatch.setattr(client, "health", dict)
    assert client.facts().runtime_config == raw


def test_two_publishes_signed_and_strategy_enabled(tmp_path, proof):  # noqa: F811
    strategy, _, target = proof
    pub = _publisher(tmp_path)
    # Use the strategy fixture's integration and secret directory.
    pub._store.state_dir = target.parent
    facts = replace(_facts(equity_usdt=2000), runtime_config=strategy.config)
    nonces = []
    for index in range(2):
        out = _publish(
            pub,
            _plan(
                pair="ETH/USDT:USDT" if index == 0 else "SOL/USDT:USDT", plan_id=f"r16-plan-{index}"
            ),
            facts=facts,
        )
        assert out.status == "PUBLISHED", out
        payload = json.loads(target.read_text())
        block = dict(payload["attestation"])
        signature = block.pop("sig")
        secret = (target.parent / ".attest_secret").read_bytes()
        assert hmac.compare_digest(
            signature, hmac.new(secret, canonical_json(block), hashlib.sha256).hexdigest()
        )
        strategy._aigrid = payload
        assert strategy._enabled() is True
        nonces.append(block["nonce"])
    assert nonces[0] != nonces[1]
    target.write_text("{}")
    assert pub.sync_active_file(facts)
    assert json.loads(target.read_text())["attestation"]


def test_missing_runtime_never_writes(tmp_path):
    pub = _publisher(tmp_path)
    result = _publish(pub, facts=replace(_facts(), runtime_config={}))
    assert "ATTESTATION_RUNTIME_MISSING" in result.reasons
    assert not (pub._store.state_dir / "ft_integration.json").exists()


def test_service_constructs_exchange_info(tmp_path, monkeypatch):
    from aigrid import service
    from aigrid.config import AIGridSettings
    from tests.test_r14_publish import EI_PATH

    captured = []
    settings = AIGridSettings(exchange_info_path=str(EI_PATH))
    monkeypatch.setattr(service, "load_settings", lambda path: settings)
    monkeypatch.setattr(service.signal, "signal", lambda *args: None)

    def orchestrator(config, *, exchange_info):
        captured.append(exchange_info)
        return SimpleNamespace(eventlog=SimpleNamespace(close=lambda: None))

    monkeypatch.setattr(service, "AIGridOrchestrator", orchestrator)
    monkeypatch.setattr(
        service, "AIGridService", lambda *args, **kwargs: SimpleNamespace(run=lambda *args: 0)
    )
    assert service.main([]) == 0
    assert captured[-1].has("ETH/USDT:USDT")
    for content in (None, "{}", '{"symbols": "bad"}'):
        path = tmp_path / "invalid.json"
        if content is not None:
            path.write_text(content)
        settings = replace(settings, exchange_info_path=str(path))
        assert service.main([]) == 0
        assert captured[-1] is None


def test_forming_candle_never_processed_or_warmed(tmp_path, monkeypatch):
    from tests.test_r14_publish import NOW_MS

    pub = _publisher(tmp_path)
    df = _df(5, end_ms=NOW_MS)
    processed = []
    orch = SimpleNamespace(process_candle=lambda pair, candle, **kwargs: processed.append(candle))
    cycle = DeterministicCycle(orch, pub, candle_loader=lambda pair: df, whitelist=["ETH"])
    monkeypatch.setattr(cycle, "_regime", lambda *args: ("HIGH_VOLATILITY", True, 0.0))
    cycle.tick()
    assert len(processed) == 4
    assert all(row["open_ms"] + 300_000 <= NOW_MS for row in processed)
