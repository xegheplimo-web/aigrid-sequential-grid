"""R15 hermetic attestation contract tests."""

import hashlib
import hmac
import importlib.util
import json
import time
from pathlib import Path

import pytest
from aigrid import preflight
from tests.test_r10_strategy_gates import _install_ft_stubs


@pytest.fixture
def proof(tmp_path, monkeypatch):
    _install_ft_stubs()
    spec = importlib.util.spec_from_file_location(
        "r15_strategy", Path("user_data/strategies/AIGridStrategy.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    path = tmp_path / "ft_integration.json"
    monkeypatch.setattr(mod, "INTEGRATION_FILE", path)
    s = object.__new__(mod.AIGridStrategy)
    s.config = {
        "strategy": "AIGridStrategy",
        "dry_run": True,
        "trading_mode": "futures",
        "stake_amount": 40.0,
        "max_open_trades": 3,
        "exchange": {"pair_whitelist": ["ETH"]},
    }
    s._plan = {"plan_id": "r15"}
    secret = b"r" * 32
    (tmp_path / ".attest_secret").write_bytes(secret)

    def write(**changes):
        block = {
            "ts_ms": int(time.time() * 1000),
            "nonce": "1234abcd",
            "container_id": "hermetic",
            "config_hash": preflight.config_hash(s.config),
        }
        block.update(changes)
        block["sig"] = hmac.new(secret, preflight.canonical_json(block), hashlib.sha256).hexdigest()
        s._aigrid = {"attestation": block}
        path.write_text(json.dumps(s._aigrid))
        return block

    return s, write, path


def test_strategy_blocks_live_runtime(proof, caplog):
    s, write, _ = proof
    write()
    s.config["dry_run"] = False
    assert s._enabled() is False
    assert s._runtime_dryrun_checked() == (False, "RUNTIME_NOT_DRY")
    assert "DRYRUN_UNPROVEN" in caplog.text


def test_strategy_blocks_missing_attestation(proof, caplog):
    s, _, path = proof
    s._aigrid = {"dry_run_verified": True}
    path.write_text(json.dumps(s._aigrid))
    assert s._enabled() is False
    assert "ATTESTATION_MISSING" in caplog.text


def test_strategy_blocks_stale_attestation(proof, caplog):
    s, write, _ = proof
    write(ts_ms=int(time.time() * 1000) - 45 * 60 * 1000)
    assert s._enabled() is False
    assert "ATTESTATION_STALE" in caplog.text


def test_strategy_blocks_config_mismatch(proof, caplog):
    s, write, _ = proof
    write()
    s.config["stake_amount"] = 41
    assert s._enabled() is False
    assert "ATTESTATION_CONFIG_MISMATCH" in caplog.text


def test_strategy_blocks_bad_signature(proof, caplog):
    s, write, path = proof
    block = write()
    block["sig"] = ("0" if block["sig"][0] != "0" else "1") + block["sig"][1:]
    path.write_text(json.dumps(s._aigrid))
    assert s._enabled() is False
    assert "ATTESTATION_BAD_SIG" in caplog.text


def test_strategy_allows_valid_attestation(proof):
    s, write, _ = proof
    write()
    assert s._enabled() is True


def test_config_hash_stable_and_sensitive():
    cfg = {
        "strategy": "AIGridStrategy",
        "dry_run": True,
        "trading_mode": "futures",
        "stake_amount": 40,
        "max_open_trades": 3,
        "exchange": {"pair_whitelist": []},
    }
    original = preflight.config_hash(cfg)
    assert original == preflight.config_hash(dict(reversed(list(cfg.items()))))
    for key in cfg:
        changed = dict(cfg)
        changed[key] = {"pair_whitelist": ["ETH"]} if key == "exchange" else "changed"
        assert preflight.config_hash(changed) != original
    cfg["order_time_in_force"] = {"entry": "GTC"}
    assert preflight.config_hash(cfg) == original


def test_secret_created_once(tmp_path):
    path = tmp_path / ".attest_secret"
    secret = preflight.get_attestation_secret(path)
    assert len(secret) == 32
    assert preflight.get_attestation_secret(path) == secret


def test_preflight_writes_attestation_hermetic(tmp_path, monkeypatch):
    cfg = {
        "strategy": "AIGridStrategy",
        "dry_run": True,
        "trading_mode": "futures",
        "stake_amount": "40",
        "max_open_trades": 3,
        "exchange": {"pair_whitelist": ["ETH"]},
    }
    calls = []

    def get(url, *args):
        calls.append(url)
        if url.endswith("show_config"):
            return cfg
        if url.endswith("health"):
            return {"last_process": "ok"}
        return {"whitelist": ["ETH"]}

    monkeypatch.setattr(preflight, "_get_json", get)
    path = tmp_path / "ft_integration.json"
    path.write_text(json.dumps({"enabled": False}))
    preflight.fetch_runtime_config("http://fake", integration_path=path)
    data = json.loads(path.read_text())
    assert data["enabled"] is False
    block = data["attestation"]
    assert set(block) == {"ts_ms", "nonce", "container_id", "config_hash", "sig"}
    assert len(block["nonce"]) == 8
    assert block["config_hash"] == preflight.config_hash(cfg)
    sig = block.pop("sig")
    secret = (tmp_path / ".attest_secret").read_bytes()
    assert hmac.compare_digest(
        sig,
        hmac.new(secret, preflight.canonical_json(block), hashlib.sha256).hexdigest(),
    )
    assert any(url.endswith("/health") for url in calls)


def test_live_never_enabled(proof):
    s, write, _ = proof
    s.config["trading_mode"] = "live"
    write()
    assert s._enabled() is False
    assert s._runtime_dryrun_checked() == (False, "RUNTIME_LIVE_MODE")


@pytest.mark.parametrize(
    "config,reason",
    [
        (None, "NO_CONFIG"),
        ({}, "RUNTIME_NOT_DRY"),
        ({"dry_run": 1}, "RUNTIME_NOT_DRY"),
        ({"dry_run": True, "runmode": "LIVE"}, "RUNTIME_LIVE_RUNMODE"),
    ],
)
def test_runtime_fails_closed(proof, config, reason):
    s, _, _ = proof
    s.config = config
    assert s._enabled() is False
    assert s._dryrun_reason == reason


def test_runtime_cache_invalidates_immediately(proof):
    s, write, _ = proof
    write()
    assert s._enabled() is True
    s.config["dry_run"] = 1
    assert s._enabled() is False
    assert s._dryrun_reason == "RUNTIME_NOT_DRY"
    s.config["dry_run"] = True
    s.config["runmode"] = "live"
    assert s._enabled() is False
    assert s._dryrun_reason == "RUNTIME_LIVE_RUNMODE"


def test_verified_evidence_is_revoked_on_change(proof):
    s, write, path = proof
    write()
    assert s._enabled() is True
    s.config["stake_amount"] += 1
    assert s._enabled() is False
    assert s._dryrun_reason == "ATTESTATION_CONFIG_MISMATCH"
    write()
    assert s._enabled() is True
    path.unlink()
    assert s._enabled() is False
    assert s._dryrun_reason == "ATTESTATION_MISSING"


def test_missing_secret_is_not_created_by_strategy(proof):
    s, write, path = proof
    write()
    secret_path = path.parent / ".attest_secret"
    secret_path.unlink()
    assert s._enabled() is False
    assert s._dryrun_reason == "ATTESTATION_NO_SECRET"
    assert not secret_path.exists()


def test_future_evidence_and_rate_limited_logging(proof, caplog):
    s, write, _ = proof
    write(ts_ms=int(time.time() * 1000) + 61_000)
    assert s._enabled() is False
    assert s._enabled() is False
    assert s._dryrun_reason == "ATTESTATION_STALE"
    assert caplog.text.count("DRYRUN_UNPROVEN") == 1


def test_failed_health_never_writes_evidence(tmp_path, monkeypatch):
    path = tmp_path / "ft_integration.json"

    def get(url, *args):
        if url.endswith("health"):
            raise OSError("health unavailable")
        return {"dry_run": True}

    monkeypatch.setattr(preflight, "_get_json", get)
    with pytest.raises(OSError, match="health unavailable"):
        preflight.fetch_runtime_config("http://fake", integration_path=path)
    assert not path.exists()
    assert not (tmp_path / ".attest_secret").exists()


def test_rest_numeric_serialization_matches_runtime(proof):
    s, _, _ = proof
    runtime = dict(s.config, stake_amount="40.0", max_open_trades="3")
    assert preflight.config_hash(runtime) == preflight.config_hash(s.config)
