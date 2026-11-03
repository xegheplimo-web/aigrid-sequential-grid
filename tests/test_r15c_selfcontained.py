"""R15c: standalone strategy uses the same wire format as the repo signer."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from aigrid.preflight import canonical_json, config_hash, write_attestation
from tests.test_r10_strategy_gates import _install_ft_stubs

ROOT = Path(__file__).resolve().parents[1]
STRATEGY = ROOT / "user_data/strategies/AIGridStrategy.py"
KNOWN_HASH = "4145ef6aa472742cf92890d00b7225b63eaeac349c3668f5e5ba7a23fa014974"


def _config(**changes):
    return dict(
        strategy="AIGridStrategy",
        dry_run=True,
        trading_mode="futures",
        stake_amount=40,
        max_open_trades=3,
        exchange={"pair_whitelist": ["ETH"]},
        **changes,
    )


def _load_strategy():
    _install_ft_stubs()
    spec = importlib.util.spec_from_file_location("r15c_strategy", STRATEGY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fallback_matches_repo_implementation(monkeypatch):
    module = _load_strategy()
    assert module._canonical_json is canonical_json
    assert module._config_hash is config_hash
    configs = [_config(), _config(), _config()]
    configs[1].update(
        stake_amount="12.5",
        max_open_trades="8",
        exchange={"pair_whitelist": ["Z", "É", "BTC", "ETH", "SOL", "X", "Y", "A"]},
    )
    configs[2].update(stake_amount="unlimited", dry_run=False)
    configs[2].pop("exchange")
    configs[2]["pair_whitelist"] = ["BTC"]
    for cfg in configs:
        assert module._local_config_hash(cfg) == config_hash(cfg)
        assert module._local_canonical_json(cfg) == canonical_json(cfg)
        assert module._hash(cfg) == config_hash(cfg)
        assert module._canonical(cfg) == canonical_json(cfg)
    for canonical in (module._local_canonical_json, canonical_json):
        with pytest.raises(ValueError):
            canonical({"value": float("nan")})

    # A working repo implementation's errors must propagate through wrappers.
    def broken(value):
        raise RuntimeError("repo error")

    monkeypatch.setattr(module, "_config_hash", broken)
    monkeypatch.setattr(module, "_canonical_json", broken)
    for wrapper in (module._hash, module._canonical):
        with pytest.raises(RuntimeError, match="repo error"):
            wrapper(configs[0])


def test_strategy_imports_without_aigrid():
    script = """
import importlib.util
import json
import re
import sys
from tests.test_r10_strategy_gates import _install_ft_stubs
_install_ft_stubs()
for name in list(sys.modules):
    if name == 'aigrid' or name.startswith('aigrid.'):
        del sys.modules[name]
sys.modules['aigrid'] = None
spec = importlib.util.spec_from_file_location('standalone_strategy', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module._canonical_json is None
assert module._config_hash is None
digest = module._hash(json.loads(sys.argv[2]))
assert re.fullmatch('[0-9a-f]{64}', digest)
assert digest == sys.argv[3]
print(digest)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(STRATEGY), json.dumps(_config()), KNOWN_HASH],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == KNOWN_HASH


def test_attestation_still_verifies_with_fallback(tmp_path, monkeypatch):
    module = _load_strategy()
    monkeypatch.setattr(module, "_canonical_json", None)
    monkeypatch.setattr(module, "_config_hash", None)
    path = tmp_path / "ft_integration.json"
    monkeypatch.setattr(module, "INTEGRATION_FILE", path)
    config = _config()
    write_attestation(config, path)
    strategy = object.__new__(module.AIGridStrategy)
    strategy.config = config
    data = json.loads(path.read_text(encoding="utf-8"))
    assert strategy._attestation_checked(data) == (True, "")
    data["attestation"]["sig"] = "0" * 64
    assert strategy._attestation_checked(data) == (False, "ATTESTATION_BAD_SIG")
