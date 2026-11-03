"""Runtime config and production-signed evidence for strategy fixtures."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from aigrid import preflight
from aigrid.publish import RuntimeFacts

_SECRET = b"r" * 32


def make_config(**overrides):
    """Return a fresh, realistic dry-run futures runtime config."""
    config = {
        "strategy": "AIGridStrategy",
        "dry_run": True,
        "trading_mode": "futures",
        "runmode": "dry_run",
        "timeframe": "5m",
        "stake_amount": 40.0,
        "max_open_trades": 3,
        "exchange": {"pair_whitelist": ["ETH/USDT:USDT"]},
    }
    config.update(overrides)
    return config


def sign_attestation(config, *, ts_ms=None, secret=None):
    """Use the production signer, including its canonical hash and HMAC."""
    secret = _SECRET if secret is None else secret
    with TemporaryDirectory() as directory:
        path = Path(directory) / "ft_integration.json"
        (path.parent / ".attest_secret").write_bytes(secret)
        if ts_ms is None:
            return preflight.write_attestation(config, path)
        with patch.object(preflight.time, "time", return_value=ts_ms / 1000):
            return preflight.write_attestation(config, path)


def write_integration(tmp_path, config, **plan_fields):
    """Write signed integration evidence and its local verification secret."""
    path = Path(tmp_path) / "ft_integration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    secret = preflight.get_attestation_secret(path.parent / ".attest_secret")
    data = {"dry_run_verified": True, **plan_fields}
    data["attestation"] = sign_attestation(config, secret=secret)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def runtime_config(**over):
    """Return realistic /show_config evidence for publisher fixtures."""
    return make_config(**over)


def write_secret(tmp_path):
    """Create the verification secret beside the integration file."""
    return preflight.get_attestation_secret(Path(tmp_path) / ".attest_secret")


def facts(**over):
    """Return complete runtime evidence; explicit overrides can model failures."""
    args = {
        "equity_usdt": 1000,
        "open_trades": [],
        "dry_run": True,
        "runtime_config": runtime_config(),
    }
    args.update(over)
    return RuntimeFacts(**args)
