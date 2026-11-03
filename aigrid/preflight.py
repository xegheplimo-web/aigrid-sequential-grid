"""Preflight config-drift and safety checks (T-205 / T-205b / T-205c).

The check functions are pure — the caller passes dicts. They exist
because of a real incident: the bot ran 15m/3-pair while the approved
config on disk said 5m/8-pair and nothing warned. Comparing the
*runtime* config against the *approved* one makes that drift loud.

Three result groups, never merged (card-R5 / D-092):
- ``drift``        — both sides report a value and they differ -> exit 1
- ``unverifiable`` — approved does not declare the key -> exit 3
- ``match``        — values verify -> exit 0

Required safety keys (REQUIRED_SAFETY_KEYS) must be declared in the
approved baseline; a missing one is always UNVERIFIABLE, never a match —
"publish" callers must treat exit 1 and 3 as blocking. Numeric keys are
normalized through float() when both sides coerce ("40" == 40 — the FT
REST API serializes stake_amount as a string), otherwise compared raw.
pair_whitelist resolves ``exchange.pair_whitelist`` as fallback and
compares as a set.

CLI (`python -m aigrid.preflight`): reads the live runtime config from
the Freqtrade REST API (show_config + whitelist) and the approved config
from a JSON file, prints both groups, exits 1 on drift / 3 on
unverifiable / 0 on match / 2 when either side is unreadable. urllib
stdlib only; Basic-auth credentials come from env vars FT_API_USER /
FT_API_PASS — never hardcoded.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import secrets
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

FT_API_URL_ENV = "FT_API_URL"
FT_API_USER_ENV = "FT_API_USER"
FT_API_PASS_ENV = "FT_API_PASS"  # nosec B105 -- environment variable name only
APPROVED_CONFIG_ENV = "AIGRID_APPROVED_CONFIG"
DEFAULT_API_URL = "http://127.0.0.1:8080"
DEFAULT_TIMEOUT_S = 10.0
DEFAULT_INTEGRATION_PATH = Path("user_data/aigrid_state/ft_integration.json")


def canonical_json(value: dict) -> bytes:
    """One UTF-8 wire representation for both signer and verifier."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def config_hash(config: dict) -> str:
    """Hash only the six REST-visible fields (D-101).

    REST serializes stake_amount as a string; normalize numeric fields on
    both sides so the same effective runtime has the same fingerprint.
    Missing fields are rejected rather than signed as evidence.
    """
    fields = {
        key: config[key]
        for key in (
            "strategy",
            "dry_run",
            "trading_mode",
            "stake_amount",
            "max_open_trades",
        )
    }
    for key in ("stake_amount", "max_open_trades"):
        try:
            fields[key] = float(fields[key])
        except (ValueError, TypeError):
            pass
    exchange = config.get("exchange")
    pairs = exchange.get("pair_whitelist") if isinstance(exchange, dict) else None
    if pairs is None:
        pairs = config["pair_whitelist"]
    if not isinstance(pairs, list) or not all(isinstance(p, str) for p in pairs):
        raise ValueError("invalid runtime whitelist")
    fields["exchange"] = {"pair_whitelist": sorted(pairs)}
    return hashlib.sha256(canonical_json(fields)).hexdigest()


def get_attestation_secret(path: Path) -> bytes:
    """Create once with exclusive open, then read; never replace a bad key."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, "wb") as stream:
            stream.write(secrets.token_bytes(32))
    secret = path.read_bytes()
    if len(secret) != 32:
        raise ValueError("invalid attestation secret length")
    return secret


def write_attestation(runtime: dict, integration_path: Path) -> dict:
    """Add evidence without ever granting permission to trade.

    Caller must first obtain successful REST show_config and health replies.
    An atomic replace prevents readers from observing a partial signature.
    """
    block = {
        "ts_ms": int(time.time() * 1000),
        "nonce": secrets.token_hex(4),
        "container_id": os.environ.get("HOSTNAME", ""),
        "config_hash": config_hash(runtime),
    }
    secret = get_attestation_secret(integration_path.parent / ".attest_secret")
    block["sig"] = hmac.new(secret, canonical_json(block), hashlib.sha256).hexdigest()
    try:
        data = json.loads(integration_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        data = {"enabled": False}
    if not isinstance(data, dict):
        raise TypeError("integration must be an object")
    data["attestation"] = block
    fd, name = tempfile.mkstemp(dir=integration_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, allow_nan=False)
        os.replace(name, integration_path)
    finally:
        if Path(name).exists():
            Path(name).unlink()
    return block


_DRIFT_KEYS = (
    "timeframe",
    "strategy",
    "dry_run",
    "trading_mode",
    "max_open_trades",
    "stake_amount",
    "pair_whitelist",
    "max_entry_position_adjustment",
)

# Keys that decide whether the runtime is safe. The approved baseline MUST
# declare every one of them; absent in approved -> UNVERIFIABLE, never
# silently a match (D-092: publish is blocked until the baseline is
# supplemented).
REQUIRED_SAFETY_KEYS = (
    "dry_run",
    "trading_mode",
    "timeframe",
    "pair_whitelist",
    "stake_amount",
    "max_open_trades",
    "max_entry_position_adjustment",
    "leverage",
    "stoploss",
)

_CHECK_KEYS = _DRIFT_KEYS + tuple(k for k in REQUIRED_SAFETY_KEYS if k not in _DRIFT_KEYS)

# Keys the runtime source (FT show_config) cannot expose at all. A missing
# runtime value for these is NOT drift — it is unverifiable through this
# source (T-205e). ``leverage`` is per-trade: with no strategy leverage()
# callback FT always runs 1x, so an approved 1.0 verifies "by policy"
# while a declared !=1.0 provably cannot be live.
_RUNTIME_UNEXPOSED_KEYS = frozenset({"leverage"})

_LEVERAGE_POLICY_NOTE = "1x by policy; verify trade.leverage == 1.0 at first fill"
_LEVERAGE_UNEXPOSED_NOTE = "not exposed by FT show_config"

_MISSING = object()


@dataclass
class ConfigCheck:
    """Three-way compare result. ``ok`` = every checked key verified."""

    drift: list[str] = field(default_factory=list)
    unverifiable: list[str] = field(default_factory=list)
    match: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.drift and not self.unverifiable

    @property
    def blocked(self) -> bool:
        """Publish-blocked: drift (exit 1) OR unverifiable (exit 3)."""
        return not self.ok

    @property
    def exit_code(self) -> int:
        if self.drift:
            return 1
        if self.unverifiable:
            return 3
        return 0


def compare_configs(
    runtime: dict, approved: dict, *, has_leverage_callback: bool = False
) -> ConfigCheck:
    """3-way diff of runtime vs approved config (T-205c / T-205e).

    approved missing a checked key -> ``unverifiable`` (not drift): for
    required safety keys this holds even when runtime is silent too — the
    baseline must declare them. Runtime missing a key the approved side
    declares -> ``drift`` (fail-closed: a declared value not running),
    EXCEPT keys the source cannot expose at all (_RUNTIME_UNEXPOSED_KEYS,
    e.g. ``leverage`` — absent from FT show_config): those classify per
    ``_classify_unexposed``. ``has_leverage_callback`` tells whether the
    deployed strategy defines ``leverage()`` — with the callback any
    leverage may be live (unverifiable until first fill); without it FT
    runs exactly 1x, so approved 1.0 matches by policy and anything else
    is an honest drift.
    """
    check = ConfigCheck()
    for key in _CHECK_KEYS:
        rv, av = _value_of(runtime, key), _value_of(approved, key)
        if av is _MISSING:
            if rv is not _MISSING or key in REQUIRED_SAFETY_KEYS:
                check.unverifiable.append(f"{key}: runtime={_fmt(rv)} approved={_fmt(av)}")
        elif rv is _MISSING:
            if key in _RUNTIME_UNEXPOSED_KEYS:
                _classify_unexposed(check, key, av, has_leverage_callback)
            else:
                check.drift.append(f"{key}: runtime={_fmt(rv)} approved={_fmt(av)}")
        elif _values_equal(key, rv, av):
            check.match.append(key)
        else:
            check.drift.append(f"{key}: runtime={_fmt(rv)} approved={_fmt(av)}")
    return check


def _classify_unexposed(check: ConfigCheck, key: str, av, has_leverage_callback: bool) -> None:
    """Runtime cannot expose ``key`` — classify honestly (T-205e)."""
    if key == "leverage" and has_leverage_callback:
        check.unverifiable.append(
            f"leverage: runtime=<absent> approved={_fmt(av)} "
            f"({_LEVERAGE_UNEXPOSED_NOTE}; strategy defines leverage() — "
            f"verify trade.leverage at first fill)"
        )
        return
    if key == "leverage":
        try:
            declared = float(av)
        except (TypeError, ValueError):
            declared = None
        if declared == 1.0:
            check.match.append(f"leverage: match ({_LEVERAGE_POLICY_NOTE})")
        else:
            check.drift.append(
                f"leverage: runtime=<absent> approved={_fmt(av)} "
                f"({_LEVERAGE_UNEXPOSED_NOTE}; no leverage() callback — FT "
                f"runs 1x, so the declared value cannot be live)"
            )
        return
    check.unverifiable.append(
        f"{key}: runtime=<absent> approved={_fmt(av)} ({_LEVERAGE_UNEXPOSED_NOTE})"
    )


def check_config_drift(runtime: dict, approved: dict) -> list[str]:
    """Compat wrapper: every way the configs do NOT verify (drift +
    unverifiable). Empty list = fully verified match. Legacy callers stay
    fail-closed: an unverifiable config never looks like a match."""
    check = compare_configs(runtime, approved)
    return check.drift + check.unverifiable


def check_telegram_and_safety(runtime: dict) -> list[str]:
    """Warnings for dangerous runtime settings (advisory, not vetoes)."""
    warnings: list[str] = []
    if runtime.get("dry_run") is False:
        warnings.append("dry_run=False: real-money trading — confirm intended")
    stoploss = runtime.get("stoploss")
    if isinstance(stoploss, (int, float)) and stoploss <= -0.99:
        warnings.append(f"stoploss={stoploss}: static stop loss effectively disabled")
    if runtime.get("max_entry_position_adjustment") == -1:
        warnings.append("max_entry_position_adjustment=-1: unlimited position adjustments")
    return warnings


def _value_of(cfg: dict, key: str):
    """Top-level lookup with the exchange.pair_whitelist fallback — FT
    configs keep the whitelist nested under the exchange section."""
    v = cfg.get(key, _MISSING)
    if v is _MISSING and key == "pair_whitelist":
        exchange = cfg.get("exchange")
        if isinstance(exchange, dict):
            v = exchange.get("pair_whitelist", _MISSING)
    return v


def _values_equal(key: str, rv, av) -> bool:
    if key == "pair_whitelist":
        try:
            return set(map(str, rv)) == set(map(str, av))
        except TypeError:
            return rv == av
    # numeric normalization: FT serializes numbers as strings ("40"),
    # and int/float mixes are common (3.0 vs 3). float() both sides when
    # possible; non-numeric values ("unlimited") fall back to equality.
    try:
        return float(rv) == float(av)
    except (TypeError, ValueError):
        return rv == av


def _fmt(value) -> str:
    if value is _MISSING:
        return "<absent>"
    if isinstance(value, (set, frozenset)):
        return str(sorted(value))
    return str(value)


# ------------------------------------------------------------------ CLI
def _get_json(url: str, username: str | None, password: str | None, timeout: float) -> dict:
    if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
        raise ValueError(f"refusing non-HTTP URL: {url!r}")
    req = urllib.request.Request(url)
    if username:
        token = base64.b64encode(f"{username}:{password or ''}".encode()).decode()
        req.add_header("Authorization", f"Basic {token}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310 -- HTTP(S) checked above
        return json.loads(resp.read().decode("utf-8"))


def fetch_runtime_config(
    base_url: str,
    username: str | None = None,
    password: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    *,
    integration_path: Path | None = None,
) -> dict:
    """Read the live runtime config from the Freqtrade REST API.

    Merges /api/v1/show_config with the pair whitelist from
    /api/v1/whitelist so check_config_drift sees every _DRIFT_KEYS field.
    """
    base = base_url.rstrip("/")
    cfg = _get_json(f"{base}/api/v1/show_config", username, password, timeout)
    health = _get_json(f"{base}/api/v1/health", username, password, timeout)
    if not isinstance(cfg, dict) or not cfg or not isinstance(health, dict) or not health:
        raise ValueError("invalid REST config/health response")
    wl = _get_json(f"{base}/api/v1/whitelist", username, password, timeout)
    runtime = dict(cfg)
    whitelist = wl.get("whitelist", wl) if isinstance(wl, dict) else wl
    runtime["pair_whitelist"] = list(whitelist or [])
    path = (
        integration_path
        if integration_path is not None
        else Path(os.environ.get("AIGRID_INTEGRATION_FILE", str(DEFAULT_INTEGRATION_PATH)))
    )
    write_attestation(runtime, path)
    return runtime


def main(argv: list[str] | None = None) -> int:
    """Exit 0 = match; 1 = drift; 2 = unreadable input; 3 = unverifiable."""
    parser = argparse.ArgumentParser(
        prog="aigrid.preflight",
        description="Compare the live Freqtrade runtime config against the approved config JSON.",
    )
    parser.add_argument(
        "--approved",
        default=os.environ.get(APPROVED_CONFIG_ENV),
        help=f"approved config JSON path (or env {APPROVED_CONFIG_ENV})",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get(FT_API_URL_ENV, DEFAULT_API_URL),
        help=f"Freqtrade REST base URL (env {FT_API_URL_ENV}, default {DEFAULT_API_URL})",
    )
    args = parser.parse_args(argv)
    if not args.approved:
        print(
            f"preflight: no approved config given (--approved or env {APPROVED_CONFIG_ENV})",
            file=sys.stderr,
        )
        return 2
    try:
        runtime = fetch_runtime_config(
            args.api_url,
            username=os.environ.get(FT_API_USER_ENV),
            password=os.environ.get(FT_API_PASS_ENV),
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"preflight: cannot read runtime config: {exc}", file=sys.stderr)
        return 2
    try:
        approved = json.loads(Path(args.approved).read_text(encoding="utf-8"))
        if not isinstance(approved, dict):
            raise TypeError("approved config is not a JSON object")
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        print(f"preflight: cannot read approved config: {exc}", file=sys.stderr)
        return 2
    for warning in check_telegram_and_safety(runtime):
        print(f"WARN {warning}", file=sys.stderr)
    check = compare_configs(runtime, approved)
    if check.drift:
        print("CONFIG DRIFT — runtime differs from approved:")
        for d in check.drift:
            print(f"  {d}")
    if check.unverifiable:
        print("UNVERIFIABLE — approved config does not declare these keys (baseline incomplete):")
        for u in check.unverifiable:
            print(f"  {u}")
    if check.ok:
        print("preflight OK — runtime config matches approved config")
        for m in check.match:
            if ":" in m:  # annotated matches (e.g. leverage policy note)
                print(f"  {m}")
    return check.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
