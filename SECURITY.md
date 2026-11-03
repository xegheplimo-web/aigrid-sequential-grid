# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | ✅ |

## Reporting a vulnerability

Please **do not** open a public issue for a security problem. Use GitHub's
[private vulnerability reporting](https://github.com/xegheplimo-web/aigrid-sequential-grid/security/advisories/new)
or email the maintainer listed in `pyproject.toml`.

Include: affected version, a minimal reproduction, and the impact you believe it
has (e.g. "bypasses the dry-run publish gate", "allows an order to be sized above
the capital cap"). We aim to acknowledge within 72 hours.

## Threat model

The engine is designed to run **locally**, next to a trading bot, with no inbound
network surface of its own:

* **No credentials are stored by this project.** The preflight CLI reads API
  credentials from environment variables (`FT_API_USER`, `FT_API_PASS`) and uses
  read-only endpoints.
* **No order-placing code path exists in this repository.** Execution is delegated
  to the host framework; the engine emits plans and commands, not orders.
* **The AI layer is not a trust boundary.** Model output is untrusted input:
  it is schema-validated, range-checked, capital-capped and risk-vetoed before it
  can influence anything.

### In scope

* A model response (or a crafted plan file) that can cause an oversized position,
  a disabled stop, or entries on a symbol owned by another strategy.
* Bypass of the `dry_run` publish gate or of the spacing floor.
* Path traversal or unsafe deserialisation in plan/state loading.
* Supply-chain risks in declared dependencies.

### Out of scope

* Losses caused by market movement while running the engine as documented.
* Attacks that require write access to the local filesystem or the environment —
  an attacker with that access already controls the bot.
* The behaviour of third-party trading frameworks themselves.

## Hardening already in place

* Atomic writes (`tmp` + `fsync` + `os.replace`) for every control file.
* Fail-closed parsing: unreadable/corrupt/stale state disables action instead of
  guessing.
* No `eval`, no `pickle`, no dynamic imports of user-supplied names.
* Secrets are never read from, or written to, files owned by this project.

## Dependency hygiene

`pip-audit` runs in CI on every push, and Dependabot opens grouped update PRs
weekly. A CVE in a direct dependency is treated as a release blocker.
