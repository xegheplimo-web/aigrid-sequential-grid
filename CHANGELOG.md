# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

* `aigrid.preflight` CLI: read-only comparison of a running bot against a reviewed
  baseline, with three outcomes — `0` match, `1` drift, `3` unverifiable.
* Per-level cycle counter embedded in `client_order_id`
  (`<plan>:<kind>:<side>:<level>:<seq>`), so a re-armed level can never collide with
  its previous cycle; the counter is persisted and monotonic across restarts.
* Fee classes per exit path: maker for take-profit, taker for stop-loss/close-all.
* `custom_stoploss` now derives an absolute stop price through
  `stoploss_from_absolute`, with side/liquidation-distance validation and an
  explicit `SL_INVALID` rejection instead of clamping.
* Position reconciliation that treats an unreadable position state as *not flat*.
* AI pipeline deadline (`llm_total_deadline_seconds`, default 120 s), per-call
  timeout, bounded retries and a cost ledger; the fast layer never blocks on it.
* Deterministic plan builder and publish gate with a hard `dry_run` refusal.

### Changed

* Strategy `timeframe` is pinned to the operating timeframe, with a startup check
  that halts new entries when the effective config drifts.
* `max_entry_position_adjustment = -1` is only accepted when capital and risk caps
  bound the resulting exposure.

### Fixed

* The second grid cycle of a level was silently dropped as a duplicate order id.
* Emergency exits were priced at maker fee, understating cost by ~3 bp per exit.
* A strategy could lock itself out of a symbol by leasing it as an external owner.

## [0.1.0] - 2026-11-03

Initial public extraction of the engine, test suite and documentation.
Backtest, dry-run and shadow execution only; no live-trading code path.
