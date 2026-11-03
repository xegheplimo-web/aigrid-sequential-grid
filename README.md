# AIGrid — Sequential Grid Engine

[![CI](https://github.com/xegheplimo-web/aigrid-sequential-grid/actions/workflows/ci.yml/badge.svg)](https://github.com/xegheplimo-web/aigrid-sequential-grid/actions/workflows/ci.yml)
[![CodeQL](https://github.com/xegheplimo-web/aigrid-sequential-grid/actions/workflows/codeql.yml/badge.svg)](https://github.com/xegheplimo-web/aigrid-sequential-grid/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A **deterministic, risk-first grid engine** for crypto perpetual futures, with an
optional AI planning layer that can never touch an order directly.

> **Tóm tắt (VI):** Engine lưới tuần tự cho futures, tất định và kiểm thử được.
> AI chỉ được **đề xuất kế hoạch** đã qua validate; Risk Engine có quyền phủ quyết;
> mọi thứ mặc định **TẮT** và chỉ chạy dry-run. Không có lời hứa lợi nhuận.

---

## Why this exists

Most "grid bot" projects fail in the same three places:

1. **They claim Binance-style grids but execute sequentially.** A real Binance
   futures grid keeps *many* resting orders at once and self-heals partial fills.
   This engine is honest about what it is: a **Sequential Grid** — one level is
   armed, filled, exited and re-armed in a cycle, with a durable per-level cycle
   counter so a re-armed level can never reuse an order id.
2. **Costs are ignored.** A grid whose spacing is smaller than round-trip fees
   plus slippage plus funding is a machine that converts capital into fees. The
   spacing floor here is computed, not configured by hope:
   `min_spacing = k × (2 × maker_fee + slippage_buffer + funding_buffer)`.
3. **The model is unaccountable.** Every state transition is an event, every
   plan has a TTL and a `plan_id`, and every exit path is priced with the fee
   class it actually uses (maker for take-profit, taker for stop-loss/close-all).

## Design

```
                 ┌──────────────────────────── FAST LAYER (deterministic, per candle) ─┐
  candles  ──────►  Regime Engine ─► Plan builder ─► Validator ─► Grid Engine ─► Risk Engine ─► Execution
  (5m, closed)      (range/trend)     (no LLM)        (schema +     (levels,        (veto,        (shadow /
                                                     exchange        seq, fees)     ownership)    dry-run)
                                                     rules)
                 └──────────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ validated AIGridPlan (TTL, immutable once adopted)
                 ┌──────────────────┴───────── SLOW LAYER (async, optional, never blocking) ─┐
                 │  AI agents ─► structured output ─► plan proposal ─► validator ─► risk engine │
                 └──────────────────────────────────────────────────────────────────────────────┘
```

* **The LLM never places orders.** It can only emit a candidate plan, which must
  pass schema validation, exchange-rule checks, capital caps and the risk veto.
* **The fast layer never waits for the AI.** AI runs in a worker with a hard
  deadline; a late or expired plan is rejected, not adopted.
* **Everything fails closed.** Missing config, stale integration file, unknown
  position state, unverifiable runtime value → no new entries.

## Install

```bash
git clone https://github.com/xegheplimo-web/aigrid-sequential-grid.git
cd aigrid-sequential-grid
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Requires Python 3.11+. Only `pandas` and `pydantic` are mandatory; the dashboard
(`[dashboard]` extra) and the Freqtrade strategy bridge (`[freqtrade]` extra) are
optional.

## Quick start

```bash
# Hermetic unit suite — no network, no market data, ~3s
pytest -q

# Replay tier: needs the original recordings (opt-in, see tests/conftest.py)
AIGRID_FULL_FIXTURES=1 pytest -q

# Live-data smoke test against public Binance futures endpoints
AIGRID_LIVE_DATA=1 pytest -q tests/integration

# Compare a running Freqtrade instance against a reviewed baseline (read-only)
export FT_API_URL=http://127.0.0.1:8087
export FT_API_USER=... FT_API_PASS=...
export AIGRID_APPROVED_CONFIG=configs/approved_baseline.example.json
aigrid-preflight            # exit 0 = match, 1 = drift, 3 = unverifiable
```

`scripts/fetch_fixtures.py` downloads a compatible candle slice from the public
`fapi.binance.com` klines endpoint (no API key). Note that Binance answers
**HTTP 451** to US-hosted IP ranges, so CI runners in the US cannot fetch it —
that job is marked best-effort and the hermetic suite remains the gate.

## Safety model

These are enforced in code and covered by negative tests — not conventions:

| Guard | Behaviour |
|---|---|
| `dry_run` gate | Publish refuses and writes **no** control file when the runtime is not dry-run |
| Spacing floor | Plan rejected when the grid cannot pay round-trip cost; never shrunk to force more trades |
| Plan TTL | Expired plans are rejected (`PLAN_STALE_REJECTED`); expiry never silently cancels live orders |
| Position reconciliation | "Unknown" position state is treated as *not flat*, not as free to enter |
| Stoploss contract | `custom_stoploss` is computed from `current_rate` through `stoploss_from_absolute`, so the protective stop stays at an absolute price across leverage |
| Ownership | One owner per symbol; the engine cannot open a position on a symbol another strategy owns |
| Risk veto | Independent risk engine can halt entries while leaving exits and stops alive |

Full detail: [`docs/SAFETY.md`](docs/SAFETY.md).

## Evidence, not adjectives

A grid cycle counts as *proven* only when a single `plan_id` can be traced end to
end:

```
PLAN_PUBLISHED → PLAN_CONSUMED → ENTRY_REQUESTED → FT_ORDER_CREATED
             → FT_ORDER_FILLED → TP_ORDER_FILLED → LEVEL_REARMED
```

Each step must carry the same `plan_id`, `level`, and cycle counter, and be
confirmed from a source independent of the one that produced it. Backtest profit
is not evidence of execution; unit tests are not evidence of a round trip.
See [`docs/EVIDENCE.md`](docs/EVIDENCE.md).

## Project status

* **Alpha.** Dry-run / backtest only. Live trading is not supported by design.
* No profitability claim is made or implied. Grid strategies lose money in
  trends; the regime filter reduces exposure but does not remove it.
* The AI planning layer is optional and disabled by default
  (`AIGridSettings.enabled = False`).

## Repository layout

```
aigrid/                 engine package (regime, plans, grid, risk, state, data)
aigrid/preflight.py     read-only runtime-vs-baseline comparison CLI
tests/                  hermetic unit suite + integration smoke test
user_data/strategies/   Freqtrade strategy bridge (optional)
scripts/                fixture fetcher
docs/                   architecture, safety model, evidence standard
```

## Contributing

PRs are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md). The bar is simple:
tests first, no safety-regression, no new dependency without a reason.

## License & disclaimer

MIT — see [`LICENSE`](LICENSE). This software is provided for research and
education. Trading derivatives involves substantial risk of loss. Nothing here
is financial advice, and no warranty of fitness for live trading is given.
