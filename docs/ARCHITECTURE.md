# Architecture

## Layers

```
aigrid/
├── settings.py          typed configuration, loaded from TOML, all flags default off
├── selection.py         deterministic pair scan (no LLM) -> watchlist
├── regime/              regime classification + controller (range / trend / high-risk)
├── plans/               plan schema, validator, (deterministic builder)
├── capital.py           capital allocation: margin vs notional, per-level caps
├── grid/                Sequential Grid engine (levels, seq, fees) + executors
├── risk/                independent risk engine + symbol ownership registry
├── state/               persistence, recovery, reconciliation
├── data/                Binance USDT-M adapters (files + public REST), indicators
├── ta_adapter.py        AI agent pipeline: deadline, retries, cost ledger
├── orchestrator.py      candle -> plan -> grid -> execution, plus async AI dispatch
├── service.py           long-running loop: heartbeat, publish cycle, shutdown
├── preflight.py         read-only runtime-vs-baseline comparison CLI
└── backtest.py          replay harness (config C = deterministic, no LLM)
```

## Two-speed design

**Fast layer (per closed candle, deterministic).** Regime → plan build → validate →
grid engine → risk veto → execution. No I/O beyond the local state and the
execution bridge; no LLM calls; bounded runtime. This is the layer that must keep
working when everything else is degraded.

**Slow layer (asynchronous, optional).** AI agents propose plans. They run in a
worker thread with a per-call timeout, a total deadline, bounded retries and a TTL
on the result. The fast layer never waits for them: a late proposal is rejected,
not adopted. Timeout or failure sets `on_ai_unavailable`, which blocks *new*
entries while keeping stops and exits alive.

## Grid model: sequential, not simultaneous

A level is a price band with a state machine:

```
ARMED ──cross──► ENTRY_PENDING ──fill──► ENTRY_FILLED ──TP──► EXIT_PENDING
   ▲                    │                      │                    │
   │                 cancel                    │                 fill
   └──────── RE_ARMED ──┴────── LEVEL_CLOSED ◄──┴────────────────────┘
```

* One cycle per level at a time; the engine is a state machine over closed candles,
  not a book of resting orders. It therefore cannot claim parity with venue-native
  simultaneous grids, and does not.
* `client_order_id = <plan_id>:<kind>:<side>:<level>:<seq>` — `seq` increments on
  each re-arm and is persisted, so cycle N+1 never reuses cycle N's id. Retries
  within a cycle reuse the id (idempotency).
* Partial fills do not release a level.
* Fees are per exit class: maker for take-profit, taker for stop-loss/close-all.

## Capital model

Sizing is expressed in **margin**, and converted to notional with leverage:
`notional = margin × leverage`. Mixing the two units is the classic grid-sizing bug
(it silently multiplies exposure by the leverage factor), so the schema carries both
fields explicitly and the capital manager clamps:

```
per_level_margin × levels ≤ max_capital
per_level_notional ≤ max_position_size
Σ open + reserved margin ≤ total_margin_budget     # portfolio-level, not per plan
```

`max_entry_position_adjustment = -1` (unlimited DCA allowed by the framework) is only
acceptable because the two caps above bound the exposure; it is never unbounded in
money terms.

## Risk and ownership

* The risk engine is independent of the strategy: it can halt entries, force flat, or
  reject a plan, and it does so regardless of what the AI or the grid wants.
* Ownership is a per-symbol lease (`FREE`, `GRID_OWNED`, `TREND_OWNED`, `RISK_LOCKED`).
  A symbol leased to another strategy cannot be entered — this is what prevents the
  engine from stacking a grid on top of an existing directional position, and what
  prevents the strategy from locking itself out by misclassifying its own trades.

## State, recovery, reconciliation

* Every transition appends an event; state is snapshotted atomically
  (`tmp` + `fsync` + `os.replace`).
* On restart, lots and filled levels are restored **before** the engine is built;
  a mismatch between the restored book and the actual position results in
  `SAFE_HOLD` rather than silent correction.
* External positions are adopted as `TREND_OWNED` leases so the grid cannot fight
  them.
* Unknown position state is treated as *not flat*: publishing stops.

## Why determinism first

A deterministic fast layer is the only way to answer "did the plan reach an order?"
without arguing about model temperature. The AI layer is an optimisation on top: it
can be disabled entirely, and the system remains a complete, testable grid engine.
When it is enabled, its output is just another untrusted plan candidate.
