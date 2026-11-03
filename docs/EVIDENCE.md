# Evidence standard

A grid bot that "works" is a claim about seven joins between a plan and a fill.
This document defines what counts as proof, and what only looks like proof.

## The seven-stage ladder

A single cycle is proven when **one** `plan_id` can be followed through all seven
stages, with the same `level` and the same cycle counter:

| # | Stage | Emitted by | Proves |
|---|---|---|---|
| 1 | `PLAN_PUBLISHED` | publisher | a plan with a TTL was handed to the execution side |
| 2 | `PLAN_CONSUMED` | strategy | the execution side read and accepted *that* plan |
| 3 | `ENTRY_REQUESTED` | strategy | an entry decision was taken for a level |
| 4 | `FT_ORDER_CREATED` | execution bridge | an order object exists with a client order id |
| 5 | `FT_ORDER_FILLED` | execution bridge / venue | the order reached a filled state |
| 6 | `TP_ORDER_FILLED` | engine | the level closed through its take-profit path |
| 7 | `LEVEL_REARMED` | engine | the level returned to armed state with a **new** cycle counter |

If any stage is missing, the cycle is unproven. Partial fills do not release a
level; a retry reuses the original client order id; a new cycle must produce a new
id (this is why the id carries `:{seq}`).

## Source independence

Recording the same truth twice is not corroboration. The Freqtrade REST API and
its SQLite database typically read the same internal state, so agreement between
them proves consistency, not execution. Independent evidence means:

* the publisher's own append-only log (what *we* sent),
* the execution framework's order/trade records (what the framework did),
* the engine's event log (what the engine believed at each transition),

and, when it exists, a venue-side confirmation. In dry-run no venue confirmation
exists — say so explicitly instead of implying it.

## What is not evidence

* A passing unit test suite: it proves the code paths behave as specified, not
  that a plan ever reached an order.
* A profitable backtest: it proves the rules would have produced those trades on
  that history, with the modelled costs — nothing about live fills.
* "No errors in the log": absence of evidence.
* "The bot is running": liveness ≠ correctness.
* A strategy `enter_tag` alone: it links a trade to a plan, but not to an order.

## Reporting template

Any run report should answer, with numbers:

```
cycles_evaluated / candles_valid / candles_stale
regime: range / uptrend / downtrend / uncertain
plans: generated / validated / rejected(reason) / published / expired
signals: generated / vetoed(stage) / allowed
orders: created / filled / cancelled / rejected
cycles_completed (all 7 stages) / levels_rearmed
fees / funding / realized pnl / max drawdown
restarts / reconciliations / mismatches
blockers (why any stage produced zero)
```

If `cycles_completed == 0`, the report must name the stage that blocked and the
counter that proves it — not a hypothesis.
