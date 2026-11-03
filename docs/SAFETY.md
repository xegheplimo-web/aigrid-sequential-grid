# Safety model

Every rule below is enforced in code and has at least one **negative test** that
proves it refuses when it should. If a rule exists only in documentation, it does
not exist.

## 1. Defaults are off

`AIGridSettings.enabled` defaults to `False`. With the flag off the orchestrator
processes candles, records events and manages already-open positions, but it
never adopts a new plan and never publishes an integration payload.

## 2. The AI cannot trade

The AI layer returns a `AIGridPlan` candidate. Before adoption it must pass:

1. schema validation (`aigrid/plans/schema.py`),
2. content + exchange-rule validation (`aigrid/plans/validator.py`),
3. capital caps (`aigrid/capital.py`),
4. the independent risk veto (`aigrid/risk/engine.py`).

Any failure → the plan is discarded and the reason is recorded. The LLM has no
handle to an order object, an exchange client, or the grid engine.

## 3. Dry-run is a hard gate

The publish path checks `runtime.dry_run is True` **before** writing any control
file. When that check fails:

* the attempt is logged as `LIVE_REFUSED`,
* **no** control file is written — not even a disable marker,
* the loop keeps running so open positions stay protected.

This is deliberately asymmetric: a control file is the only thing the strategy
listens to, so "write nothing" is stronger than "write disabled".

## 4. Spacing must pay for itself

```
min_spacing = k × (2 × maker_fee + slippage_buffer + funding_buffer)
```

with `k` in the 2–3 band (default 2.5). If the observed range cannot host the
configured number of levels at that spacing:

1. reduce the level count,
2. re-check range fit and inventory risk,
3. still not viable → **reject the pair**.

Shrinking the spacing to generate more trades is explicitly forbidden — it is
the single most common way grid bots lose money while looking busy.

## 5. Stoploss is an absolute price, not a ratio of the entry

Freqtrade's `custom_stoploss()` returns a *risk distance measured from
`current_rate`*, and the exchange-facing stop price is derived as
`current × (1 − |ratio| / leverage)` for longs. Computing the ratio from the
entry price therefore:

* tightens the stop by the leverage factor (2× → exits at −5% instead of −10%),
* and lets the stop drift upward as price rises at 1× (an unintended trailing stop).

This project converts an absolute `stop_rate` through `stoploss_from_absolute`,
validates that the stop sits on the correct side of the current price and far
enough from liquidation, and rejects the plan otherwise (`SL_INVALID`) instead of
clamping it into something that looks valid.

## 6. Unknown position state is not flat

Before publishing, the engine reconciles its lot book against the exchange/API
view. If the position state cannot be read, the result is
`POSITION_STATE_UNKNOWN` and publishing stops. Treating silence as "no position"
is how bots double their exposure after a restart.

## 7. One owner per symbol

The ownership registry grants a lease per symbol to exactly one owner (this grid,
an external strategy, or the risk engine). A symbol leased by another strategy
cannot be entered, which prevents two systems from stacking positions on the same
market.

## 8. Stale plans cannot execute

* Plans carry `created_at` / `expires_at`; expired plans are rejected.
* The integration file is validated by age; a stale file disables the strategy
  side instead of being consumed.
* Plan expiry does **not** cancel live orders — cancelling is a separate,
  explicit action that first confirms order state.

## 9. Costs use the fee class actually paid

Take-profit exits are priced at maker fee, stop-loss and close-all exits at taker
fee. Pricing every exit as maker understates losses by ~3 bp per emergency exit,
which is enough to turn a nominally profitable grid into a losing one.

## 10. Reporting rules

* A backtest is not a live result.
* A unit test is not a round trip.
* A dry-run fill is simulated by the bot framework, **not** an exchange
  confirmation; it is never described as exchange-verified.
* "No trades" is not proof of a working filter — it requires rejection telemetry
  to explain *which* stage blocked.
