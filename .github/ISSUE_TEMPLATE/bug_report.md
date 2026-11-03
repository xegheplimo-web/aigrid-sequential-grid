---
name: Bug report
about: Something behaved differently from the documented safety guarantees
title: "[bug] "
labels: ["bug", "triage"]
assignees: []
---

## Which stage diverged?

<!-- plan generation / publishing / strategy consumption / order creation / fill / exit / reconciliation -->

## What happened

<!-- Observed behaviour, with the exact command and output. -->

## What you expected

<!-- And why: link the doc/section or test that defines the expected behaviour. -->

## Reproduction

```bash
# minimal steps
```

## Evidence

<!--
Paste the relevant counters/events. This project does not accept "it did not work"
without the stage and the counter that proves it. Redact credentials and ids.
-->

```
plan_id:
level / cycle:
events:
fees / pnl:
```

## Environment

- OS:
- Python:
- Engine version (`python -c "import aigrid; print(aigrid.__version__)"`):
- Execution backend (dry-run / backtest / other):
