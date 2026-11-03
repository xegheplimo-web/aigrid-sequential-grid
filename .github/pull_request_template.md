## What changed

<!-- One paragraph. Link the issue if there is one. -->

## Why

<!-- The problem this solves. If it is a bug: what was the root cause? -->

## Safety impact

- [ ] No change to sizing, leverage, stop distance, spacing, or exit pricing
- [ ] No new path that writes a control file or creates an order
- [ ] No fail-closed branch weakened or made advisory
- [ ] New/changed guards come with a test proving they **refuse**

## How it was verified

```bash
# exact commands + result lines
pytest -q
ruff check .
```

<!-- Paste the summary line(s), e.g. "268 passed in 3.1s". Do not paste a promise. -->

## Checklist

- [ ] Tests added/updated and failing before the fix
- [ ] `CHANGELOG.md` updated under `Unreleased`
- [ ] No credentials, ids, or generated data committed
- [ ] Public API changes documented in the docstring/README
