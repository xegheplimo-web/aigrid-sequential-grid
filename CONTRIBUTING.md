# Contributing

Thanks for taking a look. This project optimises for **verifiable behaviour over
features**, so the contribution rules are stricter than usual in one specific
area: anything that touches risk, sizing, publishing, or exits.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install --hook-type pre-commit --hook-type pre-push
pytest -q
```

The hermetic suite must pass in ~3 seconds without network access. If your change
needs market data, use `scripts/fetch_fixtures.py` and keep the test skip-gated so
CI without fixtures stays green.

## The bar for a change

1. **Tests first.** New behaviour arrives with a test that fails before the
   change. Bug fixes arrive with a regression test that reproduces the bug.
2. **Negative tests for guards.** Any new safety guard needs a test proving it
   *refuses*. A guard without a refusal test is a comment.
3. **No safety regressions.** Loosening a limit, removing a fail-closed branch, or
   making a check "advisory" is a breaking change and needs explicit discussion in
   an issue first.
4. **No silent dependency growth.** Explain why an existing dependency cannot do
   the job.
5. **Numbers, not adjectives.** In PR descriptions, replace "improved" with the
   measurement, and "should work" with the command you ran.

## Style

* `ruff check` and `ruff format --check` must pass (line length 100).
* Type annotations on public functions; `mypy aigrid` is advisory today.
* Commits: imperative subject, body explains *why*. Keep unrelated reformatting
  out of behavioural PRs.
* Public API changes need a `CHANGELOG.md` entry under `## Unreleased`.

## What will be rejected

* Anything that removes or weakens a fail-closed path.
* Grid spacing, leverage, or stop-distance changes justified by "more trades".
* Profitability claims without a run report following `docs/EVIDENCE.md`.
* Large generated artefacts (recorded candles, data dumps).

## Reporting a bug

Use the bug template. The single most useful thing you can include is the
**stage** where behaviour diverged from expectation: plan generation, publishing,
strategy consumption, order creation, fill, exit, or reconciliation.
