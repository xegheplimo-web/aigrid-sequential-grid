.PHONY: help install test cov lint fmt typecheck security build fixtures clean

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with dev extras + pre-commit hooks
	python -m pip install -e ".[dev]"
	pre-commit install --hook-type pre-commit --hook-type pre-push

test:  ## Hermetic test suite (no network, no fixtures)
	pytest -q

cov:  ## Test suite with coverage report
	pytest -q --cov=aigrid --cov-report=term-missing

lint:  ## Lint + format check
	ruff check .
	ruff format --check .

fmt:  ## Auto-fix lint and formatting
	ruff check --fix .
	ruff format .

typecheck:  ## Static type check (advisory)
	mypy aigrid

security:  ## Bandit + dependency audit
	bandit -r aigrid -c pyproject.toml
	pip-audit --desc

build:  ## Build sdist + wheel into dist/
	python -m build
	python -m twine check dist/*

fixtures:  ## Download a small slice of public Binance futures candles
	python scripts/fetch_fixtures.py --pairs 4 --candles 1500

clean:  ## Remove caches and build artefacts
	rm -rf build dist .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
