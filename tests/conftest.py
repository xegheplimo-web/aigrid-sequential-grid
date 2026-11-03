"""Shared pytest configuration for the AIGrid test-suite.

The engine is pure Python; the default suite is hermetic (no network, no market
data) and runs in a few seconds. Two optional tiers exist:

1. **Live-data tier** — ``tests/integration`` talks to public Binance endpoints.
   Enable with ``AIGRID_LIVE_DATA=1`` (see the module docstring).

2. **Replay tier** — several modules replay *recorded* candles and exchange-info
   snapshots from the original research environment
   (``user_data/data/binance/futures`` and ``.orchestrator/fapi_exchangeinfo.json``).
   Those recordings are multi-gigabyte, are not published, and their assertions are
   pinned to that specific history, so the tier is skipped unless
   ``AIGRID_FULL_FIXTURES=1`` is set. ``scripts/fetch_fixtures.py`` downloads a
   compatible candle slice for local experimentation.

Skipping is decided per module from its source, so no test file needs editing when
the recordings are absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "user_data" / "data" / "binance" / "futures"

# Strings that mark a module as belonging to the replay tier.
REPLAY_MARKERS = (
    "user_data/data/binance/futures",
    ".orchestrator/fapi_exchangeinfo.json",
)


def _full_fixtures_enabled() -> bool:
    return os.environ.get("AIGRID_FULL_FIXTURES") == "1"


def _fixtures_present() -> bool:
    try:
        return FIXTURE_DIR.is_dir() and any(FIXTURE_DIR.iterdir())
    except OSError:
        return False


def _module_source(item: pytest.Item) -> str:
    module = getattr(item, "module", None)
    path = getattr(module, "__file__", None)
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _full_fixtures_enabled():
        return

    skip = pytest.mark.skip(
        reason=(
            "replay tier needs the original recorded fixtures "
            "(user_data/data/binance/futures + .orchestrator/fapi_exchangeinfo.json); "
            "run `python scripts/fetch_fixtures.py` and set AIGRID_FULL_FIXTURES=1 "
            "to enable it"
        )
    )
    cache: dict[str, bool] = {}
    for item in items:
        path = getattr(getattr(item, "module", None), "__file__", None) or ""
        needed = cache.get(path)
        if needed is None:
            source = _module_source(item)
            needed = any(marker in source for marker in REPLAY_MARKERS)
            cache[path] = needed
        if needed:
            item.add_marker(skip)
