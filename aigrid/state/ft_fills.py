"""Freqtrade order-state fills reader — the ONLY source of "filled" (R12).

The virtual grid book (``engine.state.lots``) is a candle-driven simulation,
not execution. This module reads what Freqtrade actually did:

    GET /api/v1/status  ->  list[OpenTradeSchema]  ->  trade.orders

An order counts as a grid fill when it is an entry order
(``ft_is_entry``), has a status in ``FILL_STATUSES`` and ``filled > 0``, and
its tag identifies the plan: exact ``aigrid_{plan_id}``, that prefix plus
``:``, or ``grid_level_<digits>`` within a trade whose ``enter_tag`` is
exactly ``aigrid_{plan_id}``. Each qualifying order maps to a grid level by PRICE
within ``price_tick/2`` — the price is the level's identity (P0-5b). Orders
that match the filters but map to no level are counted in
``unmapped_orders``, never silently included.

Read-only: the reader issues GETs only, never a write call. Basic-auth
credentials come from the same env vars preflight uses (FT_API_URL /
FT_API_USER / FT_API_PASS). Any failure — REST down, unparseable response —
raises; the caller treats that as UNVERIFIED and omits the
``filled_level_prices`` key entirely (the strategy fails closed on absent).
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..preflight import (
    DEFAULT_API_URL,
    FT_API_PASS_ENV,
    FT_API_URL_ENV,
    FT_API_USER_ENV,
)

STATUS_PATH = "/api/v1/status"
DEFAULT_TIMEOUT_S = 5.0
_TIGHT_TOLERANCE = 1e-9  # no tick -> exact-match against the plan's prices

# The Freqtrade API is a localhost service — always connect directly. The
# default urlopen consults OS proxy auto-detect (WPAD), which can stall for
# seconds per call before the connection is even attempted.
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class FtFillsError(Exception):
    """The /status response was not the expected list of open trades."""


# P0-5c (R14 addendum v2): confirmed entry fills count across statuses —
# ``closed`` plus ``canceled``/``expired`` that still carry ``filled>0``
# (the exchange killed the order; the partial fill remains real exposure)
# and ``open`` orders mid partial-fill. Any other status never counts.
FILL_STATUSES = frozenset({"open", "closed", "canceled", "expired"})


@dataclass(frozen=True)
class FillsResult:
    """Per-pair mapping result: verified level prices + unmapped count."""

    prices: list[float] = field(default_factory=list)  # sorted ascending
    unmapped_orders: int = 0
    # order_id -> cumulative filled qty applied for this pair's plan tag
    order_qty: dict[str, float] = field(default_factory=dict)
    # qty resting on orders still ``open`` (live partial fills)
    partial_qty: float = 0.0


class OrderFillLedger:
    """Cumulative per-order fill quantities — idempotent across REST polls.

    Freqtrade reports ``order.filled`` as a running cumulative value. The
    ledger stores the last applied value per ``order_id`` and ``apply``
    returns only the positive increment, so replaying the same /status
    payload applies zero — repeated polls can never double-count.
    """

    def __init__(self) -> None:
        self._applied: dict[str, float] = {}

    def apply(self, order_id: str, filled: float) -> float:
        """Record a cumulative observation; return the delta to book (>=0)."""
        prev = self._applied.get(order_id, 0.0)
        current = max(prev, float(filled))
        self._applied[order_id] = current
        return current - prev

    def cumulative(self, order_id: str) -> float:
        return self._applied.get(order_id, 0.0)


class FtFillsReader:
    """Read-only client for the Freqtrade REST API (GET /api/v1/status)."""

    def __init__(
        self,
        base_url: str,
        *,
        username: str | None = None,
        password: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        opener=None,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self._username = username
        self._password = password
        self.timeout_s = float(timeout_s)
        self._opener = opener if opener is not None else _NO_PROXY_OPENER

    @classmethod
    def from_env(cls, timeout_s: float = DEFAULT_TIMEOUT_S) -> FtFillsReader:
        """Same credentials source as aigrid.preflight (env, never config)."""
        return cls(
            os.environ.get(FT_API_URL_ENV, DEFAULT_API_URL),
            username=os.environ.get(FT_API_USER_ENV),
            password=os.environ.get(FT_API_PASS_ENV),
            timeout_s=timeout_s,
        )

    def fetch_open_trades(self) -> list[dict]:
        """GET /api/v1/status -> list of open-trade dicts.

        Raises on any transport/parse failure and when the decoded payload
        is not a list — callers must treat that as UNVERIFIED.
        """
        req = urllib.request.Request(f"{self.base_url}{STATUS_PATH}")
        if self._username:
            token = base64.b64encode(f"{self._username}:{self._password or ''}".encode()).decode()
            req.add_header("Authorization", f"Basic {token}")
        with self._opener.open(req, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if not isinstance(data, list):
            raise FtFillsError("/api/v1/status did not return a list")
        return data


def filled_prices_for_pair(
    trades: Sequence[dict] | None,
    *,
    pair: str,
    plan_id: str,
    level_prices: Sequence[float],
    price_tick: float | None = None,
    ledger: OrderFillLedger | None = None,
) -> FillsResult:
    """Map filled FT entry orders to plan level prices (pure, no I/O).

    ``price_tick`` comes from the fapi_exchangeinfo snapshot; when absent
    the tolerance tightens to 1e-9 so only an (effectively) exact match
    against ``level_prices`` counts. Returns prices sorted ascending.

    ``ledger`` (optional) keys cumulative fills by ``order_id``: replaying
    the same /status payload books only the delta — never a double-count.
    """
    tag = f"aigrid_{plan_id}"
    levels = [float(p) for p in level_prices]
    tol = (float(price_tick) / 2.0) if price_tick else _TIGHT_TOLERANCE
    prices: set[float] = set()
    order_qty: dict[str, float] = {}
    partial_qty = 0.0
    unmapped = 0
    for trade in trades or ():
        if not isinstance(trade, dict) or trade.get("pair") != pair:
            continue
        orders: Any = trade.get("orders")
        if not isinstance(orders, list):
            continue
        for order in orders:
            if not _is_our_fill(order, tag, trade.get("enter_tag")):
                continue
            oid = str(order.get("order_id") or order.get("id") or "")
            filled = _to_float(order.get("filled")) or 0.0
            if oid:
                if ledger is not None:
                    ledger.apply(oid, filled)
                    cumulative = ledger.cumulative(oid)
                else:
                    cumulative = filled
                order_qty[oid] = round(cumulative, 10)
            if str(order.get("status")).lower() == "open":
                partial_qty += filled
            level = _nearest_level(_order_price(order), levels, tol)
            if level is None:
                unmapped += 1
            else:
                prices.add(level)
    return FillsResult(sorted(prices), unmapped, order_qty, partial_qty)


def _is_our_fill(order: Any, plan_tag: str, trade_tag: Any) -> bool:
    """Entry order with a confirmed (partial) fill, tagged for THIS plan.

    P0-5c: counts ``closed`` plus ``canceled``/``expired`` carrying
    ``filled>0`` and ``open`` orders mid partial-fill — every status in
    FILL_STATUSES represents real exposure."""
    if not isinstance(order, dict):
        return False
    order_tag = order.get("ft_order_tag")
    if not isinstance(order_tag, str):
        return False
    if not (
        order_tag == plan_tag
        or order_tag.startswith(f"{plan_tag}:")
        or (trade_tag == plan_tag and re.fullmatch(r"grid_level_\d+", order_tag) is not None)
    ):
        return False
    if not order.get("ft_is_entry"):
        return False
    if str(order.get("status")).lower() not in FILL_STATUSES:
        return False
    filled = _to_float(order.get("filled"))
    return filled is not None and filled > 0


def _order_price(order: dict) -> float | None:
    """Realized fill price (average); the order price is the fallback."""
    average = _to_float(order.get("average"))
    if average is not None and average > 0:
        return average
    return _to_float(order.get("price"))


def _nearest_level(price: float | None, levels: Sequence[float], tol: float) -> float | None:
    if price is None:
        return None
    best: float | None = None
    best_d = tol
    for lv in levels:
        d = abs(lv - price)
        if d <= best_d:
            best, best_d = lv, d
    return best


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
