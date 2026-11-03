"""Binance USDT-M exchange info loader (tick size, lot size, min notional).

Reads the existing snapshot at .orchestrator/fapi_exchangeinfo.json (read-only).
Used by the plan validator to enforce exchange rules on AI-proposed grids.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SymbolRules:
    symbol: str
    tick_size: float
    step_size: float
    min_qty: float
    max_qty: float
    min_notional: float
    max_price: float
    min_price: float
    price_precision: int
    quantity_precision: int


class ExchangeInfo:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        with self.path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        self._rules: dict[str, SymbolRules] = {}
        for s in data.get("symbols", []):
            if s.get("contractType") != "PERPETUAL" or s.get("status") != "TRADING":
                continue
            if s.get("quoteAsset") != "USDT":
                continue
            filters = {f["filterType"]: f for f in s.get("filters", [])}
            price = filters.get("PRICE_FILTER", {})
            lot = filters.get("LOT_SIZE", {})
            notional = filters.get("MIN_NOTIONAL", {})
            self._rules[s["symbol"]] = SymbolRules(
                symbol=s["symbol"],
                tick_size=float(price.get("tickSize", 0) or 0),
                step_size=float(lot.get("stepSize", 0) or 0),
                min_qty=float(lot.get("minQty", 0) or 0),
                max_qty=float(lot.get("maxQty", 0) or 0),
                min_notional=float(notional.get("notional", 0) or 0),
                max_price=float(price.get("maxPrice", 0) or 0),
                min_price=float(price.get("minPrice", 0) or 0),
                price_precision=int(s.get("pricePrecision", 8)),
                quantity_precision=int(s.get("quantityPrecision", 8)),
            )

    def rules_for(self, pair: str) -> SymbolRules:
        """ETH/USDT:USDT -> ETHUSDT"""
        symbol = pair.split(":", 1)[0].replace("/", "")
        rules = self._rules.get(symbol)
        if rules is None:
            raise KeyError(f"Symbol {symbol} not found in exchange info")
        return rules

    def has(self, pair: str) -> bool:
        symbol = pair.split(":", 1)[0].replace("/", "")
        return symbol in self._rules
