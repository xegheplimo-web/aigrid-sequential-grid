"""R12 tests — filled_level_prices comes ONLY from Freqtrade order state.

Card R12 (P0-5): ``engine.state.lots`` is a candle-driven VIRTUAL book, not
execution. ``persist()`` must publish ``filled_level_prices[pair]`` built
exclusively from orders Freqtrade reports on ``GET /api/v1/status``:

- only entry orders (``ft_is_entry``), ``status == "closed"``, ``filled > 0``,
  whose ``ft_order_tag == f"aigrid_{plan_id}"``
- each maps to a grid level by PRICE within ``price_tick/2`` (tick from the
  fapi_exchangeinfo snapshot; 1e-9 exact-match fallback when no tick)
- ANY failure (REST down, unparseable payload, zero trades) -> the key is
  ABSENT from the payload and ``reconciliation_state == "UNVERIFIED"``; the
  engine's virtual lots are never consulted as a fallback
- the legacy ``filled_levels`` index list is unchanged but tagged
  ``filled_levels_source == "virtual"``

All tests are hermetic: the HTTP layer is a fake — no socket is opened.
"""

from __future__ import annotations

import base64
import contextlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "user_data/data/binance/futures"
PAIR = "ETH/USDT:USDT"
PLAN_ID = "plan-r12"


# --------------------------------------------------------------------- fakes
class _FakeReader:
    """Duck-typed FtFillsReader: returns canned trades or raises."""

    def __init__(self, trades=None, exc: Exception | None = None):
        self._trades = trades
        self._exc = exc
        self.calls = 0

    def fetch_open_trades(self):
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return self._trades


def _order(**kw) -> dict:
    order = {
        "order_id": "o1",
        "pair": PAIR,
        "ft_pair": "ETHUSDT",
        "ft_order_side": "buy",
        "ft_is_entry": True,
        "ft_order_tag": f"aigrid_{PLAN_ID}",
        "status": "closed",
        "order_type": "limit",
        "price": 95.0,
        "average": 95.0,
        "amount": 1.0,
        "filled": 1.0,
        "remaining": 0.0,
    }
    order.update(kw)
    return order


def _trade(orders, pair: str = PAIR) -> dict:
    return {
        "trade_id": 1,
        "pair": pair,
        "strategy": "AIGridStrategy",
        "is_open": True,
        "orders": orders,
    }


def _plan(**kw):
    from aigrid.plans.schema import AIGridPlan, GridMode

    now = datetime.now(UTC)
    args = {
        "plan_id": PLAN_ID,
        "plan_version": 1,
        "symbol": PAIR,
        "created_at": now,
        "expires_at": now + timedelta(minutes=30),
        "market_regime": "RANGE",
        "confidence": 0.8,
        "grid_mode": GridMode.RANGE_GRID,
        "grid_lower": 85.0,
        "grid_upper": 95.0,
        "grid_levels": 3,
        "grid_spacing": 0.05,
        "max_capital": 240.0,
        "max_position_size": 720.0,
        "max_leverage": 3.0,
        "level_stake": 20.0,
        "level_notional": 40.0,
        "stop_loss": 80.0,
    }
    args.update(kw)
    return AIGridPlan(**args)  # descending levels [95.0, 90.0, 85.0]


@contextlib.contextmanager
def _orch(tmp_path, reader, *, exchange_info=None, plan=None):
    """Real orchestrator + real persist(); only the REST layer is faked."""
    from aigrid.orchestrator import AIGridOrchestrator
    from aigrid.settings import AIGridSettings

    settings = AIGridSettings(
        enabled=True,
        mode="shadow",
        symbols=(PAIR,),
        timeframe="5m",
        data_dir=DATA_DIR,
        state_dir=tmp_path / "state",
    )
    orch = AIGridOrchestrator(settings, exchange_info=exchange_info, ft_fills_reader=reader)
    try:
        orch.adopt_plan(plan or _plan())
        yield orch
    finally:
        orch.close()


def _payload(orch) -> dict:
    path = orch.store.state_dir / "ft_integration.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _fill_virtual_lot_at_85(orch) -> None:
    """Book a VIRTUAL lot at price 85.0 (level index 2) via one candle."""
    engine = orch._engines[PAIR]
    actions = engine.on_candle(1_700_000_000_000, 90.0, 90.0, 84.5, 85.0)
    assert [a.price for a in actions if a.kind == "BUY"] == [85.0]
    assert engine.state.filled_levels == {2}


def _exchange_info(tmp_path, tick: str = "0.01"):
    """Real ExchangeInfo on a minimal fixture file (hermetic)."""
    from aigrid.exchange_info import ExchangeInfo

    doc = {
        "symbols": [
            {
                "symbol": "ETHUSDT",
                "contractType": "PERPETUAL",
                "status": "TRADING",
                "quoteAsset": "USDT",
                "pricePrecision": 2,
                "quantityPrecision": 4,
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": tick},
                    {
                        "filterType": "LOT_SIZE",
                        "stepSize": "0.001",
                        "minQty": "0.001",
                        "maxQty": "1000",
                    },
                    {"filterType": "MIN_NOTIONAL", "notional": "5"},
                ],
            }
        ]
    }
    path = tmp_path / "fapi_exchangeinfo.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return ExchangeInfo(path)


# =============================================================== card cases
def test_price_list_from_ft_orders(tmp_path):
    """Fake REST: 1 trade, 2 filled entries @95 and @90 tagged for this plan,
    plus 1 filled entry tagged for a DIFFERENT plan -> [90.0, 95.0]."""
    reader = _FakeReader(
        [
            _trade(
                [
                    _order(order_id="a", price=95.0, average=95.0),
                    _order(order_id="b", price=90.0, average=90.0),
                    _order(
                        order_id="c",
                        price=85.0,
                        average=85.0,
                        ft_order_tag="aigrid_plan-old",
                    ),
                ]
            )
        ]
    )
    with _orch(tmp_path, reader) as orch:
        orch.persist()
        payload = _payload(orch)
    assert payload["filled_level_prices"][PAIR] == [90.0, 95.0]
    assert payload["unmapped_orders"][PAIR] == 0
    assert payload["reconciliation_state"] == "VERIFIED"
    assert payload["filled_levels_source"] == "virtual"


def test_virtual_lots_ignored(tmp_path):
    """The engine books a VIRTUAL lot at 85, but REST shows no qualifying
    order -> 85 must NOT appear. The lot is still visible in the legacy
    filled_levels index (kept, tagged virtual) — the two books are separate."""
    reader = _FakeReader(
        [_trade([_order(ft_is_entry=False, filled=0.5, price=85.0, average=85.0)])]
    )
    with _orch(tmp_path, reader) as orch:
        _fill_virtual_lot_at_85(orch)
        orch.persist()
        payload = _payload(orch)
    assert payload["filled_level_prices"][PAIR] == []
    assert 85.0 not in payload["filled_level_prices"][PAIR]
    assert payload["filled_levels"][PAIR] == [2]  # virtual index, kept
    assert payload["filled_levels_source"] == "virtual"
    assert payload["reconciliation_state"] == "VERIFIED"


def test_rest_unavailable_omits_key(tmp_path):
    """Reader raises -> no filled_level_prices key at all +
    reconciliation_state=UNVERIFIED — even with a virtual lot in the book,
    the engine's lots are never used as a fallback."""
    reader = _FakeReader(exc=OSError("connection refused"))
    with _orch(tmp_path, reader) as orch:
        _fill_virtual_lot_at_85(orch)
        orch.persist()
        payload = _payload(orch)
    assert "filled_level_prices" not in payload
    assert payload["reconciliation_state"] == "UNVERIFIED"
    assert payload["filled_levels"][PAIR] == [2]  # legacy index survives
    assert payload["filled_levels_source"] == "virtual"


def test_no_trades_omits_key(tmp_path):
    """/status returns an empty list -> same fail-closed shape: key absent."""
    with _orch(tmp_path, _FakeReader([])) as orch:
        orch.persist()
        payload = _payload(orch)
    assert "filled_level_prices" not in payload
    assert payload["reconciliation_state"] == "UNVERIFIED"


def test_partial_fill_below_tolerance_not_counted(tmp_path):
    """P0-5c (R14/D-100) supersedes the R12 rule: an ``open`` order mid
    partial-fill IS real exposure and must be accounted — the level is
    counted so nothing can re-enter on top of a live fill."""
    reader = _FakeReader([_trade([_order(status="open", filled=0.5, price=90.0, average=90.0)])])
    with _orch(tmp_path, reader) as orch:
        orch.persist()
        payload = _payload(orch)
    assert payload["filled_level_prices"][PAIR] == [90.0]


def test_tag_mismatch_excluded(tmp_path):
    """Filled entries belonging to an OLD plan are never counted."""
    reader = _FakeReader(
        [
            _trade(
                [
                    _order(
                        order_id="x",
                        price=90.0,
                        average=90.0,
                        ft_order_tag="aigrid_plan-old",
                    ),
                    _order(
                        order_id="y",
                        price=85.0,
                        average=85.0,
                        ft_order_tag="aigrid_other",
                    ),
                ]
            )
        ]
    )
    with _orch(tmp_path, reader) as orch:
        orch.persist()
        payload = _payload(orch)
    assert payload["filled_level_prices"][PAIR] == []


def test_price_tick_tolerance(tmp_path):
    """tick=0.01 (real ExchangeInfo fixture): a fill at 95.004 maps to the
    95.0 level (within tick/2); a fill at 95.02 does NOT map and is counted
    in unmapped_orders."""
    ei = _exchange_info(tmp_path, tick="0.01")
    reader = _FakeReader(
        [
            _trade(
                [
                    _order(order_id="a", price=95.0, average=95.004),
                    _order(order_id="b", price=95.02, average=95.02),
                ]
            )
        ]
    )
    with _orch(tmp_path, reader, exchange_info=ei) as orch:
        orch.persist()
        payload = _payload(orch)
    assert payload["filled_level_prices"][PAIR] == [95.0]
    assert payload["unmapped_orders"][PAIR] == 1


def test_exact_match_fallback_without_tick(tmp_path):
    """No exchange info -> 1e-9 tolerance: 95.0 + 5e-10 still maps, +2e-9
    does not (and counts as unmapped)."""
    reader = _FakeReader(
        [
            _trade(
                [
                    _order(order_id="a", price=95.0, average=95.0 + 5e-10),
                    _order(order_id="b", price=95.0, average=95.0 + 2e-9),
                ]
            )
        ]
    )
    with _orch(tmp_path, reader) as orch:  # exchange_info=None
        orch.persist()
        payload = _payload(orch)
    assert payload["filled_level_prices"][PAIR] == [95.0]
    assert payload["unmapped_orders"][PAIR] == 1


# ------------------------------------------------- reader unit tests (fake HTTP)
class _FakeOpener:
    """Captures the request and replays a canned response — no socket."""

    def __init__(self, body: bytes = b"[]", exc: Exception | None = None):
        self._body = body
        self._exc = exc
        self.requests = []
        self.timeouts = []

    def open(self, req, timeout=None):
        self.requests.append(req)
        self.timeouts.append(timeout)
        if self._exc is not None:
            raise self._exc

        class _Resp:
            def __init__(self, body):
                self._body = body

            def read(self):
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _Resp(self._body)


def test_reader_gets_status_with_basic_auth():
    """The reader issues exactly one GET to /api/v1/status with the same
    Basic-auth scheme preflight uses — never a write call."""
    from aigrid.state.ft_fills import FtFillsReader

    opener = _FakeOpener(b"[]")
    reader = FtFillsReader("http://ft.local:8080/", username="user", password="pass", opener=opener)
    assert reader.fetch_open_trades() == []
    req = opener.requests[0]
    assert req.full_url == "http://ft.local:8080/api/v1/status"
    assert req.get_method() == "GET"
    assert req.headers.get("Authorization") == "Basic " + base64.b64encode(b"user:pass").decode()
    assert opener.timeouts[0] and opener.timeouts[0] > 0


def test_reader_rejects_non_list_response():
    """An unparseable/malformed response raises — the orchestrator must treat
    it as UNVERIFIED, never publish a book built from garbage."""
    from aigrid.state.ft_fills import FtFillsError, FtFillsReader

    opener = _FakeOpener(b'{"unexpected": "dict"}')
    with pytest.raises(FtFillsError):
        FtFillsReader("http://ft.local:8080", opener=opener).fetch_open_trades()


def test_reader_from_env_uses_preflight_settings(monkeypatch):
    """Same env vars / default base URL as aigrid.preflight."""
    from aigrid.preflight import DEFAULT_API_URL
    from aigrid.state.ft_fills import FtFillsReader

    monkeypatch.delenv("FT_API_URL", raising=False)
    assert FtFillsReader.from_env().base_url == DEFAULT_API_URL
    monkeypatch.setenv("FT_API_URL", "http://a:1/")
    monkeypatch.setenv("FT_API_USER", "u")
    monkeypatch.setenv("FT_API_PASS", "p")
    r = FtFillsReader.from_env()
    assert r.base_url == "http://a:1"  # trailing slash stripped
