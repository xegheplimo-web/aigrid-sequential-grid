"""R18: service cycles read only closed candles from Freqtrade REST."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from aigrid import service
from aigrid.config import AIGridSettings
from aigrid.publish import FtRestClient
from aigrid.state.store import StateStore

PAIR = "ETH/USDT:USDT"
NOW = pd.Timestamp("2026-11-03T14:07:00Z")


def _frame(count=600):
    return pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 42.0},
        index=pd.date_range(end="2026-11-03T14:00:00Z", periods=count, freq="5min", name="date"),
    )


@pytest.fixture
def wired_service(monkeypatch, tmp_path):
    settings = AIGridSettings(mode="dryrun", state_dir=tmp_path, symbols=(PAIR,))
    orch = SimpleNamespace(
        settings=settings,
        store=StateStore(tmp_path),
        eventlog=Mock(),
        validator=None,
        ownership=None,
        _price_tick=Mock(return_value=0.01),
        adapter=SimpleNamespace(get_ohlcv=Mock(side_effect=AssertionError("file fallback"))),
    )
    captured = {}

    def capture_run(self, *args):
        captured["cycle"] = self._cycle
        # Use the production publisher's clock injection for deterministic closure.
        self._cycle.publisher._clock = lambda: NOW.timestamp()
        return 0

    monkeypatch.setattr(service, "load_settings", lambda _: settings)
    monkeypatch.setattr(service, "AIGridOrchestrator", lambda *a, **kw: orch)
    monkeypatch.setattr(
        FtRestClient,
        "show_config",
        Mock(
            return_value={
                "pair_whitelist": [PAIR],
                "dry_run": True,
                "trading_mode": "futures",
                "runmode": "dry_run",
            }
        ),
    )
    monkeypatch.setattr(service.AIGridService, "run", capture_run)
    monkeypatch.setattr(service.signal, "signal", lambda *a: None)
    monkeypatch.delenv(service.APPROVED_CONFIG_ENV, raising=False)
    assert service.main([]) == 0
    return captured["cycle"], orch


def test_service_cycle_uses_rest_loader(wired_service, monkeypatch, caplog):
    cycle, orch = wired_service
    expected = _frame()
    # Exercise the real REST parser as well as the service's loader wiring.
    payload = {
        "columns": ["date", "open", "high", "low", "close", "volume"],
        "data": [
            [ts.isoformat(), *row]
            for ts, row in zip(
                expected.index, expected.itertuples(index=False, name=None), strict=False
            )
        ],
    }
    rest_get = Mock(return_value=payload)
    monkeypatch.setattr(FtRestClient, "_get", rest_get)
    with caplog.at_level(logging.DEBUG, logger="aigrid.service"):
        actual = cycle._load_candles(PAIR)
    pd.testing.assert_frame_equal(actual, expected, check_freq=False)
    rest_get.assert_called_once_with(
        "/api/v1/pair_candles",
        {"pair": PAIR, "timeframe": "5m", "limit": cycle.publisher.min_closed_candles + 30},
    )
    assert "REST_CANDLES pair=ETH/USDT:USDT candle_ts_ms=1793714400000" in caplog.text
    orch.adapter.get_ohlcv.assert_not_called()


def test_forming_candle_dropped(wired_service, monkeypatch):
    cycle, orch = wired_service
    expected = _frame()
    frame = pd.concat(
        [
            expected,
            expected.iloc[[-1]].set_axis(
                pd.DatetimeIndex([NOW - pd.Timedelta(seconds=60)], name="date")
            ),
        ]
    )
    rest = Mock(return_value=frame)
    monkeypatch.setattr(FtRestClient, "pair_candles", rest)
    pd.testing.assert_frame_equal(cycle._load_candles(PAIR), expected, check_freq=False)
    rest.assert_called_once_with(PAIR, "5m", cycle.publisher.min_closed_candles + 30)
    orch.adapter.get_ohlcv.assert_not_called()


def test_insufficient_closed_candles_returns_none(wired_service, monkeypatch):
    cycle, orch = wired_service
    monkeypatch.setattr(FtRestClient, "pair_candles", Mock(return_value=_frame(100)))
    assert cycle._load_candles(PAIR) is None
    orch.eventlog.log.assert_called_with(
        "DATA_UNAVAILABLE", PAIR, {"error": "insufficient closed candles: 100"}
    )
    orch.adapter.get_ohlcv.assert_not_called()


def test_rest_error_returns_none_no_fallback(wired_service, monkeypatch, caplog):
    cycle, orch = wired_service
    rest = Mock(side_effect=OSError("REST offline"))
    monkeypatch.setattr(FtRestClient, "pair_candles", rest)
    with caplog.at_level(logging.DEBUG, logger="aigrid.service"):
        assert cycle._load_candles(PAIR) is None
    assert "REST candle loader failed" in caplog.text
    # The actual REST client also returns None on REST/parse errors.
    for invalid in (None, pd.DataFrame({"close": [1.0]})):
        rest.side_effect = None
        rest.return_value = invalid
        assert cycle._load_candles(PAIR) is None
    orch.adapter.get_ohlcv.assert_not_called()
