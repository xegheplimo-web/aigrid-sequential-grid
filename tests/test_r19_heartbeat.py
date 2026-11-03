# Replay helpers require .orchestrator/fapi_exchangeinfo.json (unpublished).
"""R19: heartbeat reflects this pass instead of persisted foreign pairs."""

from aigrid.service import AIGridService
from tests.test_r14_cycle import NOW_MS, PAIR, _cycle


def test_heartbeat_processed_pairs_only(tmp_path, monkeypatch):
    cycle, orch, _, _ = _cycle(tmp_path, whitelist=(PAIR, "SOL/USDT:USDT"))
    cycle._seen["STALE/USDT:USDT"] = 123
    monkeypatch.setattr("aigrid.publish.build_deterministic_plan", lambda *a, **kw: None)
    svc = AIGridService(orch, interval_s=0, cycle=cycle)
    svc.on_cycle = lambda _: svc.stop()
    assert svc.run() == 0
    heartbeat = orch.store.load_state("heartbeat")
    expected = [PAIR, "SOL/USDT:USDT"]
    assert heartbeat["watchlist"] == expected
    assert list(heartbeat["candle_ts_ms"]) == expected
    assert heartbeat["candle_ts_ms"] == {pair: cycle.seen_ts[pair] for pair in expected}


def test_heartbeat_retains_freshness_without_new_candle(tmp_path, monkeypatch):
    cycle, orch, _, _ = _cycle(tmp_path, whitelist=(PAIR,))
    monkeypatch.setattr("aigrid.publish.build_deterministic_plan", lambda *a, **kw: None)
    svc = AIGridService(orch, interval_s=0, cycle=cycle)
    snapshots = []
    call_counts = []

    def after_tick(_):
        snapshots.append(orch.store.load_state("heartbeat"))
        call_counts.append(len(orch.calls))
        if len(snapshots) == 2:
            svc.stop()

    svc.on_cycle = after_tick
    assert svc.run() == 0
    assert len(snapshots) == 2
    assert call_counts[0] > 0
    assert call_counts[1] == call_counts[0]
    for heartbeat in snapshots:
        assert heartbeat["candle_ts_ms"] == {PAIR: NOW_MS - 300_000}
        assert heartbeat["watchlist"] == [PAIR]
