# Replay helpers require .orchestrator/fapi_exchangeinfo.json (unpublished).
"""R19: one runtime snapshot per pass, including passes without plans."""

from unittest.mock import Mock

import pytest
from tests.test_r14_cycle import PAIR, _cycle, _facts, _integration


def test_regime_skipped_fetches_once(tmp_path, monkeypatch):
    cycle, _, _, events = _cycle(tmp_path, whitelist=(PAIR, "SOL/USDT:USDT", "XRP/USDT:USDT"))
    provider = Mock(return_value=_facts())
    cycle._facts_provider = provider
    monkeypatch.setattr("aigrid.publish.build_deterministic_plan", lambda *a, **kw: None)
    cycle.tick()
    provider.assert_called_once_with()
    assert "ATTESTATION_RUNTIME_MISSING" not in events.kinds()
    assert _integration(tmp_path / "state") is None


def test_one_plan_signed_with_same_snapshot(tmp_path):
    cycle, _, _, _ = _cycle(tmp_path)
    cycle._reconciled = True
    snapshot = _facts()
    provider = Mock(return_value=snapshot)
    cycle._facts_provider = provider
    publish = Mock(wraps=cycle.publisher.publish)
    sync = Mock(wraps=cycle.publisher.sync_active_file)
    cycle.publisher.publish = publish
    cycle.publisher.sync_active_file = sync
    assert cycle.tick()[0].status == "PUBLISHED"
    provider.assert_called_once_with()
    assert publish.call_args.kwargs["facts"] is snapshot
    assert sync.call_args.args == (snapshot,)
    payload = _integration(tmp_path / "state")
    assert payload["attestation"]
    assert payload["enabled"] is True


@pytest.mark.parametrize("raises", [False, True])
def test_unavailable_facts_no_writes(tmp_path, raises):
    cycle, _, _, events = _cycle(tmp_path)
    path = tmp_path / "state" / "ft_integration.json"
    path.write_bytes(b'{"enabled": false, "plans": {}}')
    before = path.read_bytes(), path.stat().st_mtime_ns
    cycle._facts_provider = Mock(
        return_value=None, side_effect=RuntimeError("unavailable") if raises else None
    )
    cycle.publisher._runtime_facts = _facts()  # must never reuse stale evidence
    save = Mock(wraps=cycle.publisher._store.save_state)
    cycle.publisher._store.save_state = save
    cycle.tick()
    assert events.kinds().count("FACTS_UNAVAILABLE") == 1
    assert "ATTESTATION_RUNTIME_MISSING" not in events.kinds()
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert all(call.args[0] != "ft_integration" for call in save.call_args_list)


def test_signed_unchanged_payload_no_rewrite(tmp_path):
    cycle, _, _, _ = _cycle(tmp_path)
    cycle._reconciled = True
    cycle.tick()
    path = tmp_path / "state" / "ft_integration.json"
    before = path.read_bytes(), path.stat().st_mtime_ns
    assert cycle.publisher.sync_active_file(_facts()) is False
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
