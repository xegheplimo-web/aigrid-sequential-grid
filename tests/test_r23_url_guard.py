"""R23: reject non-HTTP URLs before invoking urllib's transport."""

import pytest
from aigrid import preflight


@pytest.mark.parametrize("url", ["ftp://example/x", "file:///etc/passwd", "example/x"])
def test_non_http_url_rejected_before_network(monkeypatch, url):
    def unexpected_urlopen(*args, **kwargs):
        pytest.fail("non-HTTP URL reached urlopen")

    monkeypatch.setattr(preflight.urllib.request, "urlopen", unexpected_urlopen)
    with pytest.raises(ValueError, match="refusing non-HTTP URL"):
        preflight._get_json(url, None, None, 0.1)


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_http_url_reaches_urlopen(monkeypatch, scheme):
    calls = []

    class TransportReached(Exception):
        pass

    def intercepted_urlopen(req, *, timeout):
        calls.append((req.full_url, timeout))
        raise TransportReached

    monkeypatch.setattr(preflight.urllib.request, "urlopen", intercepted_urlopen)
    url = f"{scheme}://127.0.0.1:1/x"
    with pytest.raises(TransportReached):
        preflight._get_json(url, None, None, 0.1)
    assert calls == [(url, 0.1)]
