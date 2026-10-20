"""``X-Forwarded-For`` is only honoured for a trusted proxy (BE-4).

Before this policy the limiter keyed on the *left-most* forwarded hop, so any
client could send a random header per request and never hit its bucket. The
rules pinned here:

* no ``TRUSTED_PROXIES`` → the socket peer wins, the header is ignored;
* a trusted peer → the **right-most hop that is not a trusted proxy** is the
  client (that is what our own proxy appended; a client can only prepend);
* a broken or all-trusted chain degrades to the socket peer, never to an
  exception and never to a fresh bucket per request.

Both the pure ``client_key()`` (synthetic scope, no HTTP) and the real
middleware through ``TestClient`` are exercised, because the bug lived exactly
at the boundary between them.
"""

from __future__ import annotations

import uuid

import pytest
from app.core.config import Settings, settings
from app.main import app
from app.middleware.ratelimit import client_key, is_exempt, peer_address
from fastapi.testclient import TestClient

PROXY_PEER = "203.0.113.1"
CLIENT_A = "198.51.100.7"
CLIENT_B = "198.51.100.8"


def _unique() -> tuple[str, str]:
    """Fresh peer/client pair: buckets live in the middleware instance and are
    shared by every request in the test session."""
    tail = uuid.uuid4().int
    peer = f"10.{(tail >> 24) & 255}.{(tail >> 16) & 255}.{(tail >> 8) & 255}"
    client = f"198.51.{(tail >> 16) & 255}.{(tail >> 8) & 255}"
    return peer, client


def scope(peer: str | None, forwarded: str | None = None) -> dict:
    """Minimal ASGI scope: only what the limiter reads."""
    headers = [(b"x-forwarded-for", forwarded.encode("latin-1"))] if forwarded else []
    return {
        "type": "http",
        "path": "/classify",
        "headers": headers,
        "client": (peer, 4321) if peer else None,
    }


# --- pure policy ---------------------------------------------------------


def test_header_is_ignored_without_trusted_proxies() -> None:
    """The spoofing hole: a direct client must not pick its own bucket."""
    spoofed = scope("192.0.2.50", CLIENT_A)
    assert client_key(spoofed, trusted_proxies=()) == "192.0.2.50"


def test_trusted_peer_uses_the_forwarded_client() -> None:
    assert client_key(scope(PROXY_PEER, CLIENT_A), trusted_proxies=(PROXY_PEER,)) == CLIENT_A


def test_prepended_hops_do_not_win_over_the_right_most_one() -> None:
    """A client can prepend anything: nginx appends the real address on the right."""
    chain = f"1.1.1.1, 203.0.113.99, {CLIENT_A}"
    assert client_key(scope(PROXY_PEER, chain), trusted_proxies=(PROXY_PEER,)) == CLIENT_A


def test_chain_of_trusted_proxies_is_walked_right_to_left() -> None:
    """Two proxies in front of the API: skip every trusted hop, take the client."""
    chain = f"{CLIENT_A}, 10.0.0.5, {PROXY_PEER}"
    key = client_key(scope(PROXY_PEER, chain), trusted_proxies=(PROXY_PEER, "10.0.0.0/8"))
    assert key == CLIENT_A


def test_network_and_ipv6_entries_are_supported() -> None:
    assert client_key(scope("10.5.6.7", CLIENT_A), trusted_proxies=("10.0.0.0/8",)) == CLIENT_A
    ipv6_scope = scope("2001:db8::1", "2001:db9:1::9")
    assert client_key(ipv6_scope, trusted_proxies=("2001:db8::/32",)) == "2001:db9:1::9"


def test_all_hops_trusted_falls_back_to_the_peer() -> None:
    """No untrusted hop left: the only address the API observed is the peer."""
    chain = f"{PROXY_PEER}, 10.0.0.5"
    assert client_key(scope(PROXY_PEER, chain), trusted_proxies=(PROXY_PEER, "10.0.0.0/8")) == (
        PROXY_PEER
    )


@pytest.mark.parametrize("forwarded", ["not-an-ip", "1.2.3.4:5678", "", "  ,  "])
def test_malformed_header_never_crashes_and_falls_back_to_the_peer(forwarded: str) -> None:
    assert client_key(scope(PROXY_PEER, forwarded), trusted_proxies=(PROXY_PEER,)) == PROXY_PEER


def test_malformed_hops_are_skipped_but_a_valid_one_is_used() -> None:
    chain = f"garbage, {CLIENT_A}"
    assert client_key(scope(PROXY_PEER, chain), trusted_proxies=(PROXY_PEER,)) == CLIENT_A


def test_wildcard_is_rejected_instead_of_disabling_the_protection(monkeypatch) -> None:
    monkeypatch.setenv("TRUSTED_PROXIES", "*")
    with pytest.raises(ValueError, match="TRUSTED_PROXIES"):
        Settings()


def test_missing_socket_peer_is_reported_as_unknown() -> None:
    assert client_key(scope(None, CLIENT_A), trusted_proxies=(PROXY_PEER,)) == "unknown"
    assert peer_address({"type": "http", "headers": []}) is None


# --- the real middleware -------------------------------------------------


def _client(peer: str) -> TestClient:
    return TestClient(app, client=(peer, 50000))


def _burst(monkeypatch) -> None:
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_rps", 1)
    monkeypatch.setattr(settings, "rate_limit_burst", 2)


def _post(test_client: TestClient, forwarded: str) -> int:
    return test_client.post(
        "/classify",
        json={"name": "Ноутбук Lenovo"},
        headers={"X-Forwarded-For": forwarded},
    ).status_code


def test_forged_header_does_not_reset_the_budget(monkeypatch) -> None:
    """The regression: rotating the header used to look like new clients."""
    _burst(monkeypatch)
    monkeypatch.setattr(settings, "trusted_proxies", ())
    peer, forwarded = _unique()
    direct = _client(peer)

    assert _post(direct, forwarded) == 200
    assert _post(direct, forwarded) == 200
    # A different header from the same peer is still the same client.
    assert _post(direct, "203.0.113.200") == 429


def test_forged_header_is_still_ignored_for_an_untrusted_peer_in_a_trusted_setup(
    monkeypatch,
) -> None:
    """Having a proxy configured must not make every other peer trusted."""
    _burst(monkeypatch)
    monkeypatch.setattr(settings, "trusted_proxies", (PROXY_PEER,))
    peer, _ = _unique()
    direct = _client(peer)

    # Two different forged headers spend the same burst of two tokens…
    assert _post(direct, "198.51.100.11") == 200
    assert _post(direct, "198.51.100.12") == 200
    # …so the third request is rejected: the peer is the key, not the header.
    assert _post(direct, "198.51.100.13") == 429


def test_trusted_proxy_still_separates_real_clients(monkeypatch) -> None:
    _burst(monkeypatch)
    monkeypatch.setattr(settings, "trusted_proxies", (PROXY_PEER,))
    proxy = _client(PROXY_PEER)
    _, first = _unique()

    assert _post(proxy, first) == 200
    assert _post(proxy, first) == 200
    assert _post(proxy, first) == 429
    # A genuinely different client behind the same proxy keeps its own bucket.
    assert _post(proxy, f"198.51.100.{(uuid.uuid4().int % 200) + 20}") == 200


def test_health_is_exempt_even_for_a_forged_header(monkeypatch) -> None:
    _burst(monkeypatch)
    monkeypatch.setattr(settings, "trusted_proxies", ())
    peer, forwarded = _unique()
    direct = _client(peer)
    for _ in range(4):
        assert direct.get("/health", headers={"X-Forwarded-For": forwarded}).status_code == 200
    assert is_exempt("/health/ready") and is_exempt("/metrics") and is_exempt("/ui/index.html")


# --- configuration -------------------------------------------------------


def test_trusted_proxies_are_parsed_and_normalized(monkeypatch) -> None:
    monkeypatch.setenv("TRUSTED_PROXIES", " 10.0.0.0/8 , 127.0.0.1 ,, ::1 ")
    assert Settings().trusted_proxies == ("10.0.0.0/8", "127.0.0.1", "::1")


def test_empty_setting_trusts_nobody(monkeypatch) -> None:
    monkeypatch.delenv("TRUSTED_PROXIES", raising=False)
    assert Settings().trusted_proxies == ()


def test_a_typo_fails_at_startup(monkeypatch) -> None:
    """Silently dropping the entry would either merge all clients or reopen the hole."""
    monkeypatch.setenv("TRUSTED_PROXIES", "10.0.0.0/8,proxy.internal")
    with pytest.raises(ValueError, match="TRUSTED_PROXIES"):
        Settings()


def test_cidr_with_host_bits_is_accepted(monkeypatch) -> None:
    monkeypatch.setenv("TRUSTED_PROXIES", "10.1.2.3/8")
    assert Settings().trusted_proxies == ("10.1.2.3/8",)


def test_deployment_disables_uvicorns_own_proxy_headers() -> None:
    """uvicorn trusts 127.0.0.1 by default and rewrites the peer from the header.

    That happens *before* the middleware and is invisible to in-process tests, so
    the image must not let uvicorn decide: otherwise TRUSTED_PROXIES is void.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    dockerfile = (root / "api" / "Dockerfile").read_text(encoding="utf-8")
    assert "--no-proxy-headers" in dockerfile

    compose = (root / "docker-compose.yml").read_text(encoding="utf-8")
    assert "TRUSTED_PROXIES" in compose
    assert "RATE_LIMIT_FILE_RPS" in compose
    assert "RATE_LIMIT_FILE_BURST" in compose
