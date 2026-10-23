"""Rate limiting: burst behaviour, Retry-After, exemptions, per-IP isolation, off switch.

The client address comes from ``X-Forwarded-For``, which the API honours only for
a trusted proxy (see ``test_trusted_proxies.py`` for that policy). These tests
therefore connect from a peer that is listed in ``TRUSTED_PROXIES``, i.e. they
describe the deployment with a reverse proxy in front of the API.
"""

from __future__ import annotations

import time
import uuid

import pytest
from app.core.config import settings
from app.main import app
from fastapi.testclient import TestClient

PROXY_PEER = "203.0.113.1"
client = TestClient(app, client=(PROXY_PEER, 50000))


@pytest.fixture(autouse=True)
def _proxy_is_trusted(monkeypatch):
    """Without this the header is ignored and every request shares one bucket."""
    monkeypatch.setattr(settings, "trusted_proxies", (PROXY_PEER,))


def _ip() -> str:
    """A unique client address so tests never share a bucket."""
    return f"10.9.{uuid.uuid4().int % 250}.{uuid.uuid4().int % 250}"


def _classify(ip: str) -> int:
    response = client.post(
        "/classify",
        json={"name": "Ноутбук Lenovo"},
        headers={"X-Forwarded-For": ip},
    )
    return response.status_code


def test_burst_is_allowed_and_the_next_request_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_rps", 1)
    monkeypatch.setattr(settings, "rate_limit_burst", 3)
    ip = _ip()

    for _ in range(3):
        assert _classify(ip) == 200

    rejected = client.post(
        "/classify",
        json={"name": "Ноутбук Lenovo"},
        headers={"X-Forwarded-For": ip},
    )
    assert rejected.status_code == 429
    assert rejected.json()["detail"].startswith("Слишком много запросов")
    assert rejected.headers["Retry-After"]
    assert rejected.headers["X-RateLimit-Limit"] == "3"
    assert rejected.headers["X-RateLimit-Remaining"] == "0"
    # The observer middleware still tags rejected requests.
    assert rejected.headers["X-Request-ID"]


def test_allowed_requests_advertise_the_remaining_budget(monkeypatch) -> None:
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_rps", 1)
    monkeypatch.setattr(settings, "rate_limit_burst", 2)
    response = client.post(
        "/classify",
        json={"name": "Ноутбук Lenovo"},
        headers={"X-Forwarded-For": _ip()},
    )
    assert response.status_code == 200
    assert response.headers["X-RateLimit-Limit"] == "2"
    assert response.headers["X-RateLimit-Remaining"] == "1"


def test_bucket_refills_over_time(monkeypatch) -> None:
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_rps", 20)
    monkeypatch.setattr(settings, "rate_limit_burst", 1)
    ip = _ip()

    assert _classify(ip) == 200
    assert _classify(ip) == 429
    time.sleep(0.1)  # 20 rps → one token every 50 ms
    assert _classify(ip) == 200


def test_health_metrics_and_ui_are_exempt(monkeypatch) -> None:
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_rps", 1)
    monkeypatch.setattr(settings, "rate_limit_burst", 1)
    ip = _ip()
    headers = {"X-Forwarded-For": ip}

    for _ in range(5):
        assert client.get("/health", headers=headers).status_code == 200
        assert client.get("/health/ready", headers=headers).status_code == 200
        assert client.get("/metrics", headers=headers).status_code == 200
        assert client.get("/ui/", headers=headers).status_code == 200
        assert client.get("/ui/index.html", headers=headers).status_code == 200


def test_limits_are_per_client(monkeypatch) -> None:
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_rps", 1)
    monkeypatch.setattr(settings, "rate_limit_burst", 1)
    first, second = _ip(), _ip()

    assert _classify(first) == 200
    assert _classify(first) == 429
    # A different client has its own bucket.
    assert _classify(second) == 200


def test_admin_has_a_stricter_own_limit(monkeypatch) -> None:
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_rps", 10)
    monkeypatch.setattr(settings, "rate_limit_burst", 10)
    monkeypatch.setattr(settings, "rate_limit_admin_rps", 1)
    monkeypatch.setattr(settings, "rate_limit_admin_burst", 1)
    monkeypatch.setattr(settings, "admin_api_key", "test-secret")
    ip = _ip()
    headers = {"X-Forwarded-For": ip, "X-Admin-API-Key": "test-secret"}

    assert client.get("/admin/catalog", headers=headers).status_code == 200
    rejected = client.get("/admin/catalog", headers=headers)
    assert rejected.status_code == 429
    # The public tier is unaffected by the admin bucket.
    assert _classify(ip) == 200


def test_limiter_can_be_switched_off(monkeypatch) -> None:
    monkeypatch.setattr(settings, "rate_limit_enabled", False)
    monkeypatch.setattr(settings, "rate_limit_rps", 1)
    monkeypatch.setattr(settings, "rate_limit_burst", 1)
    ip = _ip()

    for _ in range(5):
        assert _classify(ip) == 200


def test_rejections_are_counted_in_metrics(monkeypatch) -> None:
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_rps", 1)
    monkeypatch.setattr(settings, "rate_limit_burst", 1)
    ip = _ip()

    assert _classify(ip) == 200
    assert _classify(ip) == 429
    snapshot = client.get("/metrics", headers={"Accept": "application/json"}).json()
    assert snapshot["rate_limited_total"] >= 1
    assert snapshot["rate_limited_by_path"].get("/classify", 0) >= 1
    # The same rejection is visible to a Prometheus scrape (BE-6).
    exposition = client.get("/metrics").text
    assert 'dora_http_rate_limited_total{path="/classify"}' in exposition


_CSV = "Наименование;Кол-во\nНоутбук Lenovo;1\nМышь компьютерная;2\n".encode()


def _upload(ip: str) -> int:
    """POST /classify/file as *ip*; returns the status code."""
    response = client.post(
        "/classify/file",
        files={"file": ("goods.csv", _CSV, "text/csv")},
        headers={"X-Forwarded-For": ip},
    )
    return response.status_code


def test_file_uploads_have_their_own_bucket_counted_in_files(monkeypatch) -> None:
    """One upload = up to 100 classifications, so it must not share the API bucket."""
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_rps", 10)
    monkeypatch.setattr(settings, "rate_limit_burst", 20)
    monkeypatch.setattr(settings, "rate_limit_file_rps", 1)
    monkeypatch.setattr(settings, "rate_limit_file_burst", 2)
    ip = _ip()

    assert _upload(ip) == 202
    assert _upload(ip) == 202

    rejected = client.post(
        "/classify/file",
        files={"file": ("goods.csv", _CSV, "text/csv")},
        headers={"X-Forwarded-For": ip},
    )
    assert rejected.status_code == 429
    assert rejected.headers["Retry-After"]
    assert rejected.headers["X-RateLimit-Limit"] == "2"


def test_file_and_request_buckets_do_not_consume_each_other(monkeypatch) -> None:
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_rps", 1)
    monkeypatch.setattr(settings, "rate_limit_burst", 2)
    monkeypatch.setattr(settings, "rate_limit_file_rps", 1)
    monkeypatch.setattr(settings, "rate_limit_file_burst", 1)
    ip = _ip()

    # The upload spends the file bucket only, even though it is the only request.
    assert _upload(ip) == 202
    assert _upload(ip) == 429
    for _ in range(2):
        assert _classify(ip) == 200
    assert _classify(ip) == 429

    other = _ip()
    # A spent request bucket does not block a file upload from the same client.
    assert _classify(other) == 200
    assert _classify(other) == 200
    assert _classify(other) == 429
    assert _upload(other) == 202
