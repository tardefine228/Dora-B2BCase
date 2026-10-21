"""BE-6: ``/metrics`` in the Prometheus text format + health of embedder and store.

The DoD of BE-6 is a passing scrape and counters by status/path/latency, so the
exposition is parsed here with the official ``prometheus_client`` parser — the
same code path a scraper's client library uses.
"""

from __future__ import annotations

import numpy as np
import pytest
from app.core.config import settings
from app.main import app
from app.services import classifier as classifier_service
from app.services.classifier import vector_layer_health
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

client = TestClient(app)


def _families(text: str) -> dict:
    """Parse the exposition; raises when a scraper could not read it."""
    return {family.name: family for family in text_string_to_metric_families(text)}


def _samples(families: dict, name: str) -> list:
    family = families.get(name)
    return list(family.samples) if family is not None else []


def test_scrape_returns_parseable_exposition_with_required_counters() -> None:
    client.get("/health")
    client.post("/classify", json={"name": "Ноутбук Lenovo"})

    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "# TYPE dora_http_requests_total counter" in response.text

    families = _families(response.text)

    # Counters by path/method/status (DoD: «счётчики по статусам/путям»).
    requests = _samples(families, "dora_http_requests")
    assert any(
        sample.name == "dora_http_requests_total"
        and sample.labels["path"] == "/classify"
        and sample.labels["method"] == "POST"
        and sample.labels["status"] == "200"
        and sample.value >= 1
        for sample in requests
    )

    # Latency histogram (DoD: «…латентности»).
    durations = _samples(families, "dora_http_request_duration_seconds")
    assert any(sample.name.endswith("_bucket") for sample in durations)
    assert any(sample.name.endswith("_count") for sample in durations)

    # Classification answers by status.
    classifications = _samples(families, "dora_classifications")
    assert any(
        sample.labels["mode"] == "single"
        and sample.labels["status"] in {"ok", "review", "not_found"}
        and sample.value >= 1
        for sample in classifications
    )


def test_batch_job_events_and_answers_are_counted() -> None:
    accepted = client.post("/classify/batch", json={"items": ["Цемент", "Ноутбук Lenovo"]})
    assert accepted.status_code == 202
    job_id = accepted.json()["job_id"]
    state = client.get(f"/classify/batch/{job_id}").json()
    assert state["status"] == "completed"  # in-process queue finishes synchronously

    families = _families(client.get("/metrics").text)
    events = {
        sample.labels["event"]: sample.value for sample in _samples(families, "dora_batch_jobs")
    }
    assert events.get("accepted", 0) >= 1
    assert events.get("completed", 0) >= 1
    batch_answers = [
        sample
        for sample in _samples(families, "dora_classifications")
        if sample.labels["mode"] == "batch"
    ]
    assert sum(sample.value for sample in batch_answers) >= 2


def test_dependency_health_gauges_are_exposed(monkeypatch) -> None:
    monkeypatch.setattr(settings, "vector_search", False)
    classifier_service.reset_classifier()
    try:
        response = client.get("/metrics")
        families = _families(response.text)
        assert _samples(families, "dora_classifier_up")[0].value == 1
        # Lexical-only run: nothing to load, so embedder/store are vacuously up
        # and the vector layer reports itself as disabled — an operator can see
        # the difference between "off by config" and "failed to load".
        assert _samples(families, "dora_vector_search_enabled")[0].value == 0
        assert _samples(families, "dora_embedder_up")[0].value == 1
        assert _samples(families, "dora_vector_store_up")[0].value == 1
        info = _samples(families, "dora_embedder_info")
        assert info and info[0].value == 1
        build = _samples(families, "dora_build_info")
        assert build
        assert build[0].labels["classifier_backend"] == settings.classifier_backend
        assert build[0].labels["job_store"] == settings.job_store
    finally:
        classifier_service.reset_classifier()


def test_silent_vector_degradation_is_visible(monkeypatch) -> None:
    """VECTOR_SEARCH=true but no index built ⇒ embedder_up=0, degraded=true."""
    monkeypatch.setattr(settings, "vector_search", True)
    monkeypatch.setattr(classifier_service, "_build_dense_index", lambda rows: None)
    classifier_service.reset_classifier()
    try:
        health = vector_layer_health(force=True)
        assert health["classifier_up"] is True
        assert health["vector_search_enabled"] is False
        assert health["embedder_up"] is False
        assert health["degraded"] is True

        families = _families(client.get("/metrics").text)
        assert _samples(families, "dora_embedder_up")[0].value == 0
    finally:
        classifier_service.reset_classifier()
        vector_layer_health(force=True)


def test_health_probe_is_ttl_cached(monkeypatch) -> None:
    """A scrape (or the Docker healthcheck poll) must not ping the store every time."""
    vector_layer_health(force=True)
    real_probe = classifier_service._probe_vector_health
    calls = {"probes": 0}

    def counting_probe():
        calls["probes"] += 1
        return real_probe()

    monkeypatch.setattr(classifier_service, "_probe_vector_health", counting_probe)
    try:
        vector_layer_health()
        vector_layer_health()
        assert calls["probes"] == 0  # served from the TTL cache
        vector_layer_health(force=True)
        assert calls["probes"] == 1
    finally:
        vector_layer_health(force=True)


def test_json_view_keeps_the_legacy_snapshot() -> None:
    body = client.get("/metrics", headers={"Accept": "application/json"}).json()
    for key in (
        "total_requests",
        "requests_by_status",
        "average_latency_ms",
        "requests_by_path",
        "rate_limited_total",
        "rate_limited_by_path",
        "classifications",
        "batch_jobs",
        "vector_health",
    ):
        assert key in body
    assert body["vector_health"]["classifier_up"] is True


def test_qdrant_mirror_ping_and_index_health() -> None:
    pytest.importorskip("qdrant_client")
    from ml.dense import DenseIndex, QdrantMirror

    names = ["Электрочайники", "Ноутбуки портативные"]
    vectors = np.eye(2, 8, dtype=np.float32)

    plain = DenseIndex(names, vectors, "test")
    assert plain.health() == {"store": "memory", "store_up": True}

    mirror = QdrantMirror.connect(":memory:", "", "test_health")
    mirror.publish(names, vectors, "test")
    assert mirror.ping() is True
    mirrored = DenseIndex(names, vectors, "test")
    mirrored.mirror, mirrored.store = mirror, "qdrant"
    assert mirrored.health() == {"store": "qdrant", "store_up": True}

    # A failing store answers False (the gauge drops), it never raises.
    def boom(**_kwargs):
        raise RuntimeError("qdrant is down")

    mirror._client.get_collections = boom
    assert mirror.ping() is False
    assert mirrored.health() == {"store": "qdrant", "store_up": False}
