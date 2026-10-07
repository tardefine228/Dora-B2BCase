from app.core.config import settings
from app.main import app
from app.services.classifier import reset_classifier
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metrics() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert "total_requests" in body
    assert "average_latency_ms" in body


def test_classify_single_contract() -> None:
    response = client.post("/classify", json={"name": "  Ноутбук Lenovo  "})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "Ноутбук Lenovo"
    assert 0 <= body["confidence"] <= 1
    assert body["status"] in ("ok", "review", "not_found")
    assert body["latency_ms"] >= 0


def test_classify_rejects_empty_name() -> None:
    response = client.post("/classify", json={"name": "   "})
    assert response.status_code == 422


def test_classify_direct_okpd2_code() -> None:
    response = client.post("/classify", json={"name": "14.1"})
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == "14.1"
    assert body["status"] == "ok"


def test_classify_notebook_resolves_to_catalog() -> None:
    response = client.post("/classify", json={"name": "Ноутбук Lenovo"})
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == "26.20.11"
    assert body["query"] == "Ноутбук Lenovo"


def test_classify_unknown_returns_not_found() -> None:
    response = client.post("/classify", json={"name": "qwe rty zxc"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "not_found"
    assert body["code"] is None


def test_batch_returns_job_and_progress() -> None:
    response = client.post(
        "/classify/batch",
        json={"items": ["Ноутбук Lenovo", "Мышь компьютерная"]},
    )
    assert response.status_code == 202
    accepted = response.json()
    assert accepted["status"] == "queued"
    assert accepted["total"] == 2

    status_response = client.get(accepted["status_url"])
    assert status_response.status_code == 200
    status = status_response.json()
    assert status["status"] == "completed"
    assert status["completed"] == 2
    assert status["progress_percent"] == 100
    assert len(status["results"]) == 2


def test_batch_rejects_more_than_100_items() -> None:
    response = client.post(
        "/classify/batch",
        json={"items": [f"Товар {i}" for i in range(101)]},
    )
    assert response.status_code == 422


def test_batch_status_returns_404_for_unknown_job() -> None:
    response = client.get("/classify/batch/unknown-job")
    assert response.status_code == 404


def test_classify_mock_backend(monkeypatch) -> None:
    monkeypatch.setattr(settings, "classifier_backend", "mock")
    reset_classifier()
    try:
        response = client.post("/classify", json={"name": "Ноутбук"})
        assert response.status_code == 200
        body = response.json()
        assert body["code"] == "MOCK-001"
        assert body["status"] == "review"
    finally:
        reset_classifier()
