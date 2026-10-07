from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_classify_returns_temporary_contract() -> None:
    response = client.post("/classify", json={"name": "  Ноутбук Lenovo  "})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "Ноутбук Lenovo"
    assert body["code"] == "MOCK-001"
    assert body["status"] == "review"
    assert 0 <= body["confidence"] <= 1
    assert body["latency_ms"] >= 0


def test_classify_rejects_empty_name() -> None:
    response = client.post("/classify", json={"name": "   "})

    assert response.status_code == 422


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
