from app.core.config import settings
from app.main import _resolve_frontend_dir, app
from app.services.classifier import reset_classifier
from fastapi.testclient import TestClient

client = TestClient(app)


def test_frontend_path_resolves_in_docker_image_layout(tmp_path) -> None:
    image_root = tmp_path / "app"
    module_file = image_root / "app" / "main.py"
    frontend_dir = image_root / "frontend"
    frontend_dir.mkdir(parents=True)

    assert _resolve_frontend_dir(module_file) == frontend_dir


def test_frontend_path_resolves_in_source_tree_layout(tmp_path) -> None:
    repository_root = tmp_path / "repo"
    module_file = repository_root / "api" / "app" / "main.py"
    frontend_dir = repository_root / "frontend"
    frontend_dir.mkdir(parents=True)

    assert _resolve_frontend_dir(module_file) == frontend_dir


def test_root_redirects_to_the_demo_ui() -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/ui/"


def test_health_and_readiness() -> None:
    liveness = client.get("/health")
    assert liveness.status_code == 200
    assert liveness.json()["status"] == "ok"

    readiness = client.get("/health/ready")
    assert readiness.status_code == 200
    assert readiness.json()["status"] == "ready"


def test_metrics() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert "total_requests" in body
    assert "average_latency_ms" in body
    # The middleware records a request after the endpoint has returned its snapshot.
    assert "/health" in body["requests_by_path"]


def test_classify_single_contract_and_request_id() -> None:
    response = client.post(
        "/classify",
        json={"name": "  Ноутбук Lenovo  "},
        headers={"X-Request-ID": "test-request-17"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "Ноутбук Lenovo"
    assert 0 <= body["confidence"] <= 1
    assert body["status"] in ("ok", "review", "not_found")
    assert body["latency_ms"] >= 0
    assert response.headers["X-Request-ID"] == "test-request-17"
    assert float(response.headers["X-Process-Time-Ms"]) >= 0


def test_classify_rejects_empty_name() -> None:
    response = client.post("/classify", json={"name": "   "})
    assert response.status_code == 422


def test_request_body_size_limit() -> None:
    response = client.post(
        "/classify",
        content=b"x" * 1_000_001,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413
    assert response.headers["X-Request-ID"]

    def chunks():
        yield b"x" * 600_000
        yield b"x" * 600_000

    chunked_response = client.post(
        "/classify",
        content=chunks(),
        headers={"Content-Type": "application/json"},
    )
    assert chunked_response.status_code == 413


def test_classify_rejects_oversized_name_and_extra_fields() -> None:
    too_long = client.post("/classify", json={"name": "x" * 501})
    assert too_long.status_code == 422

    extra_field = client.post("/classify", json={"name": "Ноутбук", "debug": True})
    assert extra_field.status_code == 422


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


def test_batch_rejects_invalid_items() -> None:
    empty = client.post("/classify/batch", json={"items": []})
    assert empty.status_code == 422

    whitespace = client.post("/classify/batch", json={"items": [" \t "]})
    assert whitespace.status_code == 422

    too_long = client.post("/classify/batch", json={"items": ["x" * 501]})
    assert too_long.status_code == 422


def test_batch_rejects_more_than_100_items() -> None:
    response = client.post(
        "/classify/batch",
        json={"items": [f"Товар {i}" for i in range(101)]},
    )
    assert response.status_code == 422


def test_batch_status_returns_404_for_unknown_job() -> None:
    response = client.get("/classify/batch/unknown-job")
    assert response.status_code == 404


def test_admin_reindex_requires_key_and_ml_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(settings, "admin_api_key", "test-secret")
    monkeypatch.setattr(settings, "ml_enabled", True)
    monkeypatch.setattr("app.main._do_reindex", lambda: 1126)
    monkeypatch.setattr("app.main.importlib.util.find_spec", lambda _: object())

    unauthorized = client.post("/admin/reindex")
    assert unauthorized.status_code == 403

    accepted = client.post("/admin/reindex", headers={"X-Admin-API-Key": "test-secret"})
    assert accepted.status_code == 202
    body = accepted.json()
    assert body["status"] == "queued"

    status = client.get(
        body["status_url"],
        headers={"X-Admin-API-Key": "test-secret"},
    )
    assert status.status_code == 200
    assert status.json()["status"] == "completed"
    assert status.json()["indexed"] == 1126


def test_unexpected_classifier_error_returns_sanitized_500(monkeypatch) -> None:
    class BrokenClassifier:
        def classify(self, query: str):
            raise RuntimeError("internal test secret")

    monkeypatch.setattr("app.main._classifier_or_503", lambda: BrokenClassifier())
    response = client.post(
        "/classify",
        json={"name": "Ноутбук"},
        headers={"X-Request-ID": "safe-error-test"},
    )
    assert response.status_code == 500
    assert response.json()["detail"] == "Внутренняя ошибка сервера"
    assert "internal test secret" not in response.text
    assert response.headers["X-Request-ID"] == "safe-error-test"


def test_openapi_contains_public_endpoints() -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/classify" in paths
    assert "/classify/batch" in paths
    assert "/health/ready" in paths
    assert "/admin/reindex" in paths


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
        monkeypatch.setattr(settings, "classifier_backend", "catalog")
