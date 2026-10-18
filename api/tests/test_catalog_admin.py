"""Admin catalog upload/validation endpoints (docs/catalog-update.md)."""

from __future__ import annotations

import io

import pytest
from app.core.config import settings
from app.main import app
from app.services.classifier import reset_classifier
from fastapi.testclient import TestClient

client = TestClient(app)

VALID_CSV = (
    "record_id;appendix;position;name;okpd2;code_role\n"
    "t-1;2;1;Тестовый товар А;01.11.1;Включён\n"
    "t-2;2;2;Тестовый товар Б;01.11.2;Включён\n"
)

BAD_CSV = (
    "record_id;appendix;position;name;okpd2;code_role\n"
    "t-1;2;1;;01.11.1;Включён\n"
    "t-2;2;2;Товар без кода;;Включён\n"
    "t-3;2;3;Товар с кривым кодом;ABC;Включён\n"
)


@pytest.fixture()
def admin_client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "test-secret")
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    yield client, tmp_path
    reset_classifier()


def _headers() -> dict:
    return {"X-Admin-API-Key": "test-secret"}


def test_catalog_requires_admin_key(admin_client) -> None:
    client, _tmp = admin_client
    assert client.get("/admin/catalog").status_code == 403
    assert client.post("/admin/catalog", files={"file": ("a.csv", b"x")}).status_code == 403


def test_admin_key_disabled_without_configuration(monkeypatch) -> None:
    monkeypatch.setattr(settings, "admin_api_key", "")
    response = client.get("/admin/catalog")
    assert response.status_code == 503


def test_dry_run_reports_errors_without_applying(admin_client) -> None:
    client, tmp_path = admin_client
    response = client.post(
        "/admin/catalog?dry_run=true",
        headers=_headers(),
        files={"file": ("data.csv", io.BytesIO(BAD_CSV.encode("utf-8")), "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["dry_run"] is True
    assert body["applied"] is False
    assert body["rows_total"] == 3
    # t-1 has an empty name, t-2 an empty code, t-3 a malformed code.
    assert body["rows_valid"] == 0
    assert any(issue["field"] == "name" for issue in body["errors"])
    assert any(issue["field"] == "okpd2" for issue in body["errors"])
    assert not (tmp_path / "data.csv").exists()


def test_upload_applies_and_replaces_catalog(admin_client) -> None:
    client, tmp_path = admin_client
    response = client.post(
        "/admin/catalog",
        headers=_headers(),
        files={"file": ("data.csv", io.BytesIO(VALID_CSV.encode("utf-8")), "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert body["rows_valid"] == 2
    assert body["errors"] == []
    assert body["catalog"]["rows"] == 2
    assert body["catalog"]["sha256"]

    # Both views (CSV + JSON) are regenerated so the classifier and ML agree.
    assert (tmp_path / "data.csv").is_file()
    assert (tmp_path / "data.json").is_file()

    info = client.get("/admin/catalog", headers=_headers()).json()
    assert info["rows"] == 2

    # The classifier picks the new catalog up immediately.
    result = client.post("/classify", json={"name": "Тестовый товар А"})
    assert result.status_code == 200
    assert result.json()["code"] == "01.11.1"


def test_upload_rejects_broken_files(admin_client) -> None:
    client, _tmp = admin_client
    response = client.post(
        "/admin/catalog",
        headers=_headers(),
        files={"file": ("data.csv", io.BytesIO(b"\x00\x01\x02"), "text/csv")},
    )
    assert response.status_code == 422

    missing_columns = client.post(
        "/admin/catalog",
        headers=_headers(),
        files={"file": ("data.csv", io.BytesIO(b"foo;bar\n1;2\n"), "text/csv")},
    )
    assert missing_columns.status_code == 422


def test_json_upload_is_accepted(admin_client) -> None:
    client, _tmp = admin_client
    payload = (
        '[{"record_id": "j-1", "appendix": 1, "position": "1",'
        ' "name": "Тестовый товар В", "okpd2": "02.20", "code_role": "Включён"}]'
    )
    response = client.post(
        "/admin/catalog",
        headers=_headers(),
        files={"file": ("data.json", io.BytesIO(payload.encode("utf-8")), "application/json")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["applied"] is True
    assert body["rows_valid"] == 1
