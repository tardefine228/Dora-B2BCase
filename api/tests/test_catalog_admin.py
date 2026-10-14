"""Admin catalog upload/validation endpoints (docs/catalog-update.md)."""

from __future__ import annotations

import io
import json
import zipfile

import pytest
from app.core.config import settings
from app.main import app
from app.services.classifier import reset_classifier
from app.services.source_lookup import reset_source_lookup
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
    reset_source_lookup()


def _headers() -> dict:
    return {"X-Admin-API-Key": "test-secret"}


def test_catalog_requires_admin_key(admin_client) -> None:
    client, _tmp = admin_client
    assert client.get("/admin/catalog").status_code == 403
    assert client.post("/admin/catalog", files={"file": ("a.csv", b"x")}).status_code == 403
    assert client.post("/admin/models/reload").status_code == 403


def test_model_reload_is_available_with_admin_key(admin_client):
    client, _ = admin_client
    response = client.post("/admin/models/reload", headers=_headers())
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert "model_catalog_current" in response.json()


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


def _bundle(*, page=1, url="https://example.org/decree", extra=None):
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "data.csv",
            "record_id;appendix;position;name;okpd2;code_role;pdf_page_start;pdf_page_end\n"
            f"test;2;1;Тестовый товар;26.20.11.110;Включён;{page};{page}\n",
        )
        archive.writestr("source_text.txt", "=== СТРАНИЦА 1 ===\nТестовый товар 26.20.11.110")
        archive.writestr(
            "metadata.json", json.dumps({"edition": "Тестовая редакция", "source_url": url})
        )
        if extra:
            archive.writestr(extra, "bad")
    return target.getvalue()


def test_bundle_update_binds_source_and_accepts_full_okpd2(admin_client):
    client, tmp = admin_client
    response = client.post(
        "/admin/catalog", headers=_headers(), files={"file": ("catalog.zip", _bundle())}
    )
    assert response.status_code == 200
    assert response.json()["applied"]
    source = client.get("/source/test").json()
    assert source["edition"] == "Тестовая редакция"
    assert source["source_url"] == "https://example.org/decree"
    assert source["pages"] == {"start": 1, "end": 1}
    assert "Тестовый товар" in source["text"]
    # An out-of-band modification must not return a source from the old pair.
    (tmp / "source_text.txt").write_text("=== СТРАНИЦА 1 ===\nДругой документ", encoding="utf-8")
    assert client.get("/source/test").status_code == 503


def test_plain_update_disables_previous_source(admin_client):
    client, _ = admin_client
    assert client.post(
        "/admin/catalog", headers=_headers(), files={"file": ("catalog.zip", _bundle())}
    ).json()["applied"]
    with zipfile.ZipFile(io.BytesIO(_bundle())) as archive:
        csv = archive.read("data.csv")
    assert client.post(
        "/admin/catalog", headers=_headers(), files={"file": ("data.csv", csv)}
    ).json()["applied"]
    assert client.get("/source/test").status_code == 503


@pytest.mark.parametrize("bundle", [_bundle(page=99), _bundle(extra="../data.csv")])
def test_invalid_bundle_is_not_installed(admin_client, bundle):
    client, tmp = admin_client
    response = client.post(
        "/admin/catalog", headers=_headers(), files={"file": ("catalog.zip", bundle)}
    )
    assert response.status_code == 422
    assert not (tmp / "data.csv").exists()


def test_dry_run_does_not_change_source_and_rejects_unsafe_link(admin_client):
    client, tmp = admin_client
    response = client.post(
        "/admin/catalog?dry_run=true",
        headers=_headers(),
        files={"file": ("catalog.zip", _bundle())},
    )
    assert response.status_code == 200 and not response.json()["applied"]
    assert not list(tmp.iterdir())
    response = client.post(
        "/admin/catalog",
        headers=_headers(),
        files={"file": ("catalog.zip", _bundle(url="javascript:alert(1)"))},
    )
    assert response.json()["applied"]
    assert client.get("/source/test").json()["source_url"] is None
    assert "javascript:" not in client.get("/source/test?format=html").text
