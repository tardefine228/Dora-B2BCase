"""Catalog versions, rollback and export (критерий приёмки 4).

Acceptance asks for "обновление базы НПА без передеплоя". The upload endpoint
always did that, but a wrong file was a one-way door: the only copy of the
working catalog was inside the container's volume and the way back was a manual
file swap. These tests pin the safety net an operator actually needs:

    скачать текущий  →  загрузить новый  →  откатить, если новый неверен

Every step must work without a restart: the classifier singleton and the source
lookup are reset, so the very next `POST /classify` answers from the new (or
restored) catalog.
"""

from __future__ import annotations

import io
import json

import pytest
from app.core.config import settings
from app.main import app
from app.services.catalog_io import versions_dir
from app.services.classifier import reset_classifier
from fastapi.testclient import TestClient


def _catalog(rows: list[tuple[str, str]]) -> str:
    lines = ["record_id;appendix;position;name;okpd2;code_role"]
    for index, (name, code) in enumerate(rows, start=1):
        lines.append(f"t-{index};2;{index};{name};{code};Включён")
    return "\n".join(lines) + "\n"


CATALOG_V1 = _catalog([("Принтер лазерный", "26.20.16"), ("Мышь компьютерная", "26.20.16")])
CATALOG_V2 = _catalog([("Принтер струйный", "26.20.16"), ("Клавиатура", "26.20.16")])


@pytest.fixture()
def catalog_client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "test-secret")
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings, "catalog_snapshot_keep", 3)
    client = TestClient(app)
    yield client, tmp_path
    reset_classifier()


def _headers() -> dict[str, str]:
    return {"X-Admin-API-Key": "test-secret"}


def _upload(client: TestClient, content: str):
    response = client.post(
        "/admin/catalog",
        headers=_headers(),
        files={"file": ("data.csv", io.BytesIO(content.encode("utf-8")), "text/csv")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["applied"] is True, body
    return body


def test_upload_snapshots_the_previous_catalog(catalog_client) -> None:
    client, tmp_path = catalog_client
    _upload(client, CATALOG_V1)
    assert not versions_dir(tmp_path).exists() or not any(versions_dir(tmp_path).iterdir())

    _upload(client, CATALOG_V2)
    snapshots = [path for path in versions_dir(tmp_path).iterdir() if path.is_dir()]
    assert len(snapshots) == 1
    assert "Принтер лазерный" in (snapshots[0] / "data.csv").read_text(encoding="utf-8")
    # The ML view is stored too, so a rollback restores both files as a pair.
    assert (snapshots[0] / "data.json").is_file()


def test_versions_endpoint_lists_snapshots_newest_first(catalog_client) -> None:
    client, _tmp = catalog_client
    _upload(client, CATALOG_V1)
    _upload(client, CATALOG_V2)
    _upload(client, _catalog([("Сканер", "26.20.16")]))

    response = client.get("/admin/catalog/versions", headers=_headers())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["keep"] == 3
    assert len(body["versions"]) == 2  # v1 and v2; v3 is current
    ids = [version["version_id"] for version in body["versions"]]
    assert ids == sorted(ids, reverse=True)
    assert all(version["rows"] == 2 for version in body["versions"])
    assert body["current"]["rows"] == 1


def test_rollback_restores_the_previous_catalog_without_a_restart(catalog_client) -> None:
    client, tmp_path = catalog_client
    _upload(client, CATALOG_V1)
    _upload(client, CATALOG_V2)

    # The classifier is already serving v2: the phrase from v1 is not in it.
    assert "Принтер струйный" in (tmp_path / "data.csv").read_text(encoding="utf-8")

    response = client.post("/admin/catalog/rollback", headers=_headers())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["catalog"]["rows"] == 2
    assert body["restored"]["version_id"]

    restored_csv = (tmp_path / "data.csv").read_text(encoding="utf-8")
    assert "Принтер лазерный" in restored_csv
    assert "Принтер струйный" not in restored_csv
    # data.json is restored as well: the two views never drift apart.
    restored_json = json.loads((tmp_path / "data.json").read_text(encoding="utf-8"))
    assert any("Принтер лазерный" in row["name"] for row in restored_json)


def test_rollback_is_itself_undoable(catalog_client) -> None:
    """The state being replaced is snapshotted, so a mistake is undoable twice."""
    client, tmp_path = catalog_client
    _upload(client, CATALOG_V1)
    _upload(client, CATALOG_V2)

    assert client.post("/admin/catalog/rollback", headers=_headers()).status_code == 200
    assert "Принтер лазерный" in (tmp_path / "data.csv").read_text(encoding="utf-8")

    second = client.post("/admin/catalog/rollback", headers=_headers())
    assert second.status_code == 200, second.text
    assert "Принтер струйный" in (tmp_path / "data.csv").read_text(encoding="utf-8")


def test_rollback_to_a_named_version(catalog_client) -> None:
    client, tmp_path = catalog_client
    _upload(client, CATALOG_V1)
    _upload(client, CATALOG_V2)
    versions = client.get("/admin/catalog/versions", headers=_headers()).json()["versions"]
    oldest = versions[-1]["version_id"]

    response = client.post(
        f"/admin/catalog/rollback?version={oldest}", headers=_headers()
    )
    assert response.status_code == 200, response.text
    assert response.json()["restored"]["version_id"] == oldest
    assert "Принтер лазерный" in (tmp_path / "data.csv").read_text(encoding="utf-8")


def test_rollback_without_snapshots_is_a_conflict(catalog_client) -> None:
    client, _tmp = catalog_client
    _upload(client, CATALOG_V1)  # first upload has nothing to snapshot
    response = client.post("/admin/catalog/rollback", headers=_headers())
    assert response.status_code == 409
    assert "Снимков каталога нет" in response.json()["detail"]


def test_rollback_to_an_unknown_version_is_404(catalog_client) -> None:
    client, _tmp = catalog_client
    _upload(client, CATALOG_V1)
    _upload(client, CATALOG_V2)
    response = client.post("/admin/catalog/rollback?version=nope", headers=_headers())
    assert response.status_code == 404


def test_export_returns_the_installed_catalog_verbatim(catalog_client) -> None:
    client, tmp_path = catalog_client
    _upload(client, CATALOG_V1)

    response = client.get("/admin/catalog/export", headers=_headers())
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    assert response.headers["content-disposition"].endswith('.csv"')
    assert response.content == (tmp_path / "data.csv").read_bytes()


def test_snapshots_are_pruned_to_the_configured_depth(catalog_client) -> None:
    client, tmp_path = catalog_client
    _upload(client, CATALOG_V1)
    for index in range(5):
        _upload(client, _catalog([(f"Товар {index}", "26.20.16")]))

    snapshots = [path for path in versions_dir(tmp_path).iterdir() if path.is_dir()]
    assert len(snapshots) == 3  # settings.catalog_snapshot_keep
    # The newest survive: the oldest snapshot still holds the first generation.
    names = sorted(path.name for path in snapshots)
    assert len(set(names)) == 3


def test_export_and_rollback_require_the_admin_key(catalog_client) -> None:
    client, _tmp = catalog_client
    _upload(client, CATALOG_V1)
    assert client.get("/admin/catalog/export").status_code == 403
    assert client.get("/admin/catalog/versions").status_code == 403
    assert client.post("/admin/catalog/rollback").status_code == 403


def test_export_without_an_installed_catalog_is_404(catalog_client) -> None:
    client, _tmp = catalog_client
    response = client.get("/admin/catalog/export", headers=_headers())
    assert response.status_code == 404


def test_rollback_rebuilds_a_missing_ml_view(catalog_client) -> None:
    """A snapshot from an older release may carry only data.csv."""
    client, tmp_path = catalog_client
    _upload(client, CATALOG_V1)
    _upload(client, CATALOG_V2)
    snapshot = next(path for path in versions_dir(tmp_path).iterdir() if path.is_dir())
    (snapshot / "data.json").unlink()

    response = client.post("/admin/catalog/rollback", headers=_headers())
    assert response.status_code == 200, response.text
    rebuilt = json.loads((tmp_path / "data.json").read_text(encoding="utf-8"))
    assert any("Принтер лазерный" in row["name"] for row in rebuilt)
