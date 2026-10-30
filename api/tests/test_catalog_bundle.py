import io
import json
import zipfile

import pytest
from app.core.config import settings
from app.main import app
from app.services import catalog_io
from app.services.catalog_bundle import publish
from app.services.source_lookup import get_source_lookup, reset_source_lookup
from fastapi.testclient import TestClient

from ml.catalog_revision import active_catalog_dir


def bundle(text="Первый текст", page=1, extra=None):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("data.csv", "record_id;name;okpd2;pdf_page_start;pdf_page_end\n"
                         f"r1;Товар;26.20.11;{page};{page}\n")
        archive.writestr("source_text.txt", f"=== СТРАНИЦА 1 ===\n{text}")
        archive.writestr("report.json", json.dumps({"source": "test.pdf"}))
        if extra:
            archive.writestr(extra, "x")
    return stream.getvalue()


def install(root, text):
    parsed = catalog_io.parse_catalog_bytes("edition.zip", bundle(text))
    return catalog_io.write_catalog(catalog_io.normalize_rows(parsed), root,
                                    source_files=parsed.source_files)


def test_roundtrip_and_rollback_restore_both_files(tmp_path):
    install(tmp_path, "Первая редакция")
    old = active_catalog_dir(tmp_path)
    lookup = get_source_lookup(tmp_path)
    assert "Первая" in lookup.fragment("r1").text
    install(tmp_path, "Вторая редакция")
    assert old != active_catalog_dir(tmp_path)
    assert "Первая" in lookup.fragment("r1").text  # pinned reader
    assert "Вторая" in get_source_lookup(tmp_path).fragment("r1").text
    catalog_io.restore_version(tmp_path, keep=1)
    assert "Первая" in get_source_lookup(tmp_path).fragment("r1").text
    assert "source_text.txt" in catalog_io.list_versions(tmp_path)[0].files
    reset_source_lookup()


@pytest.mark.parametrize("content", [bundle(page=2), bundle(extra="../evil"), b"not a zip"])
def test_invalid_bundle_is_rejected(content):
    with pytest.raises(ValueError):
        catalog_io.parse_catalog_bytes("edition.zip", content)


def test_failed_write_does_not_publish(tmp_path, monkeypatch):
    install(tmp_path, "Старый")
    old = active_catalog_dir(tmp_path)
    original = catalog_io._atomic_write

    def fail(path, content):
        if path.name == "source_text.txt":
            raise OSError("disk full")
        original(path, content)

    monkeypatch.setattr(catalog_io, "_atomic_write", fail)
    with pytest.raises(OSError):
        publish(tmp_path, {"data.csv": b"new", "source_text.txt": b"new"})
    assert active_catalog_dir(tmp_path) == old


def test_csv_update_preserves_source_after_bundle(tmp_path):
    install(tmp_path, "Исходный текст")
    parsed = catalog_io.parse_catalog_bytes("data.csv", b"name;okpd2\nNew;26.20.11\n")
    catalog_io.write_catalog(catalog_io.normalize_rows(parsed), tmp_path)
    assert "Исходный" in (active_catalog_dir(tmp_path) / "source_text.txt").read_text()


def test_api_dry_run_upload_and_rollback(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings, "admin_api_key", "bundle-test")
    client = TestClient(app)
    headers = {"X-Admin-API-Key": "bundle-test"}
    files = {"file": ("edition.zip", bundle())}
    assert client.post("/admin/catalog", files=files).status_code == 403
    r = client.post("/admin/catalog?dry_run=true", files=files, headers=headers)
    assert r.status_code == 200 and not r.json()["applied"]
    assert not (tmp_path / "active-catalog.json").exists()
    r = client.post("/admin/catalog", files=files, headers=headers)
    assert r.status_code == 200 and r.json()["applied"]
    assert "Первый" in get_source_lookup(tmp_path).fragment("r1").text
    client.post("/admin/catalog", files={"file": ("edition.zip", bundle("Второй"))},
                headers=headers).raise_for_status()
    client.post("/admin/catalog/rollback", headers=headers).raise_for_status()
    assert "Первый" in get_source_lookup(tmp_path).fragment("r1").text
    reset_source_lookup()


def test_classifier_and_index_loader_follow_revision(tmp_path, monkeypatch):
    from app.services.classifier import _resolve_catalog_path

    from ml import catalog_loader

    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings, "catalog_path", "")
    monkeypatch.setattr(catalog_loader, "RAW_DATA_DIR", tmp_path)
    install(tmp_path, "Первая")
    first = _resolve_catalog_path()
    assert first.parent == active_catalog_dir(tmp_path)
    assert catalog_loader.load_catalog()[0]["record_id"] == "r1"
    install(tmp_path, "Вторая")
    assert _resolve_catalog_path() != first
    assert first.is_file()


def test_real_parser_output_can_be_packaged():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "data"
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("data.csv", "source_text.txt", "report.json"):
            archive.write(root / name, name)
    parsed = catalog_io.parse_catalog_bytes("edition.zip", stream.getvalue())
    assert len(parsed.rows) > 100


def test_legacy_rollback_does_not_keep_new_source(tmp_path):
    parsed = catalog_io.parse_catalog_bytes("old.csv", b"name;okpd2\nOld;26.20.11\n")
    catalog_io.write_catalog(catalog_io.normalize_rows(parsed), tmp_path)
    install(tmp_path, "Новая редакция")
    catalog_io.restore_version(tmp_path)
    current = active_catalog_dir(tmp_path)
    assert not (current / "source_text.txt").exists()
    assert b"Old" in catalog_io.export_catalog_csv(tmp_path)
