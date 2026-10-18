"""``clause_url`` and the ``GET /source/{record_id}`` endpoint (source of record)."""

import pytest
from app.main import app
from app.services.source_lookup import SourceLookup, SourceUnavailableError, clause_url_for
from fastapi.testclient import TestClient

client = TestClient(app)

KETTLE_RECORD = "2-339-01"
KETTLE_PAGES = (64, 64)


def test_clause_url_is_filled_for_catalog_answers() -> None:
    response = client.post("/classify", json={"name": "Электрочайники"})
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == "27.51.24.110"
    assert body["status"] == "ok"
    assert body["clause_url"] == f"/source/{KETTLE_RECORD}"


def test_clause_url_is_filled_for_direct_code_lookup() -> None:
    response = client.post("/classify", json={"name": "14.1"})
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == "14.1"
    assert body["clause_url"] == "/source/1-3-01"


def test_alternatives_carry_clause_url() -> None:
    response = client.post("/classify", json={"name": "Монитор 24 дюйма"})
    assert response.status_code == 200
    alternatives = response.json()["alternatives"]
    assert alternatives, "ожидались альтернативные позиции"
    assert all(alt["clause_url"].startswith("/source/") for alt in alternatives)


def test_source_endpoint_returns_decree_fragment() -> None:
    response = client.get(f"/source/{KETTLE_RECORD}")
    assert response.status_code == 200
    body = response.json()
    assert body["record_id"] == KETTLE_RECORD
    assert body["pages"] == {"start": KETTLE_PAGES[0], "end": KETTLE_PAGES[1]}
    assert body["clause_ref"] == "Приложение 2, позиция 339"
    assert "339. Электрочайники" in body["text"]
    assert body["okpd2"] == "27.51.24.110"
    assert body["truncated"] is False
    assert body["clause_url"] == f"/source/{KETTLE_RECORD}"
    assert "max-age" in response.headers["Cache-Control"]


def test_source_endpoint_returns_404_for_unknown_record() -> None:
    response = client.get("/source/does-not-exist")
    assert response.status_code == 404
    assert "не найдена" in response.json()["detail"]


def test_source_lookup_without_source_text_reports_unavailable(tmp_path) -> None:
    lookup = SourceLookup(tmp_path)
    # Without a catalog there is nothing to resolve at all.
    with pytest.raises(SourceUnavailableError):
        lookup.fragment(KETTLE_RECORD)

    (tmp_path / "data.csv").write_text(
        "record_id;name;okpd2;appendix;position;pdf_page_start;pdf_page_end\n"
        f"{KETTLE_RECORD};Электрочайники;27.51.24.110;2;339;64;64\n",
        encoding="utf-8",
    )
    # An unknown id is answered before the (missing) decree text is needed.
    assert lookup.fragment("unknown-id") is None
    with pytest.raises(SourceUnavailableError) as exc_info:
        lookup.fragment(KETTLE_RECORD)
    assert "source_text.txt" in str(exc_info.value)


def test_clause_url_for_ignores_empty_ids() -> None:
    assert clause_url_for(None) is None
    assert clause_url_for("  ") is None
    assert clause_url_for("1-1-01") == "/source/1-1-01"
