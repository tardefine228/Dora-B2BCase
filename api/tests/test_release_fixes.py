import csv
from pathlib import Path
from types import SimpleNamespace

import pytest
from app.main import app
from app.services.classifier import CatalogClassifier, QdrantClassifier
from app.services.source_lookup import SourceLookup, SourceUnavailableError
from fastapi.testclient import TestClient

from ml.preprocessing import clean_query


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  • Ноутбук\u00a0Lenovo   16 ГБ ", "Ноутбук Lenovo 16 ГБ"),
        ("1. Ноутбук без ОС", "Ноутбук без ОС"),
        ("26.20.11", "26.20.11"),
        ("Срочно, пожалуйста, найдите", "найдите"),
        ("Поставка: Кабель USB-C 2 м, не экранированный", "Кабель USB-C 2 м, не экранированный"),
        ("Стул\u200b медицинский", "Стул медицинский"),
        ("Поставка товаров: 1. Кабель", "Кабель"),
        ("Срочность", "Срочность"),
        ("Пожалуйста, " * 10 + "Стол", "Стол"),
    ],
)
def test_safe_cleanup(raw, expected):
    assert clean_query(raw) == expected
    assert clean_query(expected) == expected


def test_excluded_code_never_becomes_positive():
    classifier = CatalogClassifier(Path("data/data.csv"))
    result = classifier.classify("10.51.40.120")
    assert result.status == "not_found"
    assert result.code is None
    assert all(row["okpd2"] for row in classifier.index)
    candidates = classifier._retrieve("Сыры 10.51.40.120", set())
    assert all(not row.get("excluded_codes") for _, row in candidates)


def test_source_html_and_json_are_both_available():
    client = TestClient(app)
    response = client.get("/source/2-339-01?format=html")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Электрочайники" in response.text
    assert "не официальная" in response.text
    assert client.get("/source/2-339-01").json()["record_id"] == "2-339-01"


def test_missing_page_does_not_show_unrelated_text(tmp_path):
    with (tmp_path / "data.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["record_id", "name", "okpd2", "pdf_page_start"], delimiter=";"
        )
        writer.writeheader()
        writer.writerow(dict(record_id="test", name="Товар", okpd2="13.2", pdf_page_start=99))
    (tmp_path / "source_text.txt").write_text("=== СТРАНИЦА 1 ===\nЧужой текст", encoding="utf-8")
    with pytest.raises(SourceUnavailableError):
        SourceLookup(tmp_path).fragment("test")


def test_stale_model_never_claims_calibrated_confidence(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("name;okpd2\nКабель;27.32\n", encoding="utf-8")
    reranker = SimpleNamespace(
        meta={"catalog_sha256": "old"},
        confidence_model=True,
        rerank=lambda query, candidates: candidates,
        evaluate=lambda query, ranked: (0.99, "ok"),
        display_confidence=lambda score: 0.999,
    )
    result = CatalogClassifier(path, reranker=reranker).classify("Кабель")
    assert result.confidence_kind == "uncalibrated"
    assert result.status == "review"
    assert result.confidence == 0.99


def test_old_qdrant_payload_cannot_reintroduce_an_excluded_code():
    classifier = QdrantClassifier.__new__(QdrantClassifier)
    classifier._catalog = CatalogClassifier(Path("data/data.csv"))
    classifier._reranker = None
    hit = SimpleNamespace(record_id="2-454-02", pp_code="10.51.40.120", name="Сыры", score=0.99)
    classifier._search_many = lambda queries, top_k: [[hit] for _ in queries]
    assert classifier.classify("Сыры").status == "not_found"
    assert classifier.classify("10.51.40.120").status == "not_found"


def test_source_html_escapes_untrusted_content():
    from app.schemas import SourceFragment
    from app.services.source_view import render_source

    html = render_source(
        SourceFragment(
            record_id="x",
            name="<script>alert(1)</script>",
            text="<script>alert(1)</script>",
            source_file="source_text.txt",
        )
    )
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
