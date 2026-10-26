"""BE-9: contract invariant «``ok``/``review`` never without ``code``» + explicit flags.

Pinned defects (docs/audit-2026-10-20.md, P1-12/P1-13/P1-14):

* six positions of appendix 4 have no OKPD2 code («Не указан») and used to be
  served as ``status="ok"`` with ``code=null`` — the schema now forbids that
  combination on every backend, and the position stays visible as a hint in
  ``alternatives``;
* the ``code_role="Исключение"`` row and OCR-doubt rows (``review_required``)
  were indistinguishable in the response — they now carry explicit
  ``code_role`` / ``catalog_review_required`` flags.

Flag *handling* is deliberately informational: whether exception positions are
excluded from answers and how OCR-doubt rows are processed are product and
catalog decisions (PO-1, DA-6, DA-7). These tests pin that the flags change
NOTHING about statuses and codes — only visibility.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from app.core.config import settings
from app.main import app
from app.schemas import ClassifyResponse
from app.services.classifier import CatalogClassifier, QdrantClassifier
from fastapi.testclient import TestClient
from pydantic import ValidationError

client = TestClient(app)

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG = REPO_ROOT / "data" / "data.csv"

MINI_CATALOG = (
    "record_id;appendix;position;name;okpd2;code_role;min_share_pct;"
    "max_foreign_share_pct;confirming_document;pdf_page_start;pdf_page_end;"
    "source_code_cell;review_required\n"
    "t-1-01;1;1;Сыры, продукты сырные и творог;10.51.40.120;Исключение;;;;76;76;"
    "10.51.40.120;True\n"
    "t-2-01;1;2;Ноутбуки портативные;26.20.11;Включён;;;;80;80;26.20.11;False\n"
    "t-3-01;1;3;Расходные материалы для аппаратов искусственной вентиляции лёгких;"
    ";Не указан;;;;90;90;;False\n"
)


# --- schema level ------------------------------------------------------------


def _response(**overrides) -> ClassifyResponse:
    payload = {
        "query": "тест",
        "code": "26.20.11",
        "title": "Компьютеры портативные",
        "confidence": 0.9,
        "confidence_raw": 0.9,
        "status": "ok",
        "latency_ms": 1.0,
    }
    payload.update(overrides)
    return ClassifyResponse(**payload)


@pytest.mark.parametrize("status", ["ok", "review"])
def test_schema_forbids_an_answer_without_code(status: str) -> None:
    with pytest.raises(ValidationError):
        _response(status=status, code=None)
    with pytest.raises(ValidationError):
        _response(status=status, code="   ")
    # The same check applies when persisted results are re-validated.
    with pytest.raises(ValidationError):
        ClassifyResponse.model_validate(
            {
                "query": "тест",
                "code": None,
                "confidence": 0.9,
                "status": status,
                "latency_ms": 1.0,
            }
        )


def test_schema_allows_not_found_without_code() -> None:
    result = _response(status="not_found", code=None, title=None, confidence=0.0)
    assert result.code is None


def test_flags_default_to_null_so_old_clients_see_no_change() -> None:
    result = _response()
    assert result.code_role is None
    assert result.catalog_review_required is None


# --- catalog backend ---------------------------------------------------------


def _mini_classifier(tmp_path: Path) -> CatalogClassifier:
    catalog = tmp_path / "mini.csv"
    catalog.write_text(MINI_CATALOG, encoding="utf-8")
    return CatalogClassifier(catalog, dense=None)


def test_codeless_position_is_not_found_but_stays_visible() -> None:
    """The exact case from the audit: «Мочеприемники и калоприемники» → ok/null."""
    response = client.post("/classify", json={"name": "Мочеприемники и калоприемники"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "not_found"
    assert body["code"] is None
    assert body["title"] is None
    assert body["confidence"] == 0.0
    assert body["confidence_raw"] == 0.0
    # The position itself is not lost — it stays the first hint, with source.
    assert body["alternatives"], "a demoted position must remain visible"
    hint = body["alternatives"][0]
    assert hint["title"] == "Мочеприемники и калоприемники"
    assert hint["code"] is None
    assert len(body["alternatives"]) <= 3  # contract: alternatives ≤ 3


def test_codeless_row_never_answers_ok_or_review(tmp_path: Path) -> None:
    classifier = _mini_classifier(tmp_path)
    result = classifier.classify(
        "Расходные материалы для аппаратов искусственной вентиляции лёгких"
    )
    assert result.status == "not_found"
    assert result.code is None
    assert any(
        alternative.title == "Расходные материалы для аппаратов искусственной вентиляции лёгких"
        and alternative.code is None
        for alternative in result.alternatives
    )


def test_exception_and_review_flags_are_explicit_but_change_nothing(tmp_path: Path) -> None:
    classifier = _mini_classifier(tmp_path)
    result = classifier.classify("Сыры, продукты сырные и творог")
    # The «Исключение» position still participates in answers (that policy is
    # DA-6/PO-1 territory) — but the response now says what the catalog says.
    assert result.code == "10.51.40.120"
    assert result.status in {"ok", "review"}
    assert result.code_role == "Исключение"
    assert result.catalog_review_required is True

    normal = classifier.classify("Ноутбуки портативные")
    assert normal.code == "26.20.11"
    assert normal.code_role == "Включён"
    assert normal.catalog_review_required is False


def test_review_required_flag_on_the_real_catalog() -> None:
    """A row visually verified against the PDF no longer carries an OCR flag."""
    classifier = CatalogClassifier(CATALOG, dense=None)
    result = classifier.classify("Нитрат натрия")  # verified against PDF page 44 on 2026-10-26
    assert result.code == "20.15.6"
    assert result.status in {"ok", "review"}  # handling unchanged — flag is informational
    assert result.catalog_review_required is False
    assert result.code_role == "Включён"


def test_duplicate_rows_merge_the_review_flag_conservatively(tmp_path: Path) -> None:
    catalog = tmp_path / "dupes.csv"
    catalog.write_text(
        MINI_CATALOG
        + "t-1-02;2;5;Сыры, продукты сырные и творог;10.51.40.120;Исключение;;;;77;77;"
        "10.51.40.120;False\n",
        encoding="utf-8",
    )
    classifier = CatalogClassifier(catalog, dense=None)
    result = classifier.classify("Сыры, продукты сырные и творог")
    # Any copy marked for re-verification marks the merged position.
    assert result.catalog_review_required is True


# --- API level: batch + CSV --------------------------------------------------


def test_batch_results_respect_the_invariant() -> None:
    items = [
        "Мочеприемники и калоприемники",
        "Ноутбук Lenovo",
        "Нитрат натрия",
        "абвгдеёжз qwerty",
    ]
    accepted = client.post("/classify/batch", json={"items": items})
    assert accepted.status_code == 202
    job_id = accepted.json()["job_id"]
    state = client.get(f"/classify/batch/{job_id}").json()
    assert state["status"] == "completed"
    assert len(state["results"]) == len(items)
    for result in state["results"]:
        if result["status"] in {"ok", "review"}:
            assert (result["code"] or "").strip(), result
        else:
            assert result["code"] is None


def test_csv_export_carries_the_catalog_flags() -> None:
    accepted = client.post("/classify/batch", json={"items": ["Нитрат натрия"]})
    job_id = accepted.json()["job_id"]
    response = client.get(f"/classify/batch/{job_id}.csv")
    assert response.status_code == 200
    rows = list(
        csv.reader(io.StringIO(response.content.decode("utf-8-sig")), delimiter=";")
    )
    header = rows[0]
    assert header[8] == "code_role"
    assert header[9] == "catalog_review_required"
    assert rows[1][8] == "Включён"
    assert rows[1][9] == "True"


# --- semantic backend (stubs, no server) -------------------------------------


def _hit(code: str, name: str, score: float, record_id: str = "", **payload):
    return SimpleNamespace(
        pp_code=code,
        name=name,
        score=score,
        position=payload.pop("position", "") or None,
        group=None,
        record_id=record_id,
        code_role=payload.pop("code_role", None),
        review_required=payload.pop("review_required", None),
    )


class _StubReranker:
    """Deterministic decisions: probability = semantic score, ok from 0.8."""

    threshold_review = 0.2
    uses_dense = False

    def rerank(self, query: str, candidates: list) -> list:
        ranked = sorted(candidates, key=lambda c: c.retrieval_score, reverse=True)
        for candidate in ranked:
            candidate.probability = candidate.retrieval_score
        return ranked

    def evaluate(self, query: str, ranked: list) -> tuple[float, str]:
        top = ranked[0].probability
        if top >= 0.8:
            status = "ok"
        elif top >= self.threshold_review:
            status = "review"
        else:
            status = "not_found"
        return top, status

    def display_confidence(self, raw: float) -> float:
        return raw


@pytest.fixture()
def qdrant_classifier(monkeypatch):
    monkeypatch.setattr(settings, "ml_enabled", True)
    monkeypatch.setattr(
        "ml.embedder.get_encoder", lambda: SimpleNamespace(tag="test-embedder", dim=8)
    )
    client_stub = SimpleNamespace(
        collection_exists=lambda name: True,
        count=lambda collection_name, exact: SimpleNamespace(count=2),
    )
    monkeypatch.setattr("ml.indexer.get_qdrant_client", lambda: client_stub)
    hits_by_query: dict[str, list] = {}
    monkeypatch.setattr(
        "ml.search.search_many",
        lambda queries, top_k=10: [hits_by_query.get(query, []) for query in queries],
    )
    classifier = QdrantClassifier()
    classifier._reranker = _StubReranker()
    return classifier, hits_by_query


def test_qdrant_flags_come_from_the_point_payload(qdrant_classifier) -> None:
    classifier, hits = qdrant_classifier
    hits["Сыры"] = [
        _hit(
            "10.51.40.120",
            "Сыры, продукты сырные и творог",
            0.95,
            record_id="2-454-02",
            code_role="Исключение",
            review_required=True,
        )
    ]
    result = classifier.classify("Сыры")
    assert result.status == "ok"
    assert result.code == "10.51.40.120"
    assert result.code_role == "Исключение"
    assert result.catalog_review_required is True


def test_qdrant_flags_fall_back_to_the_catalog_csv(qdrant_classifier) -> None:
    """Collections indexed before the payload keys existed still get flags."""
    classifier, hits = qdrant_classifier
    hits["Электрочайники"] = [_hit("27.51.24.110", "Электрочайники", 0.9, record_id="2-339-01")]
    result = classifier.classify("Электрочайники")
    assert result.status == "ok"
    assert result.code_role == "Включён"
    assert result.catalog_review_required is False


def test_qdrant_codeless_top_hit_is_demoted(qdrant_classifier) -> None:
    classifier, hits = qdrant_classifier
    hits["Мочеприемники"] = [
        _hit("", "Мочеприемники и калоприемники", 0.93, record_id="4-6-01"),
        _hit("32.50.13.190", "Контейнеры для заготовки крови", 0.5, record_id="4-2-01"),
    ]
    result = classifier.classify("Мочеприемники")
    assert result.status == "not_found"
    assert result.code is None
    assert result.alternatives[0].title == "Мочеприемники и калоприемники"
    assert result.alternatives[0].code is None
    # The coded runner-up stays available as a real suggestion.
    assert any(a.code == "32.50.13.190" for a in result.alternatives[1:])


# --- not_found hints: a lead is not a claim ---------------------------------


def test_not_found_keeps_lexically_anchored_leads() -> None:
    """«Нет в каталоге» must still hand the specialist ranked leads.

    The old filter kept runner-ups whose *pair* probability cleared
    ``threshold_review`` — but that number lives on the group-confidence scale,
    and ``not_found`` already means the group confidence is below it, so the
    filter could never pass: every out-of-catalog row in the labeled set came
    back with an empty ``alternatives`` list even when the right position ranked
    first. A lead now needs a content stem shared with the query.
    """
    from app.services.classifier import _distinct_alternatives

    from ml.reranker import RankedCandidate

    def candidate(code: str, title: str, probability: float) -> RankedCandidate:
        return RankedCandidate(
            code=code,
            title=title,
            clause_ref="1-1-01",
            retrieval_score=0.1,
            probability=probability,
            payload={},
        )

    ranked = [
        candidate("26.20.11", "Ноутбуки портативные", 0.05),
        # Shares «ноутбук» with the query but scores far below any threshold.
        candidate("26.20.12", "Ноутбуки промышленные", 0.01),
        # Dense near-miss with nothing in common (the "false friend" case).
        candidate("26.20.16", "Устройства ввода или вывода", 0.09),
        candidate("28.23", "Машины вычислительные", 0.08),
    ]
    hints = _distinct_alternatives(ranked, "not_found", {"ноутбук"})
    assert [hint.code for hint in hints] == ["26.20.12"]


def test_not_found_returns_no_hint_when_nothing_is_anchored() -> None:
    from app.services.classifier import _distinct_alternatives

    from ml.reranker import RankedCandidate

    ranked = [
        RankedCandidate(
            code="26.20.11",
            title="Ноутбуки портативные",
            clause_ref=None,
            retrieval_score=0.02,
            probability=0.02,
            payload={},
        )
    ]
    # «qwerty zxcvbn asdf»: no content stem of the query occurs in the catalog,
    # so inventing a lead would be worse than returning nothing.
    assert _distinct_alternatives(ranked, "not_found", {"qwerty"}) == []
