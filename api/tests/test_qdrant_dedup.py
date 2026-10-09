"""Alternatives in the Qdrant branch must not repeat the same position/code.

Qdrant stores one point per catalog row, and the same position often appears in
several appendices, so a raw top-10 regularly contains the same code 2-3 times.
The test drives ``QdrantClassifier.classify_many`` with a stubbed search so no
Qdrant client or embedder is needed.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.services.classifier import (
    CatalogClassifier,
    QdrantClassifier,
    _distinct_alternatives,
)

from ml.reranker import RankedCandidate


def _hit(
    code: str, name: str, score: float, position: str = "", record_id: str = ""
) -> SimpleNamespace:
    return SimpleNamespace(
        pp_code=code,
        name=name,
        score=score,
        position=position,
        group=None,
        record_id=record_id,
    )


class _StubReranker:
    """Keeps retrieval order; confidence = top retrieval score."""

    threshold_ok = 0.8
    threshold_review = 0.2

    def rerank(self, query: str, candidates: list[RankedCandidate]) -> list[RankedCandidate]:
        for candidate in candidates:
            candidate.probability = candidate.retrieval_score
        candidates.sort(key=lambda item: -item.probability)
        return candidates

    def evaluate(self, query: str, ranked: list[RankedCandidate]) -> tuple[float, str]:
        return ranked[0].probability, "review"


def _qdrant_classifier(hits: list[SimpleNamespace]) -> QdrantClassifier:
    classifier = QdrantClassifier.__new__(QdrantClassifier)
    classifier._search_many = lambda queries, top_k=10: [hits for _ in queries]
    classifier._reranker = _StubReranker()
    return classifier


def test_qdrant_alternatives_have_distinct_codes() -> None:
    hits = [
        _hit("26.20.11", "Компьютеры портативные", 0.9, "Приложение 2", "2-195-01"),
        _hit("26.20.11", "Компьютеры портативные", 0.89, "Приложение 3"),
        _hit("26.20.11", "Компьютеры портативные", 0.88, "Приложение 1"),
        _hit("26.20.14", "Машины вычислительные электронные цифровые", 0.5, "", "2-196-01"),
        _hit("26.20.14", "Машины вычислительные электронные цифровые", 0.49),
        _hit("26.20.17", "Мониторы и проекторы", 0.4),
        _hit("26.30.22", "Аппараты телефонные", 0.3),
        _hit("27.40.1", "Лампы", 0.25),
    ]
    response = _qdrant_classifier(hits).classify("ноутбук")

    assert response.code == "26.20.11"
    codes = [alternative.code for alternative in response.alternatives]
    assert codes == ["26.20.14", "26.20.17", "26.30.22"]
    assert len(codes) == len(set(codes))
    assert response.code not in codes
    assert response.clause_url == "/source/2-195-01"
    assert [alternative.clause_url for alternative in response.alternatives] == [
        "/source/2-196-01",
        None,
        None,
    ]


def test_qdrant_not_found_keeps_only_candidates_above_review() -> None:
    hits = [
        _hit("15.2", "Обувь", 0.25),
        _hit("14.1", "Одежда", 0.22),
        _hit("14.1", "Одежда", 0.21),
        _hit("13.2", "Ткани", 0.05),
    ]
    classifier = _qdrant_classifier(hits)
    classifier._reranker.evaluate = lambda query, ranked: (0.0, "not_found")  # type: ignore[method-assign]
    response = classifier.classify("что-то неизвестное")

    assert response.status == "not_found"
    assert response.code is None
    assert [alternative.code for alternative in response.alternatives] == ["14.1"]


def test_distinct_alternatives_shared_by_both_backends() -> None:
    ranked = [
        RankedCandidate(code="A", title="a", clause_ref=None, retrieval_score=0.0, probability=0.9),
        RankedCandidate(code="A", title="a", clause_ref=None, retrieval_score=0.0, probability=0.8),
        RankedCandidate(code="B", title="b", clause_ref=None, retrieval_score=0.0, probability=0.7),
        RankedCandidate(code="B", title="b", clause_ref=None, retrieval_score=0.0, probability=0.6),
        RankedCandidate(code="C", title="c", clause_ref=None, retrieval_score=0.0, probability=0.5),
        RankedCandidate(code="D", title="d", clause_ref=None, retrieval_score=0.0, probability=0.4),
        RankedCandidate(code="E", title="e", clause_ref=None, retrieval_score=0.0, probability=0.3),
    ]
    alternatives = _distinct_alternatives(ranked, "review", 0.2)
    assert [alternative.code for alternative in alternatives] == ["B", "C", "D"]
    # The catalog backend uses the very same helper.
    assert CatalogClassifier._rank_to_response.__code__.co_names.count("_distinct_alternatives")
