"""The reranker must see the lexical retrieval score of every candidate.

``retrieval_margin`` is a trained feature: the training pairs carry the lexical
shortlist score, so the API has to hand the same score to the reranker. A
constant 0.0 (the previous behaviour) silently zeroes that feature in production.
"""

from __future__ import annotations

from pathlib import Path

from app.services.classifier import CatalogClassifier

from ml.reranker import RankedCandidate

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "data" / "data.csv"


class SpyReranker:
    """Records the candidates it is given and ranks them by retrieval score."""

    def __init__(self) -> None:
        self.calls: list[list[RankedCandidate]] = []

    def rerank(self, query: str, candidates: list[RankedCandidate]) -> list[RankedCandidate]:
        self.calls.append(list(candidates))
        for candidate in candidates:
            candidate.probability = candidate.retrieval_score
        return sorted(candidates, key=lambda c: c.probability, reverse=True)

    def evaluate(self, query: str, ranked: list[RankedCandidate]) -> tuple[float, str]:
        return ranked[0].probability if ranked else 0.0, "review"


def test_reranker_receives_lexical_scores_not_zeros() -> None:
    spy = SpyReranker()
    classifier = CatalogClassifier(CATALOG, reranker=spy)

    response = classifier.classify("Электрочайник")

    assert response.code is not None
    assert len(spy.calls) == 1
    scores = [candidate.retrieval_score for candidate in spy.calls[0]]
    assert len(scores) > 1
    assert max(scores) > 0.0
    # The shortlist is handed over best-first, so the scores are non-increasing.
    assert all(later <= earlier for earlier, later in zip(scores, scores[1:], strict=False))


def test_retrieval_score_matches_the_lexical_shortlist() -> None:
    from ml.features import lexical_score_prepared, prepare_candidate, stem
    from ml.lexicon import bridged_targets, expand_query

    spy = SpyReranker()
    classifier = CatalogClassifier(CATALOG, reranker=spy)
    query = "Ноутбук Lenovo"
    classifier.classify(query)

    # Same contract as CatalogClassifier._retrieve: weighted query coverage plus
    # the lexicon-bridge bonus for candidates holding the category word.
    prepared_query = prepare_candidate(expand_query(query), idf=classifier._idf)
    bridge_stems = {stem(target) for target in set(bridged_targets(query).values())}
    top = max(spy.calls[0], key=lambda c: c.retrieval_score)
    expected = max(
        min(
            1.0,
            lexical_score_prepared(prepared_query, row["prepared"])
            + (0.15 if bridge_stems & row["prepared"].stem_set else 0.0),
        )
        for row in classifier.index
    )
    assert top.retrieval_score == max(c.retrieval_score for c in spy.calls[0])
    assert abs(top.retrieval_score - expected) < 1e-9
