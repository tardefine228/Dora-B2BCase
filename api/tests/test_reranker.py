"""Unit tests for the calibrated reranker artifact (models/reranker.json)."""

from __future__ import annotations

import json

import pytest

from ml.reranker import (
    FEATURE_NAMES,
    GROUP_FEATURE_NAMES,
    HeuristicReranker,
    LogisticReranker,
    RankedCandidate,
    extract_features,
    extract_group_features,
    load_reranker,
)


def test_extract_features_shape_and_ranges() -> None:
    features = extract_features(
        "Ноутбук Lenovo",
        "Компьютеры портативные массой не более 10 кг, такие как ноутбуки",
        "26.20.11",
    )
    assert set(features) == set(FEATURE_NAMES)
    for name, value in features.items():
        assert 0.0 <= value <= 1.0 or name == "lexical_score", (name, value)


def test_trained_artifact_loads_with_thresholds() -> None:
    reranker = load_reranker()
    if isinstance(reranker, HeuristicReranker):
        pytest.skip("models/reranker.json не собран (python scripts/build_model.py)")
    assert isinstance(reranker, LogisticReranker)
    assert 0 < reranker.threshold_review < reranker.threshold_ok <= 1
    assert reranker.confidence_model is not None
    assert set(reranker.confidence_model.weights) == set(GROUP_FEATURE_NAMES)
    assert reranker.known_stems and len(reranker.known_stems) > 500
    assert reranker.idf and "обув" in reranker.idf


def test_rerank_orders_exact_match_first() -> None:
    reranker = load_reranker()
    candidates = [
        RankedCandidate(
            code="27.11.4",
            title="Трансформаторы электрические",
            clause_ref=None,
            retrieval_score=0.0,
        ),
        RankedCandidate(
            code="27.51.24.110", title="Электрочайники", clause_ref=None, retrieval_score=0.0
        ),
        RankedCandidate(
            code="27.51.23.130", title="Утюги электрические", clause_ref=None, retrieval_score=0.0
        ),
    ]
    ranked = reranker.rerank("Электрочайники", candidates)
    assert ranked[0].code == "27.51.24.110"
    confidence, status = reranker.evaluate("Электрочайники", ranked)
    assert confidence > 0.5
    assert status in {"ok", "review"}


def test_group_features_use_distinct_candidates() -> None:
    duplicate = RankedCandidate(
        code="26.20.11", title="Компьютеры портативные", clause_ref=None, retrieval_score=0.5
    )
    duplicate.probability = 0.9
    other = RankedCandidate(
        code="25.71.11", title="Ножи кухонные", clause_ref=None, retrieval_score=0.1
    )
    other.probability = 0.1
    features = extract_group_features("ноутбук", [duplicate, duplicate, other])
    # The duplicated catalog row must not zero out the margin.
    assert features["probability_margin"] == pytest.approx(0.8)


def test_garbage_query_gets_not_found() -> None:
    reranker = load_reranker()
    if isinstance(reranker, HeuristicReranker):
        pytest.skip("models/reranker.json не собран (python scripts/build_model.py)")
    candidates = [
        RankedCandidate(
            code="14.1", title="Одежда, кроме одежды из меха", clause_ref=None, retrieval_score=0.0
        )
    ]
    ranked = reranker.rerank("абвгдеёжз qwerty", candidates)
    confidence, status = reranker.evaluate("абвгдеёжз qwerty", ranked)
    assert status == "not_found"
    assert confidence < reranker.confidence_model.threshold_review


def test_artifact_json_is_stable_and_versioned(tmp_path) -> None:
    reranker = load_reranker()
    if isinstance(reranker, HeuristicReranker):
        pytest.skip("models/reranker.json не собран")
    path = tmp_path / "reranker.json"
    reranker.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] >= 1
    assert payload["model"] == "logistic"
    assert set(payload["weights"]) == set(FEATURE_NAMES)
    reloaded = LogisticReranker.load(path)
    assert reloaded.threshold_ok == reranker.threshold_ok
