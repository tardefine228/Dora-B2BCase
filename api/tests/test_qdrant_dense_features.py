"""The semantic backend must feed its own score into the decision model.

Two defects are pinned here (BE-1):

* ``QdrantClassifier`` loaded ``reranker.json`` — an artifact fitted for the
  *lexical* shortlist — even when ``reranker.<embedder-tag>.json`` existed, so a
  model trained for the semantic backend could never be served;
* every candidate carried only ``record_id``, so ``dense_cosine``/``dense_gap``
  (and the group features derived from them) were always zero and no
  vector-feature artifact could influence a decision. The similarity was used
  for ordering only, i.e. exactly the part the reranker is supposed to refine.

The tests build the classifier through its real ``__init__`` with stubbed ML
dependencies: no model download and no Qdrant server are needed, while the tag
selection, the candidate payload and the ``/health/ready`` wiring are all real.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from app.core.config import settings
from app.services.classifier import QdrantClassifier, _qdrant_candidates

from ml import reranker as reranker_module
from ml.reranker import FEATURE_NAMES, GROUP_FEATURE_NAMES

EMBEDDER_TAG = "test-embedder"
HITS = (
    ("27.51.24.110", "Электрочайники", 0.83, "Приложение 2, позиция 339", "2-339-01"),
    (
        "27.32",
        "Провода и кабели электрические прочие",
        0.55,
        "Приложение 2, позиция 113",
        "2-113-01",
    ),
    ("26.60.12.111", "Электрокардиографы", 0.40, "Приложение 2, позиция 305", "2-305-01"),
)


def _hit(code: str, name: str, score: float, position: str = "", record_id: str = ""):
    return SimpleNamespace(
        pp_code=code,
        name=name,
        score=score,
        position=position,
        group=None,
        record_id=record_id,
    )


def _write_artifact(
    models_dir: Path, name: str, *, dense_weight: float, with_group_model: bool = False
) -> Path:
    """Minimal valid artifact; only the dense weights matter for these tests."""
    confidence_model = None
    if with_group_model:
        confidence_model = {
            "weights": {"top_dense_cosine": 2.0, "top_dense_gap": 1.0, "pair_probability": 1.0},
            "intercept": -2.0,
            "thresholds": {"ok": 0.8, "review": 0.2},
            "calibration": {"kind": "platt", "a": 1.0, "b": 0.0},
        }
    payload = {
        "version": reranker_module.ARTIFACT_VERSION,
        "model": "logistic",
        "feature_names": list(FEATURE_NAMES),
        "group_feature_names": list(GROUP_FEATURE_NAMES),
        "weights": {
            "lexical_score": 0.1,
            "dense_cosine": dense_weight,
            "dense_gap": dense_weight / 2,
        },
        "intercept": -1.0,
        "thresholds": {"ok": 0.8, "review": 0.2},
        "confidence_model": confidence_model,
        "known_stems": None,
        "stem_idf": None,
        "meta": {"test": True},
    }
    path = models_dir / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture()
def models_dir(tmp_path, monkeypatch) -> Path:
    """Isolated MODELS_DIR so the shipped artifacts never leak into a test."""
    directory = tmp_path / "models"
    directory.mkdir()
    monkeypatch.setattr(reranker_module, "MODELS_DIR", directory)
    return directory


@pytest.fixture()
def stub_ml(monkeypatch):
    """Replace encoder, Qdrant client and search with deterministic stand-ins."""
    monkeypatch.setattr(settings, "ml_enabled", True)
    monkeypatch.setattr(
        "ml.embedder.get_encoder",
        lambda: SimpleNamespace(tag=EMBEDDER_TAG, dim=8),
    )
    client = SimpleNamespace(
        collection_exists=lambda name: True,
        count=lambda collection_name, exact: SimpleNamespace(count=len(HITS)),
    )
    monkeypatch.setattr("ml.indexer.get_qdrant_client", lambda: client)
    hits = [_hit(*row) for row in HITS]
    monkeypatch.setattr(
        "ml.search.search_many",
        lambda queries, top_k=10: [list(hits) for _ in queries],
    )
    return hits


def test_tagged_artifact_of_the_active_embedder_is_used(models_dir, stub_ml) -> None:
    """A vector-feature artifact must win over the lexical one when it exists."""
    _write_artifact(models_dir, "reranker.json", dense_weight=0.0)
    _write_artifact(models_dir, f"reranker.{EMBEDDER_TAG}.json", dense_weight=2.0)

    classifier = QdrantClassifier()

    assert classifier._reranker.weights["dense_cosine"] == pytest.approx(2.0)
    assert classifier.vector_info["scoring"] == "vector-features"
    assert classifier.vector_info["reranker"] == f"reranker.{EMBEDDER_TAG}.json"


def test_missing_tagged_artifact_falls_back_to_the_lexical_one(models_dir, stub_ml, caplog) -> None:
    """Graceful degradation stays: a clear warning, a working answer."""
    _write_artifact(models_dir, "reranker.json", dense_weight=0.0)

    with caplog.at_level("WARNING", logger="dora.classifier"):
        classifier = QdrantClassifier()

    assert classifier._reranker.weights["dense_cosine"] == pytest.approx(0.0)
    assert classifier.vector_info["scoring"] == "lexical-reranker"
    assert classifier.vector_info["reranker"] == "reranker.json"
    assert "Нет реранкера для эмбеддера" in caplog.text
    assert classifier.classify("Электрочайники").code == "27.51.24.110"


def test_every_candidate_carries_its_semantic_score(stub_ml) -> None:
    """`dense`/`dense_top` are what ml.reranker.dense_values() reads."""
    candidates = _qdrant_candidates(stub_ml)

    assert [candidate.payload["dense"] for candidate in candidates] == pytest.approx(
        [0.83, 0.55, 0.40]
    )
    # The best cosine of the query across the catalog, same as fuse_shortlist().
    assert {candidate.payload["dense_top"] for candidate in candidates} == {0.83}
    # The semantic score doubles as the retrieval score (margin features).
    assert [candidate.retrieval_score for candidate in candidates] == pytest.approx(
        [0.83, 0.55, 0.40]
    )


def test_dense_features_reach_the_pair_model(models_dir, stub_ml) -> None:
    """Before the fix these were zeros: the artifact could not see the similarity."""
    _write_artifact(models_dir, f"reranker.{EMBEDDER_TAG}.json", dense_weight=2.0)
    classifier = QdrantClassifier()

    ranked = classifier._reranker.rerank("Электрочайники", _qdrant_candidates(stub_ml))
    by_title = {candidate.title: candidate for candidate in ranked}

    top = by_title["Электрочайники"]
    assert top.features["dense_cosine"] == pytest.approx(0.83)
    assert top.features["dense_gap"] == pytest.approx(0.0)

    runner_up = by_title["Провода и кабели электрические прочие"]
    assert runner_up.features["dense_cosine"] == pytest.approx(0.55)
    assert runner_up.features["dense_gap"] == pytest.approx(0.55 - 0.83)


def test_dense_features_reach_the_group_model(models_dir, stub_ml) -> None:
    """`top_dense_cosine`/`top_dense_gap` depend on the same payload keys."""
    _write_artifact(
        models_dir,
        f"reranker.{EMBEDDER_TAG}.json",
        dense_weight=2.0,
        with_group_model=True,
    )
    classifier = QdrantClassifier()
    ranked = classifier._reranker.rerank("Электрочайники", _qdrant_candidates(stub_ml))

    group = classifier._reranker.confidence_model
    features = reranker_module.extract_group_features("Электрочайники", ranked)

    assert features["top_dense_cosine"] == pytest.approx(0.83)
    assert features["top_dense_gap"] == pytest.approx(0.0)
    assert group.probability(features) > 0.0


def test_semantic_score_decides_the_ranking(models_dir, stub_ml) -> None:
    """The top answer follows the cosine, not the lexical overlap of the title."""
    _write_artifact(models_dir, f"reranker.{EMBEDDER_TAG}.json", dense_weight=2.0)
    classifier = QdrantClassifier()

    result = classifier.classify("чайник электрический")

    assert result.code == "27.51.24.110"
    assert result.status in {"ok", "review"}
    assert result.alternatives, "runner-up positions must still be reported"


def test_vector_info_reports_the_semantic_layer(models_dir, stub_ml) -> None:
    """`/health/ready` used to answer {"enabled": false} in the qdrant backend."""
    _write_artifact(models_dir, f"reranker.{EMBEDDER_TAG}.json", dense_weight=2.0)

    info = QdrantClassifier().vector_info

    assert info["enabled"] is True
    assert info["embedder"] == EMBEDDER_TAG
    assert info["dim"] == 8
    assert info["positions"] == len(HITS)
    assert info["store"] == "qdrant"
    assert info["scoring"] == "vector-features"


def test_health_ready_uses_the_semantic_vector_info(models_dir, stub_ml, monkeypatch) -> None:
    """The readiness endpoint must report what actually serves requests."""
    _write_artifact(models_dir, f"reranker.{EMBEDDER_TAG}.json", dense_weight=2.0)
    classifier = QdrantClassifier()
    monkeypatch.setattr(
        "app.main.get_classifier",
        lambda: SimpleNamespace(_classifier=classifier),
    )

    from app.main import readiness

    body = readiness()

    assert body["vector_search"] == classifier.vector_info
    assert body["vector_search"]["store"] == "qdrant"


def test_embedder_without_a_tag_is_tolerated(models_dir, monkeypatch) -> None:
    """Some backends may not expose `tag`; selection then falls back to lexical."""
    _write_artifact(models_dir, "reranker.json", dense_weight=0.0)
    monkeypatch.setattr(settings, "ml_enabled", True)
    monkeypatch.setattr("ml.embedder.get_encoder", lambda: SimpleNamespace(dim=8))
    monkeypatch.setattr(
        "ml.indexer.get_qdrant_client",
        lambda: SimpleNamespace(
            collection_exists=lambda name: True,
            count=lambda collection_name, exact: SimpleNamespace(count=1),
        ),
    )
    monkeypatch.setattr("ml.search.search_many", lambda queries, top_k=10: [[] for _ in queries])

    classifier = QdrantClassifier()

    assert classifier.vector_info["embedder"] is None
    assert classifier.vector_info["scoring"] == "lexical-reranker"


def test_lexical_switch_pins_the_shared_artifact(models_dir, stub_ml, monkeypatch, caplog) -> None:
    """Rollback must be a config change: no image edit, no rebuild."""
    _write_artifact(models_dir, "reranker.json", dense_weight=0.0)
    _write_artifact(models_dir, f"reranker.{EMBEDDER_TAG}.json", dense_weight=2.0)
    monkeypatch.setattr(settings, "semantic_reranker", "lexical")

    with caplog.at_level("WARNING", logger="dora.classifier"):
        classifier = QdrantClassifier()

    assert classifier._reranker.weights["dense_cosine"] == pytest.approx(0.0)
    assert classifier.vector_info["reranker"] == "reranker.json"
    assert classifier.vector_info["scoring"] == "lexical-reranker"
    assert "SEMANTIC_RERANKER=lexical" in caplog.text


def test_invalid_switch_value_is_rejected(monkeypatch) -> None:
    """A typo must fail loudly at startup, not silently pick a default.

    A fresh ``Settings()`` is built instead of reloading the module: reloading
    would replace the process-wide object that other modules already imported.
    """
    from app.core.config import Settings

    monkeypatch.setenv("SEMANTIC_RERANKER", "tagged")
    with pytest.raises(ValueError, match="SEMANTIC_RERANKER"):
        Settings()


def test_switch_defaults_to_auto() -> None:
    assert settings.semantic_reranker == "auto"
