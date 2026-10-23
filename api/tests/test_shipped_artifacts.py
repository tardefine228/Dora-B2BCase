"""The Docker image has an empty ./models mount: shipped artifacts must still be used."""

from pathlib import Path

from ml import config, embedder, reranker


def test_shipped_artifacts_exist():
    assert (config.ARTIFACTS_DIR / reranker.RERANKER_ARTIFACT_NAME).is_file()
    assert (config.ARTIFACTS_DIR / embedder.TFIDF_ARTIFACT_NAME).is_file()


def test_empty_models_dir_falls_back_to_shipped_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(reranker, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(embedder, "MODELS_DIR", tmp_path)
    assert reranker.reranker_artifact_path().parent == config.ARTIFACTS_DIR
    assert embedder.tfidf_artifact_path().parent == config.ARTIFACTS_DIR
    assert isinstance(reranker.load_reranker(), reranker.LogisticReranker)


def test_models_dir_overrides_shipped_artifacts(tmp_path, monkeypatch):
    custom = tmp_path / reranker.RERANKER_ARTIFACT_NAME
    custom.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(reranker, "MODELS_DIR", Path(tmp_path))
    assert reranker.reranker_artifact_path() == custom


# --- vector-feature artifact and the gate report that let it ship ----------------------
#
# The shipped lexical ``reranker.json`` predates the current trainer and is *not*
# reproducible from this repository (it stores 29277 pairs and thresholds
# 0.621/0.097 while a fresh build stores 27858 pairs and 0.306/0.227). It stays as
# is until the product owner picks a working point; see eval/experiments.md, exp. 17.
# The tfidf pair below is reproducible and is what the vector path actually loads.

import hashlib  # noqa: E402
import json  # noqa: E402

DENSE_ARTIFACT = "reranker.tfidf.json"
GATE_REPORT = "dense-gate.json"


def _dense_artifact() -> dict:
    return json.loads((config.ARTIFACTS_DIR / DENSE_ARTIFACT).read_text(encoding="utf-8"))


def test_shipped_dense_artifact_actually_uses_vector_features():
    artifact = _dense_artifact()
    assert artifact["weights"]["dense_cosine"] != 0.0
    assert artifact["weights"]["dense_gap"] != 0.0
    assert {"dense_cosine", "dense_gap"} <= set(artifact["feature_names"])
    confidence = artifact["confidence_model"]
    assert any("dense" in name for name in confidence["weights"])


def test_dense_artifact_is_found_by_the_active_embedder_tag(tmp_path, monkeypatch):
    monkeypatch.setattr(reranker, "MODELS_DIR", tmp_path)  # empty ./models, as in the image
    assert reranker.reranker_artifact_path("tfidf").name == DENSE_ARTIFACT
    loaded = reranker.load_reranker(tag="tfidf")
    assert isinstance(loaded, reranker.LogisticReranker)
    assert loaded.weights["dense_cosine"] == _dense_artifact()["weights"]["dense_cosine"]


def test_gate_report_matches_the_shipped_artifact_and_kept_it():
    report = json.loads((config.ARTIFACTS_DIR / GATE_REPORT).read_text(encoding="utf-8"))
    artifact_bytes = (config.ARTIFACTS_DIR / DENSE_ARTIFACT).read_bytes()

    assert report["artifact"] == DENSE_ARTIFACT
    assert report["artifact_sha256"] == hashlib.sha256(artifact_bytes).hexdigest()
    assert report["keep"] is True and report["reasons"] == []
    # The baseline must have been produced by the same code, not the stale shipped file.
    assert report["baseline"]["trained_in_run"] is True

    deltas = {row["metric"]: row["delta_correct"] for row in report["rows"]}
    assert deltas["top1_expected"] == 3
    assert deltas["top3_expected"] == 4
    assert deltas["top1_all"] == 3
    assert deltas["correct_silence"] == 0
    assert deltas["exact_top1"] == 0


def test_shipped_tfidf_embedder_is_fitted_on_the_current_catalog():
    """A refit on today's catalog must reproduce the shipped document frequencies.

    The file once carried a fit from 2026-10-18, before the catalog cleanup, which
    silently changed the vector space under every tfidf-based artifact.
    """
    from ml.catalog_loader import item_to_text, load_catalog

    shipped = json.loads((config.ARTIFACTS_DIR / embedder.TFIDF_ARTIFACT_NAME).read_text())
    current = embedder.TfidfEncoder()
    current.fit([item_to_text(item) for item in load_catalog()])

    assert shipped["doc_count"] == current.doc_count
    assert shipped["dim"] == current.dim
    assert shipped["doc_frequencies"] == current.doc_frequencies


def test_dense_artifact_was_trained_by_the_current_trainer():
    """Cheap guard against shipping an artifact from an older trainer.

    The shipped lexical ``reranker.json`` is exactly this case (it has no
    ``split_strategy``), which is why the gate retrains its baseline instead of
    trusting whatever file is in the models directory.
    """
    meta = _dense_artifact()["meta"]
    assert meta["seed"] == 1875
    assert meta["split_strategy"] == "normalized_query_groups"
    assert meta["confidence_candidates"] == "serving_retrieval_no_forced_positive"
