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


# --- shipped artifacts, their provenance and the gate that let the dense one ship -------
#
# Both rerankers now come from the same trainer: ``reranker.json`` (lexical) is
# reproducible byte for byte, ``reranker.tfidf.json`` (vector features) passes the gate
# in ``eval/dense_gate.py`` and is what the catalog backend actually loads when the
# tfidf embedder is active. The retired, non-reproducible artifact is documented in
# ml/artifacts/README.md and can be recovered from git; see eval/experiments.md, exp. 17.

import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402

ARTIFACTS_README = config.ARTIFACTS_DIR / "README.md"
DENSE_ARTIFACT = "reranker.tfidf.json"
GATE_REPORT = "dense-gate.json"


def _load(name: str) -> dict:
    return json.loads((config.ARTIFACTS_DIR / name).read_text(encoding="utf-8"))


def _dense_artifact() -> dict:
    return _load(DENSE_ARTIFACT)


def _sha256(name: str) -> str:
    return hashlib.sha256((config.ARTIFACTS_DIR / name).read_bytes()).hexdigest()


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
    assert report["artifact"] == DENSE_ARTIFACT
    assert report["artifact_sha256"] == _sha256(DENSE_ARTIFACT)
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


def test_both_rerankers_were_trained_by_the_current_trainer():
    """Cheap guard against shipping an artifact from an older trainer.

    The retired lexical file had no ``split_strategy`` and could not be rebuilt at
    all; both shipped rerankers must carry the current trainer's provenance.
    """
    for name in (reranker.RERANKER_ARTIFACT_NAME, DENSE_ARTIFACT):
        meta = _load(name)["meta"]
        assert meta["seed"] == 1875, name
        assert meta["split_strategy"] == "normalized_query_groups", name
        assert meta["confidence_candidates"] == "serving_retrieval_no_forced_positive", name


def test_shipped_lexical_artifact_has_no_vector_features():
    artifact = _load(reranker.RERANKER_ARTIFACT_NAME)
    assert artifact["weights"]["dense_cosine"] == 0.0
    assert artifact["weights"]["dense_gap"] == 0.0


def test_gate_baseline_is_the_shipped_lexical_artifact():
    """The gate retrains its baseline; that file is now shipped, so the two must agree.

    If they diverge, either the artifact was rebuilt without updating the report or
    the trainer stopped being deterministic — both are worth failing the build for.
    """
    report = json.loads((config.ARTIFACTS_DIR / GATE_REPORT).read_text(encoding="utf-8"))
    assert report["baseline"]["sha256"] == _sha256(reranker.RERANKER_ARTIFACT_NAME)


def test_provenance_table_lists_every_shipped_artifact_with_its_hash():
    """ml/artifacts/README.md is the only record of where these files came from.

    The retired artifact's hash appears in prose (it is not shipped), so only rows of
    the "what is here now" table are checked.
    """
    text = ARTIFACTS_README.read_text(encoding="utf-8")
    listed = dict(re.findall(r"\| `([^`]+\.json)` \| `([0-9a-f]{64})` \|", text))
    shipped = {path.name for path in config.ARTIFACTS_DIR.glob("*.json")}
    assert set(listed) == shipped, "таблица провенанса расходится с содержимым каталога"
    for name, expected in listed.items():
        assert _sha256(name) == expected, f"обновите sha256 для {name} в ml/artifacts/README.md"
