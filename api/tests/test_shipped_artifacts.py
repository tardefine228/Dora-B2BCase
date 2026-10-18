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
