"""Vector search: dense index, Qdrant mirror, hybrid shortlist, classifier integration.

A deterministic hashing encoder stands in for the real embedder, so the tests need no
model downloads and run in milliseconds.
"""

from pathlib import Path

import numpy as np
import pytest
from app.core.config import settings
from app.services.classifier import CatalogClassifier

from ml import embedder as embedder_module
from ml.dense import DenseIndex, QdrantMirror, fuse_shortlist
from ml.embedder import TfidfEncoder

CATALOG = Path(__file__).resolve().parents[2] / "data" / "data.csv"


@pytest.fixture(autouse=True)
def hashing_encoder(monkeypatch):
    encoder = TfidfEncoder(dim=256)
    encoder.fit(["ноутбуки", "чайники электрические", "станки токарные", "мебель"])
    monkeypatch.setattr(embedder_module, "get_encoder", lambda: encoder)
    monkeypatch.setattr(embedder_module, "encoder_tag", lambda: "test-hashing")
    return encoder


def _names():
    return ["Ноутбуки", "Чайники электрические", "Станки токарные", "Мебель деревянная"]


def test_dense_index_finds_nearest_and_caches_vectors(tmp_path):
    first = DenseIndex.build(_names(), cache_dir=tmp_path)
    assert first.top("электрический чайник", 1)[0][0] == 1
    assert list(tmp_path.glob("dense-test-hashing-*.npy"))
    again = DenseIndex.build(_names(), cache_dir=tmp_path)  # loaded from the cache
    assert np.allclose(first.vectors, again.vectors)


def test_cache_is_keyed_by_catalog_content(tmp_path):
    DenseIndex.build(_names(), cache_dir=tmp_path)
    DenseIndex.build(_names() + ["Обувь"], cache_dir=tmp_path)
    assert len(list(tmp_path.glob("dense-*.npy"))) == 2


def test_qdrant_mirror_serves_the_same_neighbours_as_memory():
    pytest.importorskip("qdrant_client")
    mirror = QdrantMirror.connect(":memory:", "", "test_dense")
    served = DenseIndex.build(_names(), mirror=mirror)
    plain = DenseIndex.build(_names())
    assert served.store == "qdrant"
    for query in ("чайник", "станок токарный", "мебель"):
        assert [i for i, _ in served.top(query, 3)] == [i for i, _ in plain.top(query, 3)]


def test_qdrant_failure_falls_back_to_memory():
    pytest.importorskip("qdrant_client")
    mirror = QdrantMirror.connect(":memory:", "", "test_dense")
    index = DenseIndex.build(_names(), mirror=mirror)

    def boom(*_args, **_kwargs):
        raise RuntimeError("qdrant is down")

    mirror.top = boom
    assert index.top("чайник", 1)[0][0] == 1
    assert index.store == "memory"


def test_qdrant_mirror_replaces_superseded_collections():
    pytest.importorskip("qdrant_client")
    mirror = QdrantMirror.connect(":memory:", "", "test_dense")
    DenseIndex.build(_names(), mirror=mirror)
    DenseIndex.build(_names() + ["Обувь"], mirror=mirror)
    collections = [c.name for c in mirror._client.get_collections().collections]
    assert len(collections) == 1


def test_fuse_shortlist_adds_dense_only_candidates():
    rows = [{"name": name, "okpd2": str(i)} for i, name in enumerate(_names())]
    index = DenseIndex.build([row["name"] for row in rows])
    lexical = [(0.9, rows[3])]  # lexical matcher only knows the furniture row
    fused = fuse_shortlist(index, rows, lexical, "электрический чайник", k_dense=2)
    names = [row["name"] for _score, row, _cos, _top in fused]
    assert names[0] == "Мебель деревянная"  # lexical hits keep their place first
    assert "Чайники электрические" in names  # found by embedding only
    assert all(top >= cos for _s, _r, cos, top in fused)


def test_classifier_reports_vector_layer(monkeypatch):
    monkeypatch.setattr(settings, "vector_search", True)
    monkeypatch.setattr(settings, "vector_store", "memory")
    monkeypatch.setattr(settings, "data_dir", str(Path("/tmp/dora-test-dense")))
    classifier = CatalogClassifier(CATALOG)
    info = classifier.vector_info
    assert info["enabled"] and info["embedder"] == "test-hashing" and info["store"] == "memory"
    assert classifier.classify("Ноутбук Lenovo").code == "26.20.11"


def test_catalog_backend_honours_the_lexical_rollback(monkeypatch):
    """SEMANTIC_RERANKER=lexical must roll back the *scoring* artifact, not the vectors.

    Turning VECTOR_SEARCH off would also drop vector candidates from the shortlist, so
    the escape hatch has to swap only the reranker — and it must work in the catalog
    backend, not just in the qdrant one, because the catalog backend is the default.
    """
    from app.services import classifier as classifier_module

    requested: list[str | None] = []

    from ml import config as ml_config
    from ml.reranker import LogisticReranker

    lexical_artifact = LogisticReranker.load(ml_config.ARTIFACTS_DIR / "reranker.json")

    def fake_load_reranker(tag=None):
        requested.append(tag)
        return lexical_artifact

    monkeypatch.setattr(classifier_module, "_load_reranker", fake_load_reranker)
    monkeypatch.setattr(settings, "vector_search", True)
    monkeypatch.setattr(settings, "vector_store", "memory")
    monkeypatch.setattr(settings, "data_dir", str(Path("/tmp/dora-test-dense")))

    monkeypatch.setattr(settings, "semantic_reranker", "auto")
    classifier = CatalogClassifier(CATALOG)
    assert requested[-1] == "test-hashing"
    assert classifier.vector_info["enabled"] is True

    monkeypatch.setattr(settings, "semantic_reranker", "lexical")
    classifier = CatalogClassifier(CATALOG)
    assert requested[-1] is None, "откат обязан просить лексический артефакт"
    assert classifier.vector_info["enabled"] is True, "векторы остаются кандидатами"
    assert classifier.vector_info["reranker"] == "reranker.json"
    assert classifier.vector_info["scoring"] == "lexical-reranker"


def test_classifier_survives_embedder_failure(monkeypatch):
    monkeypatch.setattr(settings, "vector_search", True)

    def broken():
        raise RuntimeError("no embedder")

    monkeypatch.setattr(embedder_module, "encoder_tag", broken)
    classifier = CatalogClassifier(CATALOG)
    assert classifier.vector_info["enabled"] is False
    assert classifier.classify("Ноутбук Lenovo").code == "26.20.11"


def test_auto_backend_never_downloads_a_model(monkeypatch, tmp_path):
    monkeypatch.setattr(embedder_module, "EMBEDDER_BACKEND", "auto")
    monkeypatch.setattr(embedder_module, "MODEL_NAME", str(tmp_path / "missing-model"))
    monkeypatch.setattr(embedder_module, "sentence_transformers_available", lambda: True)
    assert embedder_module.resolve_backend() != "sentence-transformers"
    (tmp_path / "model").mkdir()
    monkeypatch.setattr(embedder_module, "MODEL_NAME", str(tmp_path / "model"))
    assert embedder_module.resolve_backend() == "sentence-transformers"
