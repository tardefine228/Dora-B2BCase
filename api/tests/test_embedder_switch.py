"""Switching the embedder must partition every cached/derived artifact.

Replacing ``intfloat/multilingual-e5-small`` (384 dims, e5 prefixes) with a
different bi-encoder — ``BAAI/bge-m3`` (1024 dims, no prefixes) is the case that
motivated this — must never reuse vectors or a reranker produced by the other
model. The sources of truth are already tag-keyed; this test pins the contract
end to end with a synthetic 1024-dim encoder, so it runs without weights,
torch, or a Qdrant server.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from ml import embedder as embedder_module
from ml.dense import DenseIndex, QdrantMirror

DIM = 1024
BGE_TAG = "st-bge-m3-v2-b118c3d8643d"
E5_TAG = "st-multilingual-e5-small-v2-9b1a13eefd5f"


class _StubEncoder:
    """Deterministic encoder with a BGE-M3 shape: 1024 dims, no prefixes."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.dim = DIM

    def _vector(self, text: str) -> list[float]:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        vector = np.random.default_rng(seed).normal(size=DIM)
        return (vector / np.linalg.norm(vector)).tolist()

    def encode(self, texts, normalize=True):
        return [self._vector(text) for text in texts]

    encode_queries = encode
    encode_documents = encode


@pytest.fixture
def bge_encoder(monkeypatch):
    encoder = _StubEncoder(BGE_TAG)
    monkeypatch.setattr(embedder_module, "get_encoder", lambda: encoder)
    monkeypatch.setattr(embedder_module, "encoder_tag", lambda: encoder.tag)
    return encoder


def test_dense_cache_is_per_embedder(bge_encoder, tmp_path, monkeypatch):
    names = ["Ноутбуки", "Электрочайники", "Станки металлорежущие"]
    index = DenseIndex.build(names, cache_dir=tmp_path)

    assert index.dim == DIM
    assert index.tag == BGE_TAG
    cache_files = sorted(path.name for path in tmp_path.glob("dense-*.npy"))
    assert len(cache_files) == 1
    assert BGE_TAG in cache_files[0]

    # A different embedder must not read those vectors back as its own.
    monkeypatch.setattr(embedder_module, "encoder_tag", lambda: E5_TAG)
    other = DenseIndex.build(names, cache_dir=tmp_path)
    assert other.tag == E5_TAG
    assert len(sorted(tmp_path.glob("dense-*.npy"))) == 2


def test_qdrant_mirror_collection_carries_the_embedder_tag(bge_encoder, tmp_path):
    from qdrant_client import QdrantClient

    names = ["Ноутбуки", "Электрочайники"]
    index = DenseIndex.build(names, cache_dir=tmp_path)
    mirror = QdrantMirror(QdrantClient(path=str(tmp_path / "qdrant")), "pp1875_items")

    collection = mirror.publish(names, index.vectors, index.tag)

    assert BGE_TAG.replace("-", "_") in collection
    assert mirror.top(index.vectors[0], 1)[0][0] == 0  # exact hit on its own vector


def test_reranker_artifact_name_is_per_embedder():
    from ml.reranker import reranker_artifact_name

    lexical = reranker_artifact_name()
    bge = reranker_artifact_name(BGE_TAG)
    e5 = reranker_artifact_name(E5_TAG)

    assert lexical == "reranker.json"
    assert bge == f"reranker.{BGE_TAG}.json"
    assert len({lexical, bge, e5}) == 3


def test_model_identity_survives_the_embedder_copy(tmp_path):
    """``/opt/models/embedder`` must keep the hub ID it was downloaded from."""
    import json

    baked = tmp_path / "embedder"
    baked.mkdir()
    (baked / "dora_model.json").write_text(json.dumps({"model_id": "BAAI/bge-m3"}))

    tag, query_prefix, doc_prefix = embedder_module._transformer_spec(str(baked))

    assert tag == BGE_TAG
    assert (query_prefix, doc_prefix) == ("", "")  # bge needs no e5 prefixes


def test_health_reports_the_real_embedder_dim(bge_encoder, tmp_path, monkeypatch):
    from app.services.classifier import CatalogClassifier

    catalog = tmp_path / "catalog.csv"
    catalog.write_text(
        "record_id;position;name;okpd2;code_role;review_required\n"
        "2-1-01;1;Ноутбуки;26.20.11;Включён;False\n"
        "2-2-01;2;Электрочайники;27.51.24.110;Включён;False\n",
        encoding="utf-8",
    )
    index = DenseIndex.build(["Ноутбуки", "Электрочайники"], cache_dir=tmp_path / "cache")
    classifier = CatalogClassifier(Path(catalog), reranker=None, dense=index)

    info = classifier.vector_info
    assert info["embedder"] == BGE_TAG
    assert info["dim"] == DIM
