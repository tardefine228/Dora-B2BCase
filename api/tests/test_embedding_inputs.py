"""Regression checks for renamed E5 models and query/document routing (no weights)."""

import json
import sys
from types import SimpleNamespace

import numpy as np

from ml import embedder, search


def test_renamed_e5_keeps_prefixes_and_cache_identity(tmp_path):
    local = tmp_path / "embedder"
    local.mkdir()
    (local / "dora_model.json").write_text(
        json.dumps({"model_id": "intfloat/multilingual-e5-small"})
    )
    tag, query, document = embedder._transformer_spec(str(local))
    assert (query, document) == ("query: ", "passage: ")
    assert tag == embedder._transformer_spec("intfloat/multilingual-e5-small")[0]
    assert tag not in {"st-embedder", "st-multilingual-e5-small"}
    assert embedder._transformer_spec("BAAI/bge-m3")[1:] == ("", "")


def test_encoder_passes_asymmetric_inputs_to_model(tmp_path, monkeypatch):
    (tmp_path / "dora_model.json").write_text(
        json.dumps({"model_id": "intfloat/multilingual-e5-small"})
    )
    calls = []

    class Model:
        def __init__(self, *args, **kwargs):
            pass

        def get_sentence_embedding_dimension(self):
            return 2

        def encode(self, texts, **kwargs):
            calls.append(texts)
            return np.asarray([[1.0, 0.0] for _ in texts])

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model)
    )
    monkeypatch.setattr(embedder, "MODEL_NAME", str(tmp_path))
    encoder = embedder.SentenceTransformerEncoder()
    encoder.encode_queries(["ноутбук"])
    encoder.encode_documents(["Компьютеры портативные"])
    assert calls == [["query: ноутбук"], ["passage: Компьютеры портативные"]]


def test_qdrant_search_uses_query_encoder(monkeypatch):
    seen = []

    def encode_queries(texts):
        seen.extend(texts)
        return [[0.0, 1.0] for _ in texts]

    class Client:
        def query_batch_points(self, *, collection_name, requests):
            assert requests[0].query == [0.0, 1.0]
            return [SimpleNamespace(points=[]) for _ in requests]

    monkeypatch.setattr(search, "encode_queries", encode_queries)
    monkeypatch.setattr(search.indexer, "get_qdrant_client", lambda: Client())
    assert search.search_many(["ноутбук"]) == [[]]
    assert seen == ["ноутбук"]
