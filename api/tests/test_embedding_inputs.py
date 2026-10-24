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


def _fake_transformer(monkeypatch, model_name: str):
    """Minimal SentenceTransformer stub with a configurable context window."""

    class Model:
        max_seq_length = 8192  # BAAI/bge-m3 default

        def __init__(self, *args, **kwargs):
            pass

        def get_sentence_embedding_dimension(self):
            return 1024

        def encode(self, texts, **kwargs):
            return np.zeros((len(texts), 1024), dtype=np.float32)

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model)
    )
    monkeypatch.setattr(embedder, "MODEL_NAME", model_name)
    return Model


def test_context_length_is_capped_for_long_context_models(tmp_path, monkeypatch):
    """bge-m3 defaults to 8192 tokens; catalog names need a few hundred."""
    _fake_transformer(monkeypatch, str(tmp_path))
    monkeypatch.setattr(embedder, "MODEL_MAX_SEQ_LENGTH", 512)
    encoder = embedder.SentenceTransformerEncoder()
    assert encoder.dim == 1024
    assert encoder._model.max_seq_length == 512


def test_zero_keeps_the_model_default_context(tmp_path, monkeypatch):
    _fake_transformer(monkeypatch, str(tmp_path))
    monkeypatch.setattr(embedder, "MODEL_MAX_SEQ_LENGTH", 0)
    encoder = embedder.SentenceTransformerEncoder()
    assert encoder._model.max_seq_length == 8192


def test_auto_mode_warns_when_transformer_is_installed_but_missing(
    tmp_path, monkeypatch, caplog
):
    """Silent degradation to spaCy must leave a trace in the log."""
    monkeypatch.setattr(embedder, "EMBEDDER_BACKEND", "auto")
    monkeypatch.setattr(embedder, "MODEL_NAME", str(tmp_path / "missing-weights"))
    monkeypatch.setattr(embedder, "sentence_transformers_available", lambda: True)
    monkeypatch.setattr(embedder, "spacy_available", lambda: False)
    monkeypatch.setattr(embedder, "_fallback_warned", False, raising=False)

    with caplog.at_level("WARNING", logger="dora.ml.embedder"):
        assert embedder.resolve_backend() == "tfidf"

    assert "MODEL_NAME" in caplog.text
    assert "scripts/download_model.py" in caplog.text
    # Warn once per process, not on every probe.
    caplog.clear()
    with caplog.at_level("WARNING", logger="dora.ml.embedder"):
        embedder.resolve_backend()
    assert caplog.text == ""


def test_explicit_backend_never_silently_degrades(monkeypatch):
    monkeypatch.setattr(embedder, "EMBEDDER_BACKEND", "sentence-transformers")
    assert embedder.resolve_backend() == "sentence-transformers"
