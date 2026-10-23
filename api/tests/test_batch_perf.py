"""BE-7: batch performance mechanics.

Three properties are pinned here (measurements and the concurrency rationale
live in ``eval/api-load-test.md``):

* query vectors of a batch are encoded in ONE encoder call, duplicates included
  once — a transformer amortizes its overhead only in a batched call;
* ``classify_many`` answers exactly what sequential ``classify`` answers
  (batching changes latency, never decisions);
* a batch job fetches (and thus warms) the classifier exactly once, and a
  failed warm-encoding degrades to per-query encoding instead of failing.
"""

from __future__ import annotations

import numpy as np
import pytest
from app.job_store import create_job, get_job
from app.schemas import ClassifyResponse
from app.services.classifier import CatalogClassifier
from app.tasks import run_batch_job

from ml import embedder as embedder_module
from ml.dense import DenseIndex

MINI_CATALOG = (
    "record_id;appendix;position;name;okpd2;code_role;min_share_pct;"
    "max_foreign_share_pct;confirming_document;pdf_page_start;pdf_page_end;"
    "source_code_cell;review_required\n"
    "t-1-01;1;1;Сыры, продукты сырные и творог;10.51.40.120;Включён;;;;76;76;;False\n"
    "t-2-01;1;2;Ноутбуки портативные;26.20.11;Включён;;;;80;80;;False\n"
    "t-3-01;1;3;Электрочайники;27.51.24.110;Включён;;;;81;81;;False\n"
)


@pytest.fixture()
def counting_encoder(monkeypatch):
    """Deterministic per-text vectors + a call log (one entry per encoder call)."""
    state: dict = {"calls": []}

    def encode_queries(texts, normalize=True):
        state["calls"].append(list(texts))
        vectors = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vector = rng.normal(size=8).astype(np.float32)
            vectors.append((vector / np.linalg.norm(vector)).tolist())
        return vectors

    monkeypatch.setattr(embedder_module, "encode_queries", encode_queries)
    return state


def _index(names: list[str]) -> DenseIndex:
    rng = np.random.default_rng(1875)
    vectors = rng.normal(size=(len(names), 8)).astype(np.float32)
    return DenseIndex(names, vectors, "test-encoder")


def test_warm_queries_makes_one_deduplicated_call(counting_encoder) -> None:
    index = _index(["Чайник", "Ноутбук"])

    assert index.warm_queries(["чайник", "чайник", "ноутбук", ""]) == 2
    assert counting_encoder["calls"] == [["чайник", "ноутбук"]]  # ONE batch, no dupes

    # Everything is cached now: no further encoder calls from warm or lookup.
    assert index.warm_queries(["чайник", "ноутбук"]) == 0
    index.query_vector("чайник")
    index.top("ноутбук", 1)
    assert len(counting_encoder["calls"]) == 1


def test_classify_many_encodes_the_whole_chunk_in_one_call(tmp_path, counting_encoder) -> None:
    catalog = tmp_path / "mini.csv"
    catalog.write_text(MINI_CATALOG, encoding="utf-8")
    plain = CatalogClassifier(catalog, dense=None)
    classifier = CatalogClassifier(catalog, dense=_index([row["name"] for row in plain.index]))

    queries = [
        "Ноутбуки портативные",
        "  Ноутбуки портативные ",  # exact duplicate after normalization
        "Электрочайники",
    ]
    counting_encoder["calls"].clear()
    results = classifier.classify_many(queries)

    assert len(counting_encoder["calls"]) == 1  # one batched encoding per chunk
    assert counting_encoder["calls"][0] == ["Ноутбуки портативные", "Электрочайники"]

    # Decisions are identical to the sequential path...
    sequential = [classifier.classify(query) for query in queries]
    assert [(r.status, r.code) for r in results] == [(r.status, r.code) for r in sequential]
    # ...and the sequential pass re-encoded nothing (cache from the warm-up).
    assert len(counting_encoder["calls"]) == 1
    # The inner classifier answers with the normalized query; the response cache
    # wrapper (CachedClassifier) restores the caller's original text.
    assert [result.query for result in results] == [
        "Ноутбуки портативные",
        "Ноутбуки портативные",
        "Электрочайники",
    ]


def test_classify_many_without_dense_stays_sequential_and_equal(tmp_path) -> None:
    catalog = tmp_path / "mini.csv"
    catalog.write_text(MINI_CATALOG, encoding="utf-8")
    classifier = CatalogClassifier(catalog, dense=None)

    queries = ["Ноутбуки портативные", "Электрочайники", "абвгдеёжз"]
    results = classifier.classify_many(queries)
    sequential = [classifier.classify(query) for query in queries]
    assert [(r.status, r.code, r.title) for r in results] == [
        (r.status, r.code, r.title) for r in sequential
    ]


def test_warm_failure_degrades_to_per_query_encoding(tmp_path, counting_encoder, caplog) -> None:
    catalog = tmp_path / "mini.csv"
    catalog.write_text(MINI_CATALOG, encoding="utf-8")
    plain = CatalogClassifier(catalog, dense=None)
    dense = _index([row["name"] for row in plain.index])

    def boom(queries):
        raise RuntimeError("encoder exploded")

    dense.warm_queries = boom
    classifier = CatalogClassifier(catalog, dense=dense)

    results = classifier.classify_many(["Ноутбуки портативные", "Электрочайники"])
    assert len(results) == 2  # the batch still answers...
    assert all(result.query for result in results)
    # ...encoding each query separately as a fallback.
    assert len(counting_encoder["calls"]) == 2
    assert all(len(call) == 1 for call in counting_encoder["calls"])


def test_batch_job_warms_the_classifier_exactly_once() -> None:
    job = create_job(["Ноутбуки портативные", "Электрочайники", "Сыры"])
    provider_calls = {"count": 0}

    class StubClassifier:
        def classify_many(self, names):
            return [
                ClassifyResponse(
                    query=name,
                    code="26.20.11",
                    title="stub",
                    confidence=0.5,
                    status="review",
                    latency_ms=0.0,
                )
                for name in names
            ]

    def provider():
        provider_calls["count"] += 1
        return StubClassifier()

    run_batch_job(job.job_id, provider)

    assert provider_calls["count"] == 1
    stored = get_job(job.job_id)
    assert stored.status == "completed"
    assert len(stored.results) == 3
