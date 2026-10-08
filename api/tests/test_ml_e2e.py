"""End-to-end ML/Qdrant pipeline: embed → index → /admin/reindex → classify.

Runs on the offline TF-IDF embedder and qdrant-client local storage, so the
whole pipeline (including repeated reindexing with atomic alias swap) is
verified without Docker or model weights. The same code path is used with a
SentenceTransformers model when one is provided in ``models/``.
"""

from __future__ import annotations

import pytest

qdrant_client = pytest.importorskip("qdrant_client", reason="qdrant-client не установлен")

from app.core.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.services.classifier import reset_classifier  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app)


@pytest.fixture()
def ml_env(tmp_path, monkeypatch):
    import ml.embedder
    import ml.indexer

    monkeypatch.setattr(settings, "ml_enabled", True)
    monkeypatch.setattr(settings, "admin_api_key", "ml-secret")
    monkeypatch.setattr(settings, "classifier_backend", "qdrant")
    monkeypatch.setattr(ml.indexer, "QDRANT_URL", None)
    monkeypatch.setattr(ml.indexer, "QDRANT_STORAGE_DIR", tmp_path / "qdrant")
    ml.indexer.get_qdrant_client.cache_clear()
    ml.embedder.get_encoder.cache_clear()
    ml.embedder.reset_encoder_cache()
    reset_classifier()
    yield monkeypatch
    ml.indexer.get_qdrant_client.cache_clear()
    ml.embedder.get_encoder.cache_clear()
    reset_classifier()
    monkeypatch.setattr(settings, "classifier_backend", "catalog")


def _reindex_and_wait() -> dict:
    accepted = client.post("/admin/reindex", headers={"X-Admin-API-Key": "ml-secret"})
    assert accepted.status_code == 202, accepted.text
    status = client.get(accepted.json()["status_url"], headers={"X-Admin-API-Key": "ml-secret"})
    assert status.status_code == 200
    state = status.json()
    assert state["status"] == "completed", state
    return state


def test_reindex_classify_and_reindex_again(ml_env) -> None:
    # 1. First /admin/reindex builds the collection from the real catalog.
    first = _reindex_and_wait()
    assert first["indexed"] >= 1000

    # 2. Classification works through the Qdrant backend right away.
    response = client.post("/classify", json={"name": "Электрочайники"})
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == "27.51.24.110"
    assert body["status"] in {"ok", "review"}

    # 3. Reindexing again publishes a fresh collection atomically…
    second = _reindex_and_wait()
    assert second["indexed"] == first["indexed"]

    # …and the service keeps answering, without leftover staging collections.
    follow_up = client.post("/classify", json={"name": "Электрочайники"})
    assert follow_up.status_code == 200
    assert follow_up.json()["code"] == "27.51.24.110"

    from ml.config import QDRANT_COLLECTION
    from ml.indexer import get_qdrant_client

    client_qdrant = get_qdrant_client()
    names = [collection.name for collection in client_qdrant.get_collections().collections]
    assert client_qdrant.collection_exists(QDRANT_COLLECTION)
    # The stable name is an alias; the physical collection keeps its staging
    # name. What must never happen is a leftover staging collection that the
    # alias does not point to.
    alias_targets = {
        alias.collection_name
        for alias in client_qdrant.get_aliases().aliases
        if alias.alias_name == QDRANT_COLLECTION
    }
    leftovers = [name for name in names if "__staging_" in name and name not in alias_targets]
    assert not leftovers, leftovers


def test_search_returns_catalog_payload(ml_env) -> None:
    _reindex_and_wait()
    from ml.search import search

    hits = search("светодиодная лампа", top_k=5)
    assert hits
    assert any(hit.pp_code.startswith("27.40") for hit in hits)
    assert all(hit.name for hit in hits)


def test_reindex_requires_ml_enabled(monkeypatch) -> None:
    monkeypatch.setattr(settings, "ml_enabled", False)
    monkeypatch.setattr(settings, "admin_api_key", "ml-secret")
    response = client.post("/admin/reindex", headers={"X-Admin-API-Key": "ml-secret"})
    assert response.status_code == 503
