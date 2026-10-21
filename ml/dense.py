"""In-memory dense (vector) index over the PP-1875 catalog names.

The catalog has ~1000 positions, so exact cosine search is a single matrix-vector
product (<1 ms). The index gives the classifier two things the lexical matcher
cannot: candidates that share *meaning* but no words with the query ("токарный
станок" -> "Станки металлорежущие"), and a cosine score for every shortlisted
candidate that the reranker uses as a feature.

Document vectors are cached on disk (key = embedder tag + hash of the names), so
a heavy transformer encodes the catalog once, not on every start.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from pathlib import Path
from threading import RLock

import numpy as np

from ml import embedder as embedder_module

logger = logging.getLogger("dora.ml.dense")

_QUERY_CACHE_SIZE = 2048


def _names_hash(names: list[str]) -> str:
    digest = hashlib.sha1("\n".join(names).encode("utf-8")).hexdigest()
    return digest[:16]


class QdrantMirror:
    """Vector store for the catalog embeddings in Qdrant (cosine, exact top-k).

    The collection name is content-addressed (embedder tag + hash of the names), so a
    changed catalog or a different embedder creates a *new* collection and the old
    one is dropped afterwards; an unchanged catalog reuses the existing collection
    (a restart does not re-upload anything).
    """

    def __init__(self, client, prefix: str) -> None:
        self._client = client
        self._prefix = prefix
        self.collection = ""

    @classmethod
    def connect(cls, url: str, api_key: str, prefix: str) -> QdrantMirror:
        from qdrant_client import QdrantClient

        if url == ":memory:":
            client = QdrantClient(":memory:")
        else:
            client = QdrantClient(url=url, api_key=api_key or None, timeout=10)
        client.get_collections()  # fail fast when the server is unreachable
        return cls(client, prefix)

    def publish(self, names: list[str], vectors: np.ndarray, tag: str) -> str:
        from qdrant_client.http.models import Distance, PointStruct, VectorParams

        safe_tag = "".join(c if c.isalnum() else "_" for c in tag).lower()
        name = f"{self._prefix}_{safe_tag}_{_names_hash(names)}"
        client = self._client
        if client.collection_exists(name):
            count = client.count(collection_name=name, exact=True).count
            if count == len(names):
                self.collection = name
                return name
            client.delete_collection(name)
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=int(vectors.shape[1]), distance=Distance.COSINE),
        )
        for offset in range(0, len(names), 128):
            chunk = range(offset, min(offset + 128, len(names)))
            client.upsert(
                collection_name=name,
                points=[
                    PointStruct(id=i, vector=vectors[i].tolist(), payload={"name": names[i]})
                    for i in chunk
                ],
                wait=True,
            )
        # Drop superseded collections (older catalog versions / other embedders).
        for existing in client.get_collections().collections:
            if existing.name.startswith(self._prefix + "_") and existing.name != name:
                try:
                    client.delete_collection(existing.name)
                except Exception:  # noqa: BLE001 - cleanup is best effort
                    logger.warning("Не удалось удалить старую коллекцию %s", existing.name)
        logger.info("Векторы опубликованы в Qdrant: %s (%d точек)", name, len(names))
        self.collection = name
        return name

    def top(self, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        response = self._client.query_points(
            collection_name=self.collection, query=vector.tolist(), limit=k
        )
        return [(int(point.id), float(point.score)) for point in response.points]

    def ping(self, timeout: float = 2.0) -> bool:
        """Cheap reachability probe for health checks (``/metrics``, readiness).

        Bounded on purpose: a scraper or a Docker healthcheck polls often and
        must not hang for the client's full request timeout while Qdrant is
        down. The local (in-memory/on-disk) client rejects unknown kwargs, so
        it is retried without one — a local probe cannot hang anyway.
        """
        try:
            try:
                self._client.get_collections(timeout=timeout)
            except (AssertionError, TypeError):
                self._client.get_collections()
            return True
        except Exception:  # noqa: BLE001 - an unreachable store is the answer, not an error
            return False


class DenseIndex:
    """Exact cosine search over L2-normalized catalog vectors."""

    def __init__(self, names: list[str], vectors: np.ndarray, tag: str) -> None:
        if len(names) != len(vectors):
            raise ValueError("names and vectors must have the same length")
        self.names = names
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.tag = tag
        self.dim = int(self.vectors.shape[1]) if len(self.vectors) else 0
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = RLock()
        self.mirror: QdrantMirror | None = None
        self.store = "memory"

    # --- construction -------------------------------------------------------
    @classmethod
    def build(
        cls,
        names: list[str],
        cache_dir: Path | None = None,
        mirror: QdrantMirror | None = None,
    ) -> DenseIndex:
        """Embed ``names`` with the active encoder (or load the cached vectors).

        With ``mirror`` the vectors are also published to Qdrant and nearest-neighbour
        retrieval is served from there (falling back to memory if Qdrant fails).
        """
        index = cls._build_local(names, cache_dir)
        if mirror is not None:
            try:
                mirror.publish(names, index.vectors, index.tag)
                index.mirror, index.store = mirror, "qdrant"
            except Exception:  # noqa: BLE001 - the in-memory index keeps working
                logger.exception("Qdrant недоступен — поиск по векторам в памяти")
        return index

    @classmethod
    def _build_local(cls, names: list[str], cache_dir: Path | None) -> DenseIndex:
        tag = embedder_module.encoder_tag()
        cache_file: Path | None = None
        if cache_dir is not None:
            safe_tag = "".join(c if c.isalnum() or c in "-._" else "_" for c in tag)
            cache_file = Path(cache_dir) / f"dense-{safe_tag}-{_names_hash(names)}.npy"
            if cache_file.is_file():
                try:
                    vectors = np.load(cache_file)
                    if vectors.shape[0] == len(names):
                        logger.info("Векторы каталога загружены из кэша %s", cache_file)
                        return cls(names, vectors, tag)
                except Exception:  # noqa: BLE001 - corrupt cache is simply rebuilt
                    logger.warning("Кэш векторов %s повреждён — пересчитываем", cache_file)
        started = time.perf_counter()
        vectors = np.asarray(embedder_module.encode_documents(names), dtype=np.float32)
        logger.info(
            "Каталог векторизован: %d позиций, dim=%d, эмбеддер=%s, %.1f с",
            len(names),
            vectors.shape[1] if vectors.ndim == 2 else 0,
            tag,
            time.perf_counter() - started,
        )
        if cache_file is not None:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_file, vectors)
            except OSError:
                logger.warning("Не удалось записать кэш векторов %s", cache_file)
        return cls(names, vectors, tag)

    # --- search --------------------------------------------------------------
    def query_vector(self, query: str) -> np.ndarray:
        with self._lock:
            cached = self._cache.get(query)
            if cached is not None:
                self._cache.move_to_end(query)
                return cached
        vector = np.asarray(embedder_module.encode_queries([query])[0], dtype=np.float32)
        with self._lock:
            self._cache[query] = vector
            if len(self._cache) > _QUERY_CACHE_SIZE:
                self._cache.popitem(last=False)
        return vector

    def warm_queries(self, queries: list[str]) -> int:
        """Encode all uncached queries in ONE encoder call; returns the new count.

        Batch classification (``CatalogClassifier.classify_many``) needs a query
        vector per product name. A heavy transformer amortizes its overhead only
        when the texts reach it as one batch, so the whole chunk is encoded here
        up front and ``query_vector``/``top`` then serve every hit from the
        cache. Duplicate names in a batch (files with repeated positions) are
        encoded once as well.
        """
        missing: list[str] = []
        seen: set[str] = set()
        with self._lock:
            for query in queries:
                if not query or query in seen or query in self._cache:
                    continue
                seen.add(query)
                missing.append(query)
        if not missing:
            return 0
        started = time.perf_counter()
        vectors = np.asarray(embedder_module.encode_queries(missing), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(missing):
            raise RuntimeError("Эмбеддер вернул неверное число векторов при пакетном кодировании")
        with self._lock:
            for query, vector in zip(missing, vectors, strict=True):
                self._cache[query] = vector
                self._cache.move_to_end(query)
            while len(self._cache) > _QUERY_CACHE_SIZE:
                self._cache.popitem(last=False)
        logger.debug(
            "Пакетно закодировано %d запросов за %.1f мс",
            len(missing),
            (time.perf_counter() - started) * 1000,
        )
        return len(missing)

    def health(self) -> dict:
        """Live state of the store backing this index (for /metrics and readiness).

        ``store_up`` is ``True`` for the in-memory index (there is nothing
        external to fail) and a bounded ping for the Qdrant mirror.
        """
        if self.mirror is None:
            return {"store": self.store, "store_up": True}
        return {"store": self.store, "store_up": self.mirror.ping()}

    def scores(self, query: str) -> np.ndarray:
        """Cosine similarity of ``query`` to every catalog position."""
        if not len(self.vectors):
            return np.zeros(0, dtype=np.float32)
        return self.vectors @ self.query_vector(query)

    def cosine(self, query: str, index: int) -> float:
        """Cosine of ``query`` to one catalog position."""
        return float(self.vectors[index] @ self.query_vector(query))

    def top(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """``(row_index, cosine)`` of the ``k`` nearest positions, best first."""
        if self.mirror is not None:
            try:
                return self.mirror.top(self.query_vector(query), k)
            except Exception:  # noqa: BLE001 - degrade to exact in-memory search
                logger.exception("Запрос к Qdrant не удался — используем память")
                self.mirror, self.store = None, "memory"
        scores = self.scores(query)
        if not len(scores):
            return []
        k = min(k, len(scores))
        order = np.argpartition(-scores, k - 1)[:k]
        order = order[np.argsort(-scores[order])]
        return [(int(i), float(scores[i])) for i in order]


# How many nearest-by-meaning positions join the lexical shortlist. The same value is
# used at training time (ml.train_reranker) and at serving time (the classifier), so
# the reranker sees the candidate mix it was trained on.
DENSE_SHORTLIST = 8


def fuse_shortlist(
    dense: DenseIndex,
    rows: list[dict],
    lexical: list[tuple[float, dict]],
    query: str,
    prepared_query=None,
    k_dense: int = DENSE_SHORTLIST,
) -> list[tuple[float, dict, float, float]]:
    """Union of the lexical shortlist and the nearest positions by embedding.

    ``rows`` must be aligned with the dense index (``rows[i]`` <-> vector ``i``).
    Returns ``(retrieval_score, row, cosine, query_dense_top)`` tuples: lexical hits
    first (in their order), then positions found only by vector search. Rows the
    lexical matcher did not score keep a retrieval score of 0 (or their exact
    lexical score when ``prepared_query`` is given).
    """
    from ml.features import lexical_score_prepared

    if not len(dense.vectors):
        return [(s, row, 0.0, 0.0) for s, row in lexical]
    nearest = dense.top(query, max(k_dense, 1))
    dense_top = nearest[0][1] if nearest else 0.0
    position = {id(row): i for i, row in enumerate(rows)}
    fused: list[tuple[float, dict, float, float]] = []
    seen: set[int] = set()
    for score, row in lexical:
        i = position[id(row)]
        seen.add(i)
        fused.append((score, row, dense.cosine(query, i), dense_top))
    for i, cosine in nearest:
        if i in seen:
            continue
        seen.add(i)
        row = rows[i]
        lexical_score = 0.0
        if prepared_query is not None and "prepared" in row:
            lexical_score = float(lexical_score_prepared(prepared_query, row["prepared"]))
        fused.append((lexical_score, row, cosine, dense_top))
    return fused
