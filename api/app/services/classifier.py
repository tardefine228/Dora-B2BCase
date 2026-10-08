"""Classifiers for products covered by Russian Government Decree No. 1875.

The catalog implementation is deliberately lightweight and works offline. The
optional Qdrant implementation uses the ML package and a model that must already
be available locally when offline mode is enabled.
"""

from __future__ import annotations

import csv
import logging
import re
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from threading import BoundedSemaphore, RLock
from typing import Protocol

from app.core.config import settings
from app.schemas import Alternative, ClassifyResponse

logger = logging.getLogger("dora.classifier")

# OKPD2 examples: 14.1, 26.20.11, 26.20.11.110
OKPD2_RE = re.compile(r"^\d{2}(\.\d{1,2}){0,3}$")
TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
_INFERENCE_SEMAPHORE = BoundedSemaphore(settings.max_concurrent_inferences)


@contextmanager
def exclusive_inference():
    """Pause all classifications during memory-intensive model indexing."""
    acquired = 0
    try:
        for _ in range(settings.max_concurrent_inferences):
            _INFERENCE_SEMAPHORE.acquire()
            acquired += 1
        yield
    finally:
        for _ in range(acquired):
            _INFERENCE_SEMAPHORE.release()


def _tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if len(token) > 1]


def _not_found(query: str) -> ClassifyResponse:
    return ClassifyResponse(
        query=query,
        code=None,
        title=None,
        confidence=0.0,
        status="not_found",
        clause_ref=None,
        clause_url=None,
        alternatives=[],
        latency_ms=0.0,
    )


class Classifier(Protocol):
    def classify(self, query: str) -> ClassifyResponse: ...


class MockClassifier:
    """Deterministic placeholder for frontend development."""

    def classify(self, query: str) -> ClassifyResponse:
        return ClassifyResponse(
            query=query,
            code="MOCK-001",
            title="Демонстрационный ответ — ML-модуль ещё не подключён",
            confidence=0.0,
            status="review",
            clause_ref=None,
            clause_url=None,
            alternatives=[],
            latency_ms=0.0,
        )


class CatalogClassifier:
    """Lightweight matching against the checked-in CSV catalog."""

    def __init__(self, catalog_path: Path) -> None:
        self.catalog_path = Path(catalog_path)
        self.index: list[dict] = []
        self._load()
        if not self.index:
            raise ValueError(f"В каталоге нет записей: {self.catalog_path}")
        logger.info(
            "CatalogClassifier загружен: %d позиций из %s",
            len(self.index),
            self.catalog_path,
        )

    def _load(self) -> None:
        with self.catalog_path.open(encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file, delimiter=";")
            required_columns = {"name", "okpd2"}
            if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
                raise ValueError(f"В каталоге должны быть колонки {sorted(required_columns)}")
            for row in reader:
                name = (row.get("name") or "").strip()
                if not name:
                    continue
                self.index.append(
                    {
                        "okpd2": (row.get("okpd2") or "").strip(),
                        "name": name,
                        "appendix": (row.get("appendix") or "").strip(),
                        "tokens": set(_tokenize(name)),
                        "text": name.casefold(),
                    }
                )

    @staticmethod
    def _token_matches(query_token: str, tokens: set[str], text: str) -> bool:
        if query_token in tokens or query_token in text:
            return True
        # Tolerate simple inflections and singular/plural variants.
        return any(token and (token in query_token or query_token in token) for token in tokens)

    def classify(self, query: str) -> ClassifyResponse:
        normalized_query = query.strip()
        if not normalized_query:
            return _not_found(normalized_query)

        query_lower = normalized_query.casefold()

        # 1. Direct OKPD2 code lookup.
        match = OKPD2_RE.fullmatch(query_lower.replace(" ", ""))
        if match:
            code = match.group(0)
            for row in self.index:
                if row["okpd2"] == code:
                    return ClassifyResponse(
                        query=normalized_query,
                        code=code,
                        title=row["name"],
                        confidence=1.0,
                        status="ok",
                        clause_ref=f"Приложение {row['appendix']}" if row["appendix"] else None,
                        clause_url=None,
                        alternatives=[],
                        latency_ms=0.0,
                    )

        # 2. Lightweight token overlap. Treat the score as a suggestion, not a legal decision.
        query_tokens = set(_tokenize(query_lower))
        if not query_tokens:
            return _not_found(normalized_query)

        scored: list[tuple[float, dict]] = []
        for row in self.index:
            matched = [
                token
                for token in query_tokens
                if self._token_matches(token, row["tokens"], row["text"])
            ]
            if matched:
                scored.append((len(matched) / len(query_tokens), row))

        if not scored:
            return _not_found(normalized_query)

        # Stable ordering makes responses deterministic when scores tie.
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best = scored[0]
        seen_codes = {best["okpd2"]}
        alternatives: list[Alternative] = []
        for score, row in scored[1:]:
            if row["okpd2"] in seen_codes:
                continue
            seen_codes.add(row["okpd2"])
            alternatives.append(
                Alternative(
                    code=row["okpd2"] or None,
                    title=row["name"],
                    confidence=round(min(score, 1.0), 4),
                )
            )
            if len(alternatives) == 3:
                break

        confidence = round(min(best_score, 1.0), 4)
        return ClassifyResponse(
            query=normalized_query,
            code=best["okpd2"] or None,
            title=best["name"],
            confidence=confidence,
            status="ok" if confidence >= 0.5 else "review",
            clause_ref=f"Приложение {best['appendix']}" if best["appendix"] else None,
            clause_url=None,
            alternatives=alternatives,
            latency_ms=0.0,
        )

    def classify_many(self, queries: list[str]) -> list[ClassifyResponse]:
        return [self.classify(query) for query in queries]


class QdrantClassifier:
    """Adapter to semantic search in a prebuilt Qdrant collection."""

    def __init__(self) -> None:
        if not settings.ml_enabled:
            raise RuntimeError(
                "ML backend отключён. Соберите образ с INSTALL_ML=true "
                "и установите ML_ENABLED=true."
            )
        try:
            from ml.config import QDRANT_COLLECTION
            from ml.embedder import get_model
            from ml.indexer import get_qdrant_client
            from ml.search import search_many
        except Exception as exc:  # optional, heavyweight dependencies
            raise RuntimeError(f"ML/Qdrant backend недоступен: {exc}") from exc

        # Load once at startup, then reuse the model and Qdrant client for requests.
        get_model()
        client = get_qdrant_client()
        if not client.collection_exists(QDRANT_COLLECTION):
            raise RuntimeError(
                f"Коллекция Qdrant '{QDRANT_COLLECTION}' не найдена; "
                "сначала выполните POST /admin/reindex."
            )
        self._search_many = search_many

    def classify(self, query: str) -> ClassifyResponse:
        return self.classify_many([query])[0]

    def classify_many(self, queries: list[str]) -> list[ClassifyResponse]:
        normalized_queries = [query.strip() for query in queries]
        if not normalized_queries:
            return []

        try:
            hit_groups = self._search_many(normalized_queries, top_k=3)
        except Exception:
            logger.exception("Qdrant batch search failed")
            raise

        results: list[ClassifyResponse] = []
        for query, hits in zip(normalized_queries, hit_groups, strict=True):
            if not hits:
                results.append(_not_found(query))
                continue

            top = hits[0]
            confidence = max(0.0, min(float(top.score), 1.0))
            alternatives = [
                Alternative(
                    code=hit.pp_code or None,
                    title=hit.name or None,
                    confidence=max(0.0, min(float(hit.score), 1.0)),
                )
                for hit in hits[1:3]
            ]
            results.append(
                ClassifyResponse(
                    query=query,
                    code=top.pp_code or None,
                    title=top.name or None,
                    confidence=round(confidence, 4),
                    status="ok" if confidence >= 0.75 else "review",
                    clause_ref=top.position or top.group or None,
                    # pp_link currently contains a human-readable source citation, not a URL.
                    clause_url=None,
                    alternatives=alternatives,
                    latency_ms=0.0,
                )
            )
        return results


def _resolve_catalog_path() -> Path | None:
    project_root = Path(__file__).resolve().parents[3]
    candidates: list[Path] = []

    if settings.catalog_path:
        explicit_path = Path(settings.catalog_path).expanduser()
        candidates.append(
            explicit_path if explicit_path.is_absolute() else project_root / explicit_path
        )

    data_dir = Path(settings.data_dir).expanduser()
    candidates.append(
        data_dir / "data.csv" if data_dir.is_absolute() else project_root / data_dir / "data.csv"
    )
    candidates.append(project_root / "data" / "data.csv")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def build_classifier(backend: str | None = None) -> Classifier:
    selected_backend = (backend or settings.classifier_backend).strip().lower()
    if selected_backend == "mock":
        return MockClassifier()
    if selected_backend == "qdrant":
        return QdrantClassifier()
    if selected_backend != "catalog":
        raise ValueError(f"Неизвестный CLASSIFIER_BACKEND: {selected_backend}")

    catalog_path = _resolve_catalog_path()
    if catalog_path is None:
        raise FileNotFoundError("Каталог data.csv не найден. Проверьте DATA_DIR или CATALOG_PATH.")
    return CatalogClassifier(catalog_path)


class CachedClassifier:
    """Bounded LRU cache plus a concurrency limit around CPU-heavy inference."""

    def __init__(self, classifier: Classifier, *, max_entries: int) -> None:
        self._classifier = classifier
        self._max_entries = max_entries
        self._cache: OrderedDict[str, ClassifyResponse] = OrderedDict()
        self._cache_lock = RLock()
        # Shared by all wrapper instances so admin indexing can pause inference globally.
        self._inference_semaphore = _INFERENCE_SEMAPHORE

    @staticmethod
    def _key(query: str) -> str:
        return " ".join(query.casefold().split())

    def _cached(self, key: str) -> ClassifyResponse | None:
        with self._cache_lock:
            result = self._cache.get(key)
            if result is None:
                return None
            self._cache.move_to_end(key)
            return result.model_copy(deep=True)

    def _save(self, key: str, result: ClassifyResponse) -> None:
        with self._cache_lock:
            self._cache[key] = result.model_copy(deep=True)
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)

    def classify(self, query: str) -> ClassifyResponse:
        key = self._key(query)
        result = self._cached(key)
        if result is None:
            with self._inference_semaphore:
                # Another request may have populated the cache while this one waited.
                result = self._cached(key)
                if result is None:
                    result = self._classifier.classify(query)
                    self._save(key, result)
        return result.model_copy(update={"query": query})

    def classify_many(self, queries: list[str]) -> list[ClassifyResponse]:
        if not queries:
            return []

        keys = [self._key(query) for query in queries]
        results_by_key: dict[str, ClassifyResponse] = {}
        missing_queries: dict[str, str] = {}
        for key, query in zip(keys, queries, strict=True):
            cached = self._cached(key)
            if cached is not None:
                results_by_key[key] = cached
            else:
                missing_queries.setdefault(key, query)

        if missing_queries:
            with self._inference_semaphore:
                # Recheck after waiting to avoid duplicate work from simultaneous batches.
                remaining: dict[str, str] = {}
                for key, query in missing_queries.items():
                    cached = self._cached(key)
                    if cached is not None:
                        results_by_key[key] = cached
                    else:
                        remaining[key] = query

                if remaining:
                    names = list(remaining.values())
                    batch_method = getattr(self._classifier, "classify_many", None)
                    computed = (
                        batch_method(names)
                        if callable(batch_method)
                        else [self._classifier.classify(name) for name in names]
                    )
                    if len(computed) != len(names):
                        raise RuntimeError("Классификатор вернул неверное число результатов")
                    for key, result in zip(remaining, computed, strict=True):
                        self._save(key, result)
                        results_by_key[key] = result

        return [
            results_by_key[key].model_copy(update={"query": query})
            for key, query in zip(keys, queries, strict=True)
        ]

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()


_singleton: CachedClassifier | None = None
_singleton_lock = RLock()


def get_classifier() -> CachedClassifier:
    """Return the single process-wide classifier and its bounded response cache."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            base_classifier = build_classifier()
            _singleton = CachedClassifier(
                base_classifier,
                max_entries=settings.classification_cache_size,
            )
        return _singleton


def reset_classifier() -> None:
    """Reset the singleton (primarily used by tests and catalog reload tooling)."""
    global _singleton
    with _singleton_lock:
        _singleton = None
