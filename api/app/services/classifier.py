"""Classifiers for products covered by Russian Government Decree No. 1875.

Three backends share one response contract:

* ``catalog`` — offline matching against the checked-in ``data/data.csv``:
  lexical retrieval shortlist + calibrated reranker (default, demo mode);
* ``qdrant`` — semantic search in a prebuilt Qdrant collection (``ml`` package),
  reranked with the same calibrated reranker;
* ``mock`` — deterministic placeholder for frontend development.

Both real backends return the reranker's match probability as ``confidence`` and
map it to ``ok`` / ``review`` / ``not_found`` with the calibrated thresholds from
``models/reranker.json``. Results are suggestions for a specialist, not a legal
determination.
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
from app.services.source_lookup import clause_url_for

logger = logging.getLogger("dora.classifier")

# OKPD2 examples: 14.1, 26.20.11, 26.20.11.110
OKPD2_RE = re.compile(r"^\d{2}(\.\d{1,2}){0,3}$")
TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
_INFERENCE_SEMAPHORE = BoundedSemaphore(settings.max_concurrent_inferences)

_RETRIEVAL_SHORTLIST = 25


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


def _display_confidence(reranker, raw_confidence: float) -> float:
    """Calibrated scale for display; the raw value still decides the status."""
    calibrate = getattr(reranker, "display_confidence", None)
    if not callable(calibrate):
        return raw_confidence
    return min(max(float(calibrate(raw_confidence)), 0.0), 1.0)


def _not_found(query: str) -> ClassifyResponse:
    return ClassifyResponse(
        query=query,
        code=None,
        title=None,
        confidence=0.0,
        confidence_raw=0.0,
        status="not_found",
        clause_ref=None,
        clause_url=None,
        alternatives=[],
        latency_ms=0.0,
    )


_MAX_ALTERNATIVES = 3


def _distinct_alternatives(ranked, status: str, threshold_review: float) -> list[Alternative]:
    """Up to ``_MAX_ALTERNATIVES`` runner-up positions with distinct codes.

    The same position is often listed in several appendices (and Qdrant stores
    one point per appendix row), so the raw ranking contains duplicates that
    would otherwise crowd out genuinely different suggestions. For ``not_found``
    answers only candidates above the review threshold are kept.
    """
    alternatives: list[Alternative] = []
    seen_codes = {ranked[0].code}
    for candidate in ranked[1:]:
        if candidate.code in seen_codes:
            continue
        if status == "not_found" and candidate.probability < threshold_review:
            continue
        seen_codes.add(candidate.code)
        alternatives.append(
            Alternative(
                code=candidate.code,
                title=candidate.title,
                confidence=round(min(max(candidate.probability, 0.0), 1.0), 4),
                clause_ref=candidate.clause_ref,
                clause_url=clause_url_for(candidate.payload.get("record_id")),
            )
        )
        if len(alternatives) == _MAX_ALTERNATIVES:
            break
    return alternatives


def _load_reranker():
    """Return the trained reranker, or a lexical fallback when artifacts are absent."""
    try:
        from ml.reranker import load_reranker
    except Exception:  # ml package must never break the offline catalog backend
        logger.exception("ml.reranker недоступен — используется эвристический скоринг")
        return None
    try:
        return load_reranker()
    except Exception:
        logger.exception("Не удалось загрузить реранкер — используется эвристический скоринг")
        return None


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
    """Matching against the checked-in CSV catalog: lexical retrieval + reranker."""

    def __init__(self, catalog_path: Path, reranker=None) -> None:
        self.catalog_path = Path(catalog_path)
        self.index: list[dict] = []
        self._reranker = reranker if reranker is not None else _load_reranker()
        self._load()
        if not self.index:
            raise ValueError(f"В каталоге нет записей: {self.catalog_path}")
        logger.info(
            "CatalogClassifier загружен: %d позиций из %s (reranker=%s)",
            len(self.index),
            self.catalog_path,
            type(self._reranker).__name__ if self._reranker else "none",
        )

    def _load(self) -> None:
        from ml.features import prepare_candidate

        by_key: dict[tuple[str, str], dict] = {}
        with self.catalog_path.open(encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file, delimiter=";")
            required_columns = {"name", "okpd2"}
            if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
                raise ValueError(f"В каталоге должны быть колонки {sorted(required_columns)}")
            for row in reader:
                name = (row.get("name") or "").strip()
                if not name:
                    continue
                okpd2 = (row.get("okpd2") or "").strip()
                clause_ref = self._clause_ref(row)
                key = (okpd2, name)
                existing = by_key.get(key)
                if existing is not None:
                    # The same position often appears in several appendices; keep
                    # one entry and cite all of them so ranking margins stay sane.
                    if clause_ref and clause_ref not in (existing["clause_ref"] or ""):
                        existing["clause_ref"] = f"{existing['clause_ref']}; {clause_ref}"
                    continue
                by_key[key] = {
                    "okpd2": okpd2,
                    "name": name,
                    "appendix": (row.get("appendix") or "").strip(),
                    "clause_ref": clause_ref,
                    # Source of record: /source/{record_id} renders the pages of
                    # data/source_text.txt this position was parsed from.
                    "record_id": (row.get("record_id") or "").strip(),
                    # Precomputed once: retrieval scores every row per request.
                    "prepared": prepare_candidate(name),
                }
        self.index = list(by_key.values())

    @staticmethod
    def _clause_ref(row: dict) -> str | None:
        appendix = (row.get("appendix") or "").strip()
        position = (row.get("position") or "").strip()
        if appendix and position:
            return f"Приложение {appendix}, позиция {position}"
        if appendix:
            return f"Приложение {appendix}"
        return None

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
                        confidence_raw=1.0,
                        status="ok",
                        clause_ref=row["clause_ref"],
                        clause_url=clause_url_for(row.get("record_id")),
                        alternatives=[],
                        latency_ms=0.0,
                    )

        # 2. Lexical shortlist, then the calibrated reranker orders the candidates.
        query_tokens = set(_tokenize(query_lower))
        if not query_tokens:
            return _not_found(normalized_query)

        shortlist = self._retrieve(normalized_query, query_tokens)
        if not shortlist:
            return _not_found(normalized_query)

        return self._rank_to_response(normalized_query, shortlist)

    def _retrieve(self, query: str, query_tokens: set[str]) -> list[tuple[float, dict]]:
        """Lexical shortlist as ``(retrieval_score, row)`` pairs, best first."""
        from ml.features import lexical_score_prepared, prepare_candidate
        from ml.lexicon import expand_query

        prepared_query = prepare_candidate(expand_query(query))
        scored: list[tuple[float, dict]] = []
        for row in self.index:
            score = lexical_score_prepared(prepared_query, row["prepared"])
            if score <= 0.0:
                # Keep the historical token-substring tolerance as a safety net.
                matched = sum(
                    1
                    for token in query_tokens
                    if token in row["name"].casefold()
                    or any(token in other or other in token for other in _tokenize(row["name"]))
                )
                score = 0.2 * matched / len(query_tokens) if matched else 0.0
            if score > 0.0:
                scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:_RETRIEVAL_SHORTLIST]

    def _rank_to_response(
        self, query: str, shortlist: list[tuple[float, dict]]
    ) -> ClassifyResponse:
        try:
            from ml.reranker import RankedCandidate
        except Exception:  # pragma: no cover - fallback path
            logger.exception("ml.reranker недоступен — отдаём лучший lexical-кандидат")
            best = shortlist[0][1]
            return ClassifyResponse(
                query=query,
                code=best["okpd2"] or None,
                title=best["name"],
                confidence=0.5,
                status="review",
                clause_ref=best["clause_ref"],
                clause_url=clause_url_for(best.get("record_id")),
                alternatives=[],
                latency_ms=0.0,
            )

        # The reranker's retrieval_margin feature is trained on the lexical
        # shortlist scores, so the same scores must reach it at inference.
        candidates = [
            RankedCandidate(
                code=row["okpd2"] or None,
                title=row["name"],
                clause_ref=row["clause_ref"],
                retrieval_score=float(score),
                payload={"record_id": row.get("record_id") or ""},
            )
            for score, row in shortlist
        ]
        if self._reranker is not None:
            ranked = self._reranker.rerank(query, candidates)
        else:  # pragma: no cover - only when ml import failed
            ranked = candidates

        best = ranked[0]
        if self._reranker is not None:
            raw_confidence, status = self._reranker.evaluate(query, ranked)
        else:  # pragma: no cover
            raw_confidence, status = best.probability, "review"
        raw_confidence = round(min(max(raw_confidence, 0.0), 1.0), 4)
        confidence = round(_display_confidence(self._reranker, raw_confidence), 4)

        threshold_review = getattr(self._reranker, "threshold_review", 0.45)
        alternatives = _distinct_alternatives(ranked, status, threshold_review)

        if status == "not_found":
            # Too weak to suggest a position: report "не найдено" and keep at most
            # plausible near-misses in alternatives for the specialist.
            return ClassifyResponse(
                query=query,
                code=None,
                title=None,
                confidence=0.0,
                confidence_raw=0.0,
                status="not_found",
                clause_ref=None,
                clause_url=None,
                alternatives=alternatives,
                latency_ms=0.0,
            )

        return ClassifyResponse(
            query=query,
            code=best.code,
            title=best.title,
            confidence=confidence,
            confidence_raw=raw_confidence,
            status=status,
            clause_ref=best.clause_ref,
            clause_url=clause_url_for(best.payload.get("record_id")),
            alternatives=alternatives,
            latency_ms=0.0,
        )

    def classify_many(self, queries: list[str]) -> list[ClassifyResponse]:
        return [self.classify(query) for query in queries]


class QdrantClassifier:
    """Semantic search in a prebuilt Qdrant collection + shared calibrated reranker."""

    def __init__(self) -> None:
        if not settings.ml_enabled:
            raise RuntimeError(
                "ML backend отключён. Соберите образ с INSTALL_ML=true "
                "и установите ML_ENABLED=true."
            )
        try:
            from ml.config import QDRANT_COLLECTION
            from ml.embedder import get_encoder
            from ml.indexer import get_qdrant_client
            from ml.search import search_many
        except Exception as exc:  # optional, heavyweight dependencies
            raise RuntimeError(f"ML/Qdrant backend недоступен: {exc}") from exc

        # Load once at startup, then reuse the model and Qdrant client for requests.
        get_encoder()
        client = get_qdrant_client()
        if not client.collection_exists(QDRANT_COLLECTION):
            raise RuntimeError(
                f"Коллекция Qdrant '{QDRANT_COLLECTION}' не найдена; "
                "сначала выполните POST /admin/reindex."
            )
        self._search_many = search_many
        self._reranker = _load_reranker()

    def classify(self, query: str) -> ClassifyResponse:
        return self.classify_many([query])[0]

    def classify_many(self, queries: list[str]) -> list[ClassifyResponse]:
        normalized_queries = [query.strip() for query in queries]
        if not normalized_queries:
            return []

        try:
            hit_groups = self._search_many(normalized_queries, top_k=10)
        except Exception:
            logger.exception("Qdrant batch search failed")
            raise

        try:
            from ml.reranker import RankedCandidate
        except Exception:  # pragma: no cover
            RankedCandidate = None

        results: list[ClassifyResponse] = []
        for query, hits in zip(normalized_queries, hit_groups, strict=True):
            if not hits:
                results.append(_not_found(query))
                continue

            if RankedCandidate is not None and self._reranker is not None:
                candidates = [
                    RankedCandidate(
                        code=hit.pp_code or None,
                        title=hit.name or "",
                        clause_ref=hit.position or hit.group or None,
                        retrieval_score=float(hit.score),
                        payload={"record_id": getattr(hit, "record_id", "") or ""},
                    )
                    for hit in hits
                ]
                ranked = self._reranker.rerank(query, candidates)
                best = ranked[0]
                raw_confidence, status = self._reranker.evaluate(query, ranked)
                raw_confidence = round(min(max(raw_confidence, 0.0), 1.0), 4)
                confidence = round(_display_confidence(self._reranker, raw_confidence), 4)
                threshold_review = getattr(self._reranker, "threshold_review", 0.45)
                alternatives = _distinct_alternatives(ranked, status, threshold_review)
                clause_ref = best.clause_ref
                clause_url = clause_url_for(best.payload.get("record_id"))
                code, title = best.code, best.title
                if status == "not_found":
                    code, title, clause_ref, clause_url = None, None, None, None
                    confidence, raw_confidence = 0.0, 0.0
            else:  # pragma: no cover - fallback when the reranker is unavailable
                top = hits[0]
                confidence = max(0.0, min(float(top.score), 1.0))
                raw_confidence = confidence
                status = "ok" if confidence >= 0.75 else "review"
                alternatives = [
                    Alternative(
                        code=hit.pp_code or None,
                        title=hit.name or None,
                        confidence=max(0.0, min(float(hit.score), 1.0)),
                    )
                    for hit in hits[1:3]
                ]
                clause_ref = top.position or top.group or None
                clause_url = clause_url_for(top.record_id)
                code, title = top.pp_code or None, top.name or None

            results.append(
                ClassifyResponse(
                    query=query,
                    code=code,
                    title=title,
                    confidence=confidence,
                    confidence_raw=raw_confidence,
                    status=status,
                    clause_ref=clause_ref,
                    # The source citation itself lives in clause_ref; the URL points
                    # at the fragment served by GET /source/{record_id}.
                    clause_url=clause_url,
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
