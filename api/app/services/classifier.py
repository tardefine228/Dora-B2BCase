"""Классификаторы товаров по ПП № 1875.

- ``CatalogClassifier`` — рабочий базовый классификатор поверх реального
  каталога ``data/data.csv`` (эвристика на основе пересечения токенов).
  Это и есть «реальный» ответ интерфейса на демонстрационных данных до
  подключения ML-модели.
- ``MockClassifier`` — заглушка для фронтенда (детерминированный MOCK-001).
- ``QdrantClassifier`` — адаптер к ML-поиску (``ml.search``); подключается,
  когда ML-разработчик соберёт индекс в Qdrant.
"""

import csv
import logging
import re
from pathlib import Path
from typing import Protocol

from app.core.config import settings
from app.schemas import Alternative, ClassifyResponse

logger = logging.getLogger("dora.classifier")

# Код ОКПД2, например 14.1 или 26.20.11
OKPD2_RE = re.compile(r"^\d{2}(\.\d{1,2}){0,3}$")
TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if len(t) > 1]


class Classifier(Protocol):
    def classify(self, query: str) -> ClassifyResponse: ...


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


class MockClassifier:
    """Временная заглушка для разработки фронтенда."""

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
    """Реальный (базовый) классификатор по каталогу ПП № 1875."""

    def __init__(self, catalog_path: Path) -> None:
        self.catalog_path = Path(catalog_path)
        self.index: list[dict] = []
        self._load()
        logger.info(
            "CatalogClassifier загружен: %d позиций из %s",
            len(self.index),
            self.catalog_path,
        )

    def _load(self) -> None:
        with self.catalog_path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
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
                        "text": name.lower(),
                    }
                )

    @staticmethod
    def _token_matches(qtok: str, tokens: set, text: str) -> bool:
        if qtok in tokens:
            return True
        if qtok in text:
            return True
        # частичное совпадение (единственное/множественное число и т.п.)
        for t in tokens:
            if t and (t in qtok or qtok in t):
                return True
        return False

    def classify(self, query: str) -> ClassifyResponse:
        q = (query or "").strip()
        if not q:
            return _not_found(q)

        q_lc = q.lower()

        # 1. Прямое указание ОКПД2-кода
        m = OKPD2_RE.match(q_lc.replace(" ", ""))
        if m:
            code = m.group(0)
            for r in self.index:
                if r["okpd2"] == code:
                    return ClassifyResponse(
                        query=q,
                        code=r["okpd2"],
                        title=r["name"],
                        confidence=1.0,
                        status="ok",
                        clause_ref=f"Приложение {r['appendix']}" if r["appendix"] else None,
                        clause_url=None,
                        alternatives=[],
                        latency_ms=0.0,
                    )

        # 2. Совпадение по названию (токены)
        qtokens = set(_tokenize(q_lc))
        if not qtokens:
            return _not_found(q)

        scored = []
        for r in self.index:
            matched = [qt for qt in qtokens if self._token_matches(qt, r["tokens"], r["text"])]
            if not matched:
                continue
            scored.append((len(matched) / len(qtokens), r))

        if not scored:
            return _not_found(q)

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best = scored[0]
        seen_codes = {best["okpd2"]}
        alternatives = []
        for s, r in scored[1:4]:
            if r["okpd2"] in seen_codes:
                continue
            seen_codes.add(r["okpd2"])
            alternatives.append(
                Alternative(
                    code=r["okpd2"] or None,
                    title=r["name"],
                    confidence=round(min(s, 1.0), 4),
                )
            )
        conf = round(min(best_score, 1.0), 4)
        status = "ok" if conf >= 0.5 else "review"
        return ClassifyResponse(
            query=q,
            code=best["okpd2"] or None,
            title=best["name"],
            confidence=conf,
            status=status,
            clause_ref=f"Приложение {best['appendix']}" if best["appendix"] else None,
            clause_url=None,
            alternatives=alternatives,
            latency_ms=0.0,
        )


class QdrantClassifier:
    """Адаптер к ML-семантическому поиску (требует собранного индекса Qdrant)."""

    def __init__(self) -> None:
        try:
            from ml.search import search as ml_search  # lazy import тяжёлых зависимостей
        except Exception as exc:  # pragma: no cover - зависит от окружения ML
            raise RuntimeError(f"Qdrant/ML backend недоступен: {exc}") from exc
        self._search = ml_search

    def classify(self, query: str) -> ClassifyResponse:
        q = (query or "").strip()
        try:
            hits = self._search(q, top_k=3)
        except Exception:  # pragma: no cover
            logger.exception("Qdrant search failed")
            return _not_found(q)
        if not hits:
            return _not_found(q)

        top = hits[0]
        conf = max(0.0, min(float(top.score), 1.0))
        alternatives = [
            Alternative(
                code=h.pp_code or None,
                title=h.name or None,
                confidence=round(min(float(h.score), 1.0), 4),
            )
            for h in hits[1:3]
        ]
        status = "ok" if conf >= 0.75 else "review"
        return ClassifyResponse(
            query=q,
            code=top.pp_code or None,
            title=top.name or None,
            confidence=round(conf, 4),
            status=status,
            clause_ref=None,
            clause_url=top.pp_link or None,
            alternatives=alternatives,
            latency_ms=0.0,
        )


def _resolve_catalog_path() -> Path | None:
    candidates: list[Path] = []
    if settings.catalog_path:
        candidates.append(Path(settings.catalog_path))
    candidates.append(Path(settings.data_dir) / "data.csv")
    candidates.append(Path("data") / "data.csv")
    candidates.append(Path(__file__).resolve().parents[3] / "data" / "data.csv")
    for c in candidates:
        if c.exists():
            return c
    return None


def build_classifier(backend: str | None = None) -> Classifier:
    backend = (backend or settings.classifier_backend).lower()
    if backend == "mock":
        return MockClassifier()
    if backend == "qdrant":
        return QdrantClassifier()
    # catalog (по умолчанию)
    path = _resolve_catalog_path()
    if path is None:
        logger.error("Файл каталога не найден — откат к MockClassifier")
        return MockClassifier()
    try:
        return CatalogClassifier(path)
    except Exception:  # pragma: no cover
        logger.exception("Не удалось построить CatalogClassifier — откат к mock")
        return MockClassifier()


_singleton: Classifier | None = None


def get_classifier() -> Classifier:
    global _singleton
    if _singleton is None:
        _singleton = build_classifier()
    return _singleton


def reset_classifier() -> None:
    global _singleton
    _singleton = None
