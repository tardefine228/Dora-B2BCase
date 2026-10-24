"""Classifiers for products covered by Russian Government Decree No. 1875.

Three backends share one response contract:

* ``catalog`` — matching against the checked-in ``data/data.csv``: a *hybrid*
  shortlist (lexical matcher + vector search over embeddings of the catalog names)
  scored by the calibrated reranker (default). Without an embedder it degrades to
  lexical-only;
* ``qdrant`` — semantic search in a prebuilt Qdrant collection (``ml`` package),
  reranked with the same calibrated reranker;
* ``mock`` — deterministic placeholder for frontend development.

Both real backends return the reranker's match probability as ``confidence`` and
map it to ``ok`` / ``review`` / ``not_found`` with thresholds from the active
reranker artifact. Catalog floors may be overridden with ``CATALOG_*_FLOOR``; the
semantic backend uses its own ``QDRANT_*_FLOOR`` pair. Results are suggestions for
a specialist, not a legal determination.
"""

from __future__ import annotations

import csv
import logging
import re
import time
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


def _parse_review_flag(raw: object) -> bool | None:
    """Catalog ``review_required`` (OCR doubt) as ``True``/``False``/``None`` (unknown)."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if not text:
        return None
    return text in {"true", "1", "yes", "on"}


def _display_confidence(reranker, raw_confidence: float) -> float:
    """Calibrated scale for display; the raw value still decides the status."""
    calibrate = getattr(reranker, "display_confidence", None)
    if not callable(calibrate):
        return raw_confidence
    return min(max(float(calibrate(raw_confidence)), 0.0), 1.0)


# How far a configured semantic floor may sit from the artifact's own trained
# threshold before it is treated as "measured on a different artifact". Both are
# group-confidence numbers on the same scale, so a large gap means the floor was
# taken from a measurement of another file, not from noise.
_SEMANTIC_FLOOR_DRIFT = 0.05

# Measured ``review`` floor of the semantic backend, on the same raw scale the
# shipped artifact produces for qdrant candidates. 0.45 is the optimum of the
# 274-row set and the median threshold of the 5x20-fold cross-validation
# (0.452), i.e. a plateau 0.42-0.45 rather than one cell of a table
# (``eval/experiments.md``, эксп. 28). Lowering it answers more but emits codes
# on goods the list does not contain: at 0.25 those false codes are 14 against 4.
_SEMANTIC_REVIEW_FLOOR = 0.45


def _apply_semantic_floors(reranker):
    """Retune the decision floors of a loaded reranker for semantic scores.

    The artifacts carry the point measured on the lexical backend (эксп. 14). The
    semantic backend answers on a compressed part of the same raw scale, so its
    ``review`` floor has to be measured separately (эксп. 16). Only the thresholds
    move: the weights and the Platt map are shared, ``status`` is decided on the
    raw scale, and the lexical path never goes through this function.
    """
    model = getattr(reranker, "confidence_model", None)
    if model is None:
        return reranker
    original = {"ok": model.threshold_ok, "review": model.threshold_review}
    floors = {
        "ok": settings.semantic_ok_floor,
        "review": settings.semantic_review_floor,
    }
    for name, floor in floors.items():
        if floor is not None:
            setattr(model, f"threshold_{name}", float(floor))
    if model.threshold_ok <= model.threshold_review:
        # A bad pair would make `ok` unreachable and silently break the contract.
        model.threshold_ok = original["ok"]
        model.threshold_review = original["review"]
        logger.warning(
            "Semantic floors ignored (ok %s <= review %s); artifact thresholds kept",
            floors["ok"],
            floors["review"],
        )
    elif (model.threshold_ok, model.threshold_review) != (original["ok"], original["review"]):
        logger.info(
            "Semantic floors applied: ok %s -> %s, review %s -> %s",
            original["ok"],
            model.threshold_ok,
            original["review"],
            model.threshold_review,
        )
        _warn_if_floor_drifted(reranker, original, model)
    return reranker


def _warn_if_floor_drifted(reranker, original: dict, model) -> None:
    """Say it loudly when the semantic floor answers more than was measured safe.

    The semantic raw scale is compressed, so its floor cannot be copied from the
    artifact: 0.45 is what the 274-row measurement and the cross-validation agree
    on (эксп. 28). Every answered-but-wrong code is a compliance risk, so a
    *lower* configured floor is worth a warning — a higher one only silences more
    and is a legitimate conservative choice.
    """
    if not getattr(reranker, "uses_dense", False):
        return
    if model.threshold_review >= _SEMANTIC_REVIEW_FLOOR - _SEMANTIC_FLOOR_DRIFT:
        return
    logger.warning(
        "Порог semantic review=%s ниже измеренного %.2f (артефакт обучен под review=%s): "
        "на 274 строках ложных кодов вне перечня 14 против 4, верных решений 146 против "
        "152. Порог не запрещён, но он отвечает чаще, чем измерено безопасно "
        "(eval/experiments.md, эксп. 28)",
        model.threshold_review,
        _SEMANTIC_REVIEW_FLOOR,
        original["review"],
    )


def _apply_catalog_floors(reranker):
    """Apply optional catalog floors without changing trained model weights.

    The group confidence model decides statuses when present; heuristic/legacy
    rerankers use their own thresholds. If either override makes ``ok <= review``,
    discard the pair and keep the artifact's calibrated operating point intact.
    """
    if reranker is None:
        return None
    model = getattr(reranker, "confidence_model", None)
    owner = model if model is not None else reranker
    if not hasattr(owner, "threshold_ok") or not hasattr(owner, "threshold_review"):
        return reranker

    original_ok = owner.threshold_ok
    original_review = owner.threshold_review
    ok_floor = settings.catalog_ok_floor
    review_floor = settings.catalog_review_floor
    if ok_floor is not None:
        owner.threshold_ok = float(ok_floor)
    if review_floor is not None:
        owner.threshold_review = float(review_floor)

    if owner.threshold_ok <= owner.threshold_review:
        owner.threshold_ok = original_ok
        owner.threshold_review = original_review
        logger.warning(
            "Catalog floors ignored (ok %s <= review %s); artifact thresholds kept",
            ok_floor,
            review_floor,
        )
    elif (owner.threshold_ok, owner.threshold_review) != (original_ok, original_review):
        logger.info(
            "Catalog floors applied: ok %s -> %s, review %s -> %s",
            original_ok,
            owner.threshold_ok,
            original_review,
            owner.threshold_review,
        )
    return reranker


def _qdrant_candidates(hits) -> list:
    """Qdrant hits as reranker candidates, with their semantic score attached.

    ``ml.reranker.dense_values()`` reads ``payload["dense"]`` and
    ``payload["dense_top"]``; without them the ``dense_cosine``/``dense_gap``
    pair features and the ``top_dense_*`` group features are always 0, so a
    reranker trained for the semantic backend could not influence anything.
    ``dense_top`` is the best cosine of the query over the whole catalog, which
    is what ``ml.dense.fuse_shortlist`` puts there for the catalog backend —
    training and serving must see the same definition.
    """
    from ml.reranker import RankedCandidate

    scores = [float(getattr(hit, "score", 0.0) or 0.0) for hit in hits]
    dense_top = max(scores) if scores else 0.0
    return [
        RankedCandidate(
            code=hit.pp_code or None,
            title=hit.name or "",
            clause_ref=hit.position or hit.group or None,
            retrieval_score=score,
            payload={
                "record_id": getattr(hit, "record_id", "") or "",
                "dense": score,
                "dense_top": dense_top,
                # Catalog facts from the point payload; None when the collection
                # was indexed before these keys existed (BE-9) — the classifier
                # then falls back to the catalog file.
                "code_role": getattr(hit, "code_role", None),
                "catalog_review_required": getattr(hit, "review_required", None),
            },
        )
        for hit, score in zip(hits, scores, strict=True)
    ]


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


def _codeless_hint(best, confidence: float) -> Alternative:
    """A top-1 position without an OKPD2 code, kept visible as a hint (BE-9).

    Contract invariant 5 forbids answering ``ok``/``review`` without a code, so
    such results are downgraded to ``not_found`` — but the position itself is
    valuable for the specialist (it exists in the decree, its code is simply
    «Не указан»), so it stays in ``alternatives`` with its title and source.
    """
    return Alternative(
        code=None,
        title=best.title,
        confidence=round(min(max(confidence, 0.0), 1.0), 4),
        clause_ref=best.clause_ref,
        clause_url=clause_url_for(best.payload.get("record_id")),
    )


def _anchored(candidate, query_stems: set[str] | None) -> bool:
    """Does the candidate position share a content stem with the query?"""
    if not query_stems:
        return False
    from ml.features import content_stems, stems

    return bool(query_stems & content_stems(set(stems(candidate.title))))


def _query_content_stems(query: str) -> set[str]:
    """Content stems of the query, with the lexicon bridges' canonical words."""
    from ml.features import content_stems, stems
    from ml.lexicon import bridged_targets

    values = set(stems(query))
    for target in bridged_targets(query).values():
        values |= set(stems(target))
    return content_stems(values)


def _distinct_alternatives(
    ranked, status: str, query_stems: set[str] | None = None
) -> list[Alternative]:
    """Up to ``_MAX_ALTERNATIVES`` runner-up positions with distinct codes.

    The same position is often listed in several appendices (and Qdrant stores
    one point per appendix row), so the raw ranking contains duplicates that
    would otherwise crowd out genuinely different suggestions.

    A ``not_found`` answer used to keep only candidates whose *pair* probability
    cleared ``threshold_review`` — but that threshold lives on the group
    confidence scale, and ``not_found`` already means the group confidence is
    below it, so the filter could never pass and the hint channel was dead: all
    108 out-of-catalog rows in the labeled set came back with an empty
    ``alternatives`` list even when the correct position ranked first. A lead
    after «нет в каталоге» is now kept when it is lexically anchored in the
    query — at least one content stem in common — so the specialist still gets
    ranked leads, while a dense near-miss with nothing in common («false
    friend») or a meaningless query («qwerty zxcvbn asdf») yields no invented
    hint. The answer's ``status`` remains ``not_found``: a lead is not a claim.
    """
    alternatives: list[Alternative] = []
    seen_codes = {ranked[0].code}
    for candidate in ranked[1:]:
        if candidate.code in seen_codes:
            continue
        if status == "not_found" and not _anchored(candidate, query_stems):
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


def _load_reranker(tag: str | None = None):
    """Return the trained reranker, or a lexical fallback when artifacts are absent.

    ``tag`` is the active embedder: the artifact trained with that embedder's vector
    features is preferred (see ``ml.reranker.load_reranker``).
    """
    try:
        from ml.reranker import load_reranker
    except Exception:  # ml package must never break the offline catalog backend
        logger.exception("ml.reranker недоступен — используется эвристический скоринг")
        return None
    try:
        return load_reranker(tag=tag)
    except Exception:
        logger.exception("Не удалось загрузить реранкер — используется эвристический скоринг")
        return None


def _reranker_artifact_name(reranker, tag: str | None) -> str | None:
    """Which artifact is actually serving decisions (for /health and logs)."""
    if reranker is None:
        return None
    try:
        from ml.reranker import reranker_artifact_name, reranker_artifact_path
    except Exception:  # pragma: no cover - ml import already failed in _load_reranker
        return None
    if getattr(reranker, "meta", {}).get("fallback"):
        return "heuristic"
    if tag:
        name = reranker_artifact_name(tag)
        if reranker_artifact_path(tag).is_file():
            return name
    return reranker_artifact_name()


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


def _build_dense_index(rows: list[dict]):
    """Vector index over the catalog names, or ``None`` when no embedder is available."""
    if not settings.vector_search:
        return None
    try:
        from ml.dense import DenseIndex, QdrantMirror

        mirror = None
        if settings.vector_store == "qdrant":
            try:
                mirror = QdrantMirror.connect(
                    settings.qdrant_url,
                    settings.qdrant_api_key,
                    f"{settings.qdrant_collection}_dense",
                )
            except Exception:  # noqa: BLE001 - fall back to the in-memory index
                logger.exception("Qdrant недоступен (%s) — векторы в памяти", settings.qdrant_url)
        cache_dir = Path(settings.data_dir).expanduser() / "dense_cache"
        return DenseIndex.build([row["name"] for row in rows], cache_dir=cache_dir, mirror=mirror)
    except Exception:  # noqa: BLE001 - the lexical matcher must keep working
        logger.exception("Векторный поиск недоступен — работаем только на лексике")
        return None


class CatalogClassifier:
    """Hybrid matching against the CSV catalog: lexical + vector retrieval, then reranker.

    ``dense`` is ``"auto"`` (build the index when ``VECTOR_SEARCH`` is on), ``None``
    (lexical only) or a ready ``ml.dense.DenseIndex`` aligned with ``self.index``.
    """

    def __init__(self, catalog_path: Path, reranker=None, dense="auto") -> None:
        self.catalog_path = Path(catalog_path)
        self.index: list[dict] = []
        self._load()
        if not self.index:
            raise ValueError(f"В каталоге нет записей: {self.catalog_path}")
        self._dense = _build_dense_index(self.index) if dense == "auto" else dense
        tag = self._dense.tag if self._dense is not None else None
        # Same escape hatch as the qdrant backend: a wrong or untested artifact must
        # be rollable back without rebuilding the image, and without losing vector
        # candidates (VECTOR_SEARCH=false would drop those too).
        reranker_tag = tag
        if tag and settings.semantic_reranker == "lexical":
            logger.warning(
                "SEMANTIC_RERANKER=lexical: каталоговый режим скорит %s, а не "
                "реранкером эмбеддера %s (векторы остаются кандидатами)",
                "reranker.json",
                tag,
            )
            reranker_tag = None
        loaded_reranker = (
            reranker if reranker is not None else _load_reranker(reranker_tag)
        )
        self._reranker = _apply_catalog_floors(loaded_reranker)
        logger.info(
            "CatalogClassifier загружен: %d позиций из %s (reranker=%s, vector=%s)",
            len(self.index),
            self.catalog_path,
            type(self._reranker).__name__ if self._reranker else "none",
            tag or "off",
        )
        self._reranker_name = _reranker_artifact_name(self._reranker, reranker_tag)

    @property
    def vector_info(self) -> dict:
        """Status of the vector layer for /health (what is actually serving requests)."""
        if self._dense is None:
            return {
                "enabled": False,
                "embedder": None,
                "dim": 0,
                "positions": 0,
                "store": None,
                # Which artifact scores is part of the contract even without vectors:
                # eval reports record it as provenance, and the rollback must be visible.
                "reranker": self._reranker_name,
                "scoring": "vector-features"
                if getattr(self._reranker, "uses_dense", False)
                else "lexical-reranker",
            }
        return {
            "enabled": True,
            "embedder": self._dense.tag,
            "dim": self._dense.dim,
            "positions": len(self._dense.names),
            "store": self._dense.store,
            "reranker": self._reranker_name,
            "scoring": "vector-features"
            if getattr(self._reranker, "uses_dense", False)
            else "lexical-reranker",
        }

    def ping_store(self) -> bool:
        """Bounded probe of the vector store behind the dense index (BE-6).

        Without a Qdrant mirror the index is exact in-memory search — there is
        nothing external to fail, so the store counts as up.
        """
        if self._dense is None:
            return True
        return bool(self._dense.health().get("store_up", True))

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
                review_flag = _parse_review_flag(row.get("review_required"))
                existing = by_key.get(key)
                if existing is not None:
                    # The same position often appears in several appendices; keep
                    # one entry and cite all of them so ranking margins stay sane.
                    if clause_ref and clause_ref not in (existing["clause_ref"] or ""):
                        existing["clause_ref"] = f"{existing['clause_ref']}; {clause_ref}"
                    # Data-quality signals merge conservatively: if ANY copy of the
                    # position is marked for re-verification, the merged entry is too.
                    if review_flag and not existing["catalog_review_required"]:
                        existing["catalog_review_required"] = True
                    continue
                by_key[key] = {
                    "okpd2": okpd2,
                    "name": name,
                    "appendix": (row.get("appendix") or "").strip(),
                    "clause_ref": clause_ref,
                    # Source of record: /source/{record_id} renders the pages of
                    # data/source_text.txt this position was parsed from.
                    "record_id": (row.get("record_id") or "").strip(),
                    # Catalog facts exposed as explicit flags in the response (BE-9).
                    # Informational only: whether «Исключение» positions may be
                    # answered and how OCR-doubt rows are processed are product and
                    # data decisions (PO-1/DA-6/DA-7), not classifier behavior.
                    "code_role": (row.get("code_role") or "").strip() or None,
                    "catalog_review_required": review_flag,
                    # Precomputed once: retrieval scores every row per request.
                    "prepared": prepare_candidate(name),
                }
        self.index = list(by_key.values())
        # Stem importance table for weighted query coverage at retrieval time.
        from ml.features import idf_table

        self._idf = idf_table([row["name"] for row in self.index])

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
        # Typos ("Монитр", "Клавитура") are corrected against the catalog
        # vocabulary first: stems and embeddings cannot rescue absent tokens.
        from ml.fuzzy import autocorrect

        return self._classify_normalized(autocorrect(query.strip()))

    def classify_many(self, queries: list[str]) -> list[ClassifyResponse]:
        """Batch path (BE-7): one autocorrect pass, ONE batched encoding call.

        With vector search the heavy part of a chunk is embedding the queries: a
        transformer amortizes its overhead only when the texts reach it as one
        batch, so all query vectors are computed up front and land in the
        ``DenseIndex`` query cache; the per-query pipeline then never encodes.
        Repeated names in a batch are encoded once as well.
        """
        if not queries:
            return []
        from ml.fuzzy import autocorrect

        normalized = [autocorrect(query.strip()) for query in queries]
        if self._dense is not None:
            try:
                self._dense.warm_queries([name for name in normalized if name])
            except Exception:  # noqa: BLE001 - per-query encoding still works
                logger.exception(
                    "Пакетное кодирование запросов не удалось — кодируем по одному"
                )
        return [self._classify_normalized(name) for name in normalized]

    def _classify_normalized(self, normalized_query: str) -> ClassifyResponse:
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
        if self._dense is not None:
            from ml.dense import fuse_shortlist
            from ml.features import prepare_candidate
            from ml.lexicon import expand_query

            fused = fuse_shortlist(
                self._dense,
                self.index,
                shortlist,
                normalized_query,
                prepare_candidate(expand_query(normalized_query), idf=self._idf),
            )
        else:
            fused = [(score, row, 0.0, 0.0) for score, row in shortlist]
        if not fused:
            return _not_found(normalized_query)

        return self._rank_to_response(normalized_query, fused)

    def _retrieve(self, query: str, query_tokens: set[str]) -> list[tuple[float, dict]]:
        """Lexical shortlist as ``(retrieval_score, row)`` pairs, best first."""
        from ml.features import lexical_score_prepared, prepare_candidate, stem
        from ml.lexicon import bridged_targets, expand_query

        prepared_query = prepare_candidate(expand_query(query), idf=self._idf)
        # A bridged product word ("принтер" -> "вывода") is a deliberate lexicon
        # decision: candidates containing the category word get a retrieval bonus
        # so abstract official names outrank stem-accidental junk ("печать" ~ "печи").
        bridge_stems = {stem(target) for target in set(bridged_targets(query).values())}
        scored: list[tuple[float, dict]] = []
        for row in self.index:
            score = lexical_score_prepared(prepared_query, row["prepared"])
            if bridge_stems and bridge_stems & row["prepared"].stem_set:
                score = min(1.0, score + 0.15)
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

    def _rank_to_response(self, query: str, shortlist: list[tuple]) -> ClassifyResponse:
        # Entries are ``(score, row)`` (lexical) or ``(score, row, cosine, dense_top)``.
        shortlist = [entry if len(entry) == 4 else (*entry, 0.0, 0.0) for entry in shortlist]
        try:
            from ml.reranker import RankedCandidate
        except Exception:  # pragma: no cover - fallback path
            logger.exception("ml.reranker недоступен — отдаём лучший lexical-кандидат")
            best = shortlist[0][1]
            if not best["okpd2"]:
                # Invariant 5 holds on every path: a codeless position is a hint,
                # not an answer.
                return ClassifyResponse(
                    query=query,
                    code=None,
                    title=None,
                    confidence=0.0,
                    confidence_raw=0.0,
                    status="not_found",
                    alternatives=[
                        Alternative(
                            code=None,
                            title=best["name"],
                            confidence=0.5,
                            clause_ref=best["clause_ref"],
                            clause_url=clause_url_for(best.get("record_id")),
                        )
                    ],
                    latency_ms=0.0,
                )
            return ClassifyResponse(
                query=query,
                code=best["okpd2"] or None,
                title=best["name"],
                confidence=0.5,
                status="review",
                clause_ref=best["clause_ref"],
                clause_url=clause_url_for(best.get("record_id")),
                code_role=best.get("code_role"),
                catalog_review_required=best.get("catalog_review_required"),
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
                payload={
                    "record_id": row.get("record_id") or "",
                    "dense": cosine,
                    "dense_top": dense_top,
                    # Catalog facts ride along so the response can expose them
                    # without a second lookup (BE-9).
                    "code_role": row.get("code_role"),
                    "catalog_review_required": row.get("catalog_review_required"),
                },
            )
            for score, row, cosine, dense_top in shortlist
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

        query_stems = _query_content_stems(query)

        if status == "not_found" or not best.code:
            # Too weak to suggest a position — or the winning position has no
            # OKPD2 code («Не указан») and invariant 5 forbids ok/review without
            # one. Report "не найдено"; keep at most plausible near-misses in
            # alternatives, and a demoted codeless winner stays visible as the
            # first hint (with its title and source) for the specialist.
            # A demoted codeless winner is not a below-threshold silence: the
            # query did hit the catalog, so its runner-ups stay unfiltered — only
            # genuine "nothing matched" answers need the lexical-anchor rule.
            alternatives = _distinct_alternatives(
                ranked, "not_found" if status == "not_found" else "review", query_stems
            )
            if status != "not_found":
                logger.info(
                    "Позиция без кода ОКПД2 понижена до not_found: query=%r title=%r",
                    query,
                    best.title,
                )
                alternatives = [
                    _codeless_hint(best, confidence),
                    *alternatives[: _MAX_ALTERNATIVES - 1],
                ]
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

        alternatives = _distinct_alternatives(ranked, status, query_stems)
        return ClassifyResponse(
            query=query,
            code=best.code,
            title=best.title,
            confidence=confidence,
            confidence_raw=raw_confidence,
            status=status,
            clause_ref=best.clause_ref,
            clause_url=clause_url_for(best.payload.get("record_id")),
            code_role=best.payload.get("code_role"),
            catalog_review_required=best.payload.get("catalog_review_required"),
            alternatives=alternatives,
            latency_ms=0.0,
        )


def _catalog_flags_lookup() -> dict[str, tuple[str | None, bool | None]]:
    """``record_id -> (code_role, review_required)`` from the catalog CSV.

    Fallback for Qdrant collections whose payload predates the catalog facts
    (BE-9); best effort — without the file the map stays empty and the flags in
    the response are simply ``null``.
    """
    flags: dict[str, tuple[str | None, bool | None]] = {}
    path = _resolve_catalog_path()
    if path is None:
        return flags
    try:
        with path.open(encoding="utf-8-sig", newline="") as csv_file:
            for row in csv.DictReader(csv_file, delimiter=";"):
                record_id = (row.get("record_id") or "").strip()
                if not record_id:
                    continue
                code_role = (row.get("code_role") or "").strip() or None
                review_flag = _parse_review_flag(row.get("review_required"))
                existing = flags.get(record_id)
                if existing is not None:
                    # Any copy marked for re-verification marks the position.
                    flags[record_id] = (
                        existing[0] or code_role,
                        True if (existing[1] or review_flag) else existing[1],
                    )
                else:
                    flags[record_id] = (code_role, review_flag)
    except OSError:
        logger.exception("Каталог для чтения code_role/review_required недоступен")
    return flags


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
        encoder = get_encoder()
        client = get_qdrant_client()
        if not client.collection_exists(QDRANT_COLLECTION):
            raise RuntimeError(
                f"Коллекция Qdrant '{QDRANT_COLLECTION}' не найдена; "
                "сначала выполните POST /admin/reindex."
            )
        self._search_many = search_many
        # Kept for the bounded health probe behind /metrics and /health/ready (BE-6).
        self._client = client
        # Collections indexed before the catalog facts landed in the payload carry
        # no code_role/review_required; the CSV on the shared volume fills the gap.
        self._catalog_flags = _catalog_flags_lookup()
        # How many points the collection holds (for /health). Counted once: a
        # reindex rebuilds the collection with the same number of rows, and a
        # probe must not turn into a network round trip on every poll.
        try:
            self._positions: int | None = int(
                client.count(collection_name=QDRANT_COLLECTION, exact=True).count
            )
        except Exception:  # noqa: BLE001 - the count is diagnostic only
            self._positions = None
        # The embedder tag keys the reranker artifact: cosine scales differ between
        # embedders, so a model trained with one set of vector features must never
        # score another. Without the tag the semantic backend silently served the
        # lexical artifact and its dense features stayed unused.
        self._embedder_tag: str | None = getattr(encoder, "tag", None)
        self._dim: int = int(getattr(encoder, "dim", 0) or 0)
        if settings.semantic_reranker == "lexical":
            logger.warning(
                "SEMANTIC_RERANKER=lexical: семантический режим использует %s "
                "вместо реранкера эмбеддера %s",
                "reranker.json",
                self._embedder_tag or "unknown",
            )
            reranker_tag = None
        else:
            reranker_tag = self._embedder_tag
        self._reranker = _apply_semantic_floors(_load_reranker(reranker_tag))
        self._reranker_name = _reranker_artifact_name(self._reranker, reranker_tag)
        model = getattr(self._reranker, "confidence_model", None)
        floors = (
            {"ok": model.threshold_ok, "review": model.threshold_review} if model else None
        )
        logger.info(
            "QdrantClassifier готов: embedder=%s dim=%d reranker=%s thresholds=%s",
            self._embedder_tag or "unknown",
            self._dim,
            self._reranker_name or "none",
            floors or "n/a",
        )
        if getattr(self._reranker, "uses_dense", False):
            logger.info(
                "Скоринг использует векторные признаки; пороги review/ok заданы "
                "QDRANT_REVIEW_FLOOR/QDRANT_OK_FLOOR и должны быть перемерены "
                "под активный эмбеддер (eval/threshold_sweep.py --backend qdrant)"
            )

    @property
    def vector_info(self) -> dict:
        """Status of the semantic layer for /health (what actually decides answers).

        The catalog backend exposes the same shape; before this property existed
        ``/health/ready`` answered ``{"enabled": false}`` for every request in the
        ``qdrant`` backend, i.e. the probe could not see the vector layer at all.
        """
        from ml.config import QDRANT_COLLECTION

        return {
            "enabled": True,
            "embedder": self._embedder_tag,
            "dim": self._dim,
            "positions": self._positions,
            "store": "qdrant",
            "collection": QDRANT_COLLECTION,
            "reranker": self._reranker_name,
            "scoring": "vector-features"
            if getattr(self._reranker, "uses_dense", False)
            else "lexical-reranker",
        }

    def ping_store(self) -> bool:
        """Bounded reachability probe of Qdrant for the health gauges (BE-6)."""
        client = getattr(self, "_client", None)
        if client is None:  # partially built instance: nothing to probe
            return False
        try:
            try:
                client.get_collections(timeout=2.0)
            except (AssertionError, TypeError):
                # Local (on-disk/in-memory) clients reject the timeout kwarg.
                client.get_collections()
            return True
        except Exception:  # noqa: BLE001 - an unreachable store is the answer
            return False

    def classify(self, query: str) -> ClassifyResponse:
        return self.classify_many([query])[0]

    def _catalog_flags_for(self, payload: dict) -> tuple[str | None, bool | None]:
        """``(code_role, review_required)`` of a hit: payload first, then the CSV."""
        code_role = payload.get("code_role")
        review_flag = payload.get("catalog_review_required")
        if code_role is None and review_flag is None:
            record_id = payload.get("record_id") or ""
            if record_id:
                # ``getattr``: partially built instances (tests via ``__new__``)
                # have no lookup map; flags then stay ``None``.
                fallback = getattr(self, "_catalog_flags", None) or {}
                code_role, review_flag = fallback.get(record_id, (None, None))
        return code_role, review_flag

    def classify_many(self, queries: list[str]) -> list[ClassifyResponse]:
        normalized_queries = [query.strip() for query in queries]
        if not normalized_queries:
            return []

        try:
            hit_groups = self._search_many(normalized_queries, top_k=10)
        except Exception:
            logger.exception("Qdrant batch search failed")
            raise

        results: list[ClassifyResponse] = []
        for query, hits in zip(normalized_queries, hit_groups, strict=True):
            if not hits:
                results.append(_not_found(query))
                continue

            code_role: str | None = None
            review_flag: bool | None = None
            if self._reranker is not None:
                candidates = _qdrant_candidates(hits)
                ranked = self._reranker.rerank(query, candidates)
                best = ranked[0]
                raw_confidence, status = self._reranker.evaluate(query, ranked)
                raw_confidence = round(min(max(raw_confidence, 0.0), 1.0), 4)
                confidence = round(_display_confidence(self._reranker, raw_confidence), 4)
                query_stems = _query_content_stems(query)
                code_role, review_flag = self._catalog_flags_for(best.payload)
                if status == "not_found" or not best.code:
                    # Invariant 5 (BE-9): ok/review without a code is forbidden, so
                    # a codeless winner («Не указан») is demoted and stays visible
                    # as the first alternative.
                    alternatives = _distinct_alternatives(
                        ranked, "not_found" if status == "not_found" else "review", query_stems
                    )
                    if status != "not_found":
                        logger.info(
                            "Позиция без кода ОКПД2 понижена до not_found: query=%r title=%r",
                            query,
                            best.title,
                        )
                        alternatives = [
                            _codeless_hint(best, confidence),
                            *alternatives[: _MAX_ALTERNATIVES - 1],
                        ]
                    results.append(
                        ClassifyResponse(
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
                    )
                    continue
                alternatives = _distinct_alternatives(ranked, status, query_stems)
                clause_ref = best.clause_ref
                clause_url = clause_url_for(best.payload.get("record_id"))
                code, title = best.code, best.title
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
                fallback = getattr(self, "_catalog_flags", None) or {}
                code_role, review_flag = fallback.get(top.record_id or "", (None, None))
                if code is None:  # pragma: no cover - invariant on the fallback path
                    alternatives = [
                        Alternative(
                            code=None,
                            title=title,
                            confidence=confidence,
                            clause_ref=clause_ref,
                            clause_url=clause_url,
                        ),
                        *alternatives[: _MAX_ALTERNATIVES - 1],
                    ]
                    results.append(
                        ClassifyResponse(
                            query=query,
                            code=None,
                            title=None,
                            confidence=0.0,
                            confidence_raw=0.0,
                            status="not_found",
                            alternatives=alternatives,
                            latency_ms=0.0,
                        )
                    )
                    continue

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
                    code_role=code_role,
                    catalog_review_required=review_flag,
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
_singleton_signature: tuple | None = None
_singleton_lock = RLock()


def _catalog_signature() -> tuple | None:
    """``(path, mtime_ns, size)`` of the catalog file in use (``None`` for non-catalog backends)."""
    if settings.classifier_backend != "catalog":
        return None
    path = _resolve_catalog_path()
    if path is None:
        return None
    stat = path.stat()
    return (str(path), stat.st_mtime_ns, stat.st_size)


def get_classifier() -> CachedClassifier:
    """Return the process-wide classifier and its bounded response cache.

    The catalog may be replaced at runtime (``POST /admin/catalog``) by *another*
    process sharing the data volume (the API and the Celery worker are separate), so
    the catalog file's signature is checked on every call (one ``stat``) and the
    classifier, including its vector index, is rebuilt when it changes.
    """
    global _singleton, _singleton_signature
    with _singleton_lock:
        signature = _catalog_signature()
        if _singleton is not None and signature != _singleton_signature:
            logger.info("Каталог изменился на диске — пересобираем классификатор")
            _singleton = None
        if _singleton is None:
            base_classifier = build_classifier()
            _singleton = CachedClassifier(
                base_classifier,
                max_entries=settings.classification_cache_size,
            )
            _singleton_signature = signature
        return _singleton


def reset_classifier() -> None:
    """Reset the singleton (primarily used by tests and catalog reload tooling)."""
    global _singleton, _singleton_signature, _health_cache
    with _singleton_lock:
        _singleton = None
        _singleton_signature = None
    with _health_lock:
        _health_cache = None


# --- health of the embedder and the vector store (BE-6) ----------------------
#
# /metrics scrapes and the Docker healthcheck poll /health/ready every ~10 s.
# The embedder part of the probe reads already-loaded state (free); the store
# part can be a network ping, so the whole snapshot is cached for a short TTL.
_HEALTH_TTL_SECONDS = 10.0
_health_cache: tuple[float, dict] | None = None
_health_lock = RLock()


def _probe_vector_health() -> dict:
    """One uncached measurement of the vector layer's live state."""
    health: dict = {
        "classifier_up": False,
        "vector_search_enabled": False,
        "embedder_up": False,
        "embedder": None,
        "dim": 0,
        "store": None,
        "store_up": False,
        "positions": 0,
        "degraded": False,
    }
    try:
        classifier = get_classifier()
    except Exception as exc:  # noqa: BLE001 - a broken classifier is the finding
        health["error"] = str(exc)
        return health
    health["classifier_up"] = True

    inner = getattr(classifier, "_classifier", None)
    info = getattr(inner, "vector_info", None)
    info = info if isinstance(info, dict) else {"enabled": False}
    health["vector_search_enabled"] = bool(info.get("enabled"))
    health["embedder"] = info.get("embedder")
    health["dim"] = int(info.get("dim") or 0)
    health["store"] = info.get("store")
    health["positions"] = int(info.get("positions") or 0)

    # What the configuration *asked for*; silent degradation (a vector layer that
    # wanted to be there but failed to load) is exactly what operators must see.
    wants_vectors = settings.vector_search and settings.classifier_backend == "catalog"
    wants_qdrant = settings.vector_store == "qdrant"

    ping_store = getattr(inner, "ping_store", None)
    if health["vector_search_enabled"]:
        health["embedder_up"] = bool(info.get("embedder"))
        health["store_up"] = bool(ping_store()) if callable(ping_store) else True
        if health["store"] == "memory" and (
            wants_qdrant or settings.classifier_backend == "qdrant"
        ):
            # The mirror dropped to memory after a Qdrant failure (or never
            # connected): the requested store is not serving.
            health["store_up"] = False
    else:
        # No vector layer: healthy when it was not requested; degraded when
        # VECTOR_SEARCH=true but no embedder loaded (lexical-only fallback).
        health["embedder_up"] = not wants_vectors
        health["store_up"] = not (wants_vectors and wants_qdrant)

    health["degraded"] = (wants_vectors and not health["vector_search_enabled"]) or not health[
        "store_up"
    ]
    return health


def vector_layer_health(force: bool = False) -> dict:
    """TTL-cached health of the embedder and the vector store.

    Cheap enough to call on every scrape and readiness poll: inside the TTL the
    cached snapshot is returned, so a dead Qdrant adds at most one bounded ping
    (2 s) per interval, never one per request.
    """
    global _health_cache
    now = time.monotonic()
    with _health_lock:
        if not force and _health_cache is not None and now - _health_cache[0] < _HEALTH_TTL_SECONDS:
            return _health_cache[1]
    health = _probe_vector_health()
    with _health_lock:
        _health_cache = (now, health)
    return health
