"""Feature-based reranker for PP-1875 candidate positions.

Retrieval (lexical or Qdrant) shortlists candidates; this reranker reorders them
with a small calibrated logistic model over interpretable pair features and
returns a match probability. The probability doubles as the ``confidence`` of
the classification and is compared against calibrated decision thresholds:

* ``p >= threshold_ok``     → status ``ok``
* ``p >= threshold_review`` → status ``review``
* otherwise                 → the match is too weak to suggest (``not_found``
  when nothing better exists)

Weights and thresholds are produced by ``python scripts/build_model.py`` and
stored in ``models/reranker.json`` (see ``ml/train_reranker.py``). Inference is
dependency-free; training needs only numpy.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ml.config import MODELS_DIR
from ml.features import (
    LOW_SIGNAL_STEMS,
    best_match_weight,
    char_ngrams,
    content_stems,
    cosine_sparse,
    coverage,
    idf_coverage,
    idf_precision,
    is_latin_word,
    jaccard_digits,
    lexical_score,
    normalize,
    oov_cyr_share,
    query_head,
    stem,
    stems,
    tokens,
)

logger = logging.getLogger("dora.ml.reranker")

RERANKER_ARTIFACT_NAME = "reranker.json"
ARTIFACT_VERSION = 2

FEATURE_NAMES: tuple[str, ...] = (
    "lexical_score",
    "idf_coverage",
    "known_coverage",
    "candidate_precision",
    "idf_precision",
    "char_cosine",
    "phrase_match",
    "full_cover",
    "bridge_cover",
    "bridge_canonical",
    "title_head_match",
    "query_head_match",
    "digit_overlap",
    "oov_cyr_share",
    "code_match",
)

# Second stage: confidence that the top-1 candidate is the right position for
# the whole query (short queries look weak per-pair even when the leader is
# clearly ahead of the alternatives).
GROUP_FEATURE_NAMES: tuple[str, ...] = (
    "pair_probability",
    "probability_margin",
    "softmax_share",
    "top_idf_coverage",
    "top_known_coverage",
    "top_query_head_match",
    "retrieval_margin",
)
SOFTMAX_TEMPERATURE = 0.15


@dataclass
class RankedCandidate:
    """A candidate position with its rerank probability and feature values."""

    code: str | None
    title: str
    clause_ref: str | None
    retrieval_score: float
    probability: float = 0.0
    features: dict[str, float] = field(default_factory=dict)
    payload: dict = field(default_factory=dict)


def extract_features(
    query: str,
    candidate_title: str,
    candidate_code: str = "",
    known_stems: set[str] | None = None,
    idf: dict[str, float] | None = None,
) -> dict[str, float]:
    """Interpretable similarity features for one (query, candidate) pair.

    ``known_stems`` is the catalog vocabulary: brand names and typos in real
    queries are out-of-vocabulary, and coverage is measured over the known part
    of the query so that "Ноутбук Lenovo" still matches the portable-computers
    position. ``idf`` downweights ubiquitous words ("электрический") in favor of
    rare, decisive ones ("чайник"). Product words are bridged to catalog
    category words first (see ``ml.lexicon``).
    """
    from ml.lexicon import CANONICAL_CODES, bridged_targets, substitute_query

    # The candidate is the canonical position of a bridged product word
    # ("сапоги" → "Обувь" 15.2, not "Обувь ортопедическая детская").
    bridge_canonical = 0.0
    if candidate_code:
        for target in set(bridged_targets(query).values()):
            canonical = CANONICAL_CODES.get(target)
            if canonical and (
                candidate_code == canonical or candidate_code.startswith(canonical + ".")
            ):
                bridge_canonical = 1.0
                break

    original_stems = set(stems(query))
    candidate_stems = stems(candidate_title)
    # Bridged product words are *replaced* by their category words so that
    # "костюм" is not an unknown Russian word penalizing the match (a product
    # word the candidate itself mentions is kept).
    query = substitute_query(query, set(candidate_stems))
    query_stems = stems(query)
    added_stems = set(query_stems) - original_stems
    bridge_cover = 0.0
    if added_stems:
        bridge_cover = max(best_match_weight(added, set(candidate_stems)) for added in added_stems)
    query_cover, candidate_precision = coverage(query, candidate_title)
    if known_stems is not None and query_stems:
        known_query = [
            stem
            for stem in content_stems(set(query_stems))
            # Latin unknowns are brands/model indices and unknown low-signal
            # modifiers ("женский") describe a variant; Cyrillic unknowns stay.
            if stem in known_stems or not (is_latin_word(stem) or stem in LOW_SIGNAL_STEMS)
        ]
        if known_query:
            known_hits = sum(best_match_weight(stem, set(candidate_stems)) for stem in known_query)
            known_coverage = known_hits / len(known_query)
        else:
            known_coverage = 0.0
    else:
        known_coverage = query_cover

    normalized_query = normalize(query)
    normalized_candidate = normalize(candidate_title)
    phrase_match = float(
        bool(normalized_query)
        and bool(normalized_candidate)
        and (normalized_query in normalized_candidate or normalized_candidate in normalized_query)
    )
    # The query is entirely explained by this position ("Материалы
    # лакокрасочные" for the long paint-materials name).
    full_cover = 0.0
    if query_stems:
        covered = [
            best_match_weight(stem_value, set(candidate_stems))
            for stem_value in content_stems(set(query_stems))
        ]
        full_cover = float(all(weight >= 0.7 for weight in covered))

    title_head_match = 0.0
    candidate_tokens = tokens(candidate_title)
    if candidate_tokens:
        title_head_match = best_match_weight(stem(candidate_tokens[0]), set(query_stems))

    # Russian product names lead with the head noun ("Мышь компьютерная"): a
    # candidate that matches only the modifiers ("Томограф компьютерный") is a
    # different product even when coverage looks decent.
    query_head_match = 0.0
    head = query_head(query)
    if head is not None:
        query_head_match = best_match_weight(head, set(candidate_stems))

    code_match = 0.0
    normalized_code = normalize(candidate_code or "")
    if normalized_code:
        code_match = float(normalized_code in normalized_query.replace(" ", ""))
        if code_match == 0.0:
            # Partial code matches ("26.20" for "26.20.11") still count a little.
            code_match = 0.5 * float(normalized_query.replace(" ", "").startswith(normalized_code))
    return {
        "lexical_score": lexical_score(query, candidate_title),
        "idf_coverage": idf_coverage(query, candidate_title, idf) if idf else query_cover,
        "known_coverage": known_coverage,
        "candidate_precision": candidate_precision,
        "idf_precision": idf_precision(query, candidate_title, idf) if idf else candidate_precision,
        "char_cosine": cosine_sparse(char_ngrams(query), char_ngrams(candidate_title)),
        "phrase_match": phrase_match,
        "full_cover": full_cover,
        "bridge_cover": bridge_cover,
        "bridge_canonical": bridge_canonical,
        "title_head_match": title_head_match,
        "query_head_match": query_head_match,
        "digit_overlap": jaccard_digits(query, candidate_title),
        "oov_cyr_share": oov_cyr_share(set(query_stems), idf) if idf else 0.0,
        "code_match": code_match,
    }


class Reranker(Protocol):
    def score(self, query: str, candidate_title: str, candidate_code: str = "") -> float: ...


def _sigmoid(linear: float) -> float:
    if linear >= 0:
        return 1.0 / (1.0 + pow(2.718281828459045, -linear))
    exponent = pow(2.718281828459045, linear)
    return exponent / (1.0 + exponent)


def extract_group_features(query: str, ranked: list[RankedCandidate]) -> dict[str, float]:
    """Query-level features describing how confident the ranking itself is."""
    if not ranked:
        return {name: 0.0 for name in GROUP_FEATURE_NAMES}
    # Duplicate catalog rows (same position listed in several appendices) must
    # not zero out the margin between distinct candidates.
    distinct: list[RankedCandidate] = []
    seen: set[tuple[str | None, str]] = set()
    for candidate in ranked:
        key = (candidate.code, candidate.title)
        if key in seen:
            continue
        seen.add(key)
        distinct.append(candidate)

    top = distinct[0]
    probabilities = [candidate.probability for candidate in distinct[:5]]
    temperature = SOFTMAX_TEMPERATURE
    exp_values = [pow(2.718281828459045, value / temperature) for value in probabilities]
    softmax_share = exp_values[0] / sum(exp_values) if exp_values else 0.0

    second_probability = distinct[1].probability if len(distinct) > 1 else 0.0
    retrieval_scores = [candidate.retrieval_score for candidate in distinct]
    second_retrieval = retrieval_scores[1] if len(retrieval_scores) > 1 else 0.0
    return {
        "pair_probability": top.probability,
        "probability_margin": top.probability - second_probability,
        "softmax_share": softmax_share,
        "top_idf_coverage": top.features.get("idf_coverage", 0.0),
        "top_known_coverage": top.features.get("known_coverage", 0.0),
        "top_query_head_match": top.features.get("query_head_match", 0.0),
        "retrieval_margin": retrieval_scores[0] - second_retrieval,
    }


class ConfidenceModel:
    """Logistic model over GROUP_FEATURE_NAMES; calibrated in train_reranker."""

    def __init__(
        self,
        weights: dict[str, float],
        intercept: float,
        threshold_ok: float,
        threshold_review: float,
    ) -> None:
        unknown = set(weights) - set(GROUP_FEATURE_NAMES)
        if unknown:
            raise ValueError(f"Неизвестные признаки модели уверенности: {sorted(unknown)}")
        self.weights = {name: float(weights.get(name, 0.0)) for name in GROUP_FEATURE_NAMES}
        self.intercept = float(intercept)
        self.threshold_ok = float(threshold_ok)
        self.threshold_review = float(threshold_review)

    def probability(self, features: dict[str, float]) -> float:
        linear = self.intercept + sum(
            self.weights[name] * features[name] for name in GROUP_FEATURE_NAMES
        )
        return _sigmoid(linear)

    def to_payload(self) -> dict:
        return {
            "weights": {name: round(self.weights[name], 6) for name in GROUP_FEATURE_NAMES},
            "intercept": round(self.intercept, 6),
            "thresholds": {
                "ok": round(self.threshold_ok, 4),
                "review": round(self.threshold_review, 4),
            },
        }

    @classmethod
    def from_payload(cls, payload: dict) -> ConfidenceModel:
        thresholds = payload.get("thresholds", {})
        return cls(
            weights=payload.get("weights", {}),
            intercept=float(payload.get("intercept", 0.0)),
            threshold_ok=float(thresholds.get("ok", 0.75)),
            threshold_review=float(thresholds.get("review", 0.45)),
        )


class LogisticReranker:
    """Logistic regression over ``FEATURE_NAMES``; pure Python at inference."""

    def __init__(
        self,
        weights: dict[str, float],
        intercept: float,
        threshold_ok: float,
        threshold_review: float,
        meta: dict | None = None,
        known_stems: set[str] | None = None,
        confidence_model: ConfidenceModel | None = None,
        idf: dict[str, float] | None = None,
    ) -> None:
        unknown = set(weights) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"Неизвестные признаки в артефакте реранкера: {sorted(unknown)}")
        self.weights = {name: float(weights.get(name, 0.0)) for name in FEATURE_NAMES}
        self.intercept = float(intercept)
        self.threshold_ok = float(threshold_ok)
        self.threshold_review = float(threshold_review)
        self.meta = meta or {}
        self.known_stems = known_stems
        self.confidence_model = confidence_model
        self.idf = idf

    def probability(self, features: dict[str, float]) -> float:
        linear = self.intercept + sum(self.weights[name] * features[name] for name in FEATURE_NAMES)
        return _sigmoid(linear)

    def score(self, query: str, candidate_title: str, candidate_code: str = "") -> float:
        return self.probability(
            extract_features(query, candidate_title, candidate_code, self.known_stems, self.idf)
        )

    def rerank(self, query: str, candidates: list[RankedCandidate]) -> list[RankedCandidate]:
        """Score and stably sort candidates by descending match probability."""
        for candidate in candidates:
            candidate.features = extract_features(
                query, candidate.title, candidate.code or "", self.known_stems, self.idf
            )
            candidate.probability = self.probability(candidate.features)
        candidates.sort(key=lambda item: (-item.probability, item.title))
        return candidates

    def evaluate(self, query: str, ranked: list[RankedCandidate]) -> tuple[float, str]:
        """Return (confidence, status) for a whole query after reranking."""
        if not ranked:
            return 0.0, "not_found"
        if self.confidence_model is None:
            confidence = min(max(ranked[0].probability, 0.0), 1.0)
            return confidence, self.status_for(ranked[0].probability)
        features = extract_group_features(query, ranked)
        confidence = self.confidence_model.probability(features)
        if confidence >= self.confidence_model.threshold_ok:
            status = "ok"
        elif confidence >= self.confidence_model.threshold_review:
            status = "review"
        else:
            status = "not_found"
        return min(max(confidence, 0.0), 1.0), status

    def status_for(self, probability: float) -> str:
        if probability >= self.threshold_ok:
            return "ok"
        if probability >= self.threshold_review:
            return "review"
        return "not_found"

    def save(self, path: Path) -> None:
        payload = {
            "version": ARTIFACT_VERSION,
            "model": "logistic",
            "feature_names": list(FEATURE_NAMES),
            "group_feature_names": list(GROUP_FEATURE_NAMES),
            "weights": {name: round(self.weights[name], 6) for name in FEATURE_NAMES},
            "intercept": round(self.intercept, 6),
            "thresholds": {
                "ok": round(self.threshold_ok, 4),
                "review": round(self.threshold_review, 4),
            },
            "confidence_model": self.confidence_model.to_payload()
            if self.confidence_model
            else None,
            "known_stems": sorted(self.known_stems) if self.known_stems else None,
            "stem_idf": (
                {stem: round(value, 4) for stem, value in sorted(self.idf.items())}
                if self.idf
                else None
            ),
            "meta": self.meta,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LogisticReranker:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model") != "logistic":
            raise ValueError(f"Неизвестный формат артефакта реранкера: {path}")
        if payload.get("version") != ARTIFACT_VERSION:
            logger.warning(
                "Артефакт реранкера %s версии %s, код ожидает %s — новые признаки получат "
                "нулевой вес; пересоберите модель (python scripts/build_model.py)",
                path,
                payload.get("version"),
                ARTIFACT_VERSION,
            )
        thresholds = payload.get("thresholds", {})
        known = payload.get("known_stems")
        confidence_payload = payload.get("confidence_model")
        idf_payload = payload.get("stem_idf")
        return cls(
            weights=payload.get("weights", {}),
            intercept=float(payload.get("intercept", 0.0)),
            threshold_ok=float(thresholds.get("ok", 0.75)),
            threshold_review=float(thresholds.get("review", 0.45)),
            meta=payload.get("meta", {}),
            known_stems=set(known) if known else None,
            confidence_model=ConfidenceModel.from_payload(confidence_payload)
            if confidence_payload
            else None,
            idf={stem: float(value) for stem, value in idf_payload.items()}
            if idf_payload
            else None,
        )


class HeuristicReranker:
    """Fallback scorer when no trained artifact exists: the lexical score itself.

    The lexical score is compressed into roughly [0, 1] with most real matches
    between 0.2 and 0.8, so its thresholds differ from the calibrated logistic
    probabilities produced by ``ml.train_reranker``.
    """

    def __init__(self, threshold_ok: float = 0.35, threshold_review: float = 0.12) -> None:
        self.threshold_ok = threshold_ok
        self.threshold_review = threshold_review
        self.meta = {"fallback": "lexical"}

    def score(self, query: str, candidate_title: str, candidate_code: str = "") -> float:
        return lexical_score(query, candidate_title)

    def rerank(self, query: str, candidates: list[RankedCandidate]) -> list[RankedCandidate]:
        for candidate in candidates:
            candidate.features = extract_features(query, candidate.title, candidate.code or "")
            candidate.probability = self.score(query, candidate.title, candidate.code or "")
        candidates.sort(key=lambda item: (-item.probability, item.title))
        return candidates

    def evaluate(self, query: str, ranked: list[RankedCandidate]) -> tuple[float, str]:
        if not ranked:
            return 0.0, "not_found"
        confidence = min(max(ranked[0].probability, 0.0), 1.0)
        return confidence, self.status_for(confidence)

    def status_for(self, probability: float) -> str:
        if probability >= self.threshold_ok:
            return "ok"
        if probability >= self.threshold_review:
            return "review"
        return "not_found"


def reranker_artifact_path() -> Path:
    return MODELS_DIR / RERANKER_ARTIFACT_NAME


def load_reranker(path: Path | None = None) -> LogisticReranker | HeuristicReranker:
    """Load the trained artifact when present; otherwise use the heuristic."""
    artifact = path or reranker_artifact_path()
    if artifact.is_file():
        reranker = LogisticReranker.load(artifact)
        logger.info(
            "Реранкер загружен из %s (пороги ok=%.2f review=%.2f)",
            artifact,
            reranker.threshold_ok,
            reranker.threshold_review,
        )
        return reranker
    logger.warning(
        "Артефакт реранкера %s не найден — используется эвристический скоринг. "
        "Выполните python scripts/build_model.py для калибровки.",
        artifact,
    )
    return HeuristicReranker()
