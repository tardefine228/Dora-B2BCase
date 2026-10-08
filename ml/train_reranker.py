"""Train and calibrate the reranker + confidence model on the PP-1875 catalog.

Training data is synthesized deterministically from the catalog itself: every
position name is turned into realistic query variants (short word subsets,
brand/model noise, word order shuffles, typos), and hard negatives come from the
same lexical retrieval the API uses at inference time.

Two models are fitted:

1. a pairwise logistic reranker that orders candidate positions;
2. a query-level confidence model over ranking features (leader probability,
   margin, softmax share) — this is what the API reports as ``confidence`` and
   what is compared against the calibrated ``ok``/``review`` thresholds.

The human-labeled set ``eval/labeled_examples.csv`` is intentionally *not* used
for training or calibration — it stays a clean measurement set (see
``eval/run_eval.py``).

Usage:
    python -m ml.train_reranker --output models/reranker.json
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from ml.catalog_loader import load_catalog
from ml.features import (
    build_known_stems,
    idf_table,
    lexical_score_prepared,
    prepare_candidate,
    stem,
    stems,
    tokens,
)
from ml.lexicon import expand_query
from ml.reranker import (
    FEATURE_NAMES,
    GROUP_FEATURE_NAMES,
    ConfidenceModel,
    LogisticReranker,
    RankedCandidate,
    extract_features,
    extract_group_features,
)

SEED = 1875
NOISE_WORDS = (
    "новый",
    "оригинальный",
    "бренд",
    "модель",
    "для дома",
    "в упаковке",
    "импорт",
    "рос",
    "купить",
    "оптом",
    "ремонт",
    "замена",
    # Real modifiers that legitimately do not appear in category names.
    "мужской",
    "женский",
    "детский",
    "бытовой",
    "промышленный",
    "настольный",
    "напольный",
    "настенный",
    "для работы",
    "хлопковый",
    "пластиковый",
    "стальной",
)
# Real product queries carry brands and model indices that never appear in the
# official position names; the reranker must learn to ignore them.
BRAND_WORDS = (
    "lenovo",
    "asus",
    "bosch",
    "xiaomi",
    "samsung",
    "haier",
    "tcl",
    "acer",
    "makita",
    "philips",
    "lg",
    "hp",
    "huawei",
    "dexp",
    "irbis",
    "prestigio",
    "oppo",
    "vitek",
    "scarlett",
    "redmond",
    "polaris",
    "electrolux",
    "tesla",
)
MODEL_WORDS = (
    "x200",
    "pro",
    "max",
    "s230",
    "v12",
    "e27",
    "220v",
    "15w",
    "ultra",
    "plus",
    "mini",
    "air",
    "lite",
    "master",
    "eco",
    "m2",
    "t7",
)


@dataclass
class TrainingPair:
    query: str
    title: str
    code: str
    label: int
    retrieval_score: float = 0.0


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 4:
        return word
    index = rng.randrange(1, len(word) - 1)
    action = rng.random()
    if action < 0.4:
        return word[:index] + word[index + 1 :]
    if action < 0.7:
        return word[:index] + word[index + 1] + word[index]
    return word[:index] + word[index] * 2 + word[index + 1 :]


# Catalog style words that carry little signal in a user query.
GENERIC_WORDS = frozenset(
    {
        "прочие",
        "прочий",
        "прочая",
        "прочее",
        "включенные",
        "включенный",
        "включающие",
        "не",
        "в",
        "другие",
        "другой",
        "группировки",
        "такие",
        "как",
        "массой",
        "более",
        "менее",
        "кроме",
        "типа",
        "предназначенные",
        "предназначенный",
        "используемые",
        "используемый",
        "том",
        "числе",
        "на",
        "для",
        "из",
        "с",
        "и",
        "или",
        "которые",
        "который",
        "проч",
        "аналогичные",
        "аналогичный",
        "силовые",
        "установки",
        "устройства",
        "устройство",
        "аппараты",
        "аппарат",
        "машины",
        "машина",
        "оборудование",
        "материалы",
        "изделия",
        "изделие",
    }
)


def salient_words(name: str, idf: dict[str, float] | None) -> list[str]:
    """Content words of a catalog name, ordered by informativeness."""
    word_list = [word for word in tokens(name) if word not in GENERIC_WORDS]
    if not word_list:
        word_list = tokens(name)
    if idf:
        word_list.sort(key=lambda word: idf.get(stem(word), 0.0), reverse=True)
    return word_list


def make_query_variants(
    name: str,
    rng: random.Random,
    idf: dict[str, float] | None = None,
    bridge_sources: list[str] | None = None,
) -> list[str]:
    word_list = tokens(name)
    content_words = salient_words(name, idf)
    variants: list[str] = [name]
    if not word_list:
        return variants

    # Short query: a few salient words (how specialists actually type).
    if len(content_words) >= 2:
        keep = rng.randint(2, max(2, min(4, len(content_words))))
        variants.append(" ".join(rng.sample(content_words, keep)))

    # Brand/model noise around 1-2 salient words ("ноутбук lenovo", "лампа e27").
    head = rng.sample(content_words, min(len(content_words), rng.randint(1, 2)))
    noise = [rng.choice(BRAND_WORDS)]
    if rng.random() < 0.5:
        noise.append(rng.choice(MODEL_WORDS))
    variants.append(" ".join(noise + head))

    # Head word + Russian modifier ("ткань хлопковая", "костюм мужской") —
    # the typical shape of a real procurement query.
    if content_words:
        variants.append(f"{content_words[0]} {rng.choice(NOISE_WORDS)}")
        if rng.random() < 0.5:
            variants.append(f"{rng.choice(NOISE_WORDS)} {content_words[0]}")

    # Shuffled word order with filler words.
    if len(word_list) >= 2:
        shuffled = word_list[:]
        rng.shuffle(shuffled)
        filler = [rng.choice(NOISE_WORDS)]
        if rng.random() < 0.4:
            filler.append(f"{rng.choice(NOISE_WORDS)} {rng.randrange(1, 99)}")
        variants.append(" ".join(filler + shuffled))

    # Typos in the first words.
    if rng.random() < 0.5:
        variants.append(" ".join(_typo(word, rng) for word in word_list[: min(4, len(word_list))]))

    # Lexicon bridges: everyday product words for an abstract category name
    # ("кроссовки" for "Обувь") — these are queries the API must resolve.
    for source in bridge_sources or ():
        if source not in word_list:
            variants.append(source)
            if rng.random() < 0.5:
                variants.append(f"{source} {rng.choice(BRAND_WORDS)}")
    return variants


def bridge_sources_by_name(catalog: list[dict]) -> dict[str, list[str]]:
    """Attach bridge product words only to the canonical (shortest) category row.

    "кроссовки" must be a positive example for "Обувь", not for "Обувь
    ортопедическая детская" — otherwise the same query gets conflicting labels
    and the model learns to distrust it.
    """
    from ml.features import best_match_weight
    from ml.lexicon import BRIDGES

    canonical: dict[str, tuple[int, str]] = {}  # target -> (stem count, name)
    for item in catalog:
        name = item["name"]
        name_stems = stems(name)
        for _source, target in BRIDGES.items():
            target_stem = stems(target)[0]
            if best_match_weight(target_stem, set(name_stems)) <= 0:
                continue
            current = canonical.get(target)
            if current is None or len(name_stems) < current[0]:
                canonical[target] = (len(name_stems), name)

    sources: dict[str, list[str]] = {}
    for source, target in BRIDGES.items():
        if target in canonical:
            sources.setdefault(canonical[target][1], []).append(source)
    return sources


def build_pairs(
    catalog: list[dict],
    rng: random.Random,
    max_queries: int,
    idf: dict[str, float] | None = None,
) -> list[TrainingPair]:
    sampled = rng.sample(catalog, min(max_queries, len(catalog)))
    pairs: list[TrainingPair] = []
    code_by_title = {item["name"]: item.get("pp_code") or "" for item in catalog}
    # Precompute candidate features once: training scores ~10^6 pairs.
    prepared_titles = [(item["name"], prepare_candidate(item["name"])) for item in catalog]
    bridges = bridge_sources_by_name(catalog)

    for item in sampled:
        true_title = item["name"]
        for query in make_query_variants(true_title, rng, idf, bridges.get(true_title)):
            prepared_query = prepare_candidate(expand_query(query))
            # Shortlist exactly the way the API does, then label the rows.
            scored = sorted(
                (
                    (lexical_score_prepared(prepared_query, prepared), title)
                    for title, prepared in prepared_titles
                ),
                key=lambda pair: pair[0],
                reverse=True,
            )
            score_by_title = {title: score for score, title in scored}
            top_titles = [title for score, title in scored[:12] if score > 0]
            if true_title not in top_titles:
                # The positive must always be present, whatever retrieval returns.
                top_titles.append(true_title)
            seen: set[str] = set()
            for title in top_titles:
                if title in seen:
                    continue
                seen.add(title)
                pairs.append(
                    TrainingPair(
                        query=query,
                        title=title,
                        code=code_by_title.get(title, ""),
                        label=int(title == true_title),
                        retrieval_score=score_by_title.get(title, 0.0),
                    )
                )
            for title in rng.sample([t for t, _p in prepared_titles], 3):
                if title not in seen:
                    pairs.append(
                        TrainingPair(
                            query=query,
                            title=title,
                            code=code_by_title.get(title, ""),
                            label=0,
                            retrieval_score=0.0,
                        )
                    )
    return pairs


def fit_logistic_named(
    feature_names: tuple[str, ...],
    feature_rows: list[list[float]],
    labels: list[int],
    *,
    epochs: int = 400,
    learning_rate: float = 0.5,
    l2: float = 1e-3,
) -> tuple[dict[str, float], float]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - training is an explicit task
        raise RuntimeError("Для обучения реранкера нужен numpy (pip install numpy)") from exc

    x = np.asarray(feature_rows, dtype=float)
    y = np.asarray(labels, dtype=float)
    weights = np.zeros(x.shape[1], dtype=float)
    intercept = 0.0
    count = x.shape[0]
    for _epoch in range(epochs):
        scores = x @ weights + intercept
        predictions = 1.0 / (1.0 + np.exp(-scores))
        errors = predictions - y
        gradient = x.T @ errors / count + l2 * weights
        weights -= learning_rate * gradient
        intercept -= learning_rate * float(errors.mean())
    return (
        {name: float(weights[index]) for index, name in enumerate(feature_names)},
        float(intercept),
    )


# Operating point of the service: minimum confidence for showing a
# suggestion, below it the answer is not_found. Chosen by the threshold
# sweep on eval/labeled_examples.csv (eval/experiments.md): below ~0.14 the
# service floods the user with weak hints, above ~0.20 it silences correct
# suggestions. Calibration only picks threshold_ok (precision-targeted).
REVIEW_CONFIDENCE_FLOOR = 0.14


def calibrate_thresholds(
    probabilities: list[float],
    correct: list[bool],
    *,
    target_precision: float = 0.92,
    min_window: int = 8,
) -> tuple[float, float]:
    """Pick (threshold_ok, threshold_review) from held-out top-1 predictions.

    ``threshold_ok`` is the lowest cut whose window keeps precision ≥
    target_precision (a single confident mistake must not spoil the whole
    calibration). ``threshold_review`` separates "worth showing as a suggestion"
    from "retrieval is unreliable", maximizing balanced accuracy.
    """
    ranked = sorted(
        zip(probabilities, correct, strict=True), key=lambda pair: pair[0], reverse=True
    )
    total_positives = sum(1 for _p, is_correct in ranked if is_correct)
    total_negatives = len(ranked) - total_positives
    if total_positives == 0 or total_negatives == 0:
        return 0.75, 0.45

    threshold_ok = 0.75
    for index in range(min_window, len(ranked) + 1):
        window = [is_correct for _p, is_correct in ranked[:index]]
        precision = sum(window) / len(window)
        if precision >= target_precision:
            threshold_ok = ranked[index - 1][0]

    threshold_review = 0.45
    best_f2 = -1.0
    for cut in sorted({round(p, 3) for p, _ in ranked}):
        predicted_positive = sum(1 for p, _c in ranked if p >= cut)
        true_positive = sum(1 for p, c in ranked if p >= cut and c)
        false_positive = predicted_positive - true_positive
        if predicted_positive == 0:
            continue
        recall = true_positive / total_positives
        specificity = (total_negatives - false_positive) / total_negatives
        # Wrong suggestions are expensive for a legal-assistant tool: keep at
        # most 30% of wrong top-1 results visible, then maximize recall (F2).
        if specificity < 0.70:
            continue
        precision = true_positive / predicted_positive
        f2 = 5 * precision * recall / (4 * precision + recall) if (precision + recall) else 0.0
        if f2 > best_f2:
            best_f2 = f2
            threshold_review = cut

    # Keep the ordering sane even on degenerate inputs.
    threshold_ok = max(threshold_ok, threshold_review + 0.05)
    return round(min(threshold_ok, 0.99), 3), round(threshold_review, 3)


def _groups(pairs: list[TrainingPair]) -> dict[str, list[TrainingPair]]:
    by_query: dict[str, list[TrainingPair]] = {}
    for pair in pairs:
        by_query.setdefault(pair.query, []).append(pair)
    return by_query


def _as_candidates(group: list[TrainingPair]) -> list[RankedCandidate]:
    return [
        RankedCandidate(
            code=pair.code or None,
            title=pair.title,
            clause_ref=None,
            retrieval_score=pair.retrieval_score,
            payload={"label": pair.label},
        )
        for pair in group
    ]


def train(
    output: Path,
    *,
    max_queries: int = 400,
    seed: int = SEED,
    validation_share: float = 0.25,
) -> LogisticReranker:
    rng = random.Random(seed)
    catalog = load_catalog()
    # Catalog vocabulary: coverage features ignore brand names and typos in queries.
    known_stems = build_known_stems([item["name"] for item in catalog])
    stem_idf = idf_table([item["name"] for item in catalog])

    pairs = build_pairs(catalog, rng, max_queries=max_queries, idf=stem_idf)
    rng.shuffle(pairs)
    split = int(len(pairs) * (1 - validation_share))
    train_pairs, validation_pairs = pairs[:split], pairs[split:]

    # --- Stage 1: pairwise reranker ---------------------------------------
    feature_rows = [
        [
            extract_features(pair.query, pair.title, pair.code, known_stems, stem_idf)[name]
            for name in FEATURE_NAMES
        ]
        for pair in train_pairs
    ]
    labels = [pair.label for pair in train_pairs]
    weights, intercept = fit_logistic_named(FEATURE_NAMES, feature_rows, labels)

    reranker = LogisticReranker(
        weights=weights,
        intercept=intercept,
        threshold_ok=0.75,
        threshold_review=0.45,
        known_stems=known_stems,
        idf=stem_idf,
    )

    # --- Stage 2: query-level confidence model ----------------------------
    by_query = _groups(validation_pairs)
    group_rows: list[list[float]] = []
    group_labels: list[int] = []
    pair_probabilities: list[float] = []
    pair_correct: list[bool] = []
    for query, group in by_query.items():
        ranked = reranker.rerank(query, _as_candidates(group))
        top_label = int(ranked[0].payload.get("label", 0))
        features = extract_group_features(query, ranked)
        group_rows.append([features[name] for name in GROUP_FEATURE_NAMES])
        group_labels.append(top_label)
        pair_probabilities.append(features["pair_probability"])
        pair_correct.append(top_label == 1)

    group_weights, group_intercept = fit_logistic_named(
        GROUP_FEATURE_NAMES, group_rows, group_labels
    )
    confidence_model = ConfidenceModel(
        weights=group_weights,
        intercept=group_intercept,
        threshold_ok=0.75,
        threshold_review=0.45,
    )

    # Calibrate both decision layers on the held-out queries.
    pair_threshold_ok, pair_threshold_review = calibrate_thresholds(
        pair_probabilities, pair_correct
    )
    confidences = [
        confidence_model.probability(
            {name: row[index] for index, name in enumerate(GROUP_FEATURE_NAMES)}
        )
        for row in group_rows
    ]
    conf_ok, _conf_review = calibrate_thresholds(confidences, [bool(y) for y in group_labels])
    conf_review = REVIEW_CONFIDENCE_FLOOR

    reranker.threshold_ok = pair_threshold_ok
    reranker.threshold_review = pair_threshold_review
    confidence_model.threshold_ok = conf_ok
    confidence_model.threshold_review = conf_review
    reranker.confidence_model = confidence_model

    ok_decisions = [
        is_correct
        for probability, is_correct in zip(
            confidences, [bool(y) for y in group_labels], strict=True
        )
        if probability >= conf_ok
    ]
    reranker.meta = {
        "seed": seed,
        "train_pairs": len(train_pairs),
        "validation_queries": len(by_query),
        "pair_thresholds": {"ok": pair_threshold_ok, "review": pair_threshold_review},
        "confidence_thresholds": {"ok": conf_ok, "review": conf_review},
        "confidence_ok_precision": round(sum(ok_decisions) / len(ok_decisions), 4)
        if ok_decisions
        else None,
        "corpus_rows": len(catalog),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    reranker.save(output)
    print(f"Trained reranker on {len(train_pairs)} pairs ({len(by_query)} validation queries)")
    print(f"  pair thresholds: ok={pair_threshold_ok} review={pair_threshold_review}")
    print(f"  confidence thresholds: ok={conf_ok} review={conf_review}")
    print(f"  confidence ok-precision={reranker.meta['confidence_ok_precision']}")
    print(f"  pair weights={ {name: round(reranker.weights[name], 3) for name in FEATURE_NAMES} }")
    print(
        "  confidence weights="
        f"{ {name: round(confidence_model.weights[name], 3) for name in GROUP_FEATURE_NAMES} }"
    )
    print(f"Saved: {output}")
    return reranker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/reranker.json"))
    parser.add_argument("--max-queries", type=int, default=400)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    train(args.output, max_queries=args.max_queries, seed=args.seed)


if __name__ == "__main__":
    main()
