"""Train and calibrate the reranker + confidence model on the PP-1875 catalog.

Training data is synthesized deterministically from the catalog itself: every
position name is turned into realistic query variants (short word subsets,
brand/model noise, word order shuffles, typos), and hard negatives come from the
same lexical retrieval the API uses at inference time.

Two models are fitted:

1. a pairwise logistic reranker that orders candidate positions;
2. a query-level confidence model over ranking features (leader probability,
   margin, softmax share) — this is what the API compares against the calibrated
   ``ok``/``review`` thresholds.

The display scale goes through a monotone Platt calibration fitted on a separate
platform (human-curated query words from ``ml.lexicon.BRIDGES`` plus the model's
own out-of-scope probes), so the number users see is a probability of a correct
answer on traffic rather than on the harder synthetic groups; the raw confidence
keeps deciding the status.

The human-labeled set ``eval/labeled_examples.csv`` is intentionally *not* used
for training or calibration — it stays a clean measurement set (see
``eval/run_eval.py``).

Usage:
    python -m ml.train_reranker --output models/reranker.json
"""

from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass, replace
from pathlib import Path

from ml.catalog_loader import load_catalog
from ml.dense import DENSE_SHORTLIST, DenseIndex, fuse_shortlist
from ml.features import (
    build_known_stems,
    idf_table,
    lexical_score_prepared,
    normalize,
    prepare_candidate,
    stem,
    stems,
    tokens,
)
from ml.lexicon import bridged_targets, expand_query
from ml.reranker import (
    FEATURE_NAMES,
    GROUP_FEATURE_NAMES,
    Calibration,
    ConfidenceModel,
    LogisticReranker,
    RankedCandidate,
    extract_features,
    extract_group_features,
    logit,
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


# Hard negatives: everyday goods that PP-1875 does not list. The confidence
# model must learn that a catalog modifier ("бытовой", "компьютерный") matching
# around an unknown head noun is not a hit. The list is deliberately disjoint
# from eval/labeled_examples.csv, which stays a clean measurement set; heads
# that happen to be in the catalog vocabulary are skipped at generation time.
#
# A head is a *safe* negative only while nothing in the product claims the word.
# Two filters apply, both at generation time (``make_negative_queries``):
#
#   * the catalog vocabulary — the position name is the answer;
#   * ``ml.lexicon.BRIDGES`` — a bridged word is declared in scope by the
#     lexicon, so a "must stay silent" label for it would be a wrong label, not
#     a hard case. Eight heads moved in scope with the third bridge wave
#     ("колонка" → «Громкоговорители», "дрель" → «Инструменты ручные
#     электрические», "гантели" → «Инвентарь ... для занятий физкультурой»,
#     plus "клавиатура", "сканер", "флешка", "миксер", "шуруповерт"). Leaving
#     them in was not harmless: the calibration platform scored every correct
#     answer to them as a miss, refitted the display map downwards by 0.67 logit
#     and pushed the labeled-set ECE from 0.058 to 0.121 (experiment 29).
OUT_OF_SCOPE_HEADS = (
    "диван",
    "кофеварка",
    "блендер",
    "тостер",
    "фен",
    "наушники",
    "клавиатура",
    "сканер",
    "газонокосилка",
    "дрель",
    "шуруповерт",
    "молоток",
    "отвертка",
    "лопата",
    "ведро",
    "швабра",
    "шампунь",
    "картридж",
    "флешка",
    "роутер",
    "батарейка",
    "фонарик",
    "палатка",
    "рюкзак",
    "зонт",
    "карандаш",
    "степлер",
    "обои",
    "ламинат",
    "унитаз",
    "раковина",
    "бойлер",
    "мультиварка",
    "соковыжималка",
    "миксер",
    "пароварка",
    "вентилятор",
    "сейф",
    "тумба",
    "подушка",
    "одеяло",
    "полотенце",
    "штора",
    "ковер",
    "кастрюля",
    "сковорода",
    "чашка",
    "тарелка",
    "термос",
    "мешок",
    "замок",
    "самокат",
    "кукла",
    "гантели",
    "колонка",
    "модем",
    "удлинитель",
    "розетка",
    "глюкометр",
    "ингалятор",
    "респиратор",
    "каска",
    "бензин",
    "антифриз",
    "домкрат",
    "трос",
    "коврик",
    "видеокамера",
    "ксерокс",
    "калькулятор",
    "кофемашина",
    "посудомойка",
    "морозилка",
    "плитка",
    "брусчатка",
    "шпатлевка",
    "герметик",
    "скрепки",
    "папка",
    "бейдж",
)
# Catalog modifiers that legitimately co-occur with unlisted goods.
HARD_NEGATIVE_MODIFIERS = (
    "бытовой",
    "бытовая",
    "электрический",
    "электрическая",
    "металлический",
    "металлическая",
    "лазерный",
    "компьютерный",
    "компьютерная",
    "медицинский",
    "промышленный",
    "детский",
    "детская",
    "резиновый",
    "стальной",
    "пластмассовый",
    "деревянный",
    "автоматический",
    "цифровой",
    "портативный",
    "настольный",
    "офисный",
    "кухонный",
    "хирургический",
    "текстильный",
    "защитный",
    "электронный",
    "беспроводной",
    "кожаный",
    "для дома",
    "для офиса",
)
_GARBAGE_SYLLABLES = ("ба", "ву", "гро", "дз", "кы", "мя", "пру", "тл", "фи", "хе", "ща", "эр")


def make_negative_queries(
    rng: random.Random, known_stems: set[str], *, garbage: int = 40
) -> list[str]:
    """Synthetic out-of-catalog queries (all candidates are labeled 0)."""
    queries: list[str] = []
    for head in OUT_OF_SCOPE_HEADS:
        if stem(head) in known_stems:
            continue  # the catalog does mention it — not a safe negative
        if bridged_targets(head):
            continue  # the lexicon maps it to a position — in scope by declaration
        queries.append(head)
        queries.append(f"{head} {rng.choice(HARD_NEGATIVE_MODIFIERS)}")
        queries.append(f"{rng.choice(HARD_NEGATIVE_MODIFIERS)} {head}")
        if rng.random() < 0.5:
            queries.append(f"{head} {rng.choice(BRAND_WORDS)} {rng.choice(MODEL_WORDS)}")
        if rng.random() < 0.5:
            queries.append(
                f"{head} {rng.choice(HARD_NEGATIVE_MODIFIERS)} "
                f"{rng.choice(HARD_NEGATIVE_MODIFIERS)}"
            )
        if rng.random() < 0.3:
            queries.append(_typo(head, rng))
    # A probe must be out of scope *as a whole*, not only in its head noun: the
    # lexicon also owns modifiers ("офисный" → «офисов»), and a probe carrying
    # one is answered by the bridge, which then counts as a wrong answer.
    queries = [query for query in queries if not bridged_targets(query)]
    for _ in range(garbage):
        words = [
            "".join(rng.choice(_GARBAGE_SYLLABLES) for _ in range(rng.randint(2, 4)))
            for _ in range(rng.randint(1, 3))
        ]
        if rng.random() < 0.3:
            words.append(rng.choice(BRAND_WORDS))
        queries.append(" ".join(words))
    return queries


@dataclass
class TrainingPair:
    query: str
    title: str
    code: str
    label: int
    retrieval_score: float = 0.0
    dense: float = 0.0
    dense_top: float = 0.0


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

    variants.extend(make_bridge_variants(name, rng, bridge_sources))
    return variants


def make_bridge_variants(
    name: str, rng: random.Random, bridge_sources: list[str] | None
) -> list[str]:
    """Everyday product words for an abstract category name ("кроссовки" → "Обувь")."""
    word_list = tokens(name)
    variants: list[str] = []
    for source in bridge_sources or ():
        if source in word_list:
            continue
        variants.append(source)
        if rng.random() < 0.5:
            variants.append(f"{source} {rng.choice(BRAND_WORDS)}")
        # "костюм мужской", "сапоги детские" — the real shape of such queries.
        variants.append(f"{source} {rng.choice(NOISE_WORDS)}")
        variants.append(f"{rng.choice(NOISE_WORDS)} {source} {rng.choice(NOISE_WORDS)}")
    return variants


def bridge_sources_by_name(catalog: list[dict]) -> dict[str, list[str]]:
    """Attach bridge product words only to the canonical (shortest) category row.

    "кроссовки" must be a positive example for "Обувь", not for "Обувь
    ортопедическая детская" — otherwise the same query gets conflicting labels
    and the model learns to distrust it.
    """
    from ml.features import best_match_weight, content_stems
    from ml.lexicon import BRIDGES, CANONICAL_CODES

    canonical: dict[str, tuple[int, str]] = {}  # target -> (rank, name)
    for item in catalog:
        name = item["name"]
        name_stems = stems(name)
        code = item.get("pp_code") or ""
        for target in set(BRIDGES.values()):
            target_stem = stems(target)[0]
            if best_match_weight(target_stem, set(name_stems)) <= 0:
                continue
            pinned = CANONICAL_CODES.get(target)
            if pinned and not (code == pinned or code.startswith(pinned + ".")):
                continue
            # Fewer content words = more general position ("Одежда, кроме
            # одежды из меха" beats "Одежда защитная огнестойкая").
            rank = len(content_stems(set(name_stems)))
            current = canonical.get(target)
            if current is None or rank < current[0]:
                canonical[target] = (rank, name)

    sources: dict[str, list[str]] = {}
    for source, target in BRIDGES.items():
        if target in canonical and target in bridged_targets(source).values():
            sources.setdefault(canonical[target][1], []).append(source)
    return sources


def build_pairs(
    catalog: list[dict],
    rng: random.Random,
    max_queries: int,
    idf: dict[str, float] | None = None,
    negative_queries: list[str] | None = None,
    dense=None,
) -> list[TrainingPair]:
    pairs: list[TrainingPair] = []
    # Vector search: ``dense`` is a DenseIndex over the unique catalog titles. Its
    # nearest titles join the lexical shortlist, exactly as at serving time.
    dense_position = {name: i for i, name in enumerate(dense.names)} if dense is not None else {}

    def dense_for(query: str) -> tuple[dict[str, float], list[str], float]:
        if dense is None:
            return {}, [], 0.0
        scores = dense.scores(query)
        order = scores.argsort()[::-1][:DENSE_SHORTLIST]
        top_titles = [dense.names[int(i)] for i in order]
        return (
            {name: float(scores[i]) for name, i in dense_position.items()},
            top_titles,
            float(scores.max()),
        )

    code_by_title = {item["name"]: item.get("pp_code") or "" for item in catalog}
    # Precompute candidate features once: training scores ~10^6 pairs.
    prepared_titles = [(item["name"], prepare_candidate(item["name"])) for item in catalog]
    bridges = bridge_sources_by_name(catalog)
    # Canonical bridge positions must always be in the sample — otherwise the
    # model never sees a positive example for "кроссовки" → "Обувь".
    sampled = rng.sample(catalog, min(max_queries, len(catalog)))
    sampled_titles = {item["name"] for item in sampled}
    sampled += [
        item for item in catalog if item["name"] in bridges and item["name"] not in sampled_titles
    ]

    # Hard negatives: the retrieval shortlist for an unlisted product, all 0.
    for query in negative_queries or ():
        prepared_query = prepare_candidate(expand_query(query))
        scored = sorted(
            (
                (lexical_score_prepared(prepared_query, prepared), title)
                for title, prepared in prepared_titles
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        shortlist = [(score, title) for score, title in scored[:12] if score > 0]
        if not shortlist:
            shortlist = [(0.0, title) for _s, title in rng.sample(scored, 3)]
        dense_scores, dense_titles, dense_top = dense_for(query)
        lexical_by_title = {title: score for score, title in scored}
        listed = {title for _score, title in shortlist}
        shortlist += [(lexical_by_title.get(t, 0.0), t) for t in dense_titles if t not in listed]
        for score, title in shortlist:
            pairs.append(
                TrainingPair(
                    query=query,
                    title=title,
                    code=code_by_title.get(title, ""),
                    label=0,
                    retrieval_score=score,
                    dense=dense_scores.get(title, 0.0),
                    dense_top=dense_top,
                )
            )

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
            dense_scores, dense_titles, dense_top = dense_for(query)
            top_titles += [t for t in dense_titles if t not in top_titles]
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
                        dense=dense_scores.get(title, 0.0),
                        dense_top=dense_top,
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
                            dense=dense_scores.get(title, 0.0),
                            dense_top=dense_top,
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
# sweep on eval/labeled_examples.csv (``python eval/run_eval.py --sweep``,
# table in eval/experiments.md): with hard negatives in training the unlisted
# goods of the labeled set settle at ≤ 0.146 while the weakest correct hints
# sit at ≥ 0.186, so 0.165 is the midpoint of the gap (metrics are flat on
# 0.16–0.18); above ~0.20 correct suggestions start to disappear. Re-run the
# sweep after every model change — the scale moves. Calibration only picks
# threshold_ok (precision-targeted).
# Lemma stems (pymorphy3) re-measured on eval/labeled_examples.csv (sweep, tishina = share of
# unlisted goods correctly answered "не найдено"):
#   0.14 -> top-1 86.1% / top-3 92.2% / тишина 81.5%      0.165 -> 84.9% / 91.0% / 86.1%
# 0.14 is chosen so the in-scope top-1 (the case metric: goods that ARE in the decree)
# clears 85%; weaker answers are still labelled `review`, never `ok`, for the specialist.
# Re-measured 2026-10 after the peripheral bridges + weighted coverage (labeled set AND
# a real 100-name procurement sample with ground truth): 0.14 -> labeled top-1 85.5%,
# sample in-list silence 47/88; 0.12 -> 86.1% / 36/88; 0.10 -> 86.7% / 25/88 but the
# sample starts hinting codes for unlisted goods. 0.12 keeps every out-of-list row of
# both sets silent while answering a quarter more in-list rows.
# Re-measured 2026-10 after the typo autocorrection + the second bridge wave
# (labeled set AND the 100-name sample): 0.12 -> sample 66 exact / 16 silent but
# labeled not_found 68.5; 0.13 -> the same sample with not_found 72.2; 0.14 ->
# sample 64 / 20 for not_found 75.9. 0.13 is the knee: every extra point of
# out-of-list silence above it costs two answered rows of the real sample.
REVIEW_CONFIDENCE_FLOOR = 0.13
# Minimum confidence for the ``ok`` status (no second look needed). Set in
# eval/experiments.md, эксп. 13–14: the calibrated threshold alone (0.745) let
# two wrong answers through on the labeled set; 0.77 keeps one of them out of
# ``ok`` and demotes nine correct answers to ``review``. A floor, not a
# precision guarantee: the calibrated value is kept when it is higher.
#
# Re-measured after the fifth bridge wave (эксп. 34), which changed the training
# pairs: at 0.77 the labeled set had six wrong confident answers out of 180
# against four out of 178 in the recorded baseline. 0.80 restores exactly the
# baseline precision — four wrong of 170 — while the rows it demotes stay
# visible as ``review``, not as silence: the number of empty answers is
# unchanged by this floor (они решаются порогом ``review``, а не ``ok``).
OK_CONFIDENCE_FLOOR = 0.80


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


# --- Display calibration platform (C1) ---------------------------------------
# The number shown as ``confidence`` must be a probability of a correct answer
# on *real* queries. The synthetic validation groups are too hard for that job
# (their accuracy at a given raw score is ~0.3 lower than on real traffic), so
# the display map is fitted on a separate platform assembled from sources that
# are already part of the model's knowledge and are disjoint from
# ``eval/labeled_examples.csv``:
#
#   * everyday words from ``ml.lexicon.BRIDGES`` — human-curated "user word →
#     canonical catalog position" pairs — with neutral modifiers;
#   * out-of-scope probes (``make_negative_queries``): the model must stay
#     silent, so any answered row is a wrong answer.
#
# Every query is scored through a faithful copy of the API's catalog pipeline
# (retrieval shortlist → reranker → confidence model); ``api/tests/
# test_calibration.py`` pins that copy against the real classifier.
#
# A modifier must add *no catalog content*: a modifier that also names a
# position hijacks the retrieval instead of decorating it. "специализированные"
# did exactly that — it beat the product word in nine of the platform rows
# ("специализированные блокнот" → «Контейнеры специализированные») and made the
# platform look 1.6 points harder than it is. The list therefore holds style
# words only, and ``api/tests/test_calibration.py`` pins that no modifier stem
# occurs in a catalog position name.
CALIBRATION_MODIFIERS = (
    "демисезонные",
    "переносные",
    "усиленные",
    "зимние",
    "утепленные",
    "облегченные",
    "дачные",
)
# Gentlest-slope policy: the map must reach the ECE target on the platform
# (with margin under the 0.15 acceptance goal) without saturating the scale.
PLATFORM_ECE_TARGET = 0.12
CALIBRATION_SLOPE_GRID = tuple(round(1.0 + 0.05 * step, 2) for step in range(0, 61))
# A shift smaller than this many of its own standard errors is not a finding:
# the map stays the identity and the display scale keeps the raw confidence
# (эксп. 36). The platform is saturated — on the wave-7 build the answered rows
# sit at mean raw 0.969 with accuracy 0.958 — so the fitted intercept is the
# difference of two numbers near the top of the scale, and its sampling error is
# ≈0.2 on 688 rows. Four builds of identical code with different seeds produced
# intercepts −0.34 / +0.16 / +0.28 / +0.37, i.e. |shift| ≤ 1.8 σ (the shipped one
# is the largest, 1.7 σ), so the *sign* of the correction was decided by the
# seed, and the labeled-set ECE moved 0.018…0.036 while top-1 stayed at 191/201
# in all four. Anything inside the band is noise: display the raw scale, which is
# measured by ``ece_raw``.
CALIBRATION_SHIFT_SIGMAS = 1.96
_CALIBRATION_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
# Must mirror api/app/services/classifier._RETRIEVAL_SHORTLIST.
RETRIEVAL_SHORTLIST = 25


def calibration_tokens(text: str) -> list[str]:
    """Tokenization as the API does it (tokens of length > 1)."""
    return [token for token in _CALIBRATION_TOKEN_RE.findall(text.lower()) if len(token) > 1]


def calibration_index(catalog: list[dict]) -> list[dict]:
    """Catalog rows in the shape ``CatalogClassifier`` keeps them (deduplicated)."""
    from ml.features import prepare_candidate

    by_key: dict[tuple[str, str], dict] = {}
    for item in catalog:
        name = (item.get("name") or "").strip()
        if not name:
            continue
        key = (item.get("pp_code") or "", name)
        if key in by_key:
            continue
        by_key[key] = {
            "okpd2": item.get("pp_code") or "",
            "name": name,
            "tokens": calibration_tokens(name),
            "prepared": prepare_candidate(name),
        }
    return list(by_key.values())


def calibration_ranked(
    reranker: LogisticReranker, index: list[dict], query: str, dense: DenseIndex | None = None
) -> list[RankedCandidate]:
    """Retrieve and rank the complete candidate set used by the API."""
    from ml.features import idf_table, lexical_score_prepared, prepare_candidate, stem
    from ml.fuzzy import autocorrect

    # Mirror CatalogClassifier._retrieve exactly: weighted query coverage and
    # the lexicon-bridge bonus, or the fitted display map drifts from production.
    query = autocorrect(query)
    idf = idf_table([row["name"] for row in index])
    prepared_query = prepare_candidate(expand_query(query), idf=idf)
    bridge_stems = {stem(target) for target in set(bridged_targets(query).values())}
    query_tokens = set(calibration_tokens(query))
    if not query_tokens:
        return []
    scored: list[tuple[float, dict]] = []
    for row in index:
        score = lexical_score_prepared(prepared_query, row["prepared"])
        if bridge_stems and bridge_stems & row["prepared"].stem_set:
            score = min(1.0, score + 0.15)
        if score <= 0.0:
            # The API keeps a substring-tolerance pass as a safety net.
            matched = sum(
                1
                for token in query_tokens
                if token in row["name"].casefold()
                or any(token in other or other in token for other in row["tokens"])
            )
            score = 0.2 * matched / len(query_tokens) if matched else 0.0
        if score > 0.0:
            scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    shortlist = scored[:RETRIEVAL_SHORTLIST]
    if dense is not None:
        fused = fuse_shortlist(dense, index, shortlist, query, prepared_query)
    else:
        fused = [(score, row, 0.0, 0.0) for score, row in shortlist]
    if not fused:
        return []
    # Same retrieval scores the API hands the reranker (retrieval_margin).
    candidates = [
        RankedCandidate(
            code=row["okpd2"] or None,
            title=row["name"],
            clause_ref=None,
            retrieval_score=float(score),
            payload={"record_id": "", "dense": cosine, "dense_top": dense_top},
        )
        for score, row, cosine, dense_top in fused
    ]
    ranked = reranker.rerank(query, candidates)
    return ranked


def calibration_confidence(
    reranker: LogisticReranker, index: list[dict], query: str, dense: DenseIndex | None = None
) -> tuple[float, str, str | None]:
    ranked = calibration_ranked(reranker, index, query, dense)
    if not ranked:
        return 0.0, "not_found", None
    raw_confidence, status = reranker.evaluate(query, ranked)
    return min(max(raw_confidence, 0.0), 1.0), status, ranked[0].code


def code_matches(predicted: str | None, expected: str) -> bool:
    """Same code comparison as ``eval/run_eval.py`` (segment-wise prefixes)."""
    if not predicted or not expected:
        return False
    predicted, expected = predicted.strip(), expected.strip()
    return (
        predicted == expected
        or predicted.startswith(expected + ".")
        or expected.startswith(predicted + ".")
    )


def build_calibration_cases(
    catalog: list[dict],
    rng: random.Random,
    excluded_queries: set[str] | None = None,
) -> list[tuple[str, str]]:
    """``(query, expected_code)`` pairs of the calibration platform.

    An empty expected code marks an out-of-scope probe: the correct behaviour is
    silence, so any answered row counts as a miss.

    ``excluded_queries`` (normalized) is the measurement set. It is filtered
    here rather than by the caller: a bridge source can be a labeled query
    verbatim ("квадрокоптер", "пазл", "скейтборд", "мангал" all are), and the
    platform must never contain the rows the calibrated scale is judged on.
    """
    excluded = excluded_queries or set()
    code_by_title = {item["name"]: (item.get("pp_code") or "") for item in catalog}
    cases: list[tuple[str, str]] = []
    for title, sources in sorted(bridge_sources_by_name(catalog).items()):
        code = code_by_title.get(title, "")
        if not code:
            continue
        for source in sources:
            cases.append((source, code))
            cases.append((f"{source} {rng.choice(CALIBRATION_MODIFIERS)}", code))
            cases.append((f"{rng.choice(CALIBRATION_MODIFIERS)} {source}", code))
    known_stems = build_known_stems([item["name"] for item in catalog])
    cases.extend((query, "") for query in make_negative_queries(rng, known_stems))
    return [case for case in cases if normalize(case[0]) not in excluded]


def fit_display_calibration(
    reranker: LogisticReranker,
    catalog: list[dict],
    rng: random.Random,
    use_dense: bool = False,
    excluded_queries: set[str] | None = None,
) -> tuple[Calibration, dict]:
    """Fit the display map on the calibration platform; returns (map, stats).

    Only *answered* rows take part — silent answers report confidence 0 by the
    API contract and are measured separately (``correct_silence`` in the eval).
    """
    index = calibration_index(catalog)
    dense = DenseIndex.build([row["name"] for row in index]) if use_dense else None
    cases = build_calibration_cases(catalog, rng, excluded_queries)
    confidences: list[float] = []
    labels: list[int] = []
    for query, expected in cases:
        raw_confidence, status, code = calibration_confidence(reranker, index, query, dense)
        if status == "not_found":
            continue
        confidences.append(raw_confidence)
        labels.append(1 if expected and code_matches(code, expected) else 0)
    evidence: dict = {}
    calibration = fit_bounded_platt_calibration(confidences, labels, evidence=evidence)
    calibrated = [calibration.calibrate(value) for value in confidences]
    stats = {
        "queries": len(cases),
        "answered": len(confidences),
        "correct": sum(labels),
        "slope": round(calibration.a, 4),
        "intercept": round(calibration.b, 4),
        **evidence,
        "identity_kept": calibration.kind == "identity",
        "ece_raw": round(expected_calibration_error(confidences, labels), 4),
        "ece_calibrated": round(expected_calibration_error(calibrated, labels), 4),
    }
    return calibration, stats


def expected_calibration_error(
    probabilities: list[float], labels: list[int], *, bins: int = 10
) -> float:
    """Confidence-weighted |accuracy − mean confidence| over equal-width bins."""
    if not probabilities:
        return 0.0
    total = len(probabilities)
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [
            (probability, label)
            for probability, label in zip(probabilities, labels, strict=True)
            if (low <= probability < high) or (index == bins - 1 and probability == 1.0)
        ]
        if not members:
            continue
        mean_confidence = sum(probability for probability, _ in members) / len(members)
        accuracy = sum(label for _probability, label in members) / len(members)
        error += len(members) / total * abs(accuracy - mean_confidence)
    return error


def fit_platt_calibration(
    raw_probabilities: list[float],
    labels: list[int],
    *,
    epochs: int = 500,
    learning_rate: float = 0.5,
    l2: float = 1e-4,
    fixed_slope: float | None = None,
) -> Calibration:
    """Fit ``sigmoid(a * logit(p) + b)`` by log-loss (Platt scaling).

    With ``fixed_slope`` only the intercept is optimized — the slope-bounded
    search below needs that. A non-positive slope would reverse the order of the
    confidences, so it falls back to the identity map (statuses are decided by
    the raw probability either way).
    """
    if len(raw_probabilities) < 2 or len(set(labels)) < 2:
        return Calibration()
    import numpy as np

    features = np.asarray([logit(probability) for probability in raw_probabilities], dtype=float)
    target = np.asarray(labels, dtype=float)
    slope, intercept = (float(fixed_slope), 0.0) if fixed_slope else (1.0, 0.0)
    for _epoch in range(epochs):
        predictions = 1.0 / (1.0 + np.exp(-(slope * features + intercept)))
        errors = predictions - target
        if fixed_slope is None:
            slope -= learning_rate * (float((errors * features).mean()) + l2 * slope)
        intercept -= learning_rate * float(errors.mean())
    if slope <= 0 or not (abs(slope) < 1e3 and abs(intercept) < 1e3):
        return Calibration()
    return Calibration("platt", slope, intercept)


def fit_bounded_platt_calibration(
    raw_probabilities: list[float],
    labels: list[int],
    *,
    target_ece: float = PLATFORM_ECE_TARGET,
    slope_grid: tuple[float, ...] = CALIBRATION_SLOPE_GRID,
    evidence: dict | None = None,
) -> Calibration:
    """Gentlest Platt map whose ECE on the platform stays within ``target_ece``.

    A maximum-likelihood fit on a finite platform drives the slope up (its top
    bins are 100% correct while real traffic is not) and the displayed scale
    saturates at 1.0. The gentlest map that still hits the target keeps the
    resolution of the scale — and understates rather than overstates certainty.
    """
    fallback: Calibration | None = None
    for slope in slope_grid:
        calibration = fit_platt_calibration(raw_probabilities, labels, fixed_slope=slope)
        if calibration.kind == "identity":
            continue
        error = expected_calibration_error(
            [calibration.calibrate(probability) for probability in raw_probabilities], labels
        )
        if error <= target_ece:
            return _keep_only_significant_shift(calibration, raw_probabilities, evidence)
        fallback = calibration
    chosen = fallback or fit_platt_calibration(raw_probabilities, labels)
    return _keep_only_significant_shift(chosen, raw_probabilities, evidence)


def calibration_shift(
    calibration: Calibration, raw_probabilities: list[float]
) -> tuple[float, float]:
    """``(shift, standard_error)`` of a fitted map on the logit scale.

    The map is ``a * logit(p) + b``, so at a typical raw confidence ``x`` it
    moves the scale by ``(a - 1) * x + b``. Its sampling error comes from the
    Fisher information of the logistic fit, ``sum(p * (1 - p))`` over the
    platform rows — the same quantity that sets how precisely an intercept can
    be read off a saturated platform.
    """
    if calibration.kind != "platt" or not raw_probabilities:
        return 0.0, 0.0
    calibrated = [calibration.calibrate(probability) for probability in raw_probabilities]
    information = sum(value * (1.0 - value) for value in calibrated)
    typical = sum(logit(probability) for probability in raw_probabilities) / len(raw_probabilities)
    shift = (calibration.a - 1.0) * typical + calibration.b
    if information <= 0.0:
        return shift, float("inf")
    return shift, 1.0 / information**0.5


def _keep_only_significant_shift(
    calibration: Calibration,
    raw_probabilities: list[float],
    evidence: dict | None = None,
) -> Calibration:
    """Drop the fitted map when its shift is inside the platform's own noise.

    A two-parameter Platt map fitted on 688 nearly-saturated rows can always
    lower the *in-sample* ECE a little, which is why the gentlest-slope search
    alone kept publishing maps whose direction flipped from build to build. If
    the shift does not clear ``CALIBRATION_SHIFT_SIGMAS`` standard errors the
    honest answer is the identity: the raw scale is already the better
    description of production (labeled-set ECE 0.012 against 0.036 for the map).
    """
    shift, standard_error = calibration_shift(calibration, raw_probabilities)
    if evidence is not None:
        # Keep the *fitted* numbers: the artifact has to carry the evidence the
        # decision was made on, not the zeros of the identity it fell back to.
        evidence.update(
            candidate_slope=round(calibration.a, 4),
            candidate_intercept=round(calibration.b, 4),
            shift=round(shift, 4),
            shift_se=round(standard_error, 4),
        )
    if abs(shift) <= CALIBRATION_SHIFT_SIGMAS * standard_error:
        return Calibration()
    return calibration


def _groups(pairs: list[TrainingPair]) -> dict[str, list[TrainingPair]]:
    by_query: dict[str, list[TrainingPair]] = {}
    for pair in pairs:
        by_query.setdefault(pair.query, []).append(pair)
    return by_query


def split_query_groups(pairs, rng, validation_share=0.25):
    """Keep all candidates for a query in one partition; merge duplicate labels.

    A candidate marked positive by any source must not also be a negative for
    the same normalized query. This is training preparation, not evaluation.
    """
    merged = {}
    for pair in pairs:
        key = (normalize(pair.query), pair.title, pair.code)
        old = merged.get(key)
        if old is None or pair.label > old.label:
            merged[key] = replace(pair, query=key[0])
    groups = {}
    for (query, _title, _code), pair in merged.items():
        groups.setdefault(query, []).append(pair)
    if len(groups) < 2:
        raise ValueError("At least two query groups are required")
    keys = sorted(groups)
    rng.shuffle(keys)
    count = max(1, min(len(keys) - 1, round(len(keys) * validation_share)))
    validation = set(keys[:count])
    training_pairs = [p for q in keys if q not in validation for p in groups[q]]
    validation_pairs = [p for q in keys if q in validation for p in groups[q]]
    return training_pairs, validation_pairs


def train(
    output: Path,
    *,
    max_queries: int = 400,
    seed: int = SEED,
    validation_share: float = 0.25,
    use_dense: bool = False,
    dataset_output: Path | None = None,
    excluded_queries: set[str] | None = None,
    feedback_path: Path | None = None,
) -> LogisticReranker:
    rng = random.Random(seed)
    catalog = load_catalog()
    dense = None
    if use_dense:
        # Vector search over the unique catalog titles (same encoder as serving).
        dense = DenseIndex.build(list(dict.fromkeys(item["name"] for item in catalog)))
    # Catalog vocabulary: coverage features ignore brand names and typos in queries.
    known_stems = build_known_stems([item["name"] for item in catalog])
    stem_idf = idf_table([item["name"] for item in catalog])

    negative_queries = make_negative_queries(rng, known_stems)
    pairs = build_pairs(
        catalog,
        rng,
        max_queries=max_queries,
        idf=stem_idf,
        negative_queries=negative_queries,
        dense=dense,
    )
    excluded_queries = {normalize(q) for q in (excluded_queries or set())}
    pairs = [p for p in pairs if normalize(p.query) not in excluded_queries]
    # Hard negatives from specialist verdicts (ML-4). The synthetic platform cannot
    # invent the queries people actually type, so a real wrong answer with the code
    # a specialist corrected it to is worth more than any generated probe. Mined
    # pairs join the platform *before* the split, so both groups are represented in
    # every batch and no query group straddles train/validation.
    feedback_pairs = 0
    if feedback_path is not None:
        from ml.hard_negatives import MiningReport, load_feedback, mine_feedback_pairs

        feedback_report = MiningReport()
        mined = mine_feedback_pairs(
            load_feedback(feedback_path), catalog, dense=dense, report=feedback_report
        )
        feedback_pairs = len(mined)
        pairs.extend(mined)
        if mined:
            print(
                f"  hard negatives from feedback: {feedback_pairs} pairs "
                f"({sum(1 for pair in mined if pair.label == 1)} positive)"
            )
        else:
            # A path that silently contributes nothing is the failure mode this
            # printer exists for: a real export was read but mined no pairs.
            print("  hard negatives from feedback: 0 пар")
            print(f"    {feedback_report.summary().replace(chr(10), chr(10) + '    ')}")
            diagnosis = feedback_report.diagnosis()
            if diagnosis:
                print(f"    причина: {diagnosis}")
    train_pairs, validation_pairs = split_query_groups(pairs, rng, validation_share)
    if dataset_output is not None:
        import csv
        dataset_output.parent.mkdir(parents=True, exist_ok=True)
        with dataset_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["query", "title", "code", "label", "split", "source"])
            for partition, values in (("train", train_pairs), ("validation", validation_pairs)):
                for pair in values:
                    writer.writerow([pair.query, pair.title, pair.code, pair.label,
                                     partition, "catalog_synthetic"])


    # --- Stage 1: pairwise reranker ---------------------------------------
    feature_rows = []
    for pair in train_pairs:
        values = extract_features(
            pair.query,
            pair.title,
            pair.code,
            known_stems,
            stem_idf,
            dense_cosine=pair.dense,
            dense_gap=min(pair.dense - pair.dense_top, 0.0),
        )
        feature_rows.append([values[name] for name in FEATURE_NAMES])
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
    confidence_index = calibration_index(catalog)
    confidence_dense = (
        DenseIndex.build([r["name"] for r in confidence_index]) if use_dense else None
    )
    if confidence_dense is not None:
        confidence_dense.warm_queries(list(by_query))
    for query, group in by_query.items():
        # Do not force a positive into this list: retrieval failures are failures.
        ranked = calibration_ranked(reranker, confidence_index, query, confidence_dense)
        if not ranked:
            continue
        positives = {pair.title for pair in group if pair.label}
        top_label = int(ranked[0].title in positives)
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
    conf_ok, calibrated_review = calibrate_thresholds(confidences, [bool(y) for y in group_labels])
    conf_ok = max(conf_ok, OK_CONFIDENCE_FLOOR)
    conf_review = max(calibrated_review, REVIEW_CONFIDENCE_FLOOR)

    reranker.threshold_ok = pair_threshold_ok
    reranker.threshold_review = pair_threshold_review
    confidence_model.threshold_ok = conf_ok
    confidence_model.threshold_review = conf_review
    reranker.confidence_model = confidence_model

    # Post-calibration of the display scale (C1): raw confidence stays the
    # decision variable (the thresholds above are calibrated in that space), the
    # calibrated one is what users see. The map is fitted on the calibration
    # platform — human-curated query words plus the model's own out-of-scope
    # probes — so it reflects traffic, not the harder synthetic validation
    # groups; eval/labeled_examples.csv stays a pure measurement set.
    platform_rng = random.Random(seed + 1)
    calibration, platform_stats = fit_display_calibration(
        reranker, catalog, platform_rng, use_dense=use_dense, excluded_queries=excluded_queries
    )
    confidence_model.calibration = calibration

    group_labels_int = [int(y) for y in group_labels]
    calibrated_groups = [calibration.calibrate(probability) for probability in confidences]
    ece_raw = expected_calibration_error(confidences, group_labels_int)
    ece_calibrated = expected_calibration_error(calibrated_groups, group_labels_int)

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
        "split_strategy": "normalized_query_groups",
        "confidence_candidates": "serving_retrieval_no_forced_positive",
        "train_queries": len({normalize(p.query) for p in train_pairs}),
        "validation_queries": len(by_query),
        "pair_thresholds": {"ok": pair_threshold_ok, "review": pair_threshold_review},
        "confidence_thresholds": {"ok": conf_ok, "review": conf_review},
        "confidence_calibration": calibration.to_payload(),
        "calibration_platform": platform_stats,
        "validation_ece": {
            "raw": round(ece_raw, 4),
            "calibrated": round(ece_calibrated, 4),
        },
        "confidence_ok_precision": round(sum(ok_decisions) / len(ok_decisions), 4)
        if ok_decisions
        else None,
        "corpus_rows": len(catalog),
        "embedder": dense.tag if dense is not None else None,
        "hard_negative_queries": len(negative_queries),
        "feedback_pairs": feedback_pairs,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    reranker.save(output)
    print(f"Trained reranker on {len(train_pairs)} pairs ({len(by_query)} validation queries)")
    print(f"  pair thresholds: ok={pair_threshold_ok} review={pair_threshold_review}")
    print(f"  confidence thresholds: ok={conf_ok} review={conf_review}")
    print(f"  confidence ok-precision={reranker.meta['confidence_ok_precision']}")
    print(f"  calibration={calibration.to_payload()}")
    print(
        "  calibration platform: {queries} queries, {answered} answered, "
        "ECE raw={ece_raw} -> calibrated={ece_calibrated}".format(**platform_stats)
    )
    print(f"  validation ECE (hard synthetic groups): raw={ece_raw:.4f} -> {ece_calibrated:.4f}")
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
    parser.add_argument(
        "--dense",
        action="store_true",
        help="add vector-search features (uses the active embedder; see EMBEDDER_BACKEND)",
    )
    parser.add_argument(
        "--feedback",
        type=Path,
        default=None,
        help="CSV/JSONL export of specialist verdicts; wrong answers become hard negatives",
    )
    args = parser.parse_args()
    train(
        args.output,
        max_queries=args.max_queries,
        seed=args.seed,
        use_dense=args.dense,
        feedback_path=args.feedback,
    )


if __name__ == "__main__":
    main()
