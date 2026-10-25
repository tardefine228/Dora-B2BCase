"""Dependency-free text features shared by the catalog ranker and the ML reranker.

The goal is robust matching of short Russian product names against the official
PP-1875 position names: light stemming tolerates inflections, character n-grams
tolerate typos and word order changes, and digit tokens keep measurement details
(``10 м``, ``220 В``) significant. Everything here is pure Python so the catalog
backend keeps working without numpy or ML dependencies installed.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from math import log, sqrt

TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
DIGIT_RE = re.compile(r"\d+(?:[.,]\d+)?")

# Common Russian endings; stripped only when the remaining stem is long enough.
_RU_SUFFIXES = (
    "иями",
    "ями",
    "ами",
    "ого",
    "его",
    "ому",
    "ему",
    "ыми",
    "ими",
    "ешь",
    "ишь",
    "ует",
    "ают",
    "яют",
    "ения",
    "ение",
    "ость",
    "ости",
    "ать",
    "ять",
    "лась",
    "лось",
    "лись",
    "тся",
    "ться",
    "ший",
    "щий",
    "ого",
    "ние",
    "ния",
    "ый",
    "ий",
    "ой",
    "ая",
    "яя",
    "ое",
    "ее",
    "ые",
    "ие",
    "ам",
    "ям",
    "ах",
    "ях",
    "ом",
    "ем",
    "ую",
    "юю",
    "ов",
    "ев",
    "ью",
    "ия",
    "ья",
    "ье",
    "ии",
    "ы",
    "и",
    "а",
    "я",
    "у",
    "ю",
    "е",
    "о",
    "ь",
)

_EN_SUFFIXES = ("ing", "ies", "es", "ed", "er", "ly", "s")


def normalize(text: str) -> str:
    """Casefold and fold ``ё``/``е`` so spelling variants match."""
    return text.casefold().replace("ё", "е")


def tokens(text: str) -> list[str]:
    """Alphanumeric tokens of length > 1 (single letters add noise)."""
    return [token for token in TOKEN_RE.findall(normalize(text)) if len(token) > 1]


_CYRILLIC_RE = re.compile(r"[а-яё]+")
_morph = None
_morph_checked = False


def _analyzer():
    """pymorphy3 analyzer, or ``None`` when it is not installed (suffix stripping only)."""
    global _morph, _morph_checked
    if not _morph_checked:
        _morph_checked = True
        if os.getenv("STEMMER", "lemma").strip().lower() == "lemma":
            try:
                import pymorphy3

                _morph = pymorphy3.MorphAnalyzer()
            except Exception:  # noqa: BLE001 - optional dependency
                _morph = None
    return _morph


def lemmatizer_active() -> bool:
    """True when stems are built from real lemmas (trained artifacts record this)."""
    return _analyzer() is not None


@lru_cache(maxsize=200_000)
def _lemma(word: str) -> str:
    analyzer = _analyzer()
    if analyzer is None:
        return word
    parses = analyzer.parse(word)
    return normalize(parses[0].normal_form) if parses else word


def _strip_suffix(word: str) -> str:
    for suffix in _RU_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    for suffix in _EN_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


@lru_cache(maxsize=200_000)
def stem(token: str) -> str:
    """Lemma-based Russian stem (pymorphy3) with the light suffix stripper on top.

    Inflected forms collapse to one lemma first (\"станки\" and \"станок\", \"гобои\"
    and \"гобой\"), which a pure suffix stripper cannot do; the stripper then
    trims derivational endings so \"токарный\"/\"токарные\" share a stem. Without
    pymorphy3 only the suffix stripper runs.
    """
    word = normalize(token)
    if word.isdigit() or len(word) <= 3:
        return word
    if _CYRILLIC_RE.fullmatch(word) and _analyzer() is not None:
        word = _lemma(word)
    return _strip_suffix(word)


def stems(text: str) -> list[str]:
    return [stem(token) for token in tokens(text)]


def stem_set(text: str) -> set[str]:
    return set(stems(text))


def digits(text: str) -> set[str]:
    """Numeric fragments such as ``220`` or ``2.5`` for measurement matching."""
    return {match.replace(",", ".") for match in DIGIT_RE.findall(normalize(text))}


def char_ngrams(text: str, sizes: tuple[int, ...] = (3, 4)) -> Counter[str]:
    """Character n-grams over the whitespace-normalized text, padded by word."""
    counts: Counter[str] = Counter()
    for word in normalize(text).split():
        padded = f"^{word}$"
        for size in sizes:
            for offset in range(len(padded) - size + 1):
                counts[padded[offset : offset + size]] += 1
    return counts


def cosine_sparse(left: Counter[str], right: Counter[str]) -> float:
    """Cosine similarity between two sparse count vectors."""
    if not left or not right:
        return 0.0
    small, large = (left, right) if len(left) <= len(right) else (right, left)
    dot = sum(count * large.get(key, 0) for key, count in small.items())
    if dot == 0.0:
        return 0.0
    norm_left = sqrt(sum(count * count for count in left.values()))
    norm_right = sqrt(sum(count * count for count in right.values()))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


def stems_match(query_stem: str, candidate_stem: str) -> bool:
    return match_weight(query_stem, candidate_stem) > 0.0


# Compound nouns like "электрочайник" (электро+чайник) embed the head word as a
# suffix; plain substring matching would also fire on "монитор" inside
# "мониторинг", so only recognized compound prefixes count.
COMPOUND_PREFIXES = frozenset(
    {
        "электро",
        "электронно",
        "гидро",
        "авто",
        "авиа",
        "мото",
        "фото",
        "видео",
        "кино",
        "радио",
        "теле",
        "тепло",
        "холодо",
        "газо",
        "водо",
        "ветро",
        "лако",
        "лазеро",
        "плазмо",
        "пневмо",
        "микро",
        "макро",
        "мега",
        "мини",
        "поли",
        "металло",
        "дерево",
        "нефте",
        "угле",
        "сельхоз",
        "станко",
        "прибо",
        "судо",
        "вертолето",
        "паро",
        "тракторо",
        "комбайно",
    }
)


# Derivational suffixes between a noun stem and its relational adjective:
# "металл" ~ "металлическ", "двер" ~ "дверн", "кож" ~ "кожан". Only applied to
# stems of 4+ characters so that "газ"/"газов" style collisions stay out.
# Note: no bare "ат" — it glued unrelated stems ("печать" ~ "печи"),
# dragging therapeutic lasers and microwave ovens into printer shortlists.
DERIVATIONAL_SUFFIXES = frozenset(
    {"ическ", "еск", "ск", "ов", "ев", "ан", "ян", "н", "ин", "енн", "онн", "ист"}
)
DERIVATIONAL_WEIGHT = 0.8
MIN_DERIVATIONAL_STEM = 4


def match_weight(query_stem: str, candidate_stem: str) -> float:
    """1.0 for the same stem, 0.8 for noun/adjective derivation, 0.7 for a compound hit."""
    if query_stem == candidate_stem:
        return 1.0
    longer, shorter = (
        (candidate_stem, query_stem)
        if len(candidate_stem) >= len(query_stem)
        else (query_stem, candidate_stem)
    )
    if not shorter or shorter not in longer:
        return 0.0
    if (
        longer.startswith(shorter)
        and len(shorter) >= MIN_DERIVATIONAL_STEM
        and longer[len(shorter) :] in DERIVATIONAL_SUFFIXES
    ):
        return DERIVATIONAL_WEIGHT
    remainder = longer.replace(shorter, "", 1)
    if remainder in COMPOUND_PREFIXES:
        return 0.7
    return 0.0


def best_match_weight(query_stem: str, candidate_stems: set[str]) -> float:
    if not candidate_stems:
        return 0.0
    if query_stem in candidate_stems:
        return 1.0
    return max(match_weight(query_stem, candidate_stem) for candidate_stem in candidate_stems)


# Function words and catalog boilerplate that must not count as "unmatched
# candidate content" — "Одежда, кроме одежды из меха" is a 2-word position.
STOPWORDS = frozenset(
    {
        "из",
        "или",
        "для",
        "не",
        "на",
        "по",
        "от",
        "до",
        "при",
        "без",
        "под",
        "над",
        "их",
        "его",
        "ее",
        "том",
        "числе",
        "кроме",
        "включая",
        "такие",
        "как",
        "так",
        "же",
        "прочие",
        "проч",
        "другие",
        "друг",
        "группировки",
        "группировк",
        "включенные",
        "включенн",
        "аналогичные",
        "аналогичн",
        "массой",
        "масс",
        "более",
        "бол",
        "менее",
        "мен",
        "только",
        "тольк",
        "виде",
        "вид",
        "части",
        "част",
    }
)


STOPWORD_STEMS = frozenset(STOPWORDS | {stem(word) for word in STOPWORDS})


def content_stems(stem_values: set[str]) -> set[str]:
    """Drop stopwords/boilerplate; fall back to the full set if nothing is left."""
    kept = {value for value in stem_values if value not in STOPWORD_STEMS}
    return kept or set(stem_values)


# Query modifiers that describe a variant of a product, never the product
# itself: gender/season/material adjectives and measurement words. When such a
# word is missing from the catalog vocabulary it must not be treated as "an
# unknown product word" (see OOV_CYRILLIC_WEIGHT).
LOW_SIGNAL_WORDS = frozenset(
    {
        "мужской",
        "женский",
        "зимний",
        "летний",
        "демисезонный",
        "рабочий",
        "повседневный",
        "спортивный",
        "хлопковый",
        "кожаный",
        "шерстяной",
        "синтетический",
        "тканевый",
        "утепленный",
        "новый",
        "большой",
        "маленький",
        "белый",
        "черный",
        "цветной",
        "домашний",
        "школьный",
        "универсальный",
        "классический",
        "современный",
        "стандартный",
        "обычный",
        "качественный",
        "недорогой",
        "мобильный",
        "кнопочный",
        "сенсорный",
        "сотовый",
        "настенный",
        "напольный",
        "настольный",
        "потолочный",
        "ручной",
        "переносной",
        "диагональ",
        "дюйм",
        # Spec/packaging words that never name a position of the decree list
        # (zero occurrences in data/data.csv): counting them as "unknown product
        # words" diluted query coverage and silenced in-list products
        # ("Электрочайник … корпус из нержавеющей стали; 1.7 л").
        "экран",
        "память",
        "разрешение",
        "поколение",
        "корпус",
        "литр",
        "см",
        "мм",
        "метр",
        "кг",
        "грамм",
        "ватт",
        "вольт",
        "шт",
        "штука",
        "упаковка",
        "размер",
        "цвет",
        "вес",
        "объем",
        "мощность",
        "модель",
        "артикул",
    }
)
LOW_SIGNAL_STEMS = frozenset(stem(word) for word in LOW_SIGNAL_WORDS)
LOW_SIGNAL_WEIGHT = 0.5


def coverage(query: str, candidate: str) -> tuple[float, float]:
    """Return (query stem coverage in candidate, candidate content-stem precision)."""
    query_stems = stem_set(query)
    candidate_stems = stem_set(candidate)
    if not query_stems or not candidate_stems:
        return 0.0, 0.0
    query_content = content_stems(query_stems)
    query_hits = sum(best_match_weight(stem, candidate_stems) for stem in query_content)
    return query_hits / len(query_content), candidate_precision(query_stems, candidate)


# --- Long official names (ML-7) -----------------------------------------
# The decree's positions are legal texts: "Компьютеры портативные
# массой не более 10 кг, такие как ноутбуки, планшетные компьютеры,
# карманные компьютеры, ..." runs to 200–1100 characters. Scoring such
# a name as one field dilutes the precision features: the query
# matches the head ("Компьютеры портативные") but the storm of
# qualifiers ("в том числе совмещающие функции…") dominates the
# denominator, so correct long-name candidates look imprecise.
# BM25F-style fix: split a long name into a *head* (everything before
# the first field separator) and *qualifiers* (the rest), and weigh
# a qualifier hit a fraction of a head hit.
LONG_TITLE_CHARS = 200
QUALIFIER_WEIGHT = 0.3
# Off by default. The split changes how long official names are scored, and the
# labeled set has only six rows whose expected position is that long, so the gate
# cannot see the change at all: on the long-name slice (``eval/long_names.py``) it
# buys +2.2 p.p. top-1 while turning 13 ``review`` hints into confident *wrong*
# answers — a product trade-off, not a free win. Measure first, then switch on
# deliberately: ``LONG_TITLE_SPLIT=1``.
LONG_TITLE_SPLIT_ENABLED = os.environ.get("LONG_TITLE_SPLIT", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
# A head ends at the first list separator. Parentheses open an
# enumerative clause ("…10 кг, такие как ноутбуки, …"), so they end
# the head too; a colon separates a definition from its list.
_TITLE_HEAD_SPLIT_RE = re.compile(r"[,;:(]")


def split_title(name: str) -> tuple[str, str]:
    """``(head, qualifiers)`` of a position name.

    Names up to ``LONG_TITLE_CHARS`` stay one field (qualifiers are
    empty), so short positions keep their exact previous scoring.
    """
    if not LONG_TITLE_SPLIT_ENABLED or len(name) <= LONG_TITLE_CHARS:
        return name, ""
    cut = -1
    for match in _TITLE_HEAD_SPLIT_RE.finditer(name):
        if match.start() >= 20:  # the head keeps at least a few words
            cut = match.start()
            break
    if cut < 0:
        # No separator: cut at a word boundary near the middle.
        cut = len(name) // 2
        while cut > 20 and name[cut - 1] not in " \t":
            cut -= 1
    head = name[:cut].strip(" ,;:(–—")
    qualifiers = name[cut:].strip(" ,;:(–—")
    return head, qualifiers


def _field_precision(
    query_stems: set[str], fields: list[tuple[str, float]]
) -> float:
    """Share of candidate content stems explained by the query.

    ``fields`` are ``(text, weight)`` pairs; a hit in a lightly weighted
    qualifier field counts less than a hit in the head.
    """
    denominator = 0.0
    hits = 0.0
    for field_text, weight in fields:
        content = content_stems(stem_set(field_text))
        if not content:
            continue
        denominator += weight * len(content)
        hits += weight * sum(
            best_match_weight(stem_value, query_stems) for stem_value in content
        )
    return hits / denominator if denominator else 0.0


def candidate_precision(query_stems: set[str], candidate: str) -> float:
    """Candidate-side precision of a match, field-weighted for long names."""
    head, qualifiers = split_title(candidate)
    if not qualifiers:
        return _field_precision(query_stems, [(candidate, 1.0)])
    return _field_precision(query_stems, [(head, 1.0), (qualifiers, QUALIFIER_WEIGHT)])


def idf_precision(query: str, candidate: str, table: dict[str, float]) -> float:
    """IDF-weighted share of candidate content words explained by the query.

    Distinguishes "Обувь" from "Обувь ортопедическая детская" for the query
    "обувь детская": the narrower position carries a rare unmatched word.
    Long names are field-weighted (see :func:`split_title`).
    """
    query_stems = stem_set(query)
    if not query_stems:
        return 0.0
    head, qualifiers = split_title(candidate)
    if not qualifiers:
        fields = [(candidate, 1.0)]
    else:
        fields = [(head, 1.0), (qualifiers, QUALIFIER_WEIGHT)]
    total = 0.0
    hit = 0.0
    for field_text, weight in fields:
        for stem_value in content_stems(stem_set(field_text)):
            word_weight = table.get(stem_value, 1.0) * weight
            total += word_weight
            hit += word_weight * best_match_weight(stem_value, query_stems)
    return hit / total if total else 0.0


def idf_table(names: list[str]) -> dict[str, float]:
    """Document frequency table over stems of catalog names."""
    from math import log

    document_frequency: dict[str, int] = {}
    total = 0
    for name in names:
        total += 1
        for stem in stem_set(name):
            document_frequency[stem] = document_frequency.get(stem, 0) + 1
    table = {
        stem: log((total + 1) / (frequency + 1)) + 1.0
        for stem, frequency in document_frequency.items()
    }
    return expand_compound_stems(table)


def expand_compound_stems(table: dict[str, float]) -> dict[str, float]:
    """Add heads of compound words: "электрочайник" also teaches us "чайник"."""
    expanded = dict(table)
    for stem, value in list(table.items()):
        for prefix in COMPOUND_PREFIXES:
            if stem.startswith(prefix) and len(stem) - len(prefix) >= 4:
                head = stem[len(prefix) :]
                expanded.setdefault(head, value)
    return expanded


def build_known_stems(names: list[str]) -> set[str]:
    """Catalog vocabulary, including heads of compound words."""
    known: set[str] = set()
    for name in names:
        known |= stem_set(name)
    return set(expand_compound_stems({stem: 1.0 for stem in known}))


def is_latin_word(stem_value: str) -> bool:
    return any("a" <= char <= "z" or "0" <= char <= "9" for char in stem_value) and not any(
        "а" <= char <= "я" or char == "е" for char in stem_value
    )


def idf_coverage(query: str, candidate: str, table: dict[str, float]) -> float:
    """IDF-weighted query coverage: rare words dominate ("чайник" vs "электрический").

    Latin out-of-vocabulary stems (brand names, model indices) are ignored — the
    catalog never contains them. Cyrillic out-of-vocabulary stems are kept in the
    denominator with a high idf: a Russian product word missing from the catalog
    usually means the product itself is not listed.
    """
    query_stems = stem_set(query)
    candidate_stems = stem_set(candidate)
    if not query_stems or not table:
        return 0.0
    weights = query_stem_weights(query_stems, table, head=query_head(query))
    total_weight = 0.0
    hit_weight = 0.0
    for stem, weight in weights.items():
        total_weight += weight
        hit_weight += weight * best_match_weight(stem, candidate_stems)
    return hit_weight / total_weight if total_weight else 0.0


OOV_CYRILLIC_WEIGHT = 3.0


# Russian product names lead with the head noun ("Покрышка грузовая", "Мышь
# компьютерная"); it decides the category far more than the modifiers.
HEAD_WEIGHT_FACTOR = 2.0


def query_head(query: str) -> str | None:
    """First content stem of the query (the head noun in product names).

    Low-signal modifiers are skipped: in "мобильный телефон" the head is
    "телефон".
    """
    for stem_value in stems(query):
        if stem_value not in STOPWORD_STEMS and stem_value not in LOW_SIGNAL_STEMS:
            return stem_value
    return None


def query_stem_weights(
    query_stems: set[str], table: dict[str, float], head: str | None = None
) -> dict[str, float]:
    """Weights of query stems for coverage features.

    In-vocabulary stems use their idf. Latin unknowns (brands, model indices)
    are ignored entirely. Low-signal modifiers ("женский", "дюйм", "мобильный")
    get a small fixed weight. Other Cyrillic unknowns keep a
    moderate fixed weight: a Russian product word missing from the catalog is a
    signal that the product itself may not be listed ("Мышь компьютерная"), but
    unmatched modifiers ("мужские", "бытовой") must not crush the score either.
    The head noun (``head``) weighs ``HEAD_WEIGHT_FACTOR`` times more.
    """
    weights: dict[str, float] = {}
    for stem in content_stems(query_stems):
        if stem in LOW_SIGNAL_STEMS:
            # A modifier stays a modifier even when the catalog uses the word
            # ("мобильный", "настольный").
            weights[stem] = LOW_SIGNAL_WEIGHT
        elif stem in table:
            weights[stem] = table[stem]
        elif not is_latin_word(stem):
            weights[stem] = OOV_CYRILLIC_WEIGHT
    if head is not None and head in weights:
        weights[head] *= HEAD_WEIGHT_FACTOR
    return weights


def oov_cyr_share(query_stems: set[str], table: dict[str, float]) -> float:
    """Share of query weight carried by unmatched-style Cyrillic unknowns."""
    weights = query_stem_weights(query_stems, table)
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    oov = sum(weight for stem, weight in weights.items() if stem not in table)
    return oov / total


def jaccard_digits(query: str, candidate: str) -> float:
    query_digits = digits(query)
    candidate_digits = digits(candidate)
    if not query_digits or not candidate_digits:
        return 0.0
    union = query_digits | candidate_digits
    return len(query_digits & candidate_digits) / len(union)


def idf_weight(total_docs: int, doc_frequency: int) -> float:
    """Standard smoothed idf; used by the hashing embedder and the ranker."""
    return log((total_docs + 1) / (doc_frequency + 1)) + 1.0


def lexical_score(query: str, candidate: str) -> float:
    """Heuristic retrieval score used to shortlist candidates before reranking.

    Combines stem coverage of the query, phrase containment, and character
    similarity. The value is only meaningful for ordering candidates.
    """
    return lexical_score_prepared(prepare_candidate(query), prepare_candidate(candidate))


@dataclass
class PreparedText:
    """Precomputed features so retrieval can score many candidates cheaply.

    ``weights`` is optional: stem importance for the *query* side (head noun and
    bridged category words dominate, spec modifiers and Cyrillic unknowns weigh
    less). Without it coverage stays uniform, as before.

    ``head_stem_set``/``qualifier_stem_set`` are filled only for long names
    (see :func:`split_title`): the BM25F-style field split keeps the
    qualifier storm of a legal text from diluting the precision features.
    """

    raw: str
    normalized: str
    stem_list: list[str]
    stem_set: set[str]
    chars: Counter[str]
    digit_set: set[str]
    weights: dict[str, float] | None = None
    head_stem_set: set[str] | None = None
    qualifier_stem_set: set[str] | None = None


def prepare_candidate(text: str, idf: dict[str, float] | None = None) -> PreparedText:
    stem_list = stems(text)
    weights = None
    if idf is not None:
        weights = query_stem_weights(set(stem_list), idf, head=query_head(text))
    head_stems: set[str] | None = None
    qualifier_stems: set[str] | None = None
    if LONG_TITLE_SPLIT_ENABLED and len(text) > LONG_TITLE_CHARS:
        head, qualifiers = split_title(text)
        head_stems = stem_set(head)
        qualifier_stems = stem_set(qualifiers)
    return PreparedText(
        raw=text,
        normalized=normalize(text),
        stem_list=stem_list,
        stem_set=set(stem_list),
        chars=char_ngrams(text),
        digit_set=digits(text),
        weights=weights,
        head_stem_set=head_stems,
        qualifier_stem_set=qualifier_stems,
    )


def prepared_precision(query_stems: set[str], candidate: PreparedText) -> float:
    """Candidate-side precision of a match, field-weighted for long names."""
    if candidate.head_stem_set is None:
        content = content_stems(candidate.stem_set)
        if not content:
            return 0.0
        return sum(best_match_weight(stem, query_stems) for stem in content) / len(
            content
        )
    head_content = content_stems(candidate.head_stem_set)
    qualifier_content = content_stems(candidate.qualifier_stem_set or set())
    denominator = len(head_content) + QUALIFIER_WEIGHT * len(qualifier_content)
    if not denominator:
        return 0.0
    hits = sum(best_match_weight(stem, query_stems) for stem in head_content)
    hits += QUALIFIER_WEIGHT * sum(
        best_match_weight(stem, query_stems) for stem in qualifier_content
    )
    return hits / denominator


def lexical_score_prepared(query: PreparedText, candidate: PreparedText) -> float:
    if not query.raw.strip() or not candidate.raw.strip():
        return 0.0
    if not query.stem_list:
        return 0.0

    query_stems = query.stem_set
    candidate_stems = candidate.stem_set
    if query.weights:
        # Weighted coverage: a hit of the head noun ("электрочайник", bridged
        # "устройства вывода") must not be diluted by unmatched spec words.
        total = sum(query.weights.values())
        hit = sum(
            query.weights[stem] * best_match_weight(stem, candidate_stems)
            for stem in query_stems
            if stem in query.weights
        )
        cover = hit / total if total else 0.0
    else:
        query_hits = sum(best_match_weight(stem, candidate_stems) for stem in query_stems)
        cover = query_hits / len(query_stems) if query_stems else 0.0

    if cover == 0.0:
        # Character similarity still catches typos/transliterations.
        return 0.35 * cosine_sparse(query.chars, candidate.chars)

    precision = prepared_precision(query_stems, candidate)
    phrase_bonus = 0.0
    if query.normalized in candidate.normalized or candidate.normalized in query.normalized:
        phrase_bonus = 0.2
    char_similarity = cosine_sparse(query.chars, candidate.chars)
    return min(
        1.0,
        0.45 * cover + 0.15 * precision + 0.25 * char_similarity + phrase_bonus,
    )
