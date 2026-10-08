"""Dependency-free text features shared by the catalog ranker and the ML reranker.

The goal is robust matching of short Russian product names against the official
PP-1875 position names: light stemming tolerates inflections, character n-grams
tolerate typos and word order changes, and digit tokens keep measurement details
(``10 м``, ``220 В``) significant. Everything here is pure Python so the catalog
backend keeps working without numpy or ML dependencies installed.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
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


def stem(token: str) -> str:
    """Very light Russian/English suffix stripper, safe for short tokens."""
    word = normalize(token)
    if word.isdigit() or len(word) <= 3:
        return word
    for suffix in _RU_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    for suffix in _EN_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


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


def match_weight(query_stem: str, candidate_stem: str) -> float:
    """1.0 for the same stem, 0.7 for a compound-word hit, 0.0 otherwise."""
    if query_stem == candidate_stem:
        return 1.0
    longer, shorter = (
        (candidate_stem, query_stem)
        if len(candidate_stem) >= len(query_stem)
        else (query_stem, candidate_stem)
    )
    if not shorter or shorter not in longer:
        return 0.0
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


def coverage(query: str, candidate: str) -> tuple[float, float]:
    """Return (query stem coverage in candidate, candidate stem precision)."""
    query_stems = stem_set(query)
    candidate_stems = stem_set(candidate)
    if not query_stems or not candidate_stems:
        return 0.0, 0.0
    query_hits = sum(best_match_weight(stem, candidate_stems) for stem in query_stems)
    candidate_hits = sum(best_match_weight(stem, query_stems) for stem in candidate_stems)
    return query_hits / len(query_stems), candidate_hits / len(candidate_stems)


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
    weights = query_stem_weights(query_stems, table)
    total_weight = 0.0
    hit_weight = 0.0
    for stem, weight in weights.items():
        total_weight += weight
        hit_weight += weight * best_match_weight(stem, candidate_stems)
    return hit_weight / total_weight if total_weight else 0.0


OOV_CYRILLIC_WEIGHT = 3.0


def query_stem_weights(query_stems: set[str], table: dict[str, float]) -> dict[str, float]:
    """Weights of query stems for coverage features.

    In-vocabulary stems use their idf. Latin unknowns (brands, model indices)
    are ignored entirely. Cyrillic unknowns keep a moderate fixed weight: a
    Russian product word missing from the catalog is a signal that the product
    itself may not be listed ("Мышь компьютерная"), but unmatched modifiers
    ("мужские", "бытовой") must not crush the score either.
    """
    weights: dict[str, float] = {}
    for stem in query_stems:
        if stem in table:
            weights[stem] = table[stem]
        elif not is_latin_word(stem):
            weights[stem] = OOV_CYRILLIC_WEIGHT
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
    """Precomputed features so retrieval can score many candidates cheaply."""

    raw: str
    normalized: str
    stem_list: list[str]
    stem_set: set[str]
    chars: Counter[str]
    digit_set: set[str]


def prepare_candidate(text: str) -> PreparedText:
    stem_list = stems(text)
    return PreparedText(
        raw=text,
        normalized=normalize(text),
        stem_list=stem_list,
        stem_set=set(stem_list),
        chars=char_ngrams(text),
        digit_set=digits(text),
    )


def lexical_score_prepared(query: PreparedText, candidate: PreparedText) -> float:
    if not query.raw.strip() or not candidate.raw.strip():
        return 0.0
    if not query.stem_list:
        return 0.0

    query_stems = query.stem_set
    candidate_stems = candidate.stem_set
    query_hits = sum(best_match_weight(stem, candidate_stems) for stem in query_stems)
    cover = query_hits / len(query_stems) if query_stems else 0.0

    if cover == 0.0:
        # Character similarity still catches typos/transliterations.
        return 0.35 * cosine_sparse(query.chars, candidate.chars)

    candidate_hits = sum(best_match_weight(stem, query_stems) for stem in candidate_stems)
    precision = candidate_hits / len(candidate_stems) if candidate_stems else 0.0
    phrase_bonus = 0.0
    if query.normalized in candidate.normalized or candidate.normalized in query.normalized:
        phrase_bonus = 0.2
    char_similarity = cosine_sparse(query.chars, candidate.chars)
    return min(
        1.0,
        0.45 * cover + 0.15 * precision + 0.25 * char_similarity + phrase_bonus,
    )
