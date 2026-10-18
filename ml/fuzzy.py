"""Autocorrection of query typos against the catalog vocabulary.

Specialists type fast: «Монитр», «Клавитура», «Бумга», «Дрел Makitta». Neither
stem matching nor embeddings can rescue a token that is absent from the
vocabulary, so before retrieval the query tokens are corrected against the
catalog's stem vocabulary: bounded edit distance, length+first-letter bucketed
search, and only a unique best match is accepted (ties keep the typo, which
then simply behaves like an unknown word).
"""

from __future__ import annotations

import re
from functools import lru_cache

from ml.features import stem

_TOKEN_RE = re.compile(r"[а-яёa-z]+", re.IGNORECASE)
# Single-letter inflection tails: only these make a prefix relation a typo
# ("дрел" -> "дрель"); longer tails mean a different word ("футбол" -> "футболка").
_INFLECTION_TAILS = {"ь", "я", "ы", "й", "а", "о", "е", "и"}
# pymorphy parse score at or above which a token counts as a real dictionary
# word: those are never "corrected" ("мойка" must not become "майка").
_DICTIONARY_SCORE = 0.6


@lru_cache(maxsize=1)
def _morph():
    try:
        import pymorphy3

        return pymorphy3.MorphAnalyzer()
    except Exception:  # pragma: no cover - morph is an optional extra
        return None


def _is_dictionary_word(token: str) -> bool:
    morph = _morph()
    if morph is None:
        return False
    return morph.parse(token)[0].score >= _DICTIONARY_SCORE


def _edit_distance(a: str, b: str, cap: int) -> int:
    """Levenshtein distance with an early-exit cap."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        best = i
        for j, cb in enumerate(b, 1):
            value = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            cur.append(value)
            if value < best:
                best = value
        if best > cap:
            return cap + 1
        prev = cur
    return prev[-1]


@lru_cache(maxsize=1)
def _catalog_index() -> dict[tuple[int, str], list[tuple[str, str]]]:
    """``(stem length, first letter) -> [(stem, example token)]``.

    The vocabulary is the catalog itself plus the lexicon's product words:
    a typo of a bridged word («Клавитура», «Дрел») must land on the canonical
    product word so the bridge still fires afterwards.
    """
    from ml.catalog_loader import load_catalog
    from ml.lexicon import BRIDGES

    buckets: dict[tuple[int, str], list[tuple[str, str]]] = {}
    seen: set[str] = set()

    def add(token: str) -> None:
        s = stem(token)
        if len(s) < 4 or s in seen:
            return
        seen.add(s)
        buckets.setdefault((len(s), s[0]), []).append((s, token))

    for item in load_catalog():
        for token in _TOKEN_RE.findall(item["name"].lower()):
            add(token)
    for product_word in BRIDGES:
        add(product_word)
    return buckets


def autocorrect(text: str) -> str:
    """Replace typo tokens with the catalog token they almost surely meant."""
    index = _catalog_index()
    replacements: list[tuple[str, str]] = []
    for token in set(_TOKEN_RE.findall(text.lower())):
        s = stem(token)
        if len(s) < 4:
            continue
        if _is_dictionary_word(token):
            continue  # a real Russian word: absence from the catalog is not a typo
        bucket = index.get((len(s), s[0]))
        if bucket and any(s == entry for entry, _ in bucket):
            continue  # a real catalog word, nothing to correct
        cap = 1 if len(s) < 8 else 2
        best: tuple[int, str, str] | None = None
        tie = False
        for delta in (0, -1, 1):
            for candidate, example in index.get((len(s) + delta, s[0]), ()):
                distance = _edit_distance(s, candidate, cap)
                if distance > cap:
                    continue
                if best is None or distance < best[0]:
                    best, tie = (distance, candidate, example), False
                elif distance == best[0]:
                    tie = True
        if best is not None and not tie and best[1] != s:
            candidate, example = best[1], best[2]
            # Prefix relations are different words, not typos, unless the tail
            # is a single inflection letter ("дрел" -> "дрель").
            if candidate.startswith(s):
                if candidate[len(s) :] not in _INFLECTION_TAILS:
                    continue
            elif s.startswith(candidate):
                if s[len(candidate) :] not in _INFLECTION_TAILS:
                    continue
            replacements.append((token, example))
    for token, fixed in replacements:
        text = re.sub(
            rf"(?<![а-яёa-z]){re.escape(token)}(?![а-яёa-z])", fixed, text, flags=re.IGNORECASE
        )
    return text
