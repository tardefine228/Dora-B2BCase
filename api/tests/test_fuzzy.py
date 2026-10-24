"""Typo autocorrection must fix typos and leave real Russian words alone.

The guard used to be a pymorphy *parse score* threshold (0.6). pymorphy spreads
probability over interpretations, so ordinary inflected words whose top reading
wins only half the mass fell below it and were "corrected" into any similar
catalog word: «Платье женское летнее» → «… лесного», «Грабли садовые» → «…
судовые», «Корм для собак» → «корда для собак». Nine of the 274 labeled
queries were rewritten that way, so retrieval and every reranker feature saw a
different product than the specialist typed; 11 of the 27 noise words the
trainer adds to training queries were mangled too («мужской», «стальной»).
The guard is now ``Parse.is_known`` — "this word form is in the dictionary" —
which is exactly the property that separates real words from typos.
"""

from __future__ import annotations

import pytest

from ml.fuzzy import autocorrect
from ml.train_reranker import NOISE_WORDS

pytest.importorskip("pymorphy3", reason="autocorrection needs pymorphy3")

# Real Russian words that a score-based guard used to "correct".
REAL_WORDS = [
    "Платье женское летнее",
    "Двери металлические входные",
    "Матрас ватный",
    "Огнетушитель порошковый",
    "Часы наручные механические",
    "Шезлонг садовый",
    "Зонт складной",
    "Грабли садовые",
    "Корм для собак",
    "Мойка",
]

# Genuine typos the autocorrection exists for.
TYPOS = {
    "Монитр": "монитор",
    "Клавитура": "клавиатура",
    "Бумга": "бумага",
    "Дрел": "дрель",
}


@pytest.mark.parametrize("query", REAL_WORDS)
def test_real_words_are_never_rewritten(query: str) -> None:
    assert autocorrect(query) == query


@pytest.mark.parametrize(("typo", "expected"), sorted(TYPOS.items()))
def test_typos_are_still_corrected(typo: str, expected: str) -> None:
    assert expected in autocorrect(typo).lower()


@pytest.mark.parametrize("noise", NOISE_WORDS)
def test_training_noise_words_are_not_corrected(noise: str) -> None:
    """The trainer appends these to catalog names; mangling them corrupts training.

    Eleven of them («мужской» 0.33, «стальной» 0.29, «новый» 0.56, «бытовой»
    0.54, «настольный» 0.50, …) used to fall below the score gate.
    """
    assert autocorrect(noise) == noise


def test_unknown_word_without_catalog_neighbour_is_kept() -> None:
    # Nothing in the catalog is one edit away: the token must survive unchanged
    # instead of snapping to some distant position name.
    assert autocorrect("Ксерокопия") == "Ксерокопия"
