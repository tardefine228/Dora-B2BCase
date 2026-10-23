"""Pair-level bans (ML-3): the five confident false friends.

A pair veto is a deliberate, reviewable ban of a (query word,
position) combination: the query word and the position name
legitimately share words, so no threshold and no embedder can
separate them — only an explicit rule can. The regression pins
both directions: the false friends stay silent AND the
legitimate queries that share the anchor word keep answering.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.lexicon import (
    PAIR_VETO,
    PAIR_VETO_CONTEXT,
    code_is_vetoed,
    vetoed_codes,
)

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "data" / "data.csv"


def _codes() -> set[str]:
    import csv

    with CATALOG.open(encoding="utf-8-sig", newline="") as handle:
        return {
        row["okpd2"].strip()
        for row in csv.DictReader(handle, delimiter=";")
        if row.get("okpd2")
    }


# --- the table itself ------------------------------------------------------


def test_every_vetoed_prefix_is_a_real_catalog_code() -> None:
    codes = _codes()
    banned: set[str] = set()
    for prefixes in PAIR_VETO.values():
        banned |= set(prefixes)
    for _anchor, (_context, prefixes) in PAIR_VETO_CONTEXT.items():
        banned |= set(prefixes)
    for prefix in banned:
        assert any(
            code == prefix or code.startswith(prefix + ".") for code in codes
        ), f"запрет ведёт в несуществующий код: {prefix}"


def test_camera_veto_is_context_gated() -> None:
    # Surveillance/web/digital cameras ban the rubber-tube position...
    assert "22.11.1" in vetoed_codes("Камера видеонаблюдения")
    assert "22.11.1" in vetoed_codes("Веб-камера")
    assert "22.11.1" in vetoed_codes("IP-камера")
    assert "22.11.1" in vetoed_codes("Видеокамера")
    # ...but a bicycle tube or a cold chamber is a legitimate
    # «камера» and must NOT be banned.
    assert vetoed_codes("Камера велосипедная") == set()
    assert vetoed_codes("Камера холодильная сборная") == set()
    assert vetoed_codes("Шины покрышки и камеры резиновые новые") == set()


def test_code_is_vetoed_matches_prefixes() -> None:
    assert code_is_vetoed("Шредер для бумаги", "17.12")
    assert code_is_vetoed("Шредер для бумаги", "17.12.1")
    assert not code_is_vetoed("Шредер для бумаги", "17.13")
    assert not code_is_vetoed("Камера велосипедная", "22.11.1")


# --- end-to-end: the five false friends ------------------------------------


@pytest.fixture(scope="module")
def classifier():
    from app.services.classifier import CatalogClassifier

    from ml.reranker import load_reranker

    return CatalogClassifier(CATALOG, reranker=load_reranker())


FALSE_FRIENDS = [
    ("Камера видеонаблюдения", "22.11.1"),
    ("Веб-камера", "22.11.1"),
    ("Шредер для бумаги", "17.12"),
    ("Конструктор детский", "32.4"),
    ("Огнетушитель порошковый", "20.41.32"),
]


@pytest.mark.parametrize("query,banned", FALSE_FRIENDS)
def test_confident_false_friends_are_silenced(classifier, query, banned) -> None:
    response = classifier.classify(query)
    assert response.status == "not_found", (
        f"ложный друг {query!r} всё ещё предлагает {response.code}"
    )
    assert response.code is None
    # The banned position must not hide in the alternatives either.
    assert all(
        not (alt.code == banned or alt.code.startswith(banned + "."))
        for alt in response.alternatives
    )


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Камера велосипедная", "22.11.1"),
        ("Камера холодильная сборная", "28.25.13.112"),
        ("Шины покрышки и камеры резиновые новые", "22.11.1"),
        ("Ноутбук Lenovo IdeaPad", "26.20.11"),
        ("Кроссовки", "15.2"),
    ],
)
def test_legitimate_queries_with_anchor_words_still_answer(
    classifier, query, expected
) -> None:
    response = classifier.classify(query)
    assert response.code is not None
    assert response.code == expected or expected.startswith(
        response.code + "."
    ) or response.code.startswith(expected + "."), (
        f"{query!r}: ожидался {expected}, получен {response.code}"
    )


def test_exact_catalog_names_with_camera_word_survive(classifier) -> None:
    # «Камера холодильная сборная» is a literal catalog position:
    # the context gate must never silence a verbatim name.
    response = classifier.classify("Камера холодильная сборная")
    assert response.code == "28.25.13.112"
