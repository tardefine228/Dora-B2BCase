"""Lexicon bridges: everyday product words must reach their abstract category,
and literal catalog names must keep resolving to themselves.

Pure-function tests run always; end-to-end expectations need the trained
artifact (``python scripts/build_model.py``) and are skipped without it.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from app.services.classifier import CatalogClassifier

from ml.features import STOPWORD_STEMS, content_stems, match_weight, stem, stems
from ml.lexicon import (
    BRIDGES,
    CANONICAL_CODES,
    bridged_targets,
    expand_query,
    substitute_query,
)
from ml.reranker import HeuristicReranker, load_reranker

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "data" / "data.csv"


def _catalog_rows() -> list[dict]:
    with CATALOG.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


# --- pure lexicon / feature helpers --------------------------------------------


def test_every_bridge_target_is_a_catalog_word_with_a_canonical_code() -> None:
    catalog_stems: set[str] = set()
    codes: set[str] = set()
    for row in _catalog_rows():
        catalog_stems |= set(stems(row["name"]))
        codes.add(row["okpd2"])
    for target in set(BRIDGES.values()):
        assert stem(target) in catalog_stems, f"мост ведёт в слово вне каталога: {target}"
        assert target in CANONICAL_CODES, f"у цели моста нет канонического кода: {target}"
    for target, code in CANONICAL_CODES.items():
        assert any(c == code or c.startswith(code + ".") for c in codes), (target, code)


def test_expand_keeps_query_and_appends_category_words() -> None:
    assert expand_query("Костюм мужской") == "Костюм мужской одежда"
    assert expand_query("Сапоги детские").endswith("обувь")
    assert expand_query("Материалы лакокрасочные") == "Материалы лакокрасочные"


def test_substitute_replaces_or_keeps_product_word_depending_on_candidate() -> None:
    assert substitute_query("Костюм мужской") == "одежда мужской"
    assert substitute_query("Шуба норковая") == "меховые норковая"
    # The candidate mentions the product word itself → keep it next to the category.
    assert substitute_query("ноутбук для работы", {"ноутбук"}) == "ноутбук портативные для работы"
    assert substitute_query("ноутбук для работы", {"кухон"}) == "портативные для работы"
    assert bridged_targets("двери металлические входные") == {"двери": "двери"}


def test_derivational_and_stopword_helpers() -> None:
    assert match_weight("металл", "металлическ") == pytest.approx(0.8)
    assert match_weight("двер", "дверн") == pytest.approx(0.8)
    assert match_weight("монитор", "мониторинг") == 0.0
    assert "кром" in STOPWORD_STEMS
    assert content_stems({"одежд", "кром", "из", "мех"}) == {"одежд", "мех"}
    assert content_stems({"из"}) == {"из"}


# --- end-to-end with the trained artifact ---------------------------------------


@pytest.fixture(scope="module")
def classifier() -> CatalogClassifier:
    reranker = load_reranker()
    if isinstance(reranker, HeuristicReranker):
        pytest.skip("models/reranker.json не собран (python scripts/build_model.py)")
    return CatalogClassifier(CATALOG, reranker=reranker)


@pytest.mark.parametrize(
    ("query", "code"),
    [
        ("Костюм мужской", "14.1"),
        ("Платье женское летнее", "14.1"),
        ("Куртка зимняя мужская", "14.1"),
        ("Шуба норковая", "14.20"),
        ("Сапоги детские", "15.2"),
        ("Ботинки кожаные мужские", "15.2"),
        ("Ситец набивной", "13.2"),
        ("Покрышка грузовая", "22.11.1"),
        ("Двери металлические входные", "25.12.10.000"),
    ],
)
def test_bridged_queries_reach_their_category(classifier, query: str, code: str) -> None:
    response = classifier.classify(query)
    assert response.code == code, (query, response.code, response.title, response.confidence)
    assert response.status in {"ok", "review"}


@pytest.mark.parametrize(
    ("query", "code"),
    [
        ("Материалы лакокрасочные", "20.30"),
        ("Электрочайники", "27.51.24.110"),
        ("Обувь", "15.2"),
        ("Одежда, кроме одежды из меха", "14.1"),
        ("Изделия меховые", "14.20"),
        ("Ткани текстильные", "13.2"),
        ("Шины, покрышки и камеры резиновые новые", "22.11.1"),
        ("Двери, окна и их рамы и пороги для дверей из металлов", "25.12.10.000"),
        ("Обувь ортопедическая детская", "32.50.22.150"),
        ("Одежда защитная огнестойкая", "32.99.11.140"),
    ],
)
def test_exact_catalog_names_are_not_broken_by_bridges(classifier, query: str, code: str) -> None:
    response = classifier.classify(query)
    assert response.code == code, (query, response.code, response.title)
    assert response.confidence >= 0.5


def test_lemma_stems_unify_inflected_forms():
    from ml.features import lemmatizer_active, stems

    if not lemmatizer_active():
        pytest.skip("pymorphy3 is not installed")
    assert stems("Станки") == stems("станок")
    assert stems("Гобои") == stems("Гобой")
    assert stems("Счётчики") == stems("счетчик")


@pytest.mark.parametrize("query", ["Камера видеонаблюдения сетевая", "Камеры телевизионные", "Огнетушитель порошковый", "Порошок металлический"])
def test_ambiguous_products_do_not_bridge_to_unrelated_categories(query):
    assert not ({"покрышки", "моющие"} & set(bridged_targets(query).values()))


@pytest.mark.parametrize(("query", "target"), [("Камера велосипедная", "покрышки"), ("Камеры резиновые", "покрышки"), ("Порошок стиральный", "моющие")])
def test_ambiguous_products_keep_bridges_in_confirmed_context(query, target):
    assert target in bridged_targets(query).values()


def test_training_does_not_label_ambiguous_bare_words_as_positive():
    from ml.train_reranker import bridge_sources_by_name
    catalog = [
        {"name": "Шины, покрышки и камеры резиновые новые", "pp_code": "22.11.1"},
        {"name": "Средства моющие и стиральные", "pp_code": "20.41.32"},
    ]
    sources = {word for words in bridge_sources_by_name(catalog).values() for word in words}
    assert "камера" not in sources
    assert "порошок" not in sources
    assert "шина" in sources
