"""A weak but anchored lead is shown as ``review`` instead of «нет в перечне».

Real purchase lines carry specifiers the register never mentions
("Осциллограф цифровой запоминающий"): every unknown word lowers the pair
probability, so a correct rank used to reach the specialist as an empty cell
(эксп. 34). The rescue is deliberately narrow — it may only promote a lead whose
*product word* is written in the position's own name, it never returns ``ok``,
and it leaves scope modifiers («офисный») out of the anchor, so a product the
register does not contain stays silent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from app.services.classifier import CatalogClassifier, _head_anchored, _query_head_stems

from ml.features import stems
from ml.reranker import HeuristicReranker, load_reranker

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "data" / "data.csv"


@pytest.fixture(scope="module")
def classifier() -> CatalogClassifier:
    reranker = load_reranker()
    if isinstance(reranker, HeuristicReranker):
        pytest.skip("models/reranker.json не собран (python scripts/build_model.py)")
    return CatalogClassifier(CATALOG, reranker=reranker)


# --- the anchor itself ---------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected_head"),
    [
        # Голова строки — первое слово-товар, а не модификатор области.
        ("Сейф офисный", "сейф"),
        ("Счетчик электрической энергии трехфазный", "счетчик"),
        ("Щётка для обуви", "щетк"),
        ("Осциллограф цифровой запоминающий", "осциллограф"),
    ],
)
def test_head_skips_scope_modifiers_and_stopwords(query, expected_head):
    assert expected_head in _query_head_stems(query)


def test_bridge_target_anchors_the_position_the_lexicon_points_at():
    """«Портландцемент» — не словарное слово перечня, но мост ведёт на «цемент»."""
    heads = _query_head_stems("Портландцемент общестроительный М500")
    assert "портландцемент" in heads
    assert "цемент" in heads


class _Lead:
    def __init__(self, title: str, probability: float = 0.2) -> None:
        self.title = title
        self.probability = probability


@pytest.mark.parametrize(
    ("query", "title", "anchored"),
    [
        ("Осциллограф цифровой запоминающий", "Осциллоскопы и осциллографы", True),
        # «Офисный» — область, а не товар: сейф не становится офисной машиной.
        ("Сейф офисный", "Машины офисные прочие", False),
        # Зависимое слово не делает щётку обувью.
        ("Щётка для обуви", "Обувь, кроме одежды из меха", False),
        ("Бульдозер колесный тяжелый", "Бульдозеры и бульдозеры с поворотным отвалом", True),
    ],
)
def test_anchor_uses_the_head_word_of_the_line(query, title, anchored):
    assert _head_anchored(_Lead(title), query) is anchored


def test_the_title_stem_set_is_what_the_anchor_compares_against():
    """Guard against a silent change of the comparison (it is stems, not text)."""
    assert "бульдозер" in set(stems("Бульдозеры и бульдозеры с поворотным отвалом"))


# --- the decision --------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "code"),
    [
        # Позиция есть в перечне, но строка несёт спецификаторы: раньше «—».
        ("Осциллограф цифровой запоминающий", "26.51.42"),
        ("Трансформатор силовой масляный", "27.11.4"),
        ("Счетчик электрической энергии трехфазный", "26.51.63.130"),
        ("Бульдозер колесный тяжелый", "28.92.21"),
        ("Самосвал карьерный трехосный", "28.92.29"),
    ],
)
def test_spec_heavy_line_answers_with_its_position(classifier, query, code):
    answer = classifier.classify(query)
    assert answer.status != "not_found", query
    assert answer.code == code


@pytest.mark.parametrize(
    "query",
    [
        # Позиции нет в перечне, и «офисный» не должен её подсказывать.
        "Сейф офисный",
        "Зеркало настенное офисное",
        "Аптечка первой помощи офисная",
    ],
)
def test_absent_product_with_a_scope_modifier_stays_silent(classifier, query):
    assert classifier.classify(query).status == "not_found", query


def test_rescue_never_claims_ok(classifier):
    """A lexical lead is a question for the specialist, never a confident answer."""
    for query in (
        "Осциллограф цифровой запоминающий",
        "Внешний жесткий диск (HDD)",
        "Станок шлифовальный круглошлифовальный",
    ):
        answer = classifier.classify(query)
        assert answer.status in {"review", "ok", "not_found"}
        if answer.status == "review":
            # A rescued answer must carry the code, a source link and the score.
            assert answer.code
            assert answer.confidence is not None
            assert answer.confidence_raw is not None


def test_rescue_keeps_runner_up_positions(classifier):
    answer = classifier.classify("Станок фрезерный с ЧПУ вертикальный")
    assert answer.status == "review"
    assert answer.alternatives or answer.code


def test_invariant_five_holds_for_rescued_answers(classifier):
    """No code, no answer: a codeless position can never be promoted by the rescue."""
    for query in (
        "Осциллограф цифровой запоминающий",
        "Самосвал карьерный трехосный",
        "Трансформатор силовой масляный",
    ):
        answer = classifier.classify(query)
        if answer.status != "not_found":
            assert answer.code
