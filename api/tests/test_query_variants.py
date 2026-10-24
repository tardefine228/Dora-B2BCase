"""Purchase lines are not product names (эксп. 30).

Two behaviours are pinned here:

* ``ml.query`` decides what part of a line is the name and what is a spec;
* ``CatalogClassifier`` scores both variants and keeps the more confident one,
  because a spec tail lowers confidence but may also disambiguate.
"""

from pathlib import Path

import pytest
from app.schemas import ClassifyResponse
from app.services.classifier import CatalogClassifier, _best_answer

from ml.query import name_variants, split_purchase_query


def _catalog_path() -> Path:
    path = Path(__file__).resolve().parents[2] / "data" / "data.csv"
    assert path.exists(), path
    return path


@pytest.fixture(scope="module")
def classifier() -> CatalogClassifier:
    return CatalogClassifier(_catalog_path())


@pytest.mark.parametrize(
    ("query", "head", "spec"),
    [
        # No separator: the whole line is the name.
        ("Шкаф металлический архивный", "Шкаф металлический архивный", ""),
        # ERP export: name, then attributes.
        (
            "Машина стиральная LG F2J3NS0W; бытовая; загрузка 6 кг; 1 шт",
            "Машина стиральная LG F2J3NS0W",
            "бытовая; загрузка 6 кг; 1 шт",
        ),
        # Spreadsheet variant with a vertical bar.
        ("Ноутбук Lenovo ThinkPad | 16 ГБ | 512 ГБ", "Ноутбук Lenovo ThinkPad", "16 ГБ | 512 ГБ"),
        # A separator inside a specification is not a name boundary.
        ("1 шт; 2 кг", "1 шт; 2 кг", ""),
        ("  ;;;  ", ";;;", ""),
        ("12; 5", "12; 5", ""),
        # Whitespace is collapsed, so variants are stable for the cache keys.
        ("Стол  письменный ;   дуб", "Стол письменный", "дуб"),
        ("", "", ""),
    ],
)
def test_split_purchase_query(query: str, head: str, spec: str) -> None:
    assert split_purchase_query(query) == (head, spec)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # A line without a spec is scored once: no second pass, no extra latency.
        ("Стул офисный", ("Стул офисный",)),
        # A line with a spec is scored twice, full line first (ties go to it).
        ("Стул офисный; кожзам; 1 шт", ("Стул офисный; кожзам; 1 шт", "Стул офисный")),
    ],
)
def test_name_variants(query: str, expected: tuple[str, ...]) -> None:
    assert name_variants(query) == expected


def _answer(status: str, raw: float, code: str | None = "31.01.1") -> ClassifyResponse:
    return ClassifyResponse(
        query="q",
        code=code if status != "not_found" else None,
        title=None if status == "not_found" else "Мебель для офисов и предприятий торговли",
        confidence=raw,
        confidence_raw=raw,
        status=status,  # type: ignore[arg-type]
        latency_ms=0.0,
    )


def test_best_answer_prefers_answered_over_silent() -> None:
    # A silent answer has raw 0.0 and would lose anyway, but the rule must be
    # explicit: "нет в перечне" is never an improvement over a code.
    silent = _answer("not_found", 0.0)
    review = _answer("review", 0.6)
    assert _best_answer([silent, review]) is review
    assert _best_answer([review, silent]) is review


def test_best_answer_prefers_higher_raw_confidence() -> None:
    low = _answer("review", 0.61)
    high = _answer("ok", 0.93)
    assert _best_answer([low, high]) is high


def test_best_answer_tie_goes_to_the_last_variant() -> None:
    # The last variant is the name alone: the cleaner input on equal evidence.
    full = _answer("ok", 0.9)
    name = _answer("ok", 0.9).model_copy(update={"code": "31.09.11"})
    assert _best_answer([full, name]) is name


def test_spec_tail_no_longer_costs_the_answer(classifier: CatalogClassifier) -> None:
    """A real ERP line: the tail used to push this below the floor, now it is a lead.

    The full line carries four specifiers the register never mentions, so the
    model stays below ``ok`` — but the position is head-anchored («Машина
    стиральная» is in the position's own name), so the answer is ``review``, not
    «нет в перечне» (эксп. 34). The name-only pass is confident, and the batch
    answer keeps that one.
    """
    line = "Машина стиральная LG F2J3NS0W; бытовая; загрузка 6 кг; отжим 1000 об/мин"
    full = classifier._classify_normalized(line)
    assert (full.status, full.code) == ("review", "27.51.13")
    answer = classifier.classify(line)
    assert (answer.status, answer.code) == ("ok", "27.51.13")
    # The response still echoes exactly what the user sent.
    assert answer.query == line


def test_disambiguating_tail_keeps_the_full_line_answer(classifier: CatalogClassifier) -> None:
    """The tail can also carry the decisive word, so it must not be cut away.

    «Плита Kronospan MDF» alone drifts to 22.21.4 (plastics); the tail
    «древесно-волокнистая» pins 16.21.14 (fibreboard).
    """
    line = "Плита Kronospan MDF; древесно-волокнистая; 16 мм; 2800x2070"
    answer = classifier.classify(line)
    assert (answer.status, answer.code) == ("ok", "16.21.14")


def test_classify_many_returns_one_answer_per_query(classifier: CatalogClassifier) -> None:
    queries = [
        "Шкаф металлический архивный; 2000х900х400; 1 шт",
        "Стул офисный",
        "Стол письменный; дуб; 1 шт",
    ]
    answers = classifier.classify_many(queries)
    assert [answer.query for answer in answers] == queries
    assert all(answer.code for answer in answers), [a.model_dump() for a in answers]
