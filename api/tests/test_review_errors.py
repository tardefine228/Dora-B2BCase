"""The error-review tool must separate model errors from labelling questions.

``eval/review_errors.py`` exists to answer one question about every wrong answer:
is the mechanic behind it one the dataset elsewhere counts as *correct*? If yes,
the row is a labelling decision, not a model defect — and optimizing the model
against it would learn a contradiction (measured: the rule that removes 5 answers
on absent classes costs 24 correct ones).
"""

from __future__ import annotations

from types import SimpleNamespace

from eval.review_errors import (
    answer_mechanic,
    classify_kind,
    code_family,
    head_stems,
)


def test_code_family_groups_sibling_positions() -> None:
    assert code_family("25.21.11.150") == code_family("25.21.11.160")
    assert code_family("32.20.12.111") != code_family("32.30.11.110")


def test_head_stems_takes_the_first_word_only() -> None:
    assert head_stems("Шредер для бумаги") == head_stems("Шредер")
    assert head_stems("Шредер для бумаги") & head_stems("Бумага и картон") == set()


def test_answer_through_a_bridge_on_a_generic_position_is_flagged() -> None:
    # «Принтер лазерный» → «Устройства ввода или вывода» comes from the lexicon
    # bridge and shares no word with the head noun: the mechanic the dataset
    # never marks correct.
    row = {"query": "Принтер лазерный", "expected_okpd2": ""}
    response = SimpleNamespace(code="26.20.16", title="Устройства ввода или вывода")
    assert classify_kind(row, response) == "generic_category"


def test_sibling_subcode_is_reported_as_such() -> None:
    row = {"query": "Конвектор отопительный стальной", "expected_okpd2": "25.21.11.160"}
    response = SimpleNamespace(code="25.21.11.150", title="Конвекторы отопительные стальные")
    assert classify_kind(row, response) == "sibling_code"


def test_exact_match_is_its_own_kind_not_a_sibling_code() -> None:
    """Every correct answer shares a code family with itself.

    Counting those as "sibling codes" made the mirror column meaningless: it read
    "the same mechanic is marked correct in 140 rows" for a mechanic that had no
    correct rows at all.
    """
    row = {"query": "Кроссовки мужские", "expected_okpd2": "15.2"}
    response = SimpleNamespace(code="15.2", title="Обувь")
    assert classify_kind(row, response) == "exact"


def test_mechanic_ignores_the_verdict() -> None:
    """The contradiction is between mechanics: same way of answering, opposite labels."""
    correct = {"query": "Кроссовки мужские", "expected_okpd2": "15.2"}
    wrong = {"query": "Конструктор детский", "expected_okpd2": ""}
    correct_response = SimpleNamespace(code="15.2", title="Обувь")
    wrong_response = SimpleNamespace(code="32.4", title="Игры и игрушки")

    assert answer_mechanic(correct, correct_response) == "generic_category"
    assert answer_mechanic(wrong, wrong_response) == "generic_category"
    # …while their verdict-level kinds differ, which is exactly what the dataset says.
    assert classify_kind(correct, correct_response) == "exact"
    assert classify_kind(wrong, wrong_response) == "generic_category"
