"""The specifier vocabulary: what is an attribute of the goods and what is the goods.

`ml/specifiers.py` is used twice: `strip_specifiers` builds an *extra* scoring
variant (measured and rejected as a serving rule, эксп. 35 — it recovers 2 rows of
347 and can drop an identity word such as «чугунные», which separates radiators),
and the same vocabulary documents which words a purchase line may add without
changing the answer. These tests pin the mechanics, not a policy.
"""

from __future__ import annotations

from ml.specifiers import (
    SPEC_ADJECTIVES,
    SPECIFIER_STEMS,
    is_specifier_token,
    strip_specifiers,
)


def test_vocabulary_is_non_empty_and_stemmed():
    assert len(SPEC_ADJECTIVES) >= 50
    assert len(SPECIFIER_STEMS) == len(SPEC_ADJECTIVES)
    assert "цифровой" in SPEC_ADJECTIVES


def test_units_and_numbers_are_specifiers():
    for token in ("10", "220В", "1000Вт", "IP54", "16ГБ", "500мм", "Ду50", "10бар"):
        assert is_specifier_token(token), token


def test_product_words_and_latin_names_are_not_specifiers():
    # Latin tokens carry identity ("ssd", "usb", brand codes) and are bridged by
    # the lexicon, so they are never stripped.
    for token in ("ssd", "usb", "ups", "ноутбук", "осциллограф", "контейнер"):
        assert not is_specifier_token(token), token


def test_strip_keeps_the_product_words():
    assert strip_specifiers("Манометр цифровой технический") == "Манометр технический"
    assert strip_specifiers("Осциллограф цифровой запоминающий") == "Осциллограф"


def test_strip_returns_empty_when_there_is_nothing_to_strip():
    # No change and "everything was an attribute" both mean "no extra variant":
    # the caller then keeps the answer it already has.
    assert strip_specifiers("Осциллограф") == ""
    assert strip_specifiers("Светодиодный") == ""
    assert strip_specifiers("") == ""


def test_identity_adjectives_document_the_rejected_serving_rule():
    """«Чугунные» separates 25.21.11.110 from 25.21.11.120 — stripping is not free.

    This is why the stripper stayed a measurement tool: on the synthetic stress
    set the best-of-two-variants rule recovered 2 rows of 347 while turning a
    correct radiator answer into a neighbouring one (эксп. 35).
    """
    assert is_specifier_token("чугунные")
    assert strip_specifiers("Радиаторы центрального отопления и их секции чугунные") is not None
