"""Long-title field split (ML-7): head + qualifiers.

Official PP-1875 names run to 200–1100 characters; scoring them as one field
lets the qualifier storm dilute the precision features. The split keeps short
names untouched (one field, identical scores) and weighs qualifier hits less for
long ones.

The split is **off by default** (``LONG_TITLE_SPLIT=1`` turns it on): the
labeled set has six rows whose expected position is over 200 characters, so the
quality gate cannot see the change at all. On the long-name slice it is a
trade-off — +2.2 p.p. top-1, but 13 ``review`` hints become confident *wrong*
answers — and that is a product decision, not a free win. These tests therefore pin both modes:
default off (identical to no split), and on (the field-weighted behaviour).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

import ml.features as features_module
from ml.features import (
    LONG_TITLE_CHARS,
    QUALIFIER_WEIGHT,
    candidate_precision,
    prepare_candidate,
    split_title,
    stem_set,
)


@pytest.fixture
def split_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable the field split — the mode these tests describe."""
    monkeypatch.setattr(features_module, "LONG_TITLE_SPLIT_ENABLED", True)


pytestmark = pytest.mark.usefixtures("split_on")

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "data" / "data.csv"


def _catalog_names() -> list[str]:
    with CATALOG.open(encoding="utf-8-sig", newline="") as handle:
        return [
            row["name"]
            for row in csv.DictReader(handle, delimiter=";")
            if row.get("name")
        ]


def test_short_names_stay_one_field() -> None:
    assert split_title("Обувь") == ("Обувь", "")
    assert split_title("Шины, покрышки и камеры резиновые новые") == (
        "Шины, покрышки и камеры резиновые новые",
        "",
    )


def test_long_name_splits_into_nonempty_head_and_qualifiers() -> None:
    long_names = [n for n in _catalog_names() if len(n) > LONG_TITLE_CHARS]
    assert long_names, "в каталоге нет длинных названий для теста"
    for name in long_names:
        head, qualifiers = split_title(name)
        assert head, name
        assert qualifiers, name
        # The split is lossless: the fields tile the original name.
        assert name.startswith(head) and name.endswith(qualifiers), name


def test_head_keeps_the_head_noun() -> None:
    name = next(
        n
        for n in _catalog_names()
        if n.startswith("Компьютеры портативные")
    )
    head, qualifiers = split_title(name)
    assert head.startswith("Компьютеры портативные")
    assert len(qualifiers) > 0
    assert len(head) < len(name)


def test_precision_rises_when_query_matches_the_head() -> None:
    name = next(
        n for n in _catalog_names() if n.startswith("Компьютеры портативные")
    )
    query = "компьютеры портативные"
    # Field-weighted precision beats the plain one exactly when
    # the query explains the head and misses most qualifiers.
    field_weighted = candidate_precision(stem_set(query), name)
    query_stems = stem_set(query)
    from ml.features import content_stems

    plain_content = content_stems(stem_set(name))
    plain = sum(
        1 for stem_value in plain_content if stem_value in query_stems
    ) / len(plain_content)
    assert field_weighted > plain


def test_precision_of_qualifier_only_match_is_discounted() -> None:
    name = next(
        n for n in _catalog_names() if n.startswith("Компьютеры портативные")
    )
    head, _qualifiers = split_title(name)
    # A word that lives only in the qualifiers ("ноутбук") must
    # weigh less than a head word ("компьютеры") of the same name.
    head_precision = candidate_precision(stem_set("компьютеры"), name)
    qualifier_precision = candidate_precision(stem_set("ноутбук"), name)
    assert head_precision > qualifier_precision


def test_prepared_text_carries_fields_only_for_long_names() -> None:
    short = prepare_candidate("Обувь")
    assert short.head_stem_set is None
    long_name = next(
        n for n in _catalog_names() if len(n) > LONG_TITLE_CHARS
    )
    long_prepared = prepare_candidate(long_name)
    assert long_prepared.head_stem_set is not None
    assert long_prepared.qualifier_stem_set is not None
    # The fields partition the stems of the full name.
    combined = long_prepared.head_stem_set | (
        long_prepared.qualifier_stem_set or set()
    )
    assert combined <= long_prepared.stem_set
    assert len(combined) > 0


def test_qualifier_weight_is_a_discount() -> None:
    assert 0.0 < QUALIFIER_WEIGHT < 1.0


def test_split_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the switch the scoring must be byte-identical to no split.

    The shipped artifact's weights were trained on one-field scores, so the
    default path may not change under it by accident.
    """
    monkeypatch.setattr(features_module, "LONG_TITLE_SPLIT_ENABLED", False)
    long_names = [n for n in _catalog_names() if len(n) > LONG_TITLE_CHARS]
    assert long_names
    for name in long_names:
        assert split_title(name) == (name, "")
        prepared = prepare_candidate(name)
        assert prepared.head_stem_set is None
        assert prepared.qualifier_stem_set is None
        # Field-weighted and plain precision coincide again.
        query = stem_set(name.split()[0])
        plain = features_module.coverage(name.split()[0], name)[1]
        assert candidate_precision(query, name) == pytest.approx(plain)
