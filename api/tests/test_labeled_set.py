"""The measurement set must stay internally consistent.

``eval/labeled_examples.csv`` is the only ground truth the project has, so its
invariants are tested, not trusted:

* every row declares a ``label_kind`` (``code`` / ``absent`` / ``unclear``);
* a ``code`` row carries a code, and that code is a real catalog position;
* ``absent`` and ``unclear`` rows carry no code — otherwise the silence metric
  and the top-1 metric would both count the same row;
* every row whose verdict is not obvious is explained in ``eval/label_audit.csv``.

Why the third kind exists: the original "outside the list" rows were selected by
searching the product name in ``data/data.csv``, which is not the same question as
"does some position of the list cover these goods". Rows where only a residual
position («прочие / не включённые в другие группировки») could cover the goods are
kept visible but excluded from the percentages until a lawyer rules on them
(see ``docs/labeling-doctrine.md``).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LABELED = ROOT / "eval" / "labeled_examples.csv"
AUDIT = ROOT / "eval" / "label_audit.csv"
CATALOG = ROOT / "data" / "data.csv"
KINDS = {"code", "absent", "unclear"}


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


@pytest.fixture(scope="module")
def labeled() -> list[dict]:
    return _read(LABELED)


@pytest.fixture(scope="module")
def catalog_codes() -> set[str]:
    return {
        (row.get("okpd2") or "").strip()
        for row in _read(CATALOG)
        if (row.get("okpd2") or "").strip()
    }


def test_every_row_declares_a_known_kind(labeled):
    kinds = {(row.get("label_kind") or "").strip() for row in labeled}
    assert kinds <= KINDS, f"неизвестные label_kind: {kinds - KINDS}"
    assert all((row.get("label_kind") or "").strip() for row in labeled)


def test_code_rows_point_at_real_positions(labeled, catalog_codes):
    """A labelled code must exist in the catalog, or be a parent of a real code."""
    broken = []
    for row in labeled:
        if row["label_kind"] != "code":
            continue
        code = (row.get("expected_okpd2") or "").strip()
        if not code:
            broken.append((row["query"], "пустой код"))
        elif code not in catalog_codes and not any(
            catalog.startswith(code + ".") for catalog in catalog_codes
        ):
            broken.append((row["query"], code))
    assert not broken, f"коды вне каталога: {broken}"


def test_silence_rows_carry_no_code(labeled):
    leaking = [
        row["query"]
        for row in labeled
        if row["label_kind"] in {"absent", "unclear"} and (row.get("expected_okpd2") or "").strip()
    ]
    assert not leaking, f"ожидается тишина, но код заполнен: {leaking}"


def test_contested_rows_are_explained(labeled):
    """Every ``unclear`` row must say why, in the audit file."""
    audited = {
        row["query"]: row
        for row in _read(AUDIT)
        if (row.get("label_kind") or "").strip() == "unclear"
    }
    missing = [
        row["query"]
        for row in labeled
        if row["label_kind"] == "unclear" and row["query"] not in audited
    ]
    assert not missing, f"спорные строки без обоснования: {missing}"
    assert all((row.get("rationale") or "").strip() for row in audited.values())


def test_audit_matches_the_labels(labeled):
    """The audit is not a historical note: it must describe the current file."""
    by_query = {row["query"]: row for row in labeled}
    mismatched = []
    for entry in _read(AUDIT):
        row = by_query.get(entry["query"])
        if row is None:
            mismatched.append((entry["query"], "нет строки"))
            continue
        expected = (row.get("expected_okpd2") or "").strip() or "(silence)"
        if expected != entry["new_expected"] or row["label_kind"] != entry["label_kind"]:
            mismatched.append((entry["query"], entry["new_expected"], expected))
    assert not mismatched, f"аудит разошёлся с разметкой: {mismatched}"
