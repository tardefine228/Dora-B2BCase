"""OKPD2 code grammar in the catalog validator (DA-3).

The old pattern ``^\\d{2}(\\.\\d{1,2}){0,3}$`` rejected every full category
code with a three-digit tail — 731 of 1126 catalog rows, including valid codes
such as ``22.29.29.190``. The replacement follows the fixed-level OKPD2 grammar
(OK 034-2014): ``XX | XX.X | XX.XX | XX.XX.X | XX.XX.XX | XX.XX.XX.XXX``.
"""

from __future__ import annotations

import re

import pytest
from app.services.catalog_io import OKPD2_PATTERN, parse_catalog_bytes, validate_rows

OKPD2_RE = re.compile(OKPD2_PATTERN)

VALID_CODES = [
    "13",  # класс
    "13.2",  # подкласс
    "14.20",  # группа
    "20.59.1",  # подгруппа
    "16.21.13",  # вид
    "22.29.29.190",  # категория (раньше ошибочно отвергался)
    "10.61.11.000",
    "01.11.11.110",
]

INVALID_CODES = [
    "ABC",
    "1.2",  # класс всегда двузначный
    "13.",
    ".13.2",
    "012.2",
    "22.x",
    "22..2",
    "22.29.29.19",  # хвост категории — ровно 3 цифры
    "22.29.29.1900",  # четыре цифры в хвосте
    "22.29.190",  # трёхзначная группа на промежуточном уровне
    "22.299.1",  # трёхзначная группа на уровне группы
    "99.9.9.999",  # пропущен разряд группового уровня
    "01.1.11.110",
    "22.29.2.19",
]


@pytest.mark.parametrize("code", VALID_CODES)
def test_valid_okpd2_codes_pass(code: str) -> None:
    assert OKPD2_RE.match(code)


@pytest.mark.parametrize("code", INVALID_CODES)
def test_malformed_okpd2_codes_are_rejected(code: str) -> None:
    assert not OKPD2_RE.match(code)


def _catalog_csv(codes: list[str]) -> bytes:
    lines = ["record_id;appendix;position;name;okpd2;code_role"]
    lines += [
        f"t-{i};1;{i};Тестовая позиция {i};{code};Включён" for i, code in enumerate(codes, start=1)
    ]
    return ("\n".join(lines)).encode("utf-8")


def test_full_category_codes_are_no_longer_errors() -> None:
    parsed = parse_catalog_bytes("data.csv", _catalog_csv(["22.29.29.190", "10.61.11.000", "13.2"]))
    report = validate_rows(parsed)

    okpd2_errors = [issue for issue in report.errors if issue.field == "okpd2"]
    assert okpd2_errors == []
    assert report.rows_valid == 3


def test_malformed_code_still_blocks_the_row() -> None:
    parsed = parse_catalog_bytes("data.csv", _catalog_csv(["22.29.29.190", "ABC"]))
    report = validate_rows(parsed)

    okpd2_errors = [issue for issue in report.errors if issue.field == "okpd2"]
    assert len(okpd2_errors) == 1
    assert "ABC" in okpd2_errors[0].message
    assert report.rows_valid == 1


def test_shipped_catalog_and_snapshot_pass_every_code() -> None:
    """Regression guard: every non-empty okpd2 cell obeys the documented grammar."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "data" / "data.csv",
        root / "eval" / "source-snapshots" / "2026-10-26" / "pp_1875_dataframe.csv",
    ]
    for path in paths:
        parsed = parse_catalog_bytes(path.name, path.read_bytes())
        report = validate_rows(parsed)
        assert not [issue for issue in report.errors if issue.field == "okpd2"]
