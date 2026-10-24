"""The labeling loop converts specialist verdicts — and nothing else — into labels."""

import csv
import sqlite3
import subprocess
import sys
from pathlib import Path

from eval.feedback_loop import HEADER, LABELED, import_rows, read_feedback_sqlite, write_labels

ROOT = Path(__file__).resolve().parents[2]


def _labels(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def _row(query, verdict, code="", corrected="", status=""):
    return {
        "query": query,
        "user_verdict": verdict,
        "code": code,
        "corrected_code": corrected,
        "status": status,
    }


def test_correct_and_incorrect_become_code_labels():
    report = import_rows(
        [
            _row("Стул офисный", "correct", "31.01.1", status="review"),
            _row("Сейф офисный", "incorrect", "31.01.1", "28.23.23", status="ok"),
        ],
        known=set(),
        existing=set(),
    )
    assert [item.expected_okpd2 for item in report.imported] == ["31.01.1", "28.23.23"]
    assert all(item.label_kind == "code" for item in report.imported)
    assert "review" in report.imported[0].notes


def test_absent_verdict_becomes_an_absent_row():
    report = import_rows(
        [_row("ИБП 650 ВА", "absent", status="not_found")], known=set(), existing=set()
    )
    assert report.imported[0].label_kind == "absent"
    assert report.imported[0].expected_okpd2 == ""


def test_rows_without_a_verdict_are_not_labels():
    """The normal state of a fresh export: logs, not labels."""
    report = import_rows(
        [_row("Ноутбук", "", "26.20.11", status="ok")], known=set(), existing=set()
    )
    assert report.imported == []
    assert report.skipped_no_verdict == 1


def test_verdicts_without_the_code_they_need_are_skipped():
    report = import_rows(
        [
            _row("Ноутбук", "correct", code="", status="review"),  # "верно" без кода
            _row("Стул", "incorrect", "31.01.1", corrected="", status="review"),  # нет исправления
            _row("Мышь", "maybe", "26.20.16"),  # неизвестный вердикт
        ],
        known=set(),
        existing=set(),
    )
    assert report.imported == []
    assert report.skipped_invalid == 3
    assert len(report.reasons) == 3


def test_duplicates_against_the_measurement_set_and_the_staging_file_are_dropped():
    known = {" ".join("Стул офисный".casefold().split())}
    report = import_rows(
        [
            _row("стул   ОФИСНЫЙ", "correct", "31.01.1"),
            _row("Тумба приставная", "correct", "31.01.1"),
            _row("Тумба приставная", "correct", "31.01.1"),
        ],
        known=known,
        existing=set(),
    )
    assert [item.query for item in report.imported] == ["Тумба приставная"]
    assert report.duplicates == 2


def test_report_counts_the_review_band_and_false_codes():
    report = import_rows(
        [
            _row("Стул", "correct", "31.01.1", status="review"),
            _row("Кресло", "correct", "31.01.1", status="review"),
            _row("Кофемашина", "absent", status="ok"),
        ],
        known=set(),
        existing=set(),
    )
    assert report.by_verdict_status[("correct", "review")] == 2
    assert report.by_verdict_status[("absent", "ok")] == 1


def test_write_labels_appends_and_keeps_the_header(tmp_path):
    out = tmp_path / "labels.csv"
    first = import_rows(
        [_row("Стул", "correct", "31.01.1")], known=set(), existing=set()
    )
    write_labels(out, first.imported)
    second = import_rows(
        [_row("Кресло", "correct", "31.01.1")], known=set(), existing=set()
    )
    write_labels(out, second.imported)
    rows = _labels(out)
    assert [row["query"] for row in rows] == ["Стул", "Кресло"]
    assert list(rows[0]) == HEADER


def test_feedback_sqlite_is_readable(tmp_path):
    path = tmp_path / "classification_feedback.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE classification_feedback (
              id TEXT PRIMARY KEY, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              query TEXT NOT NULL, predicted_code TEXT, verdict TEXT NOT NULL, corrected_code TEXT
            )"""
        )
        connection.execute(
            "INSERT INTO classification_feedback"
            " (id, query, predicted_code, verdict, corrected_code)"
            " VALUES ('1', 'Стул офисный', '31.01.1', 'correct', NULL)"
        )
    rows = read_feedback_sqlite(path)
    assert rows == [
        {
            "query": "Стул офисный",
            "predicted_code": "31.01.1",
            "verdict": "correct",
            "corrected_code": None,
        }
    ]
    report = import_rows(rows, known=set(), existing=set())
    assert report.imported[0].expected_okpd2 == "31.01.1"


def test_cli_reports_an_export_without_verdicts(tmp_path):
    export = tmp_path / "results.csv"
    with export.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";", lineterminator="\n")
        writer.writerow([
            "query", "code", "title", "status", "confidence", "confidence_raw",
            "clause_ref", "clause_url", "user_verdict", "corrected_code",
        ])
        writer.writerow([
            "Стул офисный", "31.01.1", "Мебель", "ok", "0.9", "0.9", "", "", "", "",
        ])
    result = subprocess.run(
        [sys.executable, str(ROOT / "eval" / "feedback_loop.py"),
         "--input", str(export), "--dry-run"],
        capture_output=True, text=True, check=True, cwd=ROOT,
    )
    assert "Импортировано новых строк: 0" in result.stdout
    assert "журнал классификаций, а не разметка" in result.stdout


def test_the_measurement_set_is_never_written_by_the_tool():
    # A doctrine guard: the staging file is separate, and the tool has no flag
    # to point --out at the measurement set silently.
    assert LABELED.name == "labeled_examples.csv"
    assert LABELED.parent.name == "eval"
