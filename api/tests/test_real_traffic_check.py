"""Pure logic of the real-traffic drift check (no classifier needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

from real_traffic_check import diff_predictions, read_exports, render_markdown  # noqa: E402

HEADER = (
    "query;code;title;status;confidence;confidence_raw;clause_ref;clause_url;"
    "user_verdict;corrected_code;verifier;verifier_note;row"
)


def _export(path: Path, rows: list[str]) -> None:
    path.write_text("\ufeff" + HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def _line(query: str, code: str, status: str, confidence: str) -> str:
    return f"{query};{code};title;{status};{confidence};{confidence};;;;;;1"


def test_read_exports_tags_rows_and_skips_empty_queries(tmp_path: Path) -> None:
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    _export(first, [_line("Ноутбук", "26.20.11", "ok", "0.95"), _line("", "", "not_found", "")])
    _export(second, [_line("Коньки", "15.2", "ok", "0.81")])

    rows = read_exports([first, second])

    assert len(rows) == 2
    assert {row["query"] for row in rows} == {"Ноутбук", "Коньки"}
    assert rows[0]["source_file"] == "a.csv"
    assert rows[0]["source_row"] == 2
    assert len({row["source_sha256"] for row in rows}) == 2
    assert rows[0]["old_confidence"] == 0.95


def _answers() -> list[dict]:
    # same order as the two rows in test_read_exports
    return [
        {"code": "26.20.11", "status": "ok", "confidence": 0.94},  # unchanged
        {
            "code": "32.30.11.131",
            "status": "review",
            "confidence": 0.64,
        },  # code and status moved
    ]


def test_diff_counts_transitions_code_changes_and_bands(tmp_path: Path) -> None:
    path = tmp_path / "a.csv"
    _export(
        path,
        [
            _line("Ноутбук", "26.20.11", "ok", "0.95"),
            _line("Коньки", "15.2", "ok", "0.81"),
        ],
    )
    rows = read_exports([path])

    diff = diff_predictions(rows, _answers())

    assert diff["rows"] == 2
    assert diff["unchanged_answers"] == 1
    assert diff["transitions"] == {"ok -> review": 1}
    assert diff["status_now"] == {"ok": 1, "review": 1}
    assert len(diff["code_changes"]) == 1
    change = diff["code_changes"][0]
    assert change["query"] == "Коньки"
    assert change["old_code"] == "15.2" and change["new_code"] == "32.30.11.131"
    assert diff["confidence_bands"] == {">=0.8": 1, "0.6–0.8": 1}
    # Confidence drift is measured only on fully unchanged answers.
    assert diff["confidence_delta_on_unchanged"]["n"] == 1
    assert diff["confidence_delta_on_unchanged"]["max"] == 0.01


def test_not_found_to_ok_is_flagged_as_the_risky_direction(tmp_path: Path) -> None:
    path = tmp_path / "a.csv"
    _export(path, [_line("Соль", "", "not_found", "")])
    rows = read_exports([path])
    answers = [{"code": "10.84.30.130", "status": "ok", "confidence": 0.91}]

    diff = diff_predictions(rows, answers)
    report = render_markdown(diff, [{"name": "a.csv", "rows": 1, "sha256": "x" * 64}], {})

    assert diff["transitions"] == {"not_found -> ok": 1}
    risky = [
        c
        for c in diff["code_changes"]
        if c["old_status"] == "not_found" and c["new_status"] == "ok"
    ]
    assert len(risky) == 1
    assert "not_found -> ok" in report
    assert "Соль" in report


def test_diff_refuses_mismatched_lengths(tmp_path: Path) -> None:
    path = tmp_path / "a.csv"
    _export(path, [_line("Ноутбук", "26.20.11", "ok", "0.9")])
    rows = read_exports([path])

    with pytest.raises(ValueError, match="same length"):
        diff_predictions(rows, [])
