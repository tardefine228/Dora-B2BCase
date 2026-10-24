"""The threshold cross-validation must be able to say "this is noise".

``eval/threshold_cv.py`` exists because the sweep in ``eval/eval_report.md``
shows a tempting cell (``review = 0.655``: same top-1/top-3, four more rows of
correct silence) that was read off the same 274 rows the metric is measured on.
The tool answers whether picking such a cell generalises. These tests pin the
mechanics it relies on, not the measurement itself (that lives in
``eval/experiments.md``, experiment 26).
"""

from __future__ import annotations

import random

from eval.threshold_cv import (
    best_threshold,
    candidate_thresholds,
    metrics_at,
    stratified_folds,
)


def _row(raw: float, expected: str, code: str | None, correct: bool = True) -> dict:
    return {
        "query": "q",
        "expected": expected,
        "code": code,
        "raw": raw,
        "answered_alts": [],
        "silent_alts": [],
        "baseline_status": "ok",
        "disputed": False,
        "_correct": correct,
    }


def test_threshold_only_decides_whether_to_answer() -> None:
    """Raising the threshold cannot change which code is first — only silence."""
    rows = [_row(0.9, "26.20.11", "26.20.11"), _row(0.5, "26.20.11", "26.20.11")]
    answered = metrics_at(rows, 0.4)
    strict = metrics_at(rows, 0.7)
    assert answered["top1_expected"] == 2
    assert strict["top1_expected"] == 1
    assert strict["answered"] == 1


def test_out_of_scope_rows_count_as_correct_only_when_silent() -> None:
    rows = [_row(0.9, "", "17.12"), _row(0.3, "", "13.2")]
    assert metrics_at(rows, 0.5)["silence"] == 1
    assert metrics_at(rows, 0.1)["silence"] == 0


def test_a_silent_row_can_still_earn_top3_through_an_anchored_hint() -> None:
    rows = [_row(0.9, "17.12", None)]
    assert metrics_at(rows, 0.5)["top3_expected"] == 0
    rows[0]["silent_alts"] = ["17.12"]
    assert metrics_at(rows, 0.5)["top3_expected"] == 1


def test_candidate_thresholds_cover_every_distinct_decision() -> None:
    rows = [_row(0.9, "a", "a"), _row(0.5, "a", "a"), _row(0.5, "a", "a")]
    values = candidate_thresholds(rows, 0.62)
    assert 0.62 in values
    # 0.5 appears once although two rows share it: same decision.
    assert len([value for value in values if value == 0.5]) == 1


def test_best_threshold_prefers_silence_on_a_tie() -> None:
    """When two thresholds score the same, the safer one wins."""
    rows = [
        _row(0.9, "a", "a"),  # in scope, correct
        _row(0.9, "", "b"),  # out of scope, answering is wrong
        _row(0.4, "", "c"),  # out of scope, silent only above 0.4
    ]
    # Answering the second row costs silence without buying top-1, so the tool must
    # climb above 0.9 — a candidate that "answer nothing" needs.
    assert best_threshold(rows, "top1_all") > 0.9


def test_folds_keep_both_classes_in_every_split() -> None:
    rows = [_row(0.9, "a", "a") for _ in range(10)] + [_row(0.3, "", None) for _ in range(10)]
    for fold in stratified_folds(rows, 5, random.Random(1)):
        expected = sum(1 for index in fold if rows[index]["expected"])
        assert 0 < expected < 10
        assert 0 < len(fold) - expected < 10
