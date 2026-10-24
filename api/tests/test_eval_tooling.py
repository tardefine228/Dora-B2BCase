"""The measurement tooling must count what it prints.

Everything the gates decide on comes from ``metric_counts``: the report table, the
recorded baselines and the two gates. If the printed rate and the compared count ever
came from different places, a build could pass while the report showed a regression.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

from run_eval import metric_counts, summarize, sweep_thresholds  # noqa: E402


def _row(expected: str, correct: bool, *, status: str = "ok", top3: bool | None = None) -> dict:
    return {
        "expected": expected,
        "predicted": "x" if correct else "y",
        "status": status,
        "confidence": 0.9,
        "confidence_raw": 0.9,
        "correct_top1": correct,
        "correct_top3": correct if top3 is None else top3,
        "disputed": False,
    }


def _report() -> dict:
    return {
        "results": [
            _row("26.20.11", True),
            _row("26.20.11", False),
            _row("26.20.11", False, status="review", top3=True),
            _row("(not_found)", True, status="not_found"),
            _row("(not_found)", False, status="not_found"),
        ],
        "exact": [_row("26.20.11", True), _row("26.20.11", False)],
    }


def test_counts_use_the_right_denominators():
    counts = metric_counts(_report())
    assert counts["top1_expected"].correct == 1 and counts["top1_expected"].total == 3
    assert counts["top3_expected"].correct == 2 and counts["top3_expected"].total == 3
    assert counts["correct_silence"].correct == 1 and counts["correct_silence"].total == 2
    assert counts["top1_all"].correct == 2 and counts["top1_all"].total == 5
    assert counts["exact_top1"].correct == 1 and counts["exact_top1"].total == 2


def test_ok_precision_counts_confident_answers_only():
    """`review` and `not_found` answers are not confident, so they are outside it."""
    counts = metric_counts(_report())
    # Two rows have status "ok": one right, one wrong. The wrong "review" row is out.
    assert counts["ok_precision"].correct == 1 and counts["ok_precision"].total == 2


def test_summary_rates_are_derived_from_the_same_counts():
    report = _report()
    summary = summarize(report)
    counts = metric_counts(report)
    for metric in ("top1_expected", "top3_expected", "correct_silence", "top1_all", "ok_precision"):
        assert summary[metric] == round(counts[metric].rate, 1), metric


def test_sweep_thresholds_stay_inside_the_observed_scale():
    """A fixed grid cannot describe both the raw qdrant scale and the calibrated one."""
    confidences = [0.58, 0.61, 0.63, 0.7, 0.72, 0.75, 0.8, 0.85, 0.9, 1.0]
    thresholds = sweep_thresholds(confidences, own=0.62, points=4)
    assert thresholds == sorted(thresholds)
    assert 0.58 <= thresholds[0] and thresholds[-1] <= 1.0
    assert 0.62 in thresholds, "своя рабочая точка обязана попасть в таблицу"
    assert all(value > 0 for value in thresholds)


def test_sweep_thresholds_ignore_silent_rows_and_handle_an_empty_set():
    assert sweep_thresholds([], own=0.5) == []
    assert sweep_thresholds([0.0, 0.0], own=None) == []
    only_answers = sweep_thresholds([0.0, 0.4, 0.0, 0.8], own=None, points=2)
    assert only_answers == [0.4, 0.8]


def test_sweep_thresholds_do_not_repeat_values():
    thresholds = sweep_thresholds([0.5] * 20, own=0.5, points=6)
    assert thresholds == [0.5]
