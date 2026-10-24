"""Comparing artifacts at equal silence, not at equal threshold.

The numbers are the shipped ones: the lexical artifact answers 163 of 274 rows, the
vector artifact 166, and the retired September file 189 — which is the whole reason
its headline top-1 looked better. The tool has to reproduce that reading.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

from pareto import (  # noqa: E402
    Point,
    _quantile_thresholds,
    best_at_silence,
    render_detail,
    render_table,
    throughput_gain,
)

from eval.dense_gate import Counts  # noqa: E402


def _point(threshold: float, silence: int, top1: int, *, ece: float = 0.05) -> Point:
    return Point(
        threshold=threshold,
        silence=Counts(silence, 108),
        top1=Counts(top1, 166),
        top3=Counts(min(166, top1 + 5), 166),
        top1_all=Counts(silence + top1, 274),
        ece=ece,
        answered=274 - 108 + silence,
    )


QUIET = _point(0.70, 92, 137)  # the lexical artifact at its own threshold
LOUD = _point(0.13, 80, 145)  # the retired file at its own threshold
MIDDLE = _point(0.45, 86, 141)


def test_the_most_talkative_point_that_keeps_the_silence_budget_wins():
    """At 80 refusals the loud configuration is legitimate; at 85 it is not."""
    at_80 = best_at_silence([QUIET, LOUD, MIDDLE], 80)
    assert at_80 is LOUD
    assert best_at_silence([QUIET, LOUD, MIDDLE], 85) is MIDDLE
    assert best_at_silence([QUIET, LOUD, MIDDLE], 92) is QUIET


def test_a_budget_nobody_meets_returns_none_instead_of_the_best_effort():
    assert best_at_silence([QUIET, LOUD], 100) is None


def test_ties_go_to_the_higher_overall_accuracy_then_to_the_lower_threshold():
    a = _point(0.50, 90, 140)
    b = _point(0.40, 90, 140)
    assert best_at_silence([a, b], 85) is b  # same top-1, b answers more


def test_gain_is_measured_in_rows_at_the_same_budget():
    """The vector artifact wins 3 rows over the lexical one and 5 over the retired file."""
    assert throughput_gain(QUIET, MIDDLE) == 4
    quiet_retired = _point(0.247, 92, 135)
    assert throughput_gain(QUIET, quiet_retired) == -2


def test_table_shows_every_artifact_and_marks_missing_points():
    table = render_table({"лексика": [QUIET], "вектор": [MIDDLE]}, (80, 85, 95))
    assert "лексика" in table and "вектор" in table
    assert "137/166" in table  # the lexical point is the only one that meets 95
    assert "—" in table  # the vector series has nothing at 95


def test_detail_prints_the_own_operating_point():
    detail = render_detail(
        {"лексика": [QUIET], "старый": [LOUD]}, {"лексика": 0.70, "старый": 0.13}
    )
    assert "74.1%" in detail  # the retired file's silence, the reason it was retired
    assert "85.2%" in detail
    assert "0.102" not in detail  # ECE comes from the point, not from thin air


def test_thresholds_are_taken_from_the_artifact_own_confidence_distribution():
    confidences = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # 10 values, index = round(fraction * 9): 0 → 0.1, 0.5 → 0.5, 1.0 → 1.0.
    thresholds = _quantile_thresholds(confidences, (0.0, 0.5, 1.0))
    assert thresholds == [0.1, 0.5, 1.0]
    assert _quantile_thresholds([], (0.5,)) == []


def test_silence_rate_is_a_rate_of_the_silence_rows_only():
    assert QUIET.silence_rate == pytest.approx(85.2, abs=0.1)
    assert QUIET.top1_rate == pytest.approx(82.5, abs=0.1)
