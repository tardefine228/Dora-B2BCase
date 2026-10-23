"""The dense-artifact gate must be conservative, countable and explainable.

Pinned here is the case that motivated the rewrite: on the 274-row labeled set a
vector-feature artifact improves top-1 (137/166 → 140/166), top-3 (145 → 149),
overall top-1 (229 → 232) and ECE (0.053 → 0.036) while losing nothing on
silence and literal names. The old rule compared *rounded rates* and rejected it,
because ``ok_precision`` moved by one row (131/143 → 133/146); the artifact was
deleted and the vector signal never reached a decision in production.
"""

from __future__ import annotations

import pytest

from eval.dense_gate import ECE_TOLERANCE, ROW_TOLERANCE, Counts, evaluate_gate

# --- the measured case (tfidf embedder, seed 1875, 2026-10-23) -------------------------

LEXICAL = {
    "top1_expected": Counts(137, 166),
    "top3_expected": Counts(145, 166),
    "top1_all": Counts(229, 274),
    "correct_silence": Counts(92, 108),
    "exact_top1": Counts(112, 112),
    "ok_precision": Counts(131, 143),
}
VECTOR = {
    "top1_expected": Counts(140, 166),
    "top3_expected": Counts(149, 166),
    "top1_all": Counts(232, 274),
    "correct_silence": Counts(92, 108),
    "exact_top1": Counts(112, 112),
    "ok_precision": Counts(133, 146),
}
LEXICAL_ECE, VECTOR_ECE = 0.0534, 0.0356


def _copy(counts: dict[str, Counts], **overrides: Counts) -> dict[str, Counts]:
    merged = dict(counts)
    merged.update(overrides)
    return merged


def _verdict(base=None, new=None, *, base_ece=LEXICAL_ECE, new_ece=VECTOR_ECE):
    return evaluate_gate(
        LEXICAL if base is None else base,
        VECTOR if new is None else new,
        base_ece=base_ece,
        new_ece=new_ece,
    )


def test_measured_vector_artifact_is_accepted():
    verdict = _verdict()
    assert verdict.keep, verdict.reasons
    assert verdict.reasons == ()
    assert verdict.headline_gains == {
        "top1_expected": 3,
        "top3_expected": 4,
        "top1_all": 3,
    }


def test_one_extra_wrong_confident_answer_is_tolerated_as_noise():
    # 143 ok answers with 12 wrong -> 146 with 13 wrong: the rate drops, the harm does not.
    row = next(r for r in _verdict().rows if r.key == "ok_precision")
    assert row.verdict == "tolerated"
    assert row.base.rate > row.new.rate  # the rate really did fall


def test_silence_is_a_hard_guard():
    verdict = _verdict(new=_copy(VECTOR, correct_silence=Counts(91, 108)))
    assert not verdict.keep
    assert any("верный not_found" in reason for reason in verdict.reasons)


def test_exact_catalog_names_are_a_hard_guard():
    verdict = _verdict(new=_copy(VECTOR, exact_top1=Counts(111, 112)))
    assert not verdict.keep
    assert any("точные названия" in reason for reason in verdict.reasons)


def test_two_more_wrong_confident_answers_are_rejected():
    verdict = _verdict(new=_copy(VECTOR, ok_precision=Counts(131, 146)))
    assert not verdict.keep
    assert any("неверных ok 12 → 15" in reason for reason in verdict.reasons)


def test_ok_precision_compares_wrong_counts_not_rates():
    """Equal rates are not equal harm: 1 wrong in 5 vs 10 wrong in 50."""
    base = _copy(LEXICAL, ok_precision=Counts(4, 5))
    new = _copy(VECTOR, ok_precision=Counts(40, 50))
    verdict = _verdict(base=base, new=new)
    assert not verdict.keep
    assert any("неверных ok 1 → 10" in reason for reason in verdict.reasons)


def test_identical_measurements_are_rejected_as_pointless():
    verdict = _verdict(new=LEXICAL, new_ece=LEXICAL_ECE)
    assert not verdict.keep
    assert any("ни одна из ключевых метрик не улучшилась" in r for r in verdict.reasons)


def test_one_row_loss_in_a_headline_metric_is_tolerated_when_others_improve():
    verdict = _verdict(new=_copy(VECTOR, top3_expected=Counts(144, 166)))
    assert verdict.keep, verdict.reasons
    row = next(r for r in verdict.rows if r.key == "top3_expected")
    assert row.delta == -1
    assert row.verdict == "tolerated"


def test_headline_loss_beyond_tolerance_is_rejected():
    verdict = _verdict(
        new=_copy(VECTOR, top1_expected=Counts(137 - ROW_TOLERANCE - 1, 166))
    )
    assert not verdict.keep
    assert any("top-1 (с кодом)" in reason for reason in verdict.reasons)


def test_calibration_may_not_grow_beyond_the_tolerance():
    assert _verdict(new_ece=LEXICAL_ECE + ECE_TOLERANCE).keep
    verdict = _verdict(new_ece=LEXICAL_ECE + ECE_TOLERANCE + 0.001)
    assert not verdict.keep
    assert any("ECE вырос" in reason for reason in verdict.reasons)


def test_all_failures_are_reported_together():
    verdict = _verdict(
        new=_copy(
            VECTOR,
            correct_silence=Counts(90, 108),
            exact_top1=Counts(110, 112),
            top1_expected=Counts(130, 166),
            ok_precision=Counts(120, 146),
        ),
        new_ece=0.20,
    )
    assert not verdict.keep
    assert len(verdict.reasons) >= 4


def test_mismatched_measurement_sets_are_refused():
    with pytest.raises(ValueError, match="разных наборах"):
        _verdict(new=_copy(VECTOR, top1_all=Counts(232, 270)))


def test_missing_metrics_are_refused():
    incomplete = dict(LEXICAL)
    incomplete.pop("ok_precision")
    with pytest.raises(ValueError, match="Нет измерений"):
        evaluate_gate(incomplete, VECTOR, base_ece=0.05, new_ece=0.03)


def test_counts_reject_impossible_values():
    with pytest.raises(ValueError, match="Некорректные счётчики"):
        Counts(5, 4)
    with pytest.raises(ValueError, match="Некорректные счётчики"):
        Counts(-1, 4)


def test_report_explains_every_metric_and_the_verdict():
    table = _verdict().table()
    for label in ("top-1 (с кодом)", "top-3 (с кодом)", "верный not_found", "ECE", "ПРИНЯТ"):
        assert label in table

    rejection = _verdict(new=_copy(VECTOR, correct_silence=Counts(80, 108))).table()
    assert "ОТКЛОНЁН" in rejection
    assert "верный not_found" in rejection


def test_rate_row_is_reported_by_wrong_answers_not_by_correct_ones():
    """``ok_precision`` counts answers the model chose, so its raw Δ correct is a trap.

    In the measured case the vector artifact answers 3 more rows confidently and is
    right on 2 more of them: correct goes 131 → 133 (+2), wrong goes 12 → 13 (+1).
    A row that renders "-16" while the verdict says "tolerated" would be unreadable.
    """
    verdict = _verdict()
    rate = next(row for row in verdict.rows if row.key == "ok_precision")
    assert rate.delta == 2 and rate.delta_wrong == 1

    table = verdict.table()
    assert "12 неверн./143" in table and "13 неверн./146" in table
    assert "+1 неверн." in table


def test_report_is_json_serialisable_for_the_build_log():
    payload = _verdict().as_dict()
    assert payload["keep"] is True
    assert payload["ece"] == {"base": 0.0534, "new": 0.0356}
    row = next(r for r in payload["rows"] if r["metric"] == "top1_expected")
    assert row["base"] == {"correct": 137, "total": 166, "rate": 82.5}
    assert row["new"] == {"correct": 140, "total": 166, "rate": 84.3}
    assert row["delta_correct"] == 3
