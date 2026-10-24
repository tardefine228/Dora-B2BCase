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

from eval.dense_gate import (
    ECE_TOLERANCE,
    ROW_TOLERANCE,
    Counts,
    evaluate_gate,
    pool_counts,
)

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


# --- pooling several training runs ----------------------------------------------
#
# A single run measures one split of the training data. With identical code the
# lexical artifact answered 93 / 88 / 93 / 92 of the 108 out-of-catalog rows at
# seeds 1875-1878 and the vector one 92 / 91 / 91 / 93, so the zero-tolerance
# silence guard was deciding on a coin flip: the same autocorrection fix was
# rejected at seed 1875 and accepted at 1876. Pooling keeps the policy (silence
# is a guard) and makes the measurement describe the artifact, not the split.


def test_pooling_sums_counts_and_keeps_totals() -> None:
    first = dict(LEXICAL)
    second = _copy(LEXICAL, correct_silence=Counts(88, 108), top1_expected=Counts(139, 166))
    pooled = pool_counts([first, second])
    assert pooled["correct_silence"] == Counts(180, 216)
    assert pooled["top1_expected"] == Counts(276, 332)


def test_pooling_rejects_empty_and_mismatched_runs() -> None:
    with pytest.raises(ValueError):
        pool_counts([])
    with pytest.raises(ValueError):
        pool_counts([LEXICAL, {"top1_expected": Counts(1, 1)}])


def test_one_lost_silence_row_is_noise_when_pooled_but_fatal_alone() -> None:
    """The exact case that blocked the autocorrection fix at a single seed."""
    lost = _copy(VECTOR, correct_silence=Counts(91, 108))
    alone = evaluate_gate(LEXICAL, lost, base_ece=LEXICAL_ECE, new_ece=VECTOR_ECE)
    assert alone.keep is False
    assert any("верный not_found" in reason for reason in alone.reasons)

    # Over two runs the same artifact loses one row in one run and gains it back
    # in the other: silence is equal, so the guard passes while top-1 still wins.
    base = pool_counts([LEXICAL, LEXICAL])
    new = pool_counts([lost, _copy(VECTOR, correct_silence=Counts(93, 108))])
    pooled = evaluate_gate(base, new, base_ece=LEXICAL_ECE, new_ece=VECTOR_ECE, runs=2)
    assert pooled.keep is True
    assert pooled.runs == 2
    assert pooled.as_dict()["runs"] == 2


def test_pooled_systematic_silence_loss_is_still_fatal() -> None:
    """Pooling must not launder a real regression: losing a row in every run fails."""
    lost = _copy(VECTOR, correct_silence=Counts(91, 108))
    base = pool_counts([LEXICAL, LEXICAL, LEXICAL])
    new = pool_counts([lost, lost, lost])
    verdict = evaluate_gate(base, new, base_ece=LEXICAL_ECE, new_ece=VECTOR_ECE, runs=3)
    assert verdict.keep is False
    assert any("верный not_found" in reason for reason in verdict.reasons)


def test_pooling_scales_the_headline_tolerance_with_the_runs() -> None:
    base = pool_counts([LEXICAL, LEXICAL])
    three_rows_worse = _copy(LEXICAL, top1_expected=Counts(134, 166))
    new = pool_counts([three_rows_worse, three_rows_worse])
    # −6 rows over two runs exceeds the −2 tolerance that two runs allow.
    verdict = evaluate_gate(base, new, base_ece=LEXICAL_ECE, new_ece=VECTOR_ECE, runs=2)
    assert verdict.keep is False
    assert any("top-1" in reason for reason in verdict.reasons)


# --- pooled calibration ------------------------------------------------------


def test_reliability_table_reproduces_the_scalar_ece() -> None:
    """The table is the same measurement as ``eval.run_eval.calibration``."""
    from eval.dense_gate import ece_from_bins, reliability_bins

    pairs = [(0.95, True)] * 40 + [(0.95, False)] * 2 + [(0.55, True)] * 8 + [(0.55, False)] * 3
    bins = reliability_bins(pairs)
    assert sum(row.count for row in bins) == len(pairs)
    expected = (42 / 53) * abs(40 / 42 - 0.95) + (11 / 53) * abs(8 / 11 - 0.55)
    assert ece_from_bins(bins) == pytest.approx(expected)
    # 1.0 must land in the last bin, not in an out-of-range one.
    assert reliability_bins([(1.0, True)])[-1].count == 1


def test_pooling_tables_removes_the_bias_between_runs() -> None:
    """Pooling is why the gate does not average four ``abs()`` values.

    Two runs disagree in opposite directions inside the same bin: each looks
    0.1 miscalibrated on its own, while the pooled table is exactly calibrated.
    Averaging the two ECEs would report 0.1 where there is no bias left.
    """
    from eval.dense_gate import ece_from_bins, pool_bins, reliability_bins

    first = reliability_bins([(0.9, True)] * 10)
    second = reliability_bins([(0.9, True)] * 8 + [(0.9, False)] * 2)
    assert ece_from_bins(first) == pytest.approx(0.1)
    assert ece_from_bins(second) == pytest.approx(0.1)
    assert ece_from_bins(pool_bins([first, second])) == pytest.approx(0.0)
    # Pooling a single table with itself must not change the measurement.
    assert ece_from_bins(pool_bins([first, first])) == pytest.approx(ece_from_bins(first))


def test_pooling_rejects_tables_of_different_shape() -> None:
    from eval.dense_gate import pool_bins, reliability_bins

    with pytest.raises(ValueError):
        pool_bins([reliability_bins([(0.9, True)]), reliability_bins([(0.9, True)], bins=4)])
