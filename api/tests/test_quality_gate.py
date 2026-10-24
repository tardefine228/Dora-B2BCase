"""The CI gate may not trade a regression for a green build.

The numbers here are the ones measured on 2026-10-23 (274 labeled rows): the vector
artifact beats the lexical one by 3 rows of top-1 without losing a single row of
correct silence. The gate must accept exactly that delta and reject anything larger.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.dense_gate import ECE_TOLERANCE, ROW_TOLERANCE, Counts
from eval.quality_gate import (
    MEASUREMENT_VERSION,
    evaluate_regression,
    load_measurement,
    measurement,
)

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "eval" / "baselines"

# Shipped lexical run (eval/baselines/lexical.json).
LEXICAL = {
    "correct_silence": (92, 108),
    "exact_top1": (112, 112),
    "top1_expected": (137, 166),
    "top3_expected": (145, 166),
    "top1_all": (229, 274),
    "ok_precision": (131, 143),
}
LEXICAL_ECE = 0.0534


def _record(counts: dict[str, tuple[int, int]], ece: float) -> dict:
    return measurement(
        {key: Counts(correct, total) for key, (correct, total) in counts.items()},
        ece,
        config={"test": True},
    )


def _with(**overrides: tuple[int, int]) -> dict:
    counts = dict(LEXICAL)
    counts.update(overrides)
    return _record(counts, LEXICAL_ECE)


# --- policy ---------------------------------------------------------------------------


def test_identical_measurement_is_accepted_as_equal():
    verdict = evaluate_regression(_record(LEXICAL, LEXICAL_ECE), _record(LEXICAL, LEXICAL_ECE))
    assert verdict.keep is True and verdict.reasons == ()
    assert {row.verdict for row in verdict.rows} == {"equal"}


def test_one_lost_row_is_noise_and_two_are_a_regression():
    """On 166 rows one row is 0.6 p.p. — below what this set can resolve."""
    tolerated = evaluate_regression(
        _record(LEXICAL, LEXICAL_ECE), _with(top1_expected=(136, 166))
    )
    assert tolerated.keep is True
    assert next(r for r in tolerated.rows if r.key == "top1_expected").verdict == "tolerated"

    # Two rows lost: one more than the tolerance, i.e. 137 → 135.
    fatal = evaluate_regression(
        _record(LEXICAL, LEXICAL_ECE), _with(top1_expected=(137 - ROW_TOLERANCE - 1, 166))
    )
    assert fatal.keep is False
    assert any("top-1 (с кодом)" in reason and "−2 строк" in reason for reason in fatal.reasons)


def test_lost_silence_is_reported_with_its_own_rows():
    verdict = evaluate_regression(_record(LEXICAL, LEXICAL_ECE), _with(correct_silence=(89, 108)))
    assert verdict.keep is False
    assert any("верный not_found" in reason and "−3" in reason for reason in verdict.reasons)


def test_all_failures_are_reported_together():
    verdict = evaluate_regression(
        _record(LEXICAL, LEXICAL_ECE),
        _record({**LEXICAL, "correct_silence": (80, 108), "top3_expected": (140, 166)}, 0.2),
    )
    assert len(verdict.reasons) == 3  # silence + top-3 + ECE
    assert any("ECE" in reason for reason in verdict.reasons)


def test_ok_precision_is_compared_by_wrong_answers_not_by_correct_ones():
    """Its denominator is the model's own choice, so Δ correct is meaningless.

    131/143 → 133/146 is two more correct answers *and* one more wrong one.
    """
    same = evaluate_regression(_record(LEXICAL, LEXICAL_ECE), _with(ok_precision=(133, 146)))
    assert same.keep is True
    row = next(r for r in same.rows if r.key == "ok_precision")
    assert row.delta == 2 and row.delta_wrong == 1 and row.verdict == "equal"

    worse = evaluate_regression(_record(LEXICAL, LEXICAL_ECE), _with(ok_precision=(140, 160)))
    assert worse.keep is False
    assert any("неверных ok 12 → 20" in reason for reason in worse.reasons)


def test_calibration_tolerance_is_exclusive():
    ok = evaluate_regression(
        _record(LEXICAL, LEXICAL_ECE), _record(LEXICAL, LEXICAL_ECE + ECE_TOLERANCE)
    )
    assert ok.keep is True
    fatal = evaluate_regression(
        _record(LEXICAL, LEXICAL_ECE), _record(LEXICAL, LEXICAL_ECE + ECE_TOLERANCE + 0.001)
    )
    assert fatal.keep is False


def test_different_dataset_is_an_error_not_a_regression():
    """Adding labeled rows changes the measurement; silently passing would hide it."""
    bigger = dict(LEXICAL)
    bigger["top1_all"] = (230, 280)
    with pytest.raises(ValueError, match="разные наборы"):
        evaluate_regression(_record(LEXICAL, LEXICAL_ECE), _record(bigger, LEXICAL_ECE))


def test_missing_metric_is_an_error():
    incomplete = dict(LEXICAL)
    incomplete.pop("ok_precision")
    with pytest.raises(ValueError, match="Нет измерений"):
        evaluate_regression(_record(LEXICAL, LEXICAL_ECE), _record(incomplete, LEXICAL_ECE))


# --- report format --------------------------------------------------------------------


def test_measurement_round_trips_through_json(tmp_path):
    record = _record(LEXICAL, LEXICAL_ECE)
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    assert load_measurement(path) == record
    assert record["version"] == MEASUREMENT_VERSION
    assert record["metrics"]["top1_expected"] == {"correct": 137, "total": 166}
    assert record["ece"] == LEXICAL_ECE


def test_measurement_rejects_unknown_version_and_missing_numbers(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"version": 99, "metrics": {}, "ece": 0.1}), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        load_measurement(broken)

    partial = _record(LEXICAL, LEXICAL_ECE)
    partial["metrics"].pop("exact_top1")
    no_exact = tmp_path / "partial.json"
    no_exact.write_text(json.dumps(partial), encoding="utf-8")
    with pytest.raises(ValueError, match="exact_top1"):
        load_measurement(no_exact)


def test_table_states_the_verdict_and_the_numbers():
    accepted = evaluate_regression(_record(LEXICAL, LEXICAL_ECE), _record(LEXICAL, LEXICAL_ECE))
    assert "В ПРЕДЕЛАХ" in accepted.table()
    rejected = evaluate_regression(_record(LEXICAL, LEXICAL_ECE), _with(top1_expected=(130, 166)))
    table = rejected.table()
    assert "РЕГРЕССИЯ" in table and "130/166" in table and "137/166" in table


# --- shipped baselines ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "reranker", "top1", "ece"),
    (
        ("lexical.json", "reranker.json", (137, 166), 0.0558),
        ("dense-tfidf.json", "reranker.tfidf.json", (140, 166), 0.0311),
    ),
)
def test_shipped_baselines_describe_the_shipped_configuration(name, reranker, top1, ece):
    """A baseline is only meaningful together with the artifact it was measured on."""
    baseline = load_measurement(BASELINES / name)
    config = baseline["config"]
    assert config["backend"] == "catalog"
    assert config["reranker"] == reranker
    assert config["embedder_backend"] == "tfidf"
    assert config["vector_search"] is (reranker == "reranker.tfidf.json")
    assert config["labeled_rows"] == 274
    assert baseline["metrics"]["top1_expected"]["correct"] == top1[0]
    assert baseline["metrics"]["top1_expected"]["total"] == top1[1]
    assert baseline["ece"] == ece


def test_baseline_artifact_hash_matches_the_shipped_file():
    import hashlib

    from ml import config as ml_config

    baseline = load_measurement(BASELINES / "dense-tfidf.json")
    shipped = ml_config.ARTIFACTS_DIR / "reranker.tfidf.json"
    expected = hashlib.sha256(shipped.read_bytes()).hexdigest()
    assert baseline["config"]["artifact_sha256"]["reranker.tfidf.json"] == expected


def test_baselines_can_be_compared_with_each_other_row_by_row():
    """Same dataset, so the comparison must work — and the vector run must win."""
    lexical = load_measurement(BASELINES / "lexical.json")
    dense = load_measurement(BASELINES / "dense-tfidf.json")
    verdict = evaluate_regression(lexical, dense)
    assert verdict.keep is True
    assert verdict.headline_gains["top1_expected"] == 3
