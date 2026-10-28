"""The CI gate may not trade a regression for a green build.

The numbers here are the ones measured on 2026-10-24 (274 labeled rows, 201 with a
code): the vector artifact beats the lexical one by 2 rows of top-1 and 4 of top-3
without losing a single row of correct silence. The gate must accept exactly that
delta and reject anything larger.
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
    "correct_silence": (30, 32),
    "exact_top1": (112, 112),
    "top1_expected": (189, 201),
    "top3_expected": (195, 201),
    "top1_all": (219, 233),
    "ok_precision": (174, 178),
}
LEXICAL_ECE = 0.0131


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
    """On 201 rows one row is 0.5 p.p. — below what this set can resolve."""
    tolerated = evaluate_regression(
        _record(LEXICAL, LEXICAL_ECE), _with(top1_expected=(188, 201))
    )
    assert tolerated.keep is True
    assert next(r for r in tolerated.rows if r.key == "top1_expected").verdict == "tolerated"

    # Two rows lost: one more than the tolerance, i.e. 189 → 187.
    fatal = evaluate_regression(
        _record(LEXICAL, LEXICAL_ECE), _with(top1_expected=(189 - ROW_TOLERANCE - 1, 201))
    )
    assert fatal.keep is False
    assert any("top-1 (с кодом)" in reason and "−2 строк" in reason for reason in fatal.reasons)


def test_lost_silence_is_reported_with_its_own_rows():
    verdict = evaluate_regression(_record(LEXICAL, LEXICAL_ECE), _with(correct_silence=(27, 32)))
    assert verdict.keep is False
    assert any("верный not_found" in reason and "−3" in reason for reason in verdict.reasons)


def test_all_failures_are_reported_together():
    verdict = evaluate_regression(
        _record(LEXICAL, LEXICAL_ECE),
        _record({**LEXICAL, "correct_silence": (20, 32), "top3_expected": (185, 201)}, 0.2),
    )
    assert len(verdict.reasons) == 3  # silence + top-3 + ECE
    assert any("ECE" in reason for reason in verdict.reasons)


def test_ok_precision_is_compared_by_wrong_answers_not_by_correct_ones():
    """Its denominator is the model's own choice, so Δ correct is meaningless.

    174/178 → 176/181 is two more correct answers *and* one more wrong one.
    """
    same = evaluate_regression(_record(LEXICAL, LEXICAL_ECE), _with(ok_precision=(176, 181)))
    assert same.keep is True
    row = next(r for r in same.rows if r.key == "ok_precision")
    assert row.delta == 2 and row.delta_wrong == 1 and row.verdict == "equal"

    worse = evaluate_regression(_record(LEXICAL, LEXICAL_ECE), _with(ok_precision=(160, 180)))
    assert worse.keep is False
    assert any("неверных ok 4 → 20" in reason for reason in worse.reasons)


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
    bigger["top1_all"] = (220, 280)
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
    assert record["metrics"]["top1_expected"] == {"correct": 189, "total": 201}
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
    rejected = evaluate_regression(_record(LEXICAL, LEXICAL_ECE), _with(top1_expected=(180, 201)))
    table = rejected.table()
    assert "РЕГРЕССИЯ" in table and "180/201" in table and "189/201" in table


# --- shipped baselines ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "reranker", "top1", "ece"),
    (
        # После аудита разметки (docs/labeling-doctrine.md): 201 строка с кодом
        # вместо 166, поэтому числа и ECE другие — это другое измерение, не регрессия.
        # эксп. 35 (волна 6 лексикона: приборы, сетевое оборудование, рукава,
        # системный блок, фильтр): лексика 191/201, top-3 198/201, ECE 0.019.
        # эксп. 36 (волна 7: ИБП, датчик температуры, термостат): top-1 и top-3
        # те же, «ok» точнее — 183/190 против 175/181. ECE 0.0192 → 0.0124 не
        # потому, что модель стала лучше, а потому, что показная шкала больше не
        # перекалибровывается на шум площадки (identity, см. CALIBRATION_SHIFT_SIGMAS).
        # эксп. 37 (2026-10-25): векторный артефакт пересобран тем же тренером и тем же
        # правилом калибровки — 193/201 top-1 против 192, ECE 0.031 против 0.029.
        # База — это замер отгруженной конфигурации, поэтому обновляется вместе с ней.
        ("lexical.json", "reranker.json", (191, 201), 0.0124),
        ("dense-tfidf.json", "reranker.tfidf.json", (193, 201), 0.0311),
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
    """Same dataset, so the comparison must work and show what vectors buy.

    On the shipped run the vector artifact wins where it was trained to win
    (+1 top-1, +1 top-3) and its *calibrated* confidence lands inside the 0.01
    tolerance (0.029 against 0.037), so this cross-check is green — it was red
    for one release (эксп. 30: 0.027 against 0.0131). The tolerance itself is not
    proof: the per-run ECE of one artifact spreads over 0.013–0.045 depending on
    the training split (``ece_per_run`` in ``ml/artifacts/dense-gate.json``),
    which is why admission compares the reliability table pooled over four runs.
    The two files are records of their own runs: ``dense-tfidf.json`` was measured
    on 2026-10-24, before the fifth lexicon wave and the 0.80 ``ok`` floor, and the
    vector file has not been retrained since, so the delta is informative rather
    than a head-to-head of today's code. It is also the one comparison the labeled
    set resolves with a single row: +1 top-1 is exactly one query.
    """
    lexical = load_measurement(BASELINES / "lexical.json")
    dense = load_measurement(BASELINES / "dense-tfidf.json")
    verdict = evaluate_regression(lexical, dense)
    assert verdict.headline_gains["top1_expected"] == 2
    assert verdict.headline_gains["top3_expected"] == 1
    # Every row metric is equal or better: the vector path buys rows without
    # giving up silence, literal names or confident-answer precision.
    assert [row.verdict for row in verdict.rows if row.verdict not in {"equal", "better"}] == []
    # … and the only line that does not compare is the ECE. After эксп. 37 both
    # artifacts are calibrated by the same rule (identity unless the platform's
    # shift clears its own noise), so this is no longer "old map against new
    # rule" — it is the resolution of the set: one build's ECE is quantised over
    # ~200 answered rows, and the difference +0.019 has a paired-bootstrap
    # interval of −0.005…+0.025 on *these* rows (eval/calibration_noise.md).
    # Admission therefore compares reliability tables pooled over four training
    # runs (0.0232 against 0.0242 in ml/artifacts/dense-gate.json), not two
    # single-run ECEs; this cross-check deliberately keeps the strict 0.01
    # tolerance so the unresolved number stays visible instead of being rounded
    # into a pass.
    assert verdict.keep is False
    assert len(verdict.reasons) == 1 and "ECE" in verdict.reasons[0]
