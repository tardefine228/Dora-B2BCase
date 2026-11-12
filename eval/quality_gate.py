"""CI quality gate: the shipped configuration may not regress against a recorded run.

Why a recorded measurement instead of absolute floors
-----------------------------------------------------
CI used to compare the labeled-set metrics with numbers hard-coded in the workflow
("top-1 86.1 / top-3 92.2 / silence 81.5 ... minus one row"). Those numbers belonged
to the artifact shipped in September. The trainer then moved on and the artifact
could no longer be rebuilt, so the floors stopped describing anything at all: a full
rebuild measures 82.5 / 87.3 / 85.2 against floors 85 / 91 / 80 and the job has been
red ever since — for a reason nobody could see in the log.

A recorded measurement is the same idea without the drift. ``eval/baselines/*.json``
is written by ``eval/run_eval.py --report`` for one exact configuration (backend,
embedder, artifact, thresholds), carries the sha256 of the artifact it was measured
with, and is compared row by row:

* every metric may lose at most ``ROW_TOLERANCE`` row on 166/274 rows — one row is
  0.6 / 0.4 p.p., i.e. below the resolution of this set;
* ``ok_precision`` is compared by the number of *wrong* confident answers, because
  its denominator is chosen by the model itself;
* dangerous overconfidence ECE may not grow by more than ``ECE_TOLERANCE``;
  total and underconfidence ECE stay visible in the report. An old baseline has
  only total ECE, which is used as a conservative upper bound of its
  overconfidence component, so this split cannot admit more confident errors;
* new rows or a different denominator mean the dataset changed: that is not a
  regression, it is a different measurement, and it is rejected as such.

Refreshing a baseline is therefore an explicit, reviewable act: measure, explain the
delta in the pull request, copy the file. Nothing refreshes itself.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.dense_gate import ECE_TOLERANCE, ROW_TOLERANCE, Counts, GateRow, GateVerdict

MEASUREMENT_VERSION = 1

# Same keys as the dense gate, so a measurement can be compared by the same report.
METRIC_LABELS: dict[str, str] = {
    "correct_silence": "верный not_found",
    "exact_top1": "точные названия: код/уточнение",
    "top1_expected": "верное решение: код/уточнение",
    "top3_expected": "top-3 (с кодом)",
    "top1_all": "top-1 по всем",
    "ok_precision": "точность статуса ok",
}

# Metrics whose denominator is fixed by the labeled set itself.
FIXED_SET_METRICS = tuple(key for key in METRIC_LABELS if key != "ok_precision")


def measurement(
    counts: dict[str, Counts],
    ece: float,
    *,
    config: dict | None = None,
    overconfidence_ece: float | None = None,
    underconfidence_ece: float | None = None,
) -> dict:
    """Serialisable measurement: counts, ECE and the configuration they belong to."""
    missing = set(METRIC_LABELS) - set(counts)
    if missing:
        raise ValueError(f"Нет измерений для метрик: {sorted(missing)}")
    payload = {
        "version": MEASUREMENT_VERSION,
        "config": config or {},
        "metrics": {
            key: {"correct": counts[key].correct, "total": counts[key].total}
            for key in METRIC_LABELS
        },
        "ece": round(float(ece), 4),
    }
    if overconfidence_ece is not None:
        payload["overconfidence_ece"] = round(float(overconfidence_ece), 4)
    if underconfidence_ece is not None:
        payload["underconfidence_ece"] = round(float(underconfidence_ece), 4)
    return payload


def load_measurement(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = payload.get("version")
    if version != MEASUREMENT_VERSION:
        raise ValueError(
            f"{path}: ожидается version={MEASUREMENT_VERSION}, найдено {version!r}"
        )
    missing = set(METRIC_LABELS) - set(payload.get("metrics") or {})
    if missing or "ece" not in payload:
        raise ValueError(f"{path}: нет измерений для метрик {sorted(missing)} или ECE")
    return payload


def evaluate_regression(baseline: dict, actual: dict) -> GateVerdict:
    """Compare a fresh measurement with a recorded one, row by row."""
    base_counts = {
        key: Counts(row["correct"], row["total"])
        for key, row in baseline["metrics"].items()
        if key in METRIC_LABELS
    }
    new_counts = {
        key: Counts(row["correct"], row["total"])
        for key, row in actual["metrics"].items()
        if key in METRIC_LABELS
    }
    missing = set(METRIC_LABELS) - set(base_counts) | set(METRIC_LABELS) - set(new_counts)
    if missing:
        raise ValueError(f"Нет измерений для метрик: {sorted(missing)}")
    for key in FIXED_SET_METRICS:
        if base_counts[key].total != new_counts[key].total:
            raise ValueError(
                f"{METRIC_LABELS[key]}: разные наборы — "
                f"{base_counts[key].total} и {new_counts[key].total} строк"
            )

    rows: list[GateRow] = []
    reasons: list[str] = []

    for key, label in METRIC_LABELS.items():
        old, fresh = base_counts[key], new_counts[key]
        if key == "ok_precision":
            # Denominator is the model's own choice: compare wrong confident answers.
            extra_wrong = fresh.wrong - old.wrong
            if extra_wrong > ROW_TOLERANCE:
                verdict = "fatal"
                reasons.append(
                    f"{label}: неверных ok {old.wrong} → {fresh.wrong} "
                    f"(допуск +{ROW_TOLERANCE})"
                )
            else:
                verdict = "better" if extra_wrong < 0 else "equal"
        else:
            lost = old.correct - fresh.correct
            if lost > ROW_TOLERANCE:
                verdict = "fatal"
                reasons.append(
                    f"{label}: {fresh.correct} из {fresh.total} против "
                    f"{old.correct} в базовой линии (−{lost} строк, допуск {ROW_TOLERANCE})"
                )
            elif lost > 0:
                verdict = "tolerated"
            else:
                verdict = "better" if lost < 0 else "equal"
        rows.append(GateRow(key, label, old, fresh, verdict))

    base_total_ece = float(baseline["ece"])
    new_total_ece = float(actual["ece"])
    split_calibration = "overconfidence_ece" in actual
    # Old baselines carry only total ECE. It is a conservative upper bound for
    # their overconfidence component, so the new policy is never more permissive
    # toward dangerous confident errors than the old total-ECE threshold.
    base_ece = float(baseline.get("overconfidence_ece", base_total_ece))
    new_ece = float(actual.get("overconfidence_ece", new_total_ece))
    if new_ece > base_ece + ECE_TOLERANCE:
        label = "ECE переуверенности" if split_calibration else "ECE"
        reasons.append(
            f"{label} вырос {base_ece:.3f} → {new_ece:.3f} "
            f"(допуск +{ECE_TOLERANCE})"
        )
    ece_note = None
    if split_calibration:
        new_under = float(actual.get("underconfidence_ece", new_total_ece - new_ece))
        ece_note = (
            f"полный ECE {base_total_ece:.3f} → {new_total_ece:.3f}; "
            f"недоуверенность сейчас {new_under:.3f} "
            "(показана в отчёте, не маскирует переуверенность)"
        )

    return GateVerdict(
        keep=not reasons,
        rows=tuple(rows),
        reasons=tuple(reasons),
        base_ece=base_ece,
        new_ece=new_ece,
        labels=("В ПРЕДЕЛАХ", "РЕГРЕССИЯ"),
        columns=("база", "сейчас"),
        ece_label=(
            "ECE over (max)"
            if split_calibration and "overconfidence_ece" not in baseline
            else None
        ),
        ece_note=ece_note,
    )
