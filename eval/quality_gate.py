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
* ECE may not grow by more than ``ECE_TOLERANCE``;
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
    "exact_top1": "точные названия, top-1",
    "top1_expected": "top-1 (с кодом)",
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
) -> dict:
    """Serialisable measurement: counts, ECE and the configuration they belong to."""
    missing = set(METRIC_LABELS) - set(counts)
    if missing:
        raise ValueError(f"Нет измерений для метрик: {sorted(missing)}")
    return {
        "version": MEASUREMENT_VERSION,
        "config": config or {},
        "metrics": {
            key: {"correct": counts[key].correct, "total": counts[key].total}
            for key in METRIC_LABELS
        },
        "ece": round(float(ece), 4),
    }


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

    base_ece = float(baseline["ece"])
    new_ece = float(actual["ece"])
    if new_ece > base_ece + ECE_TOLERANCE:
        reasons.append(f"ECE вырос {base_ece:.3f} → {new_ece:.3f} (допуск +{ECE_TOLERANCE})")

    return GateVerdict(
        keep=not reasons,
        rows=tuple(rows),
        reasons=tuple(reasons),
        base_ece=base_ece,
        new_ece=new_ece,
        labels=("В ПРЕДЕЛАХ", "РЕГРЕССИЯ"),
        columns=("база", "сейчас"),
    )
