"""Pick the ``ok``/``review`` floors of a backend from one classification pass.

The decision status is made on the **raw** confidence (``confidence_raw``), so the
whole threshold grid can be evaluated without re-running the model: one pass over
the labeled set records the raw value of every row, and every candidate
``(ok, review)`` pair is then a cheap recount. This is what makes a two-dimensional
sweep practical — ``run_eval.py --sweep`` moves only ``review`` and re-classifies
274 rows for every point.

Why a separate tool: the floors are **per backend**. The artifacts ship the
lexical point (``ok`` 0.77 / ``review`` 0.165, эксп. 14), but the semantic backend
answers on a different part of the raw scale, so the same numbers are meaningless
for it (эксп. 15). The chosen point is applied by the API through
``QDRANT_*``/``CATALOG_*``-style environment overrides, not by retraining.

**The labeled set is a measurement set** — it is not used to fit the model. It is
used here the same way эксп. 14 used it to settle the ``ok`` floor: as the
yardstick on which a policy point is read off and then recorded. Read the printed
frontier, do not tune on it row by row.

Usage:
    ML_ENABLED=true EMBEDDER_BACKEND=sentence-transformers MODEL_NAME=BAAI/bge-m3 \
        QDRANT_STORAGE_DIR=qdrant_storage_eval PYTHONPATH=api \
        python eval/threshold_sweep.py --backend qdrant
    python eval/threshold_sweep.py --backend catalog --from-csv eval/threshold_sweep_catalog.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "api", ROOT / "eval"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import run_eval as labeled_eval  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.services.classifier import build_classifier  # noqa: E402

OK_GRID = (0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.77, 0.80, 0.85, 0.90)
REVIEW_GRID = (0.08, 0.10, 0.12, 0.14, 0.165, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35)
CSV_FIELDS = ("query", "expected", "raw", "top1_code", "top1_correct", "silence_expected")


def collect(backend: str) -> list[dict]:
    """One classification pass: keep the raw confidence of every row."""
    settings.classifier_backend = backend
    classifier = build_classifier(backend)
    rows = []
    for row in labeled_eval.load_labeled(labeled_eval.LABELED):
        query = (row["query"] or "").strip()
        expected = (row["expected_okpd2"] or "").strip()
        response = classifier.classify(query)
        # The top-1 code is independent of the thresholds: a lower `review`
        # floor only changes the *status*, never which position leads.
        correct = (
            labeled_eval.codes_match(response.code, expected)
            if expected
            else response.status == "not_found"
        )
        rows.append(
            {
                "query": query,
                "expected": expected or "(not_found)",
                "raw": round(float(response.confidence_raw or 0.0), 4),
                "top1_code": response.code or "(not_found)",
                "top1_correct": int(bool(correct)),
                "silence_expected": int(not expected),
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    for row in rows:
        row["raw"] = float(row["raw"])
        row["top1_correct"] = int(row["top1_correct"])
        row["silence_expected"] = int(row["silence_expected"])
    return rows


def evaluate_point(rows: list[dict], ok: float, review: float) -> dict:
    """Metrics for one ``(ok, review)`` pair, recomputed from the recorded raws."""
    with_code = [r for r in rows if not r["silence_expected"]]
    silence = [r for r in rows if r["silence_expected"]]

    def status(raw: float) -> str:
        if raw >= ok:
            return "ok"
        if raw >= review:
            return "review"
        return "not_found"

    answered = [r for r in with_code if status(r["raw"]) != "not_found"]
    top1 = sum(r["top1_correct"] for r in answered)
    silence_answered = [r for r in silence if status(r["raw"]) != "not_found"]
    # Promising a position for a product that is not in the list at all is the
    # expensive mistake the customer sees; an honest `not_found` never is.
    silence_precision = (
        100 * (len(silence) - len(silence_answered)) / len(silence) if silence else 0.0
    )
    return {
        "ok": ok,
        "review": review,
        "coverage": round(100 * len(answered) / len(with_code), 1) if with_code else 0.0,
        "top1_answered": round(100 * top1 / len(answered), 1) if answered else 0.0,
        "not_found_precision": round(silence_precision, 1),
        "correct_hints": top1,
        "false_hints": len(silence_answered),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="qdrant", choices=["catalog", "qdrant"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="пересчитать сетку по сохранённым сырым уверенностям (без прогона модели)",
    )
    parser.add_argument(
        "--policy",
        type=float,
        default=90.0,
        help="минимальная точность `not_found` (в процентах), под неё берётся лучший top-1",
    )
    args = parser.parse_args()

    output = args.output or ROOT / "eval" / f"threshold_sweep_{args.backend}.csv"
    if args.from_csv:
        rows = read_csv(args.from_csv)
    else:
        rows = collect(args.backend)
        write_csv(rows, output)
        print(f"Сырые уверенности сохранены: {output}")

    with_code = sum(1 for r in rows if not r["silence_expected"])
    silence = len(rows) - with_code
    print(f"бэкенд={args.backend} строк={len(rows)} (с кодом {with_code}, вне перечня {silence})")

    points = [
        evaluate_point(rows, ok, review) for ok in OK_GRID for review in REVIEW_GRID if review < ok
    ]
    print()
    print(
        "| ok | review | покрытие | top-1 среди отвеченных | точность not_found "
        "| верных подсказок | ложных |"
    )
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for p in points:
        print(
            f"| {p['ok']:.3f} | {p['review']:.3f} | {p['coverage']}% | {p['top1_answered']}% | "
            f"{p['not_found_precision']}% | {p['correct_hints']} | {p['false_hints']} |"
        )

    eligible = [p for p in points if p["not_found_precision"] >= args.policy]
    if eligible:
        best = max(eligible, key=lambda p: (p["top1_answered"], p["coverage"]))
        print()
        print(
            f"Политика 'точность not_found >= {args.policy}%' -> лучшее top-1 среди отвеченных: "
            f"ok={best['ok']:.3f}, review={best['review']:.3f} "
            f"(top-1 {best['top1_answered']}%, покрытие {best['coverage']}%, "
            f"точность not_found {best['not_found_precision']}%, "
            f"ложных подсказок {best['false_hints']})"
        )
    else:
        print(f"\nНи одна точка не даёт точность not_found >= {args.policy}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
