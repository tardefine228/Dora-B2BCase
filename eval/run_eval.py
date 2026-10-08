"""Evaluate the classifier on the hand-labeled set eval/labeled_examples.csv.

The labeled set is a *measurement* set: neither the reranker nor the thresholds
are trained on it (see ``ml/train_reranker.py``). Match rules:

* an empty ``expected_okpd2`` means the expected answer is ``not_found``;
* otherwise the predicted code may match exactly or as a prefix of the expected
  code (or vice versa) — "26.40.20" covers all television positions;
* ``top-3`` counts the expected code among the response alternatives.

Besides the labeled set the script measures:

* **exact-name regression** — every N-th catalog position name is sent verbatim
  and must come back as top-1 with its own code (lexicon bridges and feature
  changes must never break literal queries such as "Материалы лакокрасочные");
* **calibration** — expected calibration error (ECE) and a reliability table of
  the reported ``confidence`` against top-1 correctness;
* **threshold sweep** (``--sweep``) — metrics as a function of the ``review``
  threshold, with the model fixed (used to pick ``REVIEW_CONFIDENCE_FLOOR``).

Quality gates (``--min-top1``, ``--min-top3``, ``--min-silence``,
``--min-exact``) make the script exit non-zero when a metric regresses, so CI
can fail the build.

Usage:
    PYTHONPATH=api python eval/run_eval.py [--backend catalog|qdrant] [--output eval/eval_report.md]
        [--sweep] [--min-top1 80 --min-top3 88 --min-silence 50 --min-exact 95]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "api"))

from app.core.config import settings  # noqa: E402
from app.services.classifier import build_classifier  # noqa: E402

LABELED = ROOT / "eval" / "labeled_examples.csv"
CATALOG_CSV = ROOT / "data" / "data.csv"
EXACT_NAME_STEP = 10  # every 10th catalog row → ~110 literal queries
SWEEP_THRESHOLDS = (0.10, 0.12, 0.14, 0.16, 0.165, 0.18, 0.20, 0.22, 0.25, 0.30)
ECE_BINS = 5


def codes_match(predicted: str | None, expected: str) -> bool:
    if not predicted or not expected:
        return False
    predicted = predicted.strip()
    expected = expected.strip()
    return (
        predicted == expected
        or predicted.startswith(expected + ".")
        or expected.startswith(predicted + ".")
    )


def load_labeled(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def load_exact_names(path: Path, step: int = EXACT_NAME_STEP) -> list[tuple[str, str]]:
    """Deterministic sample of (name, okpd2) rows from the catalog."""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [
            ((row.get("name") or "").strip(), (row.get("okpd2") or "").strip())
            for row in csv.DictReader(handle, delimiter=";")
        ]
    rows = [(name, code) for name, code in rows if name and code]
    return rows[::step]


def evaluate_exact_names(classifier, names: list[tuple[str, str]]) -> list[dict]:
    """Literal catalog names must come back as themselves.

    87 position names carry several OKPD2 codes in the list (one row per code),
    so a hit is either the expected code or the very same position name.
    """
    exact: list[dict] = []
    for name, code in names:
        response = classifier.classify(name)
        same_name = " ".join((response.title or "").casefold().split()) == " ".join(
            name.casefold().split()
        )
        exact.append(
            {
                "query": name,
                "expected": code,
                "predicted": response.code or "(not_found)",
                "status": response.status,
                "confidence": response.confidence,
                "correct_top1": same_name or codes_match(response.code, code),
            }
        )
    return exact


def _review_threshold(classifier) -> float | None:
    reranker = getattr(classifier, "_reranker", None)
    model = getattr(reranker, "confidence_model", None)
    return getattr(model, "threshold_review", None) if model else None


def _set_review_threshold(classifier, value: float) -> None:
    classifier._reranker.confidence_model.threshold_review = value


def run(backend: str, *, sweep: bool = False) -> dict:
    settings.classifier_backend = backend
    classifier = build_classifier(backend)
    rows = load_labeled(LABELED)

    sweep_rows: list[dict] = []
    original_threshold = _review_threshold(classifier)
    if sweep and original_threshold is not None:
        for threshold in SWEEP_THRESHOLDS:
            _set_review_threshold(classifier, threshold)
            point = summarize({"results": _classify_rows(classifier, rows)})
            point["threshold"] = threshold
            sweep_rows.append(point)
        _set_review_threshold(classifier, original_threshold)

    results = _classify_rows(classifier, rows)
    exact = evaluate_exact_names(classifier, load_exact_names(CATALOG_CSV))
    return {
        "backend": backend,
        "results": results,
        "exact": exact,
        "sweep": sweep_rows,
        "review_threshold": original_threshold,
    }


def _classify_rows(classifier, rows: list[dict]) -> list[dict]:
    results = []
    for row in rows:
        query = (row["query"] or "").strip()
        expected = (row["expected_okpd2"] or "").strip()
        disputed = (row.get("disputed") or "0").strip() == "1"
        response = classifier.classify(query)

        if not expected:
            correct_top1 = response.status == "not_found"
            correct_top3 = correct_top1
        else:
            correct_top1 = codes_match(response.code, expected)
            correct_top3 = correct_top1 or any(
                codes_match(alternative.code, expected) for alternative in response.alternatives
            )
        results.append(
            {
                "query": query,
                "expected": expected or "(not_found)",
                "predicted": response.code or "(not_found)",
                "status": response.status,
                "confidence": response.confidence,
                "correct_top1": correct_top1,
                "correct_top3": correct_top3,
                "disputed": disputed,
                "title": response.title or "",
                "notes": (row.get("notes") or "").strip(),
            }
        )
    return results


def calibration(results: list[dict], bins: int = ECE_BINS) -> dict:
    """Reliability of ``confidence`` as P(top-1 correct) over answered queries.

    ``not_found`` answers report confidence 0 and are excluded: their
    correctness is a different question (was silence right?). ECE is the
    confidence-weighted mean |accuracy − mean confidence| over equal-width bins.
    """
    answered = [r for r in results if r["status"] != "not_found"]
    if not answered:
        return {"ece": 0.0, "bins": [], "answered": 0}
    table = []
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [
            r
            for r in answered
            if (low <= r["confidence"] < high) or (index == bins - 1 and r["confidence"] == 1.0)
        ]
        if not members:
            table.append({"range": (low, high), "count": 0, "confidence": 0.0, "accuracy": 0.0})
            continue
        mean_confidence = sum(r["confidence"] for r in members) / len(members)
        accuracy = sum(1 for r in members if r["correct_top1"]) / len(members)
        ece += len(members) / len(answered) * abs(accuracy - mean_confidence)
        table.append(
            {
                "range": (low, high),
                "count": len(members),
                "confidence": round(mean_confidence, 3),
                "accuracy": round(accuracy, 3),
            }
        )
    return {"ece": round(ece, 4), "bins": table, "answered": len(answered)}


def summarize(report: dict) -> dict:
    results = report["results"]
    with_expected = [r for r in results if r["expected"] != "(not_found)"]
    without_expected = [r for r in results if r["expected"] == "(not_found)"]

    def rate(items: list[dict], key: str) -> float:
        return round(100 * sum(1 for r in items if r[key]) / len(items), 1) if items else 0.0

    summary = {
        "total": len(results),
        "top1_expected": rate(with_expected, "correct_top1"),
        "top3_expected": rate(with_expected, "correct_top3"),
        "correct_silence": rate(without_expected, "correct_top1"),
        "top1_all": rate(results, "correct_top1"),
        "expected_rows": len(with_expected),
        "not_found_rows": len(without_expected),
        "status_distribution": dict(Counter(r["status"] for r in results)),
        "disputed_top1": rate([r for r in results if r["disputed"]], "correct_top1"),
        "clean_top1": rate([r for r in results if not r["disputed"]], "correct_top1"),
    }
    exact = report.get("exact") or []
    summary["exact_rows"] = len(exact)
    summary["exact_top1"] = rate(exact, "correct_top1")
    summary["ece"] = calibration(results)["ece"]
    return summary


def render_markdown(report: dict) -> str:
    summary = summarize(report)
    lines = [
        "# Отчёт классификатора по размеченному набору",
        "",
        f"**Режим:** `{report['backend']}`  ",
        f"**Примеров:** {summary['total']} "
        f"(с ожидаемым кодом: {summary['expected_rows']}, "
        f"ожидается `not_found`: {summary['not_found_rows']})",
        "",
        "## Метрики",
        "",
        "| Метрика | Значение |",
        "| --- | ---: |",
        f"| Top-1 (примеры с кодом) | {summary['top1_expected']}% |",
        f"| Top-3 (примеры с кодом) | {summary['top3_expected']}% |",
        f"| Верный `not_found` (нет в перечне) | {summary['correct_silence']}% |",
        f"| Top-1 по всем примерам | {summary['top1_all']}% |",
        f"| Top-1 без спорных | {summary['clean_top1']}% |",
        f"| Top-1 по спорным | {summary['disputed_top1']}% |",
        f"| Точные названия каталога, top-1 ({summary['exact_rows']} шт.) "
        f"| {summary['exact_top1']}% |",
        f"| ECE уверенности (отвеченные запросы) | {summary['ece']:.3f} |",
        "",
        f"Распределение статусов: `{summary['status_distribution']}`",
        "",
    ]
    lines += _render_calibration(report["results"])
    if report.get("sweep"):
        lines += _render_sweep(report["sweep"], report.get("review_threshold"))
    exact_misses = [row for row in report.get("exact") or [] if not row["correct_top1"]]
    if exact_misses:
        lines += [
            "## Точные названия, не вернувшиеся top-1",
            "",
            "| Запрос | Ожидалось | Получено | Статус | Уверенность |",
            "| --- | --- | --- | --- | ---: |",
        ]
        for row in exact_misses:
            lines.append(
                f"| {row['query']} | {row['expected']} | {row['predicted']} | "
                f"{row['status']} | {row['confidence']:.3f} |"
            )
        lines.append("")
    lines += [
        "## Ошибки и спорные случаи",
        "",
        "| Запрос | Ожидалось | Получено | Статус | Уверенность | Вердикт |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in report["results"]:
        verdict = "ok" if row["correct_top1"] else ("top-3" if row["correct_top3"] else "MISS")
        if row["correct_top1"] and not row["disputed"]:
            continue
        lines.append(
            f"| {row['query']} | {row['expected']} | {row['predicted']} | "
            f"{row['status']} | {row['confidence']:.3f} | {verdict} |"
        )
    lines += [
        "",
        "## Полный список",
        "",
        "| Запрос | Ожидалось | Получено | Статус | Уверенность | Top-1 | Top-3 |",
        "| --- | --- | --- | --- | ---: | :-: | :-: |",
    ]
    for row in report["results"]:
        lines.append(
            f"| {row['query']} | {row['expected']} | {row['predicted']} | "
            f"{row['status']} | {row['confidence']:.3f} | "
            f"{'+' if row['correct_top1'] else '−'} | {'+' if row['correct_top3'] else '−'} |"
        )
    lines += [
        "",
        "> Набор размечен вручную и служит измерением качества, а не юридической",
        "> экспертизой. Спорные случаи помечены в `eval/labeled_examples.csv`",
        "> колонкой `disputed`.",
        "",
    ]
    return "\n".join(lines)


def _render_calibration(results: list[dict]) -> list[str]:
    info = calibration(results)
    lines = [
        f"## Калибровка уверенности (ECE = {info['ece']:.3f}, отвечено: {info['answered']})",
        "",
        "| Диапазон | Запросов | Средняя уверенность | Доля верных top-1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in info["bins"]:
        low, high = row["range"]
        lines.append(
            f"| {low:.1f}–{high:.1f} | {row['count']} | {row['confidence']:.3f} | "
            f"{row['accuracy']:.3f} |"
        )
    lines.append("")
    return lines


def _render_sweep(sweep: list[dict], current: float | None) -> list[str]:
    lines = [
        "## Свип порога `review` (модель фиксирована)",
        "",
        "| порог review | top-1 (с кодом) | top-3 (с кодом) | верный not_found | top-1 всего |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in sweep:
        marker = "**" if current is not None and abs(point["threshold"] - current) < 1e-9 else ""
        lines.append(
            f"| {marker}{point['threshold']:.3f}{marker} | {point['top1_expected']}% | "
            f"{point['top3_expected']}% | {point['correct_silence']}% | {point['top1_all']}% |"
        )
    lines.append("")
    return lines


def check_gates(summary: dict, args: argparse.Namespace) -> list[str]:
    """Return human-readable failures for every gate that is not met."""
    gates = (
        ("top-1 (с кодом)", summary["top1_expected"], args.min_top1),
        ("top-3 (с кодом)", summary["top3_expected"], args.min_top3),
        ("верный not_found", summary["correct_silence"], args.min_silence),
        ("точные названия top-1", summary["exact_top1"], args.min_exact),
    )
    return [
        f"{name}: {value}% < порога {minimum}%"
        for name, value, minimum in gates
        if minimum is not None and value < minimum
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="catalog", choices=["catalog", "qdrant", "mock"])
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "eval_report.md")
    parser.add_argument("--sweep", action="store_true", help="таблица метрик по порогам review")
    parser.add_argument("--min-top1", type=float, default=None, help="гейт: top-1 с кодом, %%")
    parser.add_argument("--min-top3", type=float, default=None, help="гейт: top-3 с кодом, %%")
    parser.add_argument("--min-silence", type=float, default=None, help="гейт: тишина, %%")
    parser.add_argument("--min-exact", type=float, default=None, help="гейт: точные названия, %%")
    args = parser.parse_args()

    report = run(args.backend, sweep=args.sweep)
    markdown = render_markdown(report)
    args.output.write_text(markdown, encoding="utf-8")
    summary = summarize(report)
    print(f"Backend: {args.backend}")
    print(f"  top-1 (с кодом):  {summary['top1_expected']}%")
    print(f"  top-3 (с кодом):  {summary['top3_expected']}%")
    print(f"  верный not_found: {summary['correct_silence']}%")
    print(f"  top-1 всего:      {summary['top1_all']}%")
    print(f"  точные названия:  {summary['exact_top1']}% ({summary['exact_rows']} шт.)")
    print(f"  ECE:              {summary['ece']:.3f}")
    if report["sweep"]:
        print("  свип порога review:")
        for point in report["sweep"]:
            print(
                f"    {point['threshold']:.3f}: top-1 {point['top1_expected']}% / "
                f"top-3 {point['top3_expected']}% / тишина {point['correct_silence']}% / "
                f"всего {point['top1_all']}%"
            )
    print(f"Report: {args.output}")

    failures = check_gates(summary, args)
    if failures:
        print("КАЧЕСТВО НИЖЕ ПОРОГА:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
