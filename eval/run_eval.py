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

Quality gates. CI does not use absolute floors any more: it measures the shipped
configuration and compares it row by row with a recorded measurement
(``--baseline eval/baselines/<config>.json``, policy in ``eval/quality_gate.py``).
A floor on a rate can silently stop describing anything once the artifact cannot be
rebuilt; a recorded measurement cannot. ``--report`` writes the actual measurement
in the same format, so refreshing a baseline is an explicit, reviewable copy.
``--min-top1``/``--min-top3``/``--min-silence``/``--min-exact`` stay for ad-hoc use.

Usage:
    PYTHONPATH=api python eval/run_eval.py [--backend catalog|qdrant] [--output eval/eval_report.md]
        [--sweep] [--baseline eval/baselines/dense-tfidf.json] [--report /tmp/actual.json]
        [--min-top1 80 --min-top3 88 --min-silence 50 --min-exact 95]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "api"))

from app.core.config import settings  # noqa: E402
from app.services.classifier import build_classifier  # noqa: E402

from eval.dense_gate import Counts  # noqa: E402

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
                "confidence_raw": response.confidence_raw
                if response.confidence_raw is not None
                else response.confidence,
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
        # Kept for provenance: which artifact and embedder produced these numbers.
        "classifier": classifier,
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
                "confidence_raw": response.confidence_raw
                if response.confidence_raw is not None
                else response.confidence,
                "correct_top1": correct_top1,
                "correct_top3": correct_top3,
                "disputed": disputed,
                "title": response.title or "",
                "notes": (row.get("notes") or "").strip(),
            }
        )
    return results


def calibration(results: list[dict], bins: int = ECE_BINS, key: str = "confidence") -> dict:
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
            r for r in answered if (low <= r[key] < high) or (index == bins - 1 and r[key] == 1.0)
        ]
        if not members:
            table.append({"range": (low, high), "count": 0, "confidence": 0.0, "accuracy": 0.0})
            continue
        mean_confidence = sum(r[key] for r in members) / len(members)
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


def metric_counts(report: dict) -> dict[str, Counts]:
    """Raw row counts behind every gated metric — the only thing the gate trusts.

    Rates are derived from these counts below, so the number printed in the report
    and the number the gate compares can never diverge.

    ``ok_precision`` deliberately has a model-chosen denominator: it is the share of
    *confident* answers (status ``ok``) that were right, and a different artifact
    legitimately labels a different number of rows as confident.
    """
    results = report["results"]
    with_expected = [r for r in results if r["expected"] != "(not_found)"]
    without_expected = [r for r in results if r["expected"] == "(not_found)"]
    confident = [r for r in results if r["status"] == "ok"]
    exact = report.get("exact") or []

    def counted(items: list[dict], key: str = "correct_top1") -> Counts:
        return Counts(sum(1 for r in items if r[key]), len(items))

    return {
        "correct_silence": counted(without_expected),
        "exact_top1": counted(exact),
        "top1_expected": counted(with_expected),
        "top3_expected": counted(with_expected, "correct_top3"),
        "top1_all": counted(results),
        "ok_precision": counted(confident),
    }


def summarize(report: dict) -> dict:
    results = report["results"]
    counts = metric_counts(report)

    def rate(items: list[dict], key: str) -> float:
        return round(100 * sum(1 for r in items if r[key]) / len(items), 1) if items else 0.0

    summary = {
        "total": len(results),
        "top1_expected": round(counts["top1_expected"].rate, 1),
        "top3_expected": round(counts["top3_expected"].rate, 1),
        "correct_silence": round(counts["correct_silence"].rate, 1),
        "top1_all": round(counts["top1_all"].rate, 1),
        "ok_precision": round(counts["ok_precision"].rate, 1),
        "expected_rows": counts["top1_expected"].total,
        "not_found_rows": counts["correct_silence"].total,
        "status_distribution": dict(Counter(r["status"] for r in results)),
        "disputed_top1": rate([r for r in results if r["disputed"]], "correct_top1"),
        "clean_top1": rate([r for r in results if not r["disputed"]], "correct_top1"),
    }
    summary["exact_rows"] = counts["exact_top1"].total
    summary["exact_top1"] = round(counts["exact_top1"].rate, 1)
    summary["ece"] = calibration(results)["ece"]
    summary["ece_raw"] = calibration(results, key="confidence_raw")["ece"]
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
        f"| ECE уверенности, калиброванная шкала (отвеченные запросы) | {summary['ece']:.3f} |",
        f"| ECE сырой уверенности (до калибровки, для сравнения) | {summary['ece_raw']:.3f} |",
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
    raw = calibration(results, key="confidence_raw")
    lines = [
        f"## Калибровка уверенности (ECE = {info['ece']:.3f}, "
        f"сырая шкала = {raw['ece']:.3f}, отвечено: {info['answered']})",
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


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _embedder_backend() -> str | None:
    try:
        from ml.embedder import resolve_backend

        return resolve_backend()
    except Exception:  # noqa: BLE001 - provenance must never break the measurement
        return None


def build_measurement(report: dict, classifier) -> dict:
    """The gated numbers plus the configuration they were measured on.

    The provenance is the point: a baseline without the artifact hash and the resolved
    upstream settings cannot be checked later, which is exactly how the old absolute
    floors rotted.
    """
    from eval.quality_gate import measurement

    summary = summarize(report)
    counts = metric_counts(report)
    info = getattr(classifier, "vector_info", None) or {}
    reranker_name = info.get("reranker")
    artifacts: dict[str, str | None] = {}
    if reranker_name:

        candidates = [ROOT / "ml" / "artifacts" / reranker_name, ROOT / "models" / reranker_name]
        for candidate in candidates:
            digest = _sha256(candidate)
            if digest:
                artifacts[reranker_name] = digest
                break
        else:  # pragma: no cover - the artifact is always shipped or rebuilt
            artifacts[reranker_name] = None

    config = {
        "backend": report["backend"],
        "vector_search": bool(getattr(settings, "vector_search", False)),
        "embedder_backend": _embedder_backend(),
        "vector_store": getattr(settings, "vector_store", None),
        "embedder": info.get("embedder"),
        "reranker": reranker_name,
        "scoring": info.get("scoring"),
        "semantic_reranker": getattr(settings, "semantic_reranker", None),
        "review_threshold": report.get("review_threshold"),
        "labeled_rows": summary["total"],
        "expected_rows": summary["expected_rows"],
        "not_found_rows": summary["not_found_rows"],
        "artifact_sha256": artifacts,
        "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return measurement(counts, summary["ece"], config=config)


def compare_with_baseline(actual: dict, baseline_path: Path) -> str | None:
    """Print the comparison; return the failure text when the build must fail."""
    from eval.quality_gate import evaluate_regression, load_measurement

    baseline = load_measurement(baseline_path)
    verdict = evaluate_regression(baseline, actual)
    print(f"  базовая линия: {baseline_path}")
    print(verdict.table())
    if verdict.keep:
        return None
    return "; ".join(verdict.reasons)


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
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="записать замер (счётчики, ECE, провенанс) в JSON — формат базовой линии",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="сравнить с записанным замером; регрессия → код возврата 1",
    )
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
    print(f"  ECE (калибровка): {summary['ece']:.3f} (сырая шкала {summary['ece_raw']:.3f})")
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

    actual = build_measurement(report, report["classifier"])
    if args.report is not None:
        args.report.write_text(
            json.dumps(actual, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Measurement: {args.report}")

    if args.baseline is not None:
        regression = compare_with_baseline(actual, args.baseline)
        if regression:
            print("РЕГРЕССИЯ ОТНОСИТЕЛЬНО БАЗОВОЙ ЛИНИИ:")
            print(f"  - {regression}")
            failures.append(regression)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
