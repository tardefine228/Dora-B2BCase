"""Evaluate the classifier on the hand-labeled set eval/labeled_examples.csv.

The labeled set is a *measurement* set: neither the reranker nor the thresholds
are trained on it (see ``ml/train_reranker.py``). Match rules:

* an empty ``expected_okpd2`` means the expected answer is ``not_found``;
* otherwise the predicted code may match exactly or as a prefix of the expected
  code (or vice versa) — "26.40.20" covers all television positions;
* ``top-3`` counts the expected code among the response alternatives.

Usage:
    PYTHONPATH=api python eval/run_eval.py [--backend catalog|qdrant] [--output eval/eval_report.md]
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


def run(backend: str) -> dict:
    settings.classifier_backend = backend
    classifier = build_classifier(backend)
    rows = load_labeled(LABELED)

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
    return {"backend": backend, "results": results}


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
        "",
        f"Распределение статусов: `{summary['status_distribution']}`",
        "",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="catalog", choices=["catalog", "qdrant", "mock"])
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "eval_report.md")
    args = parser.parse_args()

    report = run(args.backend)
    markdown = render_markdown(report)
    args.output.write_text(markdown, encoding="utf-8")
    summary = summarize(report)
    print(f"Backend: {args.backend}")
    print(f"  top-1 (с кодом):  {summary['top1_expected']}%")
    print(f"  top-3 (с кодом):  {summary['top3_expected']}%")
    print(f"  верный not_found: {summary['correct_silence']}%")
    print(f"  top-1 всего:      {summary['top1_all']}%")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
