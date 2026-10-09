"""Synthetic query benchmark: catalog positions turned into user-like queries.

Every ``--step``-th catalog position produces 3–4 variants (the full name, its
lowercase form, the first three words, the first two words). The expected code is
the source position itself, so the benchmark needs no manual labelling and can be
re-generated on any catalog dump.

The expectations are deliberately strict: a variant that the classifier resolves
to a *sibling* position with a different code counts as a miss, while a human
reviewer would often accept it (see ``eval/labeled_examples.csv`` и
``eval/experiments.md``, эксп. 11). Treat the numbers as a lower bound and read
them next to the labeled set, not instead of it.

This benchmark is informational: it does **not** feed the CI quality gate
(``.github/workflows/lint.yml``) and must not be used for calibration.

Usage:
    PYTHONPATH=api python eval/synthetic_eval.py [--step 4] [--variants 4]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.catalog_loader import load_catalog  # noqa: E402
from ml.features import tokens  # noqa: E402

DEFAULT_OUTPUT = ROOT / "eval" / "synthetic_report.md"


def codes_match(predicted: str | None, expected: str) -> bool:
    """Same comparison as ``eval/run_eval.py`` (segment-wise prefixes)."""
    if not predicted or not expected:
        return False
    predicted, expected = predicted.strip(), expected.strip()
    return (
        predicted == expected
        or predicted.startswith(expected + ".")
        or expected.startswith(predicted + ".")
    )


def unique_positions(catalog: list[dict]) -> list[dict]:
    """Catalog rows without repeated names (one position per title)."""
    seen: set[str] = set()
    positions: list[dict] = []
    for item in catalog:
        name = (item.get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        positions.append(item)
    return positions


def query_variants(name: str, *, variants: int = 4) -> list[str]:
    """The full name, its lowercase form and 2–3 word truncations."""
    words = tokens(name)
    candidates = [name, name.lower()]
    if len(words) >= 3:
        candidates.append(" ".join(words[:3]))
    if len(words) >= 2:
        candidates.append(" ".join(words[:2]))
    result: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in result:
            result.append(candidate)
        if len(result) >= variants:
            break
    return result


def build_cases(catalog: list[dict], *, step: int, variants: int) -> list[tuple[str, str]]:
    """``(query, expected_code)`` pairs, deterministic for a given catalog."""
    cases: list[tuple[str, str]] = []
    for index, item in enumerate(unique_positions(catalog)):
        if index % step:
            continue
        code = (item.get("pp_code") or "").strip()
        if not code:
            continue
        for query in query_variants(item["name"], variants=variants):
            cases.append((query, code))
    return cases


def _rate(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return round(100 * sum(1 for row in rows if row[key]) / len(rows), 1)


def summarize(rows: list[dict]) -> dict:
    answered = [row for row in rows if row["status"] != "not_found"]
    return {
        "total": len(rows),
        "answered": len(answered),
        "answered_share": round(100 * len(answered) / (len(rows) or 1), 1),
        "top1_answered": _rate(answered, "correct_top1"),
        "top3_answered": _rate(answered, "correct_top3"),
        "top1_all": _rate(rows, "correct_top1"),
        "status_distribution": dict(Counter(row["status"] for row in rows)),
        "by_status": _by_status(rows),
    }


def _by_status(rows: list[dict]) -> dict:
    """Top-1 accuracy inside each answered status bucket (``ok`` / ``review``).

    ``ok`` is the bucket a specialist is told to trust without a second look, so
    its precision matters more than the overall top-1. Keep the bucket precision
    separate from the overall number (see ``eval/experiments.md``, эксп. 13).
    """
    buckets: dict[str, dict] = {}
    for status in ("ok", "review"):
        bucket = [row for row in rows if row["status"] == status]
        buckets[status] = {
            "count": len(bucket),
            "correct": sum(1 for row in bucket if row["correct_top1"]),
            "precision": _rate(bucket, "correct_top1"),
        }
    return buckets


def render_markdown(summary: dict, *, step: int, variants: int, backend: str) -> str:
    lines = [
        "# Синтетический бенчмарк каталога",
        "",
        "Генератор: `eval/synthetic_eval.py` — каждый "
        f"{step}-й различающийся по названию пункт каталога превращается в "
        f"{variants} варианта запроса (имя, имя в нижнем регистре, первые 3 и 2 слова). "
        "Ожидаемый код — сам пункт каталога: разметка не нужна, набор воспроизводим.",
        "",
        f"**Режим:** `{backend}`  ",
        f"**Запросов:** {summary['total']} "
        f"(отвечено {summary['answered']}, {summary['answered_share']}%)",
        "",
        "| Метрика | Значение |",
        "| --- | ---: |",
        f"| Top-1 среди отвеченных | {summary['top1_answered']}% |",
        f"| Top-3 среди отвеченных | {summary['top3_answered']}% |",
        f"| Top-1 по всем запросам | {summary['top1_all']}% |",
        "",
        f"Распределение статусов: `{summary['status_distribution']}`",
        "",
        "**Точность по корзинам статусов** (top-1 внутри корзины):",
        "",
        "| Статус | Запросов | Верных | Точность |",
        "| --- | ---: | ---: | ---: |",
    ]
    for status, bucket in summary["by_status"].items():
        lines.append(
            f"| `{status}` | {bucket['count']} | {bucket['correct']} | {bucket['precision']}% |"
        )
    lines += [
        "",
        "Корзина `ok` на этом бенчмарке вероятно завышена: запрос — это буквально",
        "название позиции (или его обрезка), поэтому высокая точность не говорит о",
        "точности `ok` на живых запросах. Размеченный набор (`eval/labeled_examples.csv`)",
        "остаётся основным измерением точности `ok`.",
        "",
        "**Как читать.** Ожидание — код исходного пункта, а не канонический ответ,",
        "поэтому промах на соседнюю позицию с другим кодом считается ошибкой, хотя",
        "специалист мог бы её принять (см. `eval/eval_report.md` и раздел",
        "«Известные ограничения» в `eval/experiments.md`). Числа — нижняя оценка, и",
        "на CI-гейт они не влияют.",
        "",
    ]
    return "\n".join(lines)


def run(backend: str, *, step: int, variants: int) -> dict:
    from app.services.classifier import build_classifier

    settings_module = __import__("app.core.config", fromlist=["settings"])
    settings_module.settings.classifier_backend = backend
    classifier = build_classifier(backend)
    rows: list[dict] = []
    for query, expected in build_cases(load_catalog(), step=step, variants=variants):
        response = classifier.classify(query)
        correct_top1 = response.status != "not_found" and codes_match(response.code, expected)
        correct_top3 = correct_top1 or any(
            codes_match(alternative.code, expected) for alternative in response.alternatives
        )
        rows.append(
            {
                "query": query,
                "expected": expected,
                "predicted": response.code or "(not_found)",
                "status": response.status,
                "correct_top1": correct_top1,
                "correct_top3": correct_top3,
            }
        )
    return {"backend": backend, "rows": rows, "summary": summarize(rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="catalog", choices=["catalog", "qdrant", "mock"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--step", type=int, default=4, help="каждый N-й пункт каталога")
    parser.add_argument("--variants", type=int, default=4, help="вариантов на пункт (3–4)")
    args = parser.parse_args()

    report = run(args.backend, step=args.step, variants=args.variants)
    summary = report["summary"]
    args.output.write_text(
        render_markdown(summary, step=args.step, variants=args.variants, backend=args.backend),
        encoding="utf-8",
    )
    print(f"Backend: {args.backend}; step={args.step}, variants={args.variants}")
    print(f"  запросов:         {summary['total']} (отвечено {summary['answered']})")
    print(f"  top-1 отвеченных: {summary['top1_answered']}%")
    print(f"  top-3 отвеченных: {summary['top3_answered']}%")
    print(f"  top-1 всего:      {summary['top1_all']}%")
    print(f"  статусы:          {summary['status_distribution']}")
    for status, bucket in summary["by_status"].items():
        print(
            f"  точность {status:<7}: {bucket['precision']}% "
            f"({bucket['correct']}/{bucket['count']})"
        )
    print(f"Report: {args.output}")
    print("CI-гейт качества этот бенчмарк не проверяет (отчёт информационный).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
