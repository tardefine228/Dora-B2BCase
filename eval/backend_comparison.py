"""Compare the retrieval backends (`catalog` vs `qdrant`) on the labeled set.

The two backends share the calibrated reranker, the Platt confidence map and the
``ok``/``review`` thresholds; only the retrieval layer differs (lexical index vs
semantic Qdrant search). This script answers the question the semantic mode has
to justify itself with: **how many of the catalog backend's mistakes does the
semantic backend actually fix, and what does it break in exchange?**

For every row of ``eval/labeled_examples.csv`` both backends are asked the same
query in the same process, and the per-row outcome is written to
``eval/backend_comparison.csv`` (``;``-separated, matching the labeled set). The
summary printed to stdout and rendered with ``--markdown`` splits the deltas by
the buckets that matter for the deployment decision:

* ``catalog_silent_with_code`` — the catalog answered ``not_found`` although the
  product *is* in the list; the semantic mode is supposed to find these;
* ``catalog_false_hint`` — the catalog invented a code for a product that is not
  in the list; the semantic mode is supposed to stay silent;
* ``regressions`` — the catalog was right and the semantic mode is wrong; the
  price of switching the default backend.

The labeled set is a measurement set: this script reads it and never trains on
it. The model artifacts must exist (``python scripts/build_model.py``) and, for
the ``qdrant`` backend, the collection must be indexed
(``python -m ml.indexer``) with ``ML_ENABLED=true``.

Usage:
    ML_ENABLED=true EMBEDDER_BACKEND=sentence-transformers MODEL_NAME=BAAI/bge-m3 \
        PYTHONPATH=api python eval/backend_comparison.py
    ... --markdown eval/backend_comparison.md --output eval/backend_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "api", ROOT / "eval"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import run_eval as labeled_eval  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.services.classifier import build_classifier  # noqa: E402

BACKENDS = ("catalog", "qdrant")
CSV_FIELDS = (
    "query",
    "expected",
    "disputed",
    "catalog_code",
    "catalog_status",
    "catalog_confidence_raw",
    "catalog_margin",
    "catalog_latency_ms",
    "catalog_correct",
    "qdrant_code",
    "qdrant_status",
    "qdrant_confidence_raw",
    "qdrant_margin",
    "qdrant_latency_ms",
    "qdrant_correct",
    "verdict",
)


def _margin(response) -> float:
    """Top-1 minus top-2 confidence; 0.0 when there is no alternative to compare."""
    if not response.alternatives:
        return 0.0
    return round((response.confidence or 0.0) - (response.alternatives[0].confidence or 0.0), 4)


def _outcome(query: str, expected: str, response, latency_ms: float) -> dict:
    correct = (
        labeled_eval.codes_match(response.code, expected)
        if expected
        else response.status == "not_found"
    )
    return {
        "code": response.code or "(not_found)",
        "status": response.status,
        "raw": round(response.confidence_raw or 0.0, 4),
        "margin": _margin(response),
        "latency_ms": round(latency_ms, 1),
        "correct": bool(correct),
    }


def run(backend: str, cases: list[dict]) -> list[dict]:
    settings.classifier_backend = backend
    classifier = build_classifier(backend)
    # One warm-up call keeps model/index loading out of the per-row latency.
    if cases:
        classifier.classify(cases[0]["query"])
    outcomes = []
    for case in cases:
        started = time.perf_counter()
        response = classifier.classify(case["query"])
        elapsed_ms = (time.perf_counter() - started) * 1000
        outcomes.append(_outcome(case["query"], case["expected"], response, elapsed_ms))
    return outcomes


def collect_cases() -> list[dict]:
    return [
        {
            "query": row["query"].strip(),
            "expected": (row.get("expected_okpd2") or "").strip(),
            "disputed": (row.get("disputed") or "").strip(),
        }
        for row in labeled_eval.load_labeled(labeled_eval.LABELED)
    ]


def merge(cases: list[dict], catalog: list[dict], qdrant: list[dict]) -> list[dict]:
    rows = []
    for case, left, right in zip(cases, catalog, qdrant, strict=True):
        if left["correct"] and right["correct"]:
            verdict = "both_ok"
        elif left["correct"]:
            verdict = "regression"  # catalog right, semantic wrong
        elif right["correct"]:
            verdict = "fixed"
        else:
            verdict = "both_miss"
        rows.append({**case, "catalog": left, "qdrant": right, "verdict": verdict})
    return rows


def bucket_of(row: dict) -> str:
    """The bucket a row belongs to for the deployment decision."""
    if row["disputed"] == "1":
        return "disputed"
    if row["expected"]:
        return "with_code"
    return "not_in_list"


def summarize(rows: list[dict]) -> dict:
    summary: dict = {"rows": len(rows), "verdicts": dict(Counter(r["verdict"] for r in rows))}
    for backend in BACKENDS:
        with_code = [r for r in rows if r["expected"]]
        silence = [r for r in rows if not r["expected"]]
        summary[backend] = {
            "top1": round(
                100 * sum(r[backend]["correct"] for r in with_code) / max(len(with_code), 1), 1
            ),
            "top1_all": round(
                100 * sum(r[backend]["correct"] for r in rows) / max(len(rows), 1), 1
            ),
            "silence": round(
                100 * sum(r[backend]["correct"] for r in silence) / max(len(silence), 1), 1
            ),
            "statuses": dict(Counter(r[backend]["status"] for r in rows)),
            "latency": latency_summary([r[backend]["latency_ms"] for r in rows]),
        }

    silent_with_code = [r for r in rows if r["expected"] and r["catalog"]["status"] == "not_found"]
    false_hints = [r for r in rows if not r["expected"] and r["catalog"]["status"] != "not_found"]
    summary["catalog_silent_with_code"] = {
        "total": len(silent_with_code),
        "found_top1": sum(r["qdrant"]["correct"] for r in silent_with_code),
        "still_silent": sum(r["qdrant"]["status"] == "not_found" for r in silent_with_code),
    }
    summary["catalog_false_hint"] = {
        "total": len(false_hints),
        "silenced": sum(r["qdrant"]["status"] == "not_found" for r in false_hints),
        "still_hint": sum(r["qdrant"]["status"] != "not_found" for r in false_hints),
    }
    summary["regressions"] = [
        r for r in rows if r["expected"] and r["catalog"]["correct"] and not r["qdrant"]["correct"]
    ]
    summary["fixed"] = [
        r for r in rows if r["expected"] and not r["catalog"]["correct"] and r["qdrant"]["correct"]
    ]
    return summary


def latency_summary(values: list[float]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "p50": round(statistics.median(ordered), 1),
        "p95": round(ordered[index], 1),
        "mean": round(statistics.fmean(ordered), 1),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "query": row["query"],
                    "expected": row["expected"] or "(not_found)",
                    "disputed": row["disputed"],
                    "catalog_code": row["catalog"]["code"],
                    "catalog_status": row["catalog"]["status"],
                    "catalog_confidence_raw": row["catalog"]["raw"],
                    "catalog_margin": row["catalog"]["margin"],
                    "catalog_latency_ms": row["catalog"]["latency_ms"],
                    "catalog_correct": int(row["catalog"]["correct"]),
                    "qdrant_code": row["qdrant"]["code"],
                    "qdrant_status": row["qdrant"]["status"],
                    "qdrant_confidence_raw": row["qdrant"]["raw"],
                    "qdrant_margin": row["qdrant"]["margin"],
                    "qdrant_latency_ms": row["qdrant"]["latency_ms"],
                    "qdrant_correct": int(row["qdrant"]["correct"]),
                    "verdict": row["verdict"],
                }
            )


def render_markdown(summary: dict, rows: list[dict]) -> str:
    lines = [
        "# Сравнение бэкендов `catalog` и `qdrant` на размеченном наборе",
        "",
        f"**Строк:** {summary['rows']} · **Разметка:** `eval/labeled_examples.csv` "
        "(измерительный набор, обучение не затрагивалось)",
        "",
        "## Метрики",
        "",
        "| Метрика | catalog | qdrant |",
        "| --- | ---: | ---: |",
    ]
    for label, key in (
        ("Top-1 (примеры с кодом)", "top1"),
        ("Top-1 по всем примерам", "top1_all"),
        ("Верный `not_found`", "silence"),
        ("Латентность p50, мс", None),
        ("Латентность p95, мс", None),
    ):
        if key:
            left, right = summary["catalog"][key], summary["qdrant"][key]
            lines.append(f"| {label} | {left}% | {right}% |")
        elif label.endswith("p50, мс"):
            lines.append(
                f"| {label} | {summary['catalog']['latency']['p50']} | "
                f"{summary['qdrant']['latency']['p50']} |"
            )
        else:
            lines.append(
                f"| {label} | {summary['catalog']['latency']['p95']} | "
                f"{summary['qdrant']['latency']['p95']} |"
            )

    silent = summary["catalog_silent_with_code"]
    hints = summary["catalog_false_hint"]
    lines += [
        "",
        "## Дельты по проблемным срезам",
        "",
        "| Срез | Всего | qdrant исправил | qdrant не изменил |",
        "| --- | ---: | ---: | ---: |",
        f"| catalog промолчал, хотя код есть | {silent['total']} | "
        f"{silent['found_top1']} | {silent['still_silent']} |",
        f"| catalog выдумал подсказку вне перечня | {hints['total']} | "
        f"{hints['silenced']} | {hints['still_hint']} |",
        "",
        "| Вердикт | Строк |",
        "| --- | ---: |",
    ]
    for verdict, count in sorted(summary["verdicts"].items()):
        lines.append(f"| {verdict} | {count} |")

    lines += ["", "## Строки, где изменился вердикт", ""]
    changed = [r for r in rows if r["verdict"] in {"fixed", "regression"}]
    if not changed:
        lines.append("Изменений нет.")
    else:
        lines += [
            "| Запрос | Ожидалось | catalog | qdrant | Вердикт |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in sorted(changed, key=lambda r: (r["verdict"], r["query"])):
            lines.append(
                f"| {row['query']} | {row['expected'] or '(not_found)'} | "
                f"{row['catalog']['code']} ({row['catalog']['status']}) | "
                f"{row['qdrant']['code']} ({row['qdrant']['status']}) | {row['verdict']} |"
            )
    lines += [
        "",
        "> Отчёт информационный: он не входит в CI-гейт и не меняет ни модель,",
        "> ни пороги. Решение по режиму по умолчанию фиксируется в `eval/experiments.md`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "backend_comparison.csv")
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument(
        "--backends",
        default="catalog,qdrant",
        help="через запятую; catalog,qdrant по умолчанию",
    )
    args = parser.parse_args()

    if args.backends.strip() != "catalog,qdrant":
        parser.error("поддерживается только 'catalog,qdrant' (обе стороны сравнения обязательны)")

    cases = collect_cases()
    print(f"Размеченных строк: {len(cases)}")
    if not cases:
        return 1

    outcomes = {}
    for backend in BACKENDS:
        started = time.perf_counter()
        outcomes[backend] = run(backend, cases)
        print(f"  [{backend}] прогон {time.perf_counter() - started:.1f} с")

    rows = merge(cases, outcomes["catalog"], outcomes["qdrant"])
    summary = summarize(rows)
    write_csv(rows, args.output)

    print(f"catalog: top-1 {summary['catalog']['top1']}% / "
          f"тишина {summary['catalog']['silence']}% / "
          f"p50 {summary['catalog']['latency']['p50']} мс")
    print(f"qdrant : top-1 {summary['qdrant']['top1']}% / "
          f"тишина {summary['qdrant']['silence']}% / "
          f"p50 {summary['qdrant']['latency']['p50']} мс")
    silent = summary["catalog_silent_with_code"]
    hints = summary["catalog_false_hint"]
    print(f"catalog промолчал с кодом в перечне: {silent['total']}, "
          f"qdrant нашёл top-1: {silent['found_top1']}, всё ещё молчит: {silent['still_silent']}")
    print(f"catalog выдумал подсказку вне перечня: {hints['total']}, "
          f"qdrant замолчал: {hints['silenced']}, всё ещё подсказывает: {hints['still_hint']}")
    print(f"вердикты: {summary['verdicts']}")
    print(f"исправлено строк: {len(summary['fixed'])}, регрессий: {len(summary['regressions'])}")
    for row in summary["regressions"]:
        print(f"  регрессия: {row['query']} | ожидалось {row['expected']} | "
              f"catalog {row['catalog']['code']} | qdrant {row['qdrant']['code']}")
    print(f"CSV: {args.output}")

    if args.markdown:
        args.markdown.write_text(render_markdown(summary, rows), encoding="utf-8")
        print(f"Markdown: {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
