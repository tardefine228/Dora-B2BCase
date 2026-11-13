"""Real-traffic drift check: re-classify UI exports with the current model.

UI exports (``classification-results*.csv``) are logs, not label sets: their
``user_verdict`` column is usually empty, and a prediction is never a label.
What they *are* is the only view onto real procurement names, so this tool
replays every exported query through the currently shipped build and compares
the fresh answer with the one recorded in the file, row by row:

* status transitions (``ok``/``review``/``not_found``), including the risky
  direction ``not_found -> ok`` (a newly confident answer where an older build
  stayed silent);
* code changes, each with the fresh confidence so a reviewer can see whether
  the model became more decisive or just more talkative;
* confidence bands of the current build over the whole export.

Labeling still happens through the review queue (``scripts/audit_data.py``
builds ``query_review.json``; ``scripts/export_reviewed_queries.py`` turns
reviewed rows into ``eval/real_queries.csv``). This script never writes labels.

Usage::

    PYTHONPATH=api python eval/real_traffic_check.py \\
        --exports eval/raw-exports/2026-10-26/*.csv \\
        --output eval/real-traffic-2026-10-26.md \\
        --json eval/data-review-2026-10-26/traffic_drift.json

The classifier is imported lazily in ``main`` so the pure functions stay
importable (and testable) without the application stack.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUSES = ("ok", "review", "not_found")


def read_exports(paths: list[Path]) -> list[dict]:
    """Flatten UI exports into rows, tagged with their source file and sha256."""
    rows: list[dict] = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle, delimiter=";"), 2):
                query = (row.get("query") or "").strip()
                if not query:
                    continue
                rows.append(
                    {
                        "query": query,
                        "old_code": (row.get("code") or "").strip(),
                        "old_status": (row.get("status") or "").strip(),
                        "old_confidence": _to_float(row.get("confidence")),
                        "source_file": path.name,
                        "source_sha256": digest,
                        "source_row": index,
                    }
                )
    return rows


def _to_float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def diff_predictions(rows: list[dict], answers: list[dict]) -> dict:
    """Compare exported answers with fresh ones.

    ``answers`` mirrors ``rows`` one to one; each item carries ``code``,
    ``status`` and ``confidence`` for the current build.
    """
    if len(rows) != len(answers):
        raise ValueError("rows and answers must have the same length")
    status_now = Counter(answer["status"] for answer in answers)
    status_then = Counter(row["old_status"] for row in rows)
    transitions: Counter[tuple[str, str]] = Counter()
    code_changes = []
    confidence_deltas = []
    for row, answer in zip(rows, answers, strict=True):
        if row["old_status"] != answer["status"]:
            transitions[(row["old_status"], answer["status"])] += 1
        if row["old_code"] != answer["code"]:
            code_changes.append(
                {
                    "query": row["query"],
                    "old_code": row["old_code"],
                    "old_status": row["old_status"],
                    "new_code": answer["code"],
                    "new_status": answer["status"],
                    "confidence": answer["confidence"],
                    "source_file": row["source_file"],
                    "source_row": row["source_row"],
                }
            )
        if (
            row["old_confidence"] is not None
            and row["old_code"] == answer["code"]
            and row["old_status"] == answer["status"]
        ):
            confidence_deltas.append(abs(row["old_confidence"] - answer["confidence"]))
    bands = Counter(_confidence_band(answer) for answer in answers)
    unchanged = sum(
        1
        for row, answer in zip(rows, answers, strict=True)
        if row["old_code"] == answer["code"] and row["old_status"] == answer["status"]
    )
    transition_text = {
        f"{old} -> {new}": count for (old, new), count in sorted(transitions.items())
    }
    return {
        "rows": len(rows),
        "status_then": dict(status_then),
        "status_now": dict(status_now),
        "transitions": transition_text,
        "unchanged_answers": unchanged,
        "code_changes": code_changes,
        "confidence_bands": dict(bands),
        "confidence_delta_on_unchanged": _stats(confidence_deltas),
    }


def _confidence_band(answer: dict) -> str:
    if answer["status"] == "not_found":
        return "молчание"
    confidence = answer["confidence"]
    if confidence < 0.6:
        return "<0.6"
    if confidence < 0.8:
        return "0.6–0.8"
    return ">=0.8"


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "max": round(ordered[-1], 4),
        "median": round(ordered[len(ordered) // 2], 4),
        "exact_matches": sum(1 for value in values if value < 1e-9),
    }


def render_markdown(diff: dict, files: list[dict], config: dict) -> str:
    """Turn the diff dictionary into the committed markdown report."""
    lines = [
        "# Реальный трафик: дрифт текущей сборки относительно выгрузок",
        "",
        f"Строк: **{diff['rows']}** (уникальные запросы из {len(files)} файлов). "
        "Выгрузки — журналы без вердиктов специалиста; предсказания не считаются "
        "разметкой. Переигровка выполнена текущей сборкой каталогового режима.",
        "",
        "## Воспроизведение",
        "",
        "```bash",
        f"PYTHONPATH=api EMBEDDER_BACKEND={config.get('embedder', 'tfidf')} "
        f"VECTOR_SEARCH={config.get('vector_search', 'true')} \\",
        "  python eval/real_traffic_check.py "
        "--exports <файлы> --output <отчёт.md> --json <дрифт.json>",
        "```",
        "",
        "## Источники",
        "",
        "| Файл | строк | sha256 |",
        "| --- | ---: | --- |",
    ]
    for item in files:
        lines.append(f"| `{item['name']}` | {item['rows']} | `{item['sha256'][:16]}…` |")
    lines += [
        "",
        "## Статусы",
        "",
        "| Статус | в выгрузке | сейчас |",
        "| --- | ---: | ---: |",
    ]
    for status in STATUSES:
        lines.append(
            f"| `{status}` | {diff['status_then'].get(status, 0)} "
            f"| {diff['status_now'].get(status, 0)} |"
        )
    lines += [
        "",
        "Полностью неизменных ответов (код и статус): "
        f"**{diff['unchanged_answers']}/{diff['rows']}**.",
        "",
        "### Переходы статусов",
        "",
        "| Переход | Строк |",
        "| --- | ---: |",
    ]
    for transition, count in diff["transitions"].items():
        lines.append(f"| {transition} | {count} |")
    risky = [
        change
        for change in diff["code_changes"]
        if change["old_status"] == "not_found" and change["new_status"] == "ok"
    ]
    lines += [
        "",
        "### Полосы уверенности текущей сборки",
        "",
        "| Полоса | Строк |",
        "| --- | ---: |",
    ]
    for band in (">=0.8", "0.6–0.8", "<0.6", "молчание"):
        lines.append(f"| {band} | {diff['confidence_bands'].get(band, 0)} |")
    stats = diff["confidence_delta_on_unchanged"]
    lines += [
        "",
        "## Дрифт кода",
        "",
        f"Смен кода: **{len(diff['code_changes'])}**. "
        "Из них новый уверенный ответ там, где выгрузка молчала "
        f"(`not_found -> ok`): **{len(risky)}** — это направление требует глаза "
        "специалиста в первую очередь.",
    ]
    if stats.get("n"):
        lines += [
            "",
            f"На неизменных ответах уверенность сдвинулась: медиана |Δ| = {stats['median']}, "
            f"максимум {stats['max']}, точных совпадений {stats['exact_matches']}/{stats['n']} "
            "(непустые значения говорят, что выгрузка сделана другой сборкой).",
        ]
    lines += [
        "",
        "| Запрос | Было | Стало | Уверенность |",
        "| --- | --- | --- | ---: |",
    ]
    for change in diff["code_changes"]:
        old = (
            f"`{change['old_code']}` ({change['old_status']})"
            if change["old_code"]
            else f"молчание ({change['old_status']})"
        )
        new = (
            f"`{change['new_code']}` ({change['new_status']})"
            if change["new_code"]
            else f"молчание ({change['new_status']})"
        )
        lines.append(f"| {change['query'][:80]} | {old} | {new} | {change['confidence']:.3f} |")
    lines.append("")
    return "\n".join(lines)


def classify_rows(rows: list[dict]) -> list[dict]:
    """Replay rows through the current catalog classifier (imported lazily)."""
    from app.services.classifier import build_classifier

    classifier = build_classifier()
    answers = []
    for row in rows:
        result = classifier.classify(row["query"])
        answers.append(
            {
                "code": (result.code or "").strip(),
                "status": result.status,
                "confidence": float(result.confidence),
            }
        )
    return answers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = read_exports(args.exports)
    if not rows:
        parser.error("во выгрузках нет ни одного непустого запроса")
    answers = classify_rows(rows)
    diff = diff_predictions(rows, answers)
    files = [
        {
            "name": path.name,
            "rows": sum(1 for row in rows if row["source_file"] == path.name),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in args.exports
    ]
    config = {
        "embedder": os.environ.get("EMBEDDER_BACKEND", "auto"),
        "vector_search": os.environ.get("VECTOR_SEARCH", "true"),
    }
    markdown = render_markdown(diff, files, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"config": config, "files": files, "diff": diff},
            ensure_ascii=False,
            indent=2,
        )
        args.json_path.write_text(payload + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "rows": diff["rows"],
                "unchanged": diff["unchanged_answers"],
                "code_changes": len(diff["code_changes"]),
                "status_now": diff["status_now"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
