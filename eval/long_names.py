"""Long-name slice (ML-7): does the head/qualifier split help?

Official PP-1875 names run to 200–1100 characters ("Компьютеры портативные
массой не более 10 кг, такие как ноутбуки, планшетные компьютеры, …"). Scored
as one field, the qualifier storm dilutes the precision features, so the model
rates a *correct* long position as imprecise.

The labeled set cannot answer this question: only six of its 274 rows expect a
position whose name is over ``LONG_TITLE_CHARS``, so both modes score exactly
the same on the gate. This script measures the slice directly — for every long
position it builds a query from the head of its own name (what a person types
when the position is the right one) and reports how the pipeline answers it.

    PYTHONPATH=api python eval/long_names.py [--output eval/long_names.md]

The script runs itself twice — with ``LONG_TITLE_SPLIT=0`` and ``=1`` — because
the switch is read at import time, and prints both columns. Read the result as
a trade-off, not a win: the split lifts top-1 by making the model *more
decisive*, and decisiveness cuts both ways — part of the ``review`` hints turn
into confident wrong answers instead of "check this one". That is why the
switch ships off.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "api"))

HEAD_WORDS = 8


def _queries() -> list[tuple[str, str]]:
    """``(query, expected_code)`` pairs for every position with a long name."""
    from ml.catalog_loader import load_catalog
    from ml.features import LONG_TITLE_CHARS

    pairs: list[tuple[str, str]] = []
    for item in load_catalog():
        name = item["name"]
        if len(name) <= LONG_TITLE_CHARS:
            continue
        head = name
        for index, char in enumerate(name):
            if char in ",;:(" and index >= 20:
                head = name[:index]
                break
        query = " ".join(head.split()[:HEAD_WORDS]).strip(" ,;:(–—")
        if query:
            pairs.append((query, item["pp_code"]))
    return pairs


def measure() -> dict:
    """Run the slice in the current mode (the switch is already in the env)."""
    from app.services.classifier import build_classifier

    classifier = build_classifier()
    stats = {"total": 0, "top1": 0, "review": 0, "missed": 0, "wrong": 0}
    for query, expected in _queries():
        answer = classifier.classify(query)
        stats["total"] += 1
        status = getattr(answer, "status", "")
        code = getattr(answer, "code", None)
        if status == "not_found":
            stats["missed"] += 1
        elif code == expected:
            stats["top1"] += 1
        elif status == "review":
            stats["review"] += 1
        else:
            stats["wrong"] += 1
    return stats


def _run_mode(mode: str) -> dict:
    env = dict(os.environ, LONG_TITLE_SPLIT=mode, PYTHONPATH=os.pathsep.join(
        [str(ROOT), str(ROOT / "api"), os.environ.get("PYTHONPATH", "")]
    ))
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--json"],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )
    if result.returncode != 0:
        raise SystemExit(f"режим {mode} упал:\n{result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def _row(label: str, stats: dict) -> str:
    total = max(stats["total"], 1)
    return (
        f"| {label} | {stats['top1']}/{total} ({stats['top1'] / total * 100:.1f}%) "
        f"| {stats['review']} | {stats['wrong']} | {stats['missed']} |"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print raw counts only")
    parser.add_argument("--output", type=Path, default=None, help="write a markdown report")
    args = parser.parse_args()

    if args.json:
        print(json.dumps(measure(), ensure_ascii=False))
        return 0

    off = _run_mode("0")
    on = _run_mode("1")
    header = (
        "| Режим | top-1 по коду | review | неверный код | тишина |\n"
        "| --- | --- | --- | --- | --- |\n"
    )
    report = (
        f"# Срез длинных названий (ML-7)\n\n"
        f"Запрос = первые {HEAD_WORDS} слов головы названия; ожидаемый код — код самой позиции.\n"
        f"Длинных позиций в каталоге: {off['total']}.\n\n"
        + header
        + _row("`LONG_TITLE_SPLIT=0` (по умолчанию)", off)
        + "\n"
        + _row("`LONG_TITLE_SPLIT=1`", on)
        + "\n\n"
        "Разбиение на голову и квалификаторы поднимает top-1, но делает модель\n"
        "*решительнее*: часть подсказок `review` превращается в уверенно неверный код,\n"
        "а не в верный ответ. Подсказка «проверьте этот пункт» безопаснее, чем\n"
        "уверенная ошибка, поэтому переключатель по умолчанию выключен — решение\n"
        "продуктовое, а не техническое.\n"
    )
    print(report)
    if args.output is not None:
        args.output.write_text(report, encoding="utf-8")
        print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
