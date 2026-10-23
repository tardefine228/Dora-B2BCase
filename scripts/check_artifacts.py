"""Prove that a shipped reranker still comes from the current trainer.

The artifact in ``ml/artifacts`` is what the image serves, so it must be explainable
by the code in this repository. The shipped lexical file once was not: it stored
29277 training pairs and thresholds 0.621/0.097 while a fresh build stores 27858 and
0.306/0.227, and nothing in CI noticed for months — the only symptom was a quality
gate that failed for a reason nobody could trace.

Byte equality is *not* the right test: weights come out of a BLAS-backed optimiser and
the last digits depend on the machine. What must match is everything that describes the
training run (number of pairs, split, thresholds, feature set) and the weights to within
``TOLERANCE``.

Usage:
    python scripts/check_artifacts.py models/reranker.json ml/artifacts/reranker.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Real trainer changes move weights by ~0.1 (and change the run description), while a
# different BLAS/library build moves the last few digits. 1e-4 separates the two.
TOLERANCE = 1e-4
# Everything that ties an artifact to one training run. A different trainer changes at
# least one of these (the retired file differed in all of them).
RUN_KEYS = (
    "seed",
    "train_pairs",
    "train_queries",
    "validation_queries",
    "split_strategy",
    "confidence_candidates",
    "pair_thresholds",
    "confidence_thresholds",
    "embedder",
)


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare(fresh: dict, shipped: dict) -> list[str]:
    """Human-readable differences; an empty list means they describe the same run."""
    problems: list[str] = []

    for key in RUN_KEYS:
        old, new = shipped.get("meta", {}).get(key), fresh.get("meta", {}).get(key)
        if old != new:
            problems.append(f"meta.{key}: отгружено {old!r}, пересборка {new!r}")

    if shipped.get("feature_names") != fresh.get("feature_names"):
        problems.append("набор признаков отличается — артефакт обучен другой версией кода")

    shipped_weights = shipped.get("weights") or {}
    fresh_weights = fresh.get("weights") or {}
    if set(shipped_weights) != set(fresh_weights):
        missing = sorted(set(fresh_weights) ^ set(shipped_weights))
        problems.append(f"набор весов отличается: {missing}")
    for name in sorted(set(shipped_weights) & set(fresh_weights)):
        delta = abs(float(fresh_weights[name]) - float(shipped_weights[name]))
        if delta > TOLERANCE:
            problems.append(f"вес {name}: {shipped_weights[name]} против {fresh_weights[name]}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fresh", type=Path, help="артефакт из свежей сборки")
    parser.add_argument("shipped", type=Path, help="отгруженный артефакт")
    args = parser.parse_args(argv)

    for path in (args.fresh, args.shipped):
        if not path.is_file():
            print(f"Нет файла: {path}", file=sys.stderr)
            return 2

    fresh, shipped = load(args.fresh), load(args.shipped)
    problems = compare(fresh, shipped)
    if not problems:
        print(
            f"OK: {args.shipped} соответствует пересборке "
            f"({fresh['meta'].get('train_pairs')} пар, "
            f"пороги {fresh['meta'].get('pair_thresholds')})"
        )
        return 0

    print(f"{args.shipped} НЕ соответствует текущему тренеру:", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    print(
        "Пересоберите и отгрузите артефакт: python scripts/build_model.py, затем "
        "cp models/reranker.json ml/artifacts/ и обновите sha256 в ml/artifacts/README.md",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
