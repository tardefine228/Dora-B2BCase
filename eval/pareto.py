"""Compare reranker artifacts at equal silence, not at equal threshold.

The question "is the old artifact better?" cannot be answered by comparing headline
numbers: every artifact carries its own calibrated confidence scale, so the same
numeric threshold means a different working point for each of them. The shipped
lexical file, for instance, answers 191 of 274 rows while the retrained one answers
163 at *its* threshold — that is why it shows a higher top-1.

So the comparison is made in the plane that matters to the product:

* **silence** — share of out-of-scope queries answered with an honest ``not_found``;
* **top-1** — share of in-scope queries answered with the expected code.

For a required level of silence the tool picks, per artifact, the most talkative
threshold that still meets it, and prints the metrics there. A model that wins at
equal silence is better; a model that only answers more is merely louder.

Usage:
    PYTHONPATH=api python eval/pareto.py                     # shipped lexical + vector
    PYTHONPATH=api python eval/pareto.py --extra old=/tmp/reranker-2026-09.json
    PYTHONPATH=api python eval/pareto.py --silence 70 80 85 90
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "api"))

from eval.dense_gate import Counts  # noqa: E402

DEFAULT_SILENCE = (70, 75, 80, 85, 90, 95)
# Fractions of the observed confidence distribution to try as the `review` floor.
GRID = tuple(index / 20 for index in range(1, 20))  # 0.05 … 0.95


@dataclass(frozen=True)
class Point:
    """One working point of one artifact."""

    threshold: float
    silence: Counts
    top1: Counts
    top3: Counts
    top1_all: Counts
    ece: float
    answered: int

    @property
    def silence_rate(self) -> float:
        return self.silence.rate

    @property
    def top1_rate(self) -> float:
        return self.top1.rate


def best_at_silence(points: list[Point], min_silence_rows: int) -> Point | None:
    """The most talkative point that still keeps ``min_silence_rows`` honest refusals.

    Among the points that meet the constraint the one with the most correct codes
    wins; ties go to the lower threshold, i.e. to the one that answers more queries,
    because a correct code is worth more than a shorter answer.
    """
    eligible = [point for point in points if point.silence.correct >= min_silence_rows]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda point: (point.top1.correct, point.top1_all.correct, -point.threshold),
    )


def throughput_gain(reference: Point | None, candidate: Point | None) -> int | None:
    """Correct codes gained at the same silence level (``None`` if not comparable)."""
    if reference is None or candidate is None:
        return None
    return candidate.top1.correct - reference.top1.correct


def render_table(series: dict[str, list[Point]], targets: tuple[int, ...]) -> str:
    """Fixed-width table: rows are silence levels, columns are artifacts."""
    names = list(series)
    header = f"  {'тишина ≥':>9s} " + " ".join(f"{name:>26s}" for name in names)
    lines = [header, "  " + "-" * (len(header) - 2)]
    for target in targets:
        cells = []
        for name in names:
            point = best_at_silence(series[name], target)
            if point is None:
                cells.append(f"{'—':>26s}")
                continue
            cells.append(
                f"{point.top1_rate:5.1f}% ({point.top1.correct:3d}/{point.top1.total}) "
                f"{point.threshold:5.3f}"
            )
        lines.append(f"  {target:9d} " + " ".join(cells))
    lines.append("")
    lines.append(
        "  cell = top-1 с кодом (строк) и порог review, при котором тишина не ниже указанной"
    )
    return "\n".join(lines)


def render_detail(series: dict[str, list[Point]], owned: dict[str, float]) -> str:
    """Full picture of each artifact at its own operating point."""
    lines = [
        f"  {'артефакт':12s} {'порог':>6s} {'тишина':>16s} {'top-1':>16s} "
        f"{'top-3':>16s} {'top-1 всё':>16s} {'ECE':>6s} {'ответов':>8s}",
        "  " + "-" * 104,
    ]
    for name, points in series.items():
        own = owned[name]
        point = min(points, key=lambda item: abs(item.threshold - own))
        lines.append(
            f"  {name:12s} {point.threshold:6.3f} "
            f"{point.silence_rate:6.1f}% ({point.silence.correct:3d}/{point.silence.total}) "
            f"{point.top1_rate:9.1f}% ({point.top1.correct:3d}/{point.top1.total}) "
            f"{point.top3.rate:9.1f}% ({point.top3.correct:3d}/{point.top3.total}) "
            f"{point.top1_all.rate:9.1f}% ({point.top1_all.correct:3d}/{point.top1_all.total}) "
            f"{point.ece:6.3f} {point.answered:8d}"
        )
    return "\n".join(lines)


def _quantile_thresholds(confidences: list[float], fractions: tuple[float, ...]) -> list[float]:
    """Thresholds spread over the artifact's own confidence distribution."""
    if not confidences:
        return []
    ordered = sorted(confidences)
    last = len(ordered) - 1
    return sorted({round(ordered[min(last, int(fraction * last))], 4) for fraction in fractions})


def _measure_series(label: str, artifact: Path, dense: bool) -> tuple[list[Point], float]:
    """Scan the artifact over its own working points, from talkative to silent."""
    from app.services.classifier import CatalogClassifier

    from eval.run_eval import (
        LABELED,
        _classify_rows,
        _set_review_threshold,
        load_labeled,
        metric_counts,
        summarize,
    )
    from ml.dense import DenseIndex
    from ml.reranker import LogisticReranker

    rows = load_labeled(LABELED)
    catalog = ROOT / "data" / "data.csv"
    reranker = LogisticReranker.load(artifact)
    probe = CatalogClassifier(catalog, reranker=reranker, dense=None)
    index = DenseIndex.build([row["name"] for row in probe.index]) if dense else None
    classifier = CatalogClassifier(catalog, reranker=reranker, dense=index)
    own = classifier._reranker.confidence_model.threshold_review

    # One pass over the rows to learn the artifact's own confidence distribution: only
    # those values can change an answer, so they are the only sensible thresholds.
    confidences = [classifier.classify(row["query"].strip()).confidence for row in rows]
    thresholds = sorted(set(_quantile_thresholds(confidences, GRID)) | {round(own, 4)})

    points: list[Point] = []
    for threshold in thresholds:
        _set_review_threshold(classifier, threshold)
        report = {"results": _classify_rows(classifier, rows)}
        counts = metric_counts(report)
        summary = summarize(report)
        points.append(
            Point(
                threshold=threshold,
                silence=counts["correct_silence"],
                top1=counts["top1_expected"],
                top3=counts["top3_expected"],
                top1_all=counts["top1_all"],
                ece=summary["ece"],
                answered=len(rows) - summary["status_distribution"].get("not_found", 0),
            )
        )
    _set_review_threshold(classifier, own)
    print(f"  {label}: {len(points)} точек, свой порог review={own:.3f}")
    return points, own


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silence", type=int, nargs="+", default=list(DEFAULT_SILENCE))
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="ИМЯ=ПУТЬ[:dense]",
        help="дополнительный артефакт для сравнения",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="показать полные метрики каждого артефакта на его собственном пороге",
    )
    args = parser.parse_args(argv)

    series: dict[str, list[Point]] = {}
    owned: dict[str, float] = {}
    print("Свип порога review по каждому артефакту:")
    presets = [
        ("лексика", ROOT / "ml/artifacts/reranker.json", False),
        ("вектор", ROOT / "ml/artifacts/reranker.tfidf.json", True),
    ]
    for spec in args.extra:
        name, _, target = spec.partition("=")
        path, _, flag = target.partition(":")
        presets.append((name or path, Path(path), flag == "dense"))
    for name, path, dense in presets:
        series[name], owned[name] = _measure_series(name, path, dense)

    print()
    print(render_table(series, tuple(args.silence)))
    if args.detail:
        print()
        print("Своя рабочая точка каждого артефакта:")
        print(render_detail(series, owned))
    return 0


if __name__ == "__main__":
    sys.exit(main())
