"""Build the local model artifacts used by the catalog/ML backends.

Produces two small, reviewable artifacts in ``models/`` (weights are generated,
never committed — see .gitignore):

* ``models/tfidf_embedder.json`` — document frequencies for the offline hashing
  TF-IDF embedder, fitted on the current PP-1875 catalog;
* ``models/reranker.json`` — calibrated logistic reranker weights plus the
  ``ok``/``review`` decision thresholds used in API responses.

With ``--dense`` the reranker is trained *with vector-search features* for the
active embedder (``EMBEDDER_BACKEND``; auto = transformer -> spaCy vectors -> tfidf)
and saved as ``reranker.<embedder-tag>.json``. Cosine scales differ between
embedders, so every embedder gets its own artifact; the API picks the one that
matches the embedder it actually loaded.

Usage:
    python scripts/build_model.py [--models-dir models] [--seed 1875]
    python scripts/build_model.py --dense --models-dir ml/artifacts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.dense_gate import Counts  # noqa: E402
from ml.catalog_loader import item_to_text, load_catalog  # noqa: E402
from ml.embedder import TFIDF_ARTIFACT_NAME, TfidfEncoder  # noqa: E402
from ml.reranker import RERANKER_ARTIFACT_NAME, reranker_artifact_name  # noqa: E402
from ml.train_reranker import train  # noqa: E402


def _measure(classifier) -> tuple[dict, dict[str, Counts]]:
    """Full summary of the labeled set plus the raw counts the gate decides on."""
    from eval.dense_gate import Counts
    from eval.run_eval import (
        LABELED,
        _classify_rows,
        evaluate_exact_names,
        load_exact_names,
        load_labeled,
        summarize,
    )

    results = _classify_rows(classifier, load_labeled(LABELED))
    exact = evaluate_exact_names(classifier, load_exact_names(ROOT / "data/data.csv"))
    summary = summarize({"results": results, "exact": exact})

    with_expected = [r for r in results if r["expected"] != "(not_found)"]
    without_expected = [r for r in results if r["expected"] == "(not_found)"]
    confident = [r for r in results if r["status"] == "ok"]
    summary["ok_precision"] = (
        sum(1 for r in confident if r["correct_top1"]) / len(confident) if confident else 1.0
    )
    counts = {
        "top1_expected": Counts(
            sum(1 for r in with_expected if r["correct_top1"]), len(with_expected)
        ),
        "top3_expected": Counts(
            sum(1 for r in with_expected if r["correct_top3"]), len(with_expected)
        ),
        "top1_all": Counts(sum(1 for r in results if r["correct_top1"]), len(results)),
        "correct_silence": Counts(
            sum(1 for r in without_expected if r["correct_top1"]), len(without_expected)
        ),
        "exact_top1": Counts(sum(1 for r in exact if r["correct_top1"]), len(exact)),
        "ok_precision": Counts(
            sum(1 for r in confident if r["correct_top1"]), len(confident)
        ),
    }
    return summary, counts


def _score(classifier) -> dict:
    """Headline metrics of a classifier on eval/labeled_examples.csv."""
    return _measure(classifier)[0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lexical_baseline(
    models_dir: Path, seed: int, max_queries: int, baseline_artifact: Path | None
) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """The baseline the candidate is compared against.

    It must be produced by *the same code* as the candidate: artifacts are not
    reproducible across training-code changes, and the shipped ``reranker.json``
    predates the current trainer (it stores 29277 pairs and thresholds
    0.621/0.097, a fresh build stores 27858 pairs and 0.306/0.227). Comparing a
    fresh dense artifact with that file measured a different working point, not
    the contribution of the vector features. So the baseline is retrained here
    unless an explicit artifact is supplied.
    """
    if baseline_artifact is not None:
        if not baseline_artifact.is_file():
            raise SystemExit(f"Нет файла базового артефакта: {baseline_artifact}")
        print(f"  gate: база — {baseline_artifact} (внешний артефакт)")
        return baseline_artifact, None
    directory = tempfile.TemporaryDirectory(prefix="dora-lexical-baseline-")
    path = Path(directory.name) / RERANKER_ARTIFACT_NAME
    print("  gate: база — лексический реранкер, обученный тем же кодом (не отгруженный артефакт)")
    train(path, max_queries=max_queries, seed=seed, excluded_queries=benchmark_queries())
    return path, directory


def _measure_seed_pair(
    catalog: Path, seed: int, models_dir: Path, max_queries: int, *, reuse_candidate: Path | None
) -> tuple[dict, dict, float, float]:
    """Train (or load) both artifacts for one seed and measure them.

    Returns ``(base_counts, new_counts, base_ece, new_ece)``. With
    ``reuse_candidate`` the already-trained dense artifact is measured instead of
    retraining it (the primary seed of the build), and the lexical baseline is
    trained here; otherwise both are trained in a scratch directory.
    """
    from app.services.classifier import CatalogClassifier

    from ml.dense import DenseIndex
    from ml.embedder import encoder_tag
    from ml.reranker import LogisticReranker

    scratch = tempfile.TemporaryDirectory(prefix=f"dora-gate-seed-{seed}-")
    directory = Path(scratch.name)
    try:
        baseline = directory / RERANKER_ARTIFACT_NAME
        train(baseline, max_queries=max_queries, seed=seed, excluded_queries=benchmark_queries())
        candidate = reuse_candidate
        if candidate is None:
            candidate = directory / reranker_artifact_name(encoder_tag())
            train(
                candidate,
                max_queries=max_queries,
                seed=seed,
                use_dense=True,
                excluded_queries=benchmark_queries(),
            )
        probe = CatalogClassifier(catalog, reranker=LogisticReranker.load(baseline), dense=None)
        dense_index = DenseIndex.build([row["name"] for row in probe.index])
        lexical = CatalogClassifier(catalog, reranker=probe._reranker, dense=None)
        vector = CatalogClassifier(
            catalog, reranker=LogisticReranker.load(candidate), dense=dense_index
        )
        base_summary, base_counts = _measure(lexical)
        new_summary, new_counts = _measure(vector)
        return base_counts, new_counts, base_summary["ece"], new_summary["ece"]
    finally:
        scratch.cleanup()


def gate_dense(
    models_dir: Path,
    artifact: Path,
    *,
    seed: int = 1875,
    max_queries: int = 400,
    baseline_artifact: Path | None = None,
    report: Path | None = None,
    seeds: list[int] | None = None,
) -> bool:
    """Keep a vector-feature reranker only if it earns the right to serve decisions.

    Vector features are learned from synthetic queries, so a weak embedder can make
    real-world answers *worse*. The artifact is scored against a lexical reranker
    trained in the same run (see ``_lexical_baseline``) with
    :func:`eval.dense_gate.evaluate_gate` — a pure function over row counts, whose
    docstring spells out the policy. On rejection the artifact is deleted, so the
    API keeps using the lexical one; either way the full comparison is printed and
    written to ``<models_dir>/dense-gate.json``.
    """
    from eval.dense_gate import evaluate_gate, pool_counts

    sys.path.insert(0, str(ROOT / "api"))
    from app.services.classifier import CatalogClassifier

    from ml.dense import DenseIndex
    from ml.reranker import LogisticReranker

    catalog = ROOT / "data" / "data.csv"
    extra_seeds = [value for value in (seeds or []) if value != seed]
    pools_base: list[dict] = []
    pools_new: list[dict] = []
    base_eces: list[float] = []
    new_eces: list[float] = []

    baseline_path, scratch = _lexical_baseline(models_dir, seed, max_queries, baseline_artifact)
    try:
        lexical = CatalogClassifier(
            catalog, reranker=LogisticReranker.load(baseline_path), dense=None
        )
        probe = CatalogClassifier(catalog, reranker=lexical._reranker, dense=None)
        dense_index = DenseIndex.build([row["name"] for row in probe.index])
        vector = CatalogClassifier(
            catalog, reranker=LogisticReranker.load(artifact), dense=dense_index
        )
        base_summary, base_counts = _measure(lexical)
        new_summary, new_counts = _measure(vector)
        pools_base.append(base_counts)
        pools_new.append(new_counts)
        base_eces.append(base_summary["ece"])
        new_eces.append(new_summary["ece"])
        for extra in extra_seeds:
            print(f"  gate: дополнительный прогон, seed={extra} (обучение обоих артефактов)")
            extra_base, extra_new, extra_base_ece, extra_new_ece = _measure_seed_pair(
                catalog, extra, models_dir, max_queries, reuse_candidate=None
            )
            pools_base.append(extra_base)
            pools_new.append(extra_new)
            base_eces.append(extra_base_ece)
            new_eces.append(extra_new_ece)
        runs = len(pools_base)
        base_counts = pool_counts(pools_base) if runs > 1 else base_counts
        new_counts = pool_counts(pools_new) if runs > 1 else new_counts
        base_ece = sum(base_eces) / runs
        new_ece = sum(new_eces) / runs
        verdict = evaluate_gate(
            base_counts, new_counts, base_ece=base_ece, new_ece=new_ece, runs=runs
        )

        print(verdict.table())
        payload = {
            "artifact": artifact.name,
            "artifact_sha256": _sha256(artifact),
            "embedder_tag": getattr(dense_index, "tag", None),
            "catalog_rows": len(dense_index.names),
            "baseline": {
                "artifact": baseline_path.name,
                "sha256": _sha256(baseline_path),
                "trained_in_run": scratch is not None,
                "path": None if scratch is not None else str(baseline_path),
            },
            "labeled_rows": max(pool["top1_all"].total for pool in pools_base),
            "seeds": [seed, *extra_seeds],
            "decision_basis": "pooled" if runs > 1 else "single_run",
            **verdict.as_dict(),
        }
        destination = report or models_dir / "dense-gate.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  gate: отчёт {destination}")

        if not verdict.keep:
            artifact.unlink()
            print("  gate: артефакт удалён — векторные признаки не прошли допуск")
            return False
        print(
            "  gate: артефакт оставлен — "
            f"{json.dumps(verdict.headline_gains, ensure_ascii=False)}"
        )
        return True
    finally:
        if scratch is not None:
            scratch.cleanup()


def benchmark_queries() -> set[str]:
    import csv
    with (ROOT / "eval/labeled_examples.csv").open(encoding="utf-8") as handle:
        return {row["query"] for row in csv.DictReader(handle, delimiter=";")}


def build_dense(
    models_dir: Path,
    seed: int,
    max_queries: int,
    gate: bool = False,
    baseline_artifact: Path | None = None,
    report: Path | None = None,
    gate_seeds: list[int] | None = None,
) -> None:
    from ml.embedder import encoder_tag

    models_dir.mkdir(parents=True, exist_ok=True)
    tag = encoder_tag()
    print(f"Training the reranker with vector-search features for embedder {tag!r}...")
    path = models_dir / reranker_artifact_name(tag)
    train(
        path,
        max_queries=max_queries,
        seed=seed,
        use_dense=True,
        excluded_queries=benchmark_queries(),
    )
    if gate:
        kept = gate_dense(
            models_dir,
            path,
            seed=seed,
            max_queries=max_queries,
            baseline_artifact=baseline_artifact,
            report=report,
            seeds=gate_seeds,
        )
        if not kept:
            return
    print(f"Done: {path}")


def build(models_dir: Path, seed: int, max_queries: int) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Fitting TF-IDF embedder on the catalog...")
    catalog = load_catalog()
    encoder = TfidfEncoder()
    encoder.fit([item_to_text(item) for item in catalog])
    embedder_path = models_dir / TFIDF_ARTIFACT_NAME
    encoder.save(embedder_path)
    print(f"  documents={encoder.doc_count} dim={encoder.dim} -> {embedder_path}")

    print("Training and calibrating the reranker...")
    reranker_path = models_dir / RERANKER_ARTIFACT_NAME
    train(reranker_path, max_queries=max_queries, seed=seed, excluded_queries=benchmark_queries())

    print("Done. Point the API at the artifacts with MODELS_DIR if it is not ./models.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--seed", type=int, default=1875)
    parser.add_argument("--max-queries", type=int, default=400)
    parser.add_argument("--dense", action="store_true", help="train with vector-search features")
    parser.add_argument(
        "--gate",
        action="store_true",
        help="with --dense: drop the artifact unless it beats lexical scoring on the labeled set",
    )
    parser.add_argument(
        "--gate-seeds",
        default="",
        help=(
            "with --gate: comma-separated seeds whose runs are pooled before the verdict "
            "(e.g. 1875,1876,1877). One run measures one split of the training data, and "
            "the silence guard is zero-tolerance — pooling keeps the policy but stops the "
            "verdict from being a coin flip"
        ),
    )
    parser.add_argument(
        "--baseline-artifact",
        type=Path,
        default=None,
        help=(
            "with --gate: compare against this lexical artifact instead of retraining one "
            "(it must be produced by the same code — shipped artifacts are not reproducible "
            "across trainer changes)"
        ),
    )
    parser.add_argument(
        "--gate-report",
        type=Path,
        default=None,
        help="with --gate: where to write the JSON report (default <models-dir>/dense-gate.json)",
    )
    args = parser.parse_args()
    if args.dense:
        gate_seeds = [int(value) for value in args.gate_seeds.replace(" ", "").split(",") if value]
        if gate_seeds and not args.gate:
            parser.error("--gate-seeds имеет смысл только вместе с --gate")
        build_dense(
            args.models_dir,
            args.seed,
            args.max_queries,
            gate=args.gate,
            baseline_artifact=args.baseline_artifact,
            report=args.gate_report,
            gate_seeds=gate_seeds or None,
        )
    else:
        build(args.models_dir, args.seed, args.max_queries)


if __name__ == "__main__":
    main()
