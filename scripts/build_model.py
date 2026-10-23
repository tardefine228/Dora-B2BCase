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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.catalog_loader import item_to_text, load_catalog  # noqa: E402
from ml.embedder import TFIDF_ARTIFACT_NAME, TfidfEncoder  # noqa: E402
from ml.reranker import RERANKER_ARTIFACT_NAME, reranker_artifact_name  # noqa: E402
from ml.train_reranker import train  # noqa: E402


def _score(classifier) -> dict:
    """Headline metrics of a classifier on eval/labeled_examples.csv."""
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
    confident = [r for r in results if r["status"] == "ok"]
    summary["ok_precision"] = (
        sum(r["correct_top1"] for r in confident) / len(confident) if confident else 1.0
    )
    return summary


def gate_dense(models_dir: Path, artifact: Path) -> bool:
    """Keep a vector-feature reranker only if it is not worse than lexical-only scoring.

    Vector features are learned from synthetic queries; a weak embedder (static word
    vectors) can make real-world answers *worse* (measured: spaCy vectors cost 2.4 p.p.
    top-1). So the artifact is evaluated against the lexical reranker on the labeled set
    and deleted when it loses; the API then uses vector search only to widen the
    candidate list. Returns True when the artifact is kept.
    """
    sys.path.insert(0, str(ROOT / "api"))
    from app.services.classifier import CatalogClassifier

    from ml.dense import DenseIndex
    from ml.reranker import LogisticReranker

    catalog = ROOT / "data" / "data.csv"
    lexical = CatalogClassifier(
        catalog, reranker=LogisticReranker.load(models_dir / RERANKER_ARTIFACT_NAME), dense=None
    )
    probe = CatalogClassifier(catalog, reranker=lexical._reranker, dense=None)
    dense_index = DenseIndex.build([row["name"] for row in probe.index])
    vector = CatalogClassifier(catalog, reranker=LogisticReranker.load(artifact), dense=dense_index)
    base, new = _score(lexical), _score(vector)
    print(
        f"  gate: lexical top-1 coded {base['top1_expected']}% / all {base['top1_all']}%  |  "
        f"vector top-1 coded {new['top1_expected']}% / all {new['top1_all']}%"
    )
    higher_is_better = ("top1_expected", "top1_all", "top3_expected",
                        "correct_silence", "exact_top1", "ok_precision")
    keep = all(new[key] >= base[key] for key in higher_is_better) and new["ece"] <= base["ece"]
    if not keep:
        artifact.unlink()
        print("  gate: vector features do not help on the labeled set -> artifact removed")
    return keep


def benchmark_queries() -> set[str]:
    import csv
    with (ROOT / "eval/labeled_examples.csv").open(encoding="utf-8") as handle:
        return {row["query"] for row in csv.DictReader(handle, delimiter=";")}


def build_dense(models_dir: Path, seed: int, max_queries: int, gate: bool = False) -> None:
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
    if gate and not gate_dense(models_dir, path):
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
    args = parser.parse_args()
    if args.dense:
        build_dense(args.models_dir, args.seed, args.max_queries, gate=args.gate)
    else:
        build(args.models_dir, args.seed, args.max_queries)


if __name__ == "__main__":
    main()
