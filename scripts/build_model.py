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


def gate_dense(models_dir: Path, artifact: Path) -> dict:
    """Decide whether the vector-feature reranker ships.

    The comparison happens *at equal silence* (see ``ml.gate``):
    both artifacts are swept over the ``review`` threshold, the
    lexical artifact's shipped operating point defines the target
    silence, and the dense artifact must beat it by
    ``GATE_MARGIN_PP`` points of top-1 there. A rejected artifact
    is kept on disk with the decision recorded in the gate
    registry — the loader then ignores it — instead of being
    deleted, so every run leaves auditable evidence.
    """
    sys.path.insert(0, str(ROOT / "api"))
    from app.services.classifier import CatalogClassifier

    from ml.dense import DenseIndex
    from ml.gate import gate_decision, record_decision
    from ml.reranker import LogisticReranker

    catalog = ROOT / "data" / "data.csv"
    # The base is the artifact the API would serve without vectors:
    # MODELS_DIR when rebuilt there, the shipped one otherwise.
    from ml.reranker import reranker_artifact_path

    lexical = CatalogClassifier(
        catalog, reranker=LogisticReranker.load(reranker_artifact_path()), dense=None
    )
    probe = CatalogClassifier(catalog, reranker=lexical._reranker, dense=None)
    dense_index = DenseIndex.build([row["name"] for row in probe.index])
    vector = CatalogClassifier(catalog, reranker=LogisticReranker.load(artifact), dense=dense_index)

    from eval.run_eval import LABELED, load_labeled

    decision = gate_decision(lexical, vector, load_labeled(LABELED))
    if decision["decision"] == "kept":
        # Ship the artifact at the threshold that matches the
        # lexical operating silence, not at the raw training floor.
        _apply_threshold(artifact, vector)
    record_decision(models_dir, _embedder_tag(), decision)
    print(
        f"  gate: lexical top-1 {decision['base']['top1_expected']}% / "
        f"тишина {decision['base']['silence']}%  |  "
        f"dense top-1 {decision['candidate']['top1_expected']}% / "
        f"тишина {decision['candidate']['silence']}% (порог "
        f"{decision['candidate']['threshold']})"
    )
    print(f"  gate: {decision['decision'].upper()} — {decision['reason']}")
    return decision


def _apply_threshold(artifact: Path, classifier) -> None:
    """Write the gate-matched review threshold into the kept artifact."""
    from ml.reranker import LogisticReranker

    model = classifier._reranker.confidence_model
    reranker = LogisticReranker.load(artifact)
    reranker.confidence_model.threshold_review = round(
        model.threshold_review, 4
    )
    reranker.confidence_model.threshold_ok = round(
        max(reranker.confidence_model.threshold_ok, model.threshold_review + 0.05),
        4,
    )
    reranker.save(artifact)


def _embedder_tag() -> str:
    from ml.embedder import encoder_tag

    return encoder_tag()


def build_dense(models_dir: Path, seed: int, max_queries: int, gate: bool = False) -> None:
    tag = _embedder_tag()
    print(f"Training the reranker with vector-search features for embedder {tag!r}...")
    path = models_dir / reranker_artifact_name(tag)
    train(path, max_queries=max_queries, seed=seed, use_dense=True)
    if gate:
        gate_dense(models_dir, path)
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
    train(reranker_path, max_queries=max_queries, seed=seed)

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
