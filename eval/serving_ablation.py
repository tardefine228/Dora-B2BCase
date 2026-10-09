"""Serving ablation: how much do the train/serve differences of the catalog backend matter?

Training (``ml/train_reranker.build_pairs``) shortlists the top-12 *lexical* candidates
and never adds substring-fallback candidates. The API (``CatalogClassifier._retrieve``)
shortlists the top-25 and adds fallback candidates with score ``0.2 * matched / len``.
This script re-runs the catalog backend on the labeled set and on the synthetic
benchmark with each difference switched off, then reports how many answers change:

* ``base``        — the current API;
* ``no-fallback`` — substring fallback removed (training's candidate generation);
* ``shortlist-12``— shortlist size 12 (training's size), fallback kept.

The model is NOT retrained in any mode: the question is whether the reranker's
output depends on these choices at serving time. Results go to
``eval/experiments.md`` (эксп. 13). The script is informational and does not feed
the CI gate.

Usage:
    PYTHONPATH=api python eval/serving_ablation.py [--mode all|base|no-fallback|shortlist-12]
Requires the model artifacts (``python scripts/build_model.py``).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "api") not in sys.path:
    sys.path.insert(0, str(ROOT / "api"))
if str(ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(ROOT / "eval"))

import app.services.classifier as classifier_module  # noqa: E402
import run_eval as labeled_eval  # noqa: E402
import synthetic_eval  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.services.classifier import CatalogClassifier, build_classifier  # noqa: E402

from ml.catalog_loader import load_catalog  # noqa: E402
from ml.features import lexical_score_prepared, prepare_candidate  # noqa: E402
from ml.lexicon import expand_query  # noqa: E402

MODES = ("base", "no-fallback", "shortlist-12")
_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)


def _lexical_only_retrieve(self, query, query_tokens):
    """``CatalogClassifier._retrieve`` without the substring-fallback pass."""
    prepared_query = prepare_candidate(expand_query(query))
    scored = []
    for row in self.index:
        score = lexical_score_prepared(prepared_query, row["prepared"])
        if score > 0.0:
            scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[: classifier_module._RETRIEVAL_SHORTLIST]


def collect_cases() -> list[dict]:
    cases = []
    for row in labeled_eval.load_labeled(labeled_eval.LABELED):
        cases.append(
            {
                "src": "labeled",
                "query": row["query"].strip(),
                "expected": (row["expected_okpd2"] or "").strip(),
            }
        )
    for query, expected in synthetic_eval.build_cases(load_catalog(), step=4, variants=4):
        cases.append({"src": "synthetic", "query": query, "expected": expected})
    return cases


def run_mode(mode: str, cases: list[dict]) -> list[dict]:
    original_retrieve = CatalogClassifier._retrieve
    original_shortlist = classifier_module._RETRIEVAL_SHORTLIST
    try:
        if mode == "no-fallback":
            CatalogClassifier._retrieve = _lexical_only_retrieve
        elif mode == "shortlist-12":
            CatalogClassifier._retrieve = _lexical_only_retrieve
            classifier_module._RETRIEVAL_SHORTLIST = 12
        settings.classifier_backend = "catalog"
        classifier = build_classifier("catalog")
        outcomes = []
        for case in cases:
            response = classifier.classify(case["query"])
            if case["expected"]:
                correct = labeled_eval.codes_match(response.code, case["expected"])
            else:
                correct = response.status == "not_found"
            outcomes.append(
                {
                    **case,
                    "status": response.status,
                    "code": response.code or "(not_found)",
                    "raw": round(response.confidence_raw or 0.0, 4),
                    "correct": correct,
                }
            )
        return outcomes
    finally:
        CatalogClassifier._retrieve = original_retrieve
        classifier_module._RETRIEVAL_SHORTLIST = original_shortlist


def summarize(outcomes: list[dict]) -> dict:
    result = {}
    for src in ("labeled", "synthetic"):
        rows = [row for row in outcomes if row["src"] == src]
        with_code = [row for row in rows if row["expected"]]
        silence = [row for row in rows if not row["expected"]]
        result[src] = {
            "top1": round(100 * sum(r["correct"] for r in with_code) / max(len(with_code), 1), 1),
            "silence": round(100 * sum(r["correct"] for r in silence) / max(len(silence), 1), 1),
            "statuses": dict(Counter(r["status"] for r in rows)),
        }
    return result


def diff(base: list[dict], other: list[dict]) -> dict:
    changed_status = changed_code = changed_raw = 0
    for left, right in zip(base, other, strict=True):
        changed_status += left["status"] != right["status"]
        changed_code += left["code"] != right["code"]
        changed_raw += left["raw"] != right["raw"]
    return {
        "rows": len(base),
        "status_changed": changed_status,
        "code_changed": changed_code,
        "raw_confidence_changed": changed_raw,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="all", choices=["all", *MODES])
    args = parser.parse_args()

    cases = collect_cases()
    if args.mode == "all":
        modes = MODES
    else:
        modes = ("base",) if args.mode == "base" else ("base", args.mode)
    outcomes = {mode: run_mode(mode, cases) for mode in modes}

    for mode in modes:
        print(f"[{mode}] {summarize(outcomes[mode])}")
    if "base" in outcomes:
        for mode in modes:
            if mode != "base":
                print(f"diff base vs {mode}: {diff(outcomes['base'], outcomes[mode])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
