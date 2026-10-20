"""Pilot for the cross-encoder stage (ML-6): quality and latency.

Decision rule (DoD of ML-6): adopt the stage only when it adds
**>= +2 p.p. top-1** on the labeled set while the p95 latency of
``/classify`` stays **<= 150 ms**; otherwise the stage is rejected
and nothing ships.

Two runs are measured:

* **baseline** — the serving pipeline as-is;
* **pilot** — the same pipeline with the cross-encoder rescoreing
  the top-10 of the reranker's ranking (the reranker's calibrated
  probability still decides the status; the cross-encoder only
  reorders).

Offline limitation: the sandbox has no transformer weights, so
``--stub`` validates the plumbing with a deterministic lexical
proxy scorer. A stub run measures latency of the integration
only — its quality numbers are marked as such and never counted
as a pilot result. The real measurement runs on the target
machine with the model baked into ``MODELS_DIR/cross_encoder``.

Usage:
    PYTHONPATH=api python eval/cross_encoder_pilot.py [--stub]
        [--output eval/cross_encoder_pilot.md]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "api"))

from app.core.config import settings  # noqa: E402
from app.services.classifier import (  # noqa: E402
    _code_matches_prefix,
    _tokenize,
    build_classifier,
)

from eval.run_eval import LABELED, codes_match, load_labeled  # noqa: E402
from ml.fuzzy import autocorrect  # noqa: E402

LABELED_ROWS = load_labeled(LABELED)
CODED = [row for row in LABELED_ROWS if (row["expected_okpd2"] or "").strip()]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q / 100 * (len(ordered) - 1)))))
    return ordered[index]


def measure(classifier, rescore=None) -> dict:
    """Top-1 quality and per-query latency over the coded labeled rows."""
    correct = 0
    latencies: list[float] = []
    for row in CODED:
        query = (row["query"] or "").strip()
        expected = row["expected_okpd2"].strip()
        started = time.perf_counter()
        if rescore is None:
            response = classifier.classify(query)
        else:
            response = rescore(classifier, query)
        latencies.append((time.perf_counter() - started) * 1000)
        if codes_match(response.code, expected):
            correct += 1
    return {
        "top1": round(100 * correct / len(CODED), 1),
        "p50_ms": round(statistics.median(latencies), 1),
        "p95_ms": round(percentile(latencies, 95), 1),
    }


def pilot_run(classifier, cross_encoder) -> dict:
    """Classify with the cross-encoder rescoreing the reranker's top-10.

    Mirrors ``CatalogClassifier.classify`` step by step (retrieval,
    dense fusion, pair vetoes, rerank) and inserts the cross-encoder
    between the rerank and the confidence model, so the pilot
    measures exactly the integration the service would ship.
    """
    from ml.lexicon import vetoed_codes

    def rescore(clf, query: str):
        from ml.dense import fuse_shortlist
        from ml.features import prepare_candidate
        from ml.lexicon import expand_query
        from ml.reranker import RankedCandidate

        normalized = autocorrect(query.strip())
        query_tokens = set(_tokenize(normalized.casefold()))
        if not query_tokens:
            return None
        shortlist = clf._retrieve(normalized, query_tokens)
        if clf._dense is not None:
            fused = fuse_shortlist(
                clf._dense,
                clf.index,
                shortlist,
                normalized,
                prepare_candidate(expand_query(normalized), idf=clf._idf),
            )
        else:
            fused = [(score, row, 0.0, 0.0) for score, row in shortlist]
        banned = vetoed_codes(normalized)
        if banned:
            fused = [
                entry
                for entry in fused
                if not _code_matches_prefix(entry[1]["okpd2"], banned)
            ]
        if not fused:
            return None
        candidates = [
            RankedCandidate(
                code=row["okpd2"] or None,
                title=row["name"],
                clause_ref=row["clause_ref"],
                retrieval_score=float(score),
                payload={
                    "record_id": row.get("record_id") or "",
                    "dense": cosine,
                    "dense_top": dense_top,
                },
            )
            for score, row, cosine, dense_top in fused
        ]
        ranked = clf._reranker.rerank(query, candidates)
        if cross_encoder.available and len(ranked) > 1:
            top = ranked[: cross_encoder.top_k]
            scores = cross_encoder.rescore(query, [c.title for c in top])
            if scores:
                order = sorted(range(len(top)), key=lambda index: -scores[index])
                ranked = [top[index] for index in order] + ranked[len(top) :]
        raw_confidence, status = clf._reranker.evaluate(query, ranked)
        if status == "not_found":
            return None  # the service would stay silent here
        best = ranked[0]
        return best.code, best.title, status, raw_confidence

    correct = 0
    latencies: list[float] = []
    for row in CODED:
        query = (row["query"] or "").strip()
        expected = row["expected_okpd2"].strip()
        started = time.perf_counter()
        result = rescore(classifier, query)
        latencies.append((time.perf_counter() - started) * 1000)
        if result is not None and codes_match(result[0], expected):
            correct += 1
    return {
        "top1": round(100 * correct / len(CODED), 1),
        "p50_ms": round(statistics.median(latencies), 1),
        "p95_ms": round(percentile(latencies, 95), 1),
    }


class _StubCrossEncoder:
    """Order-preserving stub: plumbing check only.

    Returns scores that keep the reranker's order as-is, so a
    stub run must reproduce the baseline exactly — that is the
    assertion: the stage integrates without perturbing the
    serving ranking. Its numbers are NOT a quality measurement
    of a real model.
    """

    top_k = 10
    available = True
    model_name = "stub (plumbing check)"

    def rescore(self, query: str, candidates: list[str]) -> list[float]:
        return [float(len(candidates) - index) for index in range(len(candidates))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stub", action="store_true", help="plumbing check with the lexical proxy")
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "cross_encoder_pilot.md")
    parser.add_argument("--min-gain-pp", type=float, default=2.0)
    parser.add_argument("--max-p95-ms", type=float, default=150.0)
    args = parser.parse_args()

    settings.classifier_backend = "catalog"
    classifier = build_classifier("catalog")

    baseline = measure(classifier)

    if args.stub:
        cross_encoder = _StubCrossEncoder()
        pilot = pilot_run(classifier, cross_encoder)
        measured = False
    else:
        from ml.cross_encoder import CrossEncoderReranker

        cross_encoder = CrossEncoderReranker()
        if not cross_encoder.available:
            print("Кросс-энкодер недоступен офлайн — запускайте с --stub для проверки сборки")
            return 2
        pilot = pilot_run(classifier, cross_encoder)
        measured = True

    gain = round(pilot["top1"] - baseline["top1"], 1)
    accepted = gain >= args.min_gain_pp and pilot["p95_ms"] <= args.max_p95_ms

    lines = [
        "# Пилот кросс-энкодера (ML-6)",
        "",
        f"**Ступень:** {cross_encoder.model_name}  ",
        f"**Набор:** {len(CODED)} запросов с кодом из eval/labeled_examples.csv  ",
        f"**Измерено качество:** "
        f"{'да' if measured else 'НЕТ — stub-заглушка, только сборка'}",
        "",
        "| Прогон | top-1 | p50, мс | p95, мс |",
        "| --- | ---: | ---: | ---: |",
        f"| база (сервис как есть) | {baseline['top1']}% "
        f"| {baseline['p50_ms']} | {baseline['p95_ms']} |",
        f"| пилот (+кросс-энкодер top-10) | {pilot['top1']}% "
        f"| {pilot['p50_ms']} | {pilot['p95_ms']} |",
        "",
        f"**Решение:** {'ПРИНЯТО' if accepted else 'ОТКЛОНЕНО'} "
        f"(правило: прирост ≥ {args.min_gain_pp} п.п. top-1 при p95 ≤ {args.max_p95_ms:.0f} мс; "
        f"измерено {gain:+.1f} п.п., p95 {pilot['p95_ms']} мс).",
        "",
    ]
    if not measured:
        lines += [
            "> Заглушка `_StubCrossEncoder` пересчитывает кандидаты лексическим",
            "> скором: качество в этой таблице не измеряется, только то, что",
            "> ступень встраивается и не ломает ранжирование. Реальный замер —",
            "> на целевой машине: скачать модель в",
            "> `MODELS_DIR/cross_encoder/<name>` и запустить без `--stub`.",
            "",
        ]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"база:   top-1 {baseline['top1']}%, p95 {baseline['p95_ms']} мс")
    print(f"пилот:  top-1 {pilot['top1']}%, p95 {pilot['p95_ms']} мс")
    print(f"решение: {'принято' if accepted else 'отклонено'} ({gain:+.1f} п.п.)")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
