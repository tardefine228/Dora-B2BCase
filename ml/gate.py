"""Quality gate for reranker artifacts trained with vector features.

The gate answers one question: *does the dense-feature artifact beat the
lexical one at the same level of silence?* Comparing two independent
metrics (top-1 on listed goods vs top-1 on everything, as the old gate
did) mixes two different operating points and rejects artifacts that are
strictly better on the product's actual frontier — the dense artifact
measured +1.9 p.p. top-1 at equal silence and was still deleted.

How it works (ML-1):

1. Both artifacts are evaluated on the labeled set across a grid of
   ``review`` thresholds — a *frontier* of ``(silence, top-1)`` points.
2. The lexical artifact's shipped threshold defines the operating
   silence; the dense artifact's threshold is moved to the same
   silence (linear interpolation between grid points when needed).
3. At equal silence the dense artifact must beat the lexical one by at
   least ``margin_pp`` points of top-1 to be kept.

The gate never deletes an artifact. The decision is recorded in a
registry (``dense_gate.json``) next to the artifact and appended to an
audit log (``gate_decisions.jsonl``); the reranker loader
(:func:`ml.reranker.load_reranker`) refuses a *rejected* artifact, so a
failed candidate simply stays on disk as evidence while the API keeps
serving the lexical one.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ml.config import ARTIFACTS_DIR, MODELS_DIR

logger = logging.getLogger("dora.ml.gate")

# Registry of the latest gate decision per embedder tag.
GATE_REGISTRY_NAME = "dense_gate.json"
# Append-only audit log of every gate run.
GATE_LOG_NAME = "gate_decisions.jsonl"

# Threshold grid for the frontier measurement. Silence is monotone in
# the review threshold, so a modest grid brackets any operating point.
FRONTIER_THRESHOLDS = (0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.21, 0.25, 0.30)

# How close (in percentage points of silence) the candidate must come to
# the base operating point before interpolation is attempted.
SILENCE_TOLERANCE_PP = 1.5

# Default required advantage of the dense artifact at equal silence.
DEFAULT_MARGIN_PP = 1.0


def frontier(
    classifier,
    rows: list[dict],
    thresholds,
) -> list[dict]:
    """Metrics of one artifact at every ``review`` threshold.

    Uses the same measurement helpers as ``eval/run_eval.py`` so the
    gate and CI score identically. The classifier's own threshold is
    restored afterwards.
    """
    from eval.run_eval import _classify_rows, summarize

    points: list[dict] = []
    original = _review_threshold(classifier)
    try:
        for threshold in thresholds:
            _set_review_threshold(classifier, float(threshold))
            summary = summarize({"results": _classify_rows(classifier, rows)})
            points.append(
                {
                    "threshold": round(float(threshold), 4),
                    "top1_expected": summary["top1_expected"],
                    "top1_all": summary["top1_all"],
                    "silence": summary["correct_silence"],
                    "top3_expected": summary["top3_expected"],
                }
            )
    finally:
        _set_review_threshold(classifier, original)
    return points


def _review_threshold(classifier) -> float | None:
    reranker = getattr(classifier, "_reranker", None)
    model = getattr(reranker, "confidence_model", None)
    return getattr(model, "threshold_review", None) if model else None


def _set_review_threshold(classifier, value: float) -> None:
    classifier._reranker.confidence_model.threshold_review = value


def _interpolate_threshold(
    points: list[dict], target_silence: float
) -> float | None:
    """Threshold whose silence is closest to ``target_silence``.

    Linear interpolation between the two grid points bracketing the
    target; ``None`` when the target is outside the measured range.
    """
    ordered = sorted(points, key=lambda point: point["silence"])
    silences = [point["silence"] for point in ordered]
    if target_silence < silences[0] or target_silence > silences[-1]:
        return None
    for lower, upper in zip(ordered, ordered[1:], strict=False):
        if lower["silence"] <= target_silence <= upper["silence"]:
            span = upper["silence"] - lower["silence"]
            if span <= 1e-9:
                return lower["threshold"]
            share = (target_silence - lower["silence"]) / span
            return lower["threshold"] + share * (
                upper["threshold"] - lower["threshold"]
            )
    return ordered[-1]["threshold"]


def _closest_point(points: list[dict], target_silence: float) -> dict:
    return min(points, key=lambda point: abs(point["silence"] - target_silence))


def gate_decision(
    base_classifier,
    candidate_classifier,
    rows: list[dict],
    *,
    margin_pp: float = DEFAULT_MARGIN_PP,
    thresholds=FRONTIER_THRESHOLDS,
) -> dict:
    """Compare two artifacts *at equal silence*; return the decision.

    The base is measured at its shipped threshold — the
    product's current operating point. The candidate is
    measured at *its own* calibrated threshold; when its
    silence there misses the base's, its threshold is
    moved to the base silence by linear interpolation
    over the swept frontier (and written back into the
    artifact when the candidate is kept). A candidate
    that cannot reach the base silence at all is
    rejected outright.
    """
    base_point = _operating_point(base_classifier, rows)
    candidate_point = _operating_point(candidate_classifier, rows)
    target_silence = base_point["silence"]

    silence_gap = abs(candidate_point["silence"] - target_silence)
    if silence_gap > SILENCE_TOLERANCE_PP:
        candidate_frontier = frontier(candidate_classifier, rows, thresholds)
        matched_threshold = _interpolate_threshold(
            candidate_frontier, target_silence
        )
        if matched_threshold is None:
            closest = _closest_point(candidate_frontier, target_silence)
            return _decide(
                base_point,
                closest,
                decision="rejected",
                reason=(
                    f"кандидат не достигает тишины базы {target_silence}% "
                    f"ни на одном пороге сетки"
                ),
                matched_silence=closest["silence"],
            )
        _set_review_threshold(candidate_classifier, matched_threshold)
        from eval.run_eval import _classify_rows, summarize

        summary = summarize(
            {"results": _classify_rows(candidate_classifier, rows)}
        )
        candidate_point = {
            "threshold": round(matched_threshold, 4),
            "top1_expected": summary["top1_expected"],
            "top1_all": summary["top1_all"],
            "silence": summary["correct_silence"],
            "top3_expected": summary["top3_expected"],
        }

    delta = candidate_point["top1_expected"] - base_point["top1_expected"]
    keep = (
        delta >= margin_pp
        and abs(candidate_point["silence"] - target_silence)
        <= SILENCE_TOLERANCE_PP
    )
    reason = (
        f"top-1 {candidate_point['top1_expected']}% против "
        f"{base_point['top1_expected']}% "
        f"при тишины {candidate_point['silence']}% против "
        f"{target_silence}% (разница {delta:+.1f} п.п., "
        f"запас {margin_pp:+.1f} п.п.)"
    )
    return _decide(
        base_point,
        candidate_point,
        decision="kept" if keep else "rejected",
        reason=reason,
        matched_silence=target_silence,
    )


def _operating_point(classifier, rows: list[dict]) -> dict:
    """Metrics of an artifact at its shipped threshold."""
    from eval.run_eval import _classify_rows, summarize

    summary = summarize({"results": _classify_rows(classifier, rows)})
    return {
        "threshold": round(_review_threshold(classifier) or 0.0, 4),
        "top1_expected": summary["top1_expected"],
        "top1_all": summary["top1_all"],
        "silence": summary["correct_silence"],
        "top3_expected": summary["top3_expected"],
    }


def _decide(
    base_point: dict, candidate_point: dict, *, decision: str, reason: str, matched_silence: float
) -> dict:
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decision": decision,
        "reason": reason,
        "matched_silence": matched_silence,
        "base": base_point,
        "candidate": candidate_point,
    }


# --- Decision registry ----------------------------------------------------


def _registry_paths() -> list[Path]:
    return [MODELS_DIR / GATE_REGISTRY_NAME, ARTIFACTS_DIR / GATE_REGISTRY_NAME]


def load_decisions() -> dict[str, dict]:
    """Latest gate decision per embedder tag (first registry wins)."""
    for path in _registry_paths():
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return dict(payload.get("decisions", {}))
            except (OSError, ValueError):
                logger.warning("Повреждённый реестр гейта %s — игнорирую", path)
    return {}


def decision_for(tag: str | None) -> dict | None:
    if not tag:
        return None
    return load_decisions().get(tag)


def artifact_allowed(tag: str | None) -> bool:
    """False only when the gate explicitly rejected this embedder's artifact."""
    record = decision_for(tag)
    if record is None:
        return True
    return record.get("decision") != "rejected"


def record_decision(models_dir: Path, tag: str, decision: dict) -> Path:
    """Persist the decision: update the registry and append to the log."""
    models_dir.mkdir(parents=True, exist_ok=True)
    registry_path = models_dir / GATE_REGISTRY_NAME
    registry: dict = {}
    if registry_path.is_file():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8")).get(
                "decisions", {}
            )
        except (OSError, ValueError):
            logger.warning("Повреждённый реестр гейта %s — начинаю заново", registry_path)
    record = dict(decision)
    record["artifact"] = f"reranker.{tag}.json"
    record["embedder_tag"] = tag
    registry[tag] = record
    registry_path.write_text(
        json.dumps({"decisions": registry}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    log_path = models_dir / GATE_LOG_NAME
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return registry_path
