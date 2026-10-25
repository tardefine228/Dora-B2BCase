"""Dense-artifact quality gate (ML-1).

The gate compares artifacts *at equal silence*: both are
swept over the review threshold, the candidate is moved to
the base operating silence, and only a top-1 advantage of
at least ``GATE_MARGIN_PP`` keeps it. A rejected artifact
stays on disk with the decision recorded — the loader then
ignores it, so a failed experiment never silently serves.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.schemas import ClassifyResponse

import ml.reranker as reranker_module
from ml.gate import (
    artifact_allowed,
    decision_for,
    gate_decision,
    record_decision,
)
from ml.reranker import ConfidenceModel


class _MockReranker:
    """Confidence model with a settable review threshold."""

    def __init__(self, threshold_review: float = 0.16) -> None:
        self.confidence_model = ConfidenceModel(
            weights={}, intercept=0.0, threshold_ok=0.77,
            threshold_review=threshold_review,
        )


class _MockClassifier:
    """Serves fixed raw confidences; statuses follow the threshold."""

    def __init__(self, raw: dict[str, float]) -> None:
        self._raw = raw
        self._reranker = _MockReranker()

    def classify(self, query: str) -> ClassifyResponse:
        raw = self._raw.get(query, 0.0)
        model = self._reranker.confidence_model
        if raw >= model.threshold_ok:
            status = "ok"
        elif raw >= model.threshold_review:
            status = "review"
        else:
            status = "not_found"
        return ClassifyResponse(
            query=query,
            code="1.1" if status != "not_found" else None,
            title="позиция",
            confidence=raw,
            confidence_raw=raw,
            status=status,
            clause_ref=None,
            clause_url=None,
            alternatives=[],
            latency_ms=0.0,
        )


def _rows() -> list[dict]:
    """6 coded + 4 out-of-scope rows."""
    rows = [
        {"query": f"товар {i}", "expected_okpd2": "1.1", "disputed": "0"}
        for i in range(6)
    ]
    rows += [
        {"query": f"вне {i}", "expected_okpd2": "", "disputed": "0"}
        for i in range(4)
    ]
    return rows


def _base() -> _MockClassifier:
    return _MockClassifier(
        {
            "товар 0": 0.50, "товар 1": 0.40, "товар 2": 0.30,
            "товар 3": 0.20, "товар 4": 0.15, "товар 5": 0.10,
            "вне 0": 0.90, "вне 1": 0.80, "вне 2": 0.05, "вне 3": 0.02,
        }
    )


def test_gate_keeps_artifact_with_gain_at_equal_silence() -> None:
    # Same out-of-scope raws (equal silence at every threshold),
    # better on the mid-confidence coded rows.
    candidate = _MockClassifier(
        {
            "товар 0": 0.55, "товар 1": 0.45, "товар 2": 0.35,
            "товар 3": 0.25, "товар 4": 0.20, "товар 5": 0.12,
            "вне 0": 0.90, "вне 1": 0.80, "вне 2": 0.05, "вне 3": 0.02,
        }
    )
    decision = gate_decision(_base(), candidate, _rows())
    assert decision["decision"] == "kept"
    assert decision["base"]["silence"] == decision["candidate"]["silence"]
    assert decision["candidate"]["top1_expected"] > decision["base"][
        "top1_expected"
    ]


def test_gate_rejects_artifact_without_gain() -> None:
    candidate = _MockClassifier(
        {
            "товар 0": 0.50, "товар 1": 0.40, "товар 2": 0.28,
            "товар 3": 0.18, "товар 4": 0.12, "товар 5": 0.08,
            "вне 0": 0.90, "вне 1": 0.80, "вне 2": 0.05, "вне 3": 0.02,
        }
    )
    decision = gate_decision(_base(), candidate, _rows())
    assert decision["decision"] == "rejected"


def test_gate_rejects_when_silence_unreachable() -> None:
    candidate = _MockClassifier(
        {
            "товар 0": 0.55, "товар 1": 0.45, "товар 2": 0.35,
            "товар 3": 0.25, "товар 4": 0.20, "товар 5": 0.12,
            "вне 0": 0.95, "вне 1": 0.90, "вне 2": 0.85, "вне 3": 0.80,
        }
    )
    decision = gate_decision(_base(), candidate, _rows())
    assert decision["decision"] == "rejected"
    assert "тишины" in decision["reason"]


def test_gate_restores_thresholds_after_measurement() -> None:
    base = _base()
    candidate = _MockClassifier(
        {
            "товар 0": 0.55, "товар 1": 0.45, "товар 2": 0.35,
            "товар 3": 0.25, "товар 4": 0.20, "товар 5": 0.12,
            "вне 0": 0.90, "вне 1": 0.80, "вне 2": 0.05, "вне 3": 0.02,
        }
    )
    gate_decision(base, candidate, _rows())
    assert (
        base._reranker.confidence_model.threshold_review == 0.16
    )


# --- decision registry ----------------------------------------------------


def test_registry_roundtrip(tmp_path: Path) -> None:
    decision = gate_decision(_base(), _base(), _rows())
    record_decision(tmp_path, "tfidf", decision)
    registry = json.loads(
        (tmp_path / "dense_gate.json").read_text(encoding="utf-8")
    )
    assert registry["decisions"]["tfidf"]["decision"] == decision["decision"]
    log_lines = (tmp_path / "gate_decisions.jsonl").read_text(
        encoding="utf-8"
    ).strip().splitlines()
    assert len(log_lines) == 1
    assert json.loads(log_lines[0])["embedder_tag"] == "tfidf"


def test_artifact_allowed_respects_rejection(tmp_path: Path) -> None:
    decision = gate_decision(_base(), _base(), _rows())
    decision["decision"] = "rejected"
    record_decision(tmp_path, "tfidf", decision)

    import ml.config as config
    import ml.gate as gate

    original_models = config.MODELS_DIR
    original_artifacts = config.ARTIFACTS_DIR
    gate.MODELS_DIR = tmp_path
    gate.ARTIFACTS_DIR = tmp_path
    reranker_module.MODELS_DIR = tmp_path
    reranker_module.ARTIFACTS_DIR = tmp_path
    try:
        assert not artifact_allowed("tfidf")
        assert artifact_allowed("e5-small")  # no decision recorded
        assert decision_for("tfidf")["decision"] == "rejected"
    finally:
        gate.MODELS_DIR = original_models
        gate.ARTIFACTS_DIR = original_artifacts
        reranker_module.MODELS_DIR = original_models
        reranker_module.ARTIFACTS_DIR = original_artifacts


def test_loader_skips_rejected_tagged_artifact(tmp_path: Path) -> None:
    decision = gate_decision(_base(), _base(), _rows())
    decision["decision"] = "rejected"
    record_decision(tmp_path, "tfidf", decision)

    import ml.config as config
    import ml.gate as gate
    import ml.reranker as reranker_module

    original_models = config.MODELS_DIR
    gate.MODELS_DIR = tmp_path
    reranker_module.MODELS_DIR = tmp_path
    try:
        # A rejected artifact must not be served even when present.
        (tmp_path / "reranker.tfidf.json").write_text("{}", encoding="utf-8")
        assert not reranker_module.reranker_artifact_path("tfidf").is_file() or (
            reranker_module.reranker_artifact_path("tfidf").name
            == "reranker.json"
        )
    finally:
        gate.MODELS_DIR = original_models
        reranker_module.MODELS_DIR = original_models


@pytest.mark.parametrize(
    "thresholds", [(0.08, 0.12, 0.16, 0.20, 0.25, 0.30)]
)
def test_frontier_is_monotone_in_silence(thresholds) -> None:
    from ml.gate import frontier

    points = frontier(_base(), _rows(), thresholds)
    silences = [point["silence"] for point in points]
    assert silences == sorted(silences, reverse=True)
