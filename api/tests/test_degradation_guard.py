"""OPS-3: no silent runtime degradation when the trained artifact is unavailable.

The Docker build already fails on a broken ML step unless ALLOW_DEGRADE=true.
These tests cover the runtime half of the policy: with the default
``ALLOW_DEGRADE=false`` the catalog classifier must refuse to serve the
uncalibrated heuristic scorer instead of quietly switching to it.
"""

from __future__ import annotations

import sys

import pytest
from app.core.config import settings
from app.services import classifier as classifier_module
from app.services.classifier import (
    TrainedArtifactUnavailable,
    _is_trained_reranker,
    _load_reranker,
)

from ml.reranker import HeuristicReranker, LogisticReranker


@pytest.fixture()
def strict(monkeypatch):
    monkeypatch.setattr(settings, "allow_degrade", False)


@pytest.fixture()
def permissive(monkeypatch):
    monkeypatch.setattr(settings, "allow_degrade", True)


def test_default_policy_is_strict() -> None:
    # CI and the test environment never set ALLOW_DEGRADE; the shipped default
    # must be the safe one.
    assert settings.allow_degrade is False


def test_trained_reranker_is_distinguished_from_heuristic() -> None:
    assert _is_trained_reranker(HeuristicReranker()) is False
    artifact = LogisticReranker.load(_artifact_path())
    assert _is_trained_reranker(artifact) is True
    assert _is_trained_reranker(None) is False


def _artifact_path():
    from ml.config import ARTIFACTS_DIR

    return ARTIFACTS_DIR / "reranker.json"


def test_heuristic_fallback_is_rejected_in_strict_mode(strict, monkeypatch) -> None:
    import ml.reranker as reranker_module

    monkeypatch.setattr(
        reranker_module,
        "reranker_artifact_path",
        lambda *args, **kwargs: reranker_module.MODELS_DIR / "missing.json",
    )

    with pytest.raises(TrainedArtifactUnavailable, match="ALLOW_DEGRADE"):
        _load_reranker()


def test_heuristic_fallback_is_allowed_in_permissive_mode(permissive, monkeypatch) -> None:
    import ml.reranker as reranker_module

    monkeypatch.setattr(
        reranker_module,
        "reranker_artifact_path",
        lambda *args, **kwargs: reranker_module.MODELS_DIR / "missing.json",
    )

    reranker = _load_reranker()
    assert isinstance(reranker, HeuristicReranker)


def test_broken_artifact_raises_in_strict_mode(strict, monkeypatch) -> None:
    import ml.reranker as reranker_module

    def boom(*args, **kwargs):
        raise ValueError("artifact JSON corrupted")

    monkeypatch.setattr(reranker_module, "load_reranker", boom)

    with pytest.raises(TrainedArtifactUnavailable, match="не загружается"):
        _load_reranker()


def test_broken_artifact_returns_none_in_permissive_mode(permissive, monkeypatch) -> None:
    import ml.reranker as reranker_module

    def boom(*args, **kwargs):
        raise ValueError("artifact JSON corrupted")

    monkeypatch.setattr(reranker_module, "load_reranker", boom)

    assert _load_reranker() is None


def test_missing_ml_package_raises_in_strict_mode(strict, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "ml.reranker", None)
    with pytest.raises(TrainedArtifactUnavailable, match="Пакет ml.reranker недоступен"):
        _load_reranker()


def test_missing_ml_package_is_tolerated_in_permissive_mode(permissive, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "ml.reranker", None)
    assert _load_reranker() is None


def test_build_classifier_refuses_heuristic_under_strict_policy(strict, monkeypatch) -> None:
    import ml.reranker as reranker_module

    monkeypatch.setattr(
        reranker_module,
        "reranker_artifact_path",
        lambda *args, **kwargs: reranker_module.MODELS_DIR / "missing.json",
    )
    monkeypatch.setattr(settings, "vector_search", False)

    with pytest.raises(TrainedArtifactUnavailable):
        classifier_module.build_classifier("catalog")


def test_build_classifier_starts_with_heuristic_when_degradation_allowed(
    permissive, monkeypatch
) -> None:
    import ml.reranker as reranker_module

    monkeypatch.setattr(
        reranker_module,
        "reranker_artifact_path",
        lambda *args, **kwargs: reranker_module.MODELS_DIR / "missing.json",
    )
    monkeypatch.setattr(settings, "vector_search", False)

    classifier = classifier_module.build_classifier("catalog")
    assert classifier._reranker_name == "heuristic"


def test_mock_backend_never_touches_the_artifact(strict) -> None:
    # The frontend-development mock must keep working with no ML files at all.
    mock = classifier_module.build_classifier("mock")
    assert mock.classify("что угодно").code == "MOCK-001"
