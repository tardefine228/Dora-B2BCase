"""Environment-controlled decision floors for the catalog backend (BE-3)."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from app.core.config import Settings, settings
from app.services.classifier import CatalogClassifier, _apply_catalog_floors

from ml.reranker import GROUP_FEATURE_NAMES, ConfidenceModel, LogisticReranker


@pytest.fixture
def restore_catalog_floors():
    original = (settings.catalog_ok_floor, settings.catalog_review_floor)
    yield
    settings.catalog_ok_floor, settings.catalog_review_floor = original


def _model(ok: float = 0.75, review: float = 0.20) -> ConfidenceModel:
    return ConfidenceModel(
        weights={name: 0.0 for name in GROUP_FEATURE_NAMES},
        intercept=math.log(0.6 / 0.4),
        threshold_ok=ok,
        threshold_review=review,
    )


def _reranker(ok: float = 0.75, review: float = 0.20):
    return SimpleNamespace(confidence_model=_model(ok, review))


def test_catalog_floors_are_read_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("CATALOG_OK_FLOOR", "0.82")
    monkeypatch.setenv("CATALOG_REVIEW_FLOOR", "0.31")

    config = Settings()

    assert config.catalog_ok_floor == pytest.approx(0.82)
    assert config.catalog_review_floor == pytest.approx(0.31)


@pytest.mark.parametrize("name", ["CATALOG_OK_FLOOR", "CATALOG_REVIEW_FLOOR"])
def test_catalog_floors_default_to_the_artifact(monkeypatch, name: str) -> None:
    monkeypatch.delenv("CATALOG_OK_FLOOR", raising=False)
    monkeypatch.delenv("CATALOG_REVIEW_FLOOR", raising=False)

    config = Settings()

    assert getattr(config, name.lower()) is None


@pytest.mark.parametrize("value", ["-0.01", "1.01", "nan", "inf", "-inf", "oops"])
def test_catalog_floor_rejects_non_probabilities_at_startup(monkeypatch, value: str) -> None:
    monkeypatch.setenv("CATALOG_OK_FLOOR", value)

    with pytest.raises(ValueError, match="CATALOG_OK_FLOOR"):
        Settings()


def test_catalog_floor_overrides_change_the_decision_without_changing_weights(
    restore_catalog_floors,
) -> None:
    settings.catalog_ok_floor = 0.80
    settings.catalog_review_floor = 0.70
    reranker = _reranker()
    weights_before = reranker.confidence_model.weights.copy()

    _apply_catalog_floors(reranker)

    model = reranker.confidence_model
    assert model.threshold_ok == pytest.approx(0.80)
    assert model.threshold_review == pytest.approx(0.70)
    assert model.status_for(0.60) == "not_found"
    assert model.status_for(0.75) == "review"
    assert model.status_for(0.90) == "ok"
    assert model.weights == weights_before


def test_unset_catalog_floors_keep_the_artifact_thresholds(restore_catalog_floors) -> None:
    settings.catalog_ok_floor = None
    settings.catalog_review_floor = None
    reranker = _reranker(ok=0.73, review=0.18)

    _apply_catalog_floors(reranker)

    assert reranker.confidence_model.threshold_ok == pytest.approx(0.73)
    assert reranker.confidence_model.threshold_review == pytest.approx(0.18)


def test_inverted_catalog_floors_fall_back_to_artifact_pair(restore_catalog_floors) -> None:
    """Never apply half of an invalid pair: it could make one status unreachable."""
    settings.catalog_ok_floor = 0.20
    settings.catalog_review_floor = 0.80
    reranker = _reranker(ok=0.74, review=0.16)

    _apply_catalog_floors(reranker)

    assert reranker.confidence_model.threshold_ok == pytest.approx(0.74)
    assert reranker.confidence_model.threshold_review == pytest.approx(0.16)


def test_reranker_without_group_model_uses_its_own_thresholds(restore_catalog_floors) -> None:
    settings.catalog_ok_floor = 0.81
    settings.catalog_review_floor = 0.29
    reranker = SimpleNamespace(
        confidence_model=None,
        threshold_ok=0.72,
        threshold_review=0.15,
    )

    _apply_catalog_floors(reranker)

    assert reranker.threshold_ok == pytest.approx(0.81)
    assert reranker.threshold_review == pytest.approx(0.29)


def test_catalog_classifier_uses_configured_floors_for_real_decisions(
    tmp_path: Path, restore_catalog_floors
) -> None:
    catalog = tmp_path / "catalog.csv"
    catalog.write_text(
        "name;okpd2\nПринтеры лазерные;26.20.16.120\n",
        encoding="utf-8",
    )
    settings.catalog_ok_floor = 0.80
    settings.catalog_review_floor = 0.70
    pair_model = _model(ok=0.75, review=0.20)
    reranker = LogisticReranker(
        weights={},
        intercept=0.0,
        threshold_ok=0.75,
        threshold_review=0.20,
        confidence_model=pair_model,
    )

    classifier = CatalogClassifier(catalog, reranker=reranker, dense=None)
    result = classifier.classify("Принтеры лазерные")

    assert reranker.confidence_model.status_for(0.60) == "not_found"
    assert result.status == "not_found"
    assert result.code is None
