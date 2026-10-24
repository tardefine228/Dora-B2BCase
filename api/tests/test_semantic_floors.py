"""The semantic backend decides on its own floors; the lexical one must not move.

The artifact carries the lexical operating point (эксп. 14), but the semantic raw
scale is compressed, so ``review`` is measured separately (эксп. 16). These tests
pin that contract without a model or a Qdrant collection: they exercise
``_apply_semantic_floors`` on a stand-in reranker that only carries the pieces the
function touches.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from app.core.config import settings
from app.services.classifier import _apply_semantic_floors

from ml.reranker import Calibration, ConfidenceModel

LEXICAL_OK = 0.77
LEXICAL_REVIEW = 0.165
SEMANTIC_REVIEW = 0.45


def make_model(ok: float = LEXICAL_OK, review: float = LEXICAL_REVIEW) -> ConfidenceModel:
    return ConfidenceModel(
        weights={name: 0.0 for name in ("pair_probability",)},
        intercept=0.0,
        threshold_ok=ok,
        threshold_review=review,
        calibration=Calibration(kind="platt", a=1.1, b=1.81),
    )


def make_reranker(ok: float = LEXICAL_OK, review: float = LEXICAL_REVIEW):
    """Stand-in with the only attribute the helper reads."""
    return SimpleNamespace(confidence_model=make_model(ok, review))


@pytest.fixture
def restore_floors():
    """Settings are process-wide; put them back after every test."""
    original = (settings.semantic_ok_floor, settings.semantic_review_floor)
    yield
    settings.semantic_ok_floor, settings.semantic_review_floor = original


def test_semantic_review_floor_is_applied(restore_floors):
    settings.semantic_ok_floor = LEXICAL_OK
    settings.semantic_review_floor = SEMANTIC_REVIEW
    reranker = _apply_semantic_floors(make_reranker())
    assert reranker.confidence_model.threshold_review == SEMANTIC_REVIEW
    assert reranker.confidence_model.threshold_ok == LEXICAL_OK


def test_unset_floors_keep_the_artifact_point(restore_floors):
    settings.semantic_ok_floor = None
    settings.semantic_review_floor = None
    reranker = _apply_semantic_floors(make_reranker())
    assert reranker.confidence_model.threshold_review == LEXICAL_REVIEW
    assert reranker.confidence_model.threshold_ok == LEXICAL_OK


def test_inverted_floors_fall_back_to_the_artifact(restore_floors):
    """``ok <= review`` would make `ok` unreachable, so it must not be accepted."""
    settings.semantic_ok_floor = 0.10
    settings.semantic_review_floor = 0.80
    reranker = _apply_semantic_floors(make_reranker())
    assert reranker.confidence_model.threshold_ok == LEXICAL_OK
    assert reranker.confidence_model.threshold_review == LEXICAL_REVIEW


def test_platt_map_is_shared_and_untouched(restore_floors):
    """Only thresholds are per backend: the display scale stays the artifact's."""
    settings.semantic_ok_floor = LEXICAL_OK
    settings.semantic_review_floor = SEMANTIC_REVIEW
    reranker = _apply_semantic_floors(make_reranker())
    calibration = reranker.confidence_model.calibration
    assert (calibration.kind, calibration.a, calibration.b) == ("platt", 1.1, 1.81)


def test_status_still_uses_the_raw_scale(restore_floors):
    """The raised floor must turn weak answers into an honest `not_found`."""
    settings.semantic_ok_floor = LEXICAL_OK
    settings.semantic_review_floor = SEMANTIC_REVIEW
    model = _apply_semantic_floors(make_reranker()).confidence_model
    assert model.status_for(0.40) == "not_found"
    assert model.status_for(0.50) == "review"
    assert model.status_for(0.80) == "ok"


def test_helper_tolerates_a_reranker_without_a_confidence_model():
    """The optional-model branch must not raise on minimal rerankers."""
    assert _apply_semantic_floors(SimpleNamespace()) is not None


# --- a floor that answers more than was measured safe ---------------------------


def test_floor_below_the_measured_value_is_reported_loudly(restore_floors, caplog):
    """``0.25`` answers far more than the measurement allows.

    On the 274-row set with the ``label_kind`` doctrine, ``review = 0.25`` emits 14
    codes for goods the list does not contain and makes 146 correct decisions,
    against 4 and 152 at 0.45 (эксп. 28). Overriding the floor is allowed — the
    deployment owns the trade-off — but it must not pass silently.
    """
    settings.semantic_ok_floor = LEXICAL_OK
    settings.semantic_review_floor = 0.25
    reranker = make_reranker(review=0.62)
    reranker.uses_dense = True

    with caplog.at_level("WARNING"):
        _apply_semantic_floors(reranker)

    assert any("ниже измеренного" in record.message for record in caplog.records), [
        r.message for r in caplog.records
    ]


def test_floor_at_the_measured_value_is_not_reported(restore_floors, caplog):
    settings.semantic_ok_floor = LEXICAL_OK
    settings.semantic_review_floor = 0.450
    reranker = make_reranker(review=0.62)
    reranker.uses_dense = True

    with caplog.at_level("WARNING"):
        _apply_semantic_floors(reranker)

    assert not [record for record in caplog.records if record.levelname == "WARNING"]


def test_more_conservative_floor_is_not_reported(restore_floors, caplog):
    """Answering less is a legitimate choice: it only costs coverage."""
    settings.semantic_ok_floor = LEXICAL_OK
    settings.semantic_review_floor = 0.60
    reranker = make_reranker(review=0.62)
    reranker.uses_dense = True

    with caplog.at_level("WARNING"):
        _apply_semantic_floors(reranker)

    assert not [record for record in caplog.records if record.levelname == "WARNING"]
