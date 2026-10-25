"""Post-calibration of the confidence scale must never move the statuses (C1).

The decision ``ok``/``review``/``not_found`` is taken on the *raw* probability;
``confidence`` is the calibrated number users read. These tests pin both halves
of that contract: the calibration stays monotone, and every labeled example
keeps exactly the status its raw confidence implies.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import pytest
from app.services.classifier import build_classifier

from ml.reranker import Calibration, ConfidenceModel, HeuristicReranker, load_reranker

ROOT = Path(__file__).resolve().parents[2]
LABELED = ROOT / "eval" / "labeled_examples.csv"


def _reranker():
    reranker = load_reranker()
    if isinstance(reranker, HeuristicReranker):
        pytest.skip("models/reranker.json не собран (python scripts/build_model.py)")
    return reranker


def _labeled_queries() -> list[str]:
    with LABELED.open(encoding="utf-8", newline="") as handle:
        return [
            (row["query"] or "").strip()
            for row in csv.DictReader(handle, delimiter=";")
            if (row["query"] or "").strip()
        ]


def test_artifact_has_a_monotone_calibration() -> None:
    reranker = _reranker()
    model = reranker.confidence_model
    assert model is not None
    assert model.calibration.is_increasing()

    grid = [index / 100 for index in range(1, 100)]
    calibrated = [model.calibration.calibrate(value) for value in grid]
    assert all(later >= earlier for earlier, later in zip(calibrated, calibrated[1:], strict=False))
    # A calibration must not be a no-op forever: it is fitted, not hard-coded.
    assert any(abs(value - raw) > 1e-6 for value, raw in zip(calibrated, grid, strict=True))


def test_statuses_follow_the_raw_confidence_on_the_labeled_set() -> None:
    classifier = build_classifier("catalog")
    reranker = getattr(classifier, "_reranker", None)
    if reranker is None or isinstance(reranker, HeuristicReranker):
        pytest.skip("models/reranker.json не собран (python scripts/build_model.py)")

    checked = 0
    rescued = 0
    for query in _labeled_queries():
        response = classifier.classify(query)
        assert response.confidence_raw is not None
        expected_status = reranker.confidence_model.status_for(response.confidence_raw)
        if response.status != expected_status:
            # The one documented exception (эксп. 34): a below-threshold lead
            # whose product word is written in the position's own name is shown
            # as `review` instead of «нет в перечне». Nothing else may differ.
            assert expected_status == "not_found" and response.status == "review", (
                query,
                response.status,
                expected_status,
            )
            assert response.code, query
            rescued += 1
        checked += 1
    assert checked >= 89
    # The exception stays narrow: a handful of rows, not a second decision rule.
    # 9 of the 201 labeled rows (эксп. 34): one documented exception per row whose
    # head word is written in the position's own name — 4.5 %, not a second rule.
    assert rescued <= 10, rescued


def test_reported_confidence_is_the_calibrated_raw_value() -> None:
    classifier = build_classifier("catalog")
    reranker = getattr(classifier, "_reranker", None)
    if reranker is None or isinstance(reranker, HeuristicReranker):
        pytest.skip("models/reranker.json не собран (python scripts/build_model.py)")

    for query in ("Электрочайники", "Ноутбук Lenovo IdeaPad", "Смарт-часы"):
        response = classifier.classify(query)
        expected = round(reranker.display_confidence(response.confidence_raw or 0.0), 4)
        assert response.confidence == expected, query


def test_platt_scaling_is_monotone_and_bounded() -> None:
    calibration = Calibration("platt", a=2.5, b=-1.1)
    assert calibration.is_increasing()
    values = [calibration.calibrate(probability) for probability in (0.001, 0.2, 0.5, 0.9, 0.999)]
    assert all(later >= earlier for earlier, later in zip(values, values[1:], strict=False))
    assert all(0.0 <= value <= 1.0 for value in values)
    assert calibration.calibrate(1.0) == pytest.approx(1.0, abs=1e-3)


def test_negative_slope_falls_back_to_identity() -> None:
    calibration = Calibration("platt", a=-3.0, b=0.5)
    assert calibration.kind == "identity"
    assert calibration.calibrate(0.3) == pytest.approx(0.3)


def test_confidence_model_decides_on_raw_ignoring_calibration() -> None:
    model = ConfidenceModel(
        weights={},
        intercept=0.0,
        threshold_ok=0.8,
        threshold_review=0.2,
        calibration=Calibration("platt", a=4.0, b=-3.0),  # pulls everything down
    )
    # raw = 0.5 sits between the thresholds, so the status must be "review" even
    # though the calibrated value (~0.27) would fall below threshold_review.
    assert model.status_for(0.5) == "review"
    assert model.calibrate(0.5) < 0.3
    assert model.status_for(0.9) == "ok"
    assert model.status_for(0.1) == "not_found"


def test_old_artifacts_without_calibration_load_as_identity() -> None:
    model = ConfidenceModel.from_payload(
        {"weights": {}, "intercept": 0.0, "thresholds": {"ok": 0.8, "review": 0.2}}
    )
    assert model.calibration.kind == "identity"
    assert model.calibrate(0.42) == pytest.approx(0.42)


def test_calibration_platform_mirrors_the_api_confidence() -> None:
    """The build-time reproduction must match the real classifier (drift guard).

    The artifact's display map is fitted on confidences computed by
    ``ml.train_reranker.calibration_confidence``. If the API pipeline ever
    diverges, the fitted map silently stops describing production — so a sample
    of the platform queries is scored both ways here.
    """
    from ml.catalog_loader import load_catalog
    from ml.train_reranker import (
        build_calibration_cases,
        calibration_confidence,
        calibration_index,
    )

    reranker = _reranker()
    classifier = build_classifier("catalog")
    catalog = load_catalog()
    index = calibration_index(catalog)
    cases = build_calibration_cases(catalog, random.Random(1875 + 1))

    checked = 0
    for query, _expected in cases[:: max(1, len(cases) // 30)]:
        response = classifier.classify(query)
        raw, status, _code = calibration_confidence(reranker, index, query)
        assert status == response.status, query
        if status != "not_found":
            # The API zeroes the confidence of silent answers by contract, so
            # only answered rows are comparable number-wise.
            assert response.confidence_raw is not None
            assert raw == pytest.approx(response.confidence_raw, abs=5e-5), query
        checked += 1
    assert checked >= 20


def test_calibration_platform_does_not_reuse_labeled_queries() -> None:
    """The measurement set must stay out of the calibration platform."""
    from ml.catalog_loader import load_catalog
    from ml.features import normalize
    from ml.train_reranker import build_calibration_cases

    catalog = load_catalog()
    labeled = {query.casefold() for query in _labeled_queries()}
    excluded = {normalize(query) for query in labeled}
    cases = build_calibration_cases(catalog, random.Random(1875 + 1), excluded)
    platform_queries = {query.strip().casefold() for query, _expected in cases}

    assert platform_queries & labeled == set()
    assert len(cases) >= 300
    # Both halves of the platform are present: curated words and probes.
    assert any(expected for _query, expected in cases)
    assert any(not expected for _query, expected in cases)


def test_negative_probes_never_contradict_the_lexicon() -> None:
    """A word the lexicon maps to a position must not be a "stay silent" probe.

    The probes are the negative half of the calibration platform: any answer to
    them counts as a wrong answer. The third bridge wave put eight former probe
    heads in scope ("колонка" → «Громкоговорители» — the labeled set even
    expects a code for «Беспроводная колонка»), so keeping them would have
    scored correct answers as misses and refitted the display map downwards.
    """
    from ml.features import build_known_stems
    from ml.lexicon import bridged_targets
    from ml.train_reranker import OUT_OF_SCOPE_HEADS, make_negative_queries

    probe_heads = [head for head in OUT_OF_SCOPE_HEADS if not bridged_targets(head)]
    assert probe_heads, "пул негативных голов пуст"
    assert len(probe_heads) < len(OUT_OF_SCOPE_HEADS), "ни одна голова не размечена лексиконом"

    queries = make_negative_queries(random.Random(1875), build_known_stems(["Ноутбуки"]))
    assert len(queries) >= 300
    contradiction = [query for query in queries if bridged_targets(query)]
    assert contradiction == [], contradiction


def test_calibration_modifiers_carry_no_catalog_content() -> None:
    """A modifier that names a position hijacks retrieval instead of decorating it.

    "специализированные" matched «Контейнеры специализированные» and won over the
    product word in nine platform rows, which biased the platform's accuracy and
    with it the fitted display map.
    """
    from ml.catalog_loader import load_catalog
    from ml.features import stem, stems
    from ml.train_reranker import CALIBRATION_MODIFIERS

    catalog_words = {word for item in load_catalog() for word in stems(item["name"])}
    collisions = [m for m in CALIBRATION_MODIFIERS if stem(m) in catalog_words]
    assert collisions == [], collisions
    assert len(set(CALIBRATION_MODIFIERS)) >= 4
