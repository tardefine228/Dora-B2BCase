"""The shipped/built artifact check must catch a trainer change, not a BLAS rounding."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from check_artifacts import RUN_KEYS, compare, load  # noqa: E402

SHIPPED = ROOT / "ml" / "artifacts" / "reranker.json"


def _artifact() -> dict:
    return load(SHIPPED)


def _mutated(**meta) -> dict:
    artifact = json.loads(json.dumps(_artifact()))
    artifact["meta"].update(meta)
    return artifact


def test_the_shipped_artifact_matches_itself():
    assert compare(_artifact(), _artifact()) == []


def test_a_weight_rounded_differently_by_blas_is_still_the_same_run():
    """1e-9 is machine noise; failing on it would make the CI step flaky."""
    artifact = json.loads(json.dumps(_artifact()))
    name = next(iter(artifact["weights"]))
    artifact["weights"][name] += 1e-9
    assert compare(artifact, _artifact()) == []


def test_a_different_weight_is_a_different_model():
    artifact = json.loads(json.dumps(_artifact()))
    name = next(iter(artifact["weights"]))
    artifact["weights"][name] += 1e-3
    problems = compare(artifact, _artifact())
    assert any(name in problem for problem in problems)


@pytest.mark.parametrize("key", RUN_KEYS)
def test_every_run_description_field_is_compared(key):
    """Each field pins the artifact to one training run; none may be ignored."""
    artifact = _mutated(**{key: "someone-else's value"})
    problems = compare(artifact, _artifact())
    assert any(problem.startswith(f"meta.{key}") for problem in problems)


def test_a_different_feature_set_is_reported():
    artifact = json.loads(json.dumps(_artifact()))
    artifact["feature_names"] = list(artifact["feature_names"])[:-1]
    assert any("признаков" in problem for problem in compare(artifact, _artifact()))


def test_the_retired_artifact_would_have_been_rejected():
    """The failure mode this script exists for: 29277 pairs and 0.621/0.097 thresholds."""
    retired = _mutated(
        train_pairs=29277,
        pair_thresholds={"ok": 0.621, "review": 0.097},
        split_strategy=None,
    )
    problems = compare(_artifact(), retired)  # shipped file was the "fresh" one here
    assert len(problems) >= 3
    assert any("29277" in problem for problem in problems)
