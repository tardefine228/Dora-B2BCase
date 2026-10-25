"""The synthetic specifier stress set is a measurement, and it must stay one.

`eval/specifier_stress.py` builds realistic purchase lines from the catalog
(position name, its first words, or the everyday word the lexicon bridges to it,
each with and without 1-3 technical attributes) and measures how often the right
position still comes back. It exists because the labeled set is small and human:
a change that helps real ERP lines can move it by one row either way.

Two things are pinned here:

* the generator is deterministic and complete — otherwise two runs could not be
  compared, and the recorded baseline would describe nothing;
* a run of the shipped artifact is not worse than the recorded baseline beyond
  the tolerance. The policy is the same as for the labeled set: a change that
  costs accuracy on this set is a regression, whatever it does elsewhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.specifier_stress import BASES, SHAPES, build_cases, measure, summarize
from ml.specifiers import SPEC_EXAMPLES

BASELINE = Path(__file__).resolve().parents[2] / "eval" / "baselines" / "specifier_stress.json"


def test_cases_are_deterministic():
    first = build_cases(40)
    second = build_cases(40)
    assert first == second
    assert first, "генератор не вернул ни одной строки"


def test_every_shape_is_generated():
    shapes = {case["shape"] for case in build_cases(40)}
    assert shapes == set(SHAPES)
    for base in BASES:
        assert base in shapes and f"{base}+attr" in shapes


def test_truth_is_a_set_of_codes_for_the_same_position():
    """87 of 746 names are repeated in several appendices with different codes."""
    cases = {case["row"]: case for case in build_cases(None)}
    duplicates = [case for case in cases.values() if len(case["expected"]) > 1]
    assert duplicates, "в каталоге нет повторов — тест потерял смысл"
    for case in duplicates:
        assert all(code for code in case["expected"])


def test_generated_lines_carry_an_attribute():
    """Every ``+attr`` line ends with a real attribute string, not with the name."""
    for case in build_cases(40):
        if not case["shape"].endswith("+attr"):
            continue
        tail = case["query"].rsplit(";", 1)[-1].strip()
        assert any(tail.endswith(example) for example in SPEC_EXAMPLES), case["query"]


@pytest.fixture(scope="module")
def measured():  # noqa: ANN201 - fixture body is the tool's own output
    from app.services.classifier import build_classifier

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    cases = build_cases(baseline["limit"])
    return summarize(measure(cases, build_classifier("catalog"))), baseline["shapes"]


def test_stress_run_is_not_worse_than_the_recorded_baseline(measured):
    current, baseline = measured
    problems = []
    for shape in SHAPES:
        for key in ("top1", "top3"):
            if round(current[shape][key] - baseline[shape][key], 1) < -0.5:
                problems.append(
                    f"{shape}.{key}: было {baseline[shape][key]:.1f}, "
                    f"стало {current[shape][key]:.1f}"
                )
        if round(current[shape]["silence"] - baseline[shape]["silence"], 1) > 0.5:
            problems.append(
                f"{shape}.silence: было {baseline[shape]['silence']:.1f}, "
                f"стало {current[shape]['silence']:.1f}"
            )
    assert not problems, "стресс-набор просел:\n" + "\n".join(problems)
