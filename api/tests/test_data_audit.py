import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load("audit_data")
export = load("export_reviewed_queries")
REGISTRY = ROOT / "eval/data-review-2026-10-25/bridge_registry.json"


@pytest.mark.parametrize(
    "entry", json.loads(REGISTRY.read_text(encoding="utf-8")), ids=lambda r: r["source"]
)
def test_every_registered_bridge_matches_code_and_guards(entry):
    current = {r["source"]: r for r in audit.bridge_registry(ROOT / "ml/lexicon.py")}
    actual = current[entry["source"]]
    for key in ("target", "canonical_code", "require", "veto"):
        assert entry[key] == actual[key]
    codes = {r["okpd2"] for r in audit.read_csv(ROOT / "data/data.csv")}
    # Canonical codes may be parent groups containing several catalog positions.
    assert any(
        c == entry["canonical_code"] or c.startswith(entry["canonical_code"] + ".") for c in codes
    )


def test_registry_covers_all_bridges():
    actual = audit.bridge_registry(ROOT / "ml/lexicon.py")
    assert {r["source"] for r in actual} == {
        r["source"] for r in json.loads(REGISTRY.read_text(encoding="utf-8"))
    }


def test_csv_json_equivalence():
    report = audit.audit(
        audit.read_csv(ROOT / "data/data.csv"),
        json.loads((ROOT / "data/data.json").read_text(encoding="utf-8")),
    )
    assert not report["issue_counts"].get("csv_json_mismatch")
    assert not report["issue_counts"].get("invalid_record_id")
    assert not report["duplicate_json_ids"]
    assert not report["json_only_ids"]


def test_predictions_do_not_become_labels():
    assert export.verified_rows([{"label_kind": "unlabeled", "predicted_code": "1"}], {"1"}) == []


def test_review_without_evidence_is_rejected():
    with pytest.raises(ValueError, match="requires"):
        export.verified_rows([{"query": "Товар", "label_kind": "code"}], {"1"})


def test_verified_label_can_be_exported():
    item = {
        "query": "Товар",
        "query_id": "q1",
        "label_kind": "code",
        "expected_okpd2": "1",
        "reviewer": "Expert",
        "reviewed_at": "2026-10-25",
        "evidence": "Source page 1",
    }
    assert export.verified_rows([item], {"1"})[0]["expected_okpd2"] == "1"
