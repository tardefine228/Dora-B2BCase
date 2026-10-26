"""The external PP-1875 snapshot must stay equivalent to the working catalog."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "eval" / "source-snapshots" / "2026-10-26" / "pp_1875_dataframe.csv"


def load_verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_catalog_snapshot", ROOT / "scripts" / "verify_catalog_snapshot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verifier = load_verifier()


def test_committed_snapshot_matches_the_catalog():
    report = verifier.compare(
        verifier.read_catalog(SNAPSHOT),
        verifier.read_catalog(ROOT / "data" / "data.csv"),
    )

    assert report["ok"] is True
    assert report["rows"] == {"snapshot": 1126, "catalog": 1126}
    assert report["aligned_ids"] == 1126
    assert report["only_in_snapshot"] == [] and report["only_in_catalog"] == []
    # Codes, roles, pages, documents: character-identical.
    assert all(not items for items in report["other_exact_diffs"].values())
    assert report["numeric_value_diffs"] == {}
    assert report["numeric_format_only_cells"] > 0  # 80 vs 80.0 is formatting only


def test_only_the_four_pdf_corrected_names_differ():
    report = verifier.compare(
        verifier.read_catalog(SNAPSHOT),
        verifier.read_catalog(ROOT / "data" / "data.csv"),
    )

    ids = {item["record_id"] for item in report["name_diffs"]}
    assert ids == set(verifier.EXPECTED_NAME_CORRECTIONS)
    # The catalog side carries the verified inclusive bounds.
    for item in report["name_diffs"]:
        assert "≥ 500 мл ≤ 1000 мл" in item["catalog"]
        assert item["snapshot"] != item["catalog"]


def test_snapshot_lacks_only_the_qa_column():
    report = verifier.compare(
        verifier.read_catalog(SNAPSHOT),
        verifier.read_catalog(ROOT / "data" / "data.csv"),
    )

    assert report["columns"]["catalog_only"] == ["review_required"]
    assert report["columns"]["snapshot_only"] == []
    assert report["catalog_review_required_true"] == 84


def test_normalize_record_id_handles_zero_padding():
    assert verifier.normalize_record_id("1-001-01") == "1-1-1"
    assert verifier.normalize_record_id("2-402-04") == "2-402-4"


def _row(record_id: str, **overrides) -> dict:
    base = {
        "record_id": record_id,
        "appendix": "1",
        "position": "1",
        "name": "Ткани текстильные",
        "okpd2": "13.2",
        "code_role": "Включён",
        "min_share_pct": "",
        "max_foreign_share_pct": "",
        "confirming_document": "",
        "pdf_page_start": "32",
        "pdf_page_end": "32",
        "source_code_cell": "13.2",
    }
    base.update(overrides)
    return base


def _whitelisted_corrections() -> tuple[list[dict], list[dict]]:
    """The four position-402 rows, snapshot OCR text vs the PDF-verified name."""
    snapshot, catalog = [], []
    for index in range(1, 5):
        record = f"2-402-0{index}"
        snapshot.append(_row(record, name="объем контейнера 2 500 мл < 1000 мл, - - текст"))
        catalog.append(
            _row(
                f"2-402-{index}",
                name="объем контейнера ≥ 500 мл ≤ 1000 мл, соответствующие кодам",
                review_required="False",
            )
        )
    return snapshot, catalog


def test_numeric_formatting_difference_is_not_a_regression():
    snapshot, catalog = _whitelisted_corrections()
    snapshot.append(_row("1-001-01", min_share_pct="80"))
    catalog.append(_row("1-1-01", min_share_pct="80.0", review_required="False"))

    report = verifier.compare(snapshot, catalog)

    assert report["ok"] is True
    assert report["numeric_format_only_cells"] == 1
    assert report["numeric_value_diffs"] == {}


def test_unexpected_name_change_fails():
    snapshot = [_row("1-001-01", name="Старое OCR-название")]
    catalog = [_row("1-1-01", name="Исправленное название", review_required="False")]

    report = verifier.compare(snapshot, catalog)

    assert report["ok"] is False
    assert report["unexpected_name_diffs"] == ["1-1-1"]


def test_missing_whitelisted_correction_fails():
    # Same name on both sides means the expected PDF correction is gone.
    snapshot = [_row("2-402-01")]
    catalog = [_row("2-402-1", review_required="False")]

    report = verifier.compare(snapshot, catalog)

    assert report["ok"] is False
    assert "2-402-1" in report["missing_expected_corrections"]


def test_code_drift_is_caught():
    snapshot = [_row("1-001-01", okpd2="13.2")]
    catalog = [_row("1-1-01", okpd2="99.99", review_required="False")]

    report = verifier.compare(snapshot, catalog)

    assert report["ok"] is False
    assert report["other_exact_diffs"]["okpd2"][0]["record_id"] == "1-1-1"


def test_duplicate_normalized_ids_are_rejected():
    snapshot = [_row("1-001-01"), _row("1-1-01")]
    catalog = [_row("1-1-01", review_required="False")]

    with pytest.raises(ValueError, match="дублирующиеся"):
        verifier.compare(snapshot, catalog)
