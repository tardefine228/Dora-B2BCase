import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_verified_changes_match_both_catalog_formats():
    report = json.loads(
        (ROOT / "eval/source-verification-2026-10-25.json").read_text(encoding="utf-8")
    )
    with (ROOT / "data/data.csv").open(encoding="utf-8-sig", newline="") as stream:
        csv_rows = {r["record_id"]: r for r in csv.DictReader(stream, delimiter=";")}
    json_rows = {
        r["record_id"]: r for r in json.loads((ROOT / "data/data.json").read_text(encoding="utf-8"))
    }
    for change in report["changes"]:
        rid, field = change["record_id"], change["field"]
        assert json_rows[rid][field] == change["after"]
        assert str(csv_rows[rid][field]).lower() == str(change["after"]).lower()
    assert len(csv_rows) == len(json_rows) == 1126


def test_volume_boundaries_are_inclusive_in_source_view():
    text = (ROOT / "data/source_text.txt").read_text(encoding="utf-8")
    assert "контейнера ≥ 500 мл ≤ 1000 мл)," in text
    assert "контейнера > 500 мл < 1000 мл)," not in text


def test_remaining_source_flags_have_visual_evidence():
    report = json.loads(
        (ROOT / "eval/data-review-2026-10-26/source-verification.json").read_text(encoding="utf-8")
    )
    rows = {
        r["record_id"]: r
        for r in json.loads((ROOT / "data/data.json").read_text(encoding="utf-8"))
    }
    assert len(report["records"]) == 84
    assert len({r["record_id"] for r in report["records"]}) == 84
    for checked in report["records"]:
        row = rows[checked["record_id"]]
        assert row["name"] == checked["after_name"]
        assert row["okpd2"] == checked["code"]
        assert row["pdf_page_start"] == checked["page"]
        assert row["review_required"] is False
    assert all("Х$" not in r["name"] for r in rows.values())
    assert "Х$" not in (ROOT / "data/source_text.txt").read_text(encoding="utf-8")
