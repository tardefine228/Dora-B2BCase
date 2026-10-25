"""Export only explicitly reviewed queries; predictions are never training labels."""

import argparse
import csv
import json
from datetime import date
from pathlib import Path


def verified_rows(items, codes):
    output = []
    seen = set()
    for item in items:
        if item.get("label_kind") not in {"code", "not_found"}:
            continue
        query = item.get("query", "").strip()
        key = " ".join(query.casefold().split())
        if not key or key in seen:
            raise ValueError("Empty or duplicate reviewed query")
        if not all(item.get(k, "").strip() for k in ("reviewer", "reviewed_at", "evidence")):
            raise ValueError("Reviewed query requires reviewer, reviewed_at and evidence")
        date.fromisoformat(item["reviewed_at"])
        code = item.get("expected_okpd2", "").strip()
        if item["label_kind"] == "code" and code not in codes:
            raise ValueError("Reviewed code is absent from the catalog")
        if item["label_kind"] == "not_found" and code:
            raise ValueError("not_found must not carry a code")
        seen.add(key)
        output.append(
            {
                "query": query,
                "expected_okpd2": code,
                "label_kind": item["label_kind"],
                "reviewer": item["reviewer"],
                "reviewed_at": item["reviewed_at"],
                "evidence": item["evidence"],
                "query_id": item["query_id"],
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with (root / "data/data.csv").open(encoding="utf-8-sig", newline="") as stream:
        codes = {r["okpd2"] for r in csv.DictReader(stream, delimiter=";") if r["okpd2"]}
    rows = verified_rows(json.loads(args.review.read_text(encoding="utf-8")), codes)
    if not rows:
        parser.error("No verified labels; refusing to create a misleading empty dataset")
    # Never overwrite a manually curated evaluation set.
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} verified queries")


if __name__ == "__main__":
    main()
