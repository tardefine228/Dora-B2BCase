"""Reproducible catalog audit and review queues. Never edits catalog or invents labels."""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=";"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    text = str(value).strip()
    return text.lower() if text.lower() in {"true", "false"} else text


def audit(rows, json_rows):
    groups = defaultdict(list)
    ids = Counter(r.get("record_id") for r in rows)
    issues = []
    by_id = {r.get("record_id"): r for r in json_rows}
    json_ids = Counter(r.get("record_id") for r in json_rows)
    for row in rows:
        rid = row.get("record_id", "")
        groups[(row.get("okpd2", ""), row.get("name", ""))].append(rid)
        flags = []
        if not row.get("okpd2"):
            flags.append("missing_code")
        if len(row.get("name", "")) > 500:
            flags.append("long_name")
        if canonical(row.get("review_required")) == "true":
            flags.append("source_review_required")
        if row.get("code_role") == "Исключение":
            flags.append("exception_role")
        if not rid or ids[rid] > 1:
            flags.append("invalid_record_id")
        other = by_id.get(rid)
        if other is None or any(
            canonical(row.get(k)) != canonical(other.get(k)) for k in set(row) | set(other or {})
        ):
            flags.append("csv_json_mismatch")
        if flags:
            issues.append(
                {
                    "record_id": rid,
                    "issues": flags,
                    "name": row.get("name"),
                    "code": row.get("okpd2"),
                    "page": row.get("pdf_page_start"),
                    "source": "/source/" + rid,
                    "status": "pending",
                    "reviewer": "",
                    "reviewed_at": "",
                    "evidence": "",
                }
            )
    return {
        "rows": len(rows),
        "json_rows": len(json_rows),
        "duplicate_json_ids": sorted(k for k, n in json_ids.items() if n > 1),
        "json_only_ids": sorted(set(by_id) - set(ids)),
        "roles": dict(Counter(r.get("code_role") for r in rows)),
        "issue_counts": dict(Counter(f for row in issues for f in row["issues"])),
        "duplicate_code_name_groups": [v for v in groups.values() if len(v) > 1],
        "review_queue": issues,
    }


def bridge_registry(path):
    # Guards are keyed by stems, not raw product words.
    sys.path.insert(0, str(ROOT))
    from ml.features import stem
    from ml.lexicon import BRIDGE_REQUIRE, BRIDGE_VETO, BRIDGES, CANONICAL_CODES

    return [
        {
            "source": source,
            "target": target,
            "canonical_code": CANONICAL_CODES.get(target),
            "require": sorted(BRIDGE_REQUIRE.get(stem(source), [])),
            "veto": sorted(BRIDGE_VETO.get(stem(source), [])),
            "provenance": "ml/lexicon.py",
            "source_sha256": digest(path),
            "status": "legacy_unreviewed",
            "author": "unknown",
            "decision_date": "",
            "reason": "",
            "evidence": "",
        }
        for source, target in sorted(BRIDGES.items())
    ]


def query_queue(paths):
    queries = {}
    for path in paths:
        sha = digest(path)
        for index, row in enumerate(read_csv(path), 2):
            query = (row.get("query") or "").strip()
            if not query:
                continue
            key = " ".join(query.casefold().split())
            item = queries.setdefault(
                key,
                {
                    "query_id": hashlib.sha256(key.encode()).hexdigest()[:20],
                    "query": query,
                    "provenance_kind": "user_supplied_export_unverified_origin",
                    "expected_okpd2": "",
                    "label_kind": "unlabeled",
                    "reviewer": "",
                    "reviewed_at": "",
                    "evidence": "",
                    "observations": [],
                },
            )
            item["observations"].append(
                {
                    "source_file": path.name,
                    "source_sha256": sha,
                    "row": index,
                    "predicted_code": row.get("code"),
                    "user_verdict": row.get("user_verdict", ""),
                    "corrected_code": row.get("corrected_code", ""),
                }
            )
    return list(queries.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Refuse to overwrite human decisions entered into an earlier review package.
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output is not empty; choose a new folder to preserve review decisions")
    rows = read_csv(ROOT / "data/data.csv")
    report = audit(rows, json.loads((ROOT / "data/data.json").read_text(encoding="utf-8")))
    report["input_sha256"] = {
        str(p.relative_to(ROOT)): digest(p)
        for p in (ROOT / "data/data.csv", ROOT / "data/data.json", ROOT / "data/source_text.txt")
    }
    labels = read_csv(ROOT / "eval/labeled_examples.csv")
    files = {
        "catalog_audit.json": report,
        "bridge_registry.json": bridge_registry(ROOT / "ml/lexicon.py"),
        "label_review.json": [
            {**r, "reviewer": "", "reviewed_at": "", "evidence": ""}
            for r in labels
            if r.get("label_kind") == "unclear"
        ],
        "query_review.json": query_queue(args.exports),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in files.items():
        (args.output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "rows": report["rows"],
                "issues": report["issue_counts"],
                "bridges": len(files["bridge_registry.json"]),
                "unclear": len(files["label_review.json"]),
                "queries": len(files["query_review.json"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
