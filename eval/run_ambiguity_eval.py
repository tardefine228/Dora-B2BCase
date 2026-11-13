#!/usr/bin/env python3
"""Executable clarification regression on the currently configured backend.

Usage (from repository root):
    PYTHONPATH=api python eval/run_ambiguity_eval.py

The gate is intentionally separate from top-1 accuracy: a confident code on a
``clarify`` row is always a failure, even if that code could be reasonable.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

from app.services.classifier import get_classifier  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "eval" / "ambiguity_examples.csv")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.dataset.open(encoding="utf-8-sig", newline="") as handle:
        cases = list(csv.DictReader(handle))
    classifier = get_classifier()
    results = []
    for case in cases:
        answer = classifier.classify(case["query"])
        option_ids = {
            option.option_id
            for option in (answer.clarification.options if answer.clarification else [])
        }
        expected_options = set(filter(None, case["expected_value"].split("|")))
        if case["expectation"] == "clarify":
            passed = (
                answer.status == "needs_clarification"
                and answer.code is None
                and answer.title is None
                and len(option_ids) >= 2
                and expected_options <= option_ids
            )
        else:
            passed = (
                answer.status != "needs_clarification" and answer.code == case["expected_value"]
            )
        results.append(
            {
                **case,
                "passed": passed,
                "actual_status": answer.status,
                "actual_code": answer.code,
                "option_ids": sorted(option_ids),
            }
        )
    clarify = [row for row in results if row["expectation"] == "clarify"]
    controls = [row for row in results if row["expectation"] == "code"]
    summary = {
        "total": len(results),
        "clarification_recall": sum(row["passed"] for row in clarify) / max(len(clarify), 1),
        "unambiguous_pass_rate": sum(row["passed"] for row in controls) / max(len(controls), 1),
        "forced_code_on_ambiguous": sum(
            row["actual_status"] != "needs_clarification" and bool(row["actual_code"])
            for row in clarify
        ),
        "failed_case_ids": [row["case_id"] for row in results if not row["passed"]],
    }
    report = {"summary": summary, "results": results}
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not summary["failed_case_ids"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
