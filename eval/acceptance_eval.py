"""Measure exact top-1 accuracy on an explicit CSV, locally or against the serving API.

Examples (from repository root):
  python eval/acceptance_eval.py --dataset eval/labeled_examples.csv --api-url http://localhost:8000
  python eval/acceptance_eval.py --dataset test_sample_100.csv --backend catalog

Columns: query/expected_okpd2 or product_name/true_okpd2. UTF-8 BOM and comma/semicolon
CSV are supported. Empty expected code or НЕТ В ПЕРЕЧНЕ means not_found.
Optional accepted_okpd2 contains expert-approved additional exact codes separated by |.
Do not construct that column from model predictions. Hierarchical accuracy is reported
separately, never used to pass the acceptance gate. All rows, including errors, count.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
CODE = re.compile(r"\d{2}(?:\.\d{1,3})+")
NEGATIVE = {"", "НЕТ В ПЕРЕЧНЕ", "NOT_FOUND", "(NOT_FOUND)"}


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = handle.readline()
        handle.seek(0)
        delimiter = ";" if header.count(";") > header.count(",") else ","
        reader = csv.DictReader(handle, delimiter=delimiter)
        fields = set(reader.fieldnames or [])
        if {"query", "expected_okpd2"} <= fields:
            query_key, code_key = "query", "expected_okpd2"
        elif {"product_name", "true_okpd2"} <= fields:
            query_key, code_key = "product_name", "true_okpd2"
        else:
            raise ValueError("Required columns: query/expected_okpd2 or product_name/true_okpd2")
        cases = []
        for number, row in enumerate(reader, 2):
            query = (row.get(query_key) or "").strip()
            expected = (row.get(code_key) or "").strip()
            if not query or row.get(code_key) is None or None in row:
                raise ValueError(f"Malformed CSV row {number}")
            negative = expected.upper() in NEGATIVE
            extras = [x.strip() for x in (row.get("accepted_okpd2") or "").split("|") if x.strip()]
            accepted = list(dict.fromkeys(([expected] if not negative else []) + extras))
            if (negative and extras) or any(not CODE.fullmatch(c) for c in accepted):
                raise ValueError(f"Invalid expected code in CSV row {number}")
            cases.append(
                {
                    "row": number,
                    "query": query,
                    "expected": expected,
                    "accepted": accepted,
                    "negative": negative,
                    "category": row.get("category", ""),
                    "disputed": row.get("disputed", ""),
                }
            )
    if not cases:
        raise ValueError("Dataset is empty")
    return cases


def hierarchical_match(predicted: str, expected: str) -> bool:
    return bool(predicted and expected) and (
        predicted == expected
        or predicted.startswith(expected + ".")
        or expected.startswith(predicted + ".")
    )


def score(case: dict, response: dict) -> dict:
    status, code = response.get("status"), response.get("code")
    if status not in {"ok", "review", "not_found"}:
        raise ValueError("Invalid response status")
    if status == "not_found" and code is not None:
        raise ValueError("not_found response must have a null code")
    if status != "not_found" and (not isinstance(code, str) or not CODE.fullmatch(code)):
        raise ValueError("Answered response must contain an OKPD2 code")
    alternatives = response.get("alternatives") or []
    candidates = list(dict.fromkeys(c for c in [code] + [a.get("code") for a in alternatives] if c))
    if case["negative"]:
        exact = top3 = hierarchy = status == "not_found"
    else:
        exact = code in case["accepted"]
        top3 = any(c in case["accepted"] for c in candidates[:3])
        hierarchy = any(hierarchical_match(code or "", c) for c in case["accepted"])
    return {
        **case,
        "predicted": code,
        "status": status,
        "confidence": response.get("confidence"),
        "exact_top1": exact,
        "exact_top3": top3,
        "hierarchical_top1": hierarchy,
        "candidates": candidates,
        "error": None,
    }


def summarize(rows: list[dict]) -> dict:
    def rate(subset, key):
        return 100 * sum(bool(r[key]) for r in subset) / len(subset) if subset else None

    positive = [r for r in rows if not r["negative"]]
    negative = [r for r in rows if r["negative"]]
    return {
        "total": len(rows),
        "correct": sum(r["exact_top1"] for r in rows),
        "errors": sum(bool(r["error"]) for r in rows),
        "exact_top1_all_pct": rate(rows, "exact_top1"),
        "exact_top1_coded_pct": rate(positive, "exact_top1"),
        "correct_not_found_pct": rate(negative, "exact_top1"),
        "exact_top3_coded_pct": rate(positive, "exact_top3"),
        "hierarchical_top1_all_pct": rate(rows, "hierarchical_top1"),
        "coded_rows": len(positive),
        "negative_rows": len(negative),
    }


def request_json(url: str, timeout: float, payload: dict | None = None) -> dict:
    request = Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--api-url", help="Base URL of the running API")
    mode.add_argument("--backend", choices=["catalog", "qdrant"])
    parser.add_argument("--output", type=Path, default=Path("acceptance_results.json"))
    parser.add_argument("--min-accuracy", type=float, default=85)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--interval", type=float, default=0.12, help="Pause between HTTP requests")
    args = parser.parse_args()
    if not math.isfinite(args.min_accuracy) or not 0 <= args.min_accuracy <= 100:
        parser.error("min-accuracy must be between 0 and 100")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be positive")
    if not math.isfinite(args.interval) or args.interval < 0:
        parser.error("interval must be non-negative")
    cases = load_cases(args.dataset)
    metadata = {
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
    }
    if args.backend:
        sys.path[:0] = [str(ROOT), str(ROOT / "api")]
        from app.services.classifier import build_classifier

        classifier = build_classifier(args.backend)

        def predict(q):
            return classifier.classify(q).model_dump()

        metadata.update(
            backend=args.backend, vector_search=getattr(classifier, "vector_info", None)
        )
    else:
        base = (args.api_url or "http://localhost:8000").rstrip("/")

        def predict(q):
            return request_json(base + "/classify", args.timeout, {"name": q})

        try:
            metadata["health"] = request_json(base + "/health/ready", args.timeout)
        except Exception as exc:
            metadata["health_error"] = str(exc)
    results = []
    for case in cases:
        started = time.perf_counter()
        try:
            result = score(case, predict(case["query"]))
        except Exception as exc:
            result = {
                **case,
                "predicted": None,
                "status": "error",
                "confidence": None,
                "exact_top1": False,
                "exact_top3": False,
                "hierarchical_top1": False,
                "candidates": [],
                "error": str(exc),
            }
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        results.append(result)
        if not args.backend and args.interval:
            time.sleep(args.interval)
    summary = summarize(results)
    passed = summary["exact_top1_all_pct"] >= args.min_accuracy
    report = {
        "metadata": metadata,
        "summary": summary,
        "threshold_pct": args.min_accuracy,
        "passed": passed,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"{'PASS' if passed else 'FAIL'}: {summary['correct']}/{summary['total']}; {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
