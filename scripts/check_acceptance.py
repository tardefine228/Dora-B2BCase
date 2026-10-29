"""Measure API timing, result contracts and labeled accuracy (standard library only).

Input: UTF-8 CSV with query and expected_code (or expected_okpd2).
Uses the first 100 distinct queries; requires at least 20 labeled goods.
Creates ordinary classification jobs/logs; never updates the catalog or model.
"""

import argparse
import csv
import hashlib
import json
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


def expected_code(row):
    # An unclear or explicitly absent item cannot demonstrate code classification accuracy.
    if row.get("label_kind", "").strip() in ("unclear", "absent"):
        return ""
    return (row.get("expected_code") or row.get("expected_okpd2") or "").strip()


def valid_result(answer, query):
    confidence = answer.get("confidence")
    return (
        answer.get("query") == query
        and type(confidence) in (float, int)
        and 0 <= confidence <= 1
        and answer.get("status") in ("ok", "review", "not_found")
        and (
            (answer["status"] == "not_found" and not answer.get("code"))
            or (
                answer["status"] != "not_found"
                and bool(answer.get("code"))
                and bool(answer.get("clause_url"))
            )
        )
    )


def run(base, queries):
    def request(path, body=None, timeout=10):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            base.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)

    report = {"readiness": request("/health/ready"), "single": []}
    # Batch first: single-query probes must not warm these queries in the worker.
    start = time.perf_counter()
    job = request("/classify/batch", {"items": [q["query"] for q in queries]})
    state = {"status": "timeout", "results": []}
    while time.perf_counter() - start < 120:
        remaining = 120 - (time.perf_counter() - start)
        try:
            state = request(
                "/classify/batch/" + job["job_id"], timeout=max(0.1, min(10, remaining))
            )
        except TimeoutError:
            break
        if state["status"] in ("completed", "failed"):
            break
        time.sleep(min(0.5, max(0, remaining)))
    else:
        state = {"status": "timeout", "results": []}
    elapsed = time.perf_counter() - start
    results = state.get("results", [])
    contract_ok = len(results) == 100 and all(
        valid_result(answer, row["query"]) for row, answer in zip(queries, results, strict=False)
    )
    report["results"] = results
    report["result_contract_pass"] = contract_ok
    report["batch"] = {
        "job_id": job["job_id"],
        "seconds": elapsed,
        "status": state["status"],
        "count": len(results),
        "pass": elapsed < 120 and state["status"] == "completed" and len(results) == 100,
    }
    for row in [row for row in queries if expected_code(row)][:20]:
        start = time.perf_counter()
        answer = request("/classify", {"name": row["query"]}, timeout=30)
        elapsed = time.perf_counter() - start
        report["single"].append(
            {
                "query": row["query"],
                "seconds": elapsed,
                "pass": elapsed < 5 and valid_result(answer, row["query"]),
                "code": answer.get("code"),
                "expected_code": expected_code(row),
                "correct": answer.get("code") == expected_code(row),
            }
        )
    # Accuracy is reported only for supplied labels, never inferred from catalog titles.
    labeled = [(i, row) for i, row in enumerate(queries) if expected_code(row)]
    correct = sum(
        i < len(results)
        and results[i].get("query") == row["query"]
        and results[i].get("code") == expected_code(row)
        for i, row in labeled
    )
    accuracy = correct / len(labeled) if labeled else None
    report["accuracy"] = {
        "labeled": len(labeled),
        "correct": correct,
        "fraction": accuracy,
        "minimum_labeled": 20,
        "threshold": 0.85,
        "pass": len(labeled) >= 20 and accuracy >= 0.85,
        "mismatches": [
            {
                "query": row["query"],
                "expected": expected_code(row),
                "actual": results[i].get("code") if i < len(results) else None,
            }
            for i, row in labeled
            if i >= len(results) or results[i].get("code") != expected_code(row)
        ],
    }
    single_correct = sum(row["correct"] for row in report["single"])
    report["single_accuracy"] = {
        "labeled": len(report["single"]),
        "correct": single_correct,
        "pass": len(report["single"]) == 20 and single_correct >= 17,
    }
    report["pass"] = (
        report["single_accuracy"]["pass"]
        and contract_ok
        and report["batch"]["pass"]
        and report["accuracy"]["pass"]
        and len(report["single"]) == 20
        and all(r["pass"] for r in report["single"])
        and report["readiness"].get("status") == "ready"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--output", type=Path, default=Path("acceptance-report.json"))
    args = parser.parse_args()
    with args.csv.open(encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(4096)
        stream.seek(0)
        reader = csv.DictReader(stream, delimiter=";" if sample.splitlines()[0].count(";") else ",")
        unique = {}
        for row in reader:
            query = row.get("query", "").strip()
            if query:
                unique.setdefault(query, {**row, "query": query})
    if len(unique) < 100:
        parser.error("Need at least 100 distinct nonempty queries in column query")
    try:
        report = run(args.base_url, list(unique.values())[:100])
    except (OSError, ValueError, KeyError) as exc:
        report = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}
    report["checked_at"] = datetime.now(UTC).isoformat()
    report["input_sha256"] = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    report["input"] = args.csv.name
    report["base_url"] = args.base_url
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
