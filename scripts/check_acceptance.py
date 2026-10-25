"""Measure the running API against the task's timing limits (standard library only).
Input: UTF-8 CSV with query and optional expected_code. Uses the first 100 distinct queries.
This creates ordinary classification jobs/logs; it never updates the catalog or model.
"""

import argparse
import csv
import json
import time
import urllib.request
from pathlib import Path


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
        answer.get("query") == row["query"]
        and isinstance(answer.get("confidence"), (float, int))
        and 0 <= answer["confidence"] <= 1
        and (not answer.get("code") or bool(answer.get("clause_url")))
        for row, answer in zip(queries, results, strict=False)
    )
    report["result_contract_pass"] = contract_ok
    report["batch"] = {
        "seconds": elapsed,
        "status": state["status"],
        "count": len(results),
        "pass": elapsed < 120 and state["status"] == "completed" and len(results) == 100,
    }
    for row in queries[:20]:
        start = time.perf_counter()
        answer = request("/classify", {"name": row["query"]}, timeout=30)
        elapsed = time.perf_counter() - start
        report["single"].append(
            {
                "query": row["query"],
                "seconds": elapsed,
                "pass": elapsed < 5,
                "code": answer.get("code"),
            }
        )
    # Accuracy is reported only for supplied labels, never inferred from catalog titles.
    labeled = [
        (row, answer)
        for row, answer in zip(queries, results, strict=False)
        if row.get("expected_code", "").strip()
    ]
    report["accuracy"] = {
        "labeled": len(labeled),
        "correct": sum(
            row["expected_code"].strip() == answer.get("code") for row, answer in labeled
        ),
    }
    report["pass"] = (
        contract_ok and report["batch"]["pass"] and all(r["pass"] for r in report["single"])
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
    report = run(args.base_url, list(unique.values())[:100])
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
