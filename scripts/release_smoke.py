"""Measure real API requests on distinct product names, without synthetic filler.

Run against a freshly started API (empty classification cache):
    python scripts/release_smoke.py --base-url http://127.0.0.1:8000

This is a functional/latency check, not an independent accuracy benchmark.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def run(base_url: str) -> dict:
    with (ROOT / "eval/labeled_examples.csv").open(encoding="utf-8-sig") as handle:
        names = list(dict.fromkeys(row["query"] for row in csv.DictReader(handle, delimiter=";")))
    if len(names) < 120:
        raise ValueError("Need at least 120 distinct product queries")
    durations = []
    with httpx.Client(base_url=base_url, timeout=30, trust_env=False) as client:
        ready = client.get("/health/ready")
        ready.raise_for_status()
        assert ready.json()["calibrated"], "Current trained model must be loaded"
        for query in names[:20]:
            started = time.perf_counter()
            response = client.post("/classify", json={"name": query})
            response.raise_for_status()
            assert response.json()["query"] == query
            durations.append(time.perf_counter() - started)
            time.sleep(0.1)  # Respect the default public rate limit.

        # No query in this batch was used by the single-request checks above.
        batch_names = names[20:120]
        started = time.perf_counter()
        accepted = client.post("/classify/batch", json={"items": batch_names})
        accepted.raise_for_status()
        status_url = accepted.json()["status_url"]
        while True:
            response = client.get(status_url)
            response.raise_for_status()
            state = response.json()
            if state["status"] in {"completed", "failed"}:
                break
            if time.perf_counter() - started >= 120:
                raise TimeoutError("Batch exceeded the case requirement of 120 seconds")
            time.sleep(0.25)
        batch_seconds = time.perf_counter() - started
        assert state["status"] == "completed", state.get("error")
        assert [item["query"] for item in state["results"]] == batch_names
        export = client.get(status_url + ".csv")
        export.raise_for_status()
        rows = list(csv.reader(io.StringIO(export.content.decode("utf-8-sig")), delimiter=";"))
        assert len(rows) == 101

        source = client.get("/source/2-339-01?format=html")
        source.raise_for_status()
        assert "text/html" in source.headers["content-type"]
        assert "Электрочайники" in source.text
        ui = client.get("/ui/")
        ui.raise_for_status()
        assert "/ui/assets/" in ui.text
    result = {
        "environment": "local API, catalog backend; not Docker",
        "accuracy_benchmark": False,
        "single_queries": len(durations),
        "single_median_seconds": round(statistics.median(durations), 4),
        "single_max_seconds": round(max(durations), 4),
        "batch_items": 100,
        "batch_seconds": round(batch_seconds, 4),
        "batch_order_and_csv_verified": True,
        "source_html_and_built_ui_served": True,
        "browser_rendering_verified": False,
        "readiness": ready.json(),
    }
    assert max(durations) < 5, result
    assert batch_seconds < 120, result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(run(args.base_url.rstrip("/")), ensure_ascii=False, indent=2)
    print(report)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
