"""The acceptance probe must reject slow or incomplete batches, not just HTTP 200."""

import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "acceptance_probe", Path(__file__).resolve().parents[2] / "scripts/check_acceptance.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def exercise(monkeypatch, *, step=0.1, count=100):
    queries = [{"query": f"Product {i}"} for i in range(100)]
    ticks = iter(i * step for i in range(200))
    monkeypatch.setattr(probe.time, "perf_counter", lambda: next(ticks))

    class Response:
        def __init__(self, value):
            self.value = value

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(self.value).encode()

    def urlopen(req, **kwargs):
        if req.full_url.endswith("/health/ready"):
            return Response({"status": "ready"})
        if req.full_url.endswith("/classify/batch"):
            return Response({"job_id": "qa"})
        if req.full_url.endswith("/classify/batch/qa"):
            return Response(
                {
                    "status": "completed",
                    "results": [
                        {**row, "confidence": 0.9, "code": "1", "clause_url": "/source/1"}
                        for row in queries[:count]
                    ],
                }
            )
        return Response({"code": "1"})

    monkeypatch.setattr(probe.urllib.request, "urlopen", urlopen)
    return probe.run("http://test", queries)


def test_complete_fast_batch_passes(monkeypatch):
    assert exercise(monkeypatch)["pass"]


def test_http_success_with_missing_results_fails(monkeypatch):
    assert not exercise(monkeypatch, count=99)["pass"]


def test_slow_single_request_fails(monkeypatch):
    report = exercise(monkeypatch, step=6)
    assert report["batch"]["pass"]
    assert not report["pass"]
