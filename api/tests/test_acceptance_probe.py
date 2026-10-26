"""The acceptance probe must reject slow or incomplete batches, not just HTTP 200."""

import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "acceptance_probe", Path(__file__).resolve().parents[2] / "scripts/check_acceptance.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def exercise(monkeypatch, *, step=0.1, count=100, labels=20, wrong=0, single_wrong=False):
    queries = [
        {"query": f"Product {i}", "expected_code": "1" if i < labels else ""} for i in range(100)
    ]
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
                        {
                            **row,
                            "confidence": 0.9,
                            "code": "2" if i < wrong else "1",
                            "status": "ok",
                            "clause_url": "/source/1",
                        }
                        for i, row in enumerate(queries[:count])
                    ],
                }
            )
        query = json.loads(req.data)["name"]
        return Response(
            {
                "query": query,
                "confidence": 0.9,
                "code": "2" if single_wrong else "1",
                "status": "ok",
                "clause_url": "/source/1",
            }
        )

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


def test_fast_but_inaccurate_batch_fails(monkeypatch):
    report = exercise(monkeypatch, wrong=4)
    assert report["batch"]["pass"]
    assert report["accuracy"]["fraction"] == 0.8
    assert not report["pass"]


def test_exact_accuracy_threshold_passes(monkeypatch):
    assert exercise(monkeypatch, wrong=3)["pass"]


def test_no_labels_cannot_claim_acceptance(monkeypatch):
    assert not exercise(monkeypatch, labels=0)["pass"]


def test_less_than_twenty_labels_cannot_claim_acceptance(monkeypatch):
    assert not exercise(monkeypatch, labels=19)["pass"]


def test_existing_dataset_label_column_and_unclear_exclusions():
    assert probe.expected_code({"expected_okpd2": "26.20.11", "label_kind": "code"}) == "26.20.11"
    assert probe.expected_code({"expected_okpd2": "26.20.11", "label_kind": "unclear"}) == ""


def test_result_contract_rejects_false_confidence_or_missing_clause():
    answer = {"query": "q", "status": "ok", "code": "1", "clause_url": "/source/1"}
    assert not probe.valid_result({**answer, "confidence": True}, "q")
    assert not probe.valid_result({**answer, "confidence": float("nan")}, "q")
    assert not probe.valid_result({**answer, "confidence": 0.9, "clause_url": None}, "q")
    assert probe.valid_result(
        {"query": "q", "status": "not_found", "code": None, "confidence": 0}, "q"
    )


def test_batch_accuracy_cannot_hide_incorrect_single_answers(monkeypatch):
    report = exercise(monkeypatch, single_wrong=True)
    assert report["accuracy"]["pass"]
    assert not report["single_accuracy"]["pass"]
    assert not report["pass"]
