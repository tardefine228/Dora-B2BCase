from __future__ import annotations

import csv
import io

import pytest
from app.core.config import settings
from app.main import app
from app.schemas import ClassifyResponse
from app.services.classifier import reset_classifier
from app.services.procurement_report import build_procurement_report
from fastapi.testclient import TestClient

client = TestClient(app)


@pytest.fixture(autouse=True)
def lexical_backend():
    previous = settings.vector_search
    settings.vector_search = False
    reset_classifier()
    yield
    settings.vector_search = previous
    reset_classifier()


def result(status: str, **changes) -> ClassifyResponse:
    values = {
        "query": "товар",
        "code": "29.10.2" if status in {"ok", "review"} else None,
        "title": "Автомобили легковые" if status in {"ok", "review"} else None,
        "confidence": 0.9,
        "status": status,
        "code_role": "Включён",
        "catalog_review_required": False,
        "latency_ms": 0,
    }
    values.update(changes)
    return ClassifyResponse(**values)


def test_buckets_are_exclusive_and_reasons_can_overlap() -> None:
    rows = [
        result("ok"),
        result("review"),
        result("not_found", code_role=None, catalog_review_required=None),
        result(
            "review",
            attention_reasons=[
                {"code": "code_exception", "message": "исключение"},
                {"code": "catalog_review_required", "message": "сверить"},
            ],
            code_role="Исключение",
            catalog_review_required=True,
        ),
    ]
    report = build_procurement_report(rows)
    assert report.summary.total == 4
    assert report.summary.accepted == 1
    assert report.summary.needs_review == 2
    assert report.summary.not_found == 1
    assert report.summary.auto_accepted == 1
    assert report.summary.specialist_resolved == 0
    assert report.summary.reason_counts["code_exception"] == 1
    assert report.summary.reason_counts["catalog_review_required"] == 1
    assert report.items[3].risk_rank == 95
    assert [item.risk_rank for item in report.items] == [0, 70, 60, 95]
    assert sum(
        item.bucket == bucket for item in report.items for bucket in
        ("accepted", "needs_review", "not_found")
    ) == len(rows)


def test_specialist_review_is_not_silently_accepted() -> None:
    original = {
        "code": "29.10.59.140", "title": "Автомобили пожарные",
        "status": "ok", "confidence": 0.8,
    }
    resolved = result(
        "review",
        decision_source="specialist",
        resolution={
            "option_id": "1-92-01", "resolved_at": "2026-11-11T00:00:00Z",
            "original": original,
        },
        attention_reasons=[{"code": "catalog_review_required", "message": "сверить"}],
        catalog_review_required=True,
    )
    report = build_procurement_report([resolved])
    assert report.summary.needs_review == 1
    assert report.summary.accepted == report.summary.specialist_resolved == 0
    assert report.items[0].risk_rank == 85


def test_batch_api_summary_and_csv_follow_saved_resolution() -> None:
    accepted = client.post(
        "/classify/batch",
        json={"items": ["автомобиль", "Ноутбук Lenovo", "Ручка шариковая синяя"]},
    )
    job_id = accepted.json()["job_id"]
    state = client.get(f"/classify/batch/{job_id}").json()
    assert state["status"] == "completed"
    assert state["report"]["summary"]["total"] == 3
    assert state["report"]["summary"]["needs_review"] == 1
    assert state["report"]["summary"]["not_found"] == 1
    pending = state["results"][0]
    report_item = state["report"]["items"][0]
    assert report_item["bucket"] == "needs_review"
    assert report_item["risk_rank"] == 90
    assert report_item["reasons"][0]["code"] == "ambiguous_name"

    option = next(
        option for option in pending["clarification"]["options"]
        if option["code"] == "29.10.2"
    )
    response = client.post(
        f"/classify/batch/{job_id}/items/0/resolve",
        json={"option_id": option["option_id"], "catalog_sha256": pending["catalog_sha256"]},
    )
    assert response.status_code == 200
    state = client.get(f"/classify/batch/{job_id}").json()
    summary = state["report"]["summary"]
    assert summary["accepted"] == 2
    assert summary["specialist_resolved"] == 1
    assert summary["needs_review"] == 0
    assert state["report"]["items"][0]["risk_rank"] == 0

    export = client.get(f"/classify/batch/{job_id}.csv")
    rows = list(csv.DictReader(io.StringIO(export.content.decode("utf-8-sig")), delimiter=";"))
    assert rows[0]["report_bucket"] == "accepted"
    assert rows[0]["risk_rank"] == "0"
    assert rows[0]["decision_source"] == "specialist"
    assert rows[1]["report_bucket"] == "accepted"
    assert rows[2]["report_bucket"] == "not_found"
    assert "not_found" in rows[2]["report_reasons"]
