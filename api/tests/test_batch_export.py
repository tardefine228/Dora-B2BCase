"""Batch results over the wire: CSV export and the SSE progress stream."""

from __future__ import annotations

import csv
import io
import json

from app.job_store import add_batch_result, complete_batch, create_job, get_store, start_job
from app.main import app
from app.schemas import ClassifyResponse
from fastapi.testclient import TestClient

client = TestClient(app)

CSV_HEADER = (
    "query;code;title;status;confidence;clause_ref;clause_url;"
    "alt1_code;alt1_title;alt2_code;alt2_title;alt3_code;alt3_title"
)


def _run_batch(items: list[str]) -> str:
    accepted = client.post("/classify/batch", json={"items": items})
    assert accepted.status_code == 202
    return accepted.json()["job_id"]


def test_csv_export_has_bom_separators_and_headers() -> None:
    job_id = _run_batch(["Электрочайники", "Мышь компьютерная"])
    response = client.get(f"/classify/batch/{job_id}.csv")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert response.headers["content-disposition"] == f'attachment; filename="batch-{job_id}.csv"'
    assert response.content.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM for Excel

    text = response.content.decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text), delimiter=";"))
    assert rows[0] == CSV_HEADER.split(";")
    assert len(rows) == 3
    assert rows[1][0] == "Электрочайники"
    assert rows[1][1] == "27.51.24.110"
    assert rows[1][3] == "ok"
    assert rows[1][6] == "/source/2-339-01"
    # The status of the second row depends on the trained artifact (the
    # heuristic fallback answers "ok" where the model stays silent), so only
    # the shape of the column is asserted here.
    assert rows[2][0] == "Мышь компьютерная"
    assert rows[2][3] in {"ok", "review", "not_found"}


def test_csv_export_escapes_quotes_and_separators() -> None:
    job_id = _run_batch(['Кабель "витая пара"; экранированный'])
    text = client.get(f"/classify/batch/{job_id}.csv").content.decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text), delimiter=";"))

    assert rows[1][0] == 'Кабель "витая пара"; экранированный'
    assert len(rows[1]) == len(CSV_HEADER.split(";"))


def test_csv_export_rejects_unknown_and_unfinished_jobs() -> None:
    unknown = client.get("/classify/batch/unknown-job.csv")
    assert unknown.status_code == 404

    # A queued job is honest about not being ready yet.
    job = create_job(["Товар"])
    try:
        unfinished = client.get(f"/classify/batch/{job.job_id}.csv")
        assert unfinished.status_code == 409
        assert "выполняется" in unfinished.json()["detail"]
    finally:
        get_store().fail(job.job_id, "тест")


def test_csv_export_of_failed_job_returns_partial_results() -> None:
    job = create_job(["Товар 1", "Товар 2"])
    assert start_job(job.job_id)
    assert add_batch_result(
        job.job_id,
        ClassifyResponse(
            query="Товар 1",
            code="26.20.11",
            title="Компьютеры портативные",
            confidence=0.9,
            status="ok",
            latency_ms=1.0,
        ),
    )
    assert get_store().fail(job.job_id, "тестовый сбой")

    response = client.get(f"/classify/batch/{job.job_id}.csv")
    assert response.status_code == 200
    rows = list(csv.reader(io.StringIO(response.content.decode("utf-8-sig")), delimiter=";"))
    assert len(rows) == 2
    assert rows[1][0] == "Товар 1"


def test_events_stream_reports_progress_and_done() -> None:
    job_id = _run_batch(["Ноутбук Lenovo", "Мышь компьютерная"])

    with client.stream("GET", f"/classify/batch/{job_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["x-accel-buffering"] == "no"

        events: list[tuple[str, dict]] = []
        event_name = ""
        for line in response.iter_lines():
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                events.append((event_name, json.loads(line.removeprefix("data: "))))
                if event_name == "done":
                    break

    names = [name for name, _payload in events]
    assert "progress" in names
    assert names[-1] == "done"
    progress = next(payload for name, payload in events if name == "progress")
    assert progress["total"] == 2
    assert 0 <= progress["completed"] <= 2
    done = events[-1][1]
    assert done["status"] == "completed"
    assert done["completed"] == 2
    assert len(done["results"]) == 2


def test_events_unknown_job_returns_404_before_streaming() -> None:
    response = client.get("/classify/batch/unknown-job/events")
    assert response.status_code == 404
    assert "не найдена" in response.json()["detail"]


def test_events_heartbeat_keeps_a_slow_job_alive(monkeypatch) -> None:
    """A slow job keeps the stream open with ``: keep-alive`` comments.

    The stream is drained directly from the async generator: ``TestClient``
    buffers the whole body, so a live heartbeat cannot be observed through it.
    """
    import asyncio

    from app import main as main_module

    monkeypatch.setattr(main_module, "SSE_HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(main_module, "SSE_POLL_SECONDS", 0.01)

    job = create_job(["Медленный товар"])
    assert start_job(job.job_id)
    response = main_module.stream_batch_events(job.job_id)

    async def drain() -> list[str]:
        chunks: list[str] = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
            if chunk.startswith(": keep-alive"):
                complete_batch(job.job_id)
            if chunk.startswith("event: done"):
                return chunks
            if len(chunks) > 500:  # pragma: no cover - safety net
                raise AssertionError("поток SSE не завершился")
        return chunks

    chunks = asyncio.run(asyncio.wait_for(drain(), timeout=10))
    assert any(chunk.startswith(": keep-alive") for chunk in chunks)
    assert chunks[-1].startswith("event: done")


def test_events_of_a_failed_job_end_with_done_and_error() -> None:
    job = create_job(["Товар"])
    assert start_job(job.job_id)
    get_store().fail(job.job_id, "сбой обработки")

    with client.stream("GET", f"/classify/batch/{job.job_id}/events") as response:
        body = "".join(response.iter_text())

    assert "event: done" in body
    payload = json.loads(body.split("data: ")[-1].split("\n")[0])
    assert payload["status"] == "failed"
    assert payload["error"] == "сбой обработки"
