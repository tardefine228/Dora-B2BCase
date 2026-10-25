"""Verdict export for training (BE-2 → ML-4).

The feedback log used to be write-only: `POST /feedback` stored verdicts and
nothing could read them back, so hard-negative mining had a mechanism and no
data. These tests pin the whole path a specialist's "неверно" takes:

    POST /feedback  →  GET /admin/feedback/export  →  ml.hard_negatives.load_feedback

The export is checked against the *real* training loader rather than against a
hand-written expectation: a column rename would otherwise pass here and break
`python -m ml.train_reranker --feedback export.csv` in production.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3

import pytest
from app.core.config import settings
from app.main import app
from app.services.feedback_store import save_feedback
from fastapi.testclient import TestClient

from ml.hard_negatives import load_feedback, usable


@pytest.fixture()
def feedback_client(tmp_path, monkeypatch):
    """SQLite store in a temp dir with the admin key configured."""
    monkeypatch.setattr(settings, "job_store", "sqlite")
    monkeypatch.setattr(settings, "job_store_path", str(tmp_path / "jobs.sqlite"))
    monkeypatch.setattr(settings, "admin_api_key", "test-secret")
    yield TestClient(app), tmp_path


def _headers() -> dict[str, str]:
    return {"X-Admin-API-Key": "test-secret"}


def _post(client: TestClient, **overrides) -> None:
    payload = {
        "query": "Счетчик воды крыльчатый",
        "predicted_code": "28.13.14",
        "verdict": "incorrect",
        "corrected_code": "28.13.11",
        "comment": "похоже на счётчик для воды",
    }
    payload.update(overrides)
    response = client.post("/feedback", json=payload)
    assert response.status_code == 201, response.text


def test_comment_is_stored_and_survives_the_export(feedback_client) -> None:
    client, tmp_path = feedback_client
    _post(client)

    with sqlite3.connect(tmp_path / "classification_feedback.sqlite") as conn:
        row = conn.execute(
            "SELECT query, predicted_code, verdict, corrected_code, comment "
            "FROM classification_feedback"
        ).fetchone()
    assert row == (
        "Счетчик воды крыльчатый",
        "28.13.14",
        "incorrect",
        "28.13.11",
        "похоже на счётчик для воды",
    )


def test_export_round_trips_through_the_training_loader(feedback_client, tmp_path) -> None:
    """The file must be readable by `ml.hard_negatives.load_feedback` as is."""
    client, _tmp = feedback_client
    _post(client)
    _post(
        client,
        query="Газонокосилка Bosch",
        verdict="absent",
        corrected_code=None,
        comment=None,
    )

    response = client.get("/admin/feedback/export", headers=_headers())
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    assert response.headers["content-disposition"].endswith('.csv"')
    assert response.headers["x-feedback-rows"] == "2"
    assert response.headers["cache-control"] == "no-store"
    # Excel on a Russian locale needs the BOM; the loader reads utf-8-sig.
    assert response.content.startswith("\ufeff".encode())

    export_path = tmp_path / "feedback.csv"
    export_path.write_bytes(response.content)
    records = load_feedback(export_path)

    assert [record.query for record in records] == [
        "Счетчик воды крыльчатый",
        "Газонокосилка Bosch",
    ]
    assert records[0].proposed_code == "28.13.14"
    assert records[0].correct_code == "28.13.11"
    assert records[0].comment == "похоже на счётчик для воды"
    assert records[0].is_incorrect
    assert records[1].is_out_of_scope
    # Both rows carry training signal — the miner would take them.
    assert usable(records) == records


def test_export_uses_the_loader_column_names(feedback_client) -> None:
    client, _tmp = feedback_client
    _post(client)
    response = client.get("/admin/feedback/export", headers=_headers())
    header = next(csv.reader(io.StringIO(response.content.decode("utf-8-sig")), delimiter=";"))
    # `proposed_code` is the loader's name for the API's `predicted_code`.
    assert header[:5] == ["query", "proposed_code", "verdict", "correct_code", "comment"]


def test_jsonl_export_also_round_trips(feedback_client, tmp_path) -> None:
    client, _tmp = feedback_client
    _post(client)
    response = client.get("/admin/feedback/export?format=jsonl", headers=_headers())
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")

    lines = [line for line in response.text.splitlines() if line.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["proposed_code"] == "28.13.14"

    export_path = tmp_path / "feedback.jsonl"
    export_path.write_text(response.text, encoding="utf-8")
    records = load_feedback(export_path)
    assert records[0].query == "Счетчик воды крыльчатый"
    assert records[0].comment == "похоже на счётчик для воды"


def test_period_verdict_and_limit_filters(feedback_client, tmp_path) -> None:
    client, _tmp = feedback_client
    _post(client, query="Старый запрос", verdict="correct", corrected_code=None)
    _post(client, query="Новый запрос")
    # Age the first row so the period filter has something to exclude.
    with sqlite3.connect(tmp_path / "classification_feedback.sqlite") as conn:
        conn.execute(
            "UPDATE classification_feedback SET created_at = '2020-01-01 00:00:00' "
            "WHERE query = 'Старый запрос'"
        )

    since_2021 = client.get(
        "/admin/feedback/export?since=2021-01-01T00:00:00", headers=_headers()
    )
    assert since_2021.headers["x-feedback-rows"] == "1"
    assert "Новый запрос" in since_2021.content.decode("utf-8-sig")
    assert "Старый запрос" not in since_2021.content.decode("utf-8-sig")

    until_2021 = client.get(
        "/admin/feedback/export?until=2021-01-01T00:00:00", headers=_headers()
    )
    assert until_2021.headers["x-feedback-rows"] == "1"
    assert "Старый запрос" in until_2021.content.decode("utf-8-sig")

    only_correct = client.get("/admin/feedback/export?verdict=correct", headers=_headers())
    assert only_correct.headers["x-feedback-rows"] == "1"
    assert "Старый запрос" in only_correct.content.decode("utf-8-sig")

    limited = client.get("/admin/feedback/export?limit=1", headers=_headers())
    assert limited.headers["x-feedback-rows"] == "1"


def test_empty_log_exports_a_header_only_file(feedback_client, tmp_path) -> None:
    client, _tmp = feedback_client
    response = client.get("/admin/feedback/export", headers=_headers())
    assert response.status_code == 200
    assert response.headers["x-feedback-rows"] == "0"
    export_path = tmp_path / "empty.csv"
    export_path.write_bytes(response.content)
    assert load_feedback(export_path) == []


def test_export_requires_the_admin_key(feedback_client, monkeypatch) -> None:
    client, _tmp = feedback_client
    assert client.get("/admin/feedback/export").status_code == 403
    wrong_key = client.get(
        "/admin/feedback/export", headers={"X-Admin-API-Key": "nope"}
    )
    assert wrong_key.status_code == 403
    monkeypatch.setattr(settings, "admin_api_key", "")
    assert client.get("/admin/feedback/export", headers=_headers()).status_code == 503


def test_legacy_table_without_comment_is_migrated_in_place(feedback_client, tmp_path) -> None:
    """A database created before the comment field keeps its rows."""
    client, _tmp = feedback_client
    location = tmp_path / "classification_feedback.sqlite"
    with sqlite3.connect(location) as conn:
        conn.execute(
            """CREATE TABLE classification_feedback (
              id TEXT PRIMARY KEY,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              query TEXT NOT NULL, predicted_code TEXT,
              verdict TEXT NOT NULL, corrected_code TEXT
            )"""
        )
        conn.execute(
            "INSERT INTO classification_feedback (id, query, predicted_code, verdict, "
            "corrected_code) VALUES ('old', 'Старый вердикт', '01.1', 'incorrect', '01.2')"
        )

    _post(client, query="Новый вердикт")

    with sqlite3.connect(location) as conn:
        rows = conn.execute(
            "SELECT query, comment FROM classification_feedback ORDER BY created_at"
        ).fetchall()
    assert ("Старый вердикт", None) in rows
    assert ("Новый вердикт", "похоже на счётчик для воды") in rows

    export = client.get("/admin/feedback/export", headers=_headers())
    assert export.headers["x-feedback-rows"] == "2"
    assert "Старый вердикт" in export.content.decode("utf-8-sig")


def test_comment_is_optional_and_trimmed(feedback_client, tmp_path) -> None:
    client, _tmp = feedback_client
    response = client.post(
        "/feedback",
        json={"query": "Мышь компьютерная", "verdict": "correct", "comment": "   "},
    )
    assert response.status_code == 201
    exported = client.get("/admin/feedback/export", headers=_headers())
    row = next(
        csv.DictReader(
            io.StringIO(exported.content.decode("utf-8-sig")), delimiter=";"
        )
    )
    assert row["comment"] == ""


def test_overlong_comment_is_rejected(feedback_client) -> None:
    client, _tmp = feedback_client
    response = client.post(
        "/feedback",
        json={"query": "Мышь компьютерная", "verdict": "correct", "comment": "я" * 1001},
    )
    assert response.status_code == 422


def test_export_of_unreachable_storage_returns_503(feedback_client, monkeypatch) -> None:
    client, _tmp = feedback_client
    monkeypatch.setattr(settings, "job_store", "postgres")
    monkeypatch.setattr(settings, "database_url", "postgresql://nobody@127.0.0.1:1/none")
    response = client.get("/admin/feedback/export", headers=_headers())
    assert response.status_code == 503


def test_save_feedback_accepts_comment_keyword(feedback_client, tmp_path) -> None:
    """The service function keeps working for callers that bypass the API."""
    _client, tmp_path = feedback_client
    save_feedback("Проверка", "01.1", "correct", None, "комментарий")
    with sqlite3.connect(tmp_path / "classification_feedback.sqlite") as conn:
        assert conn.execute(
            "SELECT comment FROM classification_feedback WHERE query = 'Проверка'"
        ).fetchone() == ("комментарий",)
