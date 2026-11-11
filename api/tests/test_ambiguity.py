from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest
from app.core.config import settings
from app.job_store import SqliteJobStore
from app.main import app
from app.schemas import ClassifyResponse
from app.services.ambiguity import AmbiguityConflict, AmbiguityGuard
from app.services.classifier import CatalogClassifier, reset_classifier
from fastapi.testclient import TestClient
from pydantic import ValidationError

client = TestClient(app)
CATALOG = Path(__file__).resolve().parents[2] / "data" / "data.csv"


@pytest.fixture(autouse=True)
def lexical_backend():
    previous = settings.vector_search
    settings.vector_search = False
    reset_classifier()
    yield
    settings.vector_search = previous
    reset_classifier()


def test_needs_clarification_schema_forbids_an_answer() -> None:
    with pytest.raises(ValidationError):
        ClassifyResponse.model_validate(
            {
                "query": "автомобиль",
                "code": "29.10.2",
                "title": "Автомобили легковые",
                "confidence": 0.9,
                "status": "needs_clarification",
                "latency_ms": 0,
            }
        )


def test_all_curated_product_groups_ask_instead_of_forcing_top1() -> None:
    classifier = CatalogClassifier(CATALOG)
    for query in (
        "автомобиль",
        "мебель",
        "светильник",
        "кран",
        "компьютер",
        "счетчик",
        "фильтр",
        "насос",
    ):
        answer = classifier.classify(query)
        assert answer.status == "needs_clarification", query
        assert answer.code is None and answer.title is None
        assert answer.confidence == 0
        assert answer.clarification is not None
        assert len(answer.clarification.options) >= 2
        assert answer.clarification.original.code is not None
        assert answer.alternatives == []


def test_specific_modifiers_and_direct_code_do_not_trigger_generic_rule() -> None:
    classifier = CatalogClassifier(CATALOG)
    expected = {
        "легковой автомобиль": "29.10.2",
        "шаровый кран": "28.14.13.131",
        "счетчик электроэнергии": "26.51.63.130",
        "Ноутбук Lenovo": "26.20.11",
        "бумага": "17.12",
        "Обувь": "15.2",
        "Одежда": "14.1",
        "29.10.59.160": "29.10.59.160",
    }
    for query, code in expected.items():
        answer = classifier.classify(query)
        assert answer.status != "needs_clarification", query
        assert answer.code == code, query
    assert classifier.classify("29.10.59.160").decision_source == "direct_code"


def test_exact_title_with_multiple_codes_is_not_a_top1_answer() -> None:
    classifier = CatalogClassifier(CATALOG)
    answer = classifier.classify("Фильтр дыхательный (тепловлагообменник/бактериальный фильтр)")
    assert answer.status == "needs_clarification"
    assert answer.clarification is not None
    assert answer.attention_reasons[0].code == "multiple_codes"
    assert len({option.code for option in answer.clarification.options}) >= 2


def test_single_resolution_validates_option_and_catalog_hash() -> None:
    pending = client.post("/classify", json={"name": "автомобиль"}).json()
    clarification = pending["clarification"]
    selected = next(option for option in clarification["options"] if option["code"] == "29.10.2")

    invalid_option = client.post(
        "/classify/resolve",
        json={
            "query": "автомобиль",
            "option_id": "not-issued",
            "catalog_sha256": clarification["catalog_sha256"],
        },
    )
    assert invalid_option.status_code == 409

    stale = client.post(
        "/classify/resolve",
        json={
            "query": "автомобиль",
            "option_id": selected["option_id"],
            "catalog_sha256": "0" * 64,
        },
    )
    assert stale.status_code == 409

    response = client.post(
        "/classify/resolve",
        json={
            "query": "автомобиль",
            "option_id": selected["option_id"],
            "catalog_sha256": clarification["catalog_sha256"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["code"] == "29.10.2"
    assert body["decision_source"] == "specialist"
    assert body["resolution"]["option_id"] == selected["option_id"]
    assert body["resolution"]["original"]["code"] == "29.10.59.140"
    assert body["clarification"] is None


def test_resolver_rebuilds_options_instead_of_trusting_a_tampered_pending_object() -> None:
    classifier = CatalogClassifier(CATALOG)
    pending = classifier.classify("автомобиль")
    computer = classifier.classify("компьютер")
    forged = pending.model_copy(deep=True)
    assert forged.clarification is not None and computer.clarification is not None
    foreign = computer.clarification.options[0]
    forged.clarification.options.append(foreign)
    with pytest.raises(AmbiguityConflict):
        classifier._ambiguity.resolve_answer(
            forged, foreign.option_id, pending.catalog_sha256 or ""
        )

    selected = next(
        option for option in pending.clarification.options if option.code == "29.10.2"
    )
    expected_confidence = selected.confidence
    forged.clarification.options.remove(foreign)
    tampered = next(
        option for option in forged.clarification.options if option.code == "29.10.2"
    )
    tampered.confidence = 1.0
    resolved = classifier._ambiguity.resolve_answer(
        forged, selected.option_id, pending.catalog_sha256 or ""
    )
    assert resolved.confidence == expected_confidence
    assert resolved.resolution is not None
    assert resolved.resolution.original == pending.clarification.original


def test_batch_resolution_is_persisted_idempotently_rejected_and_exported() -> None:
    accepted = client.post("/classify/batch", json={"items": ["автомобиль", "Ноутбук Lenovo"]})
    assert accepted.status_code == 202
    job_id = accepted.json()["job_id"]
    state = client.get(f"/classify/batch/{job_id}").json()
    assert state["status"] == "completed"
    pending = state["results"][0]
    assert pending["status"] == "needs_clarification"
    assert pending["item_id"] == 0
    selected = next(
        option for option in pending["clarification"]["options"] if option["code"] == "29.10.2"
    )
    payload = {
        "option_id": selected["option_id"],
        "catalog_sha256": pending["clarification"]["catalog_sha256"],
    }
    response = client.post(f"/classify/batch/{job_id}/items/0/resolve", json=payload)
    assert response.status_code == 200
    assert response.json()["code"] == "29.10.2"

    # A second click must not replace the saved specialist decision.
    repeated = client.post(f"/classify/batch/{job_id}/items/0/resolve", json=payload)
    assert repeated.status_code == 409
    restored = client.get(f"/classify/batch/{job_id}").json()["results"][0]
    assert restored["decision_source"] == "specialist"
    assert restored["resolution"]["original"]["code"] == "29.10.59.140"

    export = client.get(f"/classify/batch/{job_id}.csv")
    rows = list(csv.DictReader(io.StringIO(export.content.decode("utf-8-sig")), delimiter=";"))
    assert rows[0]["code"] == "29.10.2"
    assert rows[0]["decision_source"] == "specialist"
    assert rows[0]["selected_option_id"] == selected["option_id"]
    assert rows[1]["code"] == "26.20.11"


def test_pending_batch_csv_contains_question_and_options_without_code() -> None:
    accepted = client.post("/classify/batch", json={"items": ["автомобиль"]})
    job_id = accepted.json()["job_id"]
    export = client.get(f"/classify/batch/{job_id}.csv")
    row = next(csv.DictReader(io.StringIO(export.content.decode("utf-8-sig")), delimiter=";"))
    assert row["status"] == "needs_clarification"
    assert row["code"] == row["title"] == ""
    assert "Какой именно автомобиль" in row["clarification_question"]
    assert "29.10.2" in row["clarification_options"]
    assert row["decision_source"] == "model"


def test_resolution_preserves_catalog_review_and_original_alternatives() -> None:
    rows = [
        {
            "record_id": "1-92-01",
            "name": "Автомобили легковые",
            "okpd2": "29.10.2",
            "code_role": "Включён",
            "catalog_review_required": True,
        },
        {
            "record_id": "1-94-01",
            "name": "Средства автотранспортные грузовые",
            "okpd2": "29.10.4",
            "code_role": "Включён",
            "catalog_review_required": False,
        },
    ]
    guard = AmbiguityGuard(rows, "a" * 64)
    original = ClassifyResponse(
        query="автомобиль",
        code="29.10.59.140",
        title="Автомобили пожарные",
        confidence=0.9,
        confidence_raw=0.9,
        status="ok",
        alternatives=[{"code": "29.10.2", "title": "Автомобили легковые", "confidence": 0.4}],
        latency_ms=0,
    )
    pending = guard.apply("автомобиль", original)
    assert pending.clarification.original.alternatives[0].code == "29.10.2"
    resolved = guard.resolve_answer(pending, "1-92-01", "a" * 64)
    assert resolved.status == "review"
    assert resolved.code == "29.10.2"
    assert {reason.code for reason in resolved.attention_reasons} == {"catalog_review_required"}
    with pytest.raises(AmbiguityConflict):
        guard.resolve_answer(original, "1-92-01", "a" * 64)


def test_duplicate_code_and_title_is_one_interpretation_not_two_options() -> None:
    row = {
        "name": "Автомобили легковые",
        "okpd2": "29.10.2",
        "code_role": "Включён",
        "catalog_review_required": False,
    }
    guard = AmbiguityGuard(
        [{**row, "record_id": "1-92-01"}, {**row, "record_id": "duplicate"}],
        "a" * 64,
    )
    original = ClassifyResponse(
        query="автомобиль",
        code="29.10.2",
        title="Автомобили легковые",
        confidence=0.9,
        status="ok",
        latency_ms=0,
    )
    assert guard.apply("автомобиль", original).status == "ok"


def test_sqlite_store_keeps_specialist_resolution(tmp_path) -> None:
    store = SqliteJobStore(tmp_path / "jobs.sqlite")
    job = store.create_batch(["автомобиль"])
    assert store.start(job.job_id)
    pending = CatalogClassifier(CATALOG).classify("автомобиль")
    assert store.add_batch_result(job.job_id, pending)
    assert store.complete_batch(job.job_id)
    selected = pending.clarification.options[0]
    resolved = CatalogClassifier(CATALOG).resolve_clarification(
        "автомобиль", selected.option_id, pending.clarification.catalog_sha256
    )
    assert store.update_result_if_pending(
        job.job_id, 0, pending.clarification.catalog_sha256, resolved.model_dump(mode="json")
    )
    restored = store.get(job.job_id).results[0]
    assert restored.code == selected.code
    assert restored.decision_source == "specialist"
    assert (
        store.update_result_if_pending(
            job.job_id, 0, pending.clarification.catalog_sha256, resolved.model_dump(mode="json")
        )
        is None
    )
    store.close()
