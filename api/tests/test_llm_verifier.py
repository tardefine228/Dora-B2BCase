"""The optional LLM verifier must never invent a code or break an answer.

Every test here runs offline: the transport is injected, so no key, network or
money is involved. The point of the stage is narrow (see эксп. 30): confirm one
of the top-3 hypotheses for `review`/`not_found` rows, and otherwise get out of
the way.
"""

import json
from pathlib import Path

import pytest
from app.schemas import ClassifyResponse
from app.services.classifier import CatalogClassifier
from app.services.llm_verifier import (
    Candidate,
    LlmVerifier,
    build_llm_verifier,
)

OFFERED = [
    Candidate("26.20.2", "Устройства запоминающие и прочие устройства хранения данных"),
    Candidate("32.50.22.121", "Протезы внешние"),
]


def _catalog_path() -> Path:
    path = Path(__file__).resolve().parents[2] / "data" / "data.csv"
    assert path.exists(), path
    return path


def _verifier(reply: str | Exception, **kwargs):
    calls: list[dict] = []

    def transport(payload, timeout):
        calls.append(payload)
        if isinstance(reply, Exception):
            raise reply
        return reply

    verifier = LlmVerifier(
        base_url="https://example.invalid/v1",
        model="test-model",
        transport=transport,
        **kwargs,
    )
    return verifier, calls


def test_verifier_is_off_without_a_model_or_without_the_flag():
    class Settings:
        llm_verifier_enabled = False
        llm_verifier_model = "gpt-4o-mini"

    assert build_llm_verifier(Settings()) is None

    class HalfConfigured:
        llm_verifier_enabled = True
        llm_verifier_model = ""

    assert build_llm_verifier(HalfConfigured()) is None


def test_disabled_verifier_returns_nothing():
    verifier, calls = _verifier('{"choice": "26.20.2"}')
    verifier.model = ""
    assert verifier.verify("Внешний жёсткий диск", OFFERED) is None
    assert calls == []


def test_choice_must_be_one_of_the_offered_codes():
    verifier, _ = _verifier('{"choice": "26.20.2"}')
    assert verifier.verify("Внешний жёсткий диск", OFFERED).code == "26.20.2"

    # An invented code is not an answer: the model may not propose positions
    # that were not in the shortlist.
    verifier, _ = _verifier('{"choice": "84.13.10.000"}')
    assert verifier.verify("Внешний жёсткий диск", OFFERED) is None


@pytest.mark.parametrize(
    "reply",
    [
        '{"choice": "none"}',
        '{"choice": "NONE"}',
    ],
)
def test_none_is_an_answer_but_does_not_change_the_code(reply):
    verifier, _ = _verifier(reply)
    verdict = verifier.verify("Огнетушитель порошковый", OFFERED)
    assert verdict is not None and verdict.code is None


@pytest.mark.parametrize(
    "reply",
    [
        "",  # empty body
        "не JSON",  # prose
        '{"code": "26.20.2"}',  # wrong key
        '{"choice": 42}',  # wrong type
        '[{"choice": "26.20.2"}]',  # wrong shape
    ],
)
def test_unusable_replies_are_ignored(reply):
    verifier, _ = _verifier(reply)
    assert verifier.verify("Внешний жёсткий диск", OFFERED) is None


def test_fenced_json_is_accepted():
    verifier, _ = _verifier('```json\n{"choice": "26.20.2"}\n```')
    assert verifier.verify("Внешний жёсткий диск", OFFERED).code == "26.20.2"


def test_transport_failure_fails_open():
    verifier, _ = _verifier(RuntimeError("connection reset"))
    assert verifier.verify("Внешний жёсткий диск", OFFERED) is None
    assert verifier.stats()["failed"] == 1


def test_answers_are_cached_by_query_and_candidates():
    verifier, calls = _verifier('{"choice": "26.20.2"}')
    first = verifier.verify("Внешний жёсткий диск 2 ТБ", OFFERED)
    second = verifier.verify("  внешний   ЖЁСТКИЙ диск 2 тб ", OFFERED)
    assert first.code == second.code == "26.20.2"
    assert second.cached is True
    assert len(calls) == 1
    # A different shortlist is a different question, even for the same query.
    verifier.verify("Внешний жёсткий диск 2 ТБ", OFFERED[:1])
    assert len(calls) == 2


def test_per_minute_budget_caps_the_bill_and_keeps_the_answer():
    clock = [0.0]
    verifier, calls = _verifier('{"choice": "26.20.2"}', calls_per_minute=2, clock=lambda: clock[0])
    for index in range(2):
        assert verifier.verify(f"диск {index}", OFFERED).code == "26.20.2"
    assert verifier.verify("диск 3", OFFERED) is None  # budget spent, not an error
    assert len(calls) == 2 and verifier.stats()["skipped"] == 1
    # The window slides: a minute later the stage works again.
    clock[0] = 61.0
    assert verifier.verify("диск 4", OFFERED).code == "26.20.2"


def test_cache_size_can_be_disabled():
    verifier, calls = _verifier('{"choice": "26.20.2"}', cache_size=0)
    verifier.verify("диск", OFFERED)
    verifier.verify("диск", OFFERED)
    assert len(calls) == 2


def test_prompt_offers_only_three_positions_and_names_them():
    verifier, calls = _verifier('{"choice": "none"}')
    many = [Candidate(f"26.20.{index}", f"Позиция {index}") for index in range(1, 7)]
    verifier.verify("диск", many)
    user_message = calls[0]["messages"][1]["content"]
    assert user_message.count("\n") >= 3
    for index in range(1, 4):
        assert f"26.20.{index}" in user_message
    assert "26.20.4" not in user_message
    assert calls[0]["temperature"] == 0


# --------------------------------------------------------------------- wiring


def _classifier(reply: str | Exception) -> tuple[CatalogClassifier, list[dict]]:
    verifier, calls = _verifier(reply)
    classifier = CatalogClassifier(_catalog_path(), dense=None, verifier=verifier)
    return classifier, calls


def test_silence_on_a_listed_goods_is_rescued_as_a_candidate():
    # «Шлагбаум» перечня не содержит, и головное слово не якорится ни в одной
    # позиции, поэтому правило слабых подсказок (эксп. 34) молчит: до third
    # stage строка доходит пустой — с подсказками, но без кода.
    baseline = CatalogClassifier(_catalog_path(), dense=None)
    assert baseline.classify("Шлагбаум автоматический").status == "not_found"

    classifier, _ = _classifier('{"choice": "27.12.22"}')

    answer = classifier.classify("Шлагбаум автоматический")
    assert answer.code == "27.12.22"
    # A verified code is a candidate for a human, never a confident answer.
    assert answer.status == "review"
    assert answer.verifier == "llm-top3"
    assert answer.verifier_note
    assert answer.clause_ref and answer.clause_url


def test_confident_answers_do_not_reach_the_verifier():
    classifier, calls = _classifier('{"choice": "26.20.2"}')
    answer = classifier.classify("Ноутбук Lenovo")
    assert answer.status == "ok"
    assert answer.verifier is None
    assert calls == []  # no request, no money


def test_verifier_none_keeps_the_model_silent_but_records_the_check():
    classifier, _ = _classifier('{"choice": "none"}')
    answer = classifier.classify("Огнетушитель порошковый ОП-4")
    assert answer.status == "not_found"
    assert answer.code is None
    assert answer.verifier == "llm-top3"
    assert answer.verifier_note == "нет совпадений"


def test_invented_code_never_reaches_the_response():
    classifier, _ = _classifier('{"choice": "99.99.99"}')
    answer = classifier.classify("Зеркало настенное офисное")
    assert answer.status == "not_found" and answer.code is None
    assert answer.verifier is None


def test_response_contract_keeps_the_new_fields_optional():
    plain = ClassifyResponse(
        query="x", code="26.20.2", title="t", confidence=0.9, status="ok", latency_ms=0.0
    )
    assert plain.verifier is None and plain.verifier_note is None
    verified = plain.model_copy(
        update={"verifier": "llm-top3", "verifier_note": "код подтверждён по top-3"}
    )
    assert json.loads(verified.model_dump_json())["verifier"] == "llm-top3"


def test_readiness_reports_the_verifier_state(monkeypatch):
    """"Почему код помечен проверенным" должно быть видно без чтения логов."""
    from app.core.config import settings
    from app.main import app
    from fastapi.testclient import TestClient

    monkeypatch.setattr(settings, "classifier_backend", "mock")
    monkeypatch.setattr(settings, "llm_verifier_enabled", False)
    monkeypatch.setattr(settings, "llm_verifier_model", "gpt-4o-mini")
    monkeypatch.setattr("app.services.classifier.get_classifier", lambda: None)
    monkeypatch.setattr("app.main.get_classifier", lambda: None)
    body = TestClient(app).get("/health/ready").json()
    assert body["llm_verifier"] == "off"

    monkeypatch.setattr(settings, "llm_verifier_enabled", True)
    body = TestClient(app).get("/health/ready").json()
    assert body["llm_verifier"] == "on:gpt-4o-mini"
