from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest
from app.core.config import settings
from app.job_store import SqliteJobStore
from app.main import app
from app.schemas import ClassifyResponse
from app.services.ambiguity import AmbiguityConflict, AmbiguityGuard, load_catalog_rows
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


@pytest.mark.parametrize(
    ("query", "required_option_ids"),
    [
        ("Клапан", {"3-120-01", "3-126-01", "3-122-01"}),
        ("Вентиль", {"3-120-01", "3-126-01", "3-122-01"}),
        ("Аккумулятор", {"1-31-01", "2-190-01", "3-100-01"}),
        ("Шкаф", {"1-56-01", "2-139-01", "1-138-01", "2-141-01"}),
        ("Перчатки", {"1-3-01", "3-16-01", "3-17-01", "3-18-01", "3-19-01"}),
        ("Подгузники", {"2-2-01", "2-362-01", "2-362-02", "2-362-03", "2-362-04"}),
        ("Контейнер", {"1-113-01", "1-114-01", "2-74-01", "2-378-06"}),
        ("Стол", {"2-139-01", "2-141-01", "2-320-01", "2-320-02"}),
        ("Ультразвуковой аппарат", {"2-314-01", "2-318-01", "2-380-03"}),
    ],
)
def test_observed_generic_families_offer_real_product_types(
    query: str, required_option_ids: set[str]
) -> None:
    answer = CatalogClassifier(CATALOG).classify(query)
    assert answer.status == "needs_clarification"
    assert answer.code is None and answer.title is None
    assert answer.clarification is not None
    option_ids = {option.option_id for option in answer.clarification.options}
    assert required_option_ids <= option_ids


@pytest.mark.parametrize("query", ["Аппараты ИВЛ", "Аппарат искусственной вентиляции легких"])
def test_ventilators_do_not_include_light_therapy_or_breathing_circuits(query: str) -> None:
    answer = CatalogClassifier(CATALOG).classify(query)
    assert answer.status == "needs_clarification"
    assert answer.clarification is not None
    assert {option.code for option in answer.clarification.options} == {
        "32.50.21.121",
        "32.50.21.122",
        "32.50.21.123",
    }
    assert all(not option.title.startswith("Контур") for option in answer.clarification.options)


@pytest.mark.parametrize("vector_search", ["0", "1"])
def test_observed_lexical_bridges_find_the_real_catalog_positions(
    vector_search: str, monkeypatch
) -> None:
    monkeypatch.setenv("VECTOR_SEARCH", vector_search)
    classifier = CatalogClassifier(CATALOG)
    for query, code in {
        "Компьютерная мышь": "26.20.16",
        "Смеситель": "28.14.11.160",
    }.items():
        answer = classifier.classify(query)
        assert answer.status in {"ok", "review"}, query
        assert answer.code == code, query


def test_unrelated_high_review_candidate_is_silenced() -> None:
    answer = CatalogClassifier(CATALOG).classify("Весы")
    assert answer.status == "not_found"
    assert answer.code is None and answer.title is None
    assert all("Антенн" not in (alternative.title or "") for alternative in answer.alternatives)


def test_meter_query_asks_about_the_measured_resource() -> None:
    answer = CatalogClassifier(CATALOG).classify("Прибор учёта")
    assert answer.status == "needs_clarification"
    assert answer.clarification is not None
    assert {option.option_id for option in answer.clarification.options} == {
        "1-28-01",
        "2-278-01",
        "2-279-01",
    }


@pytest.mark.parametrize("vector_search", ["0", "1"])
def test_new_generic_families_never_autoselect_a_random_branch(
    vector_search: str, monkeypatch
) -> None:
    monkeypatch.setenv("VECTOR_SEARCH", vector_search)
    classifier = CatalogClassifier(CATALOG)
    expected = {
        "Генератор": {"2-106-01", "2-107-01", "2-108-01", "2-345-01"},
        "Компрессор воздушный": {"1-34-01", "3-115-01", "3-116-01", "3-117-01", "3-118-01"},
        "Состав": {"1-131-01", "2-59-01"},
        "Платы": {"2-193-01", "2-194-01", "3-35-01", "3-36-01"},
        "Труба": {"2-66-01", "2-71-01", "2-89-01", "3-252-01"},
        "Камеры": {"1-57-01", "2-211-01"},
        "Тетради": {"2-6-01", "2-7-01"},
        "Принадлежности": {"2-8-01", "2-205-01", "1-66-01", "3-71-01"},
        "Телефон": {"2-212-01", "2-213-01"},
        "Телевизор": {"2-235-01", "2-236-01", "3-64-01"},
        "Кабель": {"2-113-01", "2-326-01"},
        "Лампа": {"2-191-01", "2-328-01", "2-332-01", "2-386-01"},
        "Монитор": {"2-201-01", "2-240-01"},
        "Микрофон": {"2-241-01", "3-68-01"},
        "Стеллаж": {"1-138-01", "1-140-01"},
        "Кресло": {"2-139-01", "2-394-01"},
        "Системы": {"2-221-01", "2-228-01", "2-262-01", "2-302-01"},
        "Установки": {"2-108-01", "3-158-01", "1-89-01", "2-255-01"},
    }
    for query, required_ids in expected.items():
        answer = classifier.classify(query)
        assert answer.status == "needs_clarification", (query, answer.model_dump())
        assert answer.code is None and answer.title is None
        assert answer.clarification is not None
        option_ids = {option.option_id for option in answer.clarification.options}
        assert required_ids <= option_ids, query


@pytest.mark.parametrize("vector_search", ["0", "1"])
def test_single_branch_generic_queries_require_review_not_autopass(
    vector_search: str, monkeypatch
) -> None:
    monkeypatch.setenv("VECTOR_SEARCH", vector_search)
    classifier = CatalogClassifier(CATALOG)
    for query, code in {
        "Трактор": "28.92.50.000",
        "Посуда": "22.29.23",
        "Кондиционер": "28.25.12.110",
        "Посудомоечная машина": "28.29.50.000",
    }.items():
        answer = classifier.classify(query)
        assert answer.status == "review", (query, answer.model_dump())
        assert answer.code == code
        assert any(reason.code == "ambiguous_name" for reason in answer.attention_reasons)


@pytest.mark.parametrize("vector_search", ["0", "1"])
def test_curated_product_phrases_override_unsafe_statistical_choices(
    vector_search: str, monkeypatch
) -> None:
    monkeypatch.setenv("VECTOR_SEARCH", vector_search)
    classifier = CatalogClassifier(CATALOG)
    expected = {
        "Ножи кухонные": "25.71.11",
        "Кухонный нож шеф-повара": "25.71.11",
        "Станок токарный с ЧПУ": "28.41.2",
        "Токарно-винторезный станок": "28.41.2",
        "Набор ручных инструментов": "25.73.60",
        "Свитер трикотажный мужской": "14.3",
        "Беговая дорожка": "32.30.14.120",
        "Пульт дистанционного управления": "26.51.20.130",
        "Внешний жесткий диск": "26.20.2",
        "Жесткий диск HDD 2 ТБ": "26.20.2",
        "Весы промышленные": "28.29.31.110",
    }
    for query, code in expected.items():
        answer = classifier.classify(query)
        assert answer.status == "ok", (query, answer.model_dump())
        assert answer.code == code, query
        assert any(
            reason.code == "curated_product_match" for reason in answer.attention_reasons
        ), query


@pytest.mark.parametrize("vector_search", ["0", "1"])
def test_curated_product_rules_have_negative_controls(
    vector_search: str, monkeypatch
) -> None:
    monkeypatch.setenv("VECTOR_SEARCH", vector_search)
    classifier = CatalogClassifier(CATALOG)
    expected = {
        "Ножи для рыбы": "25.71.14",
        "Устройства числового программного управления": "26.20.40.150",
        "Лыжи беговые": "32.30.11.110",
    }
    for query, code in expected.items():
        answer = classifier.classify(query)
        assert answer.code == code, (query, answer.model_dump())
        assert not any(
            reason.code == "curated_product_match" for reason in answer.attention_reasons
        ), query
    for query in ("Весы кухонные", "Весы бытовые электронные"):
        answer = classifier.classify(query)
        assert answer.status == "not_found", query
        assert answer.code is None and answer.title is None
        assert answer.alternatives == []


@pytest.mark.parametrize("vector_search", ["0", "1"])
def test_standalone_adjective_does_not_name_a_product(
    vector_search: str, monkeypatch
) -> None:
    monkeypatch.setenv("VECTOR_SEARCH", vector_search)
    answer = CatalogClassifier(CATALOG).classify("Медицинские")
    assert answer.status == "not_found"
    assert answer.code is None and answer.title is None
    assert answer.alternatives == []


@pytest.mark.parametrize("vector_search", ["0", "1"])
def test_specific_controls_keep_the_same_code_on_both_backends(
    vector_search: str, monkeypatch
) -> None:
    monkeypatch.setenv("VECTOR_SEARCH", vector_search)
    classifier = CatalogClassifier(CATALOG)
    expected = {
        "Платы печатные смонтированные": "26.12.1",
        "Трубы алюминиевые": "24.42.26",
        "Трубы": "32.20.13.161",
        "Генераторы сигналов электрические": "27.90.40.150",
        "Генератор дизельный": "27.11.3",
        "Компрессоры воздушные передвижные на колесных шасси": "28.13.24",
        "Компрессоры поршневые объемные": "28.13.26",
        "Трубы пластмассовые": "22.21.2",
        "Камеры холодильные сборные": "28.25.13.112",
        "Тетради различного назначения": "17.23.13.196",
        "Аппараты телефонные прочие": "26.30.23",
        "Тракторы гусеничные": "28.92.50.000",
        "Посуда пластмассовая": "22.29.23",
        "Телевизор с жидкокристаллическим экраном": "26.40.20.122",
        "Кабели волоконно-оптические": "27.31.1",
        "Кабели электрические": "27.32",
        "Лампы-вспышки фотографические": "27.40.31",
        "Микрофоны и подставки для них": "26.40.41.000",
        "Кондиционеры промышленные": "28.25.12.110",
        "Машины посудомоечные промышленного типа": "28.29.50.000",
        "Системы тревожной сигнализации": "26.30.50.115",
        "Установки генераторные электрические": "27.11.3",
    }
    for query, code in expected.items():
        answer = classifier.classify(query)
        assert answer.status == "ok", (query, answer.model_dump())
        assert answer.code == code, query

    for query, option_ids in {
        "Платы звуковые": {"2-194-01", "3-36-01"},
        "Мебель металлическая": {"3-231-01", "1-138-01"},
        "Кресла-стулья с санитарным оснащением": {"2-394-01", "2-394-02"},
    }.items():
        answer = classifier.classify(query)
        assert answer.status == "needs_clarification", (query, answer.model_dump())
        assert answer.clarification is not None
        assert option_ids <= {option.option_id for option in answer.clarification.options}


def test_medical_lights_offer_their_two_codes_instead_of_household_lights() -> None:
    answer = CatalogClassifier(CATALOG).classify("Светильники медицинские")
    assert answer.status == "needs_clarification"
    assert answer.clarification is not None
    assert {option.option_id for option in answer.clarification.options} == {
        "2-334-01",
        "2-334-02",
    }
    assert {option.code for option in answer.clarification.options} == {
        "27.40.39",
        "32.50.50.190",
    }


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
        "Клапан запорный": "28.14.13.110",
        "Шкаф холодильный": "28.25.13.111",
        "Шкаф металлический для документов": "31.09.11",
        "Перчатки рабочие тканевые": "14.1",
        "Перчатки хирургические резиновые": "22.19.60.111",
        "Подгузники детские": "17.22.12.120",
        "Контейнеры общего назначения": "29.20.21.110",
        "Аппараты ультразвукового сканирования": "26.60.12.132",
        "Аппараты ультразвуковой терапии": "26.60.13.150",
        "29.10.59.160": "29.10.59.160",
    }
    for query, code in expected.items():
        answer = classifier.classify(query)
        assert answer.status != "needs_clarification", query
        assert answer.code == code, query
    assert classifier.classify("29.10.59.160").decision_source == "direct_code"


def test_every_multi_code_title_option_is_resolvable_by_a_specialist() -> None:
    rows, digest = load_catalog_rows(CATALOG)
    by_title: dict[str, list[dict]] = {}
    for row in rows:
        by_title.setdefault(" ".join(row["name"].casefold().split()), []).append(row)
    groups = [
        group
        for group in by_title.values()
        if len({row["okpd2"] for row in group if row.get("okpd2")}) > 1
    ]
    assert len(groups) == 87
    guard = AmbiguityGuard(rows, digest)
    checked = 0
    expected_option_count = 0
    for group in groups:
        query = group[0]["name"]
        pending = guard.apply(query, guard.exact_title_answer(query))
        assert pending.status == "needs_clarification", query
        assert pending.clarification is not None
        options = pending.clarification.options
        expected_option_count += len(options)
        assert len({option.option_id for option in options}) == len(options)
        expected_codes = {row["okpd2"] for row in group if row.get("okpd2")}
        assert expected_codes <= {option.code for option in options}
        for option in options:
            resolved = guard.resolve_answer(pending, option.option_id, digest)
            assert resolved.status in {"ok", "review"}
            assert resolved.decision_source == "specialist"
            assert resolved.code == option.code
            assert resolved.resolution is not None
            assert resolved.resolution.option_id == option.option_id
            checked += 1
    assert checked == expected_option_count >= 325


def test_exact_title_with_multiple_codes_is_not_a_top1_answer() -> None:
    classifier = CatalogClassifier(CATALOG)
    answer = classifier.classify("Фильтр дыхательный (тепловлагообменник/бактериальный фильтр)")
    assert answer.status == "needs_clarification"
    assert answer.clarification is not None
    assert answer.attention_reasons[0].code == "multiple_codes"
    assert len({option.code for option in answer.clarification.options}) >= 2


@pytest.mark.parametrize(
    ("query", "record_id"),
    [
        ("Стол", "2-320-01"),
        ("Шприц", "2-385-01"),
        ("Повязки и салфетки раневые", "2-376-01"),
        ("Холодильник", "2-350-01"),
        ("Мебель медицинская (кровать больничная и пр.)", "1-140-01"),
        ("Мебель медицинская, хирургическая", "2-179-01"),
        ("Самосвал", "1-77-01"),
        ("Аккумулятор", "2-190-01"),
        ("Сыры и творог", "2-454-01"),
        ("Автомобили-самосвалы (бездорожье)", "1-77-01"),
        ("Аккумуляторы свинцовые прочие", "2-190-01"),
        ("Микроскопы медицинские", "2-379-01"),
    ],
)
def test_selected_multi_code_title_is_clarified_for_query_wordings(
    query: str, record_id: str
) -> None:
    rows, digest = load_catalog_rows(CATALOG)
    guard = AmbiguityGuard(rows, digest)
    selected = next(row for row in rows if row["record_id"] == record_id)
    original = ClassifyResponse(
        query=query,
        code=selected["okpd2"],
        title=selected["name"],
        confidence=0.95,
        confidence_raw=0.95,
        status="ok",
        latency_ms=0,
    )
    answer = guard.apply(query, original)
    assert answer.status == "needs_clarification"
    assert answer.code is None and answer.title is None
    assert answer.clarification is not None
    expected_codes = {
        row["okpd2"] for row in rows if row["name"] == selected["name"] and row["okpd2"]
    }
    assert expected_codes <= {option.code for option in answer.clarification.options}
    assert {reason.code for reason in answer.attention_reasons} <= {
        "ambiguous_name",
        "multiple_codes",
    }


@pytest.mark.parametrize(
    ("query", "record_id"),
    [
        ("Автомобиль", "1-77-01"),
        ("Светильник", "2-334-01"),
        ("Фильтр", "2-418-01"),
        ("Контейнер", "2-378-01"),
    ],
)
def test_clarification_contains_every_code_of_each_offered_title(
    query: str, record_id: str
) -> None:
    rows, _ = load_catalog_rows(CATALOG)
    selected = next(row for row in rows if row["record_id"] == record_id)
    answer = CatalogClassifier(CATALOG).classify(query)
    assert answer.status == "needs_clarification" and answer.clarification is not None
    offered = [
        option for option in answer.clarification.options if option.title == selected["name"]
    ]
    expected_codes = {
        row["okpd2"] for row in rows if row["name"] == selected["name"] and row["okpd2"]
    }
    assert len(expected_codes) >= 2
    assert {option.code for option in offered} == expected_codes
    assert len({option.option_id for option in offered}) == len(offered)


def test_every_sibling_code_can_be_resolved_without_hiding_the_choice() -> None:
    classifier = CatalogClassifier(CATALOG)
    pending = classifier.classify("Фильтр")
    assert pending.clarification is not None
    breathing = [
        option
        for option in pending.clarification.options
        if option.option_id.startswith("2-418-")
    ]
    assert len(breathing) == 9
    for selected in breathing:
        resolved = classifier.resolve_clarification(
            "Фильтр", selected.option_id, pending.clarification.catalog_sha256
        )
        assert resolved.code == selected.code
        assert resolved.title == selected.title
        assert resolved.decision_source == "specialist"
        assert resolved.resolution is not None
        assert resolved.resolution.option_id == selected.option_id
        assert resolved.clarification is None


def test_unrelated_weak_multi_code_candidate_does_not_create_a_question() -> None:
    rows, digest = load_catalog_rows(CATALOG)
    guard = AmbiguityGuard(rows, digest)
    selected = next(row for row in rows if row["record_id"] == "2-385-01")
    original = ClassifyResponse(
        query="Весы",
        code=selected["okpd2"],
        title=selected["name"],
        confidence=0.3,
        confidence_raw=0.3,
        status="review",
        latency_ms=0,
    )
    answer = guard.apply("Весы", original)
    assert answer.status == "review"
    assert answer.clarification is None


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
    assert "Приложение 1, позиция 77; Укрупнённый код" in row["clarification_options"]
    assert "Приложение 3, позиция 175; Детальный код" in row["clarification_options"]
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
        query="Автомобили легковые",
        code="29.10.2",
        title="Автомобили легковые",
        confidence=0.9,
        status="ok",
        latency_ms=0,
    )
    assert guard.apply("Автомобили легковые", original).status == "ok"


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
