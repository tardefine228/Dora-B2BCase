"""Regression cases from the user's production exports (36)–(39)."""

from __future__ import annotations

from pathlib import Path

import pytest
from app.core.config import settings
from app.services.classifier import CatalogClassifier, reset_classifier

CATALOG = Path(__file__).resolve().parents[2] / "data" / "data.csv"


@pytest.fixture(autouse=True)
def lexical_backend():
    previous = settings.vector_search
    settings.vector_search = False
    reset_classifier()
    yield
    settings.vector_search = previous
    reset_classifier()


OBSERVED_DECISIONS = {
    # (38): specific products must not inherit a merely similar category.
    "Аккумулятор свинцово-кислотный стационарный": ("ok", "27.20.22.000"),
    "Самосвал карьерный трехосный": ("ok", "28.92.29.000"),
    "Шприц медицинский одноразовый трехдетальный": ("ok", "32.50.13.110"),
    "Паста теплопроводная кремнийорганическая (термопаста)": ("not_found", None),
    "Плата печатная смонтированная": ("ok", "26.12.1"),
    "Компрессор воздушный винтовой стационарный": ("ok", "28.13.28"),
    "Датчик температуры (термопреобразователь)": ("ok", "26.51.51"),
    "Осциллограф цифровой запоминающий": ("ok", "26.51.42"),
    "Томограф магнитно-резонансный (МРТ)": ("ok", "26.60.12.131"),
    "Сервер общего назначения": ("ok", "26.20.14"),
    "Источник бесперебойного питания (ИБП)": ("ok", "26.20.4"),
    "Электродвигатель асинхронный трехфазный": ("ok", "27.11.2"),
    "Шкаф распределительный низковольтный (НКУ)": ("not_found", None),
    "Задвижка клиновая с выдвижным шпинделем": ("ok", "28.14.13.120"),
    "Кран-балка опорная электрическая": ("ok", "28.22.14.121"),
    "Кондиционер прецизионный шкафной": ("ok", "28.25.12.110"),
    "Теплообменник пластинчатый разборный": ("ok", "28.25.11"),
    "Станок фрезерный с ЧПУ вертикальный": ("ok", "28.41.2"),
    "Станок сверлильный настольный": ("ok", "28.41.3"),
    "Станок шлифовальный круглошлифовальный": ("ok", "28.41.3"),
    "Станок ленточнопильный по металлу": ("ok", "28.41.3"),
    "Экскаватор одноковшовый гусеничный": ("ok", "28.92.26.110"),
    "Каток дорожный вибрационный двухвальцовый": ("ok", "28.92.24.120"),
    "Аппарат сварочный инверторный полуавтоматический": ("ok", "27.90.31.110"),
    "Вагон метрополитена моторный головной": ("ok", "30.20.2"),
    "Вагон грузовой полувагон универсальный": ("ok", "30.20.33"),
    "Аппарат ультразвуковой диагностический (УЗИ)": ("ok", "26.60.12.132"),
    "Электрокардиограф 12-канальный портативный": ("ok", "26.60.12.111"),
    "Комплект постельного белья хлопчатобумажный": ("not_found", None),
    "Труба напорная полиэтиленовая (ПНД PE-100)": ("ok", "22.21.2"),
    "Труба стальная электросварная прямошовная": ("not_found", None),
    "Алюминий первичный в чушках": ("not_found", None),
    # (39): brand/model evidence and explicit product-type distinctions.
    "Аккумулятор VARTA Blue Dynamic D24": ("ok", "27.20.21.000"),
    "Аккумулятор DELTA DTM 1207": ("ok", "27.20.22.000"),
    "Аккумулятор Makita BL1850B": ("ok", "27.20.23.130"),
    "Видеокарта ASUS DUAL-RTX3060-O12G-V2": ("ok", "26.12.20"),
    "Сетевая плата TP-Link TG-3468": ("ok", "26.12.20"),
    "Аккумулятор BOSCH S4 005": ("ok", "27.20.21.000"),
    "Аккумулятор CSB GP1272": ("ok", "27.20.22.000"),
    "Холодильник Bosch KGV36XW21R": ("not_found", None),
    "Морозильник ATLANT М 7204-100": ("not_found", None),
    "Машина посудомоечная ABAT МПК-500Ф": ("ok", "28.29.50.000"),
    "Машина посудомоечная Bosch SMS25AW01R": ("not_found", None),
    "Клей Henkel Момент Столяр Супер ПВА": ("ok", "20.52"),
    "Камера холодильная POLAIR КХН-4.41": ("ok", "28.25.13.112"),
    "Модуль памяти Kingston ValueRAM KVR32N22S8/8": ("ok", "26.20.4"),
    "Терминал сбора данных Zebra TC21": ("ok", "26.20.11"),
    "Ротор электродвигателя для перфоратора Makita HR2470": ("not_found", None),
    "Программный продукт Microsoft Office LTSC 2024 Standard": ("ok", "58.29"),
    "Плита EGGER Eurospan E1": ("ok", "16.21.13"),
    "Плита Kronospan MDF": ("ok", "16.21.14"),
}


@pytest.mark.parametrize(("query", "expected"), OBSERVED_DECISIONS.items())
def test_observed_export_decision(query: str, expected: tuple[str, str | None]) -> None:
    answer = CatalogClassifier(CATALOG).classify(query)
    status, code = expected
    assert answer.status == status, (query, answer.model_dump())
    assert answer.code == code
    if status == "not_found":
        assert answer.title is None
        assert answer.clause_ref is None
        assert answer.clause_url is None
        assert answer.alternatives == []


def test_hierarchical_codes_keep_applicable_clauses_not_unrelated_products() -> None:
    classifier = CatalogClassifier(CATALOG)
    battery = classifier.classify(
        "Аккумуляторы свинцовые, кроме используемых для запуска поршневых двигателей"
    )
    assert battery.status == "ok" and battery.code == "27.20.22.000"
    assert battery.clause_ref == "Приложение 2, позиция 190; Приложение 3, позиция 99"

    network = classifier.classify("Сетевая плата TP-Link TG-3468")
    assert network.code == "26.12.20"
    assert network.clause_ref == "Приложение 2, позиция 194; Приложение 3, позиция 36"
    assert "позиция 35" not in (network.clause_ref or "")  # parent explicitly excludes 26.12.20

    syringe = classifier.classify("Шприц медицинский одноразовый трехдетальный")
    assert syringe.clause_ref == "Приложение 2, позиция 385"
    assert "позиция 386" not in (syringe.clause_ref or "")  # same code, different product


def test_generic_word_does_not_inherit_a_complete_title_hierarchy() -> None:
    answer = CatalogClassifier(CATALOG).classify("комплектующие")
    assert answer.status == "not_found"
    assert answer.code is None and answer.clarification is None
    assert answer.alternatives == []


def test_remaining_review_and_clarification_are_intentional() -> None:
    classifier = CatalogClassifier(CATALOG)
    for query in (
        "Аппарат искусственной вентиляции легких (ИВЛ)",
        "Ванна",
        "Компрессор FUBAG DC 320/24 CM2.5",
        "Компрессор Metabo Basic 250-24 W",
        "Шкаф ПРАКТИК AM 1845",
        "Микроскопы медицинские",
    ):
        answer = classifier.classify(query)
        assert answer.status == "needs_clarification", query
        assert answer.code is None and answer.clarification is not None

    kitchen = classifier.classify("Шкаф кухонный IKEA METOD")
    assert kitchen.status == "ok" and kitchen.code == "31.02.10"
    assert kitchen.catalog_review_required is False

    noisy = (
        "Ноутбук Lenovoo IdeaPad 3 15ITL6 / i5 / 8gb / 256gb ssd / 15.6 дюйма !!! "
        "1шт,26.20.11,Опечатки / мусор,Прил.2 поз.195"
    )
    answer = classifier.classify(noisy)
    assert answer.status == "review" and answer.code == "26.20.11"
