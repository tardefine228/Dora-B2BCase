from pathlib import Path

from app.core.config import settings
from app.schemas import ClassifyResponse
from app.services.classifier import (
    CachedClassifier,
    CatalogClassifier,
    MockClassifier,
    build_classifier,
    get_classifier,
    reset_classifier,
)


def _catalog_path() -> Path:
    # api/tests/test_classifier.py -> parents[2] is the repository root.
    path = Path(__file__).resolve().parents[2] / "data" / "data.csv"
    assert path.exists(), path
    return path


def test_mock_classifier_contract() -> None:
    result = MockClassifier().classify("Ноутбук")
    assert result.code == "MOCK-001"
    assert result.status == "review"
    assert result.query == "Ноутбук"


def test_catalog_direct_code() -> None:
    classifier = CatalogClassifier(_catalog_path())
    result = classifier.classify("14.1")
    assert result.code == "14.1"
    assert result.status == "ok"


def test_catalog_notebook() -> None:
    classifier = CatalogClassifier(_catalog_path())
    result = classifier.classify("Ноутбук Lenovo")
    assert result.code == "26.20.11"


def test_catalog_not_found() -> None:
    classifier = CatalogClassifier(_catalog_path())
    result = classifier.classify("абвгдеёжз qwerty")
    assert result.status == "not_found"
    assert result.code is None


def test_catalog_alternatives_present() -> None:
    classifier = CatalogClassifier(_catalog_path())
    result = classifier.classify("Одежда")
    assert result.code is not None
    assert isinstance(result.alternatives, list)


def test_response_cache_normalizes_whitespace_and_case() -> None:
    class CountingClassifier:
        def __init__(self) -> None:
            self.calls = 0

        def classify(self, query: str) -> ClassifyResponse:
            self.calls += 1
            return MockClassifier().classify(query)

    inner = CountingClassifier()
    cached = CachedClassifier(inner, max_entries=2)

    first = cached.classify("  Ноутбук  ")
    second = cached.classify("ноутбук")

    assert inner.calls == 1
    assert first.query == "  Ноутбук  "
    assert second.query == "ноутбук"


def test_build_classifier_default_is_catalog(monkeypatch) -> None:
    monkeypatch.setattr(settings, "catalog_path", str(_catalog_path()))
    assert isinstance(build_classifier("catalog"), CatalogClassifier)


def test_get_classifier_singleton() -> None:
    reset_classifier()
    first = get_classifier()
    second = get_classifier()
    assert first is second
    reset_classifier()


def test_office_furniture_lines_resolve_to_the_office_position() -> None:
    """Реальные строки закупки мебели (эксп. 30): предмет, а не «офисный»."""
    classifier = CatalogClassifier(_catalog_path())
    expectations = {
        "Стул офисный на металлокаркасе": "31.01.1",
        "Кресло офисное эргономичное": "31.01.1",
        "Тумба приставная": "31.01.1",
        "Диван офисный 3-местный": "31.01.1",
        "Ресепшн-стойка": "31.01.1",
        "Стол письменный офисный": "31.01.1",
        "Шкаф для документов двухстворчатый": "31.09.11",
    }
    for query, code in expectations.items():
        result = classifier.classify(query)
        assert (result.status, result.code) == ("ok", code), (query, result.model_dump())


def test_office_modifier_alone_is_not_a_furniture_claim() -> None:
    """Товар другой позиции со словом «офисный» не должен становиться мебелью."""
    classifier = CatalogClassifier(_catalog_path())
    for query in ("Ламинатор офисный", "Сервер офисный", "Кофемашина офисная"):
        result = classifier.classify(query)
        assert result.code != "31.01.1", (query, result.model_dump())
