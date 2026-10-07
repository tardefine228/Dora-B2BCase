from pathlib import Path

from app.services.classifier import (
    CatalogClassifier,
    MockClassifier,
    build_classifier,
    get_classifier,
    reset_classifier,
)


def _catalog_path() -> Path:
    # api/tests/test_classifier.py -> parents[2] == корень репозитория
    p = Path(__file__).resolve().parents[2] / "data" / "data.csv"
    assert p.exists(), p
    return p


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


def test_build_classifier_default_is_catalog() -> None:
    reset_classifier()
    classifier = build_classifier("catalog")
    assert isinstance(classifier, CatalogClassifier)
    reset_classifier()


def test_get_classifier_singleton() -> None:
    reset_classifier()
    first = get_classifier()
    second = get_classifier()
    assert first is second
    reset_classifier()
