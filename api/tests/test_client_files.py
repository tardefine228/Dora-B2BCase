"""Client-facing demo documents: does the service recognise every position?

``api/tests/fixtures/client_files`` holds eight files of the kind a procurement
specialist uploads on the main page (``POST /classify/file``): a cp1251 Excel
export, a TSV, a pasted numbered list, a headerless CSV, a JSON order, a
Markdown checklist, an XLSX with a title row above the table and a DOCX with a
requirements table.

For every file ``fixtures/client_files/expected/<file>.json`` pins:

* how the file is parsed — the exact list of product names, in order, plus the
  detected column, de-duplication and skipped-row counters;
* what each name must map to in the ПП № 1875 catalogue — the accepted ОКПД2
  code(s) and the accepted ``status`` values, with a note explaining the choice.

Codes were verified against ``data/data.csv`` by hand: an expected value is the
position that a specialist would pick, not merely today's model output. Entries
that are confidently wrong today live in ``expected/known_gaps.json`` and are
reported as ``xfail`` so an improvement shows up as ``XPASS`` instead of a
silent pass.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from app.core.config import settings
from app.main import app
from app.services.classifier import build_classifier
from app.services.file_items import detect_csv_delimiter, extract_items
from fastapi.testclient import TestClient

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "client_files"
EXPECTED_DIR = FIXTURES_DIR / "expected"
REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "data" / "data.csv"

REQUIRED_RESULT_FIELDS = {
    "query",
    "code",
    "title",
    "confidence",
    "status",
    "clause_ref",
    "clause_url",
    "alternatives",
    "latency_ms",
    "contract_version",
}

CONTENT_TYPES = {
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".json": "application/json",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


# --- expectations ------------------------------------------------------------


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def expectation_paths() -> list[Path]:
    return sorted(path for path in EXPECTED_DIR.glob("*.json") if path.name != "known_gaps.json")


def expectation_ids() -> list[str]:
    return [path.name.removesuffix(".json") for path in expectation_paths()]


def gaps() -> list[dict[str, Any]]:
    return _load(EXPECTED_DIR / "known_gaps.json")["gaps"]


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES_DIR / name).read_bytes()


def _content_type(name: str) -> str:
    return CONTENT_TYPES.get(Path(name).suffix.lower(), "application/octet-stream")


# --- helpers -----------------------------------------------------------------


@pytest.fixture(scope="module")
def classifier():
    """The classifier the API actually serves — calibrated reranker included.

    A bare ``CatalogClassifier`` scores lexically and answers differently, so the
    expectations below are pinned against this one.
    """
    assert CATALOG_PATH.exists(), CATALOG_PATH
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(settings, "catalog_path", str(CATALOG_PATH))
    try:
        yield build_classifier("catalog")
    finally:
        monkeypatch.undo()


def _checkable(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Items that the classifier must get right (known gaps are reported via xfail)."""
    return [item for item in spec["items"] if not item.get("known_gap")]


def _check_result(result: Any, item: dict[str, Any]) -> list[str]:
    """Everything that is wrong with one classification, as readable strings."""
    problems: list[str] = []

    def fail(message: str) -> None:
        problems.append(f"«{item['query']}»: {message} | {item.get('note', '')}")

    if item["codes"] is not None and result.code not in item["codes"]:
        fail(f"код {result.code!r}, ожидался один из {item['codes']}")
    if result.status not in item["statuses"]:
        fail(f"статус {result.status!r}, ожидался один из {item['statuses']}")
    if not 0.0 <= result.confidence <= 1.0:
        fail(f"confidence={result.confidence} вне диапазона 0..1")
    if result.contract_version != "1.0":
        fail(f"contract_version={result.contract_version!r}")

    if result.status == "not_found":
        if result.code is not None or result.title is not None:
            fail("при статусе not_found код и наименование должны быть пустыми")
    else:
        if not result.code:
            fail("найдена позиция без кода ОКПД2")
        if not result.title:
            fail("найдена позиция без наименования по перечню")
    return problems


def _wait_for_job(client: TestClient, status_url: str, timeout: float = 120.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    state: dict[str, Any] = {}
    while time.monotonic() < deadline:
        state = client.get(status_url).json()
        if state["status"] in {"completed", "failed", "cancelled"}:
            return state
        time.sleep(0.05)
    raise AssertionError(f"задача не завершилась за {timeout:.0f} с: {state}")


# --- fixtures themselves -----------------------------------------------------


def test_every_fixture_has_an_expectation_file() -> None:
    fixtures = {
        path.name for path in FIXTURES_DIR.iterdir() if path.is_file() and path.name != "README.md"
    }
    described = {_load(path)["file"] for path in expectation_paths()}
    assert fixtures == described, "у каждого файла клиента должен быть свой *.json с ожиданиями"


def test_fixtures_are_reproducible() -> None:
    """The committed files must be exactly what ``scripts/make_client_files.py`` builds."""
    pytest.importorskip("openpyxl")
    pytest.importorskip("docx")
    script = REPO_ROOT / "scripts" / "make_client_files.py"
    spec = importlib.util.spec_from_file_location("make_client_files", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    stale = module.differences(FIXTURES_DIR)
    assert not stale, (
        "фикстуры устарели, пересоберите: python scripts/make_client_files.py "
        f"(отличаются: {stale})"
    )


# --- parsing -----------------------------------------------------------------


@pytest.mark.parametrize("path", expectation_paths(), ids=expectation_ids())
def test_product_names_are_extracted_in_order(path: Path) -> None:
    spec = _load(path)
    result = extract_items(spec["file"], fixture_bytes(spec["file"]))

    assert result.format == spec["format"]
    assert result.items == [item["query"] for item in spec["items"]]
    assert result.total_found == len(spec["items"])
    assert result.source_column == spec["source_column"]
    assert result.duplicates == spec["duplicates"]
    assert result.skipped == spec["skipped"]
    assert not result.truncated


def test_semicolon_csv_with_decimal_commas_is_not_split_by_comma() -> None:
    """Regression: a 1С/Excel export uses «;» and decimal commas at the same time.

    ``csv.Sniffer`` used to answer «,» here, so a whole row landed in one
    "product name" (``1;Ноутбук;3;89 900`` + ``00``).
    """
    text = "№;Наименование;Кол-во;Цена\n1;Ноутбук Lenovo;2;89 900,00\n2;Монитор Dell;1;24 500,50\n"
    assert detect_csv_delimiter(text) == ";"

    result = extract_items("export.csv", text.encode("cp1251"))
    assert result.items == ["Ноутбук Lenovo", "Монитор Dell"]
    assert result.source_column == "Наименование"


def test_plain_text_keeps_numbering_out_of_names() -> None:
    content = "1. Ноутбук\n2) Монитор\n- Клавиатура\n• Мышь\n\n12345\n"
    result = extract_items("list.txt", content.encode())
    assert result.items == ["Ноутбук", "Монитор", "Клавиатура", "Мышь"]
    assert result.skipped == 1


def test_markdown_headings_are_structure_not_products() -> None:
    """A check-list heading must not become a product with a confident answer.

    Observed on a real upload: «# Перечень позиций к закупке» was classified as
    «22.22.1 Изделия пластмассовые упаковочные» with status ok, confidence 0.81.
    """
    content = (
        "# Перечень позиций к закупке\n"
        "\n"
        "## Компьютерная техника\n"
        "- Ноутбук Lenovo\n"
        "- Монитор Dell\n"
    ).encode()
    result = extract_items("perechen.md", content)
    assert result.items == ["Ноутбук Lenovo", "Монитор Dell"]
    assert result.skipped == 2
    assert any("заголовков Markdown" in warning for warning in result.warnings)

    # Plain text keeps headings: there is no markup, so «#» is literal text.
    plain = extract_items("list.txt", b"# not a heading\nNoatbook\n")
    assert plain.items == ["# not a heading", "Noatbook"]


# --- recognition of each position -------------------------------------------


@pytest.mark.parametrize("path", expectation_paths(), ids=expectation_ids())
def test_every_position_gets_the_expected_catalog_entry(path: Path, classifier) -> None:
    spec = _load(path)
    extracted = extract_items(spec["file"], fixture_bytes(spec["file"]))
    assert extracted.items == [item["query"] for item in spec["items"]]

    checkable = _checkable(spec)
    problems: list[str] = []
    for item in checkable:
        problems.extend(_check_result(classifier.classify(item["query"]), item))

    assert not problems, (
        f"{spec['file']}: неверно распознано {len(problems)} из {len(checkable)} позиций:\n  - "
        + "\n  - ".join(problems)
    )


def test_positions_outside_the_list_are_not_invented(classifier) -> None:
    """Everything the catalogue does not cover must come back as ``not_found``."""
    negatives = [
        item
        for path in expectation_paths()
        for item in _load(path)["items"]
        if item["statuses"] == ["not_found"]
    ]
    assert len(negatives) >= 5, negatives

    for item in negatives:
        result = classifier.classify(item["query"])
        assert result.status == "not_found", f"«{item['query']}»: {result.code} при {result.status}"
        assert result.code is None and result.title is None


def test_known_gaps_are_marked_in_the_expectation_files() -> None:
    flagged = {
        (path.stem.removesuffix(".json"), item["query"])
        for path in expectation_paths()
        for item in _load(path)["items"]
        if item.get("known_gap")
    }
    declared = {(gap["file"], gap["query"]) for gap in gaps()}
    assert flagged == declared, "флаги known_gap и expected/known_gaps.json разошлись"


@pytest.mark.parametrize("gap", gaps(), ids=[gap["query"] for gap in gaps()])
@pytest.mark.xfail(reason="известный промах модели, см. expected/known_gaps.json", strict=False)
def test_known_gaps_are_documented_not_forgotten(gap: dict[str, Any], classifier) -> None:
    result = classifier.classify(gap["query"])
    assert result.code in gap["expected_codes"], (
        f"«{gap['query']}» → {result.code} ({result.status}); ожидалось {gap['expected_codes']}"
    )


# --- the whole upload flow ---------------------------------------------------


@pytest.mark.parametrize("path", expectation_paths(), ids=expectation_ids())
def test_upload_endpoint_recognises_every_item_of_the_file(path: Path, classifier) -> None:
    """``POST /classify/file`` → job → results, in file order, with the same answers."""
    spec = _load(path)
    client = TestClient(app)

    response = client.post(
        "/classify/file",
        files={"file": (spec["file"], fixture_bytes(spec["file"]), _content_type(spec["file"]))},
    )
    assert response.status_code == 202, response.text
    accepted = response.json()

    queries = [item["query"] for item in spec["items"]]
    assert accepted["items"] == queries
    assert accepted["total"] == len(queries)
    assert accepted["total_found"] == len(queries)
    assert accepted["duplicates"] == spec["duplicates"]
    assert accepted["skipped"] == spec["skipped"]
    assert accepted["format"] == spec["format"]
    assert accepted["source_column"] == spec["source_column"]
    assert accepted["truncated"] is False
    assert accepted["contract_version"] == "1.0"

    state = _wait_for_job(client, accepted["status_url"])
    assert state["status"] == "completed"
    assert [row["query"] for row in state["results"]] == queries
    assert len(state["results"]) == len(queries)

    problems: list[str] = []
    for row, item in zip(state["results"], spec["items"], strict=True):
        assert REQUIRED_RESULT_FIELDS <= set(row)
        if item.get("known_gap"):
            continue
        problems.extend(_check_result(_Row(row), item))

    assert not problems, (
        f"{spec['file']} через API: неверно распознано {len(problems)} "
        f"из {len(queries)} позиций:\n  - " + "\n  - ".join(problems)
    )


class _Row:
    """Adapt a JSON row from the batch endpoint to the ``ClassifyResponse`` fields."""

    def __init__(self, row: dict[str, Any]) -> None:
        self.query = row["query"]
        self.code = row["code"]
        self.title = row["title"]
        self.confidence = row["confidence"]
        self.status = row["status"]
        self.contract_version = row["contract_version"]


def test_api_and_offline_classifier_agree(classifier) -> None:
    """The endpoint and the offline classifier must not disagree (no hidden state)."""
    spec = _load(EXPECTED_DIR / "it_zakupka.csv.json")
    client = TestClient(app)
    response = client.post(
        "/classify/file",
        files={"file": (spec["file"], fixture_bytes(spec["file"]), _content_type(spec["file"]))},
    )
    state = _wait_for_job(client, response.json()["status_url"])

    for row in state["results"]:
        offline = classifier.classify(row["query"])
        assert (row["code"], row["status"]) == (offline.code, offline.status)


def test_unsupported_and_empty_files_are_rejected_with_clear_codes() -> None:
    client = TestClient(app)
    bad = client.post(
        "/classify/file",
        files={"file": ("skan.png", b"\x89PNG\r\n", "image/png")},
    )
    assert bad.status_code == 415
    assert "не поддерживается" in bad.json()["detail"].lower()

    empty = client.post("/classify/file", files={"file": ("list.txt", b"", "text/plain")})
    assert empty.status_code == 422

    numbers_only = client.post(
        "/classify/file",
        files={"file": ("list.txt", b"1\n2\n3\n", "text/plain")},
    )
    assert numbers_only.status_code == 422


# --- the offline checker -----------------------------------------------------


@pytest.fixture(scope="module")
def comparator():
    """Import scripts/compare_client_results.py as a module (dataclasses need sys.modules)."""
    script = REPO_ROOT / "scripts" / "compare_client_results.py"
    spec = importlib.util.spec_from_file_location("compare_client_results", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compare_script_passes_on_ideal_results(tmp_path: Path, comparator) -> None:
    """``scripts/compare_client_results.py`` must accept a perfect run.

    Guards the tool used to review real uploads: a checker that fails on ideal
    data would hide actual regressions behind noise.
    """
    module = comparator

    def write_csv(name: str, rows: list[tuple[str, str | None, str]]) -> Path:
        path = tmp_path / name
        lines = ["query;code;title;status;confidence;confidence_raw;clause_ref"]
        for query, code, status in rows:
            lines.append(f'"{query}";{code or ""};;{status};0.9;0.9;')
        path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
        return path

    for expected_path in expectation_paths():
        expected = _load(expected_path)
        rows = []
        for item in expected["items"]:
            code = item["codes"][0] if item["codes"] else None
            rows.append((item["query"], code, item["statuses"][0]))
        path = write_csv(f"ok-{expected['file']}.csv", rows)
        assert module.main([str(path)]) == 0, (
            f"идеальная выгрузка {expected['file']} не прошла сверку"
        )


def test_compare_script_flags_a_wrong_code(tmp_path: Path, comparator) -> None:
    module = comparator

    path = tmp_path / "broken.csv"
    path.write_text(
        "query;code;title;status;confidence;confidence_raw;clause_ref\n"
        '"Ноутбук Lenovo ThinkPad T14";33.12.99;;ok;0.9;0.9;\n',
        encoding="utf-8-sig",
    )
    assert module.main([str(path)]) == 1
