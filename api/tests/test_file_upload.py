"""Main-page file upload: POST /classify/file and the item extractor."""

import io
import json

import pytest
from app.main import app
from app.services.file_items import FileParseError, UnsupportedFileError, extract_items
from fastapi.testclient import TestClient

client = TestClient(app)


def test_text_strips_numbering_and_duplicates() -> None:
    content = "1. Ноутбук Lenovo\n- Мышь компьютерная\n\n2) ноутбук lenovo\n12345\n".encode()
    result = extract_items("list.txt", content)
    assert result.items == ["Ноутбук Lenovo", "Мышь компьютерная"]
    assert result.duplicates == 1
    assert result.skipped == 1


def test_csv_detects_name_column_cp1251() -> None:
    content = "№;Наименование товара;Кол-во\n1;Светодиодная лампа E27;10\n2;Бумага А4;5\n"
    result = extract_items("goods.csv", content.encode("cp1251"))
    assert result.items == ["Светодиодная лампа E27", "Бумага А4"]
    assert result.source_column == "Наименование товара"


def test_csv_without_header_picks_text_column() -> None:
    content = "1,Ноутбук Lenovo IdeaPad,3\n2,Монитор Samsung 27,1\n"
    result = extract_items("goods.csv", content.encode())
    assert result.items == ["Ноутбук Lenovo IdeaPad", "Монитор Samsung 27"]
    assert result.warnings


def test_json_variants() -> None:
    plain = extract_items("a.json", json.dumps(["Принтер", "Сканер"]).encode())
    assert plain.items == ["Принтер", "Сканер"]
    objects = extract_items("b.json", json.dumps({"items": [{"id": 1, "name": "Сервер"}]}).encode())
    assert objects.items == ["Сервер"]


def test_xlsx_header_below_title() -> None:
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["Спецификация к договору"])
    sheet.append(["№", "Наименование", "Цена"])
    sheet.append([1, "Ноутбук HP", 50000])
    sheet.append([2, "Клавиатура", 900])
    buffer = io.BytesIO()
    workbook.save(buffer)
    result = extract_items("spec.xlsx", buffer.getvalue())
    assert result.items == ["Ноутбук HP", "Клавиатура"]


def test_docx_table_and_paragraphs() -> None:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    table = document.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text = "№"
    table.rows[0].cells[1].text = "Товар"
    table.rows[1].cells[0].text = "1"
    table.rows[1].cells[1].text = "Планшет"
    buffer = io.BytesIO()
    document.save(buffer)
    assert extract_items("t.docx", buffer.getvalue()).items == ["Планшет"]


def test_pdf_text() -> None:
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Laptop Lenovo\nMonitor Dell")
    result = extract_items("list.pdf", document.tobytes())
    assert result.items == ["Laptop Lenovo", "Monitor Dell"]


def test_truncation_and_errors() -> None:
    content = "\n".join(f"Товар номер {i}" for i in range(150)).encode()
    result = extract_items("big.txt", content)
    assert len(result.items) == 100 and result.truncated and result.total_found == 150
    with pytest.raises(UnsupportedFileError):
        extract_items("virus.exe", b"x")
    with pytest.raises(FileParseError):
        extract_items("empty.txt", b"\n 1 \n")


def test_endpoint_classifies_each_item() -> None:
    files = {"file": ("goods.txt", "Ноутбук Lenovo\nМышь компьютерная\n".encode(), "text/plain")}
    response = client.post("/classify/file", files=files)
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["items"] == ["Ноутбук Lenovo", "Мышь компьютерная"]
    assert body["total"] == 2
    status = client.get(body["status_url"]).json()
    assert status["status"] == "completed"
    assert [r["query"] for r in status["results"]] == body["items"]


def test_endpoint_error_codes() -> None:
    bad = client.post("/classify/file", files={"file": ("a.exe", b"MZ", "application/x")})
    assert bad.status_code == 415
    empty = client.post("/classify/file", files={"file": ("a.txt", b"", "text/plain")})
    assert empty.status_code == 422
