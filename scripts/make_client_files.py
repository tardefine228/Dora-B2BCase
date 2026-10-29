"""Build the demo "client" documents used by ``api/tests/test_client_files.py``.

The files imitate what a procurement specialist actually uploads on the main
page (``POST /classify/file``): an Excel export in cp1251, a TSV from a
spreadsheet, a pasted numbered list, a headerless CSV, a JSON API dump, a
Markdown checklist, an XLSX with title rows above the table and a DOCX
requirement sheet with a table.

Everything here is deterministic, so the fixtures are committed and reviewed as
plain files:

    python scripts/make_client_files.py          # rewrite the fixtures
    python scripts/make_client_files.py --check  # fail if they are out of date

Binary formats (XLSX/DOCX) are regenerated as well; their content is pinned by
``api/tests/fixtures/client_files/expected/*.json``.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "api" / "tests" / "fixtures" / "client_files"

# --- 1. Закупка ИТ-оборудования: CSV из 1С/Excel, cp1251, разделитель «;» ------
IT_ROWS = [
    "1;Ноутбук Lenovo ThinkPad T14;шт.;3;89 900,00",
    "2;Монитор Dell 27 дюймов;шт.;3;24 500,00",
    "3;Принтер лазерный HP LaserJet Pro;шт.;1;31 200,00",
    "4;Клавиатура проводная USB;шт.;3;1 350,00",
    "5;Мышь компьютерная оптическая;шт.;3;890,00",
    "6;Сервер стоечный 2U;шт.;1;412 000,00",
    "7;Кабель сетевой UTP cat.6;м;120;42,00",
    "8;Коммутатор сетевой 24 порта;шт.;1;58 700,00",
    "9;Источник бесперебойного питания 3000 ВА;шт.;1;64 300,00",
    "10;Картридж лазерный для принтера;шт.;2;5 400,00",
]
IT_CSV = "№;Наименование товара;Ед. изм.;Кол-во;Цена, руб.\n" + "\n".join(IT_ROWS) + "\n"

# --- 2. Канцелярия и мебель: TSV, разделитель «\t» ---------------------------
OFFICE_ROWS = [
    "Бумага офисная А4 80 г/м2\tпачка\t100",
    "Папки-регистраторы из картона\tшт.\t50",
    "Шкаф металлический для документов\tшт.\t4",
    "Стол письменный ЛДСП\tшт.\t6",
    "Стул офисный\tшт.\t12",
    "Кресло офисное\tшт.\t6",
    "Светильник светодиодный потолочный\tшт.\t8",
    "Лампа светодиодная E27 10 Вт\tшт.\t40",
]
OFFICE_TSV = "Наименование\tЕд. изм.\tКоличество\n" + "\n".join(OFFICE_ROWS) + "\n"

# --- 3. Медицинские изделия: обычный текст, нумерация и маркеры списка -------
MEDIC_TXT = """Перечень закупаемых медицинских изделий на 2026 год

1. Маска медицинская одноразовая
2) Шприц 5 мл одноразовый
3. Бинт стерильный 5 м
- Кресло-стул с санитарным оснащением
• Кровать больничная
4. Мебель медицинская
5. Халат медицинский одноразовый
6) Перчатки смотровые нитриловые
7. маска медицинская одноразовая
8. Глюкометр
"""

# --- 4. Строительные материалы: CSV без заголовка ---------------------------
STROY_CSV = """1,Цемент М500,кг,500
2,Труба стальная 57 мм,м,120
3,Краска водоэмульсионная белая,кг,80
4,Кирпич керамический полнотелый,шт,1000
5,Плитка керамическая для пола,м2,45
6,цемент м500,кг,250
"""

# --- 5. Заявка из внутренней системы: JSON с русскими ключами ---------------
ZAKAZ_ITEMS = [
    ("Перфоратор Bosch", 2),
    ("Дрель аккумуляторная", 3),
    ("Шуруповерт", 2),
    ("Болгарка", 2),
    ("Компрессор воздушный", 1),
    ("Насос погружной", 2),
    ("Подшипник 6205", 20),
    ("Кран мостовой электрический", 1),
    ("Погрузчик вилочный", 1),
    ("Лифт пассажирский", 1),
    ("Конвейер ленточный", 2),
    ("Генератор дизельный", 1),
    ("Станок токарный", 1),
    ("Сварочный аппарат инверторный", 2),
]
ZAKAZ_JSON = (
    json.dumps(
        {
            "order": "ЗК-2026-0042",
            "customer": "ООО «Ромашка»",
            "items": [
                {"id": index, "Наименование": name, "Количество": qty, "Ед. изм.": "шт."}
                for index, (name, qty) in enumerate(ZAKAZ_ITEMS, start=1)
            ],
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n"
)

# --- 6. Чек-лист в Markdown --------------------------------------------------
PERECHEN_MD = """# Перечень позиций к закупке

- Ноутбук Lenovo ThinkPad T14
- Кондиционер настенный
- Термометр инфракрасный бесконтактный
- Весы электронные настольные
- Автомобиль легковой Lada Vesta
- Шина летняя 205/55 R16
- Мясо говядина замороженная
- Огнетушитель порошковый ОП-4
"""

# --- 7. XLSX: несколько листов, заголовок таблицы не в первой строке ---------
XLSX_SHEETS = {
    "Титул": [["СПЕЦИФИКАЦИЯ к контракту № 42/2026"]],
    "Спецификация": [
        ["СПЕЦИФИКАЦИЯ к контракту № 42/2026"],
        [],
        ["№", "Наименование", "Ед. изм.", "Кол-во"],
        [1, "Стерилизатор паровой", "шт.", 2],
        [2, "Облучатель бактерицидный", "шт.", 5],
        [3, "Шкаф холодильный", "шт.", 2],
        [4, "Аппарат искусственной вентиляции легких", "шт.", 1],
        [5, "Холодильник медицинский", "шт.", 3],
        [6, "Тонометр", "шт.", 10],
        [7, "Дефибриллятор", "шт.", 2],
        [8, "Кушетка медицинская", "шт.", 4],
    ],
}

# --- 8. DOCX: таблица требований --------------------------------------------
DOCX_TABLE = [
    ["№", "Наименование товара", "Количество"],
    ["1", "Шкаф металлический для одежды", "4"],
    ["2", "Стеллаж металлический", "6"],
    ["3", "Доска маркерная", "2"],
    ["4", "Проектор мультимедийный", "2"],
    ["5", "Сканер документов", "3"],
    ["6", "МФУ лазерное", "2"],
    ["7", "Телефон IP", "10"],
    ["8", "Жесткий диск SSD 512 ГБ", "8"],
]


def _build_xlsx() -> bytes:
    from openpyxl import Workbook

    workbook = Workbook()
    workbook.remove(workbook.active)
    for title, rows in XLSX_SHEETS.items():
        sheet = workbook.create_sheet(title)
        for row in rows:
            sheet.append(row)
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _build_docx() -> bytes:
    import docx

    document = docx.Document()
    document.add_heading("Требования к поставляемому оборудованию", level=1)
    table = document.add_table(rows=len(DOCX_TABLE), cols=len(DOCX_TABLE[0]))
    for row_index, row in enumerate(DOCX_TABLE):
        for column_index, value in enumerate(row):
            table.rows[row_index].cells[column_index].text = value
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def payloads() -> dict[str, bytes]:
    """Every fixture as ``relative path -> bytes`` (stable and reviewable)."""
    return {
        "it_zakupka.csv": IT_CSV.encode("cp1251"),
        "office_supplies.tsv": OFFICE_TSV.encode("utf-8"),
        "medic_izdeliya.txt": MEDIC_TXT.encode("utf-8"),
        "stroyka_vedomost.csv": STROY_CSV.encode("utf-8"),
        "zakaz.json": ZAKAZ_JSON.encode("utf-8"),
        "perechen.md": PERECHEN_MD.encode("utf-8"),
        "trebovanie.xlsx": _build_xlsx(),
        "zayavka.docx": _build_docx(),
    }


def normalized(name: str, content: bytes) -> object:
    """Comparable representation: binary office files are compared by their text.

    XLSX/DOCX archives embed creation timestamps, so byte equality would fail on
    every rebuild. What matters for review is the sheet/table content.
    """
    if name.endswith(".xlsx"):
        from openpyxl import load_workbook

        workbook = load_workbook(io.BytesIO(content))
        return {
            sheet.title: [list(row) for row in sheet.iter_rows(values_only=True)]
            for sheet in workbook.worksheets
        }
    if name.endswith(".docx"):
        import docx

        document = docx.Document(io.BytesIO(content))
        return (
            [paragraph.text for paragraph in document.paragraphs],
            [[cell.text for cell in row.cells] for table in document.tables for row in table.rows],
        )
    return content


def differences(directory: Path) -> list[str]:
    """Fixture names whose content differs from what this script would write."""
    stale: list[str] = []
    for name, content in payloads().items():
        path = directory / name
        if not path.exists() or normalized(name, path.read_bytes()) != normalized(name, content):
            stale.append(name)
    return stale


def write(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, content in payloads().items():
        path = directory / name
        path.write_bytes(content)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="only verify that the committed fixtures match this script",
    )
    args = parser.parse_args(argv)

    if not args.check:
        for path in write(FIXTURES):
            print(f"written {path.relative_to(FIXTURES.parents[3])}")
        return 0

    stale = differences(FIXTURES)
    if stale:
        print("outdated fixtures: " + ", ".join(sorted(stale)), file=sys.stderr)
        print("run: python scripts/make_client_files.py", file=sys.stderr)
        return 1
    print("fixtures are up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
