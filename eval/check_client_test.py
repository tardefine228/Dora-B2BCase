"""Прогон клиентского теста: доля верных решений по файлу ``client_test_100.csv``.

Файл `eval/client_test_100.csv` — 100 позиций закупки с эталонными кодами ОКПД2,
собранных как заказчик: короткие названия («Ткань хлопковая»), названия с
признаками («Свитер трикотажный мужской») и несколько подробных формулировок.
Эталоны взяты из размеченного набора `eval/labeled_examples.csv` (спорные строки
не включены) и из синтетического набора `eval/golden_1875`.

Правила подсчёта повторяют `eval/run_eval.py` (методика в README, раздел
«Качество и проверки»):

* верное решение — ответ с кодом, совпадающим с эталоном (точно или как
  родитель/потомок: «27.40.1» и «27.40.12» совместимы), либо уточнение
  (`needs_clarification`), среди вариантов которого есть эталонный код;
* «Нет в перечне» и уточнение без эталонного кода — ошибка;
* порог уверенности и калибровка не оцениваются, меряется только решение.

Путь по умолчанию — лексический (catalog + reranker, без Qdrant): он
детерминирован и работает офлайн. В docker-сборке дополнительно включён
векторный поиск, поэтому число может отличаться на 1–2 процентных пункта.

Есть два набора:

* `eval/client_test_100.csv` — «клиентский»: 100 позиций, собранных из
  размеченного набора и подробных ТЗ-формулировок;
* `eval/organizer_test_100.csv` — «как от организаторов чемпионата»: 70
  официальных наименований позиций перечня ПП № 1875 и 30 реалистичных
  ТЗ-позиций с брендами и характеристиками.

Запуск::

    python eval/check_client_test.py                 # отчёт и процент (клиентский набор)
    python eval/check_client_test.py --file eval/organizer_test_100.csv
    python eval/check_client_test.py --min 96        # упасть, если ниже 96 %
    python eval/check_client_test.py --json report.json

Скрипт не тренирует и не меняет модель: это измерение, а не обучение.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILE = ROOT / "eval" / "client_test_100.csv"
DEFAULT_CATALOG = ROOT / "data" / "data.csv"
NAME_COLUMN = "Наименование"
EXPECTED_COLUMN = "Эталон (проверка)"


def codes_match(predicted: str | None, expected: str) -> bool:
    """Совпадение с учётом иерархии: родитель покрывает потомка и наоборот."""
    if not predicted or not expected:
        return False
    predicted = predicted.strip()
    expected = expected.strip()
    return (
        predicted == expected
        or predicted.startswith(expected + ".")
        or expected.startswith(predicted + ".")
    )


def clarification_has_code(response, expected: str) -> bool:
    """Вопрос засчитывается, только если среди вариантов есть эталонный код."""
    clarification = getattr(response, "clarification", None)
    return bool(
        response.status == "needs_clarification"
        and clarification is not None
        and len(clarification.options) >= 2
        and any(codes_match(option.code, expected) for option in clarification.options)
    )


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    if not rows:
        raise SystemExit(f"Пустой файл: {path}")
    missing = {NAME_COLUMN, EXPECTED_COLUMN} - set(rows[0])
    if missing:
        raise SystemExit(f"В файле нет колонок: {', '.join(sorted(missing))}")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE,
                        help="CSV с позициями и эталонами")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="data.csv перечня")
    parser.add_argument("--min", dest="minimum", type=float, default=0.0,
                        help="минимальный допустимый процент верных решений")
    parser.add_argument("--json", type=Path, default=None, help="куда записать отчёт в JSON")
    parser.add_argument("--quiet", action="store_true", help="только итоговая строка")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(ROOT / "api"))
    sys.path.insert(0, str(ROOT))
    from app.core.config import settings

    # Детерминированное измерение: лексический путь без Qdrant (см. docstring).
    settings.vector_search = False
    settings.rate_limit_enabled = False

    from app.services.classifier import CatalogClassifier

    classifier = CatalogClassifier(args.catalog)
    rows = load_rows(args.file)

    checked = []
    for row in rows:
        name = (row.get(NAME_COLUMN) or "").strip()
        expected = (row.get(EXPECTED_COLUMN) or "").strip()
        if not name or not expected:
            continue
        response = classifier.classify(name)
        correct = codes_match(response.code, expected) or clarification_has_code(response, expected)
        checked.append(
            {
                "name": name,
                "expected": expected,
                "predicted": response.code,
                "status": response.status,
                "correct": correct,
            }
        )

    total = len(checked)
    right = sum(1 for item in checked if item["correct"])
    percent = (right / total * 100) if total else 0.0

    if not args.quiet:
        misses = [item for item in checked if not item["correct"]]
        if misses:
            print(f"Ошибки ({len(misses)}):")
            for item in misses:
                got = item["predicted"] or f"нет кода ({item['status']})"
                print(f"  • {item['name'][:88]:<88} эталон {item['expected']:<12} ответ {got}")
            print()

    print(
        f"Правильно: {right} из {total} — {percent:.1f}% "
        f"(код или уточнение с эталонным кодом)"
    )

    if args.json:
        args.json.write_text(
            json.dumps(
                {"total": total, "correct": right, "percent": round(percent, 1), "items": checked},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if not args.quiet:
            print(f"Отчёт: {args.json}")

    if percent + 1e-9 < args.minimum:
        print(f"Ниже порога {args.minimum:.1f}%")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
