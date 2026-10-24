"""Разбор ошибок размеченного набора: какие из них — ошибки модели, а какие — разметки.

Зачем. Метрика `ok_precision` считает «уверенный ответ, который не совпал с
ожидаемым кодом» ошибкой. Но часть таких строк — это не ошибка модели, а
несогласованность самой разметки: одна и та же механика (ответ через мост
лексикона на обобщающую позицию перечня) в одном месте помечена верной, а в
другом — «вне перечня». Пример из набора 274 строк:

    Костюм мужской      → 14.1 «Одежда»            размечено ВЕРНО
    Конструктор детский → 32.4 «Игры и игрушки»    размечено «вне перечня»

Механика одна и та же, слова «костюм» и «конструктор» одинаково отсутствуют в
`data.csv`, обе позиции — обобщающие. Пока такие пары не разделены, любая
доработка модели упирается в противоречие в разметке: правило, убирающее вторую
строку, убирает и первую (измерено: правило «мост есть, своих слов в каталоге
нет» убирает 5 ответов на отсутствующий класс и **стоит 24 верных**).

Что делает инструмент. Для каждой строки определяет механику ответа и делит
ошибки на группы, после чего для каждой группы ищет строки с той же механикой,
помеченные противоположным образом, — это и есть кандидаты на решение
специалиста, а не на правку модели:

* ``sibling_code``  — ответ из того же семейства кодов, что и ожидаемый
  (совпадает родитель из 3–4 сегментов): тонкость разметки, не промах поиска;
* ``generic_category`` — позиция-ответ обобщающая: ни одно содержательное слово
  названия не совпадает с головой запроса, а сама позиция достигнута мостом;
* ``bridge_guess``  — сработал мост лексикона (продуктовое слово → категория);
* ``other``         — остальное.

Запуск::

    PYTHONPATH=api python eval/review_errors.py [--csv /tmp/errors.csv] [--limit 40]
    PYTHONPATH=api python eval/review_errors.py --brief docs/label-questions.md

С ``--brief`` пишется готовый документ для специалиста по нацрежиму: по каждой
спорной строке цитируются обе позиции каталога (ожидаемая и выбранная моделью) с
приложением и номером, рядом — строка с той же механикой и противоположным
вердиктом. Решение по каждой строке — «какой из двух кодов верный / нужна ли
тишина», и его можно принять, не открывая код.

Строки с одинаковой механикой и разным вердиктом печатаются парами: это
единственный способ увидеть, что «ошибка» модели на самом деле — вопрос
разметки. Инструмент ничего не меняет и не обучает: он только показывает.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

from app.services.classifier import build_classifier  # noqa: E402

from eval.run_eval import LABELED, codes_match, load_labeled  # noqa: E402
from ml.features import stems  # noqa: E402
from ml.lexicon import bridged_targets  # noqa: E402

KINDS = ("sibling_code", "generic_category", "bridge_guess", "other")

BRIEF_QUESTIONS = {
    "sibling_code": (
        "Модель выбрала соседний код того же семейства. Вопрос: какая из двух позиций "
        "покрывает товар — или обе допустимы и нужен допуск по родителю?"
    ),
    "generic_category": (
        "Модель ответила обобщающей позицией перечня, а разметка говорит «вне перечня». "
        "Вопрос: обобщающая позиция покрывает товар (как «Костюм мужской» → 14.1 «Одежда») "
        "или позиции действительно нет?"
    ),
    "bridge_guess": (
        "Ответ получен через мост лексикона, но строка помечена спорной. Вопрос: какой код "
        "верный — или товар вне перечня?"
    ),
}

# Decisions a code cannot make: they need the specialist, and they are the reason the
# same mechanic is labelled "correct" and "out of catalog" in the same file.
NEEDS_LAWYER = {"sibling_code", "generic_category", "bridge_guess"}


def code_family(code: str, depth: int = 3) -> str:
    """Parent code of ``depth`` segments — «26.20.16» and «26.20.18» share one."""
    parts = [part for part in (code or "").split(".") if part]
    return ".".join(parts[:depth])


def head_stems(query: str) -> set[str]:
    """Content stems of the query's head (first) word — the product itself."""
    words = [word for word in query.split() if word[:1].isalpha()]
    return set(stems(words[0])) if words else set()


def answer_mechanic(row: dict, response) -> str:
    """*How* the answer was reached, regardless of whether it was right.

    This is the axis the dataset contradicts itself on: «Костюм мужской» → 14.1
    «Одежда» and «Конструктор детский» → 32.4 «Игры и игрушки» are the same
    mechanic (a bridge to a generic position) with opposite verdicts. Comparing
    rows by correctness instead would hide the contradiction, because the correct
    ones would be filed as "exact matches".
    """
    title_stems = set(stems(response.title or ""))
    bridged = bool(bridged_targets(row["query"]))
    if not (title_stems & head_stems(row["query"])):
        return "generic_category" if bridged else "other"
    return "bridge_guess" if bridged else "direct"


def classify_kind(row: dict, response) -> str:
    """Verdict-level kind of one answer, used to group the errors."""
    expected = (row.get("expected_okpd2") or "").strip()
    predicted = response.code or ""
    if expected and codes_match(predicted, expected):
        return "exact"
    if expected and predicted and code_family(predicted) == code_family(expected):
        return "sibling_code"
    return answer_mechanic(row, response)


def catalog_rows_by_code(catalog_path: Path) -> dict[str, list[str]]:
    """``code -> ["Приложение 4, позиция 12: название", …]`` for quoting in the brief."""
    rows: dict[str, list[str]] = defaultdict(list)
    with catalog_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            code = (row.get("okpd2") or "").strip()
            name = (row.get("name") or "").strip()
            if not code or not name:
                continue
            appendix = (row.get("appendix") or "").strip()
            position = (row.get("position") or "").strip()
            where = " · ".join(part for part in (f"прил. {appendix}" if appendix else "",
                                                 f"поз. {position}" if position else "") if part)
            rows[code].append(f"{name}" + (f" ({where})" if where else ""))
    return rows


def _quote(rows: dict[str, list[str]], code: str, limit: int = 3) -> str:
    if not code:
        return "—"
    found = rows.get(code) or []
    if not found:
        return "в каталоге такой позиции нет"
    shown = "; ".join(found[:limit])
    extra = f" и ещё {len(found) - limit}" if len(found) > limit else ""
    return shown + extra


def render_brief(
    outcomes: list[dict],
    catalog_rows: dict[str, list[str]],
    *,
    artifact: str,
) -> str:
    """Markdown document the specialist can answer line by line."""
    answered_correct = [row for row in outcomes if row["predicted"] and row["correct"]]
    wrong = [row for row in outcomes if row["predicted"] and not row["correct"]]
    silent = [row for row in outcomes if not row["predicted"]]
    disputed = [row for row in outcomes if row["disputed"]]

    lines = [
        "# Вопросы к разметке: где кончается модель",
        "",
        f"Скорит: `{artifact}`. Набор: `eval/labeled_examples.csv` — строк: {len(outcomes)} "
        f"({sum(1 for r in outcomes if r['expected'])} с ожидаемым кодом, "
        f"{sum(1 for r in outcomes if not r['expected'])} «вне перечня»).",
        "",
        "Документ собран `eval/review_errors.py --brief`. Он перечисляет **только те** "
        "строки, где решение принимает специалист: модель ошиблась не потому, что плохо "
        "искала, а потому, что два одинаковых по механике случая размечены по-разному.",
        "",
        f"Сейчас: верных ответов {len(answered_correct)}, неверных {len(wrong)}, "
        f"тишина {len(silent)} (из них верная "
        f"{sum(1 for r in silent if r['correct'])}), помечено спорными {len(disputed)}.",
        "",
    ]

    by_kind: dict[str, list[dict]] = defaultdict(list)
    for row in wrong:
        by_kind[row["kind"]].append(row)

    for index, kind in enumerate(
        [kind for kind in KINDS if kind in NEEDS_LAWYER and by_kind.get(kind)], start=1
    ):
        items = by_kind[kind]
        # Highest-confidence examples first: they show the contradiction most clearly.
        mirror = sorted(
            (row for row in answered_correct if row["mechanic"] == kind),
            key=lambda item: -item["confidence"],
        )
        lines += [
            f"## {index}. {kind} — {len(items)} стр. "
            f"(та же механика размечена верной в {len(mirror)} строках)",
            "",
            BRIEF_QUESTIONS[kind],
            "",
            "| Запрос | Разметка ожидает | Модель ответила | Уверенность |",
            "| --- | --- | --- | ---: |",
        ]
        for row in items:
            expected = (
                f"`{row['expected']}` — {_quote(catalog_rows, row['expected'])}"
                if row["expected"]
                else "тишину (позиции нет в перечне)"
            )
            predicted = f"`{row['predicted']}` — {_quote(catalog_rows, row['predicted'])}"
            lines.append(
                f"| {row['query']} | {expected} | {predicted} | {row['confidence']:.3f} |"
            )
        lines.append("")
        if mirror:
            lines += [
                "Та же механика, но разметка считает ответ верным:",
                "",
                "| Запрос | Ответ | Разметка |",
                "| --- | --- | --- |",
            ]
            for row in mirror[:8]:
                lines.append(
                    f"| {row['query']} | `{row['predicted']}` — "
                    f"{_quote(catalog_rows, row['predicted'], limit=1)} | верно |"
                )
            lines.append("")

    remaining = [row for row in wrong if row["kind"] not in NEEDS_LAWYER]
    if remaining:
        lines += [
            f"## Остальное — {len(remaining)} стр. (похоже на настоящие ошибки модели)",
            "",
            "| Запрос | Ожидалось | Ответ | Статус | Уверенность | Примечание разметки |",
            "| --- | --- | --- | --- | ---: | --- |",
        ]
        for row in remaining:
            expected = f"`{row['expected']}`" if row["expected"] else "тишина"
            note = (row["notes"] or "").replace("|", "/")[:70]
            lines.append(
                f"| {row['query']} | {expected} | `{row['predicted']}` | {row['status']} | "
                f"{row['confidence']:.3f} | {note} |"
            )
        lines.append("")

    lines += [
        "## Как ответить",
        "",
        "Достаточно одной строки на пункт, например:",
        "",
        "```",
        "sibling_code: Конвектор отопительный стальной — верен 25.21.11.150 (модель права)",
        "generic_category: Конструктор детский — 32.4 покрывает, разметку исправить на 32.4",
        "```",
        "",
        "После правок `eval/labeled_examples.csv` перемеряются `eval/baselines/*.json` "
        "(команда в `docs/catalog-update.md`) — и только тогда имеет смысл трогать модель.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled", type=Path, default=LABELED)
    parser.add_argument("--csv", type=Path, default=None, help="куда выгрузить разбор")
    parser.add_argument("--limit", type=int, default=40, help="сколько строк печатать")
    parser.add_argument(
        "--brief",
        type=Path,
        default=None,
        help="записать документ для специалиста (markdown) вместо только консольного вывода",
    )
    parser.add_argument(
        "--catalog", type=Path, default=ROOT / "data" / "data.csv", help="каталог для цитат"
    )
    args = parser.parse_args()

    rows = load_labeled(args.labeled)
    # Catalog backend + the artifacts the image would load (MODELS_DIR first).
    classifier = build_classifier("catalog")
    reranker = getattr(classifier, "_reranker", None)
    print(
        f"Скорит: {getattr(reranker, 'meta', {}).get('artifact', 'reranker')} · "
        f"порог review={getattr(reranker, 'threshold_review', float('nan')):.3f}",
        file=sys.stderr,
    )

    outcomes: list[dict] = []
    for row in rows:
        query = (row["query"] or "").strip()
        expected = (row["expected_okpd2"] or "").strip()
        disputed = (row.get("disputed") or "0").strip() == "1"
        response = classifier.classify(query)
        answered = response.status != "not_found"
        correct = codes_match(response.code, expected) if expected else not answered
        outcomes.append(
            {
                "query": query,
                "expected": expected,
                "predicted": response.code or "",
                "status": response.status,
                "confidence": response.confidence,
                "correct": correct,
                "disputed": disputed,
                "notes": (row.get("notes") or "").strip(),
                "kind": classify_kind(row, response) if answered else "silent",
                "mechanic": answer_mechanic(row, response) if answered else "silent",
            }
        )

    wrong_answered = [row for row in outcomes if row["predicted"] and not row["correct"]]
    print(
        f"\nВсего строк: {len(outcomes)} · отвечено верно: "
        f"{sum(1 for r in outcomes if r['predicted'] and r['correct'])} · "
        f"отвечено неверно: {len(wrong_answered)} · "
        f"тишина верная: {sum(1 for r in outcomes if not r['predicted'] and r['correct'])}"
    )

    by_kind: dict[str, list[dict]] = defaultdict(list)
    for row in wrong_answered:
        by_kind[row["kind"]].append(row)
    # The mirror is found by *mechanic*, not by verdict: the point is that the same
    # way of answering is labelled correct somewhere else in the same file.
    correct_by_kind: dict[str, list[dict]] = defaultdict(list)
    for row in outcomes:
        if row["predicted"] and row["correct"]:
            correct_by_kind[row["mechanic"]].append(row)

    print("\nНеверные ответы по механике:")
    for kind in KINDS:
        items = by_kind.get(kind, [])
        if not items:
            continue
        same_mechanism_correct = len(correct_by_kind.get(kind, []))
        print(
            f"  {kind:18s} {len(items):2d} строк · та же механика помечена верной "
            f"в {same_mechanism_correct} строках"
        )

    for kind in KINDS:
        items = by_kind.get(kind, [])
        if not items:
            continue
        print(f"\n=== {kind} ({len(items)}) ===")
        for row in items[: args.limit]:
            flag = " [спорно]" if row["disputed"] else ""
            print(
                f"  {row['query'][:36]:36s} → {row['predicted']:14s} {row['status']:6s} "
                f"conf={row['confidence']:.3f} ожидалось={row['expected'] or '(тишина)'}{flag}"
            )
            if row["notes"]:
                print(f"      {row['notes'][:96]}")
        mirror = correct_by_kind.get(kind, [])
        if mirror and kind in {"generic_category", "bridge_guess"}:
            print("  та же механика, но размечено верно:")
            for row in mirror[:5]:
                print(
                    f"    {row['query'][:36]:36s} → {row['predicted']:14s} "
                    f"conf={row['confidence']:.3f} ожидалось={row['expected']}"
                )

    if args.csv:
        with args.csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "query", "expected", "predicted", "status", "confidence",
                    "correct", "disputed", "kind", "mechanic", "notes",
                ],
            )
            writer.writeheader()
            writer.writerows(outcomes)
        print(f"\nРазбор выгружен: {args.csv}")

    if args.brief:
        info = getattr(classifier, "vector_info", None) or {}
        artifact = info.get("reranker") or "reranker.json"
        args.brief.parent.mkdir(parents=True, exist_ok=True)
        args.brief.write_text(
            render_brief(outcomes, catalog_rows_by_code(args.catalog), artifact=artifact),
            encoding="utf-8",
        )
        print(f"\nДокумент для специалиста: {args.brief}")

    counts = Counter(row["kind"] for row in wrong_answered)
    disputed_wrong = sum(1 for row in wrong_answered if row["disputed"])
    print(
        f"\nИтог: неверных ответов {len(wrong_answered)}, из них помечены спорными "
        f"{disputed_wrong}; механики — {dict(counts)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
