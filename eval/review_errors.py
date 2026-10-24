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


def code_family(code: str, depth: int = 3) -> str:
    """Parent code of ``depth`` segments — «26.20.16» and «26.20.18» share one."""
    parts = [part for part in (code or "").split(".") if part]
    return ".".join(parts[:depth])


def head_stems(query: str) -> set[str]:
    """Content stems of the query's head (first) word — the product itself."""
    words = [word for word in query.split() if word[:1].isalpha()]
    return set(stems(words[0])) if words else set()


def classify_kind(row: dict, response) -> str:
    """Mechanism behind one answer, used to compare rows with different verdicts."""
    expected = (row.get("expected_okpd2") or "").strip()
    predicted = response.code or ""
    if expected and predicted and code_family(predicted) == code_family(expected):
        return "sibling_code"
    title_stems = set(stems(response.title or ""))
    if not (title_stems & head_stems(row["query"])):
        if bridged_targets(row["query"]):
            return "generic_category"
        return "other"
    if bridged_targets(row["query"]):
        return "bridge_guess"
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled", type=Path, default=LABELED)
    parser.add_argument("--csv", type=Path, default=None, help="куда выгрузить разбор")
    parser.add_argument("--limit", type=int, default=40, help="сколько строк печатать")
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
    correct_by_kind: dict[str, list[dict]] = defaultdict(list)
    for row in outcomes:
        if row["predicted"] and row["correct"]:
            correct_by_kind[row["kind"]].append(row)

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
                    "correct", "disputed", "kind", "notes",
                ],
            )
            writer.writeheader()
            writer.writerows(outcomes)
        print(f"\nРазбор выгружен: {args.csv}")

    counts = Counter(row["kind"] for row in wrong_answered)
    disputed_wrong = sum(1 for row in wrong_answered if row["disputed"])
    print(
        f"\nИтог: неверных ответов {len(wrong_answered)}, из них помечены спорными "
        f"{disputed_wrong}; механики — {dict(counts)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
