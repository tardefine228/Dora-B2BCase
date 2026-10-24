"""Turn specialist verdicts into training labels — deliberately, with a report.

The UI already writes three verdicts for every answer (`correct`, `incorrect`
with a corrected code, `absent`), and the CSV export carries them in
`user_verdict` / `corrected_code`. This tool converts those rows into the format
of ``eval/labeled_examples.csv`` so that a retrain has something to learn from.

Two rules are built in, because the wrong shortcut here destroys the only honest
measurement the project has:

* the output is a **separate** file (``eval/labeled_from_feedback.csv`` by
  default); ``eval/labeled_examples.csv`` is the fixed measurement set and is
  never modified — otherwise the next report would measure the model on the rows
  it was just trained on (``docs/labeling-doctrine.md``);
* rows are only imported with an explicit verdict. A log export with empty
  `user_verdict` (the normal state of a fresh export) imports **nothing** and
  says so, instead of silently relabelling the model's own answers as truth —
  a model cannot teach itself a new floor.

Usage::

    python eval/feedback_loop.py --input "classification-results (5).csv" --report
    python eval/feedback_loop.py --feedback-sqlite data/classification_feedback.sqlite
    python eval/feedback_loop.py --input export.csv --out eval/labeled_from_feedback.csv

The printed report splits the verdicts by the status the model gave, which is
the number a specialist actually wants: how many ``review`` rows turned out
correct (the width of the "нашёл, но не уверен" band) and how many answered
rows turned out to be absent from the list (false codes).
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELED = ROOT / "eval" / "labeled_examples.csv"
DEFAULT_OUT = ROOT / "eval" / "labeled_from_feedback.csv"
HEADER = ["query", "expected_okpd2", "expected_appendix", "disputed", "notes", "label_kind"]

VERDICTS = {"correct", "incorrect", "absent"}
STATUSES = ("ok", "review", "not_found")


@dataclass
class Imported:
    query: str
    expected_okpd2: str
    expected_appendix: str = ""
    disputed: int = 0
    notes: str = ""
    label_kind: str = "code"


@dataclass
class Report:
    imported: list[Imported] = field(default_factory=list)
    duplicates: int = 0
    skipped_no_verdict: int = 0
    skipped_invalid: int = 0
    # (verdict, model status) -> count, for the verdicts that had a status.
    by_verdict_status: Counter = field(default_factory=Counter)
    reasons: list[str] = field(default_factory=list)


def _key(query: str) -> str:
    return " ".join((query or "").casefold().split())


def _existing_queries(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {_key(row.get("query") or "") for row in csv.DictReader(handle, delimiter=";")}


def _as_label(
    verdict: str, predicted: str, corrected: str, query: str, status: str
) -> Imported | None:
    """One verdict -> one labeled row, or ``None`` when the row cannot be a label."""
    verdict = (verdict or "").strip().lower()
    predicted = (predicted or "").strip()
    corrected = (corrected or "").strip()
    stamp = f"вердикт специалиста {date.today().isoformat()}"
    source = f"{stamp}; ответ модели: {status or '—'} {predicted or '—'}"
    if verdict == "correct":
        if not predicted:
            return None  # "верно" without a code proves nothing
        return Imported(query=query, expected_okpd2=predicted, notes=source, label_kind="code")
    if verdict == "incorrect":
        if not corrected:
            return None  # the UI requires the corrected code; a log may not have it
        return Imported(query=query, expected_okpd2=corrected, notes=source, label_kind="code")
    if verdict == "absent":
        return Imported(query=query, expected_okpd2="", notes=source, label_kind="absent")
    return None


def _row_status(row: dict) -> str:
    # Exports from the UI carry ``status``; feedback rows do not, so the caller
    # may pass it in. An unknown status is reported as an empty string.
    return (row.get("status") or "").strip().lower()


def import_rows(rows: list[dict], known: set[str], existing: set[str]) -> Report:
    """Convert export/feedback rows into labels, skipping what cannot be a label."""
    report = Report()
    for row in rows:
        query = (row.get("query") or "").strip()
        if not query:
            report.skipped_invalid += 1
            continue
        verdict = (row.get("user_verdict") or row.get("verdict") or "").strip().lower()
        if not verdict:
            report.skipped_no_verdict += 1
            continue
        if verdict not in VERDICTS:
            report.skipped_invalid += 1
            report.reasons.append(f"неизвестный вердикт {verdict!r}: {query[:60]}")
            continue
        status = _row_status(row)
        report.by_verdict_status[(verdict, status)] += 1
        label = _as_label(
            verdict,
            row.get("predicted_code") or row.get("code") or "",
            row.get("corrected_code") or "",
            query,
            status,
        )
        if label is None:
            report.skipped_invalid += 1
            report.reasons.append(f"вердикт {verdict!r} без нужного кода: {query[:60]}")
            continue
        key = _key(query)
        if key in known or key in existing:
            report.duplicates += 1
            continue
        existing.add(key)
        report.imported.append(label)
    return report


def read_export(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def read_feedback_sqlite(path: Path) -> list[dict]:
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                "SELECT query, predicted_code, verdict, corrected_code FROM classification_feedback"
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [dict(row) for row in rows]


def write_labels(path: Path, labels: list[Imported]) -> None:
    """Append to the staging file, keeping one query per line and a stable header."""
    existing = read_export(path) if path.is_file() else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";", lineterminator="\n")
        writer.writerow(HEADER)
        for row in existing:
            writer.writerow([row.get(column, "") for column in HEADER])
        for label in labels:
            writer.writerow(
                [label.query, label.expected_okpd2, label.expected_appendix,
                 label.disputed, label.notes, label.label_kind]
            )


def format_report(report: Report, out: Path) -> str:
    lines = [
        "Контур разметки: вердикты специалиста -> файл меток",
        "",
        f"Импортировано новых строк: {len(report.imported)}",
        f"Пропущено дублей (уже есть в наборе): {report.duplicates}",
        f"Пропущено без вердикта: {report.skipped_no_verdict}",
        f"Пропущено как непригодные: {report.skipped_invalid}",
    ]
    if report.imported:
        kinds = Counter(label.label_kind for label in report.imported)
        lines.append("")
        composition = ", ".join(f"{kind}={count}" for kind, count in sorted(kinds.items()))
        lines.append(f"Состав импорта: {composition}")
        lines.append(f"Файл: {out} (набор измерения eval/labeled_examples.csv не изменялся)")
    if report.by_verdict_status:
        lines += [
            "",
            "Вердикты против статуса модели:",
            "",
            "| вердикт | статус | строк |",
            "| --- | --- | ---: |",
        ]
        for (verdict, status), count in sorted(report.by_verdict_status.items()):
            lines.append(f"| {verdict} | {status or '—'} | {count} |")
        review_correct = report.by_verdict_status[("correct", "review")]
        answered_absent = (
            report.by_verdict_status[("absent", "ok")]
            + report.by_verdict_status[("absent", "review")]
        )
        lines += [
            "",
            f"Полоса `review`, где специалист сказал «верно»: {review_correct} — это и есть",
            "ширина «нашёл, но не уверен»; такие строки поддерживают код, но не двигают порог.",
            f"Отвеченные строки, которые оказались «нет в перечне»: {answered_absent} — это",
            "ложные коды, и именно они (а не проценты на размеченном наборе) говорят, что",
            "порог/лексикон слишком смелы.",
        ]
    if report.skipped_no_verdict and not report.imported:
        lines += [
            "",
            "Вердиктов нет: в выгрузке пустые колонки user_verdict/corrected_code.",
            "Это журнал классификаций, а не разметка. Заполните вердикты в интерфейсе",
            "(кнопки «Код верный» / «Код неверный» / «Нет в перечне») — они сохраняются",
            "в API и попадают в следующую выгрузку.",
        ]
    if report.reasons:
        lines += [
            "",
            "Причины пропуска (первые 10):",
            *[f"- {reason}" for reason in report.reasons[:10]],
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="CSV-выгрузка результатов (с user_verdict)")
    parser.add_argument("--feedback-sqlite", type=Path, help="SQLite-база classification_feedback")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="только отчёт, без записи")
    args = parser.parse_args()

    rows: list[dict] = []
    if args.input:
        rows += read_export(args.input)
    if args.feedback_sqlite:
        rows += read_feedback_sqlite(args.feedback_sqlite)
    if not rows:
        parser.error("нужен --input и/или --feedback-sqlite с непустым содержимым")

    known = _existing_queries(LABELED)
    existing = _existing_queries(args.out)
    report = import_rows(rows, known, existing)
    print(format_report(report, args.out))
    if report.imported and not args.dry_run:
        write_labels(args.out, report.imported)
        print(
            "\nДальше: переобучение (python scripts/build_model.py --dense --gate) и отдельный"
            "\nзамер (eval/run_eval.py). Вливать файл в eval/labeled_examples.csv нельзя:"
            "\nэто набор измерения."
        )


if __name__ == "__main__":
    main()
