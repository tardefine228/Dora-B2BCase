"""Сверить выгрузку результатов сайта с эталонами файлов клиента.

Сайт отдаёт `classification-results.csv` (разделитель «;», UTF-8 BOM). Скрипт
сопоставляет каждую такую выгрузку с эталоном из
``api/tests/fixtures/client_files/expected/`` и печатает построчный вердикт:
какая позиция распознана верно, какая — нет, где потерялись строки и откуда
взялся лишний товар.

    python scripts/compare_client_results.py ~/Downloads            # папка с выгрузками
    python scripts/compare_client_results.py results/*.csv --verbose
    python scripts/compare_client_results.py results/ --report report.md

Код возврата 1, если есть хотя бы одно расхождение (кроме задокументированных
известных промахов), — так проверку можно ставить в CI.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "api" / "tests" / "fixtures" / "client_files"
EXPECTED = FIXTURES / "expected"

_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _SPACE.sub(" ", str(text)).strip().casefold()


def load_expectations(directory: Path = EXPECTED) -> list[dict]:
    specs = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "known_gaps.json":
            continue
        spec = json.loads(path.read_text(encoding="utf-8"))
        spec["_path"] = path
        specs.append(spec)
    return specs


@dataclass
class Result:
    query: str
    code: str | None
    status: str
    confidence: float


def read_results(path: Path) -> list[Result]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    rows = list(csv.DictReader(io.StringIO(text), delimiter=";"))
    if not rows:  # одиночная строка без заголовка — попробуем запятую
        rows = list(csv.DictReader(io.StringIO(text), delimiter=","))
    results = []
    for row in rows:
        query = (row.get("query") or "").strip()
        if not query:
            continue
        try:
            confidence = float(row.get("confidence") or 0.0)
        except ValueError:
            confidence = 0.0
        results.append(
            Result(
                query=query,
                code=(row.get("code") or "").strip() or None,
                status=(row.get("status") or "").strip() or "unknown",
                confidence=confidence,
            )
        )
    return results


@dataclass
class Verdict:
    file: str
    ok: list[str] = field(default_factory=list)
    wrong: list[str] = field(default_factory=list)
    known_gap: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return bool(self.wrong or self.missing or self.unexpected or self.parse_errors)


def _match_score(spec: dict, results: list[Result]) -> int:
    """Сколько строк выгрузки указывают на этот эталон (с учётом битых названий)."""
    expected_queries = [normalize(item["query"]) for item in spec["items"]]
    score = 0
    for result in results:
        query = normalize(result.query)
        if any(name in query for name in expected_queries):
            score += 1
    return score


def compare(spec: dict, results: list[Result]) -> Verdict:
    verdict = Verdict(file=spec["file"])
    consumed: set[int] = set()

    for item in spec["items"]:
        wanted = normalize(item["query"])
        match: tuple[int, Result] | None = None
        for index, result in enumerate(results):
            if index in consumed:
                continue
            if normalize(result.query) == wanted:
                match = (index, result)
                break
        exact = match is not None
        if match is None:
            for index, result in enumerate(results):
                if index in consumed:
                    continue
                if wanted and wanted in normalize(result.query):
                    match = (index, result)
                    break
        if match is None:
            verdict.missing.append(item["query"])
            continue

        index, result = match
        consumed.add(index)
        if not exact:
            verdict.parse_errors.append(
                f"«{item['query']}» найдено внутри «{result.query}» — наименование "
                "извлечено вместе с лишними столбцами (проверьте разбор файла)"
            )

        code_ok = item["codes"] is None or result.code in item["codes"]
        status_ok = result.status in item["statuses"]
        if code_ok and status_ok:
            verdict.ok.append(f"{item['query']} → {result.code} ({result.status})")
        elif item.get("known_gap"):
            verdict.known_gap.append(
                f"{item['query']} → {result.code} ({result.status}); ожидалось "
                f"{item['codes']} / {item['statuses']}"
            )
        else:
            reason = []
            if not code_ok:
                reason.append(f"код {result.code}, ожидался {item['codes']}")
            if not status_ok:
                reason.append(f"статус {result.status}, ожидался {item['statuses']}")
            verdict.wrong.append(f"{item['query']}: " + "; ".join(reason))

    for index, result in enumerate(results):
        if index not in consumed:
            verdict.unexpected.append(f"{result.query} → {result.code} ({result.status})")
    return verdict


def format_report(verdicts: list[Verdict], total_files: int) -> str:
    lines: list[str] = []
    ok = wrong = gaps = missing = unexpected = parse = 0

    for verdict in verdicts:
        ok += len(verdict.ok)
        wrong += len(verdict.wrong)
        gaps += len(verdict.known_gap)
        missing += len(verdict.missing)
        unexpected += len(verdict.unexpected)
        parse += len(verdict.parse_errors)

        status = "OK" if not verdict.failed else "РАСХОЖДЕНИЯ"
        lines.append(f"\n=== {verdict.file} — {status} ===")
        lines.append(
            f"  верно: {len(verdict.ok)} | неверно: {len(verdict.wrong)} | "
            f"известные промахи: {len(verdict.known_gap)} | "
            f"потеряно: {len(verdict.missing)} | лишних строк: {len(verdict.unexpected)}"
        )
        for message in verdict.parse_errors:
            lines.append(f"  [РАЗБОР] {message}")
        for message in verdict.wrong:
            lines.append(f"  [ОШИБКА] {message}")
        for message in verdict.missing:
            lines.append(f"  [ПОТЕРЯНО] {message}")
        for message in verdict.unexpected:
            lines.append(f"  [ЛИШНЕЕ] {message}")
        for message in verdict.known_gap:
            lines.append(f"  [известный промах] {message}")

    lines.append("\n--- ИТОГО ---")
    lines.append(f"файлов сверено: {len(verdicts)} из {total_files}")
    lines.append(f"позиций верно:        {ok}")
    lines.append(f"позиций неверно:      {wrong}")
    lines.append(f"потерянных позиций:   {missing}")
    lines.append(f"лишних строк:         {unexpected}")
    lines.append(f"ошибок разбора:       {parse}")
    lines.append(f"известных промахов:   {gaps} (задокументированы, не считаются регрессом)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="файлы или папки с classification-results*.csv")
    parser.add_argument("--expected", default=str(EXPECTED), help="папка с эталонами")
    parser.add_argument("--report", help="сохранить отчёт в markdown-файл")
    parser.add_argument("--verbose", action="store_true", help="печатать и верные позиции")
    args = parser.parse_args(argv)

    specs = load_expectations(Path(args.expected))
    files: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        files.extend(sorted(path.glob("*.csv")) if path.is_dir() else [path])
    if not files:
        print("не найдено ни одного csv", file=sys.stderr)
        return 2

    verdicts: list[Verdict] = []
    seen_files: set[str] = set()
    for path in files:
        results = read_results(path)
        if not results:
            continue
        ranked = sorted(specs, key=lambda spec: _match_score(spec, results), reverse=True)
        best = ranked[0]
        if _match_score(best, results) == 0:
            print(f"{path.name}: не похоже ни на один файл клиента — пропущен", file=sys.stderr)
            continue
        seen_files.add(best["file"])
        verdict = compare(best, results)
        verdict.file = f"{best['file']}  (выгрузка: {path.name})"
        verdicts.append(verdict)

    report = format_report(verdicts, len(specs))
    print(report)

    if args.verbose:
        print("\n--- верные позиции ---")
        for verdict in verdicts:
            for message in verdict.ok:
                print(f"  {message}")

    missing_files = {spec["file"] for spec in specs} - seen_files
    if missing_files:
        print("\nнет выгрузок для: " + ", ".join(sorted(missing_files)), file=sys.stderr)

    if args.report:
        Path(args.report).write_text("```\n" + report + "\n```\n", encoding="utf-8")
        print(f"\nотчёт сохранён: {args.report}")

    return 1 if any(verdict.failed for verdict in verdicts) else 0


if __name__ == "__main__":
    raise SystemExit(main())
