"""Retrieval recall@k on a labeled set, by slices (ML-5).

The reranker can only suggest positions that retrieval shortlisted.
This script measures how often the expected OKPD2 position is inside
the lexical shortlist (and, when vector search is on, the dense
top-k) — the *ceiling* of the whole pipeline. If recall@25 is high
(>= 90%), the reserve is in reranking/thresholds, not in semantics;
if it is low on real traffic, an embedder upgrade becomes the
priority (audit §3.3, задача ML-5).

Input: any labeled CSV in the format of ``eval/labeled_examples.csv``
(``query;expected_okpd2;...``). The intended input is the real
procurement queries of ``eval/real_queries.csv`` (задача DA-1);
until it lands, the catalog-derived labeled set is the measurable
proxy and is reported as such.

Usage:
    PYTHONPATH=api python eval/recall_at_k.py [--labeled eval/labeled_examples.csv]
        [--output eval/recall_report.md] [--backend catalog]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "api"))

from app.core.config import settings  # noqa: E402
from app.services.classifier import build_classifier  # noqa: E402

LABELED = ROOT / "eval" / "labeled_examples.csv"
CATALOG_CSV = ROOT / "data" / "data.csv"
TOP_K_VALUES = (5, 10, 25)


def codes_match(predicted: str | None, expected: str) -> bool:
    if not predicted or not expected:
        return False
    predicted, expected = predicted.strip(), expected.strip()
    return (
        predicted == expected
        or predicted.startswith(expected + ".")
        or expected.startswith(predicted + ".")
    )


def load_labeled(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def catalog_name_lengths(path: Path) -> dict[str, int]:
    """Longest official name per OKPD2 code (for the long-name slice)."""
    lengths: dict[str, int] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            code = (row.get("okpd2") or "").strip()
            name = row.get("name") or ""
            if code and len(name) > lengths.get(code, 0):
                lengths[code] = len(name)
    return lengths


def _shortlist(classifier, query: str) -> list[dict]:
    """Candidate rows exactly as the API's catalog backend builds them."""
    from ml.fuzzy import autocorrect
    from ml.lexicon import expand_query

    normalized = autocorrect(query.strip())
    tokens = {token for token in normalized.casefold().split() if len(token) > 1}
    if not tokens:
        return []
    shortlist = classifier._retrieve(normalized, tokens)
    if classifier._dense is not None:
        from ml.dense import fuse_shortlist
        from ml.features import prepare_candidate

        prepared = prepare_candidate(expand_query(normalized), idf=classifier._idf)
        fused = fuse_shortlist(classifier._dense, classifier.index, shortlist, normalized, prepared)
        return [row for _score, row, _cosine, _top in fused]
    return [row for _score, row in shortlist]


def _dense_topk(classifier, query: str, k: int) -> list[dict]:
    if classifier._dense is None:
        return []
    # The dense index is built over classifier.index in the same
    # order, so a dense row index maps directly onto the catalog rows.
    hits = classifier._dense.top(query, k)
    return [
        classifier.index[index]
        for index, _score in hits
        if index < len(classifier.index)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled", type=Path, default=LABELED)
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "recall_report.md")
    parser.add_argument("--backend", default="catalog", choices=["catalog", "qdrant"])
    parser.add_argument(
        "--long-chars",
        type=int,
        default=200,
        help="порог длины названия для среза «длинные названия»",
    )
    args = parser.parse_args()

    settings.classifier_backend = args.backend
    classifier = build_classifier(args.backend)
    if not hasattr(classifier, "index"):
        print(f"Бэкенд {args.backend} не поддерживает замер recall — нужен catalog")
        return 2

    rows = [row for row in load_labeled(args.labeled) if (row["expected_okpd2"] or "").strip()]
    name_lengths = catalog_name_lengths(CATALOG_CSV)

    # Per-row retrieval facts.
    facts: list[dict] = []
    for row in rows:
        query = (row["query"] or "").strip()
        expected = row["expected_okpd2"].strip()
        shortlist = _shortlist(classifier, query)
        shortlist_codes = [candidate["okpd2"] for candidate in shortlist]
        dense_codes = [
            candidate["okpd2"]
            for candidate in _dense_topk(classifier, query, max(TOP_K_VALUES))
        ]
        facts.append(
            {
                "query": query,
                "expected": expected,
                "words": len(query.split()),
                "disputed": (row.get("disputed") or "0").strip() == "1",
                "name_len": name_lengths.get(expected, 0),
                "lexical_ranks": [
                    index
                    for index, code in enumerate(shortlist_codes)
                    if codes_match(code, expected)
                ],
                "dense_ranks": [
                    index
                    for index, code in enumerate(dense_codes)
                    if codes_match(code, expected)
                ],
            }
        )

    def recall(rows_slice: list[dict], k: int, source: str) -> float | None:
        if not rows_slice:
            return None
        hits = 0
        for fact in rows_slice:
            ranks = fact[f"{source}_ranks"]
            if any(rank < k for rank in ranks):
                hits += 1
        return round(100 * hits / len(rows_slice), 1)

    slices: list[tuple[str, list[dict]]] = [
        ("все запросы с кодом", facts),
        ("короткие запросы (1 слово)", [f for f in facts if f["words"] == 1]),
        ("средние запросы (2 слова)", [f for f in facts if f["words"] == 2]),
        ("длинные запросы (3+ слова)", [f for f in facts if f["words"] >= 3]),
        (
            "длинные названия позиции (>200 симв.)",
            [f for f in facts if f["name_len"] > args.long_chars],
        ),
        (
            "короткие названия позиции (≤200 симв.)",
            [f for f in facts if 0 < f["name_len"] <= args.long_chars],
        ),
        ("спорные разметки", [f for f in facts if f["disputed"]]),
        ("чистые разметки", [f for f in facts if not f["disputed"]]),
    ]
    # Section slices: первые две цифры ОКПД2.
    sections: dict[str, list[dict]] = defaultdict(list)
    for fact in facts:
        sections[fact["expected"].split(".")[0]].append(fact)
    for section in sorted(sections):
        slices.append((f"раздел ОКПД2 {section}.x", sections[section]))

    lines = [
        "# Recall@k retrieval: шорт-лист до реранкера",
        "",
        f"**Набор:** {args.labeled} ({len(facts)} запросов с ожидаемым кодом"
        + (
            "; каталогный прокси — целевой вход `eval/real_queries.csv`"
            " (задача DA-1) ещё не выгружен, замер на нём той же командой"
            if args.labeled.name != "real_queries.csv"
            else "; реальный трафик, DA-1"
        )
        + ")  ",
        f"**Бэкенд:** {args.backend}  ",
        f"**Эмбеддер:** {classifier._dense.tag if classifier._dense is not None else 'выключен'}  ",
        "",
        "Recall@k — доля запросов, где ожидаемая позиция уже есть в "
        "кандидатах на ступени k. Это потолок качества: реранкер и "
        "пороги не могут предложить то, что retrieval не нашёл.",
        "",
        "| Срез | Запросов | recall@5 | recall@10 | recall@25 (лексика) | recall@25 (вектора) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    dense_available = classifier._dense is not None
    for title, slice_rows in slices:
        cells = [str(len(slice_rows))]
        for k in TOP_K_VALUES:
            value = recall(slice_rows, k, "lexical")
            cells.append("—" if value is None else f"{value}%")
        if dense_available:
            value = recall(slice_rows, 25, "dense")
            cells.append("—" if value is None else f"{value}%")
        else:
            cells.append("—")
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    lines.append("")

    # Where the pipeline loses rows: retrieval missed vs reranker missed.
    missed_by_retrieval = [
        fact for fact in facts if not any(rank < 25 for rank in fact["lexical_ranks"])
    ]
    lines += [
        "## Промахи retrieval (ожидаемая позиция вне лексического top-25)",
        "",
    ]
    if missed_by_retrieval:
        lines += ["| Запрос | Ожидаемый код | Длина названия |", "| --- | --- | ---: |"]
        for fact in missed_by_retrieval:
            lines.append(
                f"| {fact['query']} | {fact['expected']} | {fact['name_len']} |"
            )
    else:
        lines.append("Нет — retrieval находит ожидаемую позицию для всех запросов.")
    lines.append("")

    args.output.write_text("\n".join(lines), encoding="utf-8")
    overall = recall(facts, 25, "lexical")
    print(f"Recall@25 (лексика): {overall}%  ({len(facts)} запросов)")
    if dense_available:
        print(f"Recall@25 (вектора): {recall(facts, 25, 'dense')}%")
    print(f"Промахов retrieval: {len(missed_by_retrieval)}")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
