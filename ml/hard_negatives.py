"""Hard-negative mining from specialist feedback (ML-4).

The synthetic training platform cannot know the queries real
procurement specialists type. When the feedback API (задача
BE-2: ``POST /classify/feedback``) accumulates verdicts, the
*incorrect* ones are the most valuable training signal there
is: a query the model answered wrongly, with the code a
specialist corrected it to.

This module turns the feedback the product actually writes into
``TrainingPair`` rows:

* verdict ``неверно``/``incorrect`` with a corrected code — the corrected
  position becomes a **positive**, the wrongly proposed one a
  **hard negative** (both are retrieved by the same lexical
  shortlist the API uses, so the pair features match serving);
* verdict ``не в перечне``/``absent`` — every retrieved candidate is a
  **negative** (the correct behaviour is silence);
* verdict ``верно``/``correct`` — no signal (the model was right).

Input formats
-------------
The miner reads the two schemas the product really produces, plus JSONL:

1. **UI export** (``frontend/src/files.ts``): ``query;code;title;status;
   confidence;...;user_verdict;corrected_code;verifier;verifier_note;row``;
2. **feedback store** (``api/app/services/feedback_store.py``, table
   ``classification_feedback``): ``query;predicted_code;verdict;corrected_code``
   — as CSV or as the SQLite database itself (``--feedback-sqlite``);
3. **JSONL** with the store's field names.

Columns are matched by alias, so both schemas work without a conversion step:

===========  ======================================================
field        accepted columns
===========  ======================================================
query        ``query``, ``name``
proposed     ``predicted_code`` (store/API), ``proposed_code``, ``code``
verdict      ``user_verdict`` (UI export), ``verdict`` (store)
corrected    ``corrected_code`` (both), ``correct_code``
comment      ``comment``, ``verifier_note``, ``note``
===========  ======================================================

Verdicts are matched case- and separator-insensitively
(``not_in_list`` ≡ ``not in list``); Russian and English spellings are accepted.

Guards
------
* An ``incorrect`` verdict whose corrected code is not in the catalog is
  **skipped**, not mined as all-negative: the corrected position is exactly the
  row the model must learn to rank first, and labelling its neighbours as
  negatives while dropping it teaches the opposite. A skipped record is counted
  and printed, never silently swallowed.
* A fresh export carries empty verdict columns (it is a log, not labels). The
  CLI then says so and mines nothing instead of inventing signal.
* An export is a log, so one query may arrive several times, with the same
  verdict or with contradictory ones. Rows collapse to **one verdict per
  query**, and a contradiction resolves towards ``incorrect``: the correction
  names the position the model must rank first. The contradiction is counted
  and printed, never resolved silently.
* The answer the model gave is a human-verified negative even when retrieval no
  longer ranks it: it is added to the pairs explicitly instead of being lost.

Usage::

    # what would an export contribute? (DoD: >= 300 pairs)
    python -m ml.hard_negatives "export.csv" --min-pairs 300
    python -m ml.hard_negatives --feedback-sqlite data/classification_feedback.sqlite

    # training picks the same reader up
    python -m ml.train_reranker --output models/reranker.json \\
        --feedback "export.csv"
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ml.train_reranker import TrainingPair

# Verdict spellings accepted in an export or in the store. Compared after
# ``_verdict_key`` (casefold, single spaces, no trailing dot, underscores as
# spaces), so ``NOT_IN_LIST`` and ``not in list`` are the same verdict.
VERDICT_CORRECT = {"верно", "верный", "правильно", "correct", "ok"}
VERDICT_INCORRECT = {
    "неверно",
    "не верно",
    "неправильно",
    "неверный",
    "incorrect",
    "wrong",
    "error",
}
VERDICT_OUT_OF_SCOPE = {
    "не в перечне",
    "нет в перечне",
    "вне перечня",
    "вне перечня.",
    "не включено в перечень",
    # The API's own spelling (`FeedbackRequest.verdict`): a specialist pressing
    # "товара нет в перечне" sends `absent`. Without it here the strongest
    # silence signal in the log would be silently ignored at training time.
    "absent",
    "not found",
    "not in list",
    "not listed",
    "out of scope",
}

# Column aliases: the first non-empty alias wins. ``predicted_code`` is what the
# feedback store and ``POST /feedback`` write; ``code`` is what the UI export
# writes; ``proposed_code`` is the historical name of this module's own fixtures.
COLUMNS = {
    "query": ("query", "name"),
    "proposed_code": ("predicted_code", "proposed_code", "code"),
    "verdict": ("user_verdict", "verdict"),
    "correct_code": ("corrected_code", "correct_code"),
    "comment": ("comment", "verifier_note", "note"),
    "status": ("status",),
}

SQLITE_FEEDBACK_TABLE = "classification_feedback"


@dataclass(frozen=True)
class FeedbackRecord:
    """One specialist verdict on a served answer."""

    query: str
    proposed_code: str = ""
    verdict: str = ""
    correct_code: str = ""
    comment: str = ""
    # The status the model itself answered with, when the source carries it
    # (UI export does, the store does not). Only used for the report.
    status: str = ""

    @property
    def normalized_verdict(self) -> str:
        return _verdict_key(self.verdict)

    @property
    def is_incorrect(self) -> bool:
        return self.normalized_verdict in VERDICT_INCORRECT

    @property
    def is_out_of_scope(self) -> bool:
        return self.normalized_verdict in VERDICT_OUT_OF_SCOPE

    @property
    def is_correct(self) -> bool:
        return self.normalized_verdict in VERDICT_CORRECT


def _verdict_key(value: str) -> str:
    """Comparison key of a verdict: casefold, folded separators, no trailing dot."""
    text = (value or "").casefold().replace("_", " ").replace("-", " ")
    return " ".join(text.split()).strip(".")


def _normalize_code(code: str) -> str:
    return (code or "").strip().replace(" ", "").replace(",", ".")


def _pick(row: dict, field_name: str) -> str:
    """First non-empty value among the column aliases of ``field_name``."""
    for column in COLUMNS[field_name]:
        value = row.get(column)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _record_from_row(row: dict) -> FeedbackRecord:
    return FeedbackRecord(
        query=_pick(row, "query"),
        proposed_code=_normalize_code(_pick(row, "proposed_code")),
        verdict=_pick(row, "verdict"),
        correct_code=_normalize_code(_pick(row, "correct_code")),
        comment=_pick(row, "comment"),
        status=_pick(row, "status").casefold(),
    )


def _is_sqlite(path: Path) -> bool:
    return path.suffix.casefold() in {".sqlite", ".sqlite3", ".db"}


def load_feedback_sqlite(path: Path) -> list[FeedbackRecord]:
    """Read the feedback store written by ``api/app/services/feedback_store.py``."""
    if not path.is_file():
        raise FileNotFoundError(f"База вердиктов не найдена: {path}")
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                f"SELECT query, predicted_code, verdict, corrected_code "
                f"FROM {SQLITE_FEEDBACK_TABLE}"
            ).fetchall()
        except sqlite3.OperationalError as error:
            raise ValueError(
                f"В {path} нет таблицы {SQLITE_FEEDBACK_TABLE} "
                f"(это база очереди job-ов, а не журнал вердиктов): {error}"
            ) from error
    return [
        FeedbackRecord(
            query=str(row["query"] or "").strip(),
            proposed_code=_normalize_code(str(row["predicted_code"] or "")),
            verdict=str(row["verdict"] or ""),
            correct_code=_normalize_code(str(row["corrected_code"] or "")),
        )
        for row in rows
    ]


def load_feedback(path: Path) -> list[FeedbackRecord]:
    """Read a feedback export: CSV (``;``), JSONL, or the SQLite store."""
    if not path.is_file():
        raise FileNotFoundError(f"Экспорт вердиктов не найден: {path}")
    if _is_sqlite(path):
        return load_feedback_sqlite(path)
    if path.suffix.casefold() == ".jsonl":
        records: list[FeedbackRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(_record_from_row(payload))
        return records
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [_record_from_row(row) for row in csv.DictReader(handle, delimiter=";")]


def usable(records: list[FeedbackRecord]) -> list[FeedbackRecord]:
    """Verdicts that carry training signal (incorrect or out-of-scope)."""
    return [
        record
        for record in records
        if record.query.strip() and (record.is_incorrect or record.is_out_of_scope)
    ]


@dataclass
class MiningReport:
    """What the export contributed and why the rest did not."""

    records: int = 0
    by_verdict: Counter = field(default_factory=Counter)
    usable: int = 0
    pairs: int = 0
    positives: int = 0
    # ``incorrect`` verdicts whose corrected code is not a catalog position.
    unresolved: list[str] = field(default_factory=list)
    # ``incorrect`` verdicts with no corrected code at all.
    without_code: list[str] = field(default_factory=list)
    duplicate_pairs: int = 0
    # Rows for one query that carry the same verdict (two verifiers, re-export).
    duplicate_verdicts: int = 0
    # Rows for one query that disagree: ``incorrect`` beats ``absent``.
    conflicting_verdicts: list[str] = field(default_factory=list)
    # Human-confirmed wrong answers that retrieval no longer ranks in the
    # shortlist; they are added as negatives so the verdict is not lost.
    explicit_negatives: int = 0

    @property
    def hard_negatives(self) -> int:
        return self.pairs - self.positives

    @property
    def skipped_unresolved(self) -> int:
        return len(self.unresolved) + len(self.without_code)

    def summary(self) -> str:
        """Human report: what the export is, and what it bought."""
        verdicts = ", ".join(
            f"{name}: {count}" for name, count in sorted(self.by_verdict.items())
        ) or "нет"
        lines = [
            f"вердиктов в источнике: {self.records} ({verdicts})",
            f"полезных (неверно / не в перечне): {self.usable}",
            f"пар для обучения: {self.pairs} "
            f"(позитивных: {self.positives}, hard negatives: {self.hard_negatives})",
        ]
        if self.skipped_unresolved:
            lines.append(
                f"пропущено вердиктов «неверно» без пригодного кода: "
                f"{self.skipped_unresolved} "
                f"(без кода: {len(self.without_code)}, "
                f"кода нет в каталоге: {len(self.unresolved)})"
            )
            for query in (self.without_code + self.unresolved)[:5]:
                lines.append(f"  - {query[:80]}")
        if self.duplicate_verdicts:
            lines.append(
                f"повторных вердиктов по одному запросу: {self.duplicate_verdicts} "
                f"(оставлен более информативный)"
            )
        if self.conflicting_verdicts:
            lines.append(
                f"противоречивых вердиктов: {len(self.conflicting_verdicts)} "
                f"(«неверно» приоритетнее «не в перечне»)"
            )
        if self.explicit_negatives:
            lines.append(
                f"подтверждённых негативов вне короткого списка: {self.explicit_negatives}"
            )
        if self.duplicate_pairs:
            lines.append(f"свёрнуто повторов (query, title, code): {self.duplicate_pairs}")
        return "\n".join(lines)

    def diagnosis(self) -> str | None:
        """Why nothing was mined, in the terms of the file that was read."""
        if self.pairs:
            return None
        if not self.records:
            return "в источнике нет строк"
        verdicts = {key: count for key, count in self.by_verdict.items() if key != "—"}
        if not verdicts:
            return (
                "в источнике нет колонки вердикта (user_verdict/verdict): "
                "это журнал классификаций, а не разметка. Вердикты ставятся "
                "кнопками в интерфейсе и сохраняются через POST /feedback."
            )
        known = VERDICT_CORRECT | VERDICT_INCORRECT | VERDICT_OUT_OF_SCOPE
        unknown = {key: count for key, count in verdicts.items() if key not in known}
        if unknown:
            shown = ", ".join(f"{key!r} ({count})" for key, count in sorted(unknown.items()))
            return (
                f"неизвестные значения вердикта: {shown}; ожидаются "
                "correct / incorrect / absent (по-русски: верно / неверно / не в перечне)"
            )
        if not self.usable:
            return (
                "все вердикты «верно» — ошибок специалист не нашёл, и сигнала "
                "для обучения в этой выгрузке нет"
            )
        return (
            "все вердикты «неверно» остались без позитива: правленный код пуст "
            "или отсутствует в каталоге, а обучать «всё нерелевантно» нельзя"
        )


def _query_key(query: str) -> str:
    return " ".join((query or "").casefold().split())


def _verdict_rank(record: FeedbackRecord) -> int:
    """How much signal a verdict carries; ties resolve to the harsher one."""
    if record.is_incorrect:
        return 2
    if record.is_out_of_scope:
        return 1
    return 0


def one_verdict_per_query(
    records: list[FeedbackRecord],
) -> tuple[list[FeedbackRecord], int, list[str]]:
    """Collapse repeated verdicts on one query.

    An export is a log: the same query is classified, re-exported and re-verified,
    so it can arrive several times. Two rows may even disagree — a specialist
    marked it ``incorrect`` with a correction while another marked it ``absent``.
    Training on both would label the same title positive and negative, so one
    verdict wins: the correction, because it carries a position to rank first.
    """
    chosen: dict[str, FeedbackRecord] = {}
    order: list[str] = []
    duplicates = 0
    conflicts: list[str] = []
    for record in records:
        key = _query_key(record.query)
        if not key:
            continue
        previous = chosen.get(key)
        if previous is None:
            chosen[key] = record
            order.append(key)
            continue
        duplicates += 1
        if _verdict_rank(record) > _verdict_rank(previous):
            chosen[key] = record
        if _verdict_rank(record) != _verdict_rank(previous):
            # A correction and an «absent» on the same query contradict each
            # other; the order in the file decides nothing but the winner.
            conflicts.append(record.query)
    return [chosen[key] for key in order], duplicates, conflicts


def mine_feedback_pairs(
    records: list[FeedbackRecord],
    catalog: list[dict],
    *,
    dense=None,
    max_per_query: int = 12,
    report: MiningReport | None = None,
) -> list[TrainingPair]:
    """Turn verdicts into training pairs on the serving shortlist.

    Mirrors ``build_pairs``: the query is retrieved against the
    catalog exactly as the API does, and the rows are relabeled
    by the specialist's verdict instead of by construction.

    ``report``, when given, is filled with the counts and the reasons a verdict
    produced no rows (see :class:`MiningReport`).
    """
    from ml.dense import DENSE_SHORTLIST
    from ml.features import lexical_score_prepared, prepare_candidate
    from ml.lexicon import expand_query

    if report is None:
        report = MiningReport()
    report.records += len(records)
    for record in records:
        report.by_verdict[record.normalized_verdict or "—"] += 1
    signal = usable(records)
    report.usable += len(signal)
    signal, duplicates, conflicts = one_verdict_per_query(signal)
    report.duplicate_verdicts += duplicates
    report.conflicting_verdicts.extend(conflicts)

    dense_position = (
        {name: index for index, name in enumerate(dense.names)}
        if dense is not None
        else {}
    )

    def dense_for(query: str):
        if dense is None:
            return {}, [], 0.0

        scores = dense.scores(query)
        order = scores.argsort()[::-1][:DENSE_SHORTLIST]
        top_titles = [dense.names[int(index)] for index in order]
        return (
            {name: float(scores[index]) for name, index in dense_position.items()},
            top_titles,
            float(scores.max()),
        )

    prepared_titles = [
        (item["name"], prepare_candidate(item["name"])) for item in catalog
    ]
    code_by_title = {item["name"]: item.get("pp_code") or "" for item in catalog}

    pairs: list[TrainingPair] = []
    for record in signal:
        query = record.query.strip()
        positive_title = None
        if record.is_incorrect:
            # The corrected position is the positive the model failed to rank
            # first. Without it there is nowhere to put the "+1": labelling the
            # shortlist all-negative would teach the opposite of the verdict.
            if not record.correct_code:
                report.without_code.append(query)
                continue
            positive_title = _title_for_code(record.correct_code, code_by_title)
            if positive_title is None:
                report.unresolved.append(query)
                continue

        prepared_query = prepare_candidate(expand_query(query))
        scored = sorted(
            (
                (lexical_score_prepared(prepared_query, prepared), title)
                for title, prepared in prepared_titles
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        shortlist = scored[:max_per_query]
        dense_scores, dense_titles, dense_top = dense_for(query)
        shortlist += [
            (0.0, title)
            for title in dense_titles
            if title not in {listed for _score, listed in shortlist}
        ]

        shortlist_titles = {title for _score, title in shortlist}
        # The position the model answered with is a human-verified negative even
        # when the shortlist no longer ranks it (retrieval drifts between the
        # answer and the verdict).
        explicit_negative = (
            _title_for_code(record.proposed_code, code_by_title)
            if record.proposed_code and record.proposed_code != record.correct_code
            else None
        )
        for score, title in shortlist:
            label = 1 if positive_title is not None and title == positive_title else 0
            pairs.append(
                TrainingPair(
                    query=query,
                    title=title,
                    code=code_by_title.get(title, ""),
                    label=label,
                    retrieval_score=score,
                    dense=dense_scores.get(title, 0.0),
                    dense_top=dense_top,
                )
            )
        # The corrected position must be trainable even when
        # retrieval missed it (that is often why the answer was wrong).
        if positive_title is not None and positive_title not in shortlist_titles:
            pairs.append(
                TrainingPair(
                    query=query,
                    title=positive_title,
                    code=record.correct_code,
                    label=1,
                    retrieval_score=0.0,
                    dense=0.0,
                    dense_top=0.0,
                )
            )
        if (
            explicit_negative is not None
            and explicit_negative not in shortlist_titles
            and explicit_negative != positive_title
        ):
            pairs.append(
                TrainingPair(
                    query=query,
                    title=explicit_negative,
                    code=code_by_title.get(explicit_negative, ""),
                    label=0,
                    retrieval_score=0.0,
                    dense=0.0,
                    dense_top=0.0,
                )
            )
            report.explicit_negatives += 1
    # The catalog lists one position under several appendix
    # rows; a (query, title, code) pair must be trained once.
    unique: dict[tuple[str, str, str], TrainingPair] = {}
    for pair in pairs:
        unique.setdefault((pair.query, pair.title, pair.code), pair)
    report.duplicate_pairs += len(pairs) - len(unique)
    report.pairs += len(unique)
    report.positives += sum(1 for pair in unique.values() if pair.label == 1)
    return list(unique.values())


def _title_for_code(code: str, code_by_title: dict[str, str]) -> str | None:
    """A catalog title whose code matches (prefix-tolerant)."""
    if not code:
        return None
    for title, title_code in code_by_title.items():
        if title_code == code:
            return title
    for title, title_code in code_by_title.items():
        if title_code.startswith(code + ".") or code.startswith(title_code + "."):
            return title
    return None


def main() -> int:
    """CLI: inspect what an export/store would contribute to training."""
    from ml.catalog_loader import load_catalog

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "feedback",
        type=Path,
        nargs="?",
        help="экспорт вердиктов: .csv (UI-выгрузка или журнал feedback), .jsonl, .sqlite",
    )
    parser.add_argument(
        "--feedback-sqlite",
        type=Path,
        default=None,
        help="база classification_feedback (api/app/services/feedback_store.py)",
    )
    parser.add_argument("--max-per-query", type=int, default=12)
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=0,
        help="вернуть код 1, если пар меньше (DoD ML-4: 300)",
    )
    args = parser.parse_args()
    if not args.feedback and not args.feedback_sqlite:
        parser.error("нужен экспорт (позиционный аргумент) или --feedback-sqlite")

    sources: list[Path] = []
    if args.feedback:
        sources.append(args.feedback)
    if args.feedback_sqlite:
        sources.append(args.feedback_sqlite)
    records: list[FeedbackRecord] = []
    for source in sources:
        records += load_feedback(source)

    report = MiningReport()
    pairs = mine_feedback_pairs(
        records,
        load_catalog(),
        max_per_query=args.max_per_query,
        report=report,
    )
    print(report.summary())
    pairs = len(pairs)
    diagnosis = report.diagnosis()
    if diagnosis:
        print(f"\nпочему пар нет: {diagnosis}")
    if args.min_pairs and pairs < args.min_pairs:
        print(
            f"\nDoD ML-4 не выполнен: {pairs} пар против {args.min_pairs} требуемых "
            f"из реального трафика"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
