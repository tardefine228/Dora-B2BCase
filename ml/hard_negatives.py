"""Hard-negative mining from specialist feedback (ML-4).

The synthetic training platform cannot know the queries real
procurement specialists type. When the feedback API (задача
BE-2: ``POST /classify/feedback``) accumulates verdicts, the
*incorrect* ones are the most valuable training signal there
is: a query the model answered wrongly, with the code a
specialist corrected it to.

This module turns an export of the feedback log into
``TrainingPair`` rows:

* verdict ``неверно`` with a correct code — the corrected
  position becomes a **positive**, the wrongly proposed one a
  **hard negative** (both are retrieved by the same lexical
  shortlist the API uses, so the pair features match serving);
* verdict ``не в перечне`` — every retrieved candidate is a
  **negative** (the correct behaviour is silence);
* verdict ``верно`` — no signal (the model was right).

Input formats (BE-2 export, CSV with ``;`` or JSONL):

    query;proposed_code;verdict;correct_code;comment
    Счетчик воды крыльчатый;28.13.14;неверно;28.13.11;...

Verdicts are matched case-insensitively; synonyms accepted:
``верно|correct|ok``, ``неверно|incorrect|wrong|не верно``,
``не в перечне|not_found|not_in_list|вне перечня``.

Usage as a library (training picks it up automatically):

    python -m ml.train_reranker --output models/reranker.json \
        --feedback feedback_export.csv

Status: mechanism ready and tested on synthetic exports; the
>=300-pair DoD needs real traffic, which requires BE-2 (the
feedback endpoint and the ``classification_log`` columns) and
DA-1 (the real labeled set) to be live first.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from ml.train_reranker import TrainingPair

# Verdict spellings accepted in the export.
VERDICT_CORRECT = {"верно", "correct", "ok", "верный", "правильно"}
VERDICT_INCORRECT = {
    "неверно",
    "incorrect",
    "wrong",
    "не верно",
    "неправильно",
}
VERDICT_OUT_OF_SCOPE = {
    "не в перечне",
    "not_found",
    "not_in_list",
    "вне перечня",
    "вне перечня.",
}


@dataclass(frozen=True)
class FeedbackRecord:
    """One specialist verdict on a served answer."""

    query: str
    proposed_code: str = ""
    verdict: str = ""
    correct_code: str = ""
    comment: str = ""

    @property
    def normalized_verdict(self) -> str:
        return (self.verdict or "").strip().casefold()

    @property
    def is_incorrect(self) -> bool:
        return self.normalized_verdict in VERDICT_INCORRECT

    @property
    def is_out_of_scope(self) -> bool:
        return self.normalized_verdict in VERDICT_OUT_OF_SCOPE

    @property
    def is_correct(self) -> bool:
        return self.normalized_verdict in VERDICT_CORRECT


def _normalize_code(code: str) -> str:
    return (code or "").strip().replace(" ", "").replace(",", ".")


def load_feedback(path: Path) -> list[FeedbackRecord]:
    """Read a feedback export (CSV ``;``-delimited or JSONL)."""
    if not path.is_file():
        raise FileNotFoundError(f"Экспорт вердиктов не найден: {path}")
    records: list[FeedbackRecord] = []
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(
                FeedbackRecord(
                    query=str(payload.get("query", "")),
                    proposed_code=str(payload.get("proposed_code", "")),
                    verdict=str(payload.get("verdict", "")),
                    correct_code=str(payload.get("correct_code", "")),
                    comment=str(payload.get("comment", "")),
                )
            )
        return records
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            records.append(
                FeedbackRecord(
                    query=(row.get("query") or "").strip(),
                    proposed_code=_normalize_code(row.get("proposed_code")),
                    verdict=(row.get("verdict") or "").strip(),
                    correct_code=_normalize_code(row.get("correct_code")),
                    comment=(row.get("comment") or "").strip(),
                )
            )
    return records


def usable(records: list[FeedbackRecord]) -> list[FeedbackRecord]:
    """Verdicts that carry training signal (incorrect or out-of-scope)."""
    return [
        record
        for record in records
        if record.query.strip() and (record.is_incorrect or record.is_out_of_scope)
    ]


def mine_feedback_pairs(
    records: list[FeedbackRecord],
    catalog: list[dict],
    *,
    dense=None,
    max_per_query: int = 12,
) -> list[TrainingPair]:
    """Turn verdicts into training pairs on the serving shortlist.

    Mirrors ``build_pairs``: the query is retrieved against the
    catalog exactly as the API does, and the rows are relabeled
    by the specialist's verdict instead of by construction.
    """
    from ml.dense import DENSE_SHORTLIST
    from ml.features import lexical_score_prepared, prepare_candidate
    from ml.lexicon import expand_query

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
    for record in usable(records):
        query = record.query.strip()
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

        positive_title = _title_for_code(record.correct_code, code_by_title)
        shortlist_titles = {title for _score, title in shortlist}
        for score, title in shortlist:
            if record.is_incorrect and title == positive_title:
                label = 1
            else:
                label = 0
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
        if (
            record.is_incorrect
            and positive_title
            and positive_title not in shortlist_titles
        ):
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
    # The catalog lists one position under several appendix
    # rows; a (query, title, code) pair must be trained once.
    unique: dict[tuple[str, str, str], TrainingPair] = {}
    for pair in pairs:
        unique.setdefault(
            (pair.query, pair.title, pair.code), pair
        )
    return list(unique.values())


def _title_for_code(code: str, code_by_title: dict[str, str]) -> str | None:
    """A catalog title whose code matches (prefix-tolerant)."""
    if not code:
        return None
    for title, title_code in code_by_title.items():
        if title_code == code:
            return title
    for title, title_code in code_by_title.items():
        if (
            title_code.startswith(code + ".")
            or code.startswith(title_code + ".")
        ):
            return title
    return None


def main() -> int:
    """CLI: inspect what an export would contribute to training."""
    import argparse

    from ml.catalog_loader import load_catalog

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feedback", type=Path)
    parser.add_argument("--max-per-query", type=int, default=12)
    args = parser.parse_args()

    records = load_feedback(args.feedback)
    usable_records = usable(records)
    print(
        f"вердиктов: {len(records)}, полезных: {len(usable_records)} "
        f"(неверно: {sum(1 for r in usable_records if r.is_incorrect)}, "
        f"вне перечня: {sum(1 for r in usable_records if r.is_out_of_scope)})"
    )
    pairs = mine_feedback_pairs(usable_records, load_catalog())
    positives = sum(1 for pair in pairs if pair.label == 1)
    print(
        f"пар для обучения: {len(pairs)} "
        f"(позитивных: {positives}, hard negatives: {len(pairs) - positives})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
