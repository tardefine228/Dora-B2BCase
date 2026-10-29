"""Hard-negative mining from specialist verdicts (ML-4).

What is pinned here is the part that decides whether 300 real verdicts turn into
300 usable rows: the **schemas** the product actually writes (the UI export and
the feedback store), verdict spellings (Russian and English, ``absent``
included), pair construction on the serving shortlist, the guards that refuse a
poisoned label, and that ``train`` picks the same reader up.

The ``>=300``-pair DoD still needs real traffic; these tests make sure the
mechanism reads it when it arrives instead of silently importing zeros.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from pathlib import Path

from ml.hard_negatives import (
    FeedbackRecord,
    MiningReport,
    load_feedback,
    load_feedback_sqlite,
    mine_feedback_pairs,
    usable,
)
from ml.train_reranker import train

# The columns the UI writes into a downloaded file (frontend/src/files.ts).
UI_EXPORT_COLUMNS = [
    "query",
    "code",
    "title",
    "status",
    "confidence",
    "confidence_raw",
    "clause_ref",
    "clause_url",
    "user_verdict",
    "corrected_code",
    "verifier",
    "verifier_note",
    "row",
]

# The columns the feedback store keeps (api/app/services/feedback_store.py).
STORE_COLUMNS = ["query", "predicted_code", "verdict", "corrected_code"]


def _csv(columns: list[str], rows: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, delimiter=";")
    writer.writeheader()
    for row in rows:
        writer.writerow({**{column: "" for column in columns}, **row})
    return buffer.getvalue()


def _ui_export(rows: list[dict]) -> str:
    return _csv(UI_EXPORT_COLUMNS, rows)


def _store_export(rows: list[dict]) -> str:
    return _csv(STORE_COLUMNS, rows)


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_load_feedback_reads_the_ui_export(tmp_path: Path) -> None:
    """A downloaded file carries ``code``/``user_verdict``/``corrected_code``."""
    path = _write(
        tmp_path,
        "classification-results.csv",
        _ui_export(
            [
                {
                    "query": "Счетчик воды крыльчатый",
                    "code": "28.13.14",
                    "status": "ok",
                    "user_verdict": "incorrect",
                    "corrected_code": "28.13.12",
                }
            ]
        ),
    )
    records = load_feedback(path)
    assert len(records) == 1
    assert records[0].query == "Счетчик воды крыльчатый"
    assert records[0].proposed_code == "28.13.14"
    assert records[0].correct_code == "28.13.12"
    assert records[0].is_incorrect
    assert records[0].status == "ok"


def test_load_feedback_reads_the_store_schema(tmp_path: Path) -> None:
    """The store writes ``predicted_code`` and the English API verdicts."""
    path = _write(
        tmp_path,
        "feedback.csv",
        _store_export(
            [
                {
                    "query": "Газонокосилка Bosch",
                    "predicted_code": "26.20.11",
                    "verdict": "absent",
                },
                {
                    "query": "Счетчик воды крыльчатый",
                    "predicted_code": "28.13.14",
                    "verdict": "incorrect",
                    "corrected_code": "28.13.12",
                },
            ]
        ),
    )
    records = load_feedback(path)
    assert [record.proposed_code for record in records] == ["26.20.11", "28.13.14"]
    assert records[0].is_out_of_scope
    assert records[1].is_incorrect
    assert records[1].correct_code == "28.13.12"


def test_load_feedback_reads_the_sqlite_store(tmp_path: Path) -> None:
    """``--feedback-sqlite`` reads the same table the API writes."""
    path = tmp_path / "classification_feedback.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE classification_feedback (
              id TEXT PRIMARY KEY,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              query TEXT NOT NULL, predicted_code TEXT,
              verdict TEXT NOT NULL, corrected_code TEXT
            )"""
        )
        connection.execute(
            "INSERT INTO classification_feedback (id, query, predicted_code, verdict,"
            " corrected_code) VALUES (?, ?, ?, ?, ?)",
            ("1", "Судно прогулочное", "26.20.11", "absent", None),
        )
        connection.execute(
            "INSERT INTO classification_feedback (id, query, predicted_code, verdict,"
            " corrected_code) VALUES (?, ?, ?, ?, ?)",
            ("2", "Счетчик воды крыльчатый", "28.13.14", "incorrect", "28.13.12"),
        )
    records = load_feedback_sqlite(path)
    assert [record.verdict for record in records] == ["absent", "incorrect"]
    assert records[1].correct_code == "28.13.12"
    # ``load_feedback`` dispatches on the suffix, so training accepts a store path.
    assert load_feedback(path) == records


def test_load_feedback_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "feedback.jsonl"
    path.write_text(
        json.dumps(
            {
                "query": "Судно прогулочное",
                "predicted_code": "26.20.11",
                "verdict": "absent",
                "corrected_code": "",
                "comment": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records = load_feedback(path)
    assert len(records) == 1
    assert records[0].is_out_of_scope


def test_verdict_synonyms() -> None:
    assert FeedbackRecord("q", verdict="Incorrect").is_incorrect
    assert FeedbackRecord("q", verdict="не верно").is_incorrect
    assert FeedbackRecord("q", verdict="NOT_IN_LIST").is_out_of_scope
    assert FeedbackRecord("q", verdict="not in list").is_out_of_scope
    assert FeedbackRecord("q", verdict="absent").is_out_of_scope
    assert FeedbackRecord("q", verdict="вне перечня.").is_out_of_scope
    assert FeedbackRecord("q", verdict="OK").is_correct
    assert FeedbackRecord("q", verdict="correct").is_correct
    assert FeedbackRecord("q", verdict="").normalized_verdict == ""


def test_usable_filters_correct_verdicts() -> None:
    records = [
        FeedbackRecord("a", verdict="неверно", correct_code="1.1"),
        FeedbackRecord("b", verdict="верно"),
        FeedbackRecord("c", verdict="не в перечне"),
        FeedbackRecord("", verdict="неверно"),
    ]
    kept = usable(records)
    assert {record.query for record in kept} == {"a", "c"}


def _positions() -> tuple[dict, dict, list[dict]]:
    from ml.catalog_loader import load_catalog

    catalog = load_catalog()
    positive = next(item for item in catalog if item["pp_code"] == "28.13.12")
    negative = next(item for item in catalog if item["pp_code"] == "28.13.14")
    return positive, negative, catalog


def test_mine_pairs_labels_correction_positive_and_served_negative() -> None:
    positive, negative, catalog = _positions()
    records = [
        FeedbackRecord(
            query=positive["name"],
            proposed_code=negative["pp_code"],
            verdict="неверно",
            correct_code=positive["pp_code"],
        )
    ]
    report = MiningReport()
    pairs = mine_feedback_pairs(records, catalog, report=report)
    assert pairs, "короткий список не должен быть пуст"
    positive_pairs = [pair for pair in pairs if pair.label == 1]
    assert len(positive_pairs) == 1
    assert positive_pairs[0].title == positive["name"]
    assert positive_pairs[0].query == positive["name"]
    # The served wrong answer is in the shortlist as a hard negative.
    assert any(pair.title == negative["name"] and pair.label == 0 for pair in pairs)
    assert report.pairs == len(pairs)
    assert report.positives == 1
    assert report.hard_negatives > 0


def test_ui_export_mines_the_same_pairs(tmp_path: Path) -> None:
    """A real UI export (not a hand-made fixture) reaches the trainer."""
    positive, negative, catalog = _positions()
    path = _write(
        tmp_path,
        "classification-results (6).csv",
        _ui_export(
            [
                {
                    "query": positive["name"],
                    "code": negative["pp_code"],
                    "status": "ok",
                    "user_verdict": "incorrect",
                    "corrected_code": positive["pp_code"],
                },
                {
                    "query": "Газонокосилка Bosch",
                    "code": "26.20.11",
                    "status": "ok",
                    "user_verdict": "absent",
                },
                {
                    "query": "Ноутбук",  # no signal: the model was right
                    "code": "26.20.11",
                    "status": "ok",
                    "user_verdict": "correct",
                },
            ]
        ),
    )
    report = MiningReport()
    pairs = mine_feedback_pairs(load_feedback(path), catalog, report=report)
    assert report.records == 3
    assert report.usable == 2
    assert [pair.title for pair in pairs if pair.label == 1] == [positive["name"]]
    assert any(
        pair.query == "Газонокосилка Bosch" and pair.label == 0 for pair in pairs
    )


def test_incorrect_without_a_corrected_code_is_skipped_not_all_negative() -> None:
    """Labelling the whole shortlist negative would teach the opposite."""
    _, _, catalog = _positions()
    records = [
        FeedbackRecord(query="Счетчик воды крыльчатый", proposed_code="28.13.14",
                       verdict="incorrect", correct_code="")
    ]
    report = MiningReport()
    pairs = mine_feedback_pairs(records, catalog, report=report)
    assert pairs == []
    assert report.usable == 1
    assert report.without_code == ["Счетчик воды крыльчатый"]
    assert "без позитива" in (report.diagnosis() or "")


def test_corrected_code_outside_the_catalog_is_skipped_and_reported() -> None:
    _, _, catalog = _positions()
    records = [
        FeedbackRecord(query="Товар вне перечня", proposed_code="26.20.11",
                       verdict="incorrect", correct_code="99.99.99")
    ]
    report = MiningReport()
    pairs = mine_feedback_pairs(records, catalog, report=report)
    assert pairs == []
    assert report.unresolved == ["Товар вне перечня"]


def test_an_export_without_verdicts_mines_nothing_and_says_why(tmp_path: Path) -> None:
    """The normal state of a fresh export: it is a log, not a label set."""
    path = _write(
        tmp_path,
        "log.csv",
        _ui_export([{"query": "Ноутбук", "code": "26.20.11", "status": "ok"}]),
    )
    report = MiningReport()
    pairs = mine_feedback_pairs(load_feedback(path), _positions()[2], report=report)
    assert pairs == []
    assert report.records == 1
    diagnosis = report.diagnosis() or ""
    assert "журнал классификаций" in diagnosis


def test_mine_pairs_all_negative_for_out_of_scope() -> None:
    catalog = _positions()[2]
    records = [FeedbackRecord(query="Газонокосилка Bosch", verdict="не в перечне")]
    pairs = mine_feedback_pairs(records, catalog)
    assert pairs
    assert all(pair.label == 0 for pair in pairs)


def test_train_accepts_feedback_export(tmp_path: Path) -> None:
    positive, _, _ = _positions()
    path = _write(
        tmp_path,
        "feedback.csv",
        _ui_export(
            [
                {
                    "query": positive["name"],
                    "code": "28.13.14",
                    "status": "ok",
                    "user_verdict": "incorrect",
                    "corrected_code": "28.13.12",
                }
            ]
        ),
    )
    output = tmp_path / "reranker.json"
    reranker = train(output, max_queries=5, feedback_path=path)
    assert output.is_file()
    assert reranker.meta["feedback_pairs"] > 0


def test_every_api_verdict_literal_carries_the_expected_signal() -> None:
    """The API's verdict vocabulary must be the loader's vocabulary.

    `POST /feedback` accepts exactly three literals. If one of them is missing
    from the verdict sets, its rows are silently dropped at training time — the
    export would look complete while a whole class of verdicts never became a
    training pair. This test is the contract between `FeedbackRequest` and the
    miner.
    """
    from typing import get_args

    from app.schemas import FeedbackRequest

    from ml.hard_negatives import (
        VERDICT_CORRECT,
        VERDICT_INCORRECT,
        VERDICT_OUT_OF_SCOPE,
        FeedbackRecord,
    )

    verdict_field = FeedbackRequest.model_fields["verdict"]
    literals = set(get_args(verdict_field.annotation))
    assert literals == {"correct", "incorrect", "absent"}

    for literal in literals:
        record = FeedbackRecord(query="Мышь компьютерная", verdict=literal)
        buckets = {
            "correct": record.is_correct,
            "incorrect": record.is_incorrect,
            "absent": record.is_out_of_scope,
        }
        assert buckets[literal], (
            f"вердикт {literal!r} не распознан: он попадёт в выгрузку, "
            "но не станет обучающей парой"
        )
        assert sum(int(value) for value in buckets.values()) == 1, literal

    # The three sets must stay disjoint: one verdict cannot mean two things.
    assert not (VERDICT_CORRECT & VERDICT_INCORRECT)
    assert not (VERDICT_CORRECT & VERDICT_OUT_OF_SCOPE)
    assert not (VERDICT_INCORRECT & VERDICT_OUT_OF_SCOPE)


def test_one_verdict_per_query_prefers_the_correction() -> None:
    """Two verifiers disagree: the correction wins, the query is not all-negative."""
    from ml.hard_negatives import one_verdict_per_query

    positive, negative, catalog = _positions()
    records = [
        FeedbackRecord(query=positive["name"], proposed_code=negative["pp_code"],
                       verdict="absent"),
        FeedbackRecord(query=positive["name"], proposed_code=negative["pp_code"],
                       verdict="incorrect", correct_code=positive["pp_code"]),
    ]
    kept, duplicates, conflicts = one_verdict_per_query(records)
    assert len(kept) == 1
    assert kept[0].is_incorrect
    assert (duplicates, conflicts) == (1, [positive["name"]])

    report = MiningReport()
    pairs = mine_feedback_pairs(records, catalog, report=report)
    assert [pair.title for pair in pairs if pair.label == 1] == [positive["name"]]
    assert report.duplicate_verdicts == 1
    assert report.conflicting_verdicts == [positive["name"]]
    # The pair count is the one of the verdict that won, not of both.
    single = mine_feedback_pairs([records[1]], catalog)
    assert len(pairs) == len(single)


def test_a_repeated_verdict_is_mined_once() -> None:
    """A re-export is a log: the same verdict must not double the weights."""
    positive, negative, catalog = _positions()
    record = FeedbackRecord(query=positive["name"], proposed_code=negative["pp_code"],
                            verdict="неверно", correct_code=positive["pp_code"])
    report = MiningReport()
    pairs = mine_feedback_pairs([record, record], catalog, report=report)
    assert report.records == 2
    assert report.duplicate_verdicts == 1
    assert report.conflicting_verdicts == []
    assert len(pairs) == len(mine_feedback_pairs([record], catalog))


def test_the_served_answer_is_a_negative_outside_the_shortlist() -> None:
    """A verified wrong answer stays a hard negative even if retrieval drifted."""
    _, negative, catalog = _positions()
    report = MiningReport()
    pairs = mine_feedback_pairs(
        [FeedbackRecord(query="Газонокосилка Bosch", proposed_code=negative["pp_code"],
                        verdict="absent")],
        catalog,
        max_per_query=1,
        report=report,
    )
    added = [pair for pair in pairs if pair.title == negative["name"]]
    assert len(added) == 1
    assert added[0].label == 0
    assert report.explicit_negatives == 1


def test_cli_names_the_shortfall_against_the_dod(tmp_path: Path, monkeypatch, capsys) -> None:
    from ml.hard_negatives import main

    positive, negative, _catalog = _positions()
    path = _write(
        tmp_path,
        "classification-results.csv",
        _ui_export(
            [
                {
                    "query": positive["name"],
                    "code": negative["pp_code"],
                    "status": "ok",
                    "user_verdict": "incorrect",
                    "corrected_code": positive["pp_code"],
                }
            ]
        ),
    )
    monkeypatch.setattr("sys.argv", ["hard_negatives", str(path), "--min-pairs", "300"])
    assert main() == 1
    out = capsys.readouterr().out
    assert "DoD ML-4 не выполнен" in out
    assert "позитивных: 1" in out


def test_a_conflict_is_reported_in_either_order() -> None:
    from ml.hard_negatives import one_verdict_per_query

    absent = FeedbackRecord("Счетчик воды крыльчатый", verdict="absent")
    correction = FeedbackRecord(
        "Счетчик воды крыльчатый", verdict="incorrect", correct_code="28.13.12"
    )
    for first, second in ((absent, correction), (correction, absent)):
        kept, _duplicates, conflicts = one_verdict_per_query([first, second])
        assert kept == [correction]
        assert conflicts == ["Счетчик воды крыльчатый"]
