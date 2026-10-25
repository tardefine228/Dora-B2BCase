"""Hard-negative mining from specialist verdicts (ML-4).

The mechanism is tested on synthetic exports: the
>=300-pair DoD needs real traffic, which requires the
feedback API (BE-2) and the real labeled set (DA-1)
first. What is pinned here: verdict parsing, pair
construction on the serving shortlist, and that
``train`` accepts a feedback export.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ml.hard_negatives import (
    FeedbackRecord,
    load_feedback,
    mine_feedback_pairs,
    usable,
)
from ml.train_reranker import train


def _feedback_csv(rows: list[dict]) -> str:
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["query", "proposed_code", "verdict", "correct_code", "comment"],
        delimiter=";",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def test_load_feedback_csv(tmp_path: Path) -> None:
    path = tmp_path / "feedback.csv"
    path.write_text(
        _feedback_csv(
            [
                {
                    "query": "Счетчик воды крыльчатый",
                    "proposed_code": "28.13.14",
                    "verdict": "неверно",
                    "correct_code": "28.13.12",
                    "comment": "крыльчатый — центробежный",
                }
            ]
        ),
        encoding="utf-8",
    )
    records = load_feedback(path)
    assert len(records) == 1
    assert records[0].query == "Счетчик воды крыльчатый"
    assert records[0].is_incorrect
    assert not records[0].is_out_of_scope


def test_load_feedback_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "feedback.jsonl"
    path.write_text(
        json.dumps(
            {
                "query": "Судно прогулочное",
                "proposed_code": "26.20.11",
                "verdict": "не в перечне",
                "correct_code": "",
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
    assert FeedbackRecord("q", verdict="NOT_IN_LIST").normalized_verdict == (
        "not_in_list"
    )
    assert FeedbackRecord("q", verdict="not_found").is_out_of_scope
    assert FeedbackRecord("q", verdict="OK").is_correct


def test_usable_filters_correct_verdicts() -> None:
    records = [
        FeedbackRecord("a", verdict="неверно", correct_code="1.1"),
        FeedbackRecord("b", verdict="верно"),
        FeedbackRecord("c", verdict="не в перечне"),
        FeedbackRecord("", verdict="неверно"),
    ]
    kept = usable(records)
    assert {record.query for record in kept} == {"a", "c"}


def test_mine_pairs_labels_correction_positive_and_served_negative() -> None:
    from ml.catalog_loader import load_catalog

    catalog = load_catalog()
    # Two neighbouring real positions (объёмные vs центробежные
    # насосы), so the mined pairs reference actual titles and
    # both land in the same lexical shortlist.
    positive = next(item for item in catalog if item["pp_code"] == "28.13.12")
    negative = next(item for item in catalog if item["pp_code"] == "28.13.14")
    records = [
        FeedbackRecord(
            query=positive["name"],
            proposed_code=negative["pp_code"],
            verdict="неверно",
            correct_code=positive["pp_code"],
        )
    ]
    pairs = mine_feedback_pairs(records, catalog)
    assert pairs, "короткий список не должен быть пуст"
    positive_pairs = [pair for pair in pairs if pair.label == 1]
    assert len(positive_pairs) == 1
    assert positive_pairs[0].title == positive["name"]
    assert positive_pairs[0].query == positive["name"]
    # The served wrong answer is in the shortlist as a negative.
    assert any(
        pair.title == negative["name"] and pair.label == 0 for pair in pairs
    )


def test_mine_pairs_all_negative_for_out_of_scope() -> None:
    from ml.catalog_loader import load_catalog

    catalog = load_catalog()
    records = [
        FeedbackRecord(query="Газонокосилка Bosch", verdict="не в перечне")
    ]
    pairs = mine_feedback_pairs(records, catalog)
    assert pairs
    assert all(pair.label == 0 for pair in pairs)


def test_train_accepts_feedback_export(tmp_path: Path) -> None:
    from ml.catalog_loader import load_catalog

    catalog = load_catalog()
    positive = next(item for item in catalog if item["pp_code"] == "28.13.12")
    path = tmp_path / "feedback.csv"
    path.write_text(
        _feedback_csv(
            [
                {
                    "query": positive["name"],
                    "proposed_code": "28.13.14",
                    "verdict": "неверно",
                    "correct_code": "28.13.12",
                    "comment": "",
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "reranker.json"
    reranker = train(output, max_queries=5, feedback_path=path)
    assert output.is_file()
    assert reranker.meta["feedback_pairs"] > 0
