import pytest

from eval.acceptance_eval import load_cases, score, summarize


def case(expected="26.20.11"):
    return {
        "expected": expected,
        "accepted": [expected] if expected else [],
        "negative": not expected,
    }


def response(code, alternatives=(), status="review"):
    return {
        "code": code,
        "status": status,
        "confidence": 0.99,
        "alternatives": [{"code": c} for c in alternatives],
    }


def test_parent_match_is_diagnostic_not_exact_credit():
    result = score(case("26.20.11.110"), response("26.20.11"))
    assert result["hierarchical_top1"]
    assert not result["exact_top1"]


def test_top3_does_not_include_fourth_distinct_candidate():
    result = score(case(), response("26.20.12", ["26.20.13", "26.20.14", "26.20.11"]))
    assert not result["exact_top3"]
    result = score(case(), response("26.20.12", ["26.20.12", "26.20.11"]))
    assert result["exact_top3"]


def test_confidence_and_review_status_do_not_define_correctness():
    right = score(case(), response("26.20.11"))
    wrong = score(case(), response("26.20.12", status="ok"))
    assert right["exact_top1"] and not wrong["exact_top1"]


def test_not_found_and_errors_stay_in_denominator():
    right = score(case(""), response(None, status="not_found"))
    error = {
        **case(""),
        "exact_top1": False,
        "exact_top3": False,
        "hierarchical_top1": False,
        "error": "timeout",
    }
    summary = summarize([right, error])
    assert summary["exact_top1_all_pct"] == 50
    assert summary["errors"] == 1
    assert summary["exact_top1_coded_pct"] is None
    with pytest.raises(ValueError):
        score(case(""), response("26.20.11", status="not_found"))


def test_load_user_csv_bom_and_negative_marker(tmp_path):
    path = tmp_path / "test.csv"
    path.write_text(
        "product_name,true_okpd2,category,comment\n"
        "Ноутбук,26.20.11,Простой,код\n"
        "Ремонт,НЕТ В ПЕРЕЧНЕ,Вне перечня,услуга\n",
        encoding="utf-8-sig",
    )
    rows = load_cases(path)
    assert len(rows) == 2 and rows[1]["negative"]


def test_load_semicolon_csv_and_explicit_accepted_codes(tmp_path):
    path = tmp_path / "test.csv"
    path.write_text("query;expected_okpd2;accepted_okpd2\nТелевизор;26.40.20;26.40.20.122\n")
    row = load_cases(path)[0]
    assert score(row, response("26.40.20.122"))["exact_top1"]


def test_reject_empty_or_malformed_dataset(tmp_path):
    path = tmp_path / "test.csv"
    for text in [
        "query;expected_okpd2\n",
        "query;expected_okpd2\nТовар;неизвестно\n",
        "query;expected_okpd2\n;26.20.11\n",
    ]:
        path.write_text(text)
        with pytest.raises(ValueError):
            load_cases(path)
