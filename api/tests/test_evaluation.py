from types import SimpleNamespace

from app.schemas import Alternative, ClassifyResponse

from eval.run_eval import _classify_rows


def test_top3_does_not_count_the_fourth_displayed_code():
    response = ClassifyResponse(
        query="Товар",
        code="01.1",
        confidence=0.5,
        status="review",
        latency_ms=0,
        alternatives=[Alternative(code=code, confidence=0.5) for code in ("02.1", "03.1", "04.1")],
    )
    classifier = SimpleNamespace(classify=lambda query: response)
    rows = _classify_rows(classifier, [{"query": "Товар", "expected_okpd2": "04.1"}])
    assert not rows[0]["correct_top3"]
    rows = _classify_rows(classifier, [{"query": "Товар", "expected_okpd2": "03.1"}])
    assert rows[0]["correct_top3"]
