"""One deterministic decision table for the whole procurement report.

The UI and both export paths consume this service's output. They must not invent
their own buckets from confidence or treat a specialist's review flag as an
automatic acceptance. Reason counts may overlap; primary buckets never do.
"""

from collections import Counter

from app.schemas import (
    ClassifyResponse,
    ProcurementReport,
    ProcurementReportItem,
    ProcurementSummary,
    ReportReason,
    ReportReasonCode,
)

_MESSAGES: dict[ReportReasonCode, str] = {
    "multiple_codes": "Одно название встречается в перечне с разными кодами ОКПД2.",
    "code_exception": "Позиция является исключением или роль кода не подтверждена.",
    "ambiguous_name": "Общее название допускает несколько позиций; нужен выбор специалиста.",
    "catalog_review_required": "Строку каталога нужно сверить с первоисточником.",
    "low_confidence": "Модель нашла кандидата, но уверенность ниже порога автоматического ответа.",
    "code_missing": "Код ОКПД2 не назначен.",
    "not_found": "Надёжное совпадение не найдено; проверьте название вручную.",
}

# Higher rank means earlier manual attention. Ties are resolved by stable item_id.
_RISK: dict[ReportReasonCode, int] = {
    "multiple_codes": 100,
    "code_exception": 95,
    "ambiguous_name": 90,
    "catalog_review_required": 85,
    "code_missing": 80,
    "low_confidence": 70,
    "not_found": 60,
}


def _reason(code: ReportReasonCode) -> ReportReason:
    return ReportReason(code=code, message=_MESSAGES[code])


def classify_item(result: ClassifyResponse, index: int) -> ProcurementReportItem:
    reasons: list[ReportReason] = []
    codes = {reason.code for reason in result.attention_reasons}
    if result.status == "not_found":
        reasons.append(_reason("not_found"))
    else:
        for code in (
            "multiple_codes", "code_exception", "ambiguous_name", "catalog_review_required"
        ):
            if code in codes:
                reasons.append(_reason(code))
        if (
            result.status in {"ok", "review"}
            and result.code_role != "Включён"
            and "code_exception" not in codes
        ):
            reasons.append(_reason("code_exception"))
        if result.catalog_review_required is True and "catalog_review_required" not in codes:
            reasons.append(_reason("catalog_review_required"))
        if result.status == "needs_clarification" and not reasons:
            reasons.append(_reason("ambiguous_name"))
        if (
            result.status == "review"
            and result.decision_source != "specialist"
            and "low_confidence" not in codes
        ):
            reasons.append(_reason("low_confidence"))
        if result.status in {"ok", "review"} and not result.code:
            reasons.append(_reason("code_missing"))

    if result.status == "not_found":
        bucket = "not_found"
    elif result.status in {"needs_clarification", "review"} or reasons:
        bucket = "needs_review"
    else:
        bucket = "accepted"
    risk_rank = max((_RISK[reason.code] for reason in reasons), default=0)
    return ProcurementReportItem(
        item_id=result.item_id if result.item_id is not None else index,
        bucket=bucket,
        risk_rank=risk_rank,
        reasons=reasons,
    )


def build_procurement_report(results: list[ClassifyResponse]) -> ProcurementReport:
    items = [classify_item(result, index) for index, result in enumerate(results)]
    counts = Counter(item.bucket for item in items)
    reason_counts = Counter(
        reason.code for item in items for reason in item.reasons
    )
    auto_accepted = sum(
        item.bucket == "accepted" and result.decision_source != "specialist"
        for item, result in zip(items, results, strict=True)
    )
    specialist_resolved = sum(
        item.bucket == "accepted" and result.decision_source == "specialist"
        for item, result in zip(items, results, strict=True)
    )
    summary = ProcurementSummary(
        total=len(results),
        accepted=counts["accepted"],
        auto_accepted=auto_accepted,
        specialist_resolved=specialist_resolved,
        needs_review=counts["needs_review"],
        not_found=counts["not_found"],
        reason_counts=dict(reason_counts),
    )
    # This is a contract invariant, not a formatting assertion in the UI.
    assert summary.accepted + summary.needs_review + summary.not_found == summary.total
    assert summary.auto_accepted + summary.specialist_resolved == summary.accepted
    return ProcurementReport(summary=summary, items=items)


def reason_message(code: ReportReasonCode) -> str:
    return _MESSAGES[code]
