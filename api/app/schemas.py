"""Public request/response contracts; exposed automatically in OpenAPI.

The classification result format is the team-agreed contract v2.0 (see
``docs/api-contract.md``): new optional fields may be added, existing fields are
never renamed or removed within a major version.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

CONTRACT_VERSION = "2.0"


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


ProductName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
]


class ClassifyRequest(APIModel):
    name: ProductName = Field(
        description="Название товара или код ОКПД2; от 1 до 500 символов.",
        examples=["Ноутбук Lenovo IdeaPad"],
    )



class ResolveClarificationRequest(APIModel):
    option_id: str = Field(min_length=1, max_length=160)
    catalog_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")


class ResolveSingleClarificationRequest(ResolveClarificationRequest):
    query: ProductName


class FeedbackRequest(APIModel):
    query: ProductName
    predicted_code: str | None = Field(default=None, max_length=32)
    verdict: Literal["correct", "incorrect", "absent"]
    corrected_code: str | None = Field(default=None, max_length=32)
    comment: str | None = Field(
        default=None,
        max_length=1000,
        description=(
            "Свободный комментарий специалиста. Уходит в выгрузку вердиктов "
            "(GET /admin/feedback/export) вместе с остальными полями."
        ),
    )


class FeedbackSaved(APIModel):
    id: str
    saved: bool = True

class BatchClassifyRequest(APIModel):
    items: list[ProductName] = Field(
        min_length=1,
        max_length=100,
        description="От 1 до 100 названий товаров, каждое — до 500 символов.",
        examples=[["Ноутбук Lenovo", "Мышь компьютерная"]],
    )


class Alternative(APIModel):
    code: str | None = None
    title: str | None = None
    confidence: float = Field(ge=0, le=1)
    clause_ref: str | None = Field(
        default=None,
        description="Ссылка на позицию перечня (приложение, позиция), если известна.",
    )
    clause_url: str | None = Field(
        default=None,
        description="URL фрагмента источника (GET /source/{record_id}), если известен.",
    )


class ClarificationOption(APIModel):
    """One server-issued interpretation; ``option_id`` is not a client-supplied code."""

    option_id: str
    code: str
    title: str
    confidence: float = Field(ge=0, le=1)
    clause_ref: str | None = None
    clause_url: str | None = None
    code_role: str | None = None
    catalog_review_required: bool | None = None


class OriginalDecision(APIModel):
    """The model answer kept for audit before the ambiguity guard replaced it."""

    code: str | None = None
    title: str | None = None
    status: Literal["ok", "review", "not_found"]
    confidence: float = Field(ge=0, le=1)
    confidence_raw: float | None = Field(default=None, ge=0, le=1)
    alternatives: list[Alternative] = Field(default_factory=list)


class Clarification(APIModel):
    question: str
    options: list[ClarificationOption] = Field(min_length=2)
    catalog_sha256: str
    original: OriginalDecision


class Resolution(APIModel):
    option_id: str
    resolved_at: str
    original: OriginalDecision


class AttentionReason(APIModel):
    code: Literal[
        "ambiguous_name",
        "multiple_codes",
        "direct_title_match",
        "curated_product_match",
        "low_confidence",
        "code_exception",
        "catalog_review_required",
    ]
    message: str


class ClassifyResponse(APIModel):
    query: str
    code: str | None = None
    title: str | None = None
    confidence: float = Field(ge=0, le=1)
    confidence_raw: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "Уверенность до калибровки шкалы; именно по ней принимается решение "
            "ok/review/not_found (confidence — та же величина на калиброванной шкале)."
        ),
    )
    status: Literal["ok", "review", "not_found", "needs_clarification"]
    clause_ref: str | None = None
    clause_url: str | None = Field(
        default=None,
        description="URL фрагмента источника (GET /source/{record_id}); null, если неизвестен.",
    )
    code_role: str | None = Field(
        default=None,
        description=(
            "Роль кода позиции по каталогу: «Включён», «Исключение», «Не указан». "
            "Информационный флаг (BE-9): входит ли позиция-исключение в выдачу и как "
            "она обрабатывается — продуктовое решение (см. docs/api-contract.md); "
            "null, если каталог не даёт роли."
        ),
    )
    catalog_review_required: bool | None = Field(
        default=None,
        description=(
            "Строка каталога помечена к перепроверке при разборе PDF (сомнения OCR). "
            "Информационный флаг, НЕ связан со статусом ответа «review»; null, если "
            "каталог не даёт флага."
        ),
    )
    verifier: str | None = Field(
        default=None,
        description=(
            "Кто подтвердил код, если ответ пришёл от проверяющей ступени: "
            "например, «llm-top3» — модель языка выбрала одну из трёх "
            "предложенных позиций. null — ответ дан самим классификатором."
        ),
    )
    verifier_note: str | None = Field(
        default=None,
        description=(
            "Короткая пометка проверяющей ступени: «код подтверждён по top-3» "
            "или «нет совпадений». Не является юридическим заключением."
        ),
    )
    alternatives: list[Alternative] = Field(default_factory=list)
    clarification: Clarification | None = None
    resolution: Resolution | None = None
    decision_source: Literal["model", "specialist", "direct_code"] = "model"
    attention_reasons: list[AttentionReason] = Field(default_factory=list)
    catalog_sha256: str | None = None
    item_id: int | None = Field(default=None, ge=0, le=99)
    latency_ms: float = Field(ge=0)
    contract_version: str = Field(
        default=CONTRACT_VERSION,
        description="Версия формата результата (командный контракт, см. docs/api-contract.md).",
    )

    @model_validator(mode="after")
    def _decision_invariants(self) -> "ClassifyResponse":
        """A code is mandatory for an answer and forbidden before clarification.

        Contract 2.0 adds ``needs_clarification``: several interpretations are a
        question, not a weak top-1 answer. Such a response must carry at least
        two server-issued options and no winning code/title. A specialist's
        resolution is marked separately and never masquerades as model output.
        """
        if self.status in {"ok", "review"} and not (self.code or "").strip():
            raise ValueError("status 'ok'/'review' требует непустой code (контракт 2.0)")
        if self.status == "needs_clarification":
            if self.code or self.title or self.clarification is None:
                raise ValueError(
                    "needs_clarification требует code/title=null и clarification (контракт 2.0)"
                )
            if len(self.clarification.options) < 2:
                raise ValueError("needs_clarification требует минимум две опции")
        elif self.clarification is not None:
            raise ValueError("clarification допустим только при needs_clarification")
        if self.resolution is not None and self.decision_source != "specialist":
            raise ValueError("resolution требует decision_source='specialist'")
        if self.decision_source == "specialist" and self.resolution is None:
            raise ValueError("ответ специалиста требует сохранённый resolution")
        return self


class BatchJobAccepted(APIModel):
    job_id: str
    status: Literal["queued"]
    total: int = Field(ge=1, le=100)
    status_url: str
    contract_version: str = Field(
        default=CONTRACT_VERSION,
        description="Версия формата ответа batch-задачи.",
    )


class FileBatchJobAccepted(BatchJobAccepted):
    """Batch job created from an uploaded file plus what the parser extracted."""

    filename: str
    format: str
    items: list[str]
    total_found: int = Field(ge=0)
    duplicates: int = Field(ge=0)
    skipped: int = Field(ge=0)
    truncated: bool
    source_column: str | None = None
    warnings: list[str] = Field(default_factory=list)


ReportBucket = Literal["accepted", "needs_review", "not_found"]
ReportReasonCode = Literal[
    "low_confidence",
    "ambiguous_name",
    "multiple_codes",
    "code_exception",
    "catalog_review_required",
    "code_missing",
    "not_found",
]


class ReportReason(APIModel):
    code: ReportReasonCode
    message: str


class ProcurementReportItem(APIModel):
    item_id: int = Field(ge=0, le=99)
    bucket: ReportBucket
    risk_rank: int = Field(ge=0, le=100)
    reasons: list[ReportReason] = Field(default_factory=list)


class ProcurementSummary(APIModel):
    total: int = Field(ge=0, le=100)
    accepted: int = Field(ge=0, le=100)
    auto_accepted: int = Field(ge=0, le=100)
    specialist_resolved: int = Field(ge=0, le=100)
    needs_review: int = Field(ge=0, le=100)
    not_found: int = Field(ge=0, le=100)
    reason_counts: dict[ReportReasonCode, int] = Field(default_factory=dict)


class ProcurementReport(APIModel):
    summary: ProcurementSummary
    items: list[ProcurementReportItem] = Field(default_factory=list)


class BatchJobStatusResponse(APIModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    total: int = Field(ge=1, le=100)
    completed: int = Field(ge=0, le=100)
    progress_percent: float = Field(ge=0, le=100)
    results: list[ClassifyResponse] = Field(default_factory=list)
    report: ProcurementReport
    error: str | None = None
    contract_version: str = Field(
        default=CONTRACT_VERSION,
        description="Версия формата ответа batch-задачи.",
    )


class ReindexJobAccepted(APIModel):
    job_id: str
    status: Literal["queued"]
    status_url: str


class ReindexJobStatusResponse(APIModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    total: int | None = Field(default=None, ge=0)
    indexed: int = Field(ge=0)
    error: str | None = None


class SourcePages(APIModel):
    """Inclusive page range of the decree text the position was parsed from."""

    start: int | None = Field(default=None, ge=1)
    end: int | None = Field(default=None, ge=1)


class SourceFragment(APIModel):
    """Fragment of the decree text behind a catalog position (``GET /source/{id}``)."""

    record_id: str
    name: str | None = None
    okpd2: str | None = None
    clause_ref: str | None = None
    pages: SourcePages = Field(default_factory=SourcePages)
    text: str
    truncated: bool = Field(
        default=False,
        description="true, если текст обрезан по лимиту MAX_FRAGMENT_CHARS.",
    )
    source_file: str
    clause_url: str | None = None


class CatalogIssue(APIModel):
    row: int | None = Field(
        default=None, description="Номер строки файла (1-based), если применимо."
    )
    field: str | None = None
    message: str
    severity: Literal["error", "warning"]


class MaintenanceReport(APIModel):
    """Что сделала очистка по сроку хранения (BE-5)."""

    dry_run: bool = Field(default=False, description="true — только подсчёт, ничего не удалено.")
    audit_deleted: int = Field(ge=0, description="Удалено (или будет) строк журнала запросов.")
    feedback_deleted: int = Field(ge=0, description="Удалено (или будет) строк вердиктов.")
    audit_retention_days: int = Field(
        ge=0, description="Глубина журнала запросов; 0 — хранить вечно."
    )
    feedback_retention_days: int = Field(
        ge=0, description="Глубина вердиктов; 0 — хранить вечно."
    )
    skipped: str | None = Field(
        default=None,
        description="Причина, по которой очистка не выполнялась (не postgres, хранение отключено).",
    )


class CatalogInfo(APIModel):
    rows: int = Field(ge=0)
    columns: list[str] = Field(default_factory=list)
    sha256: str
    modified_at: str | None = None
    source_files: list[str] = Field(default_factory=list)


class CatalogVersion(APIModel):
    """Снимок каталога, сохранённый перед загрузкой новой редакции."""

    version_id: str
    rows: int = Field(ge=0)
    created_at: str = ""
    sha256: str = ""
    files: list[str] = Field(default_factory=list)


class CatalogVersionsResponse(APIModel):
    versions: list[CatalogVersion] = Field(default_factory=list)
    keep: int = Field(ge=1, description="Сколько последних снимков хранится.")
    current: CatalogInfo | None = None


class CatalogRollbackResponse(APIModel):
    restored: CatalogVersion
    catalog: CatalogInfo
    reindex_required: bool = Field(
        default=False,
        description="true, если после отката нужно выполнить POST /admin/reindex.",
    )


class CatalogUploadResponse(APIModel):
    dry_run: bool
    applied: bool
    rows_total: int = Field(ge=0)
    rows_valid: int = Field(ge=0)
    errors: list[CatalogIssue] = Field(default_factory=list)
    warnings: list[CatalogIssue] = Field(default_factory=list)
    catalog: CatalogInfo | None = None
    reindex_required: bool = Field(
        default=False,
        description="true, если после загрузки нужно выполнить POST /admin/reindex.",
    )
