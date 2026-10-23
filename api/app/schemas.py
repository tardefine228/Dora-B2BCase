"""Public request/response contracts; exposed automatically in OpenAPI.

The classification result format is the team-agreed contract v1.0 (see
``docs/api-contract.md``): new optional fields may be added, existing fields are
never renamed or removed within a major version.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

CONTRACT_VERSION = "1.0"


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



class FeedbackRequest(APIModel):
    query: ProductName
    predicted_code: str | None = Field(default=None, max_length=32)
    verdict: Literal["correct", "incorrect", "absent"]
    corrected_code: str | None = Field(default=None, max_length=32)


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
    status: Literal["ok", "review", "not_found"]
    clause_ref: str | None = None
    clause_url: str | None = Field(
        default=None,
        description="URL фрагмента источника (GET /source/{record_id}); null, если неизвестен.",
    )
    alternatives: list[Alternative] = Field(default_factory=list)
    latency_ms: float = Field(ge=0)
    contract_version: str = Field(
        default=CONTRACT_VERSION,
        description="Версия формата результата (командный контракт, см. docs/api-contract.md).",
    )


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


class BatchJobStatusResponse(APIModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    total: int = Field(ge=1, le=100)
    completed: int = Field(ge=0, le=100)
    progress_percent: float = Field(ge=0, le=100)
    results: list[ClassifyResponse] = Field(default_factory=list)
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


class CatalogInfo(APIModel):
    rows: int = Field(ge=0)
    columns: list[str] = Field(default_factory=list)
    sha256: str
    modified_at: str | None = None
    source_files: list[str] = Field(default_factory=list)


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
