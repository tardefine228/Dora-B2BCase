"""Public request/response contracts; exposed automatically in OpenAPI."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


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


class ClassifyResponse(APIModel):
    query: str
    code: str | None = None
    title: str | None = None
    confidence: float = Field(ge=0, le=1)
    status: Literal["ok", "review", "not_found"]
    clause_ref: str | None = None
    clause_url: str | None = None
    alternatives: list[Alternative] = Field(default_factory=list)
    latency_ms: float = Field(ge=0)


class BatchJobAccepted(APIModel):
    job_id: str
    status: Literal["queued"]
    total: int = Field(ge=1, le=100)
    status_url: str


class BatchJobStatusResponse(APIModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    total: int = Field(ge=1, le=100)
    completed: int = Field(ge=0, le=100)
    progress_percent: float = Field(ge=0, le=100)
    results: list[ClassifyResponse] = Field(default_factory=list)
    error: str | None = None


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
