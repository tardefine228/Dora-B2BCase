from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClassifyRequest(APIModel):
    name: str = Field(min_length=1, max_length=500)

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class BatchClassifyRequest(APIModel):
    items: list[str] = Field(min_length=1, max_length=100)

    @field_validator("items", mode="before")
    @classmethod
    def strip_items(cls, value: object) -> object:
        if isinstance(value, list):
            return [item.strip() if isinstance(item, str) else item for item in value]
        return value

    @field_validator("items")
    @classmethod
    def validate_items(cls, value: list[str]) -> list[str]:
        if any(not item for item in value):
            raise ValueError("Название каждого товара должно быть непустым")
        if any(len(item) > 500 for item in value):
            raise ValueError("Название товара не должно быть длиннее 500 символов")
        return value


class Alternative(APIModel):
    code: str
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
