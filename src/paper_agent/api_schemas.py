"""Web API 请求与响应模型。"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ResearchCreateRequest(BaseModel):
    """创建研究任务的参数。"""

    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=5, ge=1, le=20)
    read_limit: int = Field(default=3, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("检索 query 不能为空。")
        return value

    @field_validator("read_limit")
    @classmethod
    def validate_read_limit(cls, value: int, info: Any) -> int:
        limit = info.data.get("limit", 5)
        if value > limit:
            raise ValueError("精读数量不能大于检索数量。")
        return value


class ResearchCreateResponse(BaseModel):
    run_id: str
    status: Literal["queued"] = "queued"


class ResearchStatusResponse(BaseModel):
    run_id: str
    status: Literal["queued", "running", "completed", "failed"]
    stage: str
    message: str
    error: str | None = None
    result: dict[str, Any] | None = None

