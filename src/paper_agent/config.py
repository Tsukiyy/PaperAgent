"""应用配置。"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """从环境变量和项目根目录的 .env 加载配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = Field(
        default="openai-compatible",
        alias="LLM_PROVIDER",
    )
    llm_api_key: str | None = Field(default=None, alias="LLM_API_KEY")
    llm_base_url: str | None = Field(default=None, alias="LLM_BASE_URL")
    llm_model: str | None = Field(default=None, alias="LLM_MODEL")
    llm_max_tokens: int = Field(default=4096, ge=1, alias="LLM_MAX_TOKENS")
    http_proxy: str | None = Field(default=None, alias="PAPER_AGENT_HTTP_PROXY")

    # 三个需要接入LLM的业务模块
    review_provider: Literal["auto", "llm", "template"] = Field(
        default="auto",
        alias="PAPER_AGENT_REVIEW_PROVIDER",
    )
    # Review 阶段默认关闭 LLM 兜底，避免逐条审核造成请求量放大。
    review_llm_enabled: bool = Field(
        default=False,
        alias="PAPER_AGENT_REVIEW_LLM_ENABLED",
    )
    evidence_provider: Literal["auto", "llm", "template"] = Field(
        default="auto",
        alias="PAPER_AGENT_EVIDENCE_PROVIDER",
    )
    synthesis_provider: Literal["auto", "llm", "template"] = Field(
        default="auto",
        alias="PAPER_AGENT_SYNTHESIS_PROVIDER",
    )
    request_timeout: float = Field(
        default=90.0,
        gt=0,
        alias="PAPER_AGENT_REQUEST_TIMEOUT",
    )
    max_pdf_bytes: int = Field(
        default=50 * 1024 * 1024,
        gt=0,
        alias="PAPER_AGENT_MAX_PDF_BYTES",
    )
    max_claims_per_paper: int = Field(
        default=20,
        ge=1,
        le=20,
        alias="PAPER_AGENT_MAX_CLAIMS_PER_PAPER",
    )
    claims_review_max_retries: int = Field(
        default=1,
        ge=0,
        le=5,
        alias="CLAIMS_REVIEW_MAX_RETRIES",
    )
    output_dir: Path = Field(
        default=Path("outputs/paper-agent"),
        alias="PAPER_AGENT_OUTPUT_DIR",
    )
    data_dir: Path = Field(
        default=Path("data"),
        alias="PAPER_AGENT_DATA_DIR",
    )
    semantic_scholar_api_key: str | None = Field(
        default=None,
        alias="SEMANTIC_SCHOLAR_API_KEY",
    )
    semantic_scholar_user_agent: str = Field(
        default="PaperAgent/0.3 (academic research assistant)",
        alias="SEMANTIC_SCHOLAR_USER_AGENT",
    )
    openalex_api_key: str | None = Field(
        default=None,
        alias="OPENALEX_API_KEY",
    )
    vector_dimension: int = Field(
        default=384,
        ge=64,
        le=4096,
        alias="PAPER_AGENT_VECTOR_DIMENSION",
    )

    @field_validator("llm_provider", mode="before")
    @classmethod
    def normalize_llm_provider(cls, value: object) -> object:
        """允许环境变量使用 KIMI、QWEN 等大写写法。"""

        return value.strip().lower() if isinstance(value, str) else value

    @property
    def missing_llm_settings(self) -> tuple[str, ...]:
        """返回当前 OpenAI 兼容调用缺少的通用配置项。"""

        values = {
            "LLM_API_KEY": self.llm_api_key,
            "LLM_BASE_URL": self.llm_base_url,
            "LLM_MODEL": self.llm_model,
        }
        return tuple(name for name, value in values.items() if not value)

    @property
    def llm_configured(self) -> bool:
        """判断调用 LLM 所需的通用配置是否完整。"""

        return not self.missing_llm_settings


@lru_cache
def get_settings() -> Settings:
    """返回进程级缓存配置。"""

    return Settings()
