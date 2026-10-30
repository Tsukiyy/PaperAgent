"""Step 1 使用的领域模型；并追加 Step 2 的 PDF、章节和证据模型。"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class Paper(BaseModel):
    """统一论文元数据；后续数据源也应转换成该模型。"""

    model_config = ConfigDict(frozen=True)

    paper_id: str
    title: str
    authors: tuple[str, ...]
    abstract: str
    published_at: datetime
    updated_at: datetime
    paper_url: HttpUrl
    pdf_url: HttpUrl | None = None
    categories: tuple[str, ...] = ()
    primary_category: str | None = None
    source: str = "arXiv"
    sources: tuple[str, ...] = ("arXiv",)
    doi: str | None = None
    external_ids: dict[str, str] = Field(default_factory=dict)
    citation_count: int = Field(default=0, ge=0)
    reference_count: int = Field(default=0, ge=0)
    is_open_access: bool = False
    relevance_score: float = 0.0
    rank_score: float = 0.0


class SearchResult(BaseModel):
    """一次检索的结构化结果。"""

    model_config = ConfigDict(frozen=True)

    query: str
    papers: tuple[Paper, ...] = Field(max_length=5)
    source: str = "arXiv"


class ReviewResult(BaseModel):
    """一次 Step 1 流程的最终结果。"""

    model_config = ConfigDict(frozen=True)

    search: SearchResult
    review_body: str
    provider: str
    markdown: str
    output_path: str | None = None


class DownloadedPdf(BaseModel):
    """已下载并通过 PDF 文件头校验的公开论文。"""

    model_config = ConfigDict(frozen=True)

    paper_id: str
    source_url: HttpUrl
    local_path: str
    sha256: str
    byte_size: int = Field(gt=0)


class SectionRange(BaseModel):
    """解析出的章节及其页码范围，页码从 1 开始。"""

    model_config = ConfigDict(frozen=True)

    title: str
    start_page: int = Field(ge=1)
    end_page: int = Field(ge=1)


class DocumentChunk(BaseModel):
    """同一页、同一章节内的连续文本，是最小证据定位单元。"""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    page_number: int = Field(ge=1)
    section_title: str
    text: str


class ParsedDocument(BaseModel):
    """保留页码和章节信息的 PDF 解析结果。"""

    model_config = ConfigDict(frozen=True)

    paper_id: str
    pdf_path: str
    page_count: int = Field(ge=1)
    sections: tuple[SectionRange, ...]
    chunks: tuple[DocumentChunk, ...]


class EvidenceReference(BaseModel):
    """结论对应的原文证据，quote 必须来自指定文本块。"""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    page_number: int = Field(ge=1)
    section_title: str
    quote: str


class EvidenceClaim(BaseModel):
    """至少包含一条页码或章节证据的论文结论。"""

    model_config = ConfigDict(frozen=True)

    paper_id: str
    claim: str
    claim_type: Literal[
        "contribution",
        "method",
        "dataset",
        "metric",
        "result",
        "limitation",
        "other",
    ] = "other"
    evidence: tuple[EvidenceReference, ...] = Field(min_length=1)


class ComparisonRow(BaseModel):
    """按论文聚合的 claims 对比表行。"""

    model_config = ConfigDict(frozen=True)

    paper_id: str
    title: str
    paper_url: HttpUrl
    contributions: str = ""
    methods: str = ""
    datasets: str = ""
    metrics: str = ""
    results: str = ""
    limitations: str = ""
    evidence_locations: str = ""


class PaperEvidence(BaseModel):
    """单篇论文从下载、解析到结论提取的完整结果。"""

    model_config = ConfigDict(frozen=True)

    paper: Paper
    pdf: DownloadedPdf
    document: ParsedDocument
    claims: tuple[EvidenceClaim, ...]
    provider: str


class AnalysisFailure(BaseModel):
    """单篇论文失败记录；失败不会伪装成分析结论。"""

    model_config = ConfigDict(frozen=True)

    paper_id: str
    title: str
    stage: Literal["download", "parse", "evidence"]
    reason: str


class EvidenceRunResult(BaseModel):
    """Step 2 查询的结构化结果。"""

    model_config = ConfigDict(frozen=True)

    search: SearchResult
    analyses: tuple[PaperEvidence, ...]
    failures: tuple[AnalysisFailure, ...]
    markdown: str
    output_dir: str


class ResearchRunResult(BaseModel):
    """LangGraph 多源研究工作流的最终结果。"""

    model_config = ConfigDict(frozen=True)

    run_id: str
    query: str
    papers: tuple[Paper, ...]
    analyses: tuple[PaperEvidence, ...]
    comparisons: tuple[ComparisonRow, ...]
    failures: tuple[AnalysisFailure, ...]
    review_passed: bool
    # 结构化审核报告；保留可选以兼容旧版调用方。
    review_report: object | None = None
    output_dir: str
    outputs: dict[str, str]
    process: dict[str, str] = Field(default_factory=dict)
