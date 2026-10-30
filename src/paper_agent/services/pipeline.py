"""Step 1 检索与综述流水线。"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from paper_agent.config import Settings
from paper_agent.models import Paper, ReviewResult, SearchResult
from paper_agent.renderers.markdown import render_review_markdown
from paper_agent.services.review import ReviewService


class PaperSearcher(Protocol):
    """论文搜索器协议，方便 Step 3 接入更多数据源。"""

    def search(self, query: str, *, limit: int = 5) -> tuple[Paper, ...]: ...


class PaperSearchPipeline:
    """编排固定的检索、综述和输出流程。"""

    def __init__(
        self,
        *,
        searcher: PaperSearcher,
        review_service: ReviewService,
        settings: Settings,
    ) -> None:
        self._searcher = searcher
        self._review_service = review_service
        self._settings = settings

    def run(self, query: str, *, save: bool = True) -> ReviewResult:
        # 清洗query
        clean_query = " ".join(query.split())
        if not clean_query:
            raise ValueError("检索 query 不能为空。")

        # 通过Arxiv客户端服务获取limit条相关文献的信息
        papers = self._searcher.search(clean_query, limit=5)  # ArxivClient.search
        # 检索内容结构化{query,papers,source}
        search_result = SearchResult(query=clean_query, papers=papers)
        # 生成简短综述 ReviewService.generate，返回["综述内容","template/openai"]
        review_body, provider = self._review_service.generate(clean_query, papers)
        generated_at = datetime.now(UTC)
        # 生成带链接的md
        markdown = render_review_markdown(
            search_result,  # 检索到的内容
            review_body,  # 综述
            provider,  # 综述提供方式
            generated_at=generated_at,  # 生成时间
        )
        # 保存结果
        output_path = self._save(markdown, generated_at) if save else None
        return ReviewResult(
            search=search_result,
            review_body=review_body,
            provider=provider,
            markdown=markdown,
            output_path=str(output_path) if output_path else None,
        )

    def _save(self, markdown: str, generated_at: datetime) -> Path:
        run_id = generated_at.strftime("%Y%m%dT%H%M%S%fZ")
        # run_id 作为文件夹
        run_dir = self._settings.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        output_path = run_dir / "review.md"
        # md综述预览和连接保存为 review.md
        output_path.write_text(markdown, encoding="utf-8")
        return output_path.resolve()

