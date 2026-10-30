"""PaperAgent 命令行入口。"""

import logging
import sys
from contextlib import ExitStack

import typer

from paper_agent.clients.arxiv import ArxivClient
from paper_agent.clients.openalex import OpenAlexClient
from paper_agent.clients.semantic_scholar import SemanticScholarClient
from paper_agent.config import get_settings
from paper_agent.evidence.extractor import EvidenceService
from paper_agent.exceptions import PaperAgentError
from paper_agent.services.evidence_pipeline import EvidencePipeline
from paper_agent.services.multi_source_search import MultiSourceSearchService
from paper_agent.services.pipeline import PaperSearchPipeline
from paper_agent.services.research_workflow import ResearchWorkflow, create_run_paths
from paper_agent.services.review import ReviewService
from paper_agent.services.synthesis import ClaimSynthesisService
from paper_agent.services.trace import RunTrace
from paper_agent.storage.database import SQLiteStore
from paper_agent.storage.vector_index import FaissVectorIndex
from paper_agent.tools.pdf import PdfDownloader, PyMuPdfParser
from paper_agent.tools.research import ResearchToolContext

app = typer.Typer(
    name="paper-agent",
    help="多源检索论文，并生成摘要综述、PDF 证据和 claims 对比。",
    no_args_is_help=True,
)


def _configure_logging() -> None:
    """同时将运行日志输出到终端和项目日志文件。"""

    log_path = get_settings().output_dir / "paper_agent.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=(
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_path, encoding="utf-8"),
        ),
        force=True,
    )
    print(f"运行日志：{log_path.resolve()}", file=sys.stderr)


@app.callback()
def main() -> None:
    """PaperAgent 命令组。"""

    _configure_logging()

# 论文检索功能
@app.command()
def search(
    query: str = typer.Argument(..., help="论文检索主题，例如：agentic RAG"),
    save: bool = typer.Option(True, "--save/--no-save", help="是否保存 review.md"),
) -> None:
    """检索最多 5 篇 arXiv 论文并生成简短综述。"""

    settings = get_settings()
    try:
        # 创建ArxivClient实例，在退出块时，client会被释放
        with ArxivClient(timeout=settings.request_timeout) as client:
            pipeline = PaperSearchPipeline(
                searcher=client,  # ArxivClient
                review_service=ReviewService(settings),  # 综述服务
                settings=settings,
            )
            result = pipeline.run(query, save=save)
    except (PaperAgentError, ValueError) as exc:
        typer.echo(f"错误：{exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(result.markdown)
    if result.output_path:
        typer.echo(f"\n已保存：{result.output_path}")

# 下载PDF
@app.command()
def analyze(
    query: str = typer.Argument(..., help="论文检索主题，例如：agentic RAG"),
    limit: int = typer.Option(5, min=1, max=5, help="下载并分析的论文数"),
) -> None:
    """检索公开 PDF，解析 page/section 并生成带证据结论。"""

    settings = get_settings()
    try:
        with (
            # 创建ArxivClient搜索服务、PdfDownloader下载服务
            ArxivClient(timeout=settings.request_timeout) as search_client,
            PdfDownloader(
                timeout=settings.request_timeout,
                max_bytes=settings.max_pdf_bytes,
                proxy=settings.http_proxy,
            ) as downloader,
        ):
            pipeline = EvidencePipeline(
                searcher=search_client,
                downloader=downloader,
                parser=PyMuPdfParser(),
                evidence_service=EvidenceService(settings),
                settings=settings,
            )
            result = pipeline.run(query, limit=limit)
    except (PaperAgentError, ValueError) as exc:
        typer.echo(f"错误：{exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(result.markdown)
    typer.echo(f"\n已保存：{result.output_dir}")


@app.command()
def research(
    query: str = typer.Argument(..., help="研究主题，例如：agentic RAG"),
    limit: int = typer.Option(5, min=1, max=20, help="去重排序后保留的论文数"),
    read_limit: int = typer.Option(5, min=1, max=20, help="下载并精读的论文数"),
) -> None:
    """运行 LangGraph 多源检索、PDF 证据分析与 claims 对比工作流。"""

    settings = get_settings()
    if read_limit > limit:
        typer.echo("错误：read-limit 不能大于 limit。", err=True)
        raise typer.Exit(code=1)
    _, run_dir, pdf_dir, faiss_dir = create_run_paths(  # 在创建工作流前生成一致的 run_id 与目录路径
        settings.output_dir, settings.data_dir
    )
    try:
        with ExitStack() as stack:  # 创建一个退出栈
            # ArxivClient登记到stack，立即进入该上下文（__enter__）
            arxiv = stack.enter_context(ArxivClient(timeout=settings.request_timeout))
            # SemanticScholarClient服务
            semantic_scholar = stack.enter_context(
                SemanticScholarClient(
                    api_key=settings.semantic_scholar_api_key,
                    timeout=settings.request_timeout,
                    user_agent=settings.semantic_scholar_user_agent,
                    proxy=settings.http_proxy,
                )
            )
            # OpenAlexClient服务
            openalex = stack.enter_context(
                OpenAlexClient(
                    api_key=settings.openalex_api_key,
                    timeout=settings.request_timeout,
                    proxy=settings.http_proxy,
                )
            )
            # PdfDownloader下载器
            downloader = stack.enter_context(
                PdfDownloader(
                    timeout=settings.request_timeout,
                    max_bytes=settings.max_pdf_bytes,
                    proxy=settings.http_proxy,
                )
            )
            # SQLiteStore
            store = stack.enter_context(SQLiteStore(settings.data_dir / "paper_agent.db"))
            # 依赖集合，整个工作流共享对象
            context = ResearchToolContext(
                search_service=MultiSourceSearchService(
                    {
                        "arXiv": arxiv,
                        "Semantic Scholar": semantic_scholar,
                        "OpenAlex": openalex,
                    }
                ),
                downloader=downloader,
                parser=PyMuPdfParser(),
                evidence_service=EvidenceService(settings),
                store=store,
                vector_index=FaissVectorIndex(
                    faiss_dir, dimension=settings.vector_dimension
                ),
                trace=RunTrace(),
                pdf_dir=pdf_dir,
                run_dir=run_dir,
                # v1.1新增synthesis_service
                synthesis_service=ClaimSynthesisService(settings),
            )
            # 工作流执行
            result = ResearchWorkflow(context).run(
                query, limit=limit, read_limit=read_limit
            )
    except (PaperAgentError, ValueError) as exc:
        typer.echo(f"错误：{exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"完成：检索 {len(result.papers)} 篇，精读 {len(result.analyses)} 篇。")
    typer.echo(f"已保存：{result.output_dir}")


if __name__ == "__main__":
    app()
