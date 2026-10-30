from datetime import UTC, datetime
from pathlib import Path

from paper_agent.config import Settings
from paper_agent.evidence.extractor import EvidenceService
from paper_agent.exceptions import PdfDownloadError
from paper_agent.models import DocumentChunk, DownloadedPdf, Paper, ParsedDocument, SectionRange
from paper_agent.services.multi_source_search import MultiSourceSearchService
from paper_agent.services.research_workflow import ResearchWorkflow
from paper_agent.services.trace import RunTrace
from paper_agent.storage.database import SQLiteStore
from paper_agent.storage.vector_index import FaissVectorIndex
from paper_agent.tools.research import ResearchToolContext, download_pdf


class FakeSearchClient:
    def __init__(self, paper: Paper) -> None:
        self.paper = paper

    def search(self, query: str, *, limit: int = 5) -> tuple[Paper, ...]:
        return (self.paper,)


class FakeDownloader:
    def download(self, paper: Paper, destination: Path) -> DownloadedPdf:
        return DownloadedPdf(
            paper_id=paper.paper_id,
            source_url=paper.pdf_url,
            local_path=str(destination / "paper.pdf"),
            sha256="a" * 64,
            byte_size=100,
        )


class FakeParser:
    def parse(self, pdf: DownloadedPdf) -> ParsedDocument:
        return ParsedDocument(
            paper_id=pdf.paper_id,
            pdf_path=pdf.local_path,
            page_count=2,
            sections=(SectionRange(title="Methods", start_page=2, end_page=2),),
            chunks=(
                DocumentChunk(
                    chunk_id="p2-c1",
                    page_number=2,
                    section_title="Methods",
                    text=(
                        "We propose an agentic retrieval method that plans multiple search steps "
                        "and verifies the retrieved evidence before synthesis."
                    ),
                ),
            ),
        )


class FallbackDownloader:
    def __init__(self, failed_ids: set[str]) -> None:
        self.failed_ids = failed_ids
        self.attempted_ids: list[str] = []

    def download(self, paper: Paper, destination: Path) -> DownloadedPdf:
        self.attempted_ids.append(paper.paper_id)
        if paper.paper_id in self.failed_ids:
            raise PdfDownloadError("模拟 PDF 下载失败")
        return DownloadedPdf(
            paper_id=paper.paper_id,
            source_url=paper.pdf_url,
            local_path=str(destination / f"{paper.paper_id}.pdf"),
            sha256="a" * 64,
            byte_size=100,
        )


def test_download_pdf_skips_failures_until_read_limit_is_reached(tmp_path: Path) -> None:
    papers = tuple(
        Paper(
            paper_id=f"paper-{index}",
            title=f"Paper {index}",
            authors=("Researcher",),
            abstract="An abstract.",
            published_at=datetime(2025, 1, index, tzinfo=UTC),
            updated_at=datetime(2025, 1, index, tzinfo=UTC),
            paper_url=f"https://example.com/{index}",
            pdf_url=f"https://example.com/{index}.pdf",
        )
        for index in range(1, 5)
    )
    downloader = FallbackDownloader({"paper-2"})
    context = ResearchToolContext(
        search_service=MultiSourceSearchService({}),
        downloader=downloader,
        parser=FakeParser(),
        evidence_service=EvidenceService(Settings(PAPER_AGENT_EVIDENCE_PROVIDER="template")),
        store=SQLiteStore(tmp_path / "data" / "paper_agent.db"),
        vector_index=FaissVectorIndex(tmp_path / "data" / "faiss"),
        trace=RunTrace(),
        pdf_dir=tmp_path / "papers",
        run_dir=tmp_path / "run",
    )
    try:
        context.store.start_run("test-download-fallback", "agentic retrieval")
        context.store.save_papers("test-download-fallback", papers)
        downloaded, failures = download_pdf(papers, read_limit=3, context=context)
    finally:
        context.store.close()

    assert [pdf.paper_id for pdf in downloaded] == ["paper-1", "paper-3", "paper-4"]
    assert [failure.paper_id for failure in failures] == ["paper-2"]
    assert downloader.attempted_ids == ["paper-1", "paper-2", "paper-3", "paper-4"]


def test_langgraph_workflow_writes_exact_six_outputs(tmp_path: Path) -> None:
    now = datetime(2025, 1, 1, tzinfo=UTC)
    paper = Paper(
        paper_id="2501.00001",
        title="Agentic Retrieval",
        authors=("Ada Lovelace",),
        abstract="Agentic retrieval with planning and evidence verification.",
        published_at=now,
        updated_at=now,
        paper_url="https://arxiv.org/abs/2501.00001",
        pdf_url="https://arxiv.org/pdf/2501.00001",
        external_ids={"ArXiv": "2501.00001"},
        is_open_access=True,
    )
    run_id = "test-run"
    run_dir = tmp_path / "outputs" / run_id
    store = SQLiteStore(tmp_path / "data" / "paper_agent.db")
    context = ResearchToolContext(
        search_service=MultiSourceSearchService({"arXiv": FakeSearchClient(paper)}),
        downloader=FakeDownloader(),
        parser=FakeParser(),
        evidence_service=EvidenceService(Settings(PAPER_AGENT_EVIDENCE_PROVIDER="template")),
        store=store,
        vector_index=FaissVectorIndex(tmp_path / "data" / "faiss" / run_id),
        trace=RunTrace(),
        pdf_dir=tmp_path / "data" / "papers",
        run_dir=run_dir,
    )

    try:
        result = ResearchWorkflow(context).run("agentic retrieval", limit=1, read_limit=1)
    finally:
        store.close()

    assert result.review_passed is True
    assert {path.name for path in run_dir.iterdir()} == {
        "review.md",
        "papers.bib",
        "evidence.jsonl",
        "comparison.csv",
        "trace.jsonl",
        "eval_summary.md",
    }
    assert "p.2 §Methods" in (run_dir / "review.md").read_text(encoding="utf-8")
    assert "### 方法差异" in (run_dir / "review.md").read_text(encoding="utf-8")
    assert "methods" in (run_dir / "comparison.csv").read_text(encoding="utf-8-sig")
    assert (tmp_path / "data" / "faiss" / run_id / "chunks.faiss").exists()


def test_claim_review_failure_does_not_skip_comparison_or_synthesis(monkeypatch) -> None:
    from paper_agent.services.research_workflow import ResearchState
    from paper_agent.services.review_validation import ReviewReport

    monkeypatch.setattr(
        "paper_agent.services.research_workflow.get_settings",
        lambda: Settings(CLAIMS_REVIEW_MAX_RETRIES=1),
    )
    state: ResearchState = {
        "claims_review_report": ReviewReport(False),
        "claims_review_attempts": 2,
        "documents": (),
    }

    assert ResearchWorkflow._claims_review_route(state) == "continue"
