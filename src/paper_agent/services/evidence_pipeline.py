"""Step 2 PDF 下载、解析和证据提取流水线。"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from paper_agent.config import Settings
from paper_agent.evidence.extractor import EvidenceService
from paper_agent.exceptions import EvidenceError, PdfDownloadError, PdfParseError
from paper_agent.models import (
    AnalysisFailure,
    DownloadedPdf,
    EvidenceRunResult,
    Paper,
    PaperEvidence,
    ParsedDocument,
    SearchResult,
)
from paper_agent.renderers.evidence_markdown import render_evidence_markdown


class PaperSearcher(Protocol):
    """Step 2 论文搜索器协议。"""

    def search(self, query: str, *, limit: int = 5) -> tuple[Paper, ...]: ...


class PdfDownloadTool(Protocol):
    """PDF 下载工具协议。"""

    def download(self, paper: Paper, destination: Path) -> DownloadedPdf: ...


class PdfParseTool(Protocol):
    """PDF 解析工具协议。"""

    def parse(self, pdf: DownloadedPdf) -> ParsedDocument: ...


class EvidencePipeline:
    """逐篇执行下载、解析和证据提取，允许部分论文失败。"""

    def __init__(
        self,
        *,
        searcher: PaperSearcher,
        downloader: PdfDownloadTool,
        parser: PdfParseTool,
        evidence_service: EvidenceService,
        settings: Settings,
    ) -> None:
        self._searcher = searcher
        self._downloader = downloader
        self._parser = parser
        self._evidence_service = evidence_service
        self._settings = settings

    def run(self, query: str, *, limit: int = 5) -> EvidenceRunResult:
        clean_query = " ".join(query.split())
        if not clean_query:
            raise ValueError("检索 query 不能为空。")
        if not 1 <= limit <= 5:
            raise ValueError("Step 2 的 limit 必须在 1 到 5 之间。")

        papers = self._searcher.search(clean_query, limit=limit)
        search = SearchResult(query=clean_query, papers=papers)
        generated_at = datetime.now(UTC)
        run_id = generated_at.strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = self._settings.output_dir / run_id
        pdf_dir = run_dir / "papers"
        run_dir.mkdir(parents=True, exist_ok=False)
        analyses: list[PaperEvidence] = []
        failures: list[AnalysisFailure] = []
        for paper in papers:
            try:
                downloaded = self._downloader.download(paper, pdf_dir)
            except PdfDownloadError as exc:
                failures.append(self._failure(paper, "download", exc))
                continue
            try:
                document = self._parser.parse(downloaded)
            except PdfParseError as exc:
                failures.append(self._failure(paper, "parse", exc))
                continue
            try:
                claims, provider = self._evidence_service.extract(paper, document)
            except EvidenceError as exc:
                failures.append(self._failure(paper, "evidence", exc))
                continue
            analyses.append(
                PaperEvidence(
                    paper=paper,
                    pdf=downloaded,
                    document=document,
                    claims=claims,
                    provider=provider,
                )
            )

        analysis_tuple = tuple(analyses)
        failure_tuple = tuple(failures)
        markdown = render_evidence_markdown(
            search,
            analysis_tuple,
            failure_tuple,
            generated_at=generated_at,
        )
        self._write_outputs(run_dir, markdown, analysis_tuple, failure_tuple)
        return EvidenceRunResult(
            search=search,
            analyses=analysis_tuple,
            failures=failure_tuple,
            markdown=markdown,
            output_dir=str(run_dir.resolve()),
        )

    @staticmethod
    def _failure(
        paper: Paper,
        stage: Literal["download", "parse", "evidence"],
        error: Exception,
    ) -> AnalysisFailure:
        return AnalysisFailure(
            paper_id=paper.paper_id,
            title=paper.title,
            stage=stage,
            reason=str(error),
        )

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
        content = "".join(
            f"{json.dumps(record, ensure_ascii=False)}\n" for record in records
        )
        path.write_text(content, encoding="utf-8")

    def _write_outputs(
        self,
        run_dir: Path,
        markdown: str,
        analyses: tuple[PaperEvidence, ...],
        failures: tuple[AnalysisFailure, ...],
    ) -> None:
        (run_dir / "review.md").write_text(markdown, encoding="utf-8")
        evidence_records: list[dict[str, object]] = []
        section_records: list[dict[str, object]] = []
        for analysis in analyses:
            for claim in analysis.claims:
                evidence_records.append(
                    {
                        "paper_id": analysis.paper.paper_id,
                        "title": analysis.paper.title,
                        "paper_url": str(analysis.paper.paper_url),
                        "provider": analysis.provider,
                        **claim.model_dump(mode="json"),
                    }
                )
            for chunk in analysis.document.chunks:
                section_records.append(
                    {
                        "paper_id": analysis.paper.paper_id,
                        **chunk.model_dump(mode="json"),
                    }
                )
        self._write_jsonl(run_dir / "evidence.jsonl", evidence_records)
        self._write_jsonl(run_dir / "sections.jsonl", section_records)
        self._write_jsonl(
            run_dir / "failures.jsonl",
            [failure.model_dump(mode="json") for failure in failures],
        )
