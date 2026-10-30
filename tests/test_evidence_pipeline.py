import json
from datetime import UTC, datetime
from pathlib import Path

from paper_agent.config import Settings
from paper_agent.evidence.extractor import EvidenceService
from paper_agent.models import (
    DocumentChunk,
    DownloadedPdf,
    Paper,
    ParsedDocument,
    SectionRange,
)
from paper_agent.services.evidence_pipeline import EvidencePipeline


def make_paper(index: int) -> Paper:
    return Paper(
        paper_id=f"2401.0000{index}v1",
        title=f"Evidence Paper {index}",
        authors=("Researcher",),
        abstract="An abstract.",
        published_at=datetime(2024, 1, index, tzinfo=UTC),
        updated_at=datetime(2024, 1, index, tzinfo=UTC),
        paper_url=f"https://arxiv.org/abs/2401.0000{index}v1",
        pdf_url=f"https://arxiv.org/pdf/2401.0000{index}v1",
    )


class FakeSearcher:
    def search(self, query: str, *, limit: int = 5) -> tuple[Paper, ...]:
        return tuple(make_paper(index) for index in range(1, limit + 1))


class FakeDownloader:
    def download(self, paper: Paper, destination: Path) -> DownloadedPdf:
        return DownloadedPdf(
            paper_id=paper.paper_id,
            source_url=paper.pdf_url,
            local_path=str(destination / f"{paper.paper_id}.pdf"),
            sha256="0" * 64,
            byte_size=100,
        )


class FakeParser:
    def parse(self, pdf: DownloadedPdf) -> ParsedDocument:
        return ParsedDocument(
            paper_id=pdf.paper_id,
            pdf_path=pdf.local_path,
            page_count=2,
            sections=(SectionRange(title="2 Conclusion", start_page=2, end_page=2),),
            chunks=(
                DocumentChunk(
                    chunk_id="p2-c1",
                    page_number=2,
                    section_title="2 Conclusion",
                    text="The system preserves source locations for every generated claim.",
                ),
            ),
        )


def test_step_two_pipeline_writes_evidence_and_sections(tmp_path: Path) -> None:
    settings = Settings(
        PAPER_AGENT_EVIDENCE_PROVIDER="template",
        PAPER_AGENT_OUTPUT_DIR=tmp_path,
    )
    pipeline = EvidencePipeline(
        searcher=FakeSearcher(),
        downloader=FakeDownloader(),
        parser=FakeParser(),
        evidence_service=EvidenceService(settings),
        settings=settings,
    )

    result = pipeline.run("evidence RAG", limit=2)

    assert len(result.analyses) == 2
    output_dir = Path(result.output_dir)
    assert (output_dir / "review.md").exists()
    assert "p. 2, § 2 Conclusion" in result.markdown
    evidence_rows = [
        json.loads(line)
        for line in (output_dir / "evidence.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(evidence_rows) == 2
    assert evidence_rows[0]["evidence"][0]["page_number"] == 2
    assert evidence_rows[0]["evidence"][0]["section_title"] == "2 Conclusion"
    section_rows = (output_dir / "sections.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(section_rows) == 2

