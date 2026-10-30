import hashlib
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pymupdf
import pytest

from paper_agent.exceptions import PdfDownloadError, PdfParseError
from paper_agent.models import DownloadedPdf, Paper
from paper_agent.tools.pdf import PdfDownloader, PyMuPdfParser


def make_paper() -> Paper:
    return Paper(
        paper_id="2401.00001v1",
        title="Structured Paper",
        authors=("Researcher",),
        abstract="An abstract.",
        published_at=datetime(2024, 1, 1, tzinfo=UTC),
        updated_at=datetime(2024, 1, 1, tzinfo=UTC),
        paper_url="https://arxiv.org/abs/2401.00001v1",
        pdf_url="https://arxiv.org/pdf/2401.00001v1",
    )


def create_structured_pdf(path: Path) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Abstract", fontsize=16, fontname="hebo")
    page.insert_text(
        (72, 100),
        "This paper presents a grounded retrieval method for scientific analysis.",
        fontsize=10,
    )
    page.insert_text((72, 140), "I. INTRODUCTION", fontsize=10, fontname="helv")
    page.insert_text(
        (72, 168),
        "Reliable paper analysis requires conclusions linked to source evidence.",
        fontsize=10,
    )
    page.insert_text(
        (72, 190),
        "Results demonstrate that ordinary body sentences are not section titles.",
        fontsize=10,
    )
    page = document.new_page()
    page.insert_text((72, 72), "2 Methods", fontsize=16, fontname="hebo")
    page.insert_text(
        (72, 100),
        "The system preserves page numbers and section titles during extraction.",
        fontsize=10,
    )
    page.insert_text((72, 140), "3 Conclusion", fontsize=16, fontname="hebo")
    page.insert_text(
        (72, 168),
        "Evidence identifiers make generated claims auditable by researchers.",
        fontsize=10,
    )
    document.save(path)
    document.close()


def test_downloader_validates_pdf_and_calculates_hash(tmp_path: Path) -> None:
    content = b"%PDF-1.7\nmock content"
    transport = httpx.MockTransport(
        lambda _: httpx.Response(
            200,
            content=content,
            headers={"content-type": "application/pdf"},
        )
    )
    http_client = httpx.Client(transport=transport)
    downloader = PdfDownloader(http_client=http_client, max_bytes=1024)

    result = downloader.download(make_paper(), tmp_path)

    assert Path(result.local_path).read_bytes() == content
    assert result.sha256 == hashlib.sha256(content).hexdigest()
    assert result.byte_size == len(content)
    http_client.close()


def test_downloader_rejects_non_pdf_content(tmp_path: Path) -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, content=b"<html>no</html>"))
    http_client = httpx.Client(transport=transport)
    downloader = PdfDownloader(http_client=http_client)

    with pytest.raises(PdfDownloadError, match="文件头"):
        downloader.download(make_paper(), tmp_path)
    assert not list(tmp_path.glob("*.part"))
    http_client.close()


def test_parser_preserves_page_and_section(tmp_path: Path) -> None:
    path = tmp_path / "structured.pdf"
    create_structured_pdf(path)
    artifact = DownloadedPdf(
        paper_id="2401.00001v1",
        source_url="https://arxiv.org/pdf/2401.00001v1",
        local_path=str(path),
        sha256="0" * 64,
        byte_size=path.stat().st_size,
    )

    parsed = PyMuPdfParser().parse(artifact)

    assert parsed.page_count == 2
    assert any(
        chunk.section_title == "Abstract" and chunk.page_number == 1
        for chunk in parsed.chunks
    )
    assert any(section.title == "I. INTRODUCTION" for section in parsed.sections)
    assert any(
        chunk.section_title == "2 Methods" and chunk.page_number == 2
        for chunk in parsed.chunks
    )
    conclusion = next(section for section in parsed.sections if section.title == "3 Conclusion")
    assert conclusion.start_page == 2
    assert conclusion.end_page == 2
    assert not any(section.title.startswith("Results demonstrate") for section in parsed.sections)


def test_parser_reports_pdf_without_text(tmp_path: Path) -> None:
    path = tmp_path / "blank.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(path)
    document.close()
    artifact = DownloadedPdf(
        paper_id="blank",
        source_url="https://arxiv.org/pdf/2401.00001v1",
        local_path=str(path),
        sha256="0" * 64,
        byte_size=path.stat().st_size,
    )
    with pytest.raises(PdfParseError, match="未提取到可用文本"):
        PyMuPdfParser().parse(artifact)
