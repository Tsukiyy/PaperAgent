"""公开 PDF 下载与保留 page/section 的文本解析工具。"""

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pymupdf

from paper_agent.exceptions import PdfDownloadError, PdfParseError
from paper_agent.http_client import create_http_client
from paper_agent.models import (
    DocumentChunk,
    DownloadedPdf,
    Paper,
    ParsedDocument,
    SectionRange,
)

PDF_MAGIC = b"%PDF-"
SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
NUMBERED_HEADING = re.compile(
    r"^(?:(?:[IVXLCDM]+|[A-Z])\.|\d+(?:\.\d+)*\.?)\s+\S+"
)
HEADING_PREFIX = re.compile(r"^(?:(?:[IVXLCDM]+|[A-Z])\.|\d+(?:\.\d+)*\.?)\s+")
ABSTRACT_INLINE = re.compile(r"^Abstract\s*[—-]\s*(.+)$", re.IGNORECASE)
KNOWN_HEADING = re.compile(
    r"^(?:\d+(?:\.\d+)*[.)]?\s+)?(?:abstract|introduction|related work|background|"
    r"methods?|methodology|approach|experiments?|evaluation|results?|discussion|"
    r"limitations?|conclusions?|references|acknowledg(?:e)?ments?|appendix)\b",
    re.IGNORECASE,
)
WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class _TextLine:
    """PyMuPDF 行信息，仅供标题识别使用。"""

    text: str
    font_size: float
    bold: bool


class PdfDownloader:
    """下载 arXiv 返回的公开 PDF，并限制文件类型和大小。"""

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        max_bytes: int = 50 * 1024 * 1024,
        http_client: httpx.Client | None = None,
        proxy: str | None = None,
    ) -> None:
        self._max_bytes = max_bytes
        self._owns_client = http_client is None
        self._client = http_client or create_http_client(
            timeout=timeout,
            proxy=proxy,
            headers={"User-Agent": "PaperAgent/0.2 (academic research assistant)"},
        )

    def __enter__(self) -> "PdfDownloader":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        """只关闭由本实例创建的 HTTP 客户端。"""

        if self._owns_client:
            self._client.close()

    def download(self, paper: Paper, destination: Path) -> DownloadedPdf:
        """下载公开 PDF；当前自动流程只消费 arXiv 元数据提供的链接。"""

        if paper.pdf_url is None:
            raise PdfDownloadError(f"论文 {paper.paper_id} 没有公开 PDF 链接。")
        if paper.pdf_url.scheme not in {"http", "https"}:
            raise PdfDownloadError("仅允许下载 HTTP(S) 公开 PDF。")

        destination.mkdir(parents=True, exist_ok=True)
        safe_id = SAFE_FILENAME.sub("_", paper.paper_id).strip("._") or "paper"
        output_path = destination / f"{safe_id}.pdf"
        temporary_path = output_path.with_suffix(".pdf.part")
        digest = hashlib.sha256()
        byte_size = 0
        first_bytes = bytearray()

        try:
            with self._client.stream("GET", str(paper.pdf_url)) as response:
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > self._max_bytes:
                    raise PdfDownloadError(
                        f"PDF 超过大小限制：{content_length} > {self._max_bytes} bytes。"
                    )
                with temporary_path.open("wb") as stream:
                    for chunk in response.iter_bytes():
                        if not chunk:
                            continue
                        byte_size += len(chunk)
                        if byte_size > self._max_bytes:
                            raise PdfDownloadError(
                                f"PDF 下载超过大小限制 {self._max_bytes} bytes。"
                            )
                        if len(first_bytes) < len(PDF_MAGIC):
                            missing = len(PDF_MAGIC) - len(first_bytes)
                            first_bytes.extend(chunk[:missing])
                        digest.update(chunk)
                        stream.write(chunk)
            if bytes(first_bytes) != PDF_MAGIC:
                raise PdfDownloadError("下载内容不是有效的 PDF 文件头。")
            temporary_path.replace(output_path)
        except PdfDownloadError:
            temporary_path.unlink(missing_ok=True)
            raise
        except (httpx.HTTPError, OSError, ValueError) as exc:
            temporary_path.unlink(missing_ok=True)
            raise PdfDownloadError(f"PDF 下载失败：{exc}") from exc

        return DownloadedPdf(
            paper_id=paper.paper_id,
            source_url=paper.pdf_url,
            local_path=str(output_path.resolve()),
            sha256=digest.hexdigest(),
            byte_size=byte_size,
        )


class PyMuPdfParser:
    """用字体和标题规则将 PDF 文本切分为带页码的章节块。"""

    def parse(self, pdf: DownloadedPdf) -> ParsedDocument:
        try:
            # 按页面提取文本块、行等信息
            with pymupdf.open(pdf.local_path) as document:
                if document.page_count < 1:
                    raise PdfParseError("PDF 没有可解析页面。")
                page_lines = [self._extract_lines(page) for page in document]
        except PdfParseError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise PdfParseError(f"PDF 打开或文本解析失败：{exc}") from exc

        # 估算正文字体大小
        body_size = self._estimate_body_size(page_lines)
        # 分块，返回DocumentChunk{chunk_id, page_number, section_title, text}
        chunks = self._build_chunks(page_lines, body_size)
        if not chunks:
            raise PdfParseError("PDF 未提取到可用文本，可能是扫描件或受保护文档。")
        sections = self._build_section_ranges(chunks)
        return ParsedDocument(
            paper_id=pdf.paper_id,
            pdf_path=pdf.local_path,
            page_count=len(page_lines),
            sections=sections,
            chunks=chunks,
        )

    @staticmethod
    def _extract_lines(page: pymupdf.Page) -> tuple[_TextLine, ...]:
        raw: dict[str, Any] = page.get_text("dict", sort=False)
        page_width = float(page.rect.width)

        def block_order(block: dict[str, Any]) -> tuple[int, float, float]:
            x0, y0, x1, _ = block.get("bbox", (0.0, 0.0, page_width, 0.0))
            width = float(x1) - float(x0)
            if width >= page_width * 0.7:
                return 0, float(y0), float(x0)
            column = 1 if (float(x0) + float(x1)) / 2 < page_width / 2 else 2
            return column, float(y0), float(x0)

        lines: list[_TextLine] = []
        for block in sorted(raw.get("blocks", []), key=block_order):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = WHITESPACE.sub(
                    " ",
                    "".join(str(span.get("text", "")) for span in spans),
                ).strip()
                if not text:
                    continue
                font_size = max((float(span.get("size", 0.0)) for span in spans), default=0.0)
                bold = any(int(span.get("flags", 0)) & 16 for span in spans)
                lines.append(_TextLine(text=text, font_size=font_size, bold=bold))
        return tuple(lines)

    @staticmethod
    def _estimate_body_size(page_lines: list[tuple[_TextLine, ...]]) -> float:
        weighted_sizes: Counter[float] = Counter()
        for lines in page_lines:
            for line in lines:
                weighted_sizes[round(line.font_size, 1)] += min(len(line.text), 200)
        if not weighted_sizes:
            return 10.0
        return weighted_sizes.most_common(1)[0][0]

    @staticmethod
    def _is_heading(line: _TextLine, body_size: float, page_number: int) -> bool:
        text = line.text.strip()
        if not 2 <= len(text) <= 160:
            return False
        known = bool(KNOWN_HEADING.match(text))
        numbered = bool(NUMBERED_HEADING.match(text))
        typographic = line.font_size >= body_size * 1.25 or (
            line.bold and line.font_size >= body_size
        )
        if known or numbered:
            # IEEE 等版式的标题可能与正文同字号，因此还要识别编号和标题大小写。
            return typographic or PyMuPdfParser._looks_like_title(text)
        if page_number == 1 or text.endswith((".", "?", "!", ";")):
            return False
        return typographic and len(text.split()) <= 14

    @staticmethod
    def _looks_like_title(text: str) -> bool:
        """区分标题式文本与仅以 Results/Limitations 开头的普通句子。"""

        if text.endswith((".", "?", "!", ";")):
            return False
        if re.match(r"^[A-Z]\.\s+", text) and "," in text:
            return False
        title = HEADING_PREFIX.sub("", text).strip()
        words = re.findall(r"[A-Za-z][A-Za-z-]*", title)
        if not words or len(words) > 16:
            return False
        capitalized = sum(word[0].isupper() for word in words)
        return title.isupper() or capitalized / len(words) >= 0.6

    def _build_chunks(
        self,
        page_lines: list[tuple[_TextLine, ...]],
        body_size: float,
    ) -> tuple[DocumentChunk, ...]:
        current_section = "Front Matter"
        raw_chunks: list[tuple[int, str, str]] = []
        for page_number, lines in enumerate(page_lines, start=1):
            buffer: list[str] = []
            for line in lines:
                abstract_match = ABSTRACT_INLINE.match(line.text)
                if abstract_match:
                    self._flush(raw_chunks, page_number, current_section, buffer)
                    current_section = "Abstract"
                    buffer = [abstract_match.group(1)]
                    continue
                if self._is_heading(line, body_size, page_number):
                    self._flush(raw_chunks, page_number, current_section, buffer)
                    current_section = line.text
                    buffer = []
                else:
                    buffer.append(line.text)
            self._flush(raw_chunks, page_number, current_section, buffer)

        return tuple(
            DocumentChunk(
                chunk_id=f"p{page_number}-c{index}",
                page_number=page_number,
                section_title=section,
                text=text,
            )
            for index, (page_number, section, text) in enumerate(raw_chunks, start=1)
        )

    @staticmethod
    def _flush(
        raw_chunks: list[tuple[int, str, str]],
        page_number: int,
        section: str,
        buffer: list[str],
    ) -> None:
        text = WHITESPACE.sub(" ", " ".join(buffer)).strip()
        if text:
            raw_chunks.append((page_number, section, text))

    @staticmethod
    def _build_section_ranges(
        chunks: tuple[DocumentChunk, ...],
    ) -> tuple[SectionRange, ...]:
        ranges: dict[str, list[int]] = {}
        order: list[str] = []
        for chunk in chunks:
            if chunk.section_title not in ranges:
                ranges[chunk.section_title] = [chunk.page_number, chunk.page_number]
                order.append(chunk.section_title)
            else:
                ranges[chunk.section_title][1] = chunk.page_number
        return tuple(
            SectionRange(
                title=title,
                start_page=ranges[title][0],
                end_page=ranges[title][1],
            )
            for title in order
        )
