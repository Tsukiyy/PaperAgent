"""PaperAgent 可组合工具。"""

from paper_agent.tools.pdf import PdfDownloader, PyMuPdfParser
from paper_agent.tools.research import (
    compare_papers,
    download_pdf,
    extract_claims,
    parse_pdf,
    search_papers,
    write_review,
)

__all__ = [
    "compare_papers",
    "download_pdf",
    "extract_claims",
    "PdfDownloader",
    "parse_pdf",
    "PyMuPdfParser",
    "search_papers",
    "write_review",
]
