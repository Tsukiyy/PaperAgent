from datetime import UTC, datetime

from paper_agent.models import DocumentChunk, EvidenceClaim, EvidenceReference, Paper, PaperEvidence, ParsedDocument, DownloadedPdf
from paper_agent.services.review_validation import ReviewValidator


def _analysis(quote: str, claim: str = "The method improves accuracy.") -> PaperEvidence:
    paper = Paper(
        paper_id="p1", title="Paper", authors=("A",), abstract="", published_at=datetime(2024, 1, 1, tzinfo=UTC),
        updated_at=datetime(2024, 1, 1, tzinfo=UTC), paper_url="https://example.com/p1", pdf_url="https://example.com/p1.pdf",
    )
    chunk = DocumentChunk(chunk_id="c1", page_number=1, section_title="Results", text="The method improves accuracy.")
    return PaperEvidence(
        paper=paper,
        pdf=DownloadedPdf(paper_id="p1", source_url=paper.pdf_url, local_path="p.pdf", sha256="a" * 64, byte_size=1),
        document=ParsedDocument(paper_id="p1", pdf_path="p.pdf", page_count=1, sections=(), chunks=(chunk,)),
        claims=(EvidenceClaim(paper_id="p1", claim=claim, claim_type="result", evidence=(EvidenceReference(chunk_id="c1", page_number=1, section_title="Results", quote=quote),)),),
        provider="template",
    )


def test_validator_rejects_empty_quote():
    report = ReviewValidator().validate("accuracy", (_analysis(""),))
    assert not report.passed
    assert any(issue.rule == "quote_empty" for issue in report.issues)


def test_validator_rejects_quote_not_in_chunk():
    report = ReviewValidator().validate("accuracy", (_analysis("unrelated quote"),))
    assert not report.passed
    assert any(issue.rule == "quote_not_found" for issue in report.issues)


def test_validator_accepts_matching_quote_and_tracks_dimension():
    report = ReviewValidator().validate("accuracy", (_analysis("The method improves accuracy."),))
    assert report.passed
    assert report.dimension_coverage["result"] == ("p1",)
