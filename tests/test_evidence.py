from datetime import UTC, datetime

from paper_agent.evidence.extractor import TemplateEvidenceExtractor
from paper_agent.models import DocumentChunk, Paper, ParsedDocument, SectionRange


def make_paper() -> Paper:
    return Paper(
        paper_id="2401.00001v1",
        title="Evidence Paper",
        authors=("Researcher",),
        abstract="An abstract.",
        published_at=datetime(2024, 1, 1, tzinfo=UTC),
        updated_at=datetime(2024, 1, 1, tzinfo=UTC),
        paper_url="https://arxiv.org/abs/2401.00001v1",
        pdf_url="https://arxiv.org/pdf/2401.00001v1",
    )


def make_document() -> ParsedDocument:
    return ParsedDocument(
        paper_id="2401.00001v1",
        pdf_path="paper.pdf",
        page_count=3,
        sections=(SectionRange(title="3 Conclusion", start_page=3, end_page=3),),
        chunks=(
            DocumentChunk(
                chunk_id="p3-c1",
                page_number=3,
                section_title="3 Conclusion",
                text="The evidence-linked approach improves auditability. More details follow.",
            ),
        ),
    )


def test_template_claim_always_contains_page_section_and_quote() -> None:
    document = make_document()
    claims = TemplateEvidenceExtractor(max_claims=2).extract(make_paper(), document)

    assert len(claims) == 1
    reference = claims[0].evidence[0]
    assert reference.page_number == 3
    assert reference.section_title == "3 Conclusion"
    assert reference.quote in document.chunks[0].text


def test_template_claims_cover_comparison_dimensions() -> None:
    texts = (
        ("Introduction", "We propose a novel agentic RAG taxonomy for adaptive retrieval systems."),
        (
            "Methods",
            "The framework uses a router workflow to delegate retrieval to specialized agents.",
        ),
        (
            "Dataset",
            "We construct an evaluation dataset containing 500 grounded research questions.",
        ),
        ("Evaluation Metrics", "We evaluate answer quality using F1 score and retrieval accuracy."),
        ("Results", "The proposed system achieved 81.2% accuracy and outperformed the baseline."),
        (
            "Limitations",
            "The method is constrained by retrieval latency and limited domain coverage.",
        ),
    )
    chunks = tuple(
        DocumentChunk(
            chunk_id=f"p{index}-c{index}",
            page_number=index,
            section_title=section,
            text=text,
        )
        for index, (section, text) in enumerate(texts, start=1)
    )
    document = ParsedDocument(
        paper_id="2401.00001v1",
        pdf_path="paper.pdf",
        page_count=6,
        sections=(),
        chunks=chunks,
    )

    claims = TemplateEvidenceExtractor(max_claims=6).extract(make_paper(), document)

    assert {claim.claim_type for claim in claims} == {
        "contribution",
        "method",
        "dataset",
        "metric",
        "result",
        "limitation",
    }
