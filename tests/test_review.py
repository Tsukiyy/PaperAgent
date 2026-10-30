from datetime import UTC, datetime

from paper_agent.models import Paper, SearchResult
from paper_agent.renderers.markdown import render_review_markdown
from paper_agent.services.review import TemplateReviewGenerator


def make_paper(index: int) -> Paper:
    return Paper(
        paper_id=f"2401.0000{index}v1",
        title=f"Paper {index}",
        authors=("Alice",),
        abstract="This is the first finding. This is supporting detail.",
        published_at=datetime(2024, 1, index, tzinfo=UTC),
        updated_at=datetime(2024, 1, index, tzinfo=UTC),
        paper_url=f"https://arxiv.org/abs/2401.0000{index}v1",
        pdf_url=f"https://arxiv.org/pdf/2401.0000{index}v1",
    )


def test_template_review_only_uses_first_abstract_sentence() -> None:
    paper = make_paper(1)
    review = TemplateReviewGenerator().generate("RAG", (paper,))
    assert "This is the first finding." in review
    assert "supporting detail" not in review


def test_renderer_guarantees_every_paper_link() -> None:
    papers = tuple(make_paper(index) for index in range(1, 6))
    search = SearchResult(query="RAG", papers=papers)
    markdown = render_review_markdown(search, "简短综述。", "template")
    for paper in papers:
        assert f"[{paper.title}]({paper.paper_url})" in markdown

