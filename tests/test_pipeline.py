from datetime import UTC, datetime
from pathlib import Path

from paper_agent.config import Settings
from paper_agent.models import Paper
from paper_agent.services.pipeline import PaperSearchPipeline
from paper_agent.services.review import ReviewService


class FakeSearcher:
    def __init__(self) -> None:
        self.received_limit: int | None = None

    def search(self, query: str, *, limit: int = 5) -> tuple[Paper, ...]:
        self.received_limit = limit
        return tuple(
            Paper(
                paper_id=f"2401.0000{index}v1",
                title=f"Paper {index}",
                authors=("Researcher",),
                abstract="A grounded abstract.",
                published_at=datetime(2024, 1, index, tzinfo=UTC),
                updated_at=datetime(2024, 1, index, tzinfo=UTC),
                paper_url=f"https://arxiv.org/abs/2401.0000{index}v1",
            )
            for index in range(1, 6)
        )


def test_pipeline_requests_five_and_writes_review(tmp_path: Path) -> None:
    settings = Settings(
        LLM_API_KEY=None,
        LLM_BASE_URL=None,
        LLM_MODEL=None,
        PAPER_AGENT_REVIEW_PROVIDER="template",
        PAPER_AGENT_OUTPUT_DIR=tmp_path,
    )
    searcher = FakeSearcher()
    pipeline = PaperSearchPipeline(
        searcher=searcher,
        review_service=ReviewService(settings),
        settings=settings,
    )

    result = pipeline.run("  agentic   RAG  ")

    assert searcher.received_limit == 5
    assert len(result.search.papers) == 5
    assert result.search.query == "agentic RAG"
    assert result.output_path is not None
    assert Path(result.output_path).read_text(encoding="utf-8") == result.markdown
