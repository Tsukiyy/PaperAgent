from pathlib import Path

from paper_agent.models import DocumentChunk
from paper_agent.storage.vector_index import FaissVectorIndex


def _chunk(index: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"p1-c{index}",
        page_number=1,
        section_title="Methods",
        text=text,
    )


def test_paper_filter_is_applied_before_result_limit(tmp_path: Path) -> None:
    distractors = [
        ("other", _chunk(index, "agentic retrieval method framework"))
        for index in range(30)
    ]
    targets = [
        ("target", _chunk(100 + index, f"target evidence section {index}"))
        for index in range(3)
    ]
    index = FaissVectorIndex(tmp_path / "faiss")
    index.build([*distractors, *targets])

    matches = index.search("agentic retrieval method", limit=3, paper_id="target")

    assert set(matches) == {"p1-c100", "p1-c101", "p1-c102"}
