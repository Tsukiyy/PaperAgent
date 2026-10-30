from pathlib import Path

import httpx
import pytest

from paper_agent.clients.arxiv import ArxivClient
from paper_agent.exceptions import SearchError

FIXTURE = Path(__file__).parent / "fixtures" / "arxiv_response.xml"


def test_search_builds_relevance_query_and_parses_papers() -> None:
    captured_request: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = request
        return httpx.Response(200, content=FIXTURE.read_bytes())

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = ArxivClient(http_client=http_client)
    papers = client.search("  agentic   RAG  ", limit=5)

    assert len(papers) == 2
    assert papers[0].paper_id == "2401.00001v1"
    assert papers[0].primary_category == "cs.AI"
    assert str(papers[0].pdf_url) == "https://arxiv.org/pdf/2401.00001v1"
    assert captured_request is not None
    assert captured_request.url.params["search_query"] == "all:agentic RAG"
    assert captured_request.url.params["max_results"] == "5"
    assert captured_request.url.params["sortBy"] == "relevance"
    http_client.close()


@pytest.mark.parametrize("limit", [0, 6])
def test_search_rejects_limit_outside_step_one_boundary(limit: int) -> None:
    client = ArxivClient(http_client=httpx.Client(transport=httpx.MockTransport(lambda _: None)))
    with pytest.raises(ValueError, match="1 到 5"):
        client.search("RAG", limit=limit)


def test_search_wraps_invalid_xml() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, content=b"not xml"))
    client = ArxivClient(http_client=httpx.Client(transport=transport))
    with pytest.raises(SearchError, match="无法解析"):
        client.search("RAG")

