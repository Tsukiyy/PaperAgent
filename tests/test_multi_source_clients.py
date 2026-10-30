from datetime import UTC, datetime

import httpx

from paper_agent.clients.openalex import OpenAlexClient
from paper_agent.clients.semantic_scholar import SemanticScholarClient
from paper_agent.models import Paper
from paper_agent.services.multi_source_search import deduplicate_papers, rank_papers


def test_semantic_scholar_maps_open_pdf_and_doi() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["query"] == "agentic RAG"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "paperId": "s2-1",
                        "title": "Agentic RAG",
                        "abstract": "An agentic retrieval method.",
                        "authors": [{"name": "Ada Lovelace"}],
                        "year": 2025,
                        "publicationDate": "2025-01-02",
                        "url": "https://www.semanticscholar.org/paper/s2-1",
                        "externalIds": {"DOI": "10.1/example", "ArXiv": "2501.00001"},
                        "openAccessPdf": {"url": "https://example.org/paper.pdf"},
                        "citationCount": 12,
                        "referenceCount": 8,
                    }
                ]
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    paper = SemanticScholarClient(http_client=http_client).search("agentic RAG")[0]

    assert paper.doi == "10.1/example"
    assert str(paper.pdf_url) == "https://example.org/paper.pdf"
    assert paper.source == "Semantic Scholar"


def test_semantic_scholar_retries_rate_limit_response() -> None:
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, json={"data": []})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    papers = SemanticScholarClient(http_client=http_client).search("agentic RAG")

    assert papers == ()
    assert attempts == 3


def test_semantic_scholar_falls_back_to_bulk_without_api_key() -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        assert request.headers["user-agent"] == "PaperAgent/0.3 (Contact: test@example.com)"
        if request.url.path.endswith("/paper/search"):
            return httpx.Response(429)
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "paperId": "less-relevant",
                        "title": "Retrieval Systems",
                        "abstract": "A general retrieval paper mentioning agentic RAG.",
                        "year": 2024,
                        "citationCount": 100,
                    },
                    {
                        "paperId": "exact-match",
                        "title": "Agentic RAG: A Survey",
                        "abstract": "A survey of agentic retrieval augmented generation.",
                        "year": 2025,
                        "citationCount": 5,
                    },
                ]
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    papers = SemanticScholarClient(
        http_client=http_client,
        max_retries=0,
        user_agent="PaperAgent/0.3 (Contact: test@example.com)",
    ).search("agentic RAG", limit=1)

    assert papers[0].paper_id == "exact-match"
    assert requested_paths == [
        "/graph/v1/paper/search",
        "/graph/v1/paper/search/bulk",
    ]


def test_openalex_reconstructs_abstract() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": "https://openalex.org/W1",
                        "display_name": "Retrieval Agents",
                        "publication_year": 2024,
                        "publication_date": "2024-05-01",
                        "doi": "https://doi.org/10.2/example",
                        "authorships": [{"author": {"display_name": "Grace Hopper"}}],
                        "abstract_inverted_index": {"Retrieval": [0], "works": [1]},
                        "primary_location": {
                            "landing_page_url": "https://example.org/work",
                            "pdf_url": None,
                        },
                        "best_oa_location": {"pdf_url": "https://example.org/work.pdf"},
                        "open_access": {"is_oa": True},
                        "cited_by_count": 4,
                        "referenced_works_count": 3,
                        "ids": {"openalex": "https://openalex.org/W1"},
                        "relevance_score": 10.0,
                    }
                ]
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    paper = OpenAlexClient(http_client=http_client).search("retrieval")[0]

    assert paper.abstract == "Retrieval works"
    assert paper.doi == "10.2/example"
    assert paper.is_open_access is True


def _paper(paper_id: str, source: str, *, doi: str | None, citations: int) -> Paper:
    now = datetime(2025, 1, 1, tzinfo=UTC)
    return Paper(
        paper_id=paper_id,
        title="Agentic Retrieval Augmented Generation",
        authors=("Researcher",),
        abstract="agentic RAG retrieval planning",
        published_at=now,
        updated_at=now,
        paper_url=f"https://example.org/{paper_id}",
        source=source,
        sources=(source,),
        doi=doi,
        citation_count=citations,
        relevance_score=1.0,
    )


def test_deduplicate_merges_sources_then_rank() -> None:
    semantic = _paper("s2", "Semantic Scholar", doi="10.1/shared", citations=10)
    arxiv = _paper("2501.1", "arXiv", doi="10.1/shared", citations=3)

    unique = deduplicate_papers([semantic, arxiv])
    ranked = rank_papers("agentic RAG", unique)

    assert len(ranked) == 1
    assert ranked[0].paper_id == "2501.1"
    assert set(ranked[0].sources) == {"arXiv", "Semantic Scholar"}
    assert ranked[0].citation_count == 10
    assert ranked[0].rank_score > 0


def test_deduplicate_merges_preprint_and_published_version_by_title_authors() -> None:
    now = datetime(2022, 5, 26, tzinfo=UTC)
    preprint = Paper(
        paper_id="preprint",
        title="Are Transformers Effective for Time Series Forecasting?",
        authors=("Ailing Zeng", "Mu-Hwa Chen", "L. Zhang", "Qiang Xu"),
        abstract="same work",
        published_at=now,
        updated_at=now,
        paper_url="https://example.org/preprint",
        source="Semantic Scholar",
        sources=("Semantic Scholar",),
    )
    published = preprint.model_copy(
        update={
            "paper_id": "published",
            "authors": ("Ailing Zeng", "Muxi Chen", "Lei Zhang", "Qiang Xu"),
            "published_at": datetime(2023, 6, 26, tzinfo=UTC),
            "paper_url": "https://doi.org/10.1609/aaai.v37i9.26317",
            "source": "OpenAlex",
            "sources": ("OpenAlex",),
        }
    )

    unique = deduplicate_papers([preprint, published])

    assert len(unique) == 1
    assert set(unique[0].sources) == {"Semantic Scholar", "OpenAlex"}
