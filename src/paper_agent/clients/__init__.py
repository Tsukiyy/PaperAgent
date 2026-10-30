"""论文元数据数据源客户端。"""

from paper_agent.clients.arxiv import ArxivClient
from paper_agent.clients.openalex import OpenAlexClient
from paper_agent.clients.semantic_scholar import SemanticScholarClient

__all__ = ["ArxivClient", "OpenAlexClient", "SemanticScholarClient"]
