"""arXiv API 客户端。"""

import re
from datetime import datetime
from xml.etree import ElementTree

import httpx

from paper_agent.exceptions import SearchError
from paper_agent.models import Paper

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NAMESPACE = "http://www.w3.org/2005/Atom"
ARXIV_NAMESPACE = "http://arxiv.org/schemas/atom"
WHITESPACE_PATTERN = re.compile(r"\s+")


class ArxivClient:
    """检索 arXiv 并将 Atom XML 转为领域模型。"""

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "PaperAgent/0.1 (academic research assistant)"},
        )

    def __enter__(self) -> "ArxivClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        """只关闭由本实例创建的 HTTP 客户端。"""

        if self._owns_client:
            self._client.close()

    def search(self, query: str, *, limit: int = 5) -> tuple[Paper, ...]:
        """按相关性检索论文；Step 1 最多允许返回 5 篇。"""

        clean_query = WHITESPACE_PATTERN.sub(" ", query).strip()
        if not clean_query:
            raise ValueError("检索 query 不能为空。")
        if not 1 <= limit <= 5:
            raise ValueError("Step 1 的 limit 必须在 1 到 5 之间。")

        params = {
            "search_query": f"all:{clean_query}",
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",  # 相似度
            "sortOrder": "descending",  # 由高到低
        }
        try:
            # ARXIV_API_URL为ArXivClient请求地址，参数
            response = self._client.get(ARXIV_API_URL, params=params)
            response.raise_for_status()
        #     response结果是Atom XML格式
        except httpx.HTTPError as exc:
            raise SearchError(f"arXiv 检索失败：{exc}") from exc

        # 将每条<entry> 转换为统一的 Paper 对象
        return self._parse_feed(response.content)[:limit]

    @staticmethod
    def _parse_feed(content: bytes) -> tuple[Paper, ...]:
        namespace = {"atom": ATOM_NAMESPACE, "arxiv": ARXIV_NAMESPACE}
        try:
            root = ElementTree.fromstring(content)
            papers = tuple(
                # 将每条entry转换为Paper（Paper的基本信息）
                ArxivClient._parse_entry(entry, namespace)
                for entry in root.findall("atom:entry", namespace)
            )
        except (ElementTree.ParseError, KeyError, TypeError, ValueError) as exc:
            raise SearchError(f"无法解析 arXiv 响应：{exc}") from exc
        return papers  # 包含paper的基本信息

    @staticmethod
    def _parse_entry(entry: ElementTree.Element, namespace: dict[str, str]) -> Paper:
        def required_text(path: str) -> str:
            node = entry.find(path, namespace)
            if node is None or not node.text:
                raise ValueError(f"arXiv 条目缺少字段：{path}")
            return WHITESPACE_PATTERN.sub(" ", node.text).strip()

        entry_url = required_text("atom:id")
        paper_id = entry_url.rstrip("/").rsplit("/", maxsplit=1)[-1]
        authors = tuple(
            WHITESPACE_PATTERN.sub(" ", author.text or "").strip()
            for author in entry.findall("atom:author/atom:name", namespace)
            if author.text and author.text.strip()
        )
        links = {
            link.attrib.get("rel", "alternate"): link.attrib.get("href", "")
            for link in entry.findall("atom:link", namespace)
        }
        pdf_link = next(
            (
                link.attrib.get("href")
                for link in entry.findall("atom:link", namespace)
                if link.attrib.get("type") == "application/pdf"
            ),
            None,
        )
        category_nodes = entry.findall("atom:category", namespace)
        categories = tuple(
            node.attrib["term"] for node in category_nodes if node.attrib.get("term")
        )
        primary_node = entry.find("arxiv:primary_category", namespace)

        return Paper(
            paper_id=paper_id,
            title=required_text("atom:title"),
            authors=authors,
            abstract=required_text("atom:summary"),
            published_at=datetime.fromisoformat(
                required_text("atom:published").replace("Z", "+00:00")
            ),
            updated_at=datetime.fromisoformat(required_text("atom:updated").replace("Z", "+00:00")),
            paper_url=links.get("alternate") or entry_url,
            pdf_url=pdf_link,
            categories=categories,
            primary_category=primary_node.attrib.get("term") if primary_node is not None else None,
            source="arXiv",
            sources=("arXiv",),
            external_ids={"ArXiv": paper_id},
            is_open_access=pdf_link is not None,
        )
