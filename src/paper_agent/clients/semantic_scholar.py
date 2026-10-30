"""Semantic Scholar Academic Graph API 客户端。"""

import math
import re
import time
from datetime import UTC, datetime

import httpx

from paper_agent.exceptions import SearchError
from paper_agent.http_client import create_http_client
from paper_agent.models import Paper

SEMANTIC_SCHOLAR_SEARCH_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/search"
)
SEMANTIC_SCHOLAR_BULK_SEARCH_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
)
WORD = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
FIELDS = (
    "paperId,title,abstract,authors,year,publicationDate,url,externalIds,"
    "openAccessPdf,citationCount,referenceCount"
)


class SemanticScholarClient:
    """检索 Semantic Scholar，并将结果转换为统一论文模型。"""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        http_client: httpx.Client | None = None,
        max_retries: int = 2,
        user_agent: str = "PaperAgent/0.3 (academic research assistant)",
        proxy: str | None = None,
    ) -> None:
        self._owns_client = http_client is None
        self._has_api_key = bool(api_key)
        self._max_retries = max_retries
        headers = {"User-Agent": user_agent}
        if api_key:
            headers["x-api-key"] = api_key
        self._headers = headers
        self._client = http_client or create_http_client(
            timeout=timeout,
            proxy=proxy,
            headers=headers,
        )

    def __enter__(self) -> "SemanticScholarClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def search(self, query: str, *, limit: int = 5) -> tuple[Paper, ...]:
        clean_query = " ".join(query.split())
        if not clean_query:
            raise ValueError("检索 query 不能为空。")
        try:
            response = self._request_with_backoff(
                SEMANTIC_SCHOLAR_SEARCH_URL,
                params={"query": clean_query, "limit": limit, "fields": FIELDS},
            )
            bulk_fallback = response.status_code == 429 and not self._has_api_key
            if bulk_fallback:
                # 官方建议多数检索场景使用负载更低的 bulk 端点。匿名 relevance
                # 共享池持续限流时降级，不会额外绕过或并发冲击限流机制。
                response = self._request_with_backoff(
                    SEMANTIC_SCHOLAR_BULK_SEARCH_URL,
                    params={
                        "query": clean_query,
                        "fields": FIELDS,
                        "sort": "citationCount:desc",
                    },
                )
            response.raise_for_status()
            records = response.json().get("data", [])
            if bulk_fallback:
                records = self._rank_bulk_records(clean_query, records, limit=limit)
            else:
                records = records[:limit]
            return tuple(
                self._parse_paper(record, index, len(records))
                for index, record in enumerate(records)
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                access = "未配置 API Key 的匿名请求" if not self._has_api_key else "当前 API Key"
                raise SearchError(
                    f"Semantic Scholar 检索失败：HTTP 429，{access}已触发限流；"
                    "请稍后重试或配置/检查 SEMANTIC_SCHOLAR_API_KEY。"
                ) from exc
            raise SearchError(f"Semantic Scholar 检索失败：{exc}") from exc
        except (httpx.HTTPError, TypeError, ValueError, KeyError) as exc:
            raise SearchError(f"Semantic Scholar 检索失败：{exc}") from exc

    def _request_with_backoff(
        self,
        url: str,
        *,
        params: dict[str, object],
    ) -> httpx.Response:
        response: httpx.Response | None = None
        for attempt in range(self._max_retries + 1):
            response = self._client.get(url, params=params, headers=self._headers)
            if response.status_code != 429 or attempt >= self._max_retries:
                break
            retry_after = response.headers.get("retry-after", "")
            delay = (
                float(retry_after)
                if retry_after.replace(".", "", 1).isdigit()
                else 2.0**attempt
            )
            time.sleep(min(max(delay, 0.0), 8.0))
        if response is None:
            raise SearchError("Semantic Scholar 未返回响应。")
        return response

    @staticmethod
    def _rank_bulk_records(
        query: str,
        records: list[dict[str, object]],
        *,
        limit: int,
    ) -> list[dict[str, object]]:
        """bulk 不提供 relevance 排序，在本地优先题名匹配并兼顾摘要与引用量。"""

        query_terms = set(WORD.findall(query.casefold()))
        phrase = query.casefold()

        def score(record: dict[str, object]) -> tuple[float, int, int]:
            title = str(record.get("title") or "").casefold()
            abstract = str(record.get("abstract") or "").casefold()
            title_terms = set(WORD.findall(title))
            abstract_terms = set(WORD.findall(abstract))
            title_overlap = len(query_terms & title_terms) / max(len(query_terms), 1)
            abstract_overlap = len(query_terms & abstract_terms) / max(len(query_terms), 1)
            relevance = (
                2.0 * float(phrase in title)
                + title_overlap
                + 0.35 * abstract_overlap
            )
            citations = int(record.get("citationCount") or 0)
            year = int(record.get("year") or 0)
            return relevance, int(math.log1p(citations) * 1000), year

        return sorted(records, key=score, reverse=True)[:limit]

    @staticmethod
    def _parse_paper(record: dict[str, object], index: int, total: int) -> Paper:
        paper_id = str(record["paperId"])
        external_ids = {
            str(key): str(value)
            for key, value in (record.get("externalIds") or {}).items()
            if value
        }
        date_text = record.get("publicationDate")
        year = int(record.get("year") or 1970)
        published = (
            datetime.fromisoformat(str(date_text)).replace(tzinfo=UTC)
            if date_text
            else datetime(year, 1, 1, tzinfo=UTC)
        )
        oa_pdf = record.get("openAccessPdf") or {}
        pdf_url = oa_pdf.get("url") if isinstance(oa_pdf, dict) else None
        authors = tuple(
            str(author.get("name"))
            for author in (record.get("authors") or [])
            if isinstance(author, dict) and author.get("name")
        )
        relevance = 1.0 - (index / max(total, 1))
        return Paper(
            paper_id=paper_id,
            title=str(record.get("title") or "Untitled"),
            authors=authors,
            abstract=str(record.get("abstract") or ""),
            published_at=published,
            updated_at=published,
            paper_url=str(record.get("url") or f"https://www.semanticscholar.org/paper/{paper_id}"),
            pdf_url=str(pdf_url) if pdf_url else None,
            source="Semantic Scholar",
            sources=("Semantic Scholar",),
            doi=external_ids.get("DOI"),
            external_ids=external_ids,
            citation_count=int(record.get("citationCount") or 0),
            reference_count=int(record.get("referenceCount") or 0),
            is_open_access=bool(pdf_url),
            relevance_score=relevance,
        )
