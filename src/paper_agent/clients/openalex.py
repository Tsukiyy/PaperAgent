"""OpenAlex Works API 客户端。"""

from datetime import UTC, datetime

import httpx

from paper_agent.exceptions import SearchError
from paper_agent.http_client import create_http_client
from paper_agent.models import Paper

OPENALEX_WORKS_URL = "https://api.openalex.org/works"


class OpenAlexClient:
    """检索 OpenAlex，并将倒排摘要等字段转换为统一论文模型。"""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        http_client: httpx.Client | None = None,
        proxy: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._owns_client = http_client is None
        self._client = http_client or create_http_client(
            timeout=timeout,
            proxy=proxy,
            headers={"User-Agent": "PaperAgent/0.3 (academic research assistant)"},
        )

    def __enter__(self) -> "OpenAlexClient":
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
        params: dict[str, object] = {"search": clean_query, "per-page": limit}
        if self._api_key:
            params["api_key"] = self._api_key
        try:
            response = self._client.get(OPENALEX_WORKS_URL, params=params)
            response.raise_for_status()
            records = response.json().get("results", [])
            return tuple(
                self._parse_work(record, index, len(records))
                for index, record in enumerate(records)
            )
        except (httpx.HTTPError, TypeError, ValueError, KeyError) as exc:
            raise SearchError(f"OpenAlex 检索失败：{exc}") from exc

    @staticmethod
    def _abstract(inverted: object) -> str:
        if not isinstance(inverted, dict):
            return ""
        positioned: list[tuple[int, str]] = []
        for word, positions in inverted.items():
            if isinstance(positions, list):
                positioned.extend((int(position), str(word)) for position in positions)
        return " ".join(word for _, word in sorted(positioned))

    @staticmethod
    def _location_url(location: object, field: str) -> str | None:
        if isinstance(location, dict) and location.get(field):
            return str(location[field])
        return None

    @classmethod
    def _parse_work(cls, record: dict[str, object], index: int, total: int) -> Paper:
        openalex_id = str(record["id"]).rstrip("/").rsplit("/", maxsplit=1)[-1]
        date_text = record.get("publication_date")
        year = int(record.get("publication_year") or 1970)
        published = (
            datetime.fromisoformat(str(date_text)).replace(tzinfo=UTC)
            if date_text
            else datetime(year, 1, 1, tzinfo=UTC)
        )
        best_oa = record.get("best_oa_location") or {}
        primary = record.get("primary_location") or {}
        pdf_url = cls._location_url(best_oa, "pdf_url") or cls._location_url(primary, "pdf_url")
        landing = cls._location_url(primary, "landing_page_url")
        doi_url = str(record.get("doi") or "")
        doi = doi_url.removeprefix("https://doi.org/") or None
        authors = tuple(
            str(authorship.get("author", {}).get("display_name"))
            for authorship in (record.get("authorships") or [])
            if isinstance(authorship, dict)
            and isinstance(authorship.get("author"), dict)
            and authorship["author"].get("display_name")
        )
        ids = record.get("ids") or {}
        external_ids = {
            str(key): str(value)
            for key, value in ids.items()
            if value
        } if isinstance(ids, dict) else {}
        open_access = record.get("open_access") or {}
        is_oa = bool(open_access.get("is_oa")) if isinstance(open_access, dict) else bool(pdf_url)
        relevance = float(record.get("relevance_score") or (1.0 - index / max(total, 1)))
        return Paper(
            paper_id=openalex_id,
            title=str(record.get("display_name") or record.get("title") or "Untitled"),
            authors=authors,
            abstract=cls._abstract(record.get("abstract_inverted_index")),
            published_at=published,
            updated_at=published,
            paper_url=landing or doi_url or str(record["id"]),
            pdf_url=pdf_url,
            source="OpenAlex",
            sources=("OpenAlex",),
            doi=doi,
            external_ids=external_ids,
            citation_count=int(record.get("cited_by_count") or 0),
            reference_count=int(record.get("referenced_works_count") or 0),
            is_open_access=is_oa,
            relevance_score=relevance,
        )
