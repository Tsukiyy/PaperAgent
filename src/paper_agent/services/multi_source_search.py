"""多源论文检索、去重与统一排序。"""

import math
import re
from dataclasses import dataclass
from typing import Protocol

from paper_agent.exceptions import SearchError
from paper_agent.models import Paper

WORD = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
NON_ALNUM = re.compile(r"[^a-z0-9\u4e00-\u9fff]+")


class SearchClient(Protocol):
    def search(self, query: str, *, limit: int = 5) -> tuple[Paper, ...]: ...


@dataclass(frozen=True)
class SourceIssue:
    source: str
    reason: str


def _normalized_title(title: str) -> str:
    return NON_ALNUM.sub("", title.casefold())


def _normalized_author_signature(authors: tuple[str, ...]) -> str:
    """生成跨来源稳定的作者签名，忽略姓名顺序、大小写和标点差异。

    不使用发表年份，使预印本与正式出版版本可以按标题和作者合并。
    取每个作者的最后一个词作为姓氏，兼容 ``L. Zhang`` 与 ``Lei Zhang``。
    """

    surnames = {
        WORD.findall(author.casefold())[-1]
        for author in authors
        if author and WORD.findall(author.casefold())
    }
    return "|".join(sorted(surnames))

# 去重标识
def _identifiers(paper: Paper) -> set[str]:
    # 标识：标准化标题 + 作者签名；不再依赖发表年份。
    title_key = _normalized_title(paper.title)
    author_key = _normalized_author_signature(paper.authors)
    identifiers = {f"title-authors:{title_key}:{author_key}"}
    if paper.doi:
        identifiers.add(f"doi:{paper.doi.casefold().removeprefix('https://doi.org/')}")
    for key, value in paper.external_ids.items():
        if key.casefold() in {"arxiv", "doi"}:
            normalized = value.casefold().removeprefix("https://doi.org/")
            identifiers.add(f"{key.casefold()}:{normalized}")
    return identifiers


def _merge(left: Paper, right: Paper) -> Paper:
    """合并重复记录，优先保留更完整摘要和公开 PDF。"""

    primary = left
    if right.source == "arXiv" and left.source != "arXiv":
        primary = right
    abstract = max((left.abstract, right.abstract), key=len)
    authors = left.authors if len(left.authors) >= len(right.authors) else right.authors
    return primary.model_copy(
        update={
            "authors": authors,
            "abstract": abstract,
            "pdf_url": left.pdf_url or right.pdf_url,
            "doi": left.doi or right.doi,
            "sources": tuple(dict.fromkeys((*left.sources, *right.sources))),
            "external_ids": {**left.external_ids, **right.external_ids},
            "citation_count": max(left.citation_count, right.citation_count),
            "reference_count": max(left.reference_count, right.reference_count),
            "is_open_access": left.is_open_access or right.is_open_access,
            "relevance_score": max(left.relevance_score, right.relevance_score),
        }
    )

# 文献去重
def deduplicate_papers(papers: list[Paper]) -> tuple[Paper, ...]:
    unique: list[Paper] = []
    keys: list[set[str]] = []
    for paper in papers:
        paper_keys = _identifiers(paper)
        match = next((index for index, known in enumerate(keys) if known & paper_keys), None)
        if match is None:
            unique.append(paper)
            keys.append(paper_keys)
            continue
        # 合并元数据
        unique[match] = _merge(unique[match], paper)
        keys[match].update(paper_keys)
    return tuple(unique)

# 检索文献重排
def rank_papers(query: str, papers: tuple[Paper, ...]) -> tuple[Paper, ...]:
    query_terms = set(WORD.findall(query.casefold()))
    max_citations = max((paper.citation_count for paper in papers), default=0)
    ranked: list[Paper] = []
    for paper in papers:
        text_terms = set(WORD.findall(f"{paper.title} {paper.abstract}".casefold()))
        lexical = len(query_terms & text_terms) / max(len(query_terms), 1)
        citation = math.log1p(paper.citation_count) / max(math.log1p(max_citations), 1.0)
        source_bonus = min(len(paper.sources), 3) / 3
        score = (
            0.55 * lexical
            + 0.20 * min(max(paper.relevance_score, 0.0), 1.0)
            + 0.15 * citation
            + 0.05 * float(paper.is_open_access)
            + 0.05 * source_bonus
        )
        ranked.append(paper.model_copy(update={"rank_score": round(score, 6)}))
    return tuple(
        sorted(
            ranked,
            key=lambda paper: (paper.rank_score, paper.citation_count, paper.published_at),
            reverse=True,
        )
    )


class MultiSourceSearchService:
    """并列调用数据源；单源故障不会让整个检索失败。"""

    def __init__(self, clients: dict[str, SearchClient]) -> None:
        self._clients = clients

    def search(
        self,
        query: str,
        *,
        limit: int,
    ) -> tuple[tuple[Paper, ...], tuple[SourceIssue, ...], int]:
        raw: list[Paper] = []  # 原始文献
        issues: list[SourceIssue] = []  # 来源
        fetch_limit = max(limit * 2, 5)  # 检索文件数
        for source, client in self._clients.items():
            try:
                source_limit = min(fetch_limit, 5) if source == "arXiv" else fetch_limit
                found = client.search(query, limit=source_limit)
                count = max(len(found), 1)
                raw.extend(
                    # 根据每个来源的排序赋予relevance_score
                    paper.model_copy(update={"relevance_score": 1.0 - index / count})
                    for index, paper in enumerate(found)
                )
            # 单个来源失败不会中断
            except (SearchError, ValueError) as exc:
                issues.append(SourceIssue(source=source, reason=str(exc)))
        if not raw:
            details = "; ".join(f"{issue.source}: {issue.reason}" for issue in issues)
            raise SearchError(f"所有论文数据源均检索失败。{details}")
        # 去重
        deduplicated = deduplicate_papers(raw)
        return rank_papers(query, deduplicated)[:limit], tuple(issues), len(raw)
