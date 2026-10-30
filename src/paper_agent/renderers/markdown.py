"""Step 1 Markdown 输出。"""

from datetime import UTC, datetime

from paper_agent.models import SearchResult


def render_review_markdown(
    search: SearchResult,
    review_body: str,
    provider: str,
    *,
    generated_at: datetime | None = None,
) -> str:
    """渲染综述，并以确定性代码保证每篇论文都有可点击链接。"""

    timestamp = generated_at or datetime.now(UTC)
    provider_label = provider.upper() if provider != "template" else "本地模板"
    lines = [
        f"# {search.query}：arXiv 简短综述",
        "",
        f"> 数据源：{search.source}；论文数：{len(search.papers)}；"
        f"生成方式：{provider_label}；时间：{timestamp.isoformat(timespec='seconds')}",
        "",
        "## 综述",
        "",
        review_body.strip(),
        "",
        "## 论文与链接",
        "",
    ]
    if not search.papers:
        lines.append("本次检索没有结果。")
    else:
        for index, paper in enumerate(search.papers, start=1):
            author_text = ", ".join(paper.authors) if paper.authors else "作者信息缺失"
            lines.extend(
                [
                    f"{index}. [{paper.title}]({paper.paper_url})",
                    f"   - 作者：{author_text}",
                    f"   - 发布：{paper.published_at.date().isoformat()}；"
                    f"arXiv ID：`{paper.paper_id}`",
                ]
            )
    lines.extend(
        [
            "",
            "## 限制",
            "",
            "当前结论仅基于 arXiv 元数据和摘要，尚未解析 PDF 全文；"
            "方法、实验与局限将在 Step 2 核验。",
            "",
        ]
    )
    return "\n".join(lines)
