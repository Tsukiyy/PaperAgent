"""Step 2 带 page/section 证据的 Markdown 输出。"""

from datetime import UTC, datetime

from paper_agent.models import AnalysisFailure, PaperEvidence, SearchResult


def render_evidence_markdown(
    search: SearchResult,
    analyses: tuple[PaperEvidence, ...],
    failures: tuple[AnalysisFailure, ...],
    *,
    generated_at: datetime | None = None,
) -> str:
    """渲染结论，并由代码为每条结论附加页码与章节。"""

    timestamp = generated_at or datetime.now(UTC)
    lines = [
        f"# {search.query}：PDF 证据综述",
        "",
        f"> 数据源：{search.source}；检索论文：{len(search.papers)}；"
        f"成功解析：{len(analyses)}；失败：{len(failures)}；"
        f"时间：{timestamp.isoformat(timespec='seconds')}",
        "",
        "## 带证据结论",
        "",
    ]
    if not analyses:
        lines.append("没有论文完成 PDF 解析，因此未生成结论。")

    for analysis in analyses:
        provider = (
            analysis.provider.upper() if analysis.provider != "template" else "本地抽取"
        )
        lines.extend(
            [
                f"### [{analysis.paper.title}]({analysis.paper.paper_url})",
                "",
                f"> arXiv ID：`{analysis.paper.paper_id}`；PDF 页数："
                f"{analysis.document.page_count}；结论生成：{provider}",
                "",
            ]
        )
        for claim in analysis.claims:
            citations = "; ".join(
                f"p. {reference.page_number}, § {reference.section_title}"
                for reference in claim.evidence
            )
            lines.extend([f"- **结论：** {claim.claim} `[{citations}]`", ""])
            for reference in claim.evidence:
                lines.extend(
                    [
                        f"  > p. {reference.page_number} · {reference.section_title}："
                        f"{reference.quote}",
                        "",
                    ]
                )

    if failures:
        lines.extend(["## 未完成的论文", ""])
        for failure in failures:
            lines.append(
                f"- **{failure.title}**（`{failure.paper_id}`）："
                f"{failure.stage} 阶段失败——{failure.reason}"
            )
        lines.append("")

    lines.extend(
        [
            "## 说明",
            "",
            "- 页码从 PDF 第一页开始按 1 计数。",
            "- section 由字体、编号和常见论文标题规则识别，复杂双栏或扫描 PDF 可能需要人工核验。",
            "- 每条结论至少绑定一个原文 quote；模型模式下 quote 还会经过原文包含校验。",
            "",
        ]
    )
    return "\n".join(lines)
