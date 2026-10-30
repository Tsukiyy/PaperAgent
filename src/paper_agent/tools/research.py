"""LangGraph 工作流可调用的六个研究工具。"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from paper_agent.evidence.extractor import EvidenceService
from paper_agent.exceptions import EvidenceError, PdfDownloadError, PdfParseError
from paper_agent.models import (
    AnalysisFailure,
    ComparisonRow,
    DownloadedPdf,
    Paper,
    PaperEvidence,
    ParsedDocument,
)
from paper_agent.services.multi_source_search import MultiSourceSearchService, SourceIssue
from paper_agent.services.synthesis import ClaimSynthesisService
from paper_agent.services.trace import RunTrace
from paper_agent.storage.database import SQLiteStore
from paper_agent.storage.vector_index import FaissVectorIndex
from paper_agent.tools.pdf import PdfDownloader, PyMuPdfParser

EVIDENCE_QUERIES = (
    "contribution novelty objective survey taxonomy proposed introduced",
    "method approach framework architecture pipeline workflow algorithm agents",
    "dataset corpus benchmark evaluation set participants knowledge base",
    "metric measure accuracy precision recall score latency cost evaluation",
    "result experiment finding performance outperform improvement comparison",
    "limitation challenge drawback risk overhead constraint future work",
)


@dataclass
class ResearchToolContext:
    """六个工具共享的显式依赖，便于替换实现与单元测试。"""

    search_service: MultiSourceSearchService
    downloader: PdfDownloader
    parser: PyMuPdfParser
    evidence_service: EvidenceService
    store: SQLiteStore
    vector_index: FaissVectorIndex
    trace: RunTrace
    pdf_dir: Path
    run_dir: Path
    synthesis_service: ClaimSynthesisService | None = None


def search_papers(
    query: str,
    *,
    limit: int,
    run_id: str,
    context: ResearchToolContext,
) -> tuple[tuple[Paper, ...], tuple[SourceIssue, ...], int]:
    """从三个数据源检索后去重、排序并入库。"""

    papers, issues, raw_count = context.search_service.search(query, limit=limit)
    context.store.save_papers(run_id, papers)
    return papers, issues, raw_count


def download_pdf(
    papers: tuple[Paper, ...],
    *,
    read_limit: int,
    context: ResearchToolContext,
) -> tuple[tuple[DownloadedPdf, ...], tuple[AnalysisFailure, ...]]:
    """按排序逐篇尝试下载，直到成功达到 read_limit 或候选论文耗尽。"""

    downloaded: list[DownloadedPdf] = []
    failures: list[AnalysisFailure] = []
    # 失败论文不占用精读名额，继续尝试后续候选论文。
    for paper in papers:
        if len(downloaded) >= read_limit:
            break
        try:
            # PdfDownloader(tools/pdf)，pdf保存在pdf_dir目录中
            pdf = context.downloader.download(paper, context.pdf_dir)
            # SQLite保存pdf元数据
            context.store.save_pdf(pdf)
            downloaded.append(pdf)
        # 单篇论文下载失败，产生AnalysisFailure
        except PdfDownloadError as exc:
            failures.append(_failure(paper, "download", exc))
    return tuple(downloaded), tuple(failures)


def parse_pdf(
    pdfs: tuple[DownloadedPdf, ...],
    *,
    context: ResearchToolContext,
) -> tuple[tuple[ParsedDocument, ...], tuple[AnalysisFailure, ...]]:
    """解析 PDF 为带 page/section 的文本块，并建立 FAISS 索引。"""

    documents: list[ParsedDocument] = []
    failures: list[AnalysisFailure] = []
    for pdf in pdfs:
        try:
            # 使用PyMuPdParser解析器
            document = context.parser.parse(pdf)
            # SQLite存储pdf解析结果
            context.store.save_document(document)
            documents.append(document)
        except PdfParseError as exc:
            failures.append(
                AnalysisFailure(
                    paper_id=pdf.paper_id,
                    title=pdf.paper_id,
                    stage="parse",
                    reason=str(exc),
                )
            )
    # 汇总所有论文的文本块，paperID对应所有的chunks
    records = [
        (document.paper_id, chunk)
        for document in documents
        for chunk in document.chunks
    ]
    # 构建索引，FAISS保存chunk向量
    context.vector_index.build(records)
    return tuple(documents), tuple(failures)


def extract_claims(
    query: str,
    papers: tuple[Paper, ...],
    pdfs: tuple[DownloadedPdf, ...],
    documents: tuple[ParsedDocument, ...],
    *,
    run_id: str,
    context: ResearchToolContext,
    expanded: bool = False,
) -> tuple[tuple[PaperEvidence, ...], tuple[AnalysisFailure, ...]]:
    """先用 FAISS 召回相关块，再提取并校验每条 claim 的证据。"""

    # 建立论文和PDF映射
    paper_map = {paper.paper_id: paper for paper in papers}
    pdf_map = {pdf.paper_id: pdf for pdf in pdfs}
    analyses: list[PaperEvidence] = []
    failures: list[AnalysisFailure] = []
    for document in documents:
        paper = paper_map[document.paper_id]
        try:
            # 六个维度分别召回，避免一个宽泛 query 只命中论文中的少数主题。
            per_dimension = 8 if expanded else 4
            selected_ids = {
                chunk_id
                for evidence_query in EVIDENCE_QUERIES
                for chunk_id in context.vector_index.search(
                    f"{query} {evidence_query}",
                    limit=per_dimension,
                    paper_id=document.paper_id,
                )
            }
            selected = tuple(
                chunk for chunk in document.chunks if chunk.chunk_id in selected_ids
            )
            # 创建“聚焦版文档”，若FAISS命中则把相关chunk交给claim，未命中回退到文档全部chunk
            focused = document.model_copy(update={"chunks": selected or document.chunks})
            # 提取claims，evidence_service支持模板构造或openai生成
            claims, provider = context.evidence_service.extract(paper, focused)
            # 生成PaperEvidence
            analysis = PaperEvidence(
                paper=paper,
                pdf=pdf_map[document.paper_id],  # 下载的PDF信息
                document=document,  # 解析后的完整文档
                claims=claims,  # 提取的claims
                provider=provider,  # 提取方式
            )
            # 保存到SQLite
            context.store.save_analysis(run_id, analysis)
            analyses.append(analysis)
        except EvidenceError as exc:
            failures.append(_failure(paper, "evidence", exc))
    return tuple(analyses), tuple(failures)


def compare_papers(
    analyses: tuple[PaperEvidence, ...],
    *,
    run_id: str,
    context: ResearchToolContext,
) -> tuple[ComparisonRow, ...]:
    """按贡献、方法、数据集、指标、结果和局限聚合 claims。"""

    column_by_type = {
        "contribution": "contributions",
        "method": "methods",
        "dataset": "datasets",
        "metric": "metrics",
        "result": "results",
        "limitation": "limitations",
    }
    rows: list[ComparisonRow] = []
    # 一篇论文生成一行
    for analysis in analyses:
        values: dict[str, list[str]] = {name: [] for name in column_by_type.values()}
        locations: list[str] = []
        for claim in analysis.claims:
            # 获取claim_type如method，放入methods
            column = column_by_type.get(claim.claim_type)
            if column:
                values[column].append(claim.claim)
            locations.extend(
                f"p.{ref.page_number} §{ref.section_title}" for ref in claim.evidence
            )
        rows.append(
            ComparisonRow(
                paper_id=analysis.paper.paper_id,
                title=analysis.paper.title,
                paper_url=analysis.paper.paper_url,
                **{name: " | ".join(items) for name, items in values.items()},
                evidence_locations=" | ".join(dict.fromkeys(locations)),
            )
        )
    result = tuple(rows)
    context.store.save_comparisons(run_id, result)
    return result


def write_review(
    *,
    query: str,
    run_id: str,
    papers: tuple[Paper, ...],
    analyses: tuple[PaperEvidence, ...],
    comparisons: tuple[ComparisonRow, ...],
    failures: tuple[AnalysisFailure, ...],
    source_issues: tuple[SourceIssue, ...],
    raw_count: int,
    review_passed: bool,
    synthesis: str,
    context: ResearchToolContext,
) -> dict[str, str]:
    """输出综述、BibTeX、证据、对比表、追踪和评测摘要。"""

    context.run_dir.mkdir(parents=True, exist_ok=False)
    _write_markdown(
        context.run_dir / "review.md",
        query,
        papers,
        analyses,
        failures,
        source_issues,
        raw_count,
        synthesis,
    )
    (context.run_dir / "papers.bib").write_text(_bibtex(papers), encoding="utf-8")
    _write_evidence(context.run_dir / "evidence.jsonl", run_id, analyses)
    _write_comparison(context.run_dir / "comparison.csv", comparisons)
    _write_eval(
        context.run_dir / "eval_summary.md",
        run_id,
        papers,
        analyses,
        failures,
        source_issues,
        raw_count,
        review_passed,
    )
    # trace 在图执行完毕后重写一次，确保包含 write_review 节点自身。
    context.trace.write(context.run_dir / "trace.jsonl")
    return {
        name: str((context.run_dir / name).resolve())
        for name in (
            "review.md",
            "papers.bib",
            "evidence.jsonl",
            "comparison.csv",
            "trace.jsonl",
            "eval_summary.md",
        )
    }


def _failure(
    paper: Paper,
    stage: Literal["download", "parse", "evidence"],
    error: Exception,
) -> AnalysisFailure:
    return AnalysisFailure(
        paper_id=paper.paper_id,
        title=paper.title,
        stage=stage,
        reason=str(error),
    )


def _write_markdown(
    path: Path,
    query: str,
    papers: tuple[Paper, ...],
    analyses: tuple[PaperEvidence, ...],
    failures: tuple[AnalysisFailure, ...],
    source_issues: tuple[SourceIssue, ...],
    raw_count: int,
    synthesis: str,
) -> None:
    lines = [
        f"# {query}：多源论文证据综述",
        "",
        f"> 共检索 {raw_count} 条，去重排序后保留 {len(papers)} 篇，"
        f"完成 PDF 证据分析 {len(analyses)} 篇。",
        "",
        "## 检索结果",
        "",
    ]
    for index, paper in enumerate(papers, start=1):
        sources = ", ".join(paper.sources)
        lines.append(
            f"{index}. [{paper.title}]({paper.paper_url}) — {sources}；"
            f"rank={paper.rank_score:.3f}；引用数={paper.citation_count}"
        )
    lines.extend(["", "## 综合分析", "", synthesis, "", "## 带证据结论", ""])
    if not analyses:
        lines.append("未能从公开 PDF 形成可验证结论；以下输出不会补造论文结论。")
    for analysis in analyses:
        lines.extend([f"### [{analysis.paper.title}]({analysis.paper.paper_url})", ""])
        for claim in analysis.claims:
            citations = "; ".join(
                f"p.{ref.page_number} §{ref.section_title}" for ref in claim.evidence
            )
            lines.extend([f"- **{claim.claim_type}**：{claim.claim} `[{citations}]`", ""])
    if failures or source_issues:
        lines.extend(["## 限制与失败", ""])
        lines.extend(f"- {issue.source}：{issue.reason}" for issue in source_issues)
        lines.extend(
            f"- {failure.title}（{failure.stage}）：{failure.reason}" for failure in failures
        )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bibtex(papers: tuple[Paper, ...]) -> str:
    entries: list[str] = []
    for paper in papers:
        surname = (paper.authors[0].split()[-1] if paper.authors else "unknown").lower()
        safe = "".join(character for character in surname if character.isalnum()) or "paper"
        key = f"{safe}{paper.published_at.year}{paper.paper_id[-6:].replace('.', '')}"
        fields = [
            f"  title = {{{paper.title}}}",
            f"  author = {{{' and '.join(paper.authors)}}}",
            f"  year = {{{paper.published_at.year}}}",
            f"  url = {{{paper.paper_url}}}",
        ]
        if paper.doi:
            fields.append(f"  doi = {{{paper.doi}}}")
        entries.append(f"@article{{{key},\n" + ",\n".join(fields) + "\n}")
    return "\n\n".join(entries) + ("\n" if entries else "")


def _write_evidence(path: Path, run_id: str, analyses: tuple[PaperEvidence, ...]) -> None:
    records = [
        {
            "run_id": run_id,
            "paper_id": analysis.paper.paper_id,
            "title": analysis.paper.title,
            "paper_url": str(analysis.paper.paper_url),
            "provider": analysis.provider,
            **claim.model_dump(mode="json"),
        }
        for analysis in analyses
        for claim in analysis.claims
    ]
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_comparison(path: Path, rows: tuple[ComparisonRow, ...]) -> None:
    fieldnames = list(ComparisonRow.model_fields)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row.model_dump(mode="json") for row in rows)


def _write_eval(
    path: Path,
    run_id: str,
    papers: tuple[Paper, ...],
    analyses: tuple[PaperEvidence, ...],
    failures: tuple[AnalysisFailure, ...],
    source_issues: tuple[SourceIssue, ...],
    raw_count: int,
    review_passed: bool,
) -> None:
    citation_count = sum(len(claim.evidence) for item in analyses for claim in item.claims)
    claim_count = sum(len(item.claims) for item in analyses)
    supported_claim_count = 0
    for analysis in analyses:
        chunks = {chunk.chunk_id: chunk for chunk in analysis.document.chunks}
        for claim in analysis.claims:
            has_valid_evidence = any(
                ref.quote.strip()
                and ref.chunk_id in chunks
                and ref.page_number == chunks[ref.chunk_id].page_number
                and ref.section_title == chunks[ref.chunk_id].section_title
                and " ".join(ref.quote.split()).casefold()
                in " ".join(chunks[ref.chunk_id].text.split()).casefold()
                for ref in claim.evidence
            )
            supported_claim_count += int(has_valid_evidence)
    coverage = f"{supported_claim_count / claim_count:.2%}" if claim_count else "N/A"
    lines = [
        "# 运行评测摘要",
        "",
        f"- run_id：`{run_id}`",
        f"- 状态：{'通过' if review_passed else '证据不足'}",
        f"- 原始检索数：{raw_count}",
        f"- 去重后论文数：{len(papers)}",
        f"- 成功解析论文数：{len(analyses)}",
        f"- claims 数：{claim_count}",
        f"- 证据引用数：{citation_count}",
        f"- 至少有一条有效证据的 claim 数：{supported_claim_count}",
        f"- claim 证据覆盖率：{coverage}",
        f"- 数据源失败数：{len(source_issues)}",
        f"- 论文处理失败数：{len(failures)}",
        "",
        "> 这是单次运行的结构性检查，不替代后续 20 条固定 query 的完整评测集。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
