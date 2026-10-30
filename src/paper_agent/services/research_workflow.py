"""基于 LangGraph 的多源论文研究工作流。"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from paper_agent.config import get_settings
from paper_agent.models import (
    AnalysisFailure,
    ComparisonRow,
    DownloadedPdf,
    Paper,
    PaperEvidence,
    ParsedDocument,
    ResearchRunResult,
)
from paper_agent.services.multi_source_search import SourceIssue
from paper_agent.services.synthesis import TemplateClaimSynthesizer
from paper_agent.services.review_validation import (
    ClaimsReviewValidator,
    ComparisonReviewValidator,
    ReviewReport,
    ReviewValidator,
    SynthesisReviewValidator,
    create_semantic_review_judge,
)
from paper_agent.tools.research import (
    ResearchToolContext,
    compare_papers,
    download_pdf,
    extract_claims,
    parse_pdf,
    search_papers,
    write_review,
)


class ResearchState(TypedDict, total=False):
    """图中节点共享的状态。"""

    run_id: str
    query: str
    limit: int
    read_limit: int
    task_type: str
    papers: tuple[Paper, ...]
    source_issues: tuple[SourceIssue, ...]
    raw_count: int
    pdfs: tuple[DownloadedPdf, ...]
    documents: tuple[ParsedDocument, ...]
    analyses: tuple[PaperEvidence, ...]
    comparisons: tuple[ComparisonRow, ...]
    failures: tuple[AnalysisFailure, ...]
    synthesis: str
    review_passed: bool
    review_notes: str
    review_report: ReviewReport
    claims_review_report: ReviewReport
    comparison_review_report: ReviewReport
    synthesis_review_report: ReviewReport
    review_attempts: int
    claims_review_attempts: int
    comparison_review_attempts: int
    synthesis_review_attempts: int
    outputs: dict[str, str]
    process: dict[str, str]


class ResearchWorkflow:
    """把六个工具编排为可检查、可重试、有边界的 LangGraph。"""

    def __init__(self, context: ResearchToolContext) -> None:
        # context: ResearchToolContext 依赖集合
        self._context = context
        self._semantic_review_judge = create_semantic_review_judge(get_settings())
        self.graph = self._build_graph()

    def _traced(self, name: str, kind: str, inputs: dict[str, object]):
        return self._context.trace.span(name, kind, inputs)

    def _build_graph(self):
        builder = StateGraph(ResearchState)
        builder.add_node("planner", self._planner)
        builder.add_node("search_papers", self._search)
        builder.add_node("download_pdf", self._download)
        builder.add_node("parse_pdf", self._parse)
        builder.add_node("extract_claims", self._extract)
        builder.add_node("compare_papers", self._compare)
        builder.add_node("synthesis_agent", self._synthesize)
        builder.add_node("review_claims", self._review_claims)
        builder.add_node("review_comparison", self._review_comparison)
        builder.add_node("review_synthesis", self._review_synthesis)
        builder.add_node("write_review", self._write)

        builder.add_edge(START, "planner")
        builder.add_edge("planner", "search_papers")
        builder.add_edge("search_papers", "download_pdf")
        builder.add_edge("download_pdf", "parse_pdf")
        builder.add_edge("parse_pdf", "extract_claims")
        builder.add_edge("extract_claims", "review_claims")
        builder.add_conditional_edges("review_claims", self._claims_review_route, {"retry": "extract_claims", "continue": "compare_papers"})
        builder.add_edge("compare_papers", "review_comparison")
        builder.add_conditional_edges("review_comparison", self._comparison_review_route, {"retry": "compare_papers", "continue": "synthesis_agent"})
        builder.add_edge("synthesis_agent", "review_synthesis")
        builder.add_conditional_edges("review_synthesis", self._synthesis_review_route, {"retry": "synthesis_agent", "write": "write_review"})
        builder.add_edge("write_review", END)
        return builder.compile()

    def _review_claims(self, state: ResearchState) -> dict[str, object]:
        with self._traced("review_claims", "agent", {"analysis_count": len(state["analyses"])}) as span:
            report = ClaimsReviewValidator(self._semantic_review_judge).validate(state["query"], state["analyses"])
            attempts = state.get("claims_review_attempts", 0) + (0 if report.passed else 1)
            span["outputs"] = {"passed": report.passed, "errors": len(report.errors), "attempts": attempts}
            return {
                "claims_review_report": report,
                "claims_review_attempts": attempts,
                "process": {
                    **state.get("process", {}),
                    "evidence_review": "rules+llm" if self._semantic_review_judge else "rules",
                    "evidence_review_detail": (
                        f"{report.metrics.get('llm_fallback_candidates', 0)} 个规则灰区候选经 LLM 审核"
                        if self._semantic_review_judge
                        else "规则审核；语义 LLM 兜底未启用"
                    ),
                },
            }

    def _review_comparison(self, state: ResearchState) -> dict[str, object]:
        with self._traced("review_comparison", "agent", {"row_count": len(state["comparisons"])}) as span:
            report = ComparisonReviewValidator().validate(state["analyses"], state["comparisons"])
            attempts = state.get("comparison_review_attempts", 0) + (0 if report.passed else 1)
            span["outputs"] = {"passed": report.passed, "errors": len(report.errors), "attempts": attempts}
            return {
                "comparison_review_report": report,
                "comparison_review_attempts": attempts,
                "process": {**state.get("process", {}), "comparison_review": "rules"},
            }

    def _review_synthesis(self, state: ResearchState) -> dict[str, object]:
        with self._traced("review_synthesis", "agent", {"summary_length": len(state.get("synthesis", ""))}) as span:
            report = SynthesisReviewValidator().validate(state["analyses"], state.get("synthesis"))
            attempts = state.get("synthesis_review_attempts", 0) + (0 if report.passed else 1)
            reports = (state["claims_review_report"], state["comparison_review_report"], report)
            issues = tuple(issue for item in reports for issue in item.issues)
            metrics = {f"claims_{key}": value for key, value in reports[0].metrics.items()}
            metrics.update({f"comparison_{key}": value for key, value in reports[1].metrics.items()})
            metrics.update({f"synthesis_{key}": value for key, value in report.metrics.items()})
            final = ReviewReport(not any(item.errors for item in reports), issues, metrics, reports[0].dimension_coverage)
            span["outputs"] = {"passed": report.passed, "errors": len(report.errors), "attempts": attempts}
            return {
                "synthesis_review_report": report,
                "synthesis_review_attempts": attempts,
                "review_report": final,
                "review_passed": final.passed,
                "review_notes": "；".join(issue.message for issue in issues) or "分阶段审核通过。",
                "process": {**state.get("process", {}), "synthesis_review": "rules"},
            }

    @staticmethod
    def _claims_review_route(state: ResearchState) -> Literal["retry", "continue"]:
        report = state.get("claims_review_report")
        retries = state.get("claims_review_attempts", 0)
        if (
            report
            and report.errors
            and retries <= get_settings().claims_review_max_retries
            and state.get("documents")
        ):
            return "retry"
        return "continue"

    @staticmethod
    def _comparison_review_route(state: ResearchState) -> Literal["retry", "continue"]:
        report = state.get("comparison_review_report")
        if report and report.errors and state.get("comparison_review_attempts", 0) <= 1:
            return "retry"
        return "continue"

    @staticmethod
    def _synthesis_review_route(state: ResearchState) -> Literal["retry", "write"]:
        report = state.get("synthesis_review_report")
        return "retry" if report and report.errors and state.get("synthesis_review_attempts", 0) <= 1 else "write"

    def _planner(self, state: ResearchState) -> dict[str, object]:
        with self._traced("planner", "agent", {"query": state["query"]}) as span:
            task_type = "topic_research"  # 当前只设定为主题检索
            span["outputs"] = {"task_type": task_type}
            return {
                "task_type": task_type,
                "review_attempts": 0,
                "claims_review_attempts": 0,
                "comparison_review_attempts": 0,
                "synthesis_review_attempts": 0,
                "comparisons": (),
                "synthesis": "",
                "review_passed": False,
                "failures": (),
                "process": {},
            }

    # 检索文献
    def _search(self, state: ResearchState) -> dict[str, object]:
        with self._traced(
            "search_papers",
            "tool",
            {"query": state["query"], "limit": state["limit"]},
        ) as span:
            papers, issues, raw_count = search_papers(
                state["query"],
                limit=state["limit"],
                run_id=state["run_id"],
                context=self._context,
            )
            span["outputs"] = {
                "raw_count": raw_count,
                "deduplicated_count": len(papers),
                "source_errors": len(issues),
            }
            return {"papers": papers, "source_issues": issues, "raw_count": raw_count}

    # 下载待精读论文
    def _download(self, state: ResearchState) -> dict[str, object]:
        with self._traced(
            "download_pdf",
            "tool",
            {"paper_count": len(state["papers"]), "read_limit": state["read_limit"]},
        ) as span:
            # 精读则下载pdf，数量read_limit
            pdfs, failures = download_pdf(
                state["papers"], read_limit=state["read_limit"], context=self._context
            )
            span["outputs"] = {
                "requested": state["read_limit"],
                "attempted": len(pdfs) + len(failures),  # 尝试次数
                "downloaded": len(pdfs),
                "failed": len(failures),  # 记录失败的次数
                "candidate_exhausted": len(pdfs) < state["read_limit"],  # 所有候选论文耗尽仍不足
            }
            return {"pdfs": pdfs, "failures": failures}

    # 解析pdf参数
    def _parse(self, state: ResearchState) -> dict[str, object]:
        with self._traced(
            "parse_pdf", "tool", {"pdf_count": len(state["pdfs"])}
        ) as span:
            # state["pdfs"]即_download成功下载得到的pdf
            documents, failures = parse_pdf(state["pdfs"], context=self._context)
            span["outputs"] = {
                "documents": len(documents),
                "chunks": sum(len(document.chunks) for document in documents),
                "failed": len(failures),
            }
            return {"documents": documents, "failures": state["failures"] + failures}

    # 召回文本块并提取结论
    def _extract(self, state: ResearchState) -> dict[str, object]:
        # 判断是否为重试，首次expanded为FALSE
        expanded = state.get("claims_review_attempts", 0) > 0
        with self._traced(
            "extract_claims",
            "tool",
            {"document_count": len(state["documents"]), "expanded": expanded},
        ) as span:
            analyses, failures = extract_claims(
                state["query"],
                state["papers"],
                state["pdfs"],
                state["documents"],
                run_id=state["run_id"],
                context=self._context,
                expanded=expanded,
            )
            prior = tuple(
                failure for failure in state["failures"] if failure.stage != "evidence"
            )
            span["outputs"] = {
                "analyses": len(analyses),
                "claims": sum(len(analysis.claims) for analysis in analyses),
                "failed": len(failures),
            }
            providers = sorted({analysis.provider for analysis in analyses})
            return {
                "analyses": analyses,
                "failures": prior + failures,
                "process": {
                    **state.get("process", {}),
                    "evidence_extraction": ", ".join(providers) if providers else "not_run",
                },
            }

    # 按照6个维度对比多篇论文
    def _compare(self, state: ResearchState) -> dict[str, object]:
        with self._traced(
            "compare_papers", "tool", {"analysis_count": len(state["analyses"])}
        ) as span:
            # 输入state["analyses"]，即成功产生claims的PaperEvidence集合
            rows = compare_papers(
                state["analyses"], run_id=state["run_id"], context=self._context
            )
            span["outputs"] = {"rows": len(rows)}
            return {"comparisons": rows}

    # 生成综合分析摘要
    # v1.1版本新增横向综合服务
    def _synthesize(self, state: ResearchState) -> dict[str, object]:
        # state["analyses"]为extract召回的claims(PaperEvidence)
        # 统计claim总数
        claim_count = sum(len(analysis.claims) for analysis in state["analyses"])
        with self._traced(
            "synthesis_agent",
            "agent",
            {
                "paper_count": len(state["analyses"]),
                "claim_count": claim_count,
                "comparison_count": len(state["comparisons"]),
            },
        ) as span:
            # 统计覆盖了哪些结论类型
            # 生成摘要字符串
            if self._context.synthesis_service is None:  # synthesis_service默认auto
                synthesis = TemplateClaimSynthesizer().generate(
                    state["query"], state["analyses"], state["comparisons"]
                )
                provider = "template"
            else:
                synthesis, provider = self._context.synthesis_service.generate(
                    state["query"], state["analyses"], state["comparisons"]
                )
            span["outputs"] = {
                "provider": provider,
                "summary_length": len(synthesis),
            }
            return {
                "synthesis": synthesis,
                "process": {
                    **state.get("process", {}),
                    "synthesis_generation": provider,
                    "synthesis_generation_detail": (
                        "未成功精读论文，未生成综述"
                        if provider == "not_run"
                        else f"策略={get_settings().synthesis_provider}; 实际执行={provider}"
                    ),
                },
            }

    # 检查证据是否满足最低要求
    # v1.2 新增review证据检查
    def _review(self, state: ResearchState) -> dict[str, object]:
        with self._traced(
            "review_agent",
            "agent",
            {"analysis_count": len(state["analyses"])},
        ) as span:
            # 将所有论文analysis中的claims都取出
            claims = [claim for analysis in state["analyses"] for claim in analysis.claims]
            # 新增
            report = ReviewValidator().validate(
                state["query"], state["analyses"], state.get("comparisons", ()), state.get("synthesis")
            )
            # 验证每条claim的证据
            evidence_valid = all(
                claim.evidence  # 不为空
                and all(ref.page_number >= 1  # 每条evidence的page_number
                    and ref.section_title for ref in claim.evidence)  # section_title非空
                for claim in claims
            )
            enough_claims = all(
                len(analysis.claims) >= min(4, len(analysis.document.chunks))
                for analysis in state["analyses"]
            )
            classified_count = sum(claim.claim_type != "other" for claim in claims)
            classified_ratio = classified_count / len(claims) if claims else 0.0
            passed = (
                bool(claims)
                and evidence_valid
                and enough_claims
                and classified_ratio >= 0.7
                and report.passed
            )
            notes = "证据审查通过。" if passed else "；".join(issue.message for issue in report.issues) or "claims 数量不足或分类比例低于 70%。"
            # 更新审核次数,审核失败次数
            attempts = state.get("review_attempts", 0) + (0 if passed else 1)
            span["outputs"] = {"passed": passed, "notes": notes, "attempts": attempts}
            return {
                "review_passed": passed,  # 通过True/FALSE
                "review_notes": notes,
                "review_attempts": attempts,
                "review_report": report,
            }

    @staticmethod
    def _review_route(state: ResearchState) -> Literal["retry", "write"]:
        # 审核未通过/失败次数不超过一次/至少有一篇解析成功的文档,则重试
        if not state["review_passed"] and state["review_attempts"] <= 1 and state["documents"]:
            return "retry"
        return "write"

    # 输出检索结果
    def _write(self, state: ResearchState) -> dict[str, object]:
        with self._traced(
            "write_review", "tool", {"review_passed": state["review_passed"]}
        ) as span:
            # 创建输出目录
            outputs = write_review(
                query=state["query"],
                run_id=state["run_id"],
                papers=state["papers"],
                analyses=state["analyses"],
                comparisons=state["comparisons"],
                failures=state["failures"],
                source_issues=state["source_issues"],
                raw_count=state["raw_count"],
                review_passed=state["review_passed"],
                synthesis=state["synthesis"],
                context=self._context,
            )
            span["outputs"] = {"files": list(outputs)}
            return {"outputs": outputs}

    def run(self, query: str, *, limit: int = 5, read_limit: int = 5) -> ResearchRunResult:
        # 清洗query
        clean_query = " ".join(query.split())
        if not clean_query:
            raise ValueError("检索 query 不能为空。")
        # 检索数量
        if not 1 <= limit <= 20:
            raise ValueError("limit 必须在 1 到 20 之间。")
        # 单篇精度数量
        if not 1 <= read_limit <= limit:
            raise ValueError("read_limit 必须在 1 到 limit 之间。")
        run_id = self._context.run_dir.name  # run_id 结果目录名
        self._context.store.start_run(run_id, clean_query)  # SQLiteStore
        try:
            # 开始执行工作流
            state = self.graph.invoke(
                {
                    "run_id": run_id,
                    "query": clean_query,
                    "limit": limit,
                    "read_limit": read_limit,
                },
                {"recursion_limit": 30},
            )
            self._context.store.finish_run(run_id, status="completed")
            self._context.trace.write(self._context.run_dir / "trace.jsonl")
        except Exception as exc:
            self._context.store.finish_run(run_id, status="failed", error=str(exc))
            raise
        return ResearchRunResult(
            run_id=run_id,
            query=clean_query,
            papers=state["papers"],
            analyses=state["analyses"],
            comparisons=state["comparisons"],
            failures=state["failures"],
            review_passed=state["review_passed"],
            review_report=state.get("review_report"),
            output_dir=str(self._context.run_dir.resolve()),
            outputs=state["outputs"],
            process=state.get("process", {}),
        )


def create_run_paths(output_root: Path, data_root: Path) -> tuple[str, Path, Path, Path]:
    """为 CLI 在创建工作流前生成一致的 run_id 与目录路径。"""

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return (
        run_id,
        output_root / run_id,
        data_root / "papers",
        data_root / "faiss" / run_id,
    )
