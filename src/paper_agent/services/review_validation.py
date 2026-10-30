"""research 工作流各阶段的审核器。"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal, Protocol

from paper_agent.clients.llm import create_chat_completion, create_llm_client
from paper_agent.config import Settings
from paper_agent.models import ComparisonRow, PaperEvidence

logger = logging.getLogger(__name__)

DIMENSIONS = ("contribution", "method", "dataset", "metric", "result", "limitation")
TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
CITATION_RE = re.compile(r"\[P(?P<paper>\d+),\s*p\.(?P<page>\d+)(?:\s*§[^\]]+)?\]")
POSITIVE = ("improve", "increase", "提升", "提高", "outperform", "优于", "achiev", "达到")
NEGATIVE = ("decrease", "degrad", "下降", "降低", "worse", "劣于", "限制", "limited")
COMPARISON_FIELDS = {
    "contribution": "contributions",
    "method": "methods",
    "dataset": "datasets",
    "metric": "metrics",
    "result": "results",
    "limitation": "limitations",
}
REQUIRED_SYNTHESIS_SECTIONS = (
    "## 研究问题与范围",
    "## 分论文详细分析",
    "## 跨论文方法与实验对比",
    "## 实验结果与可比性",
    "## 局限性与证据边界",
)


@dataclass(frozen=True)
class ReviewIssue:
    rule: str
    severity: str
    message: str
    paper_id: str | None = None
    claim_index: int | None = None
    chunk_id: str | None = None


@dataclass(frozen=True)
class ReviewReport:
    passed: bool
    issues: tuple[ReviewIssue, ...] = ()
    metrics: dict[str, float | int] = field(default_factory=dict)
    dimension_coverage: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def errors(self) -> tuple[ReviewIssue, ...]:
        return tuple(item for item in self.issues if item.severity == "error")


class SemanticReviewJudge(Protocol):
    """规则无法确定时使用的语义审核接口。"""

    def judge_claim_support(self, claim: str, quote: str) -> tuple[str, float, str]: ...

    def judge_relevance(self, query: str, claim: str) -> tuple[str, float, str]: ...

    def judge_batch(self, items: list[dict[str, str]]) -> dict[str, tuple[str, float, str]]: ...


class LLMReviewJudge:
    """调用 OpenAI 兼容接口，只返回审核结论，不修改原始 claim。"""

    def __init__(self, settings: Settings) -> None:
        self._client = create_llm_client(
            api_key=settings.llm_api_key or "",
            base_url=settings.llm_base_url or "",
            proxy=settings.http_proxy,
        )
        self._model = settings.llm_model or ""
        self._max_tokens = min(settings.llm_max_tokens, 512)

    def _judge(self, task: str, payload: dict[str, str], labels: tuple[str, ...]) -> tuple[str, float, str]:
        text = create_chat_completion(
            self._client,
            model=self._model,
            system_prompt=(
                "你是学术事实审核器，只审核输入，不得重写 claim。"
                "只返回 JSON：label、confidence、reason。"
                f"label 只能是：{', '.join(labels)}。"
            ),
            user_prompt=json.dumps({"task": task, **payload}, ensure_ascii=False),
            max_tokens=self._max_tokens,
        )
        data = json.loads(text)
        label = str(data.get("label", "uncertain"))
        confidence = float(data.get("confidence", 0.0))
        reason = str(data.get("reason", ""))
        return label, confidence, reason

    def judge_claim_support(self, claim: str, quote: str) -> tuple[str, float, str]:
        return self._judge(
            "判断 quote 是否支持 claim",
            {"claim": claim, "quote": quote},
            ("supported", "insufficient", "contradicted", "uncertain"),
        )

    def judge_relevance(self, query: str, claim: str) -> tuple[str, float, str]:
        return self._judge(
            "判断 claim 是否与研究主题相关",
            {"query": query, "claim": claim},
            ("relevant", "irrelevant", "uncertain"),
        )

    def judge_batch(self, items: list[dict[str, str]]) -> dict[str, tuple[str, float, str]]:
        """一次请求批量审核所有规则灰区，避免每个 claim 单独请求。"""
        text = create_chat_completion(
            self._client,
            model=self._model,
            system_prompt=(
                "你是学术事实审核器，只审核输入，不得重写 claim。"
                "只返回 JSON：results 数组；每项包含 id、label、confidence、reason。"
                "label 只能是 supported、insufficient、contradicted、relevant、irrelevant、uncertain。"
            ),
            user_prompt=json.dumps({"task": "批量审核 claims 的 quote 支持度和研究主题相关性", "items": items}, ensure_ascii=False),
            max_tokens=self._max_tokens,
        )
        data = json.loads(text)
        results: dict[str, tuple[str, float, str]] = {}
        for item in data.get("results", []):
            item_id = str(item.get("id", ""))
            if item_id:
                results[item_id] = (
                    str(item.get("label", "uncertain")),
                    float(item.get("confidence", 0.0)),
                    str(item.get("reason", "")),
                )
        return results


def create_semantic_review_judge(settings: Settings) -> SemanticReviewJudge | None:
    """template 模式只做规则审核；auto 模式配置完整时启用 LLM 兜底。"""

    if not settings.review_llm_enabled or settings.review_provider == "template" or not settings.llm_configured:
        return None
    return LLMReviewJudge(settings)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("‑", "-").replace("–", "-")).strip().casefold()


def _tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.casefold()))


class ReviewValidator:
    """执行不依赖模型的证据、相关性和一致性检查。"""

    def validate(
        self,
        query: str,
        analyses: tuple[PaperEvidence, ...],
        comparisons: tuple[ComparisonRow, ...] = (),
        synthesis: str | None = None,
    ) -> ReviewReport:
        issues: list[ReviewIssue] = []
        covered: dict[str, set[str]] = {dimension: set() for dimension in DIMENSIONS}
        claim_count = evidence_count = support_count = relevant_count = 0
        fingerprints: dict[str, tuple[str, str, int]] = {}
        directional: list[tuple[str, str, str, int]] = []
        query_tokens = _tokens(query)
        for analysis in analyses:
            chunks = {chunk.chunk_id: chunk for chunk in analysis.document.chunks}
            minimum_claims = min(4, len(analysis.document.chunks))
            if len(analysis.claims) < minimum_claims:
                issues.append(ReviewIssue("claim_count", "error", f"claims 数量不足，至少需要 {minimum_claims} 条", analysis.paper.paper_id))
            for index, claim in enumerate(analysis.claims):
                claim_count += 1
                if not claim.claim.strip():
                    issues.append(ReviewIssue("claim_empty", "error", "claim 为空", analysis.paper.paper_id, index))
                if claim.claim_type in covered:
                    covered[claim.claim_type].add(analysis.paper.paper_id)
                # 关键词相关性判断claim是否和主题相关
                # 正则提取英文单词、数字和中文连续字符
                claim_tokens = _tokens(claim.claim)
                # 计算 query token 在 claim 中的覆盖比例
                overlap = len(claim_tokens & query_tokens) / max(1, len(query_tokens))
                if overlap >= 0.2 or any(token in _norm(claim.claim) for token in query_tokens if len(token) > 2):
                    relevant_count += 1
                else:
                    issues.append(ReviewIssue("relevance", "warning", "claim 与研究主题关键词重叠较低", analysis.paper.paper_id, index))
                fingerprint = re.sub(r"\W+", "", _norm(claim.claim))
                if fingerprint in fingerprints:
                    old_paper, _, old_index = fingerprints[fingerprint]
                    issues.append(ReviewIssue("duplicate", "warning", f"与 {old_paper} 的 claim 重复（#{old_index}）", analysis.paper.paper_id, index))
                else:
                    fingerprints[fingerprint] = (analysis.paper.paper_id, claim.claim, index)
                direction = "positive" if any(word in _norm(claim.claim) for word in POSITIVE) else "negative" if any(word in _norm(claim.claim) for word in NEGATIVE) else ""
                if direction:
                    directional.append((claim.claim_type, direction, analysis.paper.paper_id, index))
                if not claim.evidence:
                    issues.append(ReviewIssue("evidence", "error", "claim 没有 evidence", analysis.paper.paper_id, index))
                    continue
                for ref in claim.evidence:
                    evidence_count += 1
                    chunk = chunks.get(ref.chunk_id)
                    if not ref.quote.strip():
                        issues.append(ReviewIssue("quote_empty", "error", "quote 为空", analysis.paper.paper_id, index, ref.chunk_id))
                        continue
                    if chunk is None:
                        issues.append(ReviewIssue("chunk_missing", "error", "引用的 chunk 不存在", analysis.paper.paper_id, index, ref.chunk_id))
                        continue
                    if ref.page_number != chunk.page_number or ref.section_title != chunk.section_title:
                        issues.append(ReviewIssue("location_mismatch", "error", "页码或章节与原始 chunk 不一致", analysis.paper.paper_id, index, ref.chunk_id))
                    if _norm(ref.quote) not in _norm(chunk.text):
                        issues.append(ReviewIssue("quote_not_found", "error", "quote 未出现在 chunk 原文", analysis.paper.paper_id, index, ref.chunk_id))
                        continue
                    missing_numbers = set(NUMBER_RE.findall(claim.claim)) - set(NUMBER_RE.findall(ref.quote))
                    if missing_numbers:
                        issues.append(ReviewIssue("claim_unsupported_number", "error", f"claim 包含 quote 中不存在的数字：{', '.join(sorted(missing_numbers))}", analysis.paper.paper_id, index, ref.chunk_id))
                    quote_tokens = _tokens(ref.quote)
                    if claim_tokens and len(claim_tokens & quote_tokens) / len(claim_tokens) >= 0.35:
                        support_count += 1
                    else:
                        issues.append(ReviewIssue("claim_support", "warning", "claim 与 quote 的关键词支持度较低", analysis.paper.paper_id, index, ref.chunk_id))
        for i, left in enumerate(directional):
            for right in directional[i + 1 :]:
                if left[0] == right[0] and left[1] != right[1]:
                    issues.append(ReviewIssue("direction_conflict", "warning", f"检测到 {left[2]} 与 {right[2]} 的方向相反", left[2], left[3]))
        # 校验comparisons
        if comparisons:
            ReviewValidator._validate_comparisons(analyses, comparisons, covered, issues)
        # 校验synthesis
        synthesis_citations = ReviewValidator._validate_synthesis(analyses, synthesis, issues) if synthesis is not None else 0
        coverage = {dimension: tuple(sorted(papers)) for dimension, papers in covered.items()}
        metrics = {
            "claim_count": claim_count,
            "evidence_count": evidence_count,
            "quote_valid_rate": (evidence_count - sum(i.rule in {"quote_empty", "chunk_missing", "quote_not_found"} for i in issues)) / evidence_count if evidence_count else 0.0,
            "support_rate": support_count / evidence_count if evidence_count else 0.0,
            "relevance_rate": relevant_count / claim_count if claim_count else 0.0,
            "covered_dimensions": sum(bool(value) for value in coverage.values()),
            "comparison_row_count": len(comparisons),
            "synthesis_citation_count": synthesis_citations,
        }
        return ReviewReport(not any(item.severity == "error" for item in issues), tuple(issues), metrics, coverage)


class ClaimsReviewValidator:
    """审核单篇论文 claims 与原文证据。"""

    def __init__(self, judge: SemanticReviewJudge | None = None) -> None:
        self._judge = judge

    def validate(self, query: str, analyses: tuple[PaperEvidence, ...]) -> ReviewReport:
        report = ReviewValidator().validate(query, analyses)
        allowed = {"claim_count", "claim_empty", "quote_empty", "chunk_missing", "location_mismatch", "quote_not_found", "claim_unsupported_number", "claim_support", "relevance", "duplicate", "direction_conflict", "evidence"}
        issues = tuple(issue for issue in report.issues if issue.rule in allowed)
        llm_candidates = sum(
            issue.rule in {"claim_support", "relevance"} for issue in issues
        )
        if self._judge is not None:
            issues = self._apply_semantic_fallback(query, analyses, issues)
        missing = tuple(dimension for dimension in DIMENSIONS if not report.dimension_coverage.get(dimension))
        if missing:
            issues += (ReviewIssue("dimension_coverage", "warning", f"全局未覆盖维度：{', '.join(missing)}"),)
        metrics = {
            **report.metrics,
            "llm_fallback_enabled": int(self._judge is not None),
            "llm_fallback_candidates": llm_candidates,
        }
        return ReviewReport(
            not any(issue.severity == "error" for issue in issues),
            issues,
            metrics,
            report.dimension_coverage,
        )

    def _apply_semantic_fallback(self, query: str, analyses: tuple[PaperEvidence, ...], issues: tuple[ReviewIssue, ...]) -> tuple[ReviewIssue, ...]:
        """将所有规则灰区合并成一次批量 LLM 请求；默认不开启该路径。"""
        result = list(issues)
        pending: list[dict[str, str]] = []
        issue_by_id: dict[str, ReviewIssue] = {}
        for issue in issues:
            if issue.rule not in {"claim_support", "relevance"} or issue.paper_id is None or issue.claim_index is None:
                continue
            analysis = next((item for item in analyses if item.paper.paper_id == issue.paper_id), None)
            if analysis is None or issue.claim_index >= len(analysis.claims):
                continue
            claim = analysis.claims[issue.claim_index]
            item_id = f"{issue.rule}:{issue.paper_id}:{issue.claim_index}:{issue.chunk_id or ''}"
            quote = next((ref.quote for ref in claim.evidence if ref.chunk_id == issue.chunk_id), "")
            pending.append({"id": item_id, "kind": issue.rule, "query": query, "claim": claim.claim, "quote": quote})
            issue_by_id[item_id] = issue
        if not pending:
            return tuple(result)
        try:
            decisions = self._judge.judge_batch(pending)  # type: ignore[union-attr]
            for item_id, (label, confidence, reason) in decisions.items():
                issue = issue_by_id.get(item_id)
                if issue is None or confidence < 0.7:
                    continue
                if label in {"supported", "relevant"}:
                    result.remove(issue)
                elif label in {"contradicted", "irrelevant"}:
                    index = result.index(issue)
                    result[index] = ReviewIssue("claim_semantic_failure", "error", reason or "LLM 判定 claim 审核失败", issue.paper_id, issue.claim_index, issue.chunk_id)
        except Exception as exc:
            logger.warning("review 批量 LLM 兜底失败，保留规则 warning: %s", exc)
        return tuple(result)


class ComparisonReviewValidator:
    """审核六维对比表是否能由 claims 支持。"""

    def validate(self, analyses: tuple[PaperEvidence, ...], comparisons: tuple[ComparisonRow, ...]) -> ReviewReport:
        issues: list[ReviewIssue] = []
        SynthesisReviewValidator._validate_comparisons(analyses, comparisons, {dimension: set() for dimension in DIMENSIONS}, issues)
        return ReviewReport(not any(issue.severity == "error" for issue in issues), tuple(issues), {"comparison_row_count": len(comparisons)}, {})


class SynthesisReviewValidator:
    """审核综合文本中的引用能否回溯到 evidence。"""

    def validate(self, analyses: tuple[PaperEvidence, ...], synthesis: str | None) -> ReviewReport:
        issues: list[ReviewIssue] = []
        count = ReviewValidator._validate_synthesis(analyses, synthesis, issues)
        return ReviewReport(not any(issue.severity == "error" for issue in issues), tuple(issues), {"synthesis_citation_count": count}, {})

    @staticmethod
    def _validate_comparisons(
        analyses: tuple[PaperEvidence, ...],
        comparisons: tuple[ComparisonRow, ...],
        covered: dict[str, set[str]],
        issues: list[ReviewIssue],
    ) -> None:
        """校验对比表是否覆盖真实 analyses，避免 CSV 聚合结果与 claims 脱节。"""
        analysis_by_id = {item.paper.paper_id: item for item in analyses}
        seen: set[str] = set()
        for row in comparisons:
            # 是否存在重复论文行
            if row.paper_id in seen:
                issues.append(ReviewIssue("comparison_duplicate_paper", "error", "comparison 中存在重复论文行", row.paper_id))
            seen.add(row.paper_id)
            # 是否引用了不存在的论文
            analysis = analysis_by_id.get(row.paper_id)
            if analysis is None:
                issues.append(ReviewIssue("comparison_unknown_paper", "error", "comparison 引用了不存在的论文", row.paper_id))
                continue
            claims_by_type = {
                dimension: [claim.claim for claim in analysis.claims if claim.claim_type == dimension]
                for dimension in DIMENSIONS
            }
            for dimension, field_name in COMPARISON_FIELDS.items():
                value = getattr(row, field_name)
                # 某个维度有 claims 但 comparison 字段为空
                if claims_by_type[dimension] and not value.strip():
                    issues.append(ReviewIssue("comparison_missing_dimension", "warning", f"comparison 缺少 {dimension} 维度内容", row.paper_id))
                # comparison 中存在没有对应 claims 的维度
                if value.strip() and not claims_by_type[dimension]:
                    issues.append(ReviewIssue("comparison_unbacked_dimension", "warning", f"comparison 的 {dimension} 没有对应 claim", row.paper_id))
                for claim_text in claims_by_type[dimension]:
                    occurrences = value.count(claim_text)
                    if occurrences == 0:
                        issues.append(ReviewIssue("comparison_missing_claim", "warning", f"comparison 的 {dimension} 遗漏 claim", row.paper_id))
                    elif occurrences > 1:
                        issues.append(ReviewIssue("comparison_duplicate_claim", "warning", f"comparison 的 {dimension} 重复计入 claim", row.paper_id))
        # 是否遗漏已分析论文
        missing_rows = set(analysis_by_id) - seen
        for paper_id in sorted(missing_rows):
            issues.append(ReviewIssue("comparison_missing_paper", "warning", "comparison 缺少已分析论文行", paper_id))

    @staticmethod
    def _validate_synthesis(
        analyses: tuple[PaperEvidence, ...],
        synthesis: str | None,
        issues: list[ReviewIssue],
    ) -> int:
        """对现有 Markdown 综合结果做可回溯的轻量检查。"""
        if synthesis is None:
            return 0
        if not synthesis.strip():
            issues.append(ReviewIssue("synthesis_empty", "error", "synthesis 为空"))
            return 0
        valid_pages = {
            index: {ref.page_number for claim in analysis.claims for ref in claim.evidence}
            for index, analysis in enumerate(analyses, start=1)
        }
        title_indices = {
            analysis.paper.title.casefold(): index
            for index, analysis in enumerate(analyses, start=1)
        }
        citations = list(CITATION_RE.finditer(synthesis))
        for match in citations:
            paper_index = int(match.group("paper"))
            page = int(match.group("page"))
            if paper_index not in valid_pages:
                issues.append(ReviewIssue("synthesis_unknown_paper", "error", f"synthesis 引用了不存在的论文 P{paper_index}"))
            elif page not in valid_pages[paper_index]:
                issues.append(ReviewIssue("synthesis_unknown_page", "warning", f"synthesis 引用的 P{paper_index} 页码没有对应 evidence: p.{page}"))
        for section in REQUIRED_SYNTHESIS_SECTIONS:
            if section not in synthesis:
                issues.append(ReviewIssue("synthesis_missing_section", "error", f"synthesis 缺少章节：{section}"))
        cited_papers = {int(match.group("paper")) for match in citations if int(match.group("paper")) in valid_pages}
        # 模板综述使用论文标题和 [p.x §section] 定位，不一定带 P1 前缀。
        for title, index in title_indices.items():
            if title in synthesis.casefold():
                cited_papers.add(index)
        generic_pages = {
            int(page)
            for page in re.findall(r"\[p\.(\d+)(?:\s*§[^\]]+)?\]", synthesis)
        }
        for page in generic_pages:
            if len(analyses) == 1 and page not in valid_pages.get(1, set()):
                issues.append(ReviewIssue("synthesis_unknown_page", "warning", f"synthesis 引用的页码没有对应 evidence: p.{page}"))
        if len(cited_papers) < len(analyses):
            issues.append(ReviewIssue("synthesis_missing_paper", "error", "synthesis 未覆盖全部需要精读的论文"))
        if analyses and not citations and not generic_pages and not cited_papers:
            issues.append(ReviewIssue("synthesis_without_evidence", "error", "synthesis 没有引用已审核 evidence"))
        return len(citations)


# 保持旧 ReviewValidator API 兼容，旧调用仍可复用拆分后的实现。
ReviewValidator._validate_comparisons = staticmethod(SynthesisReviewValidator._validate_comparisons)
ReviewValidator._validate_synthesis = staticmethod(SynthesisReviewValidator._validate_synthesis)
