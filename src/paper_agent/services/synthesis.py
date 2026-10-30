"""基于多篇论文 claims 生成有证据定位的横向综述。"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from openai import OpenAIError

from paper_agent.clients.llm import create_chat_completion, create_llm_client
from paper_agent.config import Settings
from paper_agent.exceptions import SynthesisError
from paper_agent.models import ComparisonRow, PaperEvidence

logger = logging.getLogger(__name__)

CLAIM_DIMENSIONS = (
    "contribution",
    "method",
    "dataset",
    "metric",
    "result",
    "limitation",
    "other",
)
COMPARISON_FIELDS = {
    "contribution": "contributions",
    "method": "methods",
    "dataset": "datasets",
    "metric": "metrics",
    "result": "results",
    "limitation": "limitations",
}


def build_structured_comparison(
    analyses: tuple[PaperEvidence, ...],
    comparisons: tuple[ComparisonRow, ...],
) -> list[dict[str, Any]]:
    """按 comparison 顺序从 analyses 构建结构化输入，不解析 CSV 字符串。"""

    analysis_by_id = {analysis.paper.paper_id: analysis for analysis in analyses}
    row_by_id = {row.paper_id: row for row in comparisons}
    ordered_ids = [row.paper_id for row in comparisons if row.paper_id in analysis_by_id]
    ordered_ids.extend(
        analysis.paper.paper_id
        for analysis in analyses
        if analysis.paper.paper_id not in ordered_ids
    )

    records: list[dict[str, Any]] = []
    for index, paper_id in enumerate(ordered_ids, start=1):
        analysis = analysis_by_id[paper_id]
        comparison = row_by_id.get(paper_id)
        claims_by_dimension: dict[str, list[dict[str, Any]]] = {
            dimension: [] for dimension in CLAIM_DIMENSIONS
        }
        for claim in analysis.claims:
            claims_by_dimension[claim.claim_type].append(
                {
                    "claim": claim.claim,
                    "evidence": [
                        {
                            "page": reference.page_number,
                            "section": reference.section_title,
                            "quote": reference.quote,
                        }
                        for reference in claim.evidence
                    ],
                }
            )
        compared_dimensions = [
            dimension
            for dimension, field in COMPARISON_FIELDS.items()
            if comparison is not None and getattr(comparison, field)
        ]
        records.append(
            {
                "paper_ref": f"P{index}",
                "paper_id": paper_id,
                "title": analysis.paper.title,
                "url": str(analysis.paper.paper_url),
                "authors": list(analysis.paper.authors),
                "published_at": analysis.paper.published_at.year,
                "abstract": analysis.paper.abstract,
                "compared_dimensions": compared_dimensions,
                "claims_by_dimension": {
                    dimension: claims for dimension, claims in claims_by_dimension.items() if claims
                },
            }
        )
    return records


class ClaimSynthesizer(Protocol):
    """横向综述生成器协议。"""

    def generate(
        self,
        query: str,
        analyses: tuple[PaperEvidence, ...],
        comparisons: tuple[ComparisonRow, ...],
    ) -> str: ...


@dataclass(frozen=True)
class TemplateClaimSynthesizer:
    """在模型不可用时输出结构化、可追溯的横向摘要。"""

    def generate(
        self,
        query: str,
        analyses: tuple[PaperEvidence, ...],
        comparisons: tuple[ComparisonRow, ...],
    ) -> str:
        records = build_structured_comparison(analyses, comparisons)
        claim_count = sum(len(analysis.claims) for analysis in analyses)
        if not records:
            return "未能从公开 PDF 形成可验证 claims，因此无法开展横向综合。"

        coverage: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            for dimension in record["claims_by_dimension"]:
                coverage.setdefault(dimension, []).append(record)

        lines = [
            "## 研究问题与范围",
            "",
            f"围绕“{query}”，本次综合 {len(records)} 篇论文的 {claim_count} 条可定位 claims。",
            "当前使用本地降级模式；以下内容仅依据已提取证据，不对缺失信息作无依据推断。",
            "",
            "## 分论文详细分析",
            "",
        ]
        for record in records:
            lines.extend([f"### {record['paper_ref']} {record['title']}", ""])
            abstract = record.get("abstract", "").strip()
            if abstract:
                lines.extend([f"**研究问题与背景：** {abstract}", ""])
            for dimension, label in (
                ("contribution", "核心贡献"),
                ("method", "方法机制"),
                ("dataset", "数据集与实验设置"),
                ("metric", "评价指标"),
                ("result", "主要结果"),
                ("limitation", "局限性"),
            ):
                claims = record["claims_by_dimension"].get(dimension, [])
                if claims:
                    lines.append(f"**{label}：**")
                    lines.extend(f"- {item['claim']}（{_first_location(item)}）" for item in claims)
                    lines.append("")
        lines.extend(["## 跨论文方法与实验对比", "", "### 论文共识", ""])
        shared = {name: items for name, items in coverage.items() if len(items) >= 2}
        if shared:
            for dimension, items in shared.items():
                references = "、".join(item["paper_ref"] for item in items)
                lines.append(f"- {references} 均覆盖 `{dimension}` 维度；具体观点见下方证据。")
        else:
            lines.append("- 当前 claims 没有至少两篇论文共同覆盖的对比维度。")

        lines.extend(["", "### 方法差异", ""])
        method_found = False
        for record in records:
            for claim in record["claims_by_dimension"].get("method", []):
                method_found = True
                location = _first_location(claim)
                lines.append(f"- [{record['paper_ref']}, {location}] {claim['claim']}")
        if not method_found:
            lines.append("- 输入 claims 未覆盖方法维度，无法比较方法差异。")

        for dimension, label in (("dataset", "数据集差异"), ("metric", "评价指标差异")):
            lines.extend(["", f"### {label}", ""])
            found = False
            for record in records:
                for claim in record["claims_by_dimension"].get(dimension, []):
                    found = True
                    lines.append(
                        f"- [{record['paper_ref']}, {_first_location(claim)}] {claim['claim']}"
                    )
            if not found:
                lines.append(f"- 已提取证据未覆盖{label}，无法进行有依据的比较。")

        lines.extend(["", "### 实验设置差异", ""])
        setup_found = False
        for record in records:
            for dimension in ("dataset", "metric", "method", "result"):
                for claim in record["claims_by_dimension"].get(dimension, []):
                    setup_found = True
                    lines.append(
                        f"- [{record['paper_ref']}, {_first_location(claim)}] "
                        f"{dimension}：{claim['claim']}"
                    )
        if not setup_found:
            lines.append("- 当前没有足够的实验设置证据用于横向比较。")

        lines.extend(["", "### 实验结果与矛盾", ""])
        result_found = False
        for record in records:
            for claim in record["claims_by_dimension"].get("result", []):
                result_found = True
                location = _first_location(claim)
                lines.append(f"- [{record['paper_ref']}, {location}] {claim['claim']}")
        if result_found:
            lines.append("- 本地降级模式不判断上述结果是否语义矛盾。")
            lines.append("- 需由 LLM 或人工结合实验条件核验。")
        else:
            lines.append("- 输入 claims 未覆盖实验结果，无法分析相互矛盾的结果。")

        lines.extend(["", "## 实验结果与可比性", ""])
        lines.append("上述结果只有在数据集、预测范围、基线、指标和实验条件相近时才可直接比较；本地模板不会据此判定真正矛盾。")
        lines.extend(["", "## 局限性与证据边界", ""])
        lines.append("- 综合内容仅来自已提取 claims；未出现的维度和事实均未补造。")
        return "\n".join(lines)


class LLMClaimSynthesizer:
    """使用 OpenAI 兼容 Chat Completions 接口综合结构化全文 claims。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        max_tokens: int,
        proxy: str | None = None,
    ) -> None:
        self._client = create_llm_client(
            api_key=api_key,
            base_url=base_url,
            proxy=proxy,
        )
        self._model = model
        self._max_tokens = max_tokens

    def generate(
        self,
        query: str,
        analyses: tuple[PaperEvidence, ...],
        comparisons: tuple[ComparisonRow, ...],
    ) -> str:
        records = build_structured_comparison(analyses, comparisons)
        if not records:
            return "未能从公开 PDF 形成可验证 claims，因此无法开展横向综合。"

        text = create_chat_completion(
            self._client,
            model=self._model,
            system_prompt=(
                "你是严谨的学术综述作者。仅依据输入论文的摘要、claims、原文 quote 和证据定位，"
                "写一篇详细、连贯、可复核的中文综述，不得补充外部事实或把缺失信息当作否定结论。"
                "不要把 claims 简单罗列成清单，要解释每篇论文解决的研究问题、核心贡献、方法机制、"
                "数据集与实验设置、评价指标、主要结果和局限性。每篇论文都必须有独立的小节或完整段落。"
                "随后进行跨论文对比，具体说明方法路线、数据集、指标、实验条件的差异，指出哪些结果可比较、"
                "哪些不能直接比较及原因。只输出以下 Markdown 小节：## 研究问题与范围、"
                "## 分论文详细分析、"
                "## 跨论文方法与实验对比、## 实验结果与可比性、## 局限性与证据边界。"
                "共识必须至少由两篇论文支持；冲突分析必须区分真正冲突与实验条件不同导致的差异。"
                "每个实质判断都尽量使用 [P1, p.2 §Methods] 格式标注论文和证据位置；"
                "引用 quote 中的具体数字时必须忠实。"
                "证据不足时明确说明，不要为了完整而编造结论。"
            ),
            user_prompt=(
                f"研究问题：{query}\n"
                "结构化论文对比 JSON：\n"
                f"{json.dumps(records, ensure_ascii=False)}"
            ),
            max_tokens=self._max_tokens,
        )
        if not text:
            raise SynthesisError("LLM 返回了空的横向综述。")
        return text


class ClaimSynthesisService:
    """按配置调用模型，auto 模式失败时降级为本地横向摘要。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._template = TemplateClaimSynthesizer()

    def generate(
        self,
        query: str,
        analyses: tuple[PaperEvidence, ...],
        comparisons: tuple[ComparisonRow, ...],
    ) -> tuple[str, str]:
        # provider默认auto
        provider = self._settings.synthesis_provider
        logger.info(
            "横向综述 provider=%s llm_configured=%s",
            provider,
            self._settings.llm_configured,
        )
        if provider == "template":
            return self._template.generate(query, analyses, comparisons), "template"
        if not self._settings.llm_configured:
            logger.warning(
                "横向综述未调用 LLM：通用配置不完整，missing=%s",
                "、".join(self._settings.missing_llm_settings) or "<none>",
            )
            if provider == "llm":
                missing = "、".join(self._settings.missing_llm_settings)
                raise SynthesisError(f"llm 综合模式缺少配置：{missing}。")
            return self._template.generate(query, analyses, comparisons), "template"
        if not analyses:
            return "没有成功精读的论文及可验证 claims，无法生成论文综述。", "not_run"

        try:
            generator = LLMClaimSynthesizer(
                api_key=self._settings.llm_api_key or "",
                base_url=self._settings.llm_base_url or "",
                model=self._settings.llm_model or "",
                max_tokens=self._settings.llm_max_tokens,
                proxy=self._settings.http_proxy,
            )
            return (
                generator.generate(query, analyses, comparisons),
                self._settings.llm_provider,
            )
        except (OpenAIError, SynthesisError) as exc:
            logger.warning(
                "LLM 横向综述生成失败，降级为模板综合："
                "error_type=%s error=%s",
                type(exc).__name__,
                str(exc),
            )
            if provider == "llm":
                if isinstance(exc, SynthesisError):
                    raise
                raise SynthesisError(f"LLM 横向综述生成失败：{exc}") from exc
            return self._template.generate(query, analyses, comparisons), "template"


def _first_location(claim: dict[str, Any]) -> str:
    evidence = claim["evidence"][0]
    return f"p.{evidence['page']} §{evidence['section']}"
