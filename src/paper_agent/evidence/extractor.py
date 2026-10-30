"""从带 page/section 的文本块中提取可验证结论。"""

import json
import logging
import re
from dataclasses import dataclass

from openai import OpenAIError
from pydantic import BaseModel, Field, ValidationError

from paper_agent.clients.llm import create_chat_completion, create_llm_client
from paper_agent.config import Settings
from paper_agent.exceptions import EvidenceError
from paper_agent.models import (
    DocumentChunk,
    EvidenceClaim,
    EvidenceReference,
    Paper,
    ParsedDocument,
)

logger = logging.getLogger(__name__)
RESPONSE_PREVIEW_LENGTH = 1000

WHITESPACE = re.compile(r"\s+")
SENTENCE = re.compile(r".+?(?:[.!?。！？](?=\s|$)|$)", re.DOTALL)
JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
BROKEN_WORD = re.compile(r"(?<=[A-Za-z])[-‐‑]\s+(?=[a-z])")
SECTION_PRIORITY = (
    "conclu",
    "limitation",
    "discussion",
    "result",
    "experiment",
    "method",
    "approach",
    "introduction",
    "abstract",
)
CLAIM_TYPES = (
    "contribution",
    "method",
    "dataset",
    "metric",
    "result",
    "limitation",
)
TYPE_MARKERS = {
    "contribution": (
        "we propose", "we present", "we introduce", "this paper proposes",
        "this paper presents", "this survey", "our contribution", "novel",
        "survey examines", "survey provides", "we develop", "introduces a",
        "provides a", "first ",
    ),
    "method": (
        "method", "approach", "framework", "architecture", "pipeline", "workflow",
        "algorithm", "orchestrat", "agent", "retrieval", "router", "delegat",
    ),
    "dataset": (
        "dataset", "data set", "corpus", "evaluation set", "knowledge base",
        "participants", "deployment sites", "questions", "samples",
    ),
    "metric": (
        "metric", "measure", "accuracy", "precision", "recall", "f1", "rouge", "bleu",
        "score", "latency", "throughput", "cost", "shapiro-wilk",
    ),
    "result": (
        "results", "finding", "outperform", "improv", "increase", "decrease", "achiev",
        "demonstrat", "show that", "indicate that", "performance gain", "compared with",
        "compared to",
    ),
    "limitation": (
        "limitation", "challenge", "drawback", "shortcoming", "constraint", "overhead",
        "risk", "remain open", "future work", "future research", "further investigation", "may not",
        "is constrained", "are constrained", "resource consumption", "increase latency",
        "proprietary", "undocumented", "may be altered",
        "limiting generalization", "underperform", "rely on heuristic",
        "heavily concentrated", "limited network", "computationally intensive",
        "additional latency", "failure mode",
    ),
}


class _RawEvidence(BaseModel):
    evidence_id: str
    quote: str


class _RawClaim(BaseModel):
    claim: str
    claim_type: str = "other"
    evidence: list[_RawEvidence] = Field(min_length=1)


class _RawResponse(BaseModel):
    claims: list[_RawClaim]


def _priority(chunk: DocumentChunk) -> tuple[int, int]:
    rank = _section_priority(chunk.section_title)
    return rank, chunk.page_number


def _section_priority(section_title: str) -> int:
    title = section_title.casefold()
    rank = next(
        (index for index, marker in enumerate(SECTION_PRIORITY) if marker in title),
        len(SECTION_PRIORITY),
    )
    return rank


def _normalized(text: str) -> str:
    return WHITESPACE.sub(" ", text).strip()


def _marker_count(text: str, claim_type: str) -> int:
    return sum(marker in text for marker in TYPE_MARKERS[claim_type])


def _claim_type(section_title: str, text: str = "") -> str:
    title = section_title.casefold()
    content = text.casefold()
    content_counts = {
        claim_type: _marker_count(content, claim_type) for claim_type in CLAIM_TYPES
    }

    # 结果句通常也会出现 metric；含比较、变化或明确数值结果时优先归为 result。
    result_expression = content_counts["result"] and (
        bool(re.search(r"\b\d+(?:\.\d+)?\s*%", content))
        or any(
            marker in content
            for marker in (
                "the result", "results show", "results indicate", "findings:",
                "outperform", "performance gain", "compared", "achieved",
            )
        )
        or any(marker in title for marker in ("result", "finding", "experiment"))
    )
    context_section = any(
        marker in title for marker in ("abstract", "introduction", "background", "related work")
    )
    explicit_result = content.startswith(("findings", "results")) or any(
        marker in content for marker in ("results show", "results indicate", "results were")
    )
    if context_section and result_expression and not explicit_result and not any(
        marker in content for marker in ("we ", "our ", "this study", "this paper")
    ):
        return "other"
    if explicit_result or result_expression:
        return "result"
    if content_counts["contribution"]:
        return "contribution"
    resolves_limitation = any(
        marker in content
        for marker in (
            "addresses these limitations", "overcomes this limitation", "mitigates",
            "limitations motivate",
        )
    )
    if content_counts["limitation"] and not resolves_limitation:
        return "limitation"
    if content_counts["dataset"] and any(
        marker in content
        for marker in (
            "dataset", "data set", "corpus", "evaluation set", "participants",
            "deployment sites",
        )
    ):
        return "dataset"
    metric_expression = content_counts["metric"] and (
        any(marker in title for marker in ("metric", "evaluation", "measure"))
        or any(
            marker in content
            for marker in ("metric", "measured", "we evaluate", "we assessed", "score of")
        )
    )
    if metric_expression:
        return "metric"
    if content_counts["method"]:
        return "method"
    if any(marker in title for marker in ("result", "experiment", "finding")):
        return "result"
    if any(marker in title for marker in ("limitation", "challenge", "future")):
        return "limitation"
    if any(marker in title for marker in ("method", "approach", "methodology", "architecture")):
        return "method"
    if any(marker in title for marker in ("dataset", "data set", "corpus")):
        return "dataset"
    if "metric" in title:
        return "metric"
    if "conclu" in title and any(
        marker in content for marker in ("we ", "this paper", "this study")
    ):
        return "contribution"
    return "other"


def _clean_claim(text: str) -> str:
    """只清理 claim 的断行连字符；证据 quote 仍逐字保留解析后的原文。"""

    return BROKEN_WORD.sub("", text).strip()


def _candidate_score(section_title: str, quote: str, claim_type: str) -> int:
    content = quote.casefold()
    score = 8 if claim_type != "other" else 0
    score += min(_marker_count(content, claim_type), 3) * 3 if claim_type != "other" else 0
    score += max(0, len(SECTION_PRIORITY) - _section_priority(section_title))
    if 80 <= len(quote) <= 500:
        score += 3
    elif len(quote) > 600:
        score -= 8
    if re.search(r"\b\d+(?:\.\d+)?\s*%", quote):
        score += 2
    if quote[:1].islower() or re.match(r"^\d+[a-z]?,", quote.casefold()):
        score -= 20
    if re.fullmatch(r"(?:table|figure)\s+\w+\s+(?:summarizes|shows)\s+the\s+results\.?", content):
        score -= 12
    if claim_type == "result" and any(
        marker in content for marker in ("analysts suggest", "may experience", "may achieve")
    ):
        score -= 10
    if re.match(r"^section\s+\w+\s+(?:reviews|describes|summarizes)\b", content):
        score -= 15
    if " denotes " in content or "prompt: context:" in content:
        score -= 15
    if content.startswith(("table ", "figure ")):
        score -= 15
    if re.match(r"^\d+(?:\.\d+)+\s", content):
        score -= 20
    if content.count("figure") + content.count("table") >= 2:
        score -= 15
    return score


def _balanced_chunks(
    chunks: tuple[DocumentChunk, ...],
    *,
    limit: int,
) -> tuple[DocumentChunk, ...]:
    """按 claim 类型保留章节多样性，避免 24 个块全部来自同一种章节。"""

    ranked = sorted(chunks, key=_priority)
    selected: list[DocumentChunk] = []
    for claim_type in CLAIM_TYPES:
        matches = [
            chunk
            for chunk in ranked
            if _claim_type(chunk.section_title, chunk.text) == claim_type
        ]
        selected.extend(matches[:3])
    selected.extend(ranked)
    return tuple(dict.fromkeys(selected))[:limit]


@dataclass(frozen=True)
class TemplateEvidenceExtractor:
    """抽取原文句子作为结论与 quote，保证无模型时仍有可追溯输出。"""

    max_claims: int = 10

    def extract(self, paper: Paper, document: ParsedDocument) -> tuple[EvidenceClaim, ...]:
        candidates: list[tuple[int, str, str, str, DocumentChunk]] = []
        seen: set[str] = set()
        for chunk in document.chunks:
            if "reference" in chunk.section_title.casefold():
                continue
            for sentence_match in SENTENCE.finditer(_normalized(chunk.text)):
                quote = _normalized(sentence_match.group(0))[:700].strip()
                if len(quote) < 45:
                    continue
                claim = _clean_claim(quote)
                fingerprint = re.sub(r"\W+", "", claim.casefold())[:240]
                if not fingerprint or fingerprint in seen:
                    continue
                seen.add(fingerprint)
                claim_type = _claim_type(chunk.section_title, claim)
                score = _candidate_score(chunk.section_title, quote, claim_type)
                if score < 8:
                    continue
                candidates.append((score, claim_type, claim, quote, chunk))

        candidates.sort(key=lambda item: (-item[0], _priority(item[4])))
        selected: list[tuple[int, str, str, str, DocumentChunk]] = []
        # 先尽量覆盖六个对比维度，再用剩余额度补充高价值结论。
        for claim_type in CLAIM_TYPES:
            match = next((item for item in candidates if item[1] == claim_type), None)
            if match is not None:
                selected.append(match)
        type_counts = {
            claim_type: sum(item[1] == claim_type for item in selected)
            for claim_type in (*CLAIM_TYPES, "other")
        }
        max_per_type = max(2, (self.max_claims + 2) // 3)
        for item in candidates:
            claim_type = item[1]
            if item in selected or type_counts.get(claim_type, 0) >= max_per_type:
                continue
            selected.append(item)
            type_counts[claim_type] = type_counts.get(claim_type, 0) + 1
            if len(selected) >= self.max_claims:
                break

        claims: list[EvidenceClaim] = []
        for _, claim_type, claim, quote, chunk in selected[: self.max_claims]:
            claims.append(
                EvidenceClaim(
                    paper_id=paper.paper_id,
                    claim=claim,
                    claim_type=claim_type,
                    evidence=(
                        EvidenceReference(
                            chunk_id=chunk.chunk_id,
                            page_number=chunk.page_number,
                            section_title=chunk.section_title,
                            quote=quote,
                        ),
                    ),
                )
            )
        if not claims:
            raise EvidenceError(f"论文 {paper.paper_id} 没有足够文本形成带证据结论。")
        return tuple(claims)


class LLMEvidenceExtractor:
    """让模型基于证据 ID 生成结论，再由代码校验原文 quote。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        max_tokens: int,
        max_claims: int,
        proxy: str | None = None,
    ) -> None:
        self._client = create_llm_client(
            api_key=api_key,
            base_url=base_url,
            proxy=proxy,
        )
        self._model = model
        self._max_tokens = max_tokens
        self._max_claims = max_claims

    def extract(self, paper: Paper, document: ParsedDocument) -> tuple[EvidenceClaim, ...]:
        selected = _balanced_chunks(document.chunks, limit=24)
        chunk_map = {chunk.chunk_id: chunk for chunk in selected}
        records = [
            {
                "evidence_id": chunk.chunk_id,
                "page": chunk.page_number,
                "section": chunk.section_title,
                "text": chunk.text[:1800],
            }
            for chunk in selected
        ]
        # create_chat_completion 调用llm.py 的API调用
        text = create_chat_completion(
            self._client,
            model=self._model,
            system_prompt=(
                "你是严谨的论文证据提取器。仅使用输入文本提取论文自身的实质性 claims，"
                "不要把一般背景、研究动机或相关工作误标为论文贡献。分类必须依据句子语义，"
                "不能只依据章节标题。尽量覆盖 contribution、method、dataset、metric、result、"
                "limitation 六个维度；原文没有的维度不要臆造。"
                f"最多返回 {self._max_claims} 条中文结论。每条结论至少引用一个 evidence_id，"
                "claim_type 只能是 contribution、method、dataset、metric、result、"
                "limitation 或 other。"
                "quote 必须逐字复制自该 evidence_id 的 text，不得改写 quote。只返回 JSON："
                '{"claims":[{"claim":"...","claim_type":"method","evidence":'
                '[{"evidence_id":"p1-c1","quote":"原文"}]}]}。'
            ),
            user_prompt=(
                f"论文：{paper.title}\n"
                f"证据块 JSON：\n{json.dumps(records, ensure_ascii=False)}"
            ),
            max_tokens=self._max_tokens,
        )
        # 校验llm输出是否是JSON格式
        match = JSON_OBJECT.search(text)
        if not match:
            preview = text[:RESPONSE_PREVIEW_LENGTH].replace("\n", "\\n")
            logger.error(
                "LLM claim 响应不是 JSON object：paper_id=%s content_length=%s "
                "response_preview=%r",
                paper.paper_id,
                len(text),
                preview,
            )
            raise EvidenceError(
                f"LLM 未返回有效 JSON 结论；响应预览：{preview}"
            )
        try:
            raw = _RawResponse.model_validate_json(match.group(0))
        except ValidationError as exc:
            preview = text[:RESPONSE_PREVIEW_LENGTH].replace("\n", "\\n")
            logger.error(
                "LLM claim JSON 结构校验失败：paper_id=%s error=%s response_preview=%r",
                paper.paper_id,
                str(exc),
                preview,
            )
            raise EvidenceError(f"LLM 结论结构校验失败：{exc}") from exc

        claims: list[EvidenceClaim] = []
        for raw_claim in raw.claims[: self._max_claims]:
            references: list[EvidenceReference] = []
            for raw_evidence in raw_claim.evidence:
                chunk = chunk_map.get(raw_evidence.evidence_id)
                quote = _normalized(raw_evidence.quote)
                if chunk is None or not quote or quote not in _normalized(chunk.text):
                    continue
                references.append(
                    EvidenceReference(
                        chunk_id=chunk.chunk_id,
                        page_number=chunk.page_number,
                        section_title=chunk.section_title,
                        quote=quote,
                    )
                )
            if references and raw_claim.claim.strip():
                claims.append(
                    EvidenceClaim(
                        paper_id=paper.paper_id,
                        claim=raw_claim.claim.strip(),
                        claim_type=(
                            raw_claim.claim_type
                            if raw_claim.claim_type
                            in {
                                "contribution",
                                "method",
                                "dataset",
                                "metric",
                                "result",
                                "limitation",
                                "other",
                            }
                            else "other"
                        ),
                        evidence=tuple(references),
                    )
                )
        if not claims:
            preview = text[:RESPONSE_PREVIEW_LENGTH].replace("\n", "\\n")
            logger.error(
                "LLM claim 未通过证据引用校验：paper_id=%s response_preview=%r",
                paper.paper_id,
                preview,
            )
            raise EvidenceError("LLM 结论没有通过证据 ID 和原文 quote 校验。")
        return tuple(claims)


class EvidenceService:
    """按配置提取结论；auto 模式在模型失败时降级为原文抽取。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._template = TemplateEvidenceExtractor(settings.max_claims_per_paper)

    def extract(
        self,
        paper: Paper,
        document: ParsedDocument,
    ) -> tuple[tuple[EvidenceClaim, ...], str]:
        provider = self._settings.evidence_provider
        logger.info(
            "claim 提取 provider=%s llm_configured=%s paper_id=%s",
            provider,
            self._settings.llm_configured,
            paper.paper_id,
        )
        if provider == "template":
            return self._template.extract(paper, document), "template"
        if not self._settings.llm_configured:
            logger.warning(
                "claim 提取未调用 LLM：通用配置不完整，missing=%s",
                "、".join(self._settings.missing_llm_settings) or "<none>",
            )
            if provider == "llm":
                missing = "、".join(self._settings.missing_llm_settings)
                raise EvidenceError(f"llm 证据模式缺少配置：{missing}。")
            return self._template.extract(paper, document), "template"
        try:
            extractor = LLMEvidenceExtractor(
                api_key=self._settings.llm_api_key or "",
                base_url=self._settings.llm_base_url or "",
                model=self._settings.llm_model or "",
                max_tokens=self._settings.llm_max_tokens,
                max_claims=self._settings.max_claims_per_paper,
                proxy=self._settings.http_proxy,
            )
            return extractor.extract(paper, document), self._settings.llm_provider
        except (OpenAIError, EvidenceError) as exc:
            logger.warning(
                "LLM claim 提取失败，降级为模板抽取：paper_id=%s "
                "error_type=%s error=%s",
                paper.paper_id,
                type(exc).__name__,
                str(exc),
            )
            if provider == "llm":
                raise
            return self._template.extract(paper, document), "template"
