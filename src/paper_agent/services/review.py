"""基于论文元数据生成简短综述。"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAIError

from paper_agent.clients.llm import create_chat_completion, create_llm_client
from paper_agent.config import Settings
from paper_agent.exceptions import ReviewError
from paper_agent.models import Paper

logger = logging.getLogger(__name__)

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+")

# 根据模板生成综述
class ReviewGenerator(Protocol):
    """综述生成器协议。"""

    def generate(self, query: str, papers: tuple[Paper, ...]) -> str: ...


@dataclass(frozen=True)
class TemplateReviewGenerator:
    """无需模型的可追溯综述；只压缩 arXiv 摘要，不增加外部事实。"""

    def generate(self, query: str, papers: tuple[Paper, ...]) -> str:
        if not papers:
            return f"arXiv 未返回与“{query}”相关的论文，因此暂时无法形成综述。"

        lines = [
            f"围绕“{query}”，本次从 arXiv 检索到 {len(papers)} 篇相关论文。",
            "以下概览仅依据 arXiv 元数据与摘要生成：",
        ]
        for index, paper in enumerate(papers, start=1):
            abstract_preview = SENTENCE_BOUNDARY.split(paper.abstract, maxsplit=1)[0].strip()
            if len(abstract_preview) > 220:
                abstract_preview = f"{abstract_preview[:217].rstrip()}..."
            lines.append(f"{index}. **{paper.title}**：{abstract_preview}")
        lines.append("这些论文的具体方法、实验结论与局限仍需在 Step 2 解析全文后核验。")
        return "\n\n".join(lines)

# openai 生成综述内容
class LLMReviewGenerator:
    """使用 OpenAI 兼容 Chat Completions 接口综合多篇 arXiv 摘要。"""

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

    def generate(self, query: str, papers: tuple[Paper, ...]) -> str:
        if not papers:
            return f"arXiv 未返回与“{query}”相关的论文，因此暂时无法形成综述。"

        records = [
            {
                "index": index,
                "title": paper.title,  # 标题
                "authors": list(paper.authors),  # 作者
                "published_at": paper.published_at.date().isoformat(),  # 发布时间
                "abstract": paper.abstract,  # 摘要
                "url": str(paper.paper_url),  # 地址URL
            }
            for index, paper in enumerate(papers, start=1)
        ]
        text = create_chat_completion(
            self._client,
            model=self._model,
            system_prompt=(
                "你是严谨的学术检索助手。仅依据提供的 arXiv 元数据写中文简短综述；"
                "先用一段总结共同主题，再用精炼条目说明每篇工作的关注点。"
                "引用论文时使用 [1]、[2] 这样的编号。不得声称已经阅读全文，不得补充输入中没有的"
                "实验数字、因果结论或优劣判断；证据不足时明确说明。不要生成标题或论文链接。"
            ),
            user_prompt=(
                f"研究问题：{query}\n"
                f"arXiv 检索结果 JSON：\n{json.dumps(records, ensure_ascii=False)}"
            ),
            max_tokens=self._max_tokens,
        )
        if not text:
            raise ReviewError("LLM 返回了空综述。")
        return text


class ReviewService:
    """按配置选择模型综述，auto 模式下允许安全降级。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._template = TemplateReviewGenerator()

    # 生成简短综述
    def generate(self, query: str, papers: tuple[Paper, ...]) -> tuple[str, str]:
        # 根据配置
        provider = self._settings.review_provider
        logger.info(
            "Step 1 简短综述 provider=%s llm_configured=%s",
            provider,
            self._settings.llm_configured,
        )
        # 使用本地模板
        if provider == "template":
            return self._template.generate(query, papers), "template"
        if not self._settings.llm_configured:
            logger.warning(
                "Step 1 简短综述未调用 LLM：通用配置不完整，missing=%s",
                "、".join(self._settings.missing_llm_settings) or "<none>",
            )
            if provider == "llm":
                missing = "、".join(self._settings.missing_llm_settings)
                raise ReviewError(f"llm 模式缺少配置：{missing}。")
            return self._template.generate(query, papers), "template"

        try:
            # 调用openai
            generator = LLMReviewGenerator(
                api_key=self._settings.llm_api_key or "",
                base_url=self._settings.llm_base_url or "",
                model=self._settings.llm_model or "",
                max_tokens=self._settings.llm_max_tokens,
                proxy=self._settings.http_proxy,
            )
            return generator.generate(query, papers), self._settings.llm_provider
        except (OpenAIError, ReviewError) as exc:
            logger.warning(
                "LLM Step 1 简短综述生成失败，降级为模板综述："
                "error_type=%s error=%s",
                type(exc).__name__,
                str(exc),
            )
            if provider == "llm":
                if isinstance(exc, ReviewError):
                    raise
                raise ReviewError(f"LLM 综述生成失败：{exc}") from exc
            return self._template.generate(query, papers), "template"
#         返回tuple[str, str]--["综述内容","template/openai"]
