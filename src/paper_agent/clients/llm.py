"""OpenAI 兼容 LLM 的公共调用封装。"""

import logging

import httpx
from openai import OpenAI

logger = logging.getLogger(__name__)
RESPONSE_PREVIEW_LENGTH = 1000


def create_llm_client(
    *,
    api_key: str,
    base_url: str,
    proxy: str | None = None,
) -> OpenAI:
    """为当前配置的供应商创建 OpenAI 兼容客户端。"""

    if proxy:
        http_client = httpx.Client(proxy=proxy, trust_env=False)
        return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
    return OpenAI(api_key=api_key, base_url=base_url)


def create_chat_completion(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> str:
    """调用兼容的 Chat Completions 接口并返回纯文本内容。"""

    logger.info("LLM 请求开始：model=%s max_tokens=%s", model, max_tokens)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=(
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ),
            max_tokens=max_tokens,
            extra_body={
                "thinking": {
                    "type": "disabled"
                }
            },
        )
    except Exception as exc:
        logger.exception(
            "LLM 请求失败：model=%s error_type=%s status_code=%r "
            "request_id=%r body=%r error=%s cause=%r",
            model,
            type(exc).__name__,
            getattr(exc, "status_code", None),
            getattr(exc, "request_id", None),
            getattr(exc, "body", None),
            str(exc),
            exc.__cause__,
        )
        raise
    if not response.choices:
        logger.error("LLM 响应没有 choices：model=%s", model)
        return ""

    choice = response.choices[0]
    content = choice.message.content
    logger.info(
        "LLM 响应收到：model=%s choices=%s finish_reason=%r content_length=%s",
        model,
        len(response.choices),
        getattr(choice, "finish_reason", None),
        len(content or ""),
    )
    if not content:
        logger.warning("LLM 响应 content 为空：model=%s response=%r", model, response)
    return content.strip() if content else ""
