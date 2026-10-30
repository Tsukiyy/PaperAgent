"""项目共享的 HTTPX 客户端构造。"""

from collections.abc import Mapping

import httpx


def create_http_client(
    *,
    timeout: float,
    headers: Mapping[str, str],
    proxy: str | None = None,
) -> httpx.Client:
    """创建统一 HTTP 客户端；显式代理时不再被环境绕过规则覆盖。"""

    kwargs: dict[str, object] = {
        "timeout": timeout,
        "follow_redirects": True,
        "headers": dict(headers),
    }
    # 有 PAPER_AGENT_HTTP_PROXY，显式使用该代理
    if proxy:
        kwargs.update({"proxy": proxy, "trust_env": False})
    else:  # 使用 HTTP_PROXY、HTTPS_PROXY、ALL_PROXY 等环境配置
        kwargs["trust_env"] = True
    return httpx.Client(**kwargs)
