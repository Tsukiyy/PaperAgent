from __future__ import annotations

import argparse
import socket
import sys
from xml.etree import ElementTree

import httpx

ARXIV_ENDPOINT = "https://export.arxiv.org/api/query"
# ARXIV_ENDPOINT = "http://cn.arxiv.org/api/query"
DEFAULT_PROXY = "http://127.0.0.1:7890"
ATOM_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}


def resolve_host(host: str) -> None:
    print(f"[DNS] resolving {host} ...")

    try:
        addresses = socket.getaddrinfo(
            host,
            443,
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        print(f"[DNS] failed: {type(exc).__name__}: {exc}")
        return

    unique_addresses = []
    for item in addresses:
        address = item[4][0]
        if address not in unique_addresses:
            unique_addresses.append(address)

    for address in unique_addresses:
        print(f"[DNS] {host} -> {address}")


def run_query(
    *,
    query: str,
    limit: int,
    timeout: float,
    endpoint: str,
    trust_env: bool,
    proxy: str | None,
) -> int:
    parsed_endpoint = httpx.URL(endpoint)
    endpoint_host = parsed_endpoint.host
    # httpx 当前版本返回 str，兼容旧版本可能返回 bytes 的情况。
    endpoint_host = (
        endpoint_host.decode()
        if isinstance(endpoint_host, bytes)
        else endpoint_host
    )
    resolve_host(endpoint_host)

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": limit,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    headers = {
        "User-Agent": "PaperAgent-live-arxiv-test/1.0",
        "Accept": "application/atom+xml, application/xml, text/xml",
    }

    print()
    print("[HTTP] endpoint:", endpoint)
    print("[HTTP] params:", params)
    print("[HTTP] proxy:", proxy or "<direct>")
    print("[HTTP] trust_env:", trust_env)
    print("[HTTP] sending request ...")

    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers=headers,
            proxy=proxy,
            trust_env=trust_env,
        ) as client:
            response = client.get(endpoint, params=params)

    except httpx.ConnectError as exc:
        print()
        print("[ERROR] connection failed")
        print("[ERROR] exception type:", type(exc).__name__)
        print("[ERROR] message:", str(exc))
        print("[ERROR] repr:", repr(exc))

        if exc.__cause__ is not None:
            print("[ERROR] underlying cause type:", type(exc.__cause__).__name__)
            print("[ERROR] underlying cause:", repr(exc.__cause__))

        return 2

    except httpx.TimeoutException as exc:
        print()
        print("[ERROR] request timed out")
        print("[ERROR] exception type:", type(exc).__name__)
        print("[ERROR] message:", str(exc))
        return 3

    except httpx.HTTPError as exc:
        print()
        print("[ERROR] HTTPX error")
        print("[ERROR] exception type:", type(exc).__name__)
        print("[ERROR] message:", str(exc))
        print("[ERROR] repr:", repr(exc))
        return 4

    print()
    print("[HTTP] status:", response.status_code)
    print("[HTTP] final URL:", response.url)
    print("[HTTP] content-type:", response.headers.get("content-type"))
    print("[HTTP] content-length:", response.headers.get("content-length"))
    print("[HTTP] response bytes:", len(response.content))

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        print()
        print("[ERROR] arXiv returned an HTTP error")
        print("[ERROR] message:", str(exc))
        print("[ERROR] body preview:", response.text[:500])
        return 5

    try:
        root = ElementTree.fromstring(response.content)
    except ElementTree.ParseError as exc:
        print()
        print("[ERROR] response is not valid Atom XML")
        print("[ERROR] parser message:", str(exc))
        print("[ERROR] body preview:", response.text[:500])
        return 6

    entries = root.findall("atom:entry", ATOM_NAMESPACE)

    print()
    print(f"[RESULT] entries: {len(entries)}")

    if not entries:
        print("[RESULT] arXiv returned no matching papers")
        return 0

    for index, entry in enumerate(entries, start=1):
        title_node = entry.find("atom:title", ATOM_NAMESPACE)
        id_node = entry.find("atom:id", ATOM_NAMESPACE)
        published_node = entry.find("atom:published", ATOM_NAMESPACE)

        title = " ".join((title_node.text or "").split()) if title_node is not None else ""
        paper_id = (id_node.text or "").strip() if id_node is not None else ""
        published = (
            (published_node.text or "").strip()
            if published_node is not None
            else ""
        )

        print(f"{index}. {title}")
        print(f"   id: {paper_id}")
        print(f"   published: {published}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="独立真实访问 arXiv API 的测试脚本"
    )
    parser.add_argument(
        "--query",
        default="agentic RAG",
        help="arXiv 检索关键词，默认：agentic RAG",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="返回论文数量，默认：3",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="请求超时时间，默认：30 秒",
    )
    parser.add_argument(
        "--endpoint",
        default=ARXIV_ENDPOINT,
        help=f"arXiv API 地址，默认：{ARXIV_ENDPOINT}",
    )
    parser.add_argument(
        "--trust-env",
        action="store_true",
        help=(
            "允许 httpx 读取 HTTP_PROXY、HTTPS_PROXY、ALL_PROXY 等环境变量；"
            "默认不读取环境代理"
        ),
    )
    parser.add_argument(
        "--proxy",
        default=DEFAULT_PROXY,
        help=f"代理地址，默认：{DEFAULT_PROXY}",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="禁用显式代理，直接访问 arXiv",
    )

    args = parser.parse_args()

    if not 1 <= args.limit <= 5:
        print("limit 必须在 1 到 5 之间", file=sys.stderr)
        return 1

    return run_query(
        query=args.query,
        limit=args.limit,
        timeout=args.timeout,
        endpoint=args.endpoint,
        trust_env=args.trust_env,
        proxy=None if args.no_proxy else args.proxy,
    )


if __name__ == "__main__":
    raise SystemExit(main())
