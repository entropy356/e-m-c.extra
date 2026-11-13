"""联网搜索：Tavily Search API。

只负责「给一个 query，拿回几条标题/链接/正文片段」。
Tavily 返回的是已清洗的正文节选（content），比 SERP 摘要信息量大得多，
所以这里不再自己抓页面。**不设降级源**：拿不到就如实报错，不拿劣质结果糊弄模型。
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

ENGINE = "tavily"
ENDPOINT = "https://api.tavily.com/search"
MAX_SNIPPET = 700
TIMEOUT = 20.0


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _api_key() -> str:
    return (os.environ.get("TAVILY_API_KEY") or "").strip()


def search(query: str, *, max_results: int = 4, timeout: float = TIMEOUT) -> dict[str, Any]:
    """返回 {query, engine, results:[{title,url,snippet,score}], error}。"""
    query = _clean(query)[:400]
    out: dict[str, Any] = {"query": query, "engine": ENGINE, "results": [], "error": ""}
    if not query:
        out["error"] = "empty query"
        return out

    key = _api_key()
    if not key:
        out["error"] = "TAVILY_API_KEY 未配置"
        return out

    payload = {
        "query": query,
        "max_results": max(1, min(int(max_results), 8)),
        "search_depth": "basic",
        "topic": "general",
        "include_answer": False,
        "include_raw_content": False,
    }
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read()[:200].decode("utf-8", "replace")
        out["error"] = f"HTTP {error.code}: {detail}"
        return out
    except Exception as error:  # noqa: BLE001 - 搜索失败不能让整轮对话失败
        out["error"] = f"{type(error).__name__}: {error}"[:200]
        return out

    for item in (data.get("results") or [])[:max_results]:
        title = _clean(item.get("title") or "")
        link = _clean(item.get("url") or "")
        if not title or not link.startswith("http"):
            continue
        out["results"].append(
            {
                "title": title[:160],
                "url": link,
                "snippet": _clean(item.get("content") or "")[:MAX_SNIPPET],
                "score": round(float(item.get("score") or 0), 3),
            }
        )
    if not out["results"]:
        out["error"] = "no results parsed"
    return out


def as_tool_message(payload: dict[str, Any]) -> str:
    """把搜索结果压成模型好读的纯文本。"""
    lines = [f"搜索：{payload.get('query', '')}（{payload.get('engine', ENGINE)}）"]
    results = payload.get("results") or []
    if not results:
        lines.append(f"没有拿到结果。{payload.get('error') or ''}".strip())
        lines.append("这次搜索失败了。可以换个关键词再试一次；如果还是不行，就凭已有知识回答并说明无法核实。")
        return "\n".join(lines)
    for index, item in enumerate(results, 1):
        lines.append(f"[{index}] {item['title']}")
        lines.append(f"链接：{item['url']}")
        if item.get("snippet"):
            lines.append(f"内容：{item['snippet']}")
    lines.append("以上是网页正文节选，可能过时或不完整；引用时说清来源，不要编造没出现的细节。")
    return "\n".join(lines)
