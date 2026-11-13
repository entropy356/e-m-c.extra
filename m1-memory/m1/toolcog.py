"""工具循环认知策略：模型可以在一轮对话的任意时刻自主联网搜索。

runtime 自带的 ``direct`` 策略只发一次请求；这里换成一个小的 tool loop：
模型返回 tool_calls 就执行搜索、把结果塞回上下文、继续让它生成，直到它
给出正文。搜索次数和模型轮数都有上限，避免失控。
"""

from __future__ import annotations

import json
import time
from typing import Any, Mapping, Sequence

from emc_runtime.contracts import AssistantReply, ProviderError

from . import websearch

MAX_ROUNDS = 5
MAX_SEARCHES = 4

TOOL_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "联网搜索网页，返回标题、链接和摘要。任何需要最新、具体或你不确定的信息时"
                "都可以调用，一轮里可以多次调用，也可以在回答到一半时再次调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词，尽量具体"},
                },
                "required": ["query"],
            },
        },
    }
]

TOOL_HINT = (
    "\n\n你可以随时调用 web_search 工具联网搜索：涉及最新动态、价格、天气、新闻、"
    "具体资料或你不确定的事实时，先搜再答；一次没查到可以换关键词再搜。"
    "闲聊、记忆库能回答的问题不要搜。引用搜索结果时自然地带出来源，不要罗列链接清单。"
)


class ToolLoopCognition:
    """与 runtime 的 CognitionStrategy 同形：run(messages, provider, options)。"""

    kind = "tool-loop"

    def __init__(self) -> None:
        self.last_trace: dict[str, Any] = {}

    def run(
        self,
        messages: Sequence[dict],
        provider: Any,
        options: Mapping[str, Any],
    ) -> AssistantReply:
        trace: dict[str, Any] = {
            "rounds": 0,
            "calls": [],
            "searches": 0,
            "stopped": "",
        }
        self.last_trace = trace
        msgs = [dict(message) for message in messages]
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["content"] = (msgs[0].get("content") or "") + TOOL_HINT
        merged = {**getattr(provider, "options", {}), **dict(options)}

        for _ in range(MAX_ROUNDS):
            message = self._call(msgs, provider, merged, tools=TOOL_SPEC)
            trace["rounds"] += 1
            calls = message.get("tool_calls") or []
            content = (message.get("content") or "").strip()
            if calls:
                msgs.append(
                    {
                        "role": "assistant",
                        "content": message.get("content") or "",
                        "tool_calls": calls,
                    }
                )
                for call in calls:
                    msgs.append(self._run_tool(call, trace))
                continue
            if content:
                return AssistantReply(content=content)
            trace["stopped"] = "empty reply"
            break

        # 兜底：工具轮用完了还没正文，就让模型基于已有材料直接收尾。
        trace["stopped"] = trace["stopped"] or "round limit"
        msgs.append({"role": "user", "content": "根据上面已有的信息直接回答，不要再调用工具。"})
        message = self._call(msgs, provider, merged, tools=None)
        content = (message.get("content") or "").strip()
        if not content:
            raise ProviderError("provider returned an empty reply; the turn was aborted")
        return AssistantReply(content=content)

    # ------------------------------------------------------------------ 内部

    @staticmethod
    def _call(msgs: list[dict], provider: Any, options: Mapping[str, Any], tools: Any) -> dict:
        payload: dict[str, Any] = {"model": provider.model, "messages": msgs}
        for key in ("temperature", "top_p", "max_tokens"):
            if options.get(key) is not None:
                payload[key] = options[key]
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        result = provider._transport(  # noqa: SLF001 - 复用同一套超时/重试/代理
            provider.base_url + "/chat/completions",
            payload,
            {"Authorization": f"Bearer {provider.api_key}"},
            provider.timeout,
            provider.proxy,
            provider.max_attempts,
        )
        try:
            return result["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            raise ProviderError("invalid chat completion response") from None

    @staticmethod
    def _run_tool(call: dict, trace: dict) -> dict:
        function = call.get("function") or {}
        name = function.get("name") or ""
        try:
            args = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        query = str(args.get("query") or "")
        started = time.time()
        if name != "web_search":
            payload = {"query": query, "results": [], "error": f"unknown tool {name}"}
        elif trace["searches"] >= MAX_SEARCHES:
            payload = {"query": query, "results": [], "error": "search budget exhausted"}
        else:
            payload = websearch.search(query)
            trace["searches"] += 1
        record = {
            "query": payload.get("query", query),
            "engine": payload.get("engine", websearch.ENGINE),
            "ms": int((time.time() - started) * 1000),
            "count": len(payload.get("results") or []),
            "error": payload.get("error") or "",
            "results": [
                {"title": item["title"], "url": item["url"]}
                for item in (payload.get("results") or [])[:4]
            ],
        }
        trace["calls"].append(record)
        return {
            "role": "tool",
            "tool_call_id": call.get("id") or "",
            "content": websearch.as_tool_message(payload),
        }
