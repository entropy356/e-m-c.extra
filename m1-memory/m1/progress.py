"""轻量的进度广播：模块执行到哪一步，前端就能看到哪一步。

一个会话同一时刻只跑一轮，所以按 conversation_id 记录即可。
"""

from __future__ import annotations

import time
from typing import Any

_LIVE: dict[str, dict[str, Any]] = {}


def begin(conversation_id: str) -> None:
    _LIVE[conversation_id] = {"stage": "读取状态", "started": time.time(), "events": []}


def stage(conversation_id: str, text: str, **extra: Any) -> None:
    entry = _LIVE.get(conversation_id)
    if entry is None:
        entry = {"stage": text, "started": time.time(), "events": []}
        _LIVE[conversation_id] = entry
    entry["stage"] = text
    entry["updated"] = time.time()
    entry["events"].append({"at": round(time.time() - entry["started"], 2), "stage": text, **extra})
    entry["events"] = entry["events"][-12:]


def snapshot(conversation_id: str) -> dict[str, Any]:
    entry = _LIVE.get(conversation_id)
    if not entry:
        return {}
    return {
        "stage": entry.get("stage", ""),
        "elapsed": round(time.time() - entry.get("started", time.time()), 1),
        "events": entry.get("events", []),
    }


def clear(conversation_id: str) -> None:
    _LIVE.pop(conversation_id, None)
