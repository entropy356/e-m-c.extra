"""e-m-c · M1 记忆：把 e-m-c.runtime 跑成一个可以玩的聊天前端。

runtime 负责单轮编排、状态事务和 trace；这里只做三件事：
1. 把一轮对话变成后台任务，前端轮询运行链；
2. 暴露记忆库的只读视图和“喂料”入口；
3. 串行化所有写操作（runtime 的 revision 是全局的）。
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
# runtime 源码来自 vendor（自包含）或同级的 e-m-c.runtime（并列克隆）。
RUNTIME = os.environ.get("EMC_RUNTIME_PATH", str(BASE.parent.parent / "e-m-c.runtime" / "src"))
for _candidate in (str(BASE / "vendor"), RUNTIME):
    if Path(_candidate).is_dir():
        sys.path.insert(0, _candidate)
        break

from flask import Flask, jsonify, request, send_from_directory  # noqa: E402

from emc_runtime.config import load_config  # noqa: E402
from emc_runtime.contracts import Event  # noqa: E402
from emc_runtime.engine import build_engine  # noqa: E402
from emc_runtime.storage import ModuleStateWrite  # noqa: E402

from m1 import model, progress, websearch  # noqa: E402
from m1.module import ingest_document, make_doc_id  # noqa: E402
from m1 import toolcog  # noqa: E402
from m1.toolcog import ToolLoopCognition  # noqa: E402

CONFIG_PATH = os.environ.get("M1_CONFIG", str(BASE / "config.json"))
MODULE_ID = "m1.memory"
SCOPE_KEY = "__agent__"

app = Flask(__name__, static_folder=str(BASE / "static"), static_url_path="/static")

CONFIG = load_config(CONFIG_PATH)
ENGINE = build_engine(CONFIG)
# runtime 不动源码：把单次请求的认知策略换成带联网搜索的工具循环。
ENGINE.cognition = ToolLoopCognition()
TURN_LOCK = threading.Lock()
RUNS: dict[str, dict[str, Any]] = {}


def save_search_trace(turn_id: str, trace: dict) -> None:
    """搜索记录单独存一张表，不改 runtime 的 trace 结构。"""
    if not trace.get("calls"):
        return
    import sqlite3

    connection = sqlite3.connect(ENGINE.storage.path, timeout=10)
    try:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS search_traces (turn_id TEXT PRIMARY KEY, payload TEXT)"
        )
        connection.execute(
            "INSERT OR REPLACE INTO search_traces (turn_id, payload) VALUES (?, ?)",
            (turn_id, json.dumps(trace, ensure_ascii=False)),
        )
        connection.commit()
    finally:
        connection.close()


def search_traces(turn_ids: list[str]) -> dict[str, dict]:
    if not turn_ids:
        return {}
    import sqlite3

    connection = sqlite3.connect(f"file:{ENGINE.storage.path}?mode=ro", uri=True, timeout=10)
    try:
        exists = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='search_traces'"
        ).fetchone()
        if not exists:
            return {}
        marks = ",".join("?" for _ in turn_ids)
        rows = connection.execute(
            f"SELECT turn_id, payload FROM search_traces WHERE turn_id IN ({marks})", turn_ids
        ).fetchall()
        return {row[0]: json.loads(row[1]) for row in rows}
    finally:
        connection.close()


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------------ 状态读写

def module_instance():
    return ENGINE.registry.modules[0]


def read_state() -> dict:
    entry = ENGINE.storage.module_state(MODULE_ID, SCOPE_KEY)
    if entry is None:
        return model.new_state(module_instance().encoder.info())
    state = entry.state
    if (state.get("encoder") or {}).get("id") in (None, "", "unknown"):
        state["encoder"] = module_instance().encoder.info()
    return state


def write_state(state: dict, note: str, conversation: str = "memory-console") -> int:
    """用 runtime 的提交路径落库：单事务、校验 revision。"""
    with TURN_LOCK:
        return ENGINE.storage.commit_turn(
            expected_revision=ENGINE.storage.revision(),
            conversation_id=conversation,
            turn_id=uuid.uuid4().hex,
            event_id=uuid.uuid4().hex,
            event_kind="memory_edit",
            new_messages=[],
            module_writes=[
                ModuleStateWrite(
                    module_id=MODULE_ID,
                    scope_key=SCOPE_KEY,
                    state_version=1,
                    state=state,
                )
            ],
            trace={
                "kind": "memory_edit",
                "note": note,
                "at": _now(),
                "memory": model.stats(state),
            },
        )


# ------------------------------------------------------------------ 会话

def list_conversations() -> list[dict]:
    rows = ENGINE.storage._connect if False else None  # noqa: F841 - 明确不碰私有连接
    import sqlite3

    connection = sqlite3.connect(f"file:{ENGINE.storage.path}?mode=ro", uri=True, timeout=10)
    try:
        cursor = connection.execute(
            "SELECT conversation_id, COUNT(*) AS n, MAX(created_at) AS last "
            "FROM messages GROUP BY conversation_id ORDER BY last DESC LIMIT 50"
        )
        items = [
            {"id": row[0], "messages": row[1], "last": row[2]} for row in cursor.fetchall()
        ]
        for item in items:
            row = connection.execute(
                "SELECT content FROM messages WHERE conversation_id=? AND role='user' "
                "ORDER BY seq ASC LIMIT 1",
                (item["id"],),
            ).fetchone()
            item["preview"] = (row[0][:40] if row else "").replace("\n", " ")
    finally:
        connection.close()
    return items


def conversation_turns(conversation_id: str) -> list[dict]:
    messages = ENGINE.storage.history(conversation_id)
    turns = {turn["turn_id"]: turn for turn in ENGINE.storage.recent_turns(conversation_id, 200)}
    searches = search_traces(list(turns))
    ordered: list[dict] = []
    seen: set[str] = set()
    for message in messages:
        if message.turn_id in seen:
            continue
        seen.add(message.turn_id)
        turn = turns.get(message.turn_id)
        items = [
            {"role": item.role, "content": item.content}
            for item in messages
            if item.turn_id == message.turn_id
        ]
        trace = (turn or {}).get("trace") or {}
        module_trace = (trace.get("modules") or {}).get(MODULE_ID) or {}
        ordered.append(
            {
                "turn_id": message.turn_id,
                "messages": items,
                "created_at": (turn or {}).get("created_at"),
                "prepare": module_trace.get("prepare_trace") or {},
                "finalize": module_trace.get("finalize_trace") or {},
                "context": trace.get("context_summary") or {},
                "search": searches.get(message.turn_id) or {},
            }
        )
    return ordered


def conversation_title(conversation_id: str) -> str:
    for message in ENGINE.storage.history(conversation_id):
        if message.role == "user":
            text = " ".join(message.content.split())
            return text[:26] + ("…" if len(text) > 26 else "")
    return "新对话"


# ------------------------------------------------------------------ 运行

def start_turn(conversation_id: str, content: str) -> str:
    run_id = uuid.uuid4().hex[:12]

    def worker() -> None:
        with TURN_LOCK:
            progress.begin(conversation_id)
            started = time.time()
            RUNS[run_id] = {
                "id": run_id,
                "status": "running",
                "conversation": conversation_id,
                "started": started,
            }
            try:
                result = ENGINE.run(
                    Event(kind="user_message", conversation_id=conversation_id, content=content)
                )
                trace = (result.module_traces.get(MODULE_ID) or {})
                search_trace = dict(ENGINE.cognition.last_trace or {})
                save_search_trace(result.turn_id, search_trace)
                RUNS[run_id].update(
                    status="done",
                    reply=result.reply,
                    turn_id=result.turn_id,
                    revision=result.revision,
                    prepare=trace.get("prepare") or {},
                    finalize=trace.get("finalize") or {},
                    search=search_trace,
                )
            except Exception as exc:  # noqa: BLE001 - 错误要原样给前端看
                RUNS[run_id].update(
                    status="error",
                    error=f"{type(exc).__name__}: {exc}"[:400],
                )
            finally:
                RUNS[run_id]["ms"] = int((time.time() - started) * 1000)
                progress.clear(conversation_id)

    RUNS[run_id] = {
        "id": run_id,
        "status": "queued",
        "conversation": conversation_id,
        "started": time.time(),
    }
    threading.Thread(target=worker, daemon=True).start()
    return run_id


# ------------------------------------------------------------------ 路由

@app.get("/")
def index():
    return send_from_directory(BASE / "static", "index.html")


@app.get("/api/meta")
def meta():
    state = read_state()
    return jsonify(
        {
            "model": CONFIG.provider_params.get("model"),
            "encoder": state.get("encoder"),
            "params": model.PARAMS,
            "memory": model.stats(state),
            "search": {"enabled": True, "engine": websearch.ENGINE, "max_searches": toolcog.MAX_SEARCHES},
        }
    )


@app.get("/api/threads")
def threads():
    return jsonify({"threads": list_conversations()})


@app.post("/api/threads")
def create_thread():
    body = request.get_json(silent=True) or {}
    conversation_id = (body.get("id") or f"c-{uuid.uuid4().hex[:8]}").strip()
    return jsonify({"id": conversation_id})


@app.get("/api/threads/<conversation_id>")
def thread_detail(conversation_id: str):
    context = progress.snapshot(conversation_id)
    active = next(
        (
            run_id
            for run_id, run in RUNS.items()
            if run["conversation"] == conversation_id and run["status"] in {"queued", "running"}
        ),
        None,
    )
    return jsonify(
        {
            "id": conversation_id,
            "turns": conversation_turns(conversation_id),
            "title": conversation_title(conversation_id),
            "live": context,
            "active": active,
        }
    )


@app.post("/api/threads/<conversation_id>/messages")
def post_message(conversation_id: str):
    body = request.get_json(silent=True) or {}
    content = (body.get("content") or "").strip()
    if not content:
        return jsonify({"error": "消息不能为空"}), 400
    if any(
        run["conversation"] == conversation_id and run["status"] in {"queued", "running"}
        for run in RUNS.values()
    ):
        return jsonify({"error": "上一轮还在跑"}), 429
    return jsonify({"run": start_turn(conversation_id, content)})


@app.get("/api/runs/<run_id>")
def run_detail(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        return jsonify({"error": "not found"}), 404
    payload = dict(run)
    payload["live"] = progress.snapshot(run["conversation"])
    return jsonify(payload)


@app.get("/api/memory")
def memory():
    state = read_state()
    conversation = request.args.get("conversation", "")
    session = (state.get("sessions") or {}).get(conversation) or {}
    pointer = session.get("p") or []
    chunks = []
    for chunk in state["chunks"][-120:]:
        index = chunk["i"]
        chunks.append(
            {
                "i": index,
                "doc": chunk["doc"],
                "time": chunk["time"],
                "episode": chunk["episode"],
                "text": chunk["text"],
                "activation": round(pointer[index], 6) if index < len(pointer) else 0.0,
            }
        )
    return jsonify(
        {
            "memory": model.stats(state),
            "encoder": state.get("encoder"),
            "params": model.PARAMS,
            "sessions": {
                key: {"queries": value.get("queries", 0), "pointer": len(value.get("p") or [])}
                for key, value in (state.get("sessions") or {}).items()
            },
            "chunks": chunks,
            "docs": [
                {"id": doc["id"], "time": doc["time"], "chars": len(doc["raw"]), "chunks": len(doc["chunks"])}
                for doc in state["docs"][-60:]
            ],
        }
    )


@app.post("/api/memory/docs")
def add_doc():
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "内容不能为空"}), 400
    state = read_state()
    info = ingest_document(
        state,
        module_instance().encoder,
        raw=text,
        episode_id=body.get("episode") or make_doc_id("ep"),
        doc_id=body.get("doc_id") or make_doc_id("doc"),
    )
    write_state(state, "append-doc")
    return jsonify({"appended": info, "memory": model.stats(state)})


@app.post("/api/memory/reset-pointer")
def reset_pointer():
    body = request.get_json(silent=True) or {}
    conversation = body.get("conversation")
    if not conversation:
        return jsonify({"error": "缺少 conversation"}), 400
    state = read_state()
    model.reset_pointer(state, conversation)
    write_state(state, "reset-pointer")
    return jsonify({"ok": True})


@app.post("/api/memory/seed")
def seed():
    samples = [
        ("上周三我在杭州西湖边上跟老同学吃饭，聊到当年宿舍楼下的那家牛肉面。", "2026-11-04 20:10:00"),
        ("那家牛肉面叫“老周面馆”，老板会多给一勺辣油，我不太能吃辣，每次都让他少放。", "2026-11-04 20:12:00"),
        ("我在做一个记忆模型，核心想法是原文全部保留，靠查询驱动扩散激活去找。", "2026-11-09 22:40:00"),
        ("这个模型最让我意外的是：简单的文档相邻展开，效果和神经元层差不多。", "2026-11-09 22:44:00"),
        ("我的猫叫年糕，三岁，橘色，怕吸尘器，一开就跑床底。", "2026-11-01 09:15:00"),
        ("下个月要去深圳出差三天，住在南山，想顺便去看看海上世界。", "2026-11-12 18:20:00"),
    ]
    state = read_state()
    encoder = module_instance().encoder
    appended = []
    episode = make_doc_id("seed")
    for text, stamp in samples:
        appended.append(
            ingest_document(
                state, encoder, raw=text, time=stamp, episode_id=episode, doc_id=make_doc_id("seed")
            )
        )
    write_state(state, "seed")
    return jsonify({"appended": appended, "memory": model.stats(state)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "3000"))
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
