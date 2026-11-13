"""M1：查询驱动的神经元—长文本记忆。

所有参数取 M1 的既定取值；所有得分都是原文块上的非负分布。

状态是一份纯 JSON（可直接作为 runtime 模块状态提交），结构：

    {
      "encoder": {"id", "version", "dim"},
      "chunks": [{"i", "doc", "start", "end", "text", "episode", "time", "vec"}],
      "cols":   [{"<neuron_id>": weight, ...}, ...],   # B 的列，按块索引
      "adj":    [[neighbor_chunk, ...], ...],          # A 的邻接（仅前驱）
      "docs":   [{"id", "raw", "time", "episode", "chunks": [...]}],
      "sessions": {"<conversation_id>": {"p": [...], "queries": n}},
    }
"""

from __future__ import annotations

import math
import unicodedata
from typing import Any, Sequence

import numpy as np

# ---- 草案参数（不要随手改；改动会改变判定与历史可比性）----------------------
CHUNK_CHARS = 600
CHUNK_OVERLAP = 80
CHUNK_STRIDE = 520

NEIGHBORS = 3          # r：含自身
KAPPA = 0.9
ETA = 5.0

TOPK_EACH = 15         # L
THETA = 0.65           # 词法权重

ALPHA = 0.27
RHO = 0.16
TAU = 0.15
BETA = 0.43
TOPK_RETURN = 5        # K

BM25_K1 = 1.2
BM25_B = 0.75

MAX_ITER = 100
TOL = 1e-9

PARAMS = {
    "chunk_chars": CHUNK_CHARS,
    "chunk_overlap": CHUNK_OVERLAP,
    "chunk_stride": CHUNK_STRIDE,
    "neighbors": NEIGHBORS,
    "kappa": KAPPA,
    "eta": ETA,
    "topk_each": TOPK_EACH,
    "theta": THETA,
    "alpha": ALPHA,
    "rho": RHO,
    "tau": TAU,
    "beta": BETA,
    "topk_return": TOPK_RETURN,
    "bm25_k1": BM25_K1,
    "bm25_b": BM25_B,
    "max_iter": MAX_ITER,
    "tol": TOL,
}


# --------------------------------------------------------------------- 切块

def split_chunks(raw: str) -> list[tuple[int, int, str]]:
    """按 Unicode 码点切 600 窗口、重叠 80；末块允许不足。"""
    text = raw or ""
    if not text.strip():
        return []
    spans: list[tuple[int, int, str]] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + CHUNK_CHARS, length)
        piece = text[start:end]
        if piece.strip():
            spans.append((start, end, piece))
        if end >= length:
            break
        start += CHUNK_STRIDE
    return spans


# ------------------------------------------------------------------- 分词

def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF
        or 0x3040 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7AF
    )


def tokenize(text: str) -> list[str]:
    """NFKC + casefold；ASCII 词/数字/下划线按整词，汉字取单字与相邻双字。"""
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    tokens: list[str] = []
    word: list[str] = []
    run: list[str] = []

    def flush_word() -> None:
        if word:
            tokens.append("".join(word))
            word.clear()

    def flush_run() -> None:
        if run:
            tokens.extend(run)
            tokens.extend(run[i] + run[i + 1] for i in range(len(run) - 1))
            run.clear()

    for char in normalized:
        if char.isalnum() and not _is_cjk(char) or char == "_":
            flush_run()
            word.append(char)
        elif _is_cjk(char):
            flush_word()
            run.append(char)
        else:
            flush_word()
            flush_run()
    flush_word()
    flush_run()
    return tokens


# ------------------------------------------------------------------ 基础工具

def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.dot(a, b))


def _unit(vector: Sequence[float]) -> list[float] | None:
    array = np.asarray(vector, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        return None
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm == 0.0:
        return None
    return (array / norm).tolist()


# ------------------------------------------------------------- 写入（append）

def append_document(
    state: dict,
    *,
    doc_id: str,
    raw: str,
    time: str,
    episode_id: str,
    vectors: Sequence[Sequence[float] | None],
) -> dict:
    """把一份原文写入记忆；vectors 与 split_chunks 一一对应。"""
    spans = split_chunks(raw)
    if not spans:
        return {"doc_id": doc_id, "chunks": 0, "edges": 0, "skipped": "empty"}

    existing = len(state["chunks"])
    new_vectors = []
    for offset, (start, end, piece) in enumerate(spans):
        vector = vectors[offset] if offset < len(vectors) else None
        new_vectors.append(_unit(vector) if vector else None)

    new_chunk_ids: list[int] = []
    links = 0
    for offset, (start, end, piece) in enumerate(spans):
        index = len(state["chunks"])
        vector = new_vectors[offset]
        state["chunks"].append(
            {
                "i": index,
                "doc": doc_id,
                "start": start,
                "end": end,
                "text": piece,
                "episode": episode_id,
                "time": time,
                "vec": vector or [],
            }
        )
        new_chunk_ids.append(index)

        column: dict[str, float] = {}
        if vector is None:
            column[str(index)] = 1.0
        else:
            scored: list[tuple[float, int]] = []
            for other in range(index):
                other_vec = state["chunks"][other]["vec"]
                if not other_vec:
                    continue
                similarity = dot(vector, other_vec)
                if similarity > 0.0:
                    scored.append((similarity, other))
            scored.sort(key=lambda item: (-item[0], item[1]))
            chosen = scored[: NEIGHBORS - 1]
            if not chosen:
                column[str(index)] = 1.0
            else:
                total = sum(similarity for similarity, _ in chosen)
                column[str(index)] = KAPPA
                for similarity, other in chosen:
                    column[str(other)] = round((1.0 - KAPPA) * similarity / total, 12)
                links += len(chosen)
        state["cols"].append(column)

        neighbours: list[int] = []
        if index > 0 and state["chunks"][index - 1]["episode"] == episode_id:
            neighbours.append(index - 1)
        state["adj"].append(neighbours)

    state["docs"].append(
        {
            "id": doc_id,
            "raw": raw,
            "time": time,
            "episode": episode_id,
            "chunks": new_chunk_ids,
        }
    )
    return {
        "doc_id": doc_id,
        "chunks": len(new_chunk_ids),
        "chunk_ids": new_chunk_ids,
        "sparse_links": links,
        "first_index": existing,
    }


# ------------------------------------------------------------------ 矩阵

def _build_b(state: dict) -> np.ndarray:
    size = len(state["chunks"])
    matrix = np.zeros((size, size), dtype=np.float64)
    for column_index, column in enumerate(state["cols"]):
        for row_key, weight in column.items():
            row_index = int(row_key)
            if 0 <= row_index < size and 0 <= column_index < size:
                matrix[row_index, column_index] = float(weight)
    return matrix


def _build_t(state: dict, b_matrix: np.ndarray) -> np.ndarray:
    """T = W diag(列和)^-1，W = BB^T + η BAB^T。"""
    size = b_matrix.shape[0]
    w_matrix = b_matrix @ b_matrix.T
    for column_index, neighbours in enumerate(state["adj"]):
        for other in neighbours:
            outer = np.outer(b_matrix[:, other], b_matrix[:, column_index])
            w_matrix += ETA * (outer + outer.T)
    sums = w_matrix.sum(axis=0)
    sums[sums <= 0.0] = 1.0
    return w_matrix / sums


# ------------------------------------------------------------------ 检索

def _bm25(state: dict, terms: list[str]) -> dict[int, float]:
    chunks = state["chunks"]
    total = len(chunks)
    if total == 0 or not terms:
        return {}
    docs_tokens = [tokenize(chunk["text"]) for chunk in chunks]
    lengths = np.array([len(tokens) for tokens in docs_tokens], dtype=np.float64)
    average = float(lengths.mean()) if total else 0.0
    if average <= 0.0:
        return {}

    df: dict[str, int] = {}
    for tokens in docs_tokens:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1

    scores: dict[int, float] = {}
    unique_terms = list(dict.fromkeys(terms))
    for index, tokens in enumerate(docs_tokens):
        if not tokens:
            continue
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        score = 0.0
        for term in unique_terms:
            frequency = counts.get(term)
            if not frequency:
                continue
            idf = math.log(1.0 + (total - df[term] + 0.5) / (df[term] + 0.5))
            denominator = frequency + BM25_K1 * (
                1.0 - BM25_B + BM25_B * lengths[index] / average
            )
            score += idf * frequency * (BM25_K1 + 1.0) / denominator
        if score > 0.0:
            scores[index] = score
    return scores


def _top_normalized(scores: dict[int, float], limit: int = TOPK_EACH) -> dict[int, float]:
    positive = [(index, value) for index, value in scores.items() if value > 0.0]
    positive.sort(key=lambda item: (-item[1], item[0]))
    chosen = positive[:limit]
    total = sum(value for _, value in chosen)
    if total <= 0.0:
        return {}
    return {index: value / total for index, value in chosen}


def _semantic(state: dict, query_vector: Sequence[float] | None) -> dict[int, float]:
    if query_vector is None:
        return {}
    unit_query = _unit(query_vector)
    if unit_query is None:
        return {}
    q = np.asarray(unit_query)
    scores: dict[int, float] = {}
    for chunk in state["chunks"]:
        if not chunk["vec"]:
            continue
        similarity = max(0.0, float(np.dot(q, np.asarray(chunk["vec"]))))
        if similarity > 0.0:
            scores[chunk["i"]] = similarity ** 5
    return scores


def retrieve(
    state: dict,
    *,
    query: str,
    query_vector: Sequence[float] | None,
    session_id: str,
    preview_chars: int = 120,
) -> dict[str, Any]:
    """返回 z 排序的前 K 块与全部中间量；不修改 state。"""
    chunks = state["chunks"]
    if not chunks:
        return {"empty": "no-memory", "retrieved": [], "trace": {}}

    terms = tokenize(query)
    lexical_raw = _bm25(state, terms)
    semantic_raw = _semantic(state, query_vector)
    lexical = _top_normalized(lexical_raw)
    semantic = _top_normalized(semantic_raw)

    if not lexical and not semantic:
        return {
            "empty": "no-evidence",
            "retrieved": [],
            "trace": {
                "terms": terms,
                "lexical": {"candidates": len(lexical_raw), "hits": []},
                "semantic": {"candidates": len(semantic_raw), "hits": []},
            },
        }

    size = len(chunks)
    if lexical and semantic:
        fused = np.zeros(size)
        for index, value in lexical.items():
            fused[index] += THETA * value
        for index, value in semantic.items():
            fused[index] += (1.0 - THETA) * value
        route = "lexical+semantic"
    elif lexical:
        fused = np.zeros(size)
        for index, value in lexical.items():
            fused[index] = value
        route = "lexical"
    else:
        fused = np.zeros(size)
        for index, value in semantic.items():
            fused[index] = value
        route = "semantic"

    order = np.argsort(-fused, kind="stable")
    d1 = float(fused[order[0]]) if size else 0.0
    d2 = float(fused[order[1]]) if size > 1 else 0.0

    session = state["sessions"].get(session_id) or {}
    previous = np.asarray(session.get("p") or [], dtype=np.float64)
    has_history = bool(previous.size)

    if has_history:
        gamma = RHO * max(0.0, 1.0 - (d1 - d2) / TAU) if TAU > 0 else 0.0
    else:
        gamma = 0.0

    b_matrix = _build_b(state)
    seed = b_matrix @ fused

    if has_history:
        pointer = previous
        if pointer.size < size:
            pointer = np.concatenate([pointer, np.zeros(size - pointer.size)])
        elif pointer.size > size:
            pointer = pointer[:size]
        start = (1.0 - gamma) * seed + gamma * pointer
    else:
        pointer = np.zeros(size)
        start = seed.copy()

    t_matrix = _build_t(state, b_matrix)
    activation = start.copy()
    iterations = 0
    residual = float("inf")
    for step in range(1, MAX_ITER + 1):
        nxt = ALPHA * start + (1.0 - ALPHA) * (t_matrix @ activation)
        residual = float(np.abs(nxt - activation).sum())
        activation = nxt
        iterations = step
        if residual <= TOL:
            break

    row_sums = b_matrix.sum(axis=1)
    row_sums[row_sums <= 0.0] = 1.0
    read = (b_matrix / row_sums[:, None]).T @ activation

    z = (1.0 - BETA) * fused + BETA * read
    ranked = sorted(range(size), key=lambda i: (-float(z[i]), i))[:TOPK_RETURN]

    retrieved = []
    for index in ranked:
        chunk = chunks[index]
        if z[index] <= 0.0:
            continue
        retrieved.append(
            {
                "chunk": index,
                "doc": chunk["doc"],
                "time": chunk["time"],
                "episode": chunk["episode"],
                "text": chunk["text"],
                "preview": chunk["text"][:preview_chars],
                "direct": round(float(fused[index]), 6),
                "neural": round(float(read[index]), 6),
                "total": round(float(z[index]), 6),
                "start": chunk["start"],
                "end": chunk["end"],
            }
        )

    def hits(mapping: dict[int, float]) -> list[dict]:
        rows = sorted(mapping.items(), key=lambda item: (-item[1], item[0]))
        return [
            {
                "chunk": index,
                "score": round(value, 6),
                "preview": chunks[index]["text"][:preview_chars],
                "doc": chunks[index]["doc"],
            }
            for index, value in rows[:TOPK_EACH]
        ]

    top_neurons = [
        {"neuron": int(index), "value": round(float(activation[index]), 8)}
        for index in np.argsort(-activation)[:8]
        if activation[index] > 0
    ]

    trace = {
        "route": route,
        "terms": terms,
        "lexical": {"candidates": len(lexical_raw), "hits": hits(lexical)},
        "semantic": {"candidates": len(semantic_raw), "hits": hits(semantic)},
        "fused": {
            "d1": round(d1, 6),
            "d2": round(d2, 6),
            "hits": hits({int(index): float(fused[index]) for index in np.nonzero(fused)[0]}),
        },
        "gate": {
            "has_history": has_history,
            "gamma": round(gamma, 6),
            "rho": RHO,
            "tau": TAU,
            "previous_top": [
                {"chunk": int(index), "value": round(float(pointer[index]), 6)}
                for index in np.argsort(-pointer)[:3]
                if pointer.size and pointer[index] > 0
            ],
        },
        "seed": {"top": top_neurons[:5], "mass": round(float(seed.sum()), 6)},
        "diffusion": {
            "alpha": ALPHA,
            "iterations": iterations,
            "residual": f"{residual:.2e}",
            "activation_top": top_neurons,
            "activation_mass": round(float(activation.sum()), 6),
        },
        "matrix": {
            "chunks": size,
            "columns": len(state["cols"]),
            "edges": int(sum(len(neighbours) for neighbours in state["adj"])),
            "nonzeros": int(np.count_nonzero(b_matrix)),
        },
    }
    return {"retrieved": retrieved, "trace": trace, "activation": activation.tolist()}


def commit_pointer(state: dict, session_id: str, activation: Sequence[float]) -> None:
    """成功检索后更新同会话的跨轮指针；失败/空检索不调用。"""
    session = state["sessions"].setdefault(session_id, {"p": [], "queries": 0})
    session["p"] = [round(float(value), 10) for value in activation]
    session["queries"] = int(session.get("queries", 0)) + 1


def reset_pointer(state: dict, session_id: str) -> None:
    session = state["sessions"].get(session_id)
    if session:
        session["p"] = []


def new_state(encoder: dict) -> dict:
    return {
        "encoder": encoder,
        "chunks": [],
        "cols": [],
        "adj": [],
        "docs": [],
        "sessions": {},
    }


def stats(state: dict) -> dict:
    chunks = state["chunks"]
    return {
        "docs": len(state["docs"]),
        "chunks": len(chunks),
        "edges": int(sum(len(neighbours) for neighbours in state["adj"])),
        "neurons": len(chunks),
        "sessions": len(state["sessions"]),
        "chars": sum(len(chunk["text"]) for chunk in chunks),
    }
