"""编码器：Gemini 批量 embedding，失败时退化为确定性字符哈希。

草案要求编码器固定“名称 / 版本 / 维度”，写入时保存嵌入。这里把三者写进
记忆状态，任何一次回落都会在 trace 和块记录里留下 `encoder` 标记。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from typing import Any, Sequence

import numpy as np

BATCH_LIMIT = 100
RETRY_STATUS = {429, 500, 502, 503, 504}
BACKOFF = (3.0, 8.0, 20.0)


def _normalize(vector: Sequence[float]) -> list[float] | None:
    array = np.asarray(vector, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        return None
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm == 0.0:
        return None
    return (array / norm).tolist()


def hashing_embedding(text: str, dimension: int) -> list[float]:
    """确定性 2–3 元字符哈希嵌入；零依赖，用作回路。"""
    vector = np.zeros(dimension, dtype=np.float64)
    normalized = (text or "").strip().lower()
    if not normalized:
        return vector.tolist()
    grams: list[str] = []
    for size in (2, 3):
        for index in range(max(0, len(normalized) - size + 1)):
            grams.append(normalized[index : index + size])
    if not grams:
        grams = [normalized]
    for gram in grams:
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        slot = value % dimension
        sign = 1.0 if (value >> 63) & 1 else -1.0
        vector[slot] += sign
    return _normalize(vector) or vector.tolist()


class Encoder:
    """一次调用可以编码一批文本，返回与输入等长的向量列表。"""

    def __init__(
        self,
        *,
        protocol: str = "gemini",
        base_url: str = "",
        api_key: str = "",
        model: str = "gemini-embedding-001",
        dimension: int = 768,
        proxy: str | None = None,
        timeout: float = 60.0,
    ):
        self.protocol = protocol
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.dimension = dimension
        self.proxy = proxy
        self.timeout = timeout
        self.remote = bool(self.api_key and self.base_url and protocol != "hashing")
        self.id = model if self.remote else "hash-ngram"
        self.version = "batch-1" if self.remote else "b2-3"
        self.last: dict[str, Any] = {}
        self._cache: dict[str, list[float]] = {}

    # ------------------------------------------------------------------ public

    def embed_many(self, texts: Sequence[str]) -> list[list[float] | None]:
        started = time.time()
        pending = [text for text in texts if text and text not in self._cache]
        fallback = False
        error = ""
        if pending and self.remote:
            try:
                for offset in range(0, len(pending), BATCH_LIMIT):
                    batch = pending[offset : offset + BATCH_LIMIT]
                    vectors = self._remote_batch(batch)
                    for text, vector in zip(batch, vectors):
                        if vector is not None:
                            self._cache[text] = vector
                missing = [text for text in pending if text not in self._cache]
                if missing:
                    fallback = True
                    for text in missing:
                        self._cache[text] = hashing_embedding(text, self.dimension)
            except Exception as exc:  # noqa: BLE001 - 回落比整轮失败好
                fallback = True
                error = f"{type(exc).__name__}: {exc}"[:200]
                for text in pending:
                    if text not in self._cache:
                        self._cache[text] = hashing_embedding(text, self.dimension)
        elif pending:
            fallback = not self.remote
            for text in pending:
                self._cache[text] = hashing_embedding(text, self.dimension)

        self.last = {
            "encoder": self.id,
            "version": self.version,
            "dim": self.dimension,
            "texts": len(texts),
            "remote_calls": 0 if not self.remote else math.ceil(len(pending) / BATCH_LIMIT),
            "fallback": fallback,
            "error": error,
            "ms": int((time.time() - started) * 1000),
        }
        return [self._cache.get(text) if text else None for text in texts]

    def embed(self, text: str) -> list[float] | None:
        return self.embed_many([text])[0]

    def info(self) -> dict:
        return {
            "id": self.id,
            "version": self.version,
            "dim": self.dimension,
            "remote": self.remote,
        }

    # ----------------------------------------------------------------- private

    def _remote_batch(self, texts: Sequence[str]) -> list[list[float] | None]:
        payload = {
            "requests": [
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": text}]},
                    "outputDimensionality": self.dimension,
                }
                for text in texts
            ]
        }
        url = f"{self.base_url}/models/{self.model}:batchEmbedContents"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": self.proxy, "https": self.proxy})
            if self.proxy
            else urllib.request.ProxyHandler({})
        )
        for attempt in range(len(BACKOFF) + 1):
            request = urllib.request.Request(url, body, headers)
            try:
                with opener.open(request, timeout=self.timeout) as response:
                    result = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in RETRY_STATUS or attempt >= len(BACKOFF):
                    raise
                time.sleep(BACKOFF[attempt])
        else:  # pragma: no cover - 循环必然 break 或 raise
            raise RuntimeError("embedding request failed")

        embeddings = result.get("embeddings") or []
        vectors: list[list[float] | None] = []
        for item in embeddings:
            values = item.get("values") if isinstance(item, dict) else None
            vectors.append(_normalize(values) if values else None)
        while len(vectors) < len(texts):
            vectors.append(None)
        return vectors[: len(texts)]


def from_env() -> Encoder:
    """从 AGENT_EMBED_* 环境变量构造；缺失时自动退化为哈希编码。"""
    protocol = os.environ.get("AGENT_EMBED_PROTOCOL", "gemini")
    return Encoder(
        protocol=protocol,
        base_url=os.environ.get("AGENT_EMBED_BASE", ""),
        api_key=os.environ.get("AGENT_EMBED_KEY", ""),
        model=os.environ.get("AGENT_EMBED_MODEL", "gemini-embedding-001"),
        dimension=int(os.environ.get("AGENT_EMBED_DIMENSIONS", "768")),
        proxy=os.environ.get("AGENT_EMBED_PROXY") or None,
        timeout=float(os.environ.get("AGENT_EMBED_TIMEOUT", "60")),
    )
