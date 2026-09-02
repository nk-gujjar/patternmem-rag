"""
patternmem.backends.json_backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Zero-dependency JSON file backend for PatternMem.

This is the **first-class** backend for zero-credential mode
(``backend="json"``, ``eval="none"``, ``observability=None``).
It requires only stdlib + numpy (already a sentence-transformers transitive dep).

Storage layout
--------------
A single JSON file, default ``~/.patternmem/patterns.json``.  The file is an
object mapping ``pattern_id → serialised FailurePattern``.

Concurrency
-----------
An ``asyncio.Lock`` serialises all file reads and writes.  This is sufficient
for a single-process use case; for multi-process workloads use SQLite or Neo4j.

Similarity
----------
Cosine similarity via numpy.  Brute-force over all stored embeddings —
acceptable for the typical pattern-memory sizes (< 10 000 patterns).
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from patternmem.backend import MemoryBackend
from patternmem.types import FailurePattern, FailureType

# Default storage path
_DEFAULT_PATH = Path.home() / ".patternmem" / "patterns.json"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return cosine similarity in [−1, 1] between two vectors."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def _pattern_to_dict(p: FailurePattern) -> dict[str, Any]:
    return {
        "id": p.id,
        "query_embedding": p.query_embedding,
        "failure_type": p.failure_type.name,
        "root_cause": p.root_cause,
        "hint_text": p.hint_text,
        "score": p.score,
        "created_at": p.created_at.isoformat(),
        "decay_weight": p.decay_weight,
    }


def _dict_to_pattern(d: dict[str, Any]) -> FailurePattern:
    return FailurePattern(
        id=d["id"],
        query_embedding=d["query_embedding"],
        failure_type=FailureType[d["failure_type"]],
        root_cause=d["root_cause"],
        hint_text=d["hint_text"],
        score=d["score"],
        created_at=datetime.fromisoformat(d["created_at"]).replace(tzinfo=timezone.utc),
        decay_weight=d["decay_weight"],
    )


class JSONBackend(MemoryBackend):
    """File-backed JSON storage — zero external dependencies beyond numpy.

    Parameters
    ----------
    path:
        Path to the JSON file.  Created automatically if it does not exist.
    similarity_threshold:
        Minimum cosine similarity for a pattern to be returned by
        ``lookup_patterns``.  Should match ``PatternMemMiddleware``'s value.
    """

    def __init__(
        self,
        path: str | Path = _DEFAULT_PATH,
        similarity_threshold: float = 0.82,
    ) -> None:
        self._path = Path(path)
        self._threshold = similarity_threshold
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_dir(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _read_all(self) -> dict[str, dict[str, Any]]:
        """Load the JSON file; return empty dict if file does not exist."""
        if not self._path.exists():
            return {}
        with open(self._path, "r", encoding="utf-8") as fh:
            try:
                data: dict[str, dict[str, Any]] = json.load(fh)
            except json.JSONDecodeError:
                return {}
        return data

    def _write_all(self, data: dict[str, dict[str, Any]]) -> None:
        self._ensure_dir()
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        # Atomic replace
        tmp.replace(self._path)

    # ------------------------------------------------------------------
    # MemoryBackend implementation
    # ------------------------------------------------------------------

    async def write_pattern(self, pattern: FailurePattern) -> None:
        async with self._lock:
            data = self._read_all()
            data[pattern.id] = _pattern_to_dict(pattern)
            self._write_all(data)

    async def lookup_patterns(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[FailurePattern]:
        async with self._lock:
            data = self._read_all()

        if not data:
            return []

        scored: list[tuple[float, FailurePattern]] = []
        for raw in data.values():
            stored_emb: list[float] = raw["query_embedding"]
            if not stored_emb:
                continue
            sim = _cosine_similarity(query_embedding, stored_emb)
            if sim >= self._threshold:
                scored.append((sim, _dict_to_pattern(raw)))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [p for _, p in scored[:top_k]]

    async def get_stats(self) -> dict[str, Any]:
        async with self._lock:
            data = self._read_all()
        return {
            "count": len(data),
            "path": str(self._path),
            "backend": "json",
        }

    async def update_pattern(self, pattern_id: str, decay_weight: float) -> None:
        async with self._lock:
            data = self._read_all()
            if pattern_id not in data:
                raise KeyError(f"Pattern {pattern_id!r} not found in JSON backend")
            data[pattern_id]["decay_weight"] = decay_weight
            self._write_all(data)

    async def delete_pattern(self, pattern_id: str) -> None:
        async with self._lock:
            data = self._read_all()
            if pattern_id not in data:
                raise KeyError(f"Pattern {pattern_id!r} not found in JSON backend")
            del data[pattern_id]
            self._write_all(data)
