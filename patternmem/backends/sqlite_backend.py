"""
patternmem.backends.sqlite_backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SQLite backend for PatternMem using aiosqlite for async I/O.

Storage layout
--------------
A single SQLite database file (WAL mode) with a ``failure_patterns`` table.
The query embedding is stored as a JSON text column — no extension required.

Similarity
----------
All embeddings are loaded into numpy for brute-force cosine similarity.
This is acceptable for typical pattern-memory sizes (< 10 000 patterns).
For larger workloads, use Neo4j with its native vector index.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json
import numpy as np
import aiosqlite

from patternmem.backend import MemoryBackend
from patternmem.types import FailurePattern, FailureType

_DEFAULT_PATH = Path.home() / ".patternmem" / "patterns.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS failure_patterns (
    id           TEXT PRIMARY KEY,
    query_embedding TEXT NOT NULL,          -- JSON array of floats
    failure_type TEXT NOT NULL,
    root_cause   TEXT NOT NULL DEFAULT '',
    hint_text    TEXT NOT NULL DEFAULT '',
    score        REAL NOT NULL DEFAULT 0.0,
    created_at   TEXT NOT NULL,             -- ISO-8601 UTC
    decay_weight REAL NOT NULL DEFAULT 1.0
)
"""


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def _row_to_pattern(row: aiosqlite.Row) -> FailurePattern:
    return FailurePattern(
        id=row["id"],
        query_embedding=json.loads(row["query_embedding"]),
        failure_type=FailureType[row["failure_type"]],
        root_cause=row["root_cause"],
        hint_text=row["hint_text"],
        score=row["score"],
        created_at=datetime.fromisoformat(row["created_at"]).replace(tzinfo=timezone.utc),
        decay_weight=row["decay_weight"],
    )


class SQLiteBackend(MemoryBackend):
    """Async SQLite backend via ``aiosqlite``.

    Each public method opens a fresh ``aiosqlite`` connection using
    ``aiosqlite.connect()`` as an async context manager, performs its
    operation, and lets the context manager close it cleanly.  This is the
    correct aiosqlite usage pattern and avoids the "thread can only be started
    once" error that arises from reusing a connection object.

    Parameters
    ----------
    path:
        Path to the SQLite file.  Created automatically if absent.
    similarity_threshold:
        Minimum cosine similarity for lookup.
    """

    def __init__(
        self,
        path: str | Path = _DEFAULT_PATH,
        similarity_threshold: float = 0.82,
    ) -> None:
        self._path = Path(path)
        self._threshold = similarity_threshold

    def _ensure_dir(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

    async def _init_schema(self, conn: aiosqlite.Connection) -> None:
        """Enable WAL and create the table if needed."""
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute(_CREATE_TABLE)
        await conn.commit()

    async def write_pattern(self, pattern: FailurePattern) -> None:
        self._ensure_dir()
        async with aiosqlite.connect(str(self._path)) as conn:
            conn.row_factory = aiosqlite.Row
            await self._init_schema(conn)
            await conn.execute(
                """
                INSERT INTO failure_patterns
                    (id, query_embedding, failure_type, root_cause,
                     hint_text, score, created_at, decay_weight)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    query_embedding = excluded.query_embedding,
                    failure_type    = excluded.failure_type,
                    root_cause      = excluded.root_cause,
                    hint_text       = excluded.hint_text,
                    score           = excluded.score,
                    created_at      = excluded.created_at,
                    decay_weight    = excluded.decay_weight
                """,
                (
                    pattern.id,
                    json.dumps(pattern.query_embedding),
                    pattern.failure_type.name,
                    pattern.root_cause,
                    pattern.hint_text,
                    pattern.score,
                    pattern.created_at.isoformat(),
                    pattern.decay_weight,
                ),
            )
            await conn.commit()

    async def lookup_patterns(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[FailurePattern]:
        self._ensure_dir()
        async with aiosqlite.connect(str(self._path)) as conn:
            conn.row_factory = aiosqlite.Row
            await self._init_schema(conn)
            async with conn.execute(
                "SELECT * FROM failure_patterns"
            ) as cursor:
                rows = await cursor.fetchall()

        if not rows:
            return []

        scored: list[tuple[float, FailurePattern]] = []
        for row in rows:
            stored_emb: list[float] = json.loads(row["query_embedding"])
            if not stored_emb:
                continue
            sim = _cosine_similarity(query_embedding, stored_emb)
            if sim >= self._threshold:
                scored.append((sim, _row_to_pattern(row)))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [p for _, p in scored[:top_k]]

    async def get_stats(self) -> dict[str, Any]:
        self._ensure_dir()
        async with aiosqlite.connect(str(self._path)) as conn:
            conn.row_factory = aiosqlite.Row
            await self._init_schema(conn)
            async with conn.execute(
                "SELECT COUNT(*) as cnt FROM failure_patterns"
            ) as cursor:
                row = await cursor.fetchone()
        return {
            "count": int(row["cnt"]) if row else 0,
            "path": str(self._path),
            "backend": "sqlite",
        }

    async def update_pattern(self, pattern_id: str, decay_weight: float) -> None:
        self._ensure_dir()
        async with aiosqlite.connect(str(self._path)) as conn:
            conn.row_factory = aiosqlite.Row
            await self._init_schema(conn)
            result = await conn.execute(
                "UPDATE failure_patterns SET decay_weight = ? WHERE id = ?",
                (decay_weight, pattern_id),
            )
            await conn.commit()
            if result.rowcount == 0:
                raise KeyError(f"Pattern {pattern_id!r} not found in SQLite backend")

    async def delete_pattern(self, pattern_id: str) -> None:
        self._ensure_dir()
        async with aiosqlite.connect(str(self._path)) as conn:
            conn.row_factory = aiosqlite.Row
            await self._init_schema(conn)
            result = await conn.execute(
                "DELETE FROM failure_patterns WHERE id = ?",
                (pattern_id,),
            )
            await conn.commit()
            if result.rowcount == 0:
                raise KeyError(f"Pattern {pattern_id!r} not found in SQLite backend")
