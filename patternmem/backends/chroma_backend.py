"""
patternmem.backends.chroma_backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ChromaDB backend for PatternMem.

Uses ChromaDB's native vector index for similarity search — no brute-force
numpy cosine loop is needed.  This makes it the best choice for large
pattern memories (> 10 000 patterns) when you don't need Neo4j's graph model.

Storage layout
--------------
A single ChromaDB collection named ``"patternmem_patterns"`` (configurable).
Pattern metadata (failure_type, root_cause, hint_text, score, created_at,
decay_weight) is stored in Chroma's document metadata dict.
The query embedding is stored as the collection document embedding.

Similarity
----------
ChromaDB runs its own HNSW index for approximate nearest-neighbour search.
PatternMem's ``similarity_threshold`` is applied as a post-filter on the
returned cosine distances (Chroma returns distances not similarities, so
we convert: similarity = 1 − distance).

Requirements
------------
    pip install patternmem-rag[chroma]
    # or
    pip install chromadb>=0.4
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from patternmem.backend import MemoryBackend
from patternmem.types import FailurePattern, FailureType

try:
    import chromadb

    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False


def _pattern_to_chroma(p: FailurePattern) -> dict[str, Any]:
    """Flatten FailurePattern into a Chroma metadata dict (scalar values only)."""
    return {
        "failure_type": p.failure_type.name,
        "root_cause": p.root_cause,
        "hint_text": p.hint_text,
        "score": p.score,
        "created_at": p.created_at.isoformat(),
        "decay_weight": p.decay_weight,
    }


def _chroma_to_pattern(
    pattern_id: str,
    meta: dict[str, Any],
    embedding: list[float],
) -> FailurePattern:
    return FailurePattern(
        id=pattern_id,
        query_embedding=embedding,
        failure_type=FailureType[meta["failure_type"]],
        root_cause=meta["root_cause"],
        hint_text=meta["hint_text"],
        score=float(meta["score"]),
        created_at=datetime.fromisoformat(meta["created_at"]).replace(
            tzinfo=timezone.utc
        ),
        decay_weight=float(meta["decay_weight"]),
    )


class ChromaBackend(MemoryBackend):
    """ChromaDB-backed vector store for PatternMem failure patterns.

    Uses ChromaDB's native HNSW index for fast approximate nearest-neighbour
    search.  Recommended for pattern stores > 10 000 entries or when you
    already run a Chroma server.

    Parameters
    ----------
    collection_name:
        Name of the Chroma collection to use.  Created if it does not exist.
    persist_directory:
        Path for the persistent Chroma client.  ``None`` uses an ephemeral
        (in-memory) client — useful for tests and notebooks.
    host / port:
        If provided, connects to a running Chroma HTTP server instead of a
        local file-backed client.  ``host`` takes precedence over
        ``persist_directory``.
    similarity_threshold:
        Minimum cosine similarity (0–1) for a pattern to be returned.
        Chroma returns L2 or cosine distances; PatternMem converts and filters.
    """

    def __init__(
        self,
        collection_name: str = "patternmem_patterns",
        persist_directory: str | None = None,
        host: str | None = None,
        port: int = 8000,
        similarity_threshold: float = 0.82,
    ) -> None:
        if not _CHROMA_AVAILABLE:
            raise ImportError(
                "ChromaBackend requires 'chromadb'. "
                "Install it with: pip install patternmem-rag[chroma]"
            )
        self._threshold = similarity_threshold
        self._collection_name = collection_name

        if host is not None:
            # Connect to a running Chroma HTTP server
            self._client = chromadb.HttpClient(host=host, port=port)
        elif persist_directory is not None:
            # File-backed persistent client
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            # Ephemeral in-memory client (tests / notebooks)
            self._client = chromadb.EphemeralClient()

        # Get-or-create the collection with cosine similarity space
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # MemoryBackend implementation
    # ------------------------------------------------------------------

    async def write_pattern(self, pattern: FailurePattern) -> None:
        async with self._lock:
            await asyncio.to_thread(self._write_sync, pattern)

    def _write_sync(self, pattern: FailurePattern) -> None:
        """Upsert pattern into the Chroma collection (idempotent on pattern.id)."""
        self._collection.upsert(
            ids=[pattern.id],
            embeddings=[pattern.query_embedding],
            metadatas=[_pattern_to_chroma(pattern)],
            documents=[pattern.root_cause or " "],  # Chroma requires non-empty docs
        )

    async def lookup_patterns(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[FailurePattern]:
        # Fetch more candidates than top_k so the threshold filter has room to work
        n_results = min(top_k * 4, max(1, self._collection.count()))
        if n_results == 0:
            return []

        results = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["embeddings", "metadatas", "distances"],
        )

        ids: list[str] = results["ids"][0]
        metadatas: list[dict[str, Any]] = results["metadatas"][0]
        distances: list[float] = results["distances"][0]
        embeddings: list[list[float]] = results["embeddings"][0]

        # Chroma cosine space returns distances in [0, 2]; convert to similarity
        # similarity = 1 − (distance / 2)  (for cosine distance in [0, 2])
        # OR  similarity = 1 − distance     (for normalized cosine in [0, 1])
        # ChromaDB cosine distance = 1 − cosine_similarity → similarity = 1 − dist
        patterns: list[FailurePattern] = []
        for pid, meta, dist, emb in zip(ids, metadatas, distances, embeddings):
            similarity = 1.0 - float(dist)
            if similarity >= self._threshold:
                patterns.append(_chroma_to_pattern(pid, meta, list(emb)))

        # Already sorted by distance (ascending) = similarity descending
        return patterns[:top_k]

    async def get_stats(self) -> dict[str, Any]:
        count = await asyncio.to_thread(self._collection.count)
        return {
            "count": count,
            "collection": self._collection_name,
            "backend": "chroma",
        }

    async def update_pattern(self, pattern_id: str, decay_weight: float) -> None:
        async with self._lock:
            await asyncio.to_thread(self._update_sync, pattern_id, decay_weight)

    def _update_sync(self, pattern_id: str, decay_weight: float) -> None:
        result = self._collection.get(ids=[pattern_id], include=["metadatas"])
        if not result["ids"]:
            raise KeyError(f"Pattern {pattern_id!r} not found in Chroma backend")
        meta = result["metadatas"][0]
        meta["decay_weight"] = decay_weight
        self._collection.update(ids=[pattern_id], metadatas=[meta])

    async def delete_pattern(self, pattern_id: str) -> None:
        async with self._lock:
            await asyncio.to_thread(self._delete_sync, pattern_id)

    def _delete_sync(self, pattern_id: str) -> None:
        result = self._collection.get(ids=[pattern_id])
        if not result["ids"]:
            raise KeyError(f"Pattern {pattern_id!r} not found in Chroma backend")
        self._collection.delete(ids=[pattern_id])
