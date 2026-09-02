"""
patternmem.backends.networkx_backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In-memory NetworkX DiGraph backend for PatternMem.

Ideal for experimentation, notebooks, and unit tests that need a
zero-file-I/O graph-native store.  Optionally serialises to GraphML
via ``persist_path`` for lightweight persistence across restarts.

Similarity
----------
Cosine similarity via numpy — identical algorithm to ``JSONBackend``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import networkx as nx

    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False

from patternmem.backend import MemoryBackend
from patternmem.types import FailurePattern, FailureType


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def _pattern_to_attrs(p: FailurePattern) -> dict[str, Any]:
    """Flatten a FailurePattern into a NetworkX node-attribute dict."""
    return {
        "query_embedding": p.query_embedding,
        "failure_type": p.failure_type.name,
        "root_cause": p.root_cause,
        "hint_text": p.hint_text,
        "score": p.score,
        "created_at": p.created_at.isoformat(),
        "decay_weight": p.decay_weight,
    }


def _attrs_to_pattern(node_id: str, attrs: dict[str, Any]) -> FailurePattern:
    return FailurePattern(
        id=node_id,
        query_embedding=attrs["query_embedding"],
        failure_type=FailureType[attrs["failure_type"]],
        root_cause=attrs["root_cause"],
        hint_text=attrs["hint_text"],
        score=attrs["score"],
        created_at=datetime.fromisoformat(attrs["created_at"]).replace(
            tzinfo=timezone.utc
        ),
        decay_weight=attrs["decay_weight"],
    )


class NetworkXBackend(MemoryBackend):
    """In-memory graph backend backed by a ``networkx.DiGraph``.

    Parameters
    ----------
    persist_path:
        Optional path to a GraphML file.  If provided the graph is loaded on
        construction and saved on every write/update/delete.  ``None`` means
        pure in-memory (no persistence).
    similarity_threshold:
        Minimum cosine similarity for lookup.
    """

    def __init__(
        self,
        persist_path: str | Path | None = None,
        similarity_threshold: float = 0.82,
    ) -> None:
        if not _NX_AVAILABLE:
            raise ImportError(
                "NetworkXBackend requires 'networkx'. "
                "Install it with: pip install patternmem-rag[networkx]"
            )
        self._threshold = similarity_threshold
        self._persist_path = Path(persist_path) if persist_path else None
        self._lock = asyncio.Lock()

        self._graph: Any = nx.DiGraph()
        if self._persist_path and self._persist_path.exists():
            self._graph = nx.read_graphml(str(self._persist_path))
            # GraphML deserialises embeddings as strings; re-parse them
            for nid in list(self._graph.nodes):
                raw_emb = self._graph.nodes[nid].get("query_embedding", "[]")
                if isinstance(raw_emb, str):
                    import json
                    self._graph.nodes[nid]["query_embedding"] = json.loads(raw_emb)

    def _persist(self) -> None:
        if self._persist_path:
            import json
            # GraphML can't store lists natively — serialise embeddings to JSON str
            tmp_graph: Any = self._graph.copy()
            for nid in tmp_graph.nodes:
                emb = tmp_graph.nodes[nid].get("query_embedding", [])
                tmp_graph.nodes[nid]["query_embedding"] = json.dumps(emb)
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            nx.write_graphml(tmp_graph, str(self._persist_path))

    async def write_pattern(self, pattern: FailurePattern) -> None:
        async with self._lock:
            self._graph.add_node(pattern.id, **_pattern_to_attrs(pattern))
            self._persist()

    async def lookup_patterns(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[FailurePattern]:
        async with self._lock:
            scored: list[tuple[float, FailurePattern]] = []
            for nid, attrs in self._graph.nodes(data=True):
                stored_emb: list[float] = attrs.get("query_embedding", [])
                if not stored_emb:
                    continue
                sim = _cosine_similarity(query_embedding, stored_emb)
                if sim >= self._threshold:
                    scored.append((sim, _attrs_to_pattern(nid, attrs)))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [p for _, p in scored[:top_k]]

    async def get_stats(self) -> dict[str, Any]:
        async with self._lock:
            count = self._graph.number_of_nodes()
        return {
            "count": count,
            "persist_path": str(self._persist_path) if self._persist_path else None,
            "backend": "networkx",
        }

    async def update_pattern(self, pattern_id: str, decay_weight: float) -> None:
        async with self._lock:
            if pattern_id not in self._graph:
                raise KeyError(f"Pattern {pattern_id!r} not found in NetworkX backend")
            self._graph.nodes[pattern_id]["decay_weight"] = decay_weight
            self._persist()

    async def delete_pattern(self, pattern_id: str) -> None:
        async with self._lock:
            if pattern_id not in self._graph:
                raise KeyError(f"Pattern {pattern_id!r} not found in NetworkX backend")
            self._graph.remove_node(pattern_id)
            self._persist()
