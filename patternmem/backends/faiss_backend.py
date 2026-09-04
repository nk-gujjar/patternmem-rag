"""
patternmem.backends.faiss_backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FAISS (Facebook AI Similarity Search) backend for PatternMem.

Uses a ``faiss.IndexFlatIP`` (inner-product) index on L2-normalised embeddings,
which is mathematically equivalent to cosine similarity search.  FAISS is
extremely fast for large pattern memories and runs fully locally with no server.

Storage layout
--------------
Two files, both in the same directory:

- ``<name>.index``  — the FAISS binary index (embeddings only, sequential rows)
- ``<name>.meta.json``  — JSON sidecar: a list of FailurePattern dicts in the
  same row order as the FAISS index.

Row positions in the FAISS index map 1-to-1 with list indices in the sidecar.
On delete/update, the sidecar entry is marked ``null`` and the row is
logically ignored during lookup (a compaction rebuilds the index if needed).

Why not IndexIDMap?
-------------------
``faiss.IndexIDMap`` has a known segfault on macOS Apple Silicon with certain
builds.  Using a plain ``IndexFlatIP`` plus a positional sidecar avoids
the issue entirely while keeping the same search semantics.

Similarity
----------
``IndexFlatIP`` returns the exact inner product.  Since embeddings are
L2-normalised by MiniLM before being passed to PatternMem, inner product ==
cosine similarity.  The ``similarity_threshold`` is applied as a post-filter.

Requirements
------------
    pip install patternmem-rag[faiss]
    # or
    pip install faiss-cpu>=1.7   # CPU-only (recommended)
    # pip install faiss-gpu>=1.7  # GPU version (if CUDA is available)
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from patternmem.backend import MemoryBackend
from patternmem.types import FailurePattern, FailureType

try:
    import faiss  # type: ignore[import]

    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

# Default directory for FAISS index files
_DEFAULT_DIR = Path.home() / ".patternmem"
_DEFAULT_NAME = "patterns_faiss"

# MiniLM embedding dimension (must match PatternMemMiddleware._EMBED_MODEL)
_DIM = 384

# Sentinel for a deleted/unused row in the sidecar
_DELETED: None = None


def _to_float32(embedding: list[float]) -> "np.ndarray[Any, np.dtype[np.float32]]":
    """Convert a list of floats to a normalised float32 row-vector."""
    arr = np.array(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr /= norm
    return arr.reshape(1, -1)


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
        score=float(d["score"]),
        created_at=datetime.fromisoformat(d["created_at"]).replace(
            tzinfo=timezone.utc
        ),
        decay_weight=float(d["decay_weight"]),
    )


class FAISSBackend(MemoryBackend):
    """FAISS-backed vector index for PatternMem failure patterns.

    Extremely fast exact nearest-neighbour search, fully local, no server.
    Best for large pattern memories where brute-force numpy becomes slow.

    Uses a plain ``faiss.IndexFlatIP`` (no IndexIDMap) for maximum
    compatibility across platforms, including macOS Apple Silicon.

    Parameters
    ----------
    directory:
        Directory where the index (``.index``) and metadata (``.meta.json``)
        files are stored.  Created automatically if absent.
    name:
        Base name for the two files.  Defaults to ``"patterns_faiss"``.
    similarity_threshold:
        Minimum cosine similarity for a pattern to be returned.
    compaction_threshold:
        When the fraction of deleted (null) rows in the index exceeds this
        value, a full compaction is triggered on the next write.
        Default: 0.3 (compact when 30 %+ of rows are tombstones).
    """

    def __init__(
        self,
        directory: str | Path = _DEFAULT_DIR,
        name: str = _DEFAULT_NAME,
        similarity_threshold: float = 0.82,
        compaction_threshold: float = 0.3,
    ) -> None:
        if not _FAISS_AVAILABLE:
            raise ImportError(
                "FAISSBackend requires 'faiss-cpu' (or 'faiss-gpu'). "
                "Install it with: pip install patternmem-rag[faiss]"
            )
        self._threshold = similarity_threshold
        self._compaction_threshold = compaction_threshold
        self._dir = Path(directory)
        self._index_path = self._dir / f"{name}.index"
        self._meta_path = self._dir / f"{name}.meta.json"
        self._lock = asyncio.Lock()

        # sidecar: list of (pattern_dict | None)
        # None = row deleted / not in use
        self._sidecar: list[dict[str, Any] | None] = self._load_meta()

        # Rebuild index from sidecar on startup
        self._index: Any = self._build_index_from_sidecar()

        # id_to_row: pattern UUID → row index in the sidecar/index
        self._id_to_row: dict[str, int] = {
            d["id"]: i
            for i, d in enumerate(self._sidecar)
            if d is not None
        }

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _ensure_dir(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)

    def _load_meta(self) -> list[dict[str, Any] | None]:
        if not self._meta_path.exists():
            return []
        with open(self._meta_path, "r", encoding="utf-8") as fh:
            try:
                raw = json.load(fh)
                return raw  # list of dicts or nulls
            except json.JSONDecodeError:
                return []

    def _build_index_from_sidecar(self) -> Any:
        """Rebuild a fresh IndexFlatIP from all non-null sidecar rows."""
        idx = faiss.IndexFlatIP(_DIM)
        active = [d for d in self._sidecar if d is not None]
        if active:
            vecs = np.vstack(
                [_to_float32(d["query_embedding"]) for d in active]
            ).astype(np.float32)
            idx.add(vecs)
        return idx

    def _save(self) -> None:
        """Atomically persist the FAISS index and metadata sidecar."""
        self._ensure_dir()
        faiss.write_index(self._index, str(self._index_path))
        tmp = self._meta_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._sidecar, fh, indent=2)
        tmp.replace(self._meta_path)

    def _compact(self) -> None:
        """Remove all null rows, rebuild the index and reset id_to_row."""
        active = [(i, d) for i, d in enumerate(self._sidecar) if d is not None]
        self._sidecar = [d for _, d in active]
        self._index = self._build_index_from_sidecar()
        self._id_to_row = {d["id"]: new_i for new_i, d in enumerate(self._sidecar)}

    def _should_compact(self) -> bool:
        total = len(self._sidecar)
        if total == 0:
            return False
        nulls = sum(1 for d in self._sidecar if d is None)
        return nulls / total >= self._compaction_threshold

    # ------------------------------------------------------------------
    # MemoryBackend implementation
    # ------------------------------------------------------------------

    async def write_pattern(self, pattern: FailurePattern) -> None:
        async with self._lock:
            pid = pattern.id

            if pid in self._id_to_row:
                # Mark old row as deleted (tombstone), then append fresh entry
                old_row = self._id_to_row[pid]
                self._sidecar[old_row] = None
                del self._id_to_row[pid]

            if self._should_compact():
                self._compact()

            # Append new vector and metadata
            vec = _to_float32(pattern.query_embedding)
            self._index.add(vec)
            new_row = len(self._sidecar)
            self._sidecar.append(_pattern_to_dict(pattern))
            self._id_to_row[pid] = new_row
            self._save()

    async def lookup_patterns(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[FailurePattern]:
        async with self._lock:
            active_count = self._index.ntotal
            if active_count == 0:
                return []

            vec = _to_float32(query_embedding)
            k = min(top_k * 4, active_count)
            scores_arr, rows_arr = self._index.search(vec, k)

            # Map FAISS sequential row positions back to sidecar positions
            # After compaction, FAISS rows align with non-null sidecar entries
            active_rows = [i for i, d in enumerate(self._sidecar) if d is not None]

            results: list[tuple[float, FailurePattern]] = []
            for faiss_pos, score in zip(rows_arr[0].tolist(), scores_arr[0].tolist()):
                if faiss_pos < 0 or faiss_pos >= len(active_rows):
                    continue
                sidecar_idx = active_rows[faiss_pos]
                d = self._sidecar[sidecar_idx]
                if d is None:
                    continue
                similarity = float(score)
                if similarity >= self._threshold:
                    results.append((similarity, _dict_to_pattern(d)))

            results.sort(key=lambda t: t[0], reverse=True)
            return [p for _, p in results[:top_k]]

    async def get_stats(self) -> dict[str, Any]:
        async with self._lock:
            active = sum(1 for d in self._sidecar if d is not None)
        return {
            "count": active,
            "index_path": str(self._index_path),
            "backend": "faiss",
        }

    async def update_pattern(self, pattern_id: str, decay_weight: float) -> None:
        async with self._lock:
            if pattern_id not in self._id_to_row:
                raise KeyError(f"Pattern {pattern_id!r} not found in FAISS backend")
            row = self._id_to_row[pattern_id]
            if self._sidecar[row] is not None:
                self._sidecar[row]["decay_weight"] = decay_weight  # type: ignore[index]
            self._save()

    async def delete_pattern(self, pattern_id: str) -> None:
        async with self._lock:
            if pattern_id not in self._id_to_row:
                raise KeyError(f"Pattern {pattern_id!r} not found in FAISS backend")
            row = self._id_to_row.pop(pattern_id)
            self._sidecar[row] = None
            # Compact immediately on delete to keep the index honest
            self._compact()
            self._save()
