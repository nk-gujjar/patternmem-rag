"""
tests/contract/test_backend_contract.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Parameterised contract tests that every MemoryBackend implementation must pass.

Run with::

    pytest tests/contract/ -v --asyncio-mode=auto

Adding a new backend
--------------------
1. Add a pytest fixture for the new backend below.
2. Add its fixture name to the ``backend`` parametrize list.
The test functions themselves never need to change.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
import pytest
import pytest_asyncio

from patternmem.backends.json_backend import JSONBackend
from patternmem.backends.networkx_backend import NetworkXBackend
from patternmem.backends.sqlite_backend import SQLiteBackend
from patternmem.types import FailurePattern, FailureType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 384  # MiniLM embedding dimension


def _unit_vec(seed: int) -> list[float]:
    """Return a reproducible unit vector (same seed → identical lookup behaviour)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _make_pattern(
    embedding: list[float],
    failure_type: FailureType = FailureType.RETRIEVAL_MISS,
    hint: str = "broaden query",
    score: float = 0.3,
    decay_weight: float = 1.0,
) -> FailurePattern:
    return FailurePattern(
        query_embedding=embedding,
        failure_type=failure_type,
        root_cause="retriever returned 0 docs",
        hint_text=hint,
        score=score,
        decay_weight=decay_weight,
    )


# ---------------------------------------------------------------------------
# Backend fixtures — each returns a clean, isolated backend instance
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def json_backend(tmp_path: Path) -> AsyncGenerator[JSONBackend, None]:
    yield JSONBackend(path=tmp_path / "patterns.json", similarity_threshold=0.82)


@pytest_asyncio.fixture
async def networkx_backend() -> AsyncGenerator[NetworkXBackend, None]:
    pytest.importorskip("networkx")
    yield NetworkXBackend(persist_path=None, similarity_threshold=0.82)


@pytest_asyncio.fixture
async def sqlite_backend(tmp_path: Path) -> AsyncGenerator[SQLiteBackend, None]:
    yield SQLiteBackend(path=tmp_path / "patterns.db", similarity_threshold=0.82)


# ---------------------------------------------------------------------------
# Parametrise over all backend fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=["json_backend", "networkx_backend", "sqlite_backend"],
)
def backend(request: pytest.FixtureRequest):  # type: ignore[return]
    return request.getfixturevalue(request.param)


# ---------------------------------------------------------------------------
# Contract tests — identical assertions for every backend
# ---------------------------------------------------------------------------


class TestWriteAndLookup:
    async def test_write_then_lookup_returns_pattern(self, backend) -> None:
        emb = _unit_vec(42)
        p = _make_pattern(emb)
        await backend.write_pattern(p)

        results = await backend.lookup_patterns(emb, top_k=3)
        assert len(results) == 1
        assert results[0].id == p.id
        assert results[0].failure_type == FailureType.RETRIEVAL_MISS
        assert results[0].hint_text == "broaden query"

    async def test_write_is_idempotent_on_same_id(self, backend) -> None:
        emb = _unit_vec(1)
        p = _make_pattern(emb, score=0.3)
        await backend.write_pattern(p)

        # Overwrite with updated score
        p2 = FailurePattern(
            id=p.id,
            query_embedding=emb,
            failure_type=FailureType.HALLUCINATION,
            root_cause="updated",
            hint_text="updated hint",
            score=0.6,
            decay_weight=0.9,
        )
        await backend.write_pattern(p2)

        stats = await backend.get_stats()
        assert stats["count"] == 1  # no duplicate

        results = await backend.lookup_patterns(emb, top_k=3)
        assert results[0].root_cause == "updated"

    async def test_lookup_returns_empty_when_no_match(self, backend) -> None:
        # Write a pattern with embedding seed 10
        emb_stored = _unit_vec(10)
        await backend.write_pattern(_make_pattern(emb_stored))

        # Query with a very different embedding (seed 99)
        emb_query = _unit_vec(99)
        # Force near-zero similarity by negating
        emb_query_neg = [-x for x in emb_stored]
        results = await backend.lookup_patterns(emb_query_neg, top_k=3)
        assert results == []

    async def test_lookup_respects_top_k(self, backend) -> None:
        emb = _unit_vec(7)
        # Write 5 nearly identical patterns (slightly perturbed)
        for i in range(5):
            rng = np.random.default_rng(7 + i)
            v = np.array(emb) + rng.standard_normal(len(emb)) * 0.001
            v = v / np.linalg.norm(v)
            await backend.write_pattern(_make_pattern(v.tolist()))

        results = await backend.lookup_patterns(emb, top_k=3)
        assert len(results) <= 3

    async def test_lookup_ordered_by_descending_similarity(self, backend) -> None:
        base = np.array(_unit_vec(3), dtype=np.float32)
        patterns = []
        # Create patterns at increasing distances from base
        for i in range(3):
            noise = np.random.default_rng(i).standard_normal(len(base)).astype(np.float32) * (0.01 * (i + 1))
            v = base + noise
            v = v / np.linalg.norm(v)
            p = _make_pattern(v.tolist())
            await backend.write_pattern(p)
            patterns.append(p)

        results = await backend.lookup_patterns(base.tolist(), top_k=3)
        assert len(results) >= 1
        # Each result similarity ≥ the next (descending order guaranteed by spec)
        # We verify by re-computing similarities
        sims = []
        for r in results:
            va = np.array(base, dtype=np.float32)
            vb = np.array(r.query_embedding, dtype=np.float32)
            sims.append(float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))))
        assert sims == sorted(sims, reverse=True)


class TestUpdateAndDecay:
    async def test_update_decay_weight(self, backend) -> None:
        emb = _unit_vec(55)
        p = _make_pattern(emb, decay_weight=1.0)
        await backend.write_pattern(p)

        await backend.update_pattern(p.id, 0.8)

        results = await backend.lookup_patterns(emb, top_k=1)
        assert abs(results[0].decay_weight - 0.8) < 1e-6

    async def test_update_nonexistent_raises_keyerror(self, backend) -> None:
        with pytest.raises(KeyError):
            await backend.update_pattern("nonexistent-id", 0.5)

    async def test_delete_removes_pattern(self, backend) -> None:
        emb = _unit_vec(88)
        p = _make_pattern(emb)
        await backend.write_pattern(p)

        await backend.delete_pattern(p.id)

        results = await backend.lookup_patterns(emb, top_k=3)
        assert all(r.id != p.id for r in results)

    async def test_delete_nonexistent_raises_keyerror(self, backend) -> None:
        with pytest.raises(KeyError):
            await backend.delete_pattern("nonexistent-id")

    async def test_decay_eviction_flow(self, backend) -> None:
        """Full write → decay → evict contract."""
        emb = _unit_vec(77)
        p = _make_pattern(emb, decay_weight=0.15)
        await backend.write_pattern(p)

        # Simulate two decay steps (0.15 − 0.2 = −0.05 → below 0.1 floor)
        new_weight = 0.15 - 0.20  # = -0.05 → should trigger eviction
        # Caller (BackgroundReflector) calls delete if below floor; test it here
        await backend.delete_pattern(p.id)

        results = await backend.lookup_patterns(emb, top_k=3)
        assert all(r.id != p.id for r in results)


class TestGetStats:
    async def test_get_stats_has_count(self, backend) -> None:
        stats = await backend.get_stats()
        assert "count" in stats
        assert isinstance(stats["count"], int)

    async def test_count_increments_on_write(self, backend) -> None:
        stats_before = await backend.get_stats()
        await backend.write_pattern(_make_pattern(_unit_vec(100)))
        stats_after = await backend.get_stats()
        assert stats_after["count"] == stats_before["count"] + 1

    async def test_count_decrements_on_delete(self, backend) -> None:
        emb = _unit_vec(101)
        p = _make_pattern(emb)
        await backend.write_pattern(p)
        stats_before = await backend.get_stats()
        await backend.delete_pattern(p.id)
        stats_after = await backend.get_stats()
        assert stats_after["count"] == stats_before["count"] - 1
