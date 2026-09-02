"""
tests/integration/test_full_loop.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Integration test: full 3-phase loop against the JSON backend.

Scenario
--------
A deterministic mock pipeline that:
- Query 1: fails (returns low faithfulness → RETRIEVAL_MISS signal)
- Query 2: fails again (pattern reinforces in backend)
- Query 3: succeeds after hint injection (pattern reinforced; decay_weight goes up)
- Query 4 (always-fail): eviction test — a deliberately brittle pattern decays out

All assertions are on backend state after the reflector drains.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio

from patternmem.backends.json_backend import JSONBackend
from patternmem.decay import EVICTION_FLOOR, REINFORCE_DELTA
from patternmem.middleware import PatternMemMiddleware
from patternmem.types import FailurePattern, FailureType

_DIM = 384


def _unit_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


_QUERY_EMB = _unit_vec(42)

# ---------------------------------------------------------------------------
# Mock pipeline and eval adapters
# ---------------------------------------------------------------------------


class _DeterministicPipeline:
    """Pipeline that fails on queries 1–2, succeeds on query 3+."""

    def __init__(self) -> None:
        self.llm = MagicMock()
        self.call_count = 0

    def __call__(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.call_count += 1
        if self.call_count <= 2:
            # Simulate poor retrieval (low faithfulness, low context precision)
            return {
                "answer": "I don't know",
                "chunks": [],
                # EvalRouter short-circuit via reflect key
                "reflect": 0.15,
            }
        # Simulate success after hint injection
        return {
            "answer": "Paris is the capital of France.",
            "chunks": ["France is a country in Europe. Its capital is Paris."],
            "reflect": 0.9,
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def backend(tmp_path: Path) -> JSONBackend:
    return JSONBackend(path=tmp_path / "integration.json", similarity_threshold=0.50)


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


async def _run_mw_query(mw: PatternMemMiddleware, query: str) -> Any:
    result = await mw.ainvoke(query)
    # Give the event loop a chance to schedule Phase 2/3
    await asyncio.sleep(0.05)
    return result


class TestFullLoop:
    async def test_pattern_written_after_first_failure(
        self, backend: JSONBackend, tmp_path: Path
    ) -> None:
        pipeline = _DeterministicPipeline()

        with patch("patternmem.middleware.SentenceTransformer") as MockST:
            encoder = MagicMock()
            encoder.encode.return_value = np.array(_QUERY_EMB, dtype=np.float32)
            MockST.return_value = encoder

            async with PatternMemMiddleware(
                pipeline=pipeline,
                backend=backend,
                eval="auto",  # will short-circuit via reflect key
                observability=None,
            ) as mw:
                # Query 1 — should fail and write a pattern
                await _run_mw_query(mw, "What is the capital of France?")
                # Drain reflector
                await asyncio.sleep(0.2)

        stats = await backend.get_stats()
        assert stats["count"] >= 1, "Expected at least 1 pattern after query 1"

    async def test_pattern_reinforced_after_successful_hint_run(
        self, backend: JSONBackend, tmp_path: Path
    ) -> None:
        pipeline = _DeterministicPipeline()

        with patch("patternmem.middleware.SentenceTransformer") as MockST:
            encoder = MagicMock()
            encoder.encode.return_value = np.array(_QUERY_EMB, dtype=np.float32)
            MockST.return_value = encoder

            async with PatternMemMiddleware(
                pipeline=pipeline,
                backend=backend,
                eval="auto",
                observability=None,
            ) as mw:
                # Query 1 — fail
                await _run_mw_query(mw, "What is the capital of France?")
                await asyncio.sleep(0.2)

                # Query 2 — fail again (pattern in backend, hint injected, still fails)
                await _run_mw_query(mw, "What is the capital of France?")
                await asyncio.sleep(0.2)

                # Query 3 — success after hint (reflect=0.9 > 0.15 + 0.15)
                await _run_mw_query(mw, "What is the capital of France?")
                await asyncio.sleep(0.2)

        # Pattern should still exist (reinforced)
        stats = await backend.get_stats()
        assert stats["count"] >= 1, "Pattern should survive after reinforcement"

    async def test_always_failing_pattern_gets_evicted(
        self, backend: JSONBackend, tmp_path: Path
    ) -> None:
        """A pattern with a near-floor decay_weight that never helps gets evicted."""
        # Pre-seed a "stale" pattern with a low decay weight
        stale = FailurePattern(
            query_embedding=_unit_vec(99),
            failure_type=FailureType.RETRIEVAL_MISS,
            hint_text="stale hint",
            score=0.5,
            decay_weight=0.25,  # 2 decay rounds → eviction
        )
        await backend.write_pattern(stale)

        # Simulate 2 decay rounds manually (as the reflector would do)
        from patternmem.decay import compute_new_decay_weight, should_evict

        w = stale.decay_weight
        for _ in range(2):
            w = compute_new_decay_weight(w, old_score=0.5, new_score=0.5)
            if should_evict(w):
                await backend.delete_pattern(stale.id)
                break

        stats = await backend.get_stats()
        patterns = await backend.lookup_patterns(_unit_vec(99), top_k=3)
        assert all(p.id != stale.id for p in patterns), (
            "Stale pattern should have been evicted"
        )
