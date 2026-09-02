"""
tests/conftest.py
~~~~~~~~~~~~~~~~~
Shared pytest fixtures for the PatternMem test suite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio

from patternmem.backends.json_backend import JSONBackend
from patternmem.types import FailurePattern, FailureType

_DIM = 384


def unit_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


@pytest_asyncio.fixture
async def json_backend_tmp(tmp_path: Path) -> JSONBackend:
    return JSONBackend(path=tmp_path / "test_patterns.json", similarity_threshold=0.82)


@pytest.fixture
def sample_pattern() -> FailurePattern:
    return FailurePattern(
        query_embedding=unit_vec(1),
        failure_type=FailureType.RETRIEVAL_MISS,
        root_cause="retriever returned 0 documents",
        hint_text="broaden query",
        score=0.2,
        decay_weight=1.0,
    )
