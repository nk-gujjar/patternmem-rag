"""
tests/test_types.py
~~~~~~~~~~~~~~~~~~~
Unit tests for patternmem.types — FailureType, FailureSignal, FailurePattern,
LLMResolverError, and lane-routing helpers.

These tests have zero external dependencies.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from patternmem.types import (
    GENERATION_FAILURE_TYPES,
    RETRIEVAL_FAILURE_TYPES,
    FailurePattern,
    FailureSignal,
    FailureType,
    LLMResolverError,
    is_generation_failure,
    is_retrieval_failure,
)


# ---------------------------------------------------------------------------
# FailureType
# ---------------------------------------------------------------------------


class TestFailureType:
    def test_all_nine_members_exist(self) -> None:
        expected = {
            "RETRIEVAL_MISS",
            "RETRIEVAL_RANK",
            "RETRIEVAL_STALE",
            "RETRIEVAL_ENTITY_DROP",
            "HALLUCINATION",
            "FAITHFULNESS_DRIFT",
            "REFUSAL",
            "INCOMPLETE",
            "UNKNOWN",
        }
        assert {m.name for m in FailureType} == expected

    def test_members_are_unique(self) -> None:
        values = [m.value for m in FailureType]
        assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# Lane-routing helpers
# ---------------------------------------------------------------------------


class TestLaneRouting:
    def test_retrieval_lane_members(self) -> None:
        expected = {
            FailureType.RETRIEVAL_MISS,
            FailureType.RETRIEVAL_RANK,
            FailureType.RETRIEVAL_STALE,
            FailureType.RETRIEVAL_ENTITY_DROP,
        }
        assert RETRIEVAL_FAILURE_TYPES == expected

    def test_generation_lane_members(self) -> None:
        expected = {
            FailureType.HALLUCINATION,
            FailureType.FAITHFULNESS_DRIFT,
            FailureType.REFUSAL,
            FailureType.INCOMPLETE,
        }
        assert GENERATION_FAILURE_TYPES == expected

    def test_lanes_are_disjoint(self) -> None:
        assert RETRIEVAL_FAILURE_TYPES.isdisjoint(GENERATION_FAILURE_TYPES)

    @pytest.mark.parametrize("ft", list(RETRIEVAL_FAILURE_TYPES))
    def test_is_retrieval_failure_true(self, ft: FailureType) -> None:
        assert is_retrieval_failure(ft) is True
        assert is_generation_failure(ft) is False

    @pytest.mark.parametrize("ft", list(GENERATION_FAILURE_TYPES))
    def test_is_generation_failure_true(self, ft: FailureType) -> None:
        assert is_generation_failure(ft) is True
        assert is_retrieval_failure(ft) is False

    def test_unknown_is_neither_lane(self) -> None:
        assert not is_retrieval_failure(FailureType.UNKNOWN)
        assert not is_generation_failure(FailureType.UNKNOWN)


# ---------------------------------------------------------------------------
# FailureSignal
# ---------------------------------------------------------------------------


class TestFailureSignal:
    def test_valid_construction(self) -> None:
        sig = FailureSignal(
            score=0.3,
            failure_type=FailureType.HALLUCINATION,
            root_cause="model hallucinated a date",
            hint="add temporal grounding to retrieval query",
        )
        assert sig.score == 0.3
        assert sig.failure_type == FailureType.HALLUCINATION
        assert isinstance(sig.trace_id, str)
        # trace_id should be a valid UUID4
        uuid.UUID(sig.trace_id, version=4)

    def test_trace_id_is_unique_per_instance(self) -> None:
        a = FailureSignal(
            score=0.5, failure_type=FailureType.UNKNOWN, root_cause="", hint=""
        )
        b = FailureSignal(
            score=0.5, failure_type=FailureType.UNKNOWN, root_cause="", hint=""
        )
        assert a.trace_id != b.trace_id

    def test_explicit_trace_id_preserved(self) -> None:
        tid = "00000000-0000-4000-a000-000000000001"
        sig = FailureSignal(
            score=0.0,
            failure_type=FailureType.UNKNOWN,
            root_cause="",
            hint="",
            trace_id=tid,
        )
        assert sig.trace_id == tid

    @pytest.mark.parametrize("bad_score", [-0.01, 1.01, -100.0, 2.0])
    def test_invalid_score_raises(self, bad_score: float) -> None:
        with pytest.raises(ValueError, match="score"):
            FailureSignal(
                score=bad_score,
                failure_type=FailureType.UNKNOWN,
                root_cause="",
                hint="",
            )

    @pytest.mark.parametrize("ok_score", [0.0, 0.5, 1.0])
    def test_boundary_scores_accepted(self, ok_score: float) -> None:
        sig = FailureSignal(
            score=ok_score, failure_type=FailureType.UNKNOWN, root_cause="", hint=""
        )
        assert sig.score == ok_score


# ---------------------------------------------------------------------------
# FailurePattern
# ---------------------------------------------------------------------------


class TestFailurePattern:
    def test_default_construction(self) -> None:
        p = FailurePattern()
        assert p.decay_weight == 1.0
        assert p.failure_type == FailureType.UNKNOWN
        assert p.score == 0.0
        assert isinstance(p.created_at, datetime)
        assert p.created_at.tzinfo == timezone.utc
        uuid.UUID(p.id, version=4)

    def test_id_unique_per_instance(self) -> None:
        a = FailurePattern()
        b = FailurePattern()
        assert a.id != b.id

    def test_full_construction(self) -> None:
        emb = [0.1, 0.2, 0.3]
        p = FailurePattern(
            query_embedding=emb,
            failure_type=FailureType.RETRIEVAL_MISS,
            root_cause="retriever returned 0 docs",
            hint_text="broaden query keywords",
            score=0.25,
            decay_weight=0.8,
        )
        assert p.query_embedding == emb
        assert p.failure_type == FailureType.RETRIEVAL_MISS
        assert p.decay_weight == 0.8

    @pytest.mark.parametrize("bad_score", [-0.01, 1.01])
    def test_invalid_score_raises(self, bad_score: float) -> None:
        with pytest.raises(ValueError, match="score"):
            FailurePattern(score=bad_score)

    def test_negative_decay_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="decay_weight"):
            FailurePattern(decay_weight=-0.01)

    def test_zero_decay_weight_accepted(self) -> None:
        # Zero is valid; eviction is triggered by < 0.1 not == 0.0
        p = FailurePattern(decay_weight=0.0)
        assert p.decay_weight == 0.0


# ---------------------------------------------------------------------------
# LLMResolverError
# ---------------------------------------------------------------------------


class TestLLMResolverError:
    def test_is_runtime_error(self) -> None:
        err = LLMResolverError("no LLM found")
        assert isinstance(err, RuntimeError)

    def test_message_preserved(self) -> None:
        err = LLMResolverError("probe paths: a, b, c")
        assert "probe paths" in str(err)
