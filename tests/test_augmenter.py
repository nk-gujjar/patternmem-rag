"""
tests/test_augmenter.py
~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for Augmenter — lane routing, compound signals, framework detection,
rewrite_feedback, and allow_param_override.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from patternmem.augmenter import Augmenter
from patternmem.types import FailurePattern, FailureType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pattern(ft: FailureType, hint: str = "test hint") -> FailurePattern:
    return FailurePattern(
        failure_type=ft,
        hint_text=hint,
        score=0.3,
    )


def _augmenter(pipeline=None, rewrite_feedback=False, allow_param_override=False):
    return Augmenter(
        pipeline=pipeline or MagicMock(),
        rewrite_feedback=rewrite_feedback,
        allow_param_override=allow_param_override,
    )


# ---------------------------------------------------------------------------
# Lane routing — Invariant 4
# ---------------------------------------------------------------------------


class TestLaneRouting:
    def test_retrieval_miss_goes_to_retrieval_hint_only(self) -> None:
        aug = _augmenter()
        result = aug.build([_pattern(FailureType.RETRIEVAL_MISS)])
        assert "retrieval_hint" in result
        assert "generation_constraint" not in result

    def test_retrieval_rank_goes_to_retrieval_hint_only(self) -> None:
        aug = _augmenter()
        result = aug.build([_pattern(FailureType.RETRIEVAL_RANK)])
        assert "retrieval_hint" in result
        assert "generation_constraint" not in result

    def test_retrieval_stale_goes_to_retrieval_hint_only(self) -> None:
        aug = _augmenter()
        result = aug.build([_pattern(FailureType.RETRIEVAL_STALE)])
        assert "retrieval_hint" in result
        assert "generation_constraint" not in result

    def test_retrieval_entity_drop_goes_to_retrieval_hint_only(self) -> None:
        aug = _augmenter()
        result = aug.build([_pattern(FailureType.RETRIEVAL_ENTITY_DROP)])
        assert "retrieval_hint" in result
        assert "generation_constraint" not in result

    def test_hallucination_goes_to_generation_constraint_only(self) -> None:
        aug = _augmenter()
        result = aug.build([_pattern(FailureType.HALLUCINATION)])
        assert "generation_constraint" in result
        assert "retrieval_hint" not in result

    def test_faithfulness_drift_goes_to_generation_constraint_only(self) -> None:
        aug = _augmenter()
        result = aug.build([_pattern(FailureType.FAITHFULNESS_DRIFT)])
        assert "generation_constraint" in result
        assert "retrieval_hint" not in result

    def test_refusal_goes_to_generation_constraint_only(self) -> None:
        aug = _augmenter()
        result = aug.build([_pattern(FailureType.REFUSAL)])
        assert "generation_constraint" in result
        assert "retrieval_hint" not in result

    def test_incomplete_goes_to_generation_constraint_only(self) -> None:
        aug = _augmenter()
        result = aug.build([_pattern(FailureType.INCOMPLETE)])
        assert "generation_constraint" in result
        assert "retrieval_hint" not in result

    def test_unknown_sets_no_lane_keys(self) -> None:
        aug = _augmenter()
        result = aug.build([_pattern(FailureType.UNKNOWN)])
        assert "retrieval_hint" not in result
        assert "generation_constraint" not in result


# ---------------------------------------------------------------------------
# Compound case — both keys set, no cross-contamination
# ---------------------------------------------------------------------------


class TestCompoundCase:
    def test_compound_signals_both_keys_set(self) -> None:
        aug = _augmenter()
        patterns = [
            _pattern(FailureType.HALLUCINATION, hint="add CoT"),
            _pattern(FailureType.RETRIEVAL_MISS, hint="broaden query"),
        ]
        result = aug.build(patterns)
        assert "retrieval_hint" in result
        assert "generation_constraint" in result

    def test_no_cross_contamination(self) -> None:
        aug = _augmenter()
        patterns = [
            _pattern(FailureType.HALLUCINATION, hint="gen-only hint"),
            _pattern(FailureType.RETRIEVAL_MISS, hint="ret-only hint"),
        ]
        result = aug.build(patterns)
        # Generation hint must NOT appear in retrieval_hint
        assert "gen-only hint" not in result.get("retrieval_hint", "")
        # Retrieval hint must NOT appear in generation_constraint
        assert "ret-only hint" not in result.get("generation_constraint", "")


# ---------------------------------------------------------------------------
# Optional flags
# ---------------------------------------------------------------------------


class TestOptionalFlags:
    def test_rewrite_feedback_adds_rewrite_constraint_when_retrieval_hint_present(self) -> None:
        aug = _augmenter(rewrite_feedback=True)
        result = aug.build([_pattern(FailureType.RETRIEVAL_MISS, hint="expand synonyms")])
        assert "rewrite_constraint" in result
        assert "expand synonyms" in result["rewrite_constraint"]

    def test_rewrite_feedback_not_set_without_flag(self) -> None:
        aug = _augmenter(rewrite_feedback=False)
        result = aug.build([_pattern(FailureType.RETRIEVAL_MISS)])
        assert "rewrite_constraint" not in result

    def test_rewrite_feedback_not_added_for_generation_only(self) -> None:
        """rewrite_constraint should only appear alongside a retrieval hint."""
        aug = _augmenter(rewrite_feedback=True)
        result = aug.build([_pattern(FailureType.HALLUCINATION)])
        assert "rewrite_constraint" not in result

    def test_allow_param_override_adds_llm_call_kwargs(self) -> None:
        aug = _augmenter(allow_param_override=True)
        result = aug.build([_pattern(FailureType.HALLUCINATION)])
        assert "llm_call_kwargs" in result
        assert result["llm_call_kwargs"]["temperature"] == 0.1

    def test_allow_param_override_not_set_without_flag(self) -> None:
        aug = _augmenter(allow_param_override=False)
        result = aug.build([_pattern(FailureType.HALLUCINATION)])
        assert "llm_call_kwargs" not in result

    def test_allow_param_override_not_added_for_retrieval_only(self) -> None:
        """llm_call_kwargs should only appear alongside a generation constraint."""
        aug = _augmenter(allow_param_override=True)
        result = aug.build([_pattern(FailureType.RETRIEVAL_MISS)])
        assert "llm_call_kwargs" not in result


# ---------------------------------------------------------------------------
# Framework-specific injection
# ---------------------------------------------------------------------------


class TestFrameworkInjection:
    def test_langchain_retriever_hint_set(self) -> None:
        pipeline = MagicMock()
        pipeline.retriever = MagicMock()
        pipeline.retriever.search_kwargs = {}
        aug = Augmenter(pipeline=pipeline)
        result = aug.build([_pattern(FailureType.RETRIEVAL_MISS, hint="lc hint")])
        assert pipeline.retriever.search_kwargs.get("hint") == "lc hint"
        assert "langchain_retriever_hint" in result

    def test_llamaindex_query_bundle_set(self) -> None:
        pipeline = MagicMock(spec=[])  # no retriever attr
        pipeline.as_query_engine = MagicMock()
        aug = Augmenter(pipeline=pipeline)
        result = aug.build([_pattern(FailureType.RETRIEVAL_MISS, hint="li hint")])
        # augmented_input["retrieval_hint"] always set
        assert result.get("retrieval_hint") == "li hint"

    def test_empty_patterns_returns_empty_dict(self) -> None:
        aug = _augmenter()
        result = aug.build([])
        assert result == {}
