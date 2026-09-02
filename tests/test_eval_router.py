"""
tests/test_eval_router.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for EvalRouter — covers short-circuit, eval=none, compound failures,
and the guarantee that RAGAS/DeepEval are never called when a reflect key exists.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from patternmem.eval_router import EvalRouter, LOW_THRESHOLD, HIGH_THRESHOLD
from patternmem.types import FailureSignal, FailureType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _router(mode: str = "none") -> EvalRouter:
    return EvalRouter(mode=mode)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Short-circuit tests
# ---------------------------------------------------------------------------


class TestShortCircuit:
    def test_reflect_key_prevents_ragas_call(self) -> None:
        """When pipeline_output has 'reflect', RAGAS must NOT be invoked."""
        router = _router("auto")
        pipeline_output = {"reflect": 0.3, "answer": "some answer"}

        with patch("patternmem.eval_router._try_ragas") as mock_ragas, \
             patch("patternmem.eval_router._try_deepeval") as mock_deepeval:
            signals = router.evaluate(
                query="What is X?",
                chunks=["context"],
                answer="some answer",
                pipeline_output=pipeline_output,
            )
            mock_ragas.assert_not_called()
            mock_deepeval.assert_not_called()

        assert len(signals) >= 1
        assert all(isinstance(s, FailureSignal) for s in signals)

    def test_reflect_key_low_routes_to_generation_failure(self) -> None:
        router = _router("auto")
        # reflect=0.1 → faithfulness LOW, cp set to 1.0 → generation failure
        signals = router.evaluate(
            query="q", chunks=[], answer="a",
            pipeline_output={"reflect": 0.1},
        )
        assert len(signals) == 1
        assert signals[0].failure_type in {
            FailureType.HALLUCINATION, FailureType.FAITHFULNESS_DRIFT
        }

    def test_faithfulness_eval_key_short_circuits(self) -> None:
        router = _router("auto")
        pipeline_output = {"faithfulness_eval": {"score": 0.2}}

        with patch("patternmem.eval_router._try_ragas") as mock_ragas:
            signals = router.evaluate(
                query="q", chunks=[], answer="a",
                pipeline_output=pipeline_output,
            )
            mock_ragas.assert_not_called()

        assert len(signals) >= 1

    def test_no_short_circuit_without_reflect_key(self) -> None:
        """Without a reflect key, eval is called (mocked to return None → none fallback)."""
        router = _router("auto")
        with patch("patternmem.eval_router._try_ragas", return_value=None), \
             patch("patternmem.eval_router._try_deepeval", return_value=None):
            signals = router.evaluate(
                query="q", chunks=["c"], answer="a",
                pipeline_output={"answer": "a"},
            )
        assert signals[0].failure_type == FailureType.UNKNOWN


# ---------------------------------------------------------------------------
# eval="none" mode
# ---------------------------------------------------------------------------


class TestEvalNone:
    def test_returns_single_unknown_signal(self) -> None:
        router = _router("none")
        signals = router.evaluate("q", ["c"], "a")
        assert len(signals) == 1
        assert signals[0].failure_type == FailureType.UNKNOWN
        assert signals[0].score == 0.0

    def test_never_calls_ragas_or_deepeval(self) -> None:
        router = _router("none")
        with patch("patternmem.eval_router._try_ragas") as mock_ragas, \
             patch("patternmem.eval_router._try_deepeval") as mock_deepeval:
            router.evaluate("q", [], "a")
            mock_ragas.assert_not_called()
            mock_deepeval.assert_not_called()


# ---------------------------------------------------------------------------
# Classification matrix
# ---------------------------------------------------------------------------


class TestClassificationMatrix:
    def _signals_for_scores(self, faith: float, cp: float) -> list[FailureSignal]:
        router = _router("auto")
        with patch("patternmem.eval_router._try_ragas", return_value=(faith, cp)):
            return router.evaluate("q", ["c"], "a")

    def test_both_low_returns_two_signals(self) -> None:
        signals = self._signals_for_scores(0.1, 0.1)
        assert len(signals) == 2
        types = {s.failure_type for s in signals}
        assert FailureType.HALLUCINATION in types
        assert FailureType.RETRIEVAL_MISS in types

    def test_both_low_signals_have_different_trace_ids(self) -> None:
        signals = self._signals_for_scores(0.1, 0.1)
        assert signals[0].trace_id != signals[1].trace_id

    def test_faith_low_cp_high_is_generation_type(self) -> None:
        signals = self._signals_for_scores(0.1, 0.9)
        assert len(signals) == 1
        assert signals[0].failure_type in {
            FailureType.HALLUCINATION, FailureType.FAITHFULNESS_DRIFT
        }

    def test_faith_high_cp_low_is_retrieval_type(self) -> None:
        signals = self._signals_for_scores(0.9, 0.1)
        assert len(signals) == 1
        assert signals[0].failure_type in {
            FailureType.RETRIEVAL_MISS, FailureType.RETRIEVAL_RANK
        }

    def test_both_high_returns_unknown(self) -> None:
        signals = self._signals_for_scores(0.9, 0.9)
        assert len(signals) == 1
        assert signals[0].failure_type == FailureType.UNKNOWN


# ---------------------------------------------------------------------------
# Output type guarantee (Invariant 7)
# ---------------------------------------------------------------------------


class TestOutputTypeGuarantee:
    def test_always_returns_list_of_failure_signals(self) -> None:
        """EvalRouter output is ALWAYS list[FailureSignal] — Invariant 7."""
        for mode in ["none", "auto"]:
            router = EvalRouter(mode=mode)  # type: ignore[arg-type]
            with patch("patternmem.eval_router._try_ragas", return_value=None), \
                 patch("patternmem.eval_router._try_deepeval", return_value=None):
                result = router.evaluate("q", [], "a")
            assert isinstance(result, list)
            assert all(isinstance(s, FailureSignal) for s in result)
