"""
patternmem.eval_router
~~~~~~~~~~~~~~~~~~~~~~~
EvalRouter — the *only* permitted translation boundary between external eval
libraries (RAGAS, DeepEval) and the rest of the PatternMem system.

Invariant 7: ``EvalRouter`` output is **always** a ``list[FailureSignal]``.
No code outside this module ever touches a raw RAGAS or DeepEval object.

Behaviour summary
-----------------
1. **Short-circuit**: if the pipeline output already contains a ``"reflect"``
   key or a ``FaithfulnessEvaluator`` result dict, that score is used directly
   and neither RAGAS nor DeepEval is invoked.
2. **``eval="auto"``**: tries RAGAS, falls back to DeepEval, falls back to
   ``"none"`` mode if neither is installed.
3. **``eval="none"``**: returns a single ``FailureSignal`` with
   ``FailureType.UNKNOWN`` and ``score=0.0`` — no external calls.
4. **Compound failure**: when both ``faithfulness < LOW_THRESHOLD`` and
   ``context_precision < LOW_THRESHOLD``, returns **two** ``FailureSignal``s
   (one ``HALLUCINATION``, one ``RETRIEVAL_MISS``).
5. Return type is always ``list[FailureSignal]`` — length 1 for single
   failures, 2 for compound.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from patternmem.types import FailureSignal, FailureType

# ---------------------------------------------------------------------------
# Numeric thresholds (module-level constants, not magic numbers)
# ---------------------------------------------------------------------------

LOW_THRESHOLD: float = 0.4   # below this → metric is "low"
HIGH_THRESHOLD: float = 0.7  # above this → metric is "high"

# ---------------------------------------------------------------------------
# Internal helpers — RAGAS adapter
# ---------------------------------------------------------------------------


def _try_ragas(
    query: str,
    chunks: list[str],
    answer: str,
) -> tuple[float, float] | None:
    """Return ``(faithfulness, context_precision)`` via RAGAS, or ``None`` if unavailable."""
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, context_precision
        from datasets import Dataset

        ds = Dataset.from_dict(
            {
                "question": [query],
                "answer": [answer],
                "contexts": [chunks],
            }
        )
        result = ragas_evaluate(ds, metrics=[faithfulness, context_precision])
        faith = float(result["faithfulness"])
        cp = float(result["context_precision"])
        return faith, cp
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Internal helpers — DeepEval adapter
# ---------------------------------------------------------------------------


def _try_deepeval(
    query: str,
    chunks: list[str],
    answer: str,
) -> tuple[float, float] | None:
    """Return ``(faithfulness, context_precision)`` via DeepEval, or ``None``."""
    try:
        from deepeval.metrics import (
            FaithfulnessMetric,
            ContextualPrecisionMetric,
        )
        from deepeval.test_case import LLMTestCase

        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=chunks,
        )
        fm = FaithfulnessMetric(threshold=0.5)
        cm = ContextualPrecisionMetric(threshold=0.5)
        fm.measure(test_case)
        cm.measure(test_case)
        return float(fm.score), float(cm.score)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Short-circuit helper
# ---------------------------------------------------------------------------


def _extract_existing_reflection(
    pipeline_output: Any,
) -> tuple[float, float] | None:
    """Return ``(faithfulness, context_precision)`` from an existing reflection key.

    Checks for:
    - ``pipeline_output["reflect"]`` — a float score assumed to be faithfulness
    - ``pipeline_output["faithfulness_eval"]`` — LlamaIndex FaithfulnessEvaluator result

    If found, sets ``context_precision = 1.0`` (no retrieval signal available from
    reflection node) so classification routes to a generation-type failure only.
    """
    if not isinstance(pipeline_output, dict):
        return None

    # Key "reflect" — a raw float (0–1) indicating faithfulness
    if "reflect" in pipeline_output:
        score = pipeline_output["reflect"]
        try:
            return float(score), 1.0
        except (TypeError, ValueError):
            pass

    # LlamaIndex FaithfulnessEvaluator result — dict with "score" key
    if "faithfulness_eval" in pipeline_output:
        fe = pipeline_output["faithfulness_eval"]
        if isinstance(fe, dict) and "score" in fe:
            try:
                return float(fe["score"]), 1.0
            except (TypeError, ValueError):
                pass

    return None


# ---------------------------------------------------------------------------
# Classification matrix
# ---------------------------------------------------------------------------


def _classify(
    faithfulness: float,
    context_precision: float,
    query: str,
    hint_override: str = "",
) -> list[FailureSignal]:
    """Map (faithfulness, context_precision) → list[FailureSignal] per the spec matrix.

    Compound case: both low → two independent signals.
    """
    faith_low = faithfulness < LOW_THRESHOLD
    cp_low = context_precision < LOW_THRESHOLD
    faith_high = faithfulness >= HIGH_THRESHOLD
    cp_high = context_precision >= HIGH_THRESHOLD

    signals: list[FailureSignal] = []
    trace_id = str(uuid.uuid4())

    if faith_low and cp_low:
        # Compound case — two independent signals
        signals.append(
            FailureSignal(
                score=faithfulness,
                failure_type=FailureType.HALLUCINATION,
                root_cause="Both faithfulness and context precision are low; "
                           "generation is not grounded in retrieved context.",
                hint="Add explicit grounding instructions to the generator; "
                     "check that context is attached to the prompt.",
                trace_id=trace_id,
            )
        )
        signals.append(
            FailureSignal(
                score=context_precision,
                failure_type=FailureType.RETRIEVAL_MISS,
                root_cause="Context precision is critically low; retrieved chunks "
                           "are not relevant to the query.",
                hint="Broaden retrieval query; increase top-k; check index freshness.",
                trace_id=str(uuid.uuid4()),
            )
        )
    elif faith_low and not cp_low:
        # Retrieval found relevant docs but generator drifted
        ft = (
            FailureType.FAITHFULNESS_DRIFT
            if faithfulness >= 0.2  # partially faithful
            else FailureType.HALLUCINATION
        )
        signals.append(
            FailureSignal(
                score=faithfulness,
                failure_type=ft,
                root_cause="Generator answer diverges from retrieved context despite "
                           "relevant retrieval.",
                hint="Add citation constraints; reduce temperature; force CoT.",
                trace_id=trace_id,
            )
        )
    elif not faith_low and cp_low:
        # Generator is faithful to what it has, but retrieval quality is poor
        ft = FailureType.RETRIEVAL_MISS if context_precision < 0.2 else FailureType.RETRIEVAL_RANK
        signals.append(
            FailureSignal(
                score=context_precision,
                failure_type=ft,
                root_cause="Retrieved context has low precision relative to the query.",
                hint="Refine retrieval query; add MMR re-ranking; expand synonyms.",
                trace_id=trace_id,
            )
        )
    else:
        # Both high → no actionable failure
        signals.append(
            FailureSignal(
                score=min(faithfulness, context_precision),
                failure_type=FailureType.UNKNOWN,
                root_cause="No clear failure detected.",
                hint="",
                trace_id=trace_id,
            )
        )

    return signals


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class EvalRouter:
    """Normalises RAGAS and DeepEval outputs into ``FailureSignal`` lists.

    Parameters
    ----------
    mode:
        One of ``"ragas"``, ``"deepeval"``, ``"auto"``, ``"none"``.
        ``"auto"`` tries RAGAS → DeepEval → none in order.
    """

    def __init__(self, mode: Literal["ragas", "deepeval", "auto", "none"]) -> None:
        self._mode = mode

    def evaluate(
        self,
        query: str,
        chunks: list[str],
        answer: str,
        pipeline_output: Any = None,
    ) -> list[FailureSignal]:
        """Evaluate a single query/answer/context triple.

        Parameters
        ----------
        query:
            The original user query.
        chunks:
            Retrieved context chunks passed to the generator.
        answer:
            The generated answer.
        pipeline_output:
            Raw return value from the pipeline call.  Inspected for existing
            reflection keys before invoking any eval library.

        Returns
        -------
        list[FailureSignal]
            Length 1 for single failures; length 2 for compound (both faithfulness
            and context_precision low).  Always returns at least one signal.
        """
        # --- Short-circuit: existing reflection in pipeline output ---
        existing = _extract_existing_reflection(pipeline_output)
        if existing is not None:
            faith, cp = existing
            return _classify(faith, cp, query)

        # --- eval="none" ---
        if self._mode == "none":
            return [
                FailureSignal(
                    score=0.0,
                    failure_type=FailureType.UNKNOWN,
                    root_cause="Evaluation disabled (eval='none').",
                    hint="",
                )
            ]

        # --- Try the configured adapter(s) ---
        scores: tuple[float, float] | None = None

        if self._mode in ("ragas", "auto"):
            scores = _try_ragas(query, chunks, answer)

        if scores is None and self._mode in ("deepeval", "auto"):
            scores = _try_deepeval(query, chunks, answer)

        if scores is None:
            # Neither library available — fall back to none behaviour
            return [
                FailureSignal(
                    score=0.0,
                    failure_type=FailureType.UNKNOWN,
                    root_cause="No evaluation library available (RAGAS and DeepEval not installed).",
                    hint="",
                )
            ]

        faith, cp = scores
        return _classify(faith, cp, query)
