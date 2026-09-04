"""
patternmem.types
~~~~~~~~~~~~~~~~
Core data contracts for PatternMem RAG middleware.

This module is the **only** place where ``FailureType``, ``FailureSignal``,
and ``FailurePattern`` are defined.  Everything else in the codebase imports
from here — never redefines these types locally.

No logic lives here: only ``@dataclass`` / ``Enum`` definitions and docstrings.
The module must remain importable with zero external dependencies.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto



# ---------------------------------------------------------------------------
# FailureType
# ---------------------------------------------------------------------------


class FailureType(Enum):
    """Taxonomy of RAG failure modes detected by ``EvalRouter``.

    Members
    -------
    RETRIEVAL_MISS
        The retriever returned no documents relevant to the query.
    RETRIEVAL_RANK
        Relevant documents were retrieved but ranked too low to influence the
        generated answer.
    RETRIEVAL_STALE
        Retrieved documents are outdated relative to the query's time context.
    RETRIEVAL_ENTITY_DROP
        A key named entity in the query was dropped from the retrieved context.
    HALLUCINATION
        The generated answer contains facts not supported by retrieved context.
    FAITHFULNESS_DRIFT
        The generated answer is partially faithful but drifts from context in
        meaningful ways (softer than outright hallucination).
    REFUSAL
        The generator refused to answer (e.g. safety filter, out-of-scope).
    INCOMPLETE
        The generator produced a truncated or partial answer.
    UNKNOWN
        Failure mode cannot be determined from available signals.  This is also
        the sentinel value used by ``eval="none"`` mode.
    """

    RETRIEVAL_MISS = auto()
    RETRIEVAL_RANK = auto()
    RETRIEVAL_STALE = auto()
    RETRIEVAL_ENTITY_DROP = auto()
    HALLUCINATION = auto()
    FAITHFULNESS_DRIFT = auto()
    REFUSAL = auto()
    INCOMPLETE = auto()
    UNKNOWN = auto()


# ---------------------------------------------------------------------------
# Injection-lane classification helpers
# ---------------------------------------------------------------------------

#: Failure types that should route to the *retrieval* injection lane.
RETRIEVAL_FAILURE_TYPES: frozenset[FailureType] = frozenset(
    {
        FailureType.RETRIEVAL_MISS,
        FailureType.RETRIEVAL_RANK,
        FailureType.RETRIEVAL_STALE,
        FailureType.RETRIEVAL_ENTITY_DROP,
    }
)

#: Failure types that should route to the *generation* injection lane.
GENERATION_FAILURE_TYPES: frozenset[FailureType] = frozenset(
    {
        FailureType.HALLUCINATION,
        FailureType.FAITHFULNESS_DRIFT,
        FailureType.REFUSAL,
        FailureType.INCOMPLETE,
    }
)


def is_retrieval_failure(ft: FailureType) -> bool:
    """Return ``True`` if *ft* should route to ``augmented_input["retrieval_hint"]``."""
    return ft in RETRIEVAL_FAILURE_TYPES


def is_generation_failure(ft: FailureType) -> bool:
    """Return ``True`` if *ft* should route to ``augmented_input["generation_constraint"]``."""
    return ft in GENERATION_FAILURE_TYPES


# ---------------------------------------------------------------------------
# LLMResolverError
# ---------------------------------------------------------------------------


class LLMResolverError(RuntimeError):
    """Raised at ``PatternMemMiddleware.__init__`` time when no LLM can be resolved.

    This is an *init-time* error, never a runtime error (see Invariant 1).
    The message always includes which probe paths were attempted so the user
    can quickly diagnose the issue.
    """


# ---------------------------------------------------------------------------
# FailureSignal
# ---------------------------------------------------------------------------


@dataclass
class FailureSignal:
    """Normalised failure signal produced exclusively by ``EvalRouter``.

    ``EvalRouter`` is the *only* permitted source of ``FailureSignal`` objects.
    No other code in the codebase constructs them directly except in tests.

    Attributes
    ----------
    score : float
        Evaluation score in [0.0, 1.0].  Lower scores indicate worse quality.
        Set to ``0.0`` when ``eval="none"`` mode is active.
    failure_type : FailureType
        Classified failure mode.  ``FailureType.UNKNOWN`` when the mode cannot
        be determined.
    root_cause : str
        Human-readable description of *why* the failure occurred, extracted by
        the ``BackgroundReflector`` using the resolved LLM.
    hint : str
        Actionable hint text to be injected in the next Phase 1 pass via the
        appropriate augmentation lane.
    trace_id : str
        UUID4 string correlating this signal across logs, Langfuse, and OTel.
    """

    score: float
    failure_type: FailureType
    root_cause: str
    hint: str
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"FailureSignal.score must be in [0.0, 1.0], got {self.score!r}"
            )


# ---------------------------------------------------------------------------
# FailurePattern
# ---------------------------------------------------------------------------


@dataclass
class FailurePattern:
    """Persistent failure pattern stored in a ``MemoryBackend``.

    ``FailurePattern`` is the persisted form of a ``FailureSignal`` enriched
    with an embedding and decay bookkeeping fields.  It represents a *recurring*
    failure that the middleware has learned to anticipate and pre-empt.

    Attributes
    ----------
    id : str
        UUID4 string, unique per pattern.
    query_embedding : List[float]
        Dense embedding of the query that triggered this failure, produced by
        the MiniLM encoder.  Used for cosine-similarity lookup in Phase 1.
    failure_type : FailureType
        Classified failure mode (mirrors ``FailureSignal.failure_type``).
    root_cause : str
        Human-readable root-cause explanation.
    hint_text : str
        Actionable hint to inject during Phase 1 augmentation.
    score : float
        Evaluation score at the time this pattern was last written/updated.
    created_at : datetime
        UTC timestamp of pattern creation.
    decay_weight : float
        Starts at ``1.0`` on creation.  Increases on reinforcement (+0.1),
        decreases on non-improvement (−0.2).  Patterns below ``0.1`` are evicted.
        See ``patternmem.decay`` for the exact update logic.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_embedding: list[float] = field(default_factory=list)
    failure_type: FailureType = FailureType.UNKNOWN
    root_cause: str = ""
    hint_text: str = ""
    score: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    decay_weight: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"FailurePattern.score must be in [0.0, 1.0], got {self.score!r}"
            )
        if self.decay_weight < 0.0:
            raise ValueError(
                f"FailurePattern.decay_weight must be >= 0.0, got {self.decay_weight!r}"
            )
