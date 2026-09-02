"""
patternmem.augmenter
~~~~~~~~~~~~~~~~~~~~~
Augmenter — builds the ``augmented_input`` dict from retrieved ``FailurePattern``
objects and injects it into the pipeline call for Phase 1 of the 3-phase loop.

Invariants enforced here
------------------------
- **Invariant 2**: Never mutates the user's prompt template.  All influence
  flows through ``augmented_input``.
- **Invariant 4**: Retrieval-type hints → ``augmented_input["retrieval_hint"]``
  only.  Generation-type constraints → ``augmented_input["generation_constraint"]``
  only.  No hint ever appears in the wrong key.

Framework detection
-------------------
Duck-typing only — no ``isinstance`` against framework types, no framework imports.
"""

from __future__ import annotations

from typing import Any

from patternmem.types import (
    FailurePattern,
    is_generation_failure,
    is_retrieval_failure,
)


class Augmenter:
    """Builds ``augmented_input`` dicts from failure patterns.

    Parameters
    ----------
    pipeline:
        The wrapped pipeline object.  Used for framework-specific augmentation
        (duck-typing only).
    rewrite_feedback:
        If ``True``, add a ``rewrite_constraint`` key to ``augmented_input``
        when retrieval hints are present.  Never touches the prompt template.
    allow_param_override:
        If ``True``, add ``llm_call_kwargs`` (e.g. ``temperature=0.1``) to
        ``augmented_input`` for grounding-sensitive failure types.
    """

    def __init__(
        self,
        pipeline: Any,
        rewrite_feedback: bool = False,
        allow_param_override: bool = False,
    ) -> None:
        self._pipeline = pipeline
        self._rewrite_feedback = rewrite_feedback
        self._allow_param_override = allow_param_override

    # ------------------------------------------------------------------
    # Framework detection helpers (duck-typing, no imports)
    # ------------------------------------------------------------------

    def _is_langchain(self) -> bool:
        """Heuristic: pipeline has a ``retriever`` attribute with ``search_kwargs``."""
        try:
            return hasattr(self._pipeline, "retriever") and hasattr(
                self._pipeline.retriever, "search_kwargs"
            )
        except Exception:  # noqa: BLE001
            return False

    def _is_llamaindex(self) -> bool:
        """Heuristic: pipeline responds to ``as_query_engine()`` call."""
        try:
            return callable(getattr(self._pipeline, "as_query_engine", None))
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Public method
    # ------------------------------------------------------------------

    def build(self, patterns: list[FailurePattern]) -> dict[str, Any]:
        """Build the ``augmented_input`` dict from the given patterns.

        Parameters
        ----------
        patterns:
            Matched patterns from ``backend.lookup_patterns()``.  May contain
            both retrieval-type and generation-type failures (compound case).

        Returns
        -------
        dict[str, Any]
            Keys set only as needed:
            - ``"retrieval_hint"`` — present only if retrieval-type patterns exist
            - ``"generation_constraint"`` — present only if generation-type patterns exist
            - ``"rewrite_constraint"`` — present only if ``rewrite_feedback=True``
              and a retrieval hint was produced
            - ``"llm_call_kwargs"`` — present only if ``allow_param_override=True``
              and a generation-type constraint was produced
            - ``"langchain_retriever_hint"`` — present only for LangChain pipelines
            - ``"llamaindex_query_bundle"`` — present only for LlamaIndex pipelines

        Notes (Invariant 4)
        -------------------
        Retrieval-type ``FailureType``s → ``"retrieval_hint"`` only.
        Generation-type ``FailureType``s → ``"generation_constraint"`` only.
        This is enforced by ``is_retrieval_failure`` / ``is_generation_failure``
        defined in ``patternmem.types``.
        """
        retrieval_hints: list[str] = []
        generation_constraints: list[str] = []

        for p in patterns:
            if is_retrieval_failure(p.failure_type):
                retrieval_hints.append(p.hint_text)
            elif is_generation_failure(p.failure_type):
                generation_constraints.append(p.hint_text)
            # UNKNOWN failure type → no augmentation

        augmented: dict[str, Any] = {}

        # --- Retrieval lane ---
        if retrieval_hints:
            combined_hint = "; ".join(retrieval_hints)
            augmented["retrieval_hint"] = combined_hint

            # Framework-specific injection
            if self._is_langchain():
                try:
                    self._pipeline.retriever.search_kwargs["hint"] = combined_hint
                    augmented["langchain_retriever_hint"] = combined_hint
                except Exception:  # noqa: BLE001
                    pass
            elif self._is_llamaindex():
                augmented["llamaindex_query_bundle"] = {
                    "custom_embedding_strs": [combined_hint]
                }

            # Rewrite feedback lane (only if explicitly enabled)
            if self._rewrite_feedback:
                # Invariant 2: never patch the prompt object — append via context key only
                augmented["rewrite_constraint"] = (
                    f"Constraint (auto): {combined_hint}"
                )

        # --- Generation lane ---
        if generation_constraints:
            combined_constraint = "; ".join(generation_constraints)
            augmented["generation_constraint"] = combined_constraint

            # Optional temperature override for grounding-sensitive queries
            if self._allow_param_override:
                augmented["llm_call_kwargs"] = {"temperature": 0.1}

        return augmented
