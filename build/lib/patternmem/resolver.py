"""
patternmem.resolver
~~~~~~~~~~~~~~~~~~~~
LLMResolver — discovers a usable LLM from the wrapped pipeline at init time.

Design rules
------------
- Duck-typing only.  No ``isinstance`` against any framework type.
- No framework imports (langchain, llama_index, langgraph, etc.).
- All attribute probes are wrapped in ``try/except AttributeError`` — probing
  must never raise an unexpected exception.
- If all probe paths fail and no ``llm=`` kwarg was supplied, raises
  ``LLMResolverError`` immediately (Invariant 1: fail at init, not at runtime).

Probe priority order (do not reorder)
--------------------------------------
1. ``pipeline.llm``
2. ``pipeline.combine_docs_chain.llm_chain.llm``
3. ``pipeline.nodes["generate"].llm``
4. ``pipeline._llm``
5. ``pipeline.as_query_engine()._llm``
6. Explicit ``llm=`` parameter
"""

from __future__ import annotations

from typing import Any

from patternmem.types import LLMResolverError

# Human-readable descriptions of each probe path (used in error messages)
_PROBE_DESCRIPTIONS = [
    "pipeline.llm",
    "pipeline.combine_docs_chain.llm_chain.llm",
    'pipeline.nodes["generate"].llm',
    "pipeline._llm",
    "pipeline.as_query_engine()._llm",
]


def _probe(pipeline: Any) -> Any | None:
    """Attempt all probe paths and return the first non-None result, or None."""

    # Path 1: pipeline.llm (LangChain LLMChain, simple case)
    try:
        llm = pipeline.llm
        if llm is not None:
            return llm
    except AttributeError:
        pass

    # Path 2: pipeline.combine_docs_chain.llm_chain.llm (LangChain RetrievalQA)
    try:
        llm = pipeline.combine_docs_chain.llm_chain.llm
        if llm is not None:
            return llm
    except AttributeError:
        pass

    # Path 3: pipeline.nodes["generate"].llm (LangGraph compiled graph)
    try:
        nodes = pipeline.nodes
        if isinstance(nodes, dict) and "generate" in nodes:
            llm = nodes["generate"].llm
            if llm is not None:
                return llm
    except AttributeError:
        pass

    # Path 4: pipeline._llm (LlamaIndex query engine)
    try:
        llm = pipeline._llm  # noqa: SLF001
        if llm is not None:
            return llm
    except AttributeError:
        pass

    # Path 5: pipeline.as_query_engine()._llm (LlamaIndex index)
    try:
        engine = pipeline.as_query_engine()
        llm = engine._llm  # noqa: SLF001
        if llm is not None:
            return llm
    except (AttributeError, Exception):  # as_query_engine may raise
        pass

    return None


class LLMResolver:
    """Resolves and holds a reference to the LLM used by the wrapped pipeline.

    Parameters
    ----------
    pipeline:
        The user's pipeline object.  May be any callable or object — resolver
        uses duck-typing only.
    llm:
        Explicit LLM override.  If provided, all probe paths are skipped.

    Raises
    ------
    LLMResolverError
        If no LLM can be resolved from the pipeline and ``llm`` is ``None``.
        This is raised at construction time, never at call time.
    """

    def __init__(self, pipeline: Any, llm: Any | None = None) -> None:
        if llm is not None:
            self._llm = llm
            self._source = "explicit llm= parameter"
            return

        resolved = _probe(pipeline)
        if resolved is None:
            raise LLMResolverError(
                "PatternMemMiddleware could not resolve an LLM from the pipeline. "
                "Tried probe paths:\n"
                + "\n".join(f"  {i + 1}. {desc}" for i, desc in enumerate(_PROBE_DESCRIPTIONS))
                + "\n\nFix: pass llm= explicitly:\n"
                "  PatternMemMiddleware(pipeline=my_pipeline, llm=my_llm)"
            )

        self._llm = resolved
        self._source = "auto-resolved from pipeline"

    @property
    def llm(self) -> Any:
        """The resolved LLM object."""
        return self._llm

    @property
    def source(self) -> str:
        """Human-readable description of how the LLM was resolved."""
        return self._source

    def __repr__(self) -> str:
        return f"LLMResolver(llm={self._llm!r}, source={self._source!r})"
