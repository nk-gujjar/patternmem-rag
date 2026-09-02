"""
tests/test_resolver.py
~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for LLMResolver — covers all 5 probe paths and error case.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from patternmem.resolver import LLMResolver
from patternmem.types import LLMResolverError


# ---------------------------------------------------------------------------
# Mock pipeline shapes
# ---------------------------------------------------------------------------


class _MockLangChainChain:
    """Simulates a LangChain LLMChain (probe path 1)."""
    def __init__(self, llm: Any) -> None:
        self.llm = llm


class _MockLangChainRetrievalQA:
    """Simulates a LangChain RetrievalQA (probe path 2)."""
    def __init__(self, llm: Any) -> None:
        self.combine_docs_chain = MagicMock()
        self.combine_docs_chain.llm_chain = MagicMock()
        self.combine_docs_chain.llm_chain.llm = llm


class _MockLangGraphGraph:
    """Simulates a LangGraph compiled graph (probe path 3)."""
    def __init__(self, llm: Any) -> None:
        generate_node = MagicMock()
        generate_node.llm = llm
        self.nodes = {"generate": generate_node}


class _MockLlamaIndexEngine:
    """Simulates a LlamaIndex query engine (probe path 4)."""
    def __init__(self, llm: Any) -> None:
        self._llm = llm


class _MockLlamaIndexIndex:
    """Simulates a LlamaIndex index (probe path 5)."""
    def __init__(self, llm: Any) -> None:
        engine = MagicMock()
        engine._llm = llm
        self._engine = engine

    def as_query_engine(self) -> Any:
        return self._engine


class _NoLLMPipeline:
    """A pipeline with no detectable LLM on any probe path."""
    def __call__(self, query: str) -> str:
        return "answer"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLLMResolverProbePaths:
    def test_path1_pipeline_llm(self) -> None:
        fake_llm = MagicMock()
        pipeline = _MockLangChainChain(llm=fake_llm)
        resolver = LLMResolver(pipeline=pipeline)
        assert resolver.llm is fake_llm

    def test_path2_combine_docs_chain(self) -> None:
        fake_llm = MagicMock()
        pipeline = _MockLangChainRetrievalQA(llm=fake_llm)
        resolver = LLMResolver(pipeline=pipeline)
        assert resolver.llm is fake_llm

    def test_path3_langgraph_nodes(self) -> None:
        fake_llm = MagicMock()
        pipeline = _MockLangGraphGraph(llm=fake_llm)
        resolver = LLMResolver(pipeline=pipeline)
        assert resolver.llm is fake_llm

    def test_path4_private_llm(self) -> None:
        fake_llm = MagicMock()
        pipeline = _MockLlamaIndexEngine(llm=fake_llm)
        resolver = LLMResolver(pipeline=pipeline)
        assert resolver.llm is fake_llm

    def test_path5_as_query_engine(self) -> None:
        fake_llm = MagicMock()
        pipeline = _MockLlamaIndexIndex(llm=fake_llm)
        resolver = LLMResolver(pipeline=pipeline)
        assert resolver.llm is fake_llm


class TestLLMResolverExplicitOverride:
    def test_explicit_llm_skips_probing(self) -> None:
        fake_llm = MagicMock()
        # Pipeline has its own LLM but explicit should win
        pipeline = _MockLangChainChain(llm=MagicMock())
        resolver = LLMResolver(pipeline=pipeline, llm=fake_llm)
        assert resolver.llm is fake_llm
        assert resolver.source == "explicit llm= parameter"

    def test_explicit_llm_on_no_llm_pipeline(self) -> None:
        fake_llm = MagicMock()
        resolver = LLMResolver(pipeline=_NoLLMPipeline(), llm=fake_llm)
        assert resolver.llm is fake_llm


class TestLLMResolverError:
    def test_raises_at_init_when_no_llm_found(self) -> None:
        with pytest.raises(LLMResolverError) as exc_info:
            LLMResolver(pipeline=_NoLLMPipeline())
        # Error message must mention all probe paths
        msg = str(exc_info.value)
        assert "pipeline.llm" in msg
        assert "llm=" in msg

    def test_error_is_runtime_error(self) -> None:
        with pytest.raises(RuntimeError):
            LLMResolver(pipeline=_NoLLMPipeline())

    def test_probing_does_not_raise_on_attribute_error(self) -> None:
        """Probing a plain callable must not raise AttributeError."""
        fake_llm = MagicMock()
        # Plain function — no attributes
        resolver = LLMResolver(pipeline=lambda q: q, llm=fake_llm)
        assert resolver.llm is fake_llm
