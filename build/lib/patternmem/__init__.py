"""
patternmem
~~~~~~~~~~
Framework-agnostic RAG middleware that makes any pipeline self-improving
via persistent cross-query failure-pattern memory.

One-line integration::

    answer = await PatternMemMiddleware(pipeline=my_rag_pipeline).ainvoke(query)

Public API
----------
The following names are the stable, versioned public surface of this package.
Anything not listed here is considered internal and may change without notice.
"""

from __future__ import annotations

from patternmem.backend import MemoryBackend
from patternmem.middleware import PatternMemMiddleware
from patternmem.types import (
    FailurePattern,
    FailureSignal,
    FailureType,
    LLMResolverError,
)

__all__ = [
    "PatternMemMiddleware",
    # data contracts
    "FailureSignal",
    "FailureType",
    "FailurePattern",
    # errors
    "LLMResolverError",
    # ABC (for community backend implementors)
    "MemoryBackend",
]

__version__ = "0.1.0"
