"""
patternmem.backend
~~~~~~~~~~~~~~~~~~
Abstract base class for all PatternMem memory backends.

Every storage backend (JSON, SQLite, NetworkX, Neo4j) must subclass
``MemoryBackend`` and implement every abstract method.  The contract test suite
in ``tests/contract/test_backend_contract.py`` asserts identical behaviour
across all implementations, making them provably interchangeable.

Design rules
------------
- All methods are ``async`` — implementations may use any I/O strategy
  (asyncio-native, ``asyncio.to_thread``, etc.) as long as they do not block
  the event loop for more than ~10 ms.
- The ABC deliberately does *not* import any storage driver — backends stay
  in ``patternmem/backends/`` and import drivers themselves.
- ``update_pattern`` and ``delete_pattern`` are required by the decay loop
  (Invariant 6) but are *not* exported from ``patternmem.__init__``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from patternmem.types import FailurePattern


class MemoryBackend(ABC):
    """Abstract base class that every storage backend must implement.

    The public surface (``write_pattern``, ``lookup_patterns``, ``get_stats``)
    is exported from ``patternmem.__init__``.  The lifecycle methods
    (``update_pattern``, ``delete_pattern``) are required internally for the
    decay/eviction loop but are not part of the user-facing API.
    """

    # ------------------------------------------------------------------
    # Public API — exported via patternmem.__init__
    # ------------------------------------------------------------------

    @abstractmethod
    async def write_pattern(self, pattern: FailurePattern) -> None:
        """Persist a new ``FailurePattern`` to the backend.

        Parameters
        ----------
        pattern:
            The failure pattern to store.  The ``id`` field is already set by
            ``FailurePattern``'s ``__post_init__``.

        Notes
        -----
        Implementations must be idempotent on ``pattern.id``: writing the same
        ``id`` twice should update the existing record, not create a duplicate.
        """

    @abstractmethod
    async def lookup_patterns(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[FailurePattern]:
        """Return the top-*k* patterns whose embeddings are similar to *query_embedding*.

        Parameters
        ----------
        query_embedding:
            Dense query vector produced by the MiniLM encoder.
        top_k:
            Maximum number of patterns to return.  Implementations must apply
            the similarity threshold (0.82 by default, configurable via the
            middleware's ``similarity_threshold`` parameter) before ranking.

        Returns
        -------
        list[FailurePattern]
            Ordered by descending similarity score.  Empty list if no patterns
            exceed the similarity threshold.
        """

    @abstractmethod
    async def get_stats(self) -> dict[str, Any]:
        """Return backend health and bookkeeping statistics.

        The returned dict must contain at minimum:

        .. code-block:: python

            {"count": int}          # total patterns stored

        Implementations may include additional keys (e.g. ``"path"``,
        ``"uri"``, ``"evictions"``).
        """

    # ------------------------------------------------------------------
    # Lifecycle API — required by decay/eviction loop (Invariant 6)
    # ------------------------------------------------------------------

    @abstractmethod
    async def update_pattern(self, pattern_id: str, decay_weight: float) -> None:
        """Update the ``decay_weight`` of an existing pattern.

        Parameters
        ----------
        pattern_id:
            UUID string of the pattern to update.
        decay_weight:
            New decay weight value.  Must be stored verbatim; no clamping is
            performed here — callers (``BackgroundReflector``) own that logic.

        Raises
        ------
        KeyError
            If *pattern_id* does not exist in the backend.
        """

    @abstractmethod
    async def delete_pattern(self, pattern_id: str) -> None:
        """Permanently evict a pattern from the backend.

        Parameters
        ----------
        pattern_id:
            UUID string of the pattern to delete.

        Raises
        ------
        KeyError
            If *pattern_id* does not exist in the backend.

        Notes
        -----
        This is called by the decay loop when ``decay_weight < 0.1``.
        Implementations should ensure the deletion is durable (e.g. flushed
        to disk for file-based backends) before returning.
        """
