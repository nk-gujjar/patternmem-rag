"""
patternmem.reflector
~~~~~~~~~~~~~~~~~~~~~
BackgroundReflector — asyncio queue consumer for Phase 3 of the 3-phase loop.

Responsibilities
----------------
1. Dequeue ``(FailureSignal, query, chunks, answer, matched_pattern_id_or_None)``
   tuples from the internal asyncio queue.
2. Use the resolved LLM to extract a ``FailurePattern`` from the signal.
3. Write the pattern to the backend (or update+decay an existing one).
4. Emit the score to the configured observability backend.

Lifecycle
---------
- Started as an ``asyncio.Task`` on the first ``ainvoke()`` call.
- Cancelled and drained when ``PatternMemMiddleware.__aexit__`` is called.
- Never blocks ``ainvoke()`` (Invariant 3).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

from patternmem.backend import MemoryBackend
from patternmem.decay import compute_new_decay_weight, should_evict
from patternmem.observability import ObservabilityEmitter
from patternmem.types import FailurePattern, FailureSignal

logger = logging.getLogger(__name__)

_SENTINEL = object()  # Signals queue shutdown


@dataclass
class _ReflectorJob:
    signal: FailureSignal
    query_embedding: list[float]
    matched_pattern: Optional[FailurePattern]  # None if Phase 1 was a miss


class BackgroundReflector:
    """Async queue consumer that persists failure patterns to the backend.

    Parameters
    ----------
    backend:
        The storage backend to write/update/delete patterns in.
    llm:
        The resolved LLM.  Used to extract ``root_cause`` and ``hint_text``
        from the raw ``FailureSignal``.
    emitter:
        Observability emitter.  No-op if observability is disabled.
    """

    def __init__(
        self,
        backend: MemoryBackend,
        llm: Any,
        emitter: ObservabilityEmitter,
    ) -> None:
        self._backend = backend
        self._llm = llm
        self._emitter = emitter
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=1000)
        self._task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background consumer task."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._consume(), name="patternmem.reflector")

    async def stop(self) -> None:
        """Signal shutdown and wait for the queue to drain."""
        await self._queue.put(_SENTINEL)
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()

    # ------------------------------------------------------------------
    # Enqueue (called from Phase 2 background task in middleware)
    # ------------------------------------------------------------------

    def enqueue(
        self,
        signal: FailureSignal,
        query_embedding: list[float],
        matched_pattern: FailurePattern | None = None,
    ) -> None:
        """Non-blocking enqueue — never raises even if the queue is full."""
        try:
            self._queue.put_nowait(
                _ReflectorJob(
                    signal=signal,
                    query_embedding=query_embedding,
                    matched_pattern=matched_pattern,
                )
            )
        except asyncio.QueueFull:
            logger.warning("PatternMem reflector queue is full; dropping signal %s", signal.trace_id)

    # ------------------------------------------------------------------
    # Consumer loop
    # ------------------------------------------------------------------

    async def _consume(self) -> None:
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                self._queue.task_done()
                break
            try:
                await self._process(item)
            except Exception:  # noqa: BLE001
                logger.exception("BackgroundReflector error processing job")
            finally:
                self._queue.task_done()

    async def _process(self, job: _ReflectorJob) -> None:
        signal = job.signal
        matched = job.matched_pattern

        if matched is not None:
            # --- Pattern exists: apply decay or reinforce ---
            new_weight = compute_new_decay_weight(
                stored=matched.decay_weight,
                old_score=matched.score,
                new_score=signal.score,
            )
            if should_evict(new_weight):
                logger.debug("Evicting pattern %s (weight %.3f)", matched.id, new_weight)
                try:
                    await self._backend.delete_pattern(matched.id)
                except KeyError:
                    pass  # Already deleted — idempotent
            else:
                logger.debug(
                    "Updating pattern %s decay_weight %.3f → %.3f",
                    matched.id,
                    matched.decay_weight,
                    new_weight,
                )
                await self._backend.update_pattern(matched.id, new_weight)
        else:
            # --- New pattern: extract hint via LLM and write ---
            root_cause, hint_text = await self._extract_from_llm(signal)

            pattern = FailurePattern(
                query_embedding=job.query_embedding,
                failure_type=signal.failure_type,
                root_cause=root_cause,
                hint_text=hint_text,
                score=signal.score,
                decay_weight=1.0,
            )
            await self._backend.write_pattern(pattern)
            logger.debug("Wrote new pattern %s (%s)", pattern.id, signal.failure_type.name)

        # Emit observability score
        self._emitter.emit_score(
            trace_id=signal.trace_id,
            score=signal.score,
            failure_type=signal.failure_type.name,
        )

    async def _extract_from_llm(self, signal: FailureSignal) -> tuple[str, str]:
        """Ask the LLM to refine root_cause and hint_text from the signal.

        Falls back to the signal's own fields if LLM call fails.
        """
        if signal.hint:
            # Signal already carries a hint (from the classification matrix)
            return signal.root_cause, signal.hint

        prompt = (
            f"A RAG pipeline produced a failure of type '{signal.failure_type.name}'.\n"
            f"Score: {signal.score:.3f}\n"
            f"Root cause description: {signal.root_cause}\n\n"
            "In one sentence each:\n"
            "1. What is the root cause?\n"
            "2. What retrieval or generation hint would prevent this failure next time?\n"
            "Reply in JSON: {\"root_cause\": \"...\", \"hint\": \"...\"}"
        )
        try:
            import json as _json

            # Duck-typed LLM call — supports langchain .invoke() and raw callables
            if hasattr(self._llm, "invoke"):
                raw = self._llm.invoke(prompt)
                if hasattr(raw, "content"):
                    raw = raw.content
            elif callable(self._llm):
                raw = self._llm(prompt)
            else:
                raw = str(self._llm)

            parsed = _json.loads(str(raw))
            return str(parsed.get("root_cause", signal.root_cause)), str(
                parsed.get("hint", signal.hint)
            )
        except Exception:  # noqa: BLE001
            return signal.root_cause, signal.hint
