"""
patternmem.middleware
~~~~~~~~~~~~~~~~~~~~~~
PatternMemMiddleware — the public one-line integration point.

Usage::

    from patternmem import PatternMemMiddleware

    middleware = PatternMemMiddleware(pipeline=my_rag_pipeline)
    answer = await middleware.ainvoke("What is the capital of France?")

    # Or as an async context manager (recommended for lifecycle management):
    async with PatternMemMiddleware(pipeline=my_rag_pipeline) as m:
        answer = await m.ainvoke("What is the capital of France?")

The 3-phase loop
----------------
Phase 1 (sync, ~50 ms target):
    Embed query (MiniLM) → lookup patterns → augment → call pipeline → return.
    The caller receives the answer *immediately* upon pipeline return.

Phase 2 (async background, non-blocking):
    EvalRouter evaluates (query, chunks, answer).
    FailureSignal(s) are enqueued into the BackgroundReflector queue.

Phase 3 (async background, queue consumer):
    BackgroundReflector dequeues → LLM extracts pattern → backend write/decay.

Invariants enforced here
------------------------
- Invariant 1: LLMResolverError is raised at __init__, not at ainvoke time.
- Invariant 3: ainvoke() returns as soon as the pipeline returns.
- Invariant 5: backend="json", eval="none", observability=None works with zero deps.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Callable, Literal, Optional

from sentence_transformers import SentenceTransformer

from patternmem.augmenter import Augmenter
from patternmem.backend import MemoryBackend
from patternmem.eval_router import EvalRouter
from patternmem.observability import ObservabilityEmitter
from patternmem.reflector import BackgroundReflector
from patternmem.resolver import LLMResolver
from patternmem.types import FailurePattern, FailureSignal

logger = logging.getLogger(__name__)

# MiniLM model name — fully local, no API key
_EMBED_MODEL = "all-MiniLM-L6-v2"


def _build_backend(
    kind: str,
    similarity_threshold: float,
) -> MemoryBackend:
    """Instantiate the requested backend with sensible defaults."""
    if kind == "json":
        from patternmem.backends.json_backend import JSONBackend
        return JSONBackend(similarity_threshold=similarity_threshold)
    if kind == "sqlite":
        from patternmem.backends.sqlite_backend import SQLiteBackend
        return SQLiteBackend(similarity_threshold=similarity_threshold)
    if kind == "networkx":
        from patternmem.backends.networkx_backend import NetworkXBackend
        return NetworkXBackend(similarity_threshold=similarity_threshold)
    if kind == "chroma":
        from patternmem.backends.chroma_backend import ChromaBackend
        return ChromaBackend(similarity_threshold=similarity_threshold)
    if kind == "faiss":
        from patternmem.backends.faiss_backend import FAISSBackend
        return FAISSBackend(similarity_threshold=similarity_threshold)
    if kind == "neo4j":
        raise ValueError(
            "Neo4jBackend requires additional configuration (uri, auth). "
            "Instantiate it directly and pass a MemoryBackend instance instead."
        )
    raise ValueError(
        f"Unknown backend: {kind!r}. "
        "Choose from: json, sqlite, networkx, chroma, faiss, neo4j."
    )


def _extract_chunks(pipeline_output: Any) -> list[str]:
    """Best-effort extraction of retrieved context chunks from pipeline output."""
    if isinstance(pipeline_output, dict):
        for key in ("chunks", "context", "source_documents", "contexts"):
            val = pipeline_output.get(key)
            if isinstance(val, list):
                return [str(v) for v in val]
    return []


def _extract_answer(pipeline_output: Any) -> str:
    """Best-effort extraction of the generated answer from pipeline output."""
    if isinstance(pipeline_output, str):
        return pipeline_output
    if isinstance(pipeline_output, dict):
        for key in ("answer", "result", "output", "response", "text"):
            val = pipeline_output.get(key)
            if isinstance(val, str):
                return val
    return str(pipeline_output)


class PatternMemMiddleware:
    """Framework-agnostic RAG middleware with persistent failure-pattern memory.

    Parameters
    ----------
    pipeline:
        Any callable that accepts a query string (and optional kwargs) and
        returns an answer.  May be synchronous or asynchronous.
    llm:
        Optional explicit LLM.  If ``None``, ``LLMResolver`` probes the
        pipeline for a detectable LLM.  Raises ``LLMResolverError`` at init
        if neither succeeds.
    backend:
        Storage backend identifier, or an existing ``MemoryBackend`` instance.
        Accepts ``"json"`` (default), ``"sqlite"``, ``"networkx"``.
        For Neo4j, pass an instantiated ``Neo4jBackend`` object directly.
    eval:
        Evaluation mode.  ``"auto"`` (default) tries RAGAS then DeepEval then
        falls back to ``"none"`` behaviour.
    observability:
        ``"langfuse"``, ``"otel"``, or ``None`` (default, no-op).
    similarity_threshold:
        Cosine similarity floor for Phase 1 pattern lookup.
    allow_param_override:
        If ``True``, Augmenter may set ``llm_call_kwargs`` (e.g. temperature).
    rewrite_feedback:
        If ``True``, Augmenter adds a ``rewrite_constraint`` key.

    Raises
    ------
    LLMResolverError
        At construction time if the LLM cannot be resolved.
    """

    def __init__(
        self,
        pipeline: Callable[..., Any],
        llm: Optional[Any] = None,
        backend: "Literal['neo4j', 'sqlite', 'json', 'networkx', 'chroma', 'faiss'] | MemoryBackend" = "json",
        eval: Literal["ragas", "deepeval", "auto", "none"] = "auto",
        observability: Optional[Literal["langfuse", "otel"]] = None,
        similarity_threshold: float = 0.82,
        allow_param_override: bool = False,
        rewrite_feedback: bool = False,
    ) -> None:
        self._pipeline = pipeline
        self._similarity_threshold = similarity_threshold

        # --- Backend ---
        if isinstance(backend, MemoryBackend):
            self._backend: MemoryBackend = backend
        else:
            self._backend = _build_backend(str(backend), similarity_threshold)

        # --- LLM Resolution (Invariant 1: fail here, not at runtime) ---
        # When eval="none", we still attempt resolution so the user gets an
        # early error if they forget llm=.  They can pass llm=None explicitly
        # only when building a test double.
        self._resolver = LLMResolver(pipeline=pipeline, llm=llm)

        # --- EvalRouter ---
        self._eval_router = EvalRouter(mode=eval)

        # --- Augmenter ---
        self._augmenter = Augmenter(
            pipeline=pipeline,
            rewrite_feedback=rewrite_feedback,
            allow_param_override=allow_param_override,
        )

        # --- Observability ---
        self._emitter = ObservabilityEmitter(mode=observability)

        # --- BackgroundReflector (started lazily on first ainvoke) ---
        self._reflector = BackgroundReflector(
            backend=self._backend,
            llm=self._resolver.llm,
            emitter=self._emitter,
        )

        # --- Embedding model (MiniLM, fully local) ---
        self._encoder: SentenceTransformer | None = None  # lazy init

    def _get_encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            # token=False forces anonymous download, bypassing any cached/expired
            # HuggingFace tokens.  The model is freely public.
            self._encoder = SentenceTransformer(_EMBED_MODEL, token=False)
        return self._encoder

    def _embed(self, text: str) -> list[float]:
        enc = self._get_encoder()
        vec = enc.encode(text, normalize_embeddings=True)
        result: list[float] = vec.tolist()
        return result

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "PatternMemMiddleware":
        self._reflector.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self._reflector.stop()

    # ------------------------------------------------------------------
    # Core public method
    # ------------------------------------------------------------------

    async def ainvoke(self, query: str, **kwargs: Any) -> Any:
        """Invoke the wrapped pipeline with optional pattern-based augmentation.

        Phase 1 runs synchronously in this coroutine; Phases 2 and 3 are fired
        as background tasks.  The caller receives the pipeline's answer as soon
        as it is available (Invariant 3).

        Parameters
        ----------
        query:
            The user's natural-language query.
        **kwargs:
            Additional keyword arguments forwarded to the pipeline.

        Returns
        -------
        Any
            Whatever the wrapped pipeline returns.
        """
        # Ensure the reflector is running
        self._reflector.start()

        # ------------------------------------------------------------------
        # Phase 1 — Embed → Lookup → Augment → Call pipeline
        # ------------------------------------------------------------------
        with self._emitter.span("patternmem.phase1", {"query": query[:200]}):
            query_embedding = await asyncio.to_thread(self._embed, query)

            matched_patterns: list[FailurePattern] = await self._backend.lookup_patterns(
                query_embedding, top_k=3
            )

            if matched_patterns:
                augmented = self._augmenter.build(matched_patterns)
                logger.debug(
                    "Phase 1 HIT: %d patterns matched, augmented keys: %s",
                    len(matched_patterns),
                    list(augmented.keys()),
                )
                pipeline_kwargs = {**kwargs, **augmented}
            else:
                logger.debug("Phase 1 MISS: no patterns above threshold")
                pipeline_kwargs = kwargs

            # Call the pipeline (sync, async function, or async callable instance)
            if inspect.iscoroutinefunction(self._pipeline) or inspect.iscoroutinefunction(
                getattr(self._pipeline, "__call__", None)
            ):
                pipeline_output = await self._pipeline(query, **pipeline_kwargs)
            else:
                pipeline_output = await asyncio.to_thread(
                    self._pipeline, query, **pipeline_kwargs
                )

        # Return to caller immediately (Invariant 3)
        # ------------------------------------------------------------------
        # Phase 2 — Background eval → enqueue to reflector
        # ------------------------------------------------------------------
        asyncio.create_task(
            self._phase2(
                query=query,
                query_embedding=query_embedding,
                pipeline_output=pipeline_output,
                matched_patterns=matched_patterns,
            ),
            name="patternmem.phase2",
        )

        return pipeline_output

    # ------------------------------------------------------------------
    # Phase 2 (background)
    # ------------------------------------------------------------------

    async def _phase2(
        self,
        query: str,
        query_embedding: list[float],
        pipeline_output: Any,
        matched_patterns: list[FailurePattern],
    ) -> None:
        """Run EvalRouter in background and enqueue signals for Phase 3."""
        try:
            with self._emitter.span("patternmem.phase2", {"query": query[:200]}):
                chunks = _extract_chunks(pipeline_output)
                answer = _extract_answer(pipeline_output)

                signals: list[FailureSignal] = await asyncio.to_thread(
                    self._eval_router.evaluate,
                    query,
                    chunks,
                    answer,
                    pipeline_output,
                )

                for signal in signals:
                    # Match signal to an existing pattern by failure_type
                    matched = next(
                        (
                            p
                            for p in matched_patterns
                            if p.failure_type == signal.failure_type
                        ),
                        None,
                    )
                    # matched=None → new pattern will be written in Phase 3
                    self._reflector.enqueue(
                        signal=signal,
                        query_embedding=query_embedding,
                        matched_pattern=matched,
                    )
        except Exception:  # noqa: BLE001
            logger.exception("Phase 2 background task failed")
