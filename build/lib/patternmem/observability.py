"""
patternmem.observability
~~~~~~~~~~~~~~~~~~~~~~~~~
Thin, no-op-safe wrappers around Langfuse and OpenTelemetry.

Design rules
------------
- If the requested observability library is not installed, every function
  silently becomes a no-op.  This keeps zero-credential mode working with
  ``observability=None`` (Invariant 5).
- No observability call may block ``ainvoke()`` (Invariant 3).  All emits
  happen inside the already-async background Phase 3 task.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator, Literal, Optional


ObservabilityMode = Optional[Literal["langfuse", "otel"]]


class ObservabilityEmitter:
    """Emits trace spans and scores to the configured observability backend.

    Parameters
    ----------
    mode:
        ``"langfuse"``, ``"otel"``, or ``None`` (no-op).
    """

    def __init__(self, mode: ObservabilityMode = None) -> None:
        self._mode = mode
        self._langfuse: Any = None
        self._tracer: Any = None

        if mode == "langfuse":
            self._langfuse = self._init_langfuse()
        elif mode == "otel":
            self._tracer = self._init_otel()

    # ------------------------------------------------------------------
    # Init helpers (no-op on ImportError)
    # ------------------------------------------------------------------

    def _init_langfuse(self) -> Any:
        try:
            from langfuse import Langfuse
            return Langfuse()
        except ImportError:
            return None

    def _init_otel(self) -> Any:
        try:
            from opentelemetry import trace
            return trace.get_tracer("patternmem")
        except ImportError:
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def emit_score(
        self,
        trace_id: str,
        score: float,
        failure_type: str,
        name: str = "patternmem.failure_score",
    ) -> None:
        """Emit a score observation (non-blocking; called from Phase 3 background)."""
        if self._mode == "langfuse" and self._langfuse is not None:
            try:
                self._langfuse.score(
                    trace_id=trace_id,
                    name=name,
                    value=score,
                    comment=f"failure_type={failure_type}",
                )
            except Exception:  # noqa: BLE001
                pass

    @contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Any, None, None]:
        """Context manager that wraps a code block in a trace span.

        No-op if observability is disabled or the library is not installed.
        """
        if self._mode == "otel" and self._tracer is not None:
            with self._tracer.start_as_current_span(name) as span:
                if attributes:
                    for k, v in attributes.items():
                        span.set_attribute(k, str(v))
                yield span
        else:
            yield None
