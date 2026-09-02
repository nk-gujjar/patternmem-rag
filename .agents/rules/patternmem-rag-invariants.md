# patternmem-rag — Non-Negotiable Invariants

These rules apply to every file, every agent pass, and every code-generation step
in this codebase. Treat them as compiler errors: if any proposed change violates one,
reject the change and explain the violation before proceeding.

---

## Invariant 1 — The middleware never owns an LLM

The middleware borrows an LLM via `LLMResolver`, or the user passes `llm=`.
If resolution fails, raise `LLMResolverError` at `__init__` time.
Never fail silently at runtime.

## Invariant 2 — The middleware never mutates the user's prompt template

All influence travels through an `augmented_input` context dict.
Never string-patch a prompt object.

## Invariant 3 — The user-facing response is never blocked by evaluation or memory writes

Phase 2 (eval) and Phase 3 (classify + write) are always async/background.
`ainvoke()` returns as soon as the wrapped pipeline returns.

## Invariant 4 — Retrieval hints and generation constraints never cross injection lanes

- A retrieval-type hint → `augmented_input["retrieval_hint"]` only.
- A generation-type constraint → `augmented_input["generation_constraint"]` only.
No code may place either type in the wrong key.

## Invariant 5 — Zero-credential local mode must always work

`backend="json"`, `eval="none"`, `observability=None` is a first-class, tested
configuration, not a fallback. It must work with zero env vars and zero API keys.

## Invariant 6 — Pattern memory is not append-only

`decay_weight` drives eviction. The graph must self-prune.
Any backend implementation that silently ignores decay/eviction violates this invariant.

## Invariant 7 — EvalRouter output is always a FailureSignal

No code outside `EvalRouter` ever touches a raw RAGAS or DeepEval object.
`EvalRouter` is the only permitted translation boundary between eval libraries and
the rest of the system.
