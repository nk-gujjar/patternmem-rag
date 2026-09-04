# PatternMem RAG

[![CI](https://github.com/nk-gujjar/patternmem-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/nk-gujjar/patternmem-rag/actions)
[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/patternmem-rag/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://pypi.org/project/patternmem-rag/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Framework-agnostic Python middleware that wraps *any* existing RAG pipeline and makes it self-improving** — via persistent failure-pattern memory across queries.

```python
# Before
answer = my_rag_pipeline(query)

# After — one line
answer = await PatternMemMiddleware(pipeline=my_rag_pipeline).ainvoke(query)
```

---

## Why PatternMem?

Self-RAG, CRAG, and DSPy all reflect *within a single query*. PatternMem is the missing layer: it makes failure signals **persistent across the entire query history** — so the second time a pipeline fails on a similar question, it already knows what went wrong and pre-empts the failure.

PatternMem **does not reimplement** evaluation, LLM calling, or graph storage. It sits between your pipeline and the eval/storage libraries you already have.

---

## Quickstart

```bash
pip install patternmem-rag
```

```python
import asyncio
from patternmem import PatternMemMiddleware

async def my_rag_pipeline(query, **kwargs):
    # your existing pipeline here
    return {"answer": "...", "chunks": [...]}

async def main():
    async with PatternMemMiddleware(
        pipeline=my_rag_pipeline,
        backend="json",    # local file, zero credentials
        eval="auto",       # tries RAGAS → DeepEval → no-op
    ) as mw:
        answer = await mw.ainvoke("What is the capital of France?")
        print(answer)

asyncio.run(main())
```

## Zero-credential mode (no API keys needed)

```python
answer = await PatternMemMiddleware(
    pipeline=my_pipeline,
    backend="json",
    eval="none",
    observability=None,
).ainvoke(query)
```

Run the included demo:
```bash
python examples/zero_credential_demo.py
```

---

## How it works

```
Query ──► [Phase 1: ~50ms]
           Embed query (MiniLM, local)
           Lookup patterns (cosine similarity ≥ 0.82)
           on HIT  → Augmenter injects retrieval_hint / generation_constraint
           Pipeline called → Answer returned to caller immediately

           [Phase 2: async background]
           EvalRouter: RAGAS / DeepEval / none → FailureSignal

           [Phase 3: async background]
           BackgroundReflector: LLM extracts FailurePattern → Backend write
           Decay/eviction loop (patterns below weight 0.1 are pruned)
```

The caller **never waits** for Phases 2 or 3.

---

## Optional extras

| Extra | What it adds |
|---|---|
| `pip install patternmem-rag[ragas]` | RAGAS evaluation adapter |
| `pip install patternmem-rag[deepeval]` | DeepEval evaluation adapter |
| `pip install patternmem-rag[neo4j]` | Neo4j / AuraDB backend |
| `pip install patternmem-rag[langfuse]` | Langfuse observability |
| `pip install patternmem-rag[networkx]` | NetworkX in-memory backend |
| `pip install patternmem-rag[chroma]` | ChromaDB vector backend |
| `pip install patternmem-rag[faiss]` | FAISS local vector index backend |

---

## Configuration reference

```python
PatternMemMiddleware(
    pipeline,                          # any callable (sync or async)
    llm=None,                          # explicit LLM; auto-resolved if omitted
    backend="json",                    # "json" | "sqlite" | "networkx" | MemoryBackend
    eval="auto",                       # "ragas" | "deepeval" | "auto" | "none"
    observability=None,                # "langfuse" | "otel" | None
    similarity_threshold=0.82,         # cosine similarity floor for pattern lookup
    allow_param_override=False,        # allow temperature/CoT overrides
    rewrite_feedback=False,            # inject hints into query rewriter
)
```

---

## Backend choice guide

| Backend | Best for | Persistence | Dependencies |
|---|---|---|---|
| `"json"` | Zero-config, development | File | None |
| `"sqlite"` | Single-process production | File (WAL) | `aiosqlite` (core) |
| `"networkx"` | Notebooks, graph experiments | Optional file | `networkx` |
| `"chroma"` | Large stores, existing Chroma setup | File / HTTP server | `chromadb` |
| `"faiss"` | High-speed local search, no server | File (index + sidecar) | `faiss-cpu` |
| `"neo4j"` | Multi-process, AuraDB, scale | Native graph | `neo4j` driver |

---

## Custom backend

Implement `MemoryBackend` and pass an instance directly:

```python
from patternmem import MemoryBackend, PatternMemMiddleware

class MyRedisBackend(MemoryBackend):
    async def write_pattern(self, pattern): ...
    async def lookup_patterns(self, embedding, top_k=3): ...
    async def get_stats(self): ...
    async def update_pattern(self, id, decay_weight): ...
    async def delete_pattern(self, id): ...

mw = PatternMemMiddleware(pipeline=my_pipeline, backend=MyRedisBackend())
```

---

## FAQ

**Q: Does PatternMem replace RAGAS or DeepEval?**
No. It wraps them. It uses their scores as signals, stores the resulting patterns, and pre-empts future failures.

**Q: Does it change my prompts?**
Never. All augmentation flows through `augmented_input` kwargs — PatternMem never touches your prompt template.

**Q: What if evaluation isn't installed?**
`eval="none"` is a first-class mode. The middleware still runs the full 3-phase loop; Phase 2 returns an UNKNOWN signal and Phase 3 stores it with no external calls.

**Q: What's the LLM used for?**
Only Phase 3 (root cause extraction and hint generation from a `FailureSignal`). It borrows your pipeline's LLM — it never creates one.

**Q: What's out of scope?**
Celery integration (documented stub), Redis/Postgres backends (open ABC for community), any dashboard (use Langfuse's native UI).

---

## Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change.

- All backends must pass the contract test suite in `tests/contract/test_backend_contract.py`.
- Keep public API surface stable — anything not in `patternmem.__init__.__all__` is internal.

---

## License

MIT
