# Changelog

All notable changes to PatternMem RAG are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-09-04

### Added
- **PatternMemMiddleware** — core 3-phase async middleware loop
- **JSONBackend** — zero-dependency file-backed pattern storage
- **SQLiteBackend** — aiosqlite-powered single-process persistent backend
- **NetworkXBackend** — in-memory graph backend for notebooks
- **Neo4jBackend** — multi-process graph backend for production scale
- **EvalRouter** — normalises RAGAS / DeepEval outputs into `FailureSignal` lists
- **BackgroundReflector** — async queue consumer for pattern extraction (Phase 3)
- **Augmenter** — builds `augmented_input` dicts with retrieval/generation lanes
- **LLMResolver** — auto-discovers LLMs from wrapped pipeline (init-time, Invariant 1)
- **ObservabilityEmitter** — thin wrappers around Langfuse and OpenTelemetry
- **decay module** — pure reinforce/evict logic with configurable constants
- Zero-credential demo (`examples/zero_credential_demo.py`)
- 131 unit, contract, and integration tests across all backends
- Optional extras: `ragas`, `deepeval`, `neo4j`, `langfuse`, `langchain`, `networkx`, `celery`
- GitHub Actions CI (Python 3.10, 3.11, 3.12)

### Fixed
- Cross-platform demo path (was `/tmp/…`, now uses `tempfile.gettempdir()`)
- `typing.List` → built-in `list` in `types.py` (Python 3.10+)
- Deduplicated `_cosine_similarity` into `patternmem._utils`
- Bounded reflector queue (`maxsize=1000`) to prevent unbounded memory growth
