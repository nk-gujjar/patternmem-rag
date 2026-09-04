"""
examples/zero_credential_demo.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Zero-credential end-to-end demo of PatternMem RAG middleware.

Run with::

    python examples/zero_credential_demo.py

Requirements
------------
- No API keys
- No environment variables
- No external services

This demo uses:
- backend="json"  — stores patterns in ~/.patternmem/demo_patterns.json
- eval="none"     — no RAGAS or DeepEval; no LLM needed for evaluation
- observability=None — no Langfuse or OTel

The mock pipeline simulates a RAG system that answers questions.  It includes
a 'reflect' key in its output that PatternMem uses for feedback (without
calling any external eval library).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

# Ensure any expired/stale HuggingFace token doesn't block the public MiniLM download.
# The model is freely public — no token is needed.
for _hf_key in ("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
    os.environ.pop(_hf_key, None)

# Add the project root to sys.path so this script works from the examples/ dir
sys.path.insert(0, str(Path(__file__).parent.parent))

from patternmem import PatternMemMiddleware
from patternmem.backends.json_backend import JSONBackend


# ---------------------------------------------------------------------------
# Mock pipeline — simulates a RAG system
# ---------------------------------------------------------------------------


class MockRAGPipeline:
    """Simulates a RAG pipeline that initially fails and then improves.

    - Queries 1 and 2: low faithfulness (reflect=0.2)
    - Queries 3+: high faithfulness (reflect=0.9) after hint injection
    """

    def __init__(self) -> None:
        # LLMResolver needs to find an LLM — expose one as an attribute
        self.llm = _MockLLM()
        self.call_count = 0

    def __call__(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.call_count += 1
        retrieval_hint = kwargs.get("retrieval_hint", "")

        print(f"\n  📥 Pipeline call #{self.call_count}: {query!r}")
        if retrieval_hint:
            print(f"  💡 Hint injected: {retrieval_hint!r}")

        if self.call_count <= 2:
            return {
                "answer": "I'm not sure about that.",
                "chunks": [],
                "reflect": 0.2,  # low faithfulness — triggers pattern learning
            }
        # After hint injection on query 3, the pipeline "improves"
        return {
            "answer": "The answer is 42, based on the retrieved context.",
            "chunks": ["The answer to everything is 42. Douglas Adams, 1979."],
            "reflect": 0.92,  # high faithfulness — triggers reinforcement
        }


class _MockLLM:
    """Minimal LLM-like object (no API calls)."""

    def invoke(self, prompt: str) -> str:
        return '{"root_cause": "retriever missed relevant docs", "hint": "add synonym expansion"}'

    def __call__(self, prompt: str) -> str:
        return self.invoke(prompt)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


async def main() -> None:
    print("=" * 60)
    print("  PatternMem RAG — Zero-Credential Demo")
    print("=" * 60)
    print("\nConfiguration:")
    print("  backend      = json  (local file, no API key)")
    print("  eval         = none  (no RAGAS or DeepEval)")
    print("  observability= None  (no Langfuse or OTel)\n")

    pipeline = MockRAGPipeline()

    import tempfile
    # Use the system temp dir for cross-platform compatibility (Linux/macOS/Windows)
    demo_store = Path(tempfile.gettempdir()) / "patternmem_demo_patterns.json"
    demo_store.unlink(missing_ok=True)  # clean start

    backend = JSONBackend(
        path=demo_store,
        similarity_threshold=0.70,  # slightly lower for the demo
    )

    async with PatternMemMiddleware(
        pipeline=pipeline,
        backend=backend,
        eval="none",       # Invariant 5: zero-credential mode
        observability=None,
    ) as mw:

        queries = [
            "What is the answer to the ultimate question?",
            "What is the answer to the ultimate question?",  # same query → hint injection
            "What is the answer to the ultimate question?",  # should get hint + succeed
        ]

        for i, query in enumerate(queries, 1):
            print(f"\n{'─' * 50}")
            print(f"Query {i}: {query!r}")
            result = await mw.ainvoke(query)
            print(f"  ✅ Answer: {result['answer']!r}")
            print(f"  📊 Faithfulness: {result.get('reflect', 'N/A')}")

            # Give background tasks a moment to process
            await asyncio.sleep(0.3)

        stats = await backend.get_stats()
        print(f"\n{'─' * 50}")
        print(f"\n📦 Backend stats: {stats}")

    print("\n" + "=" * 60)
    print("  ✅ Demo complete — zero env vars, zero API keys.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
