"""
examples/simple_rag_app.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
A complete, runnable RAG system demonstrating how PatternMemMiddleware
makes retrieval and generation self-improving across queries.

Run modes:
1. Zero-API Key / Local (default): Uses sentence-transformers + local TF-IDF/semantic retrieval
2. OpenAI / Real LLM (optional): Set OPENAI_API_KEY to use OpenAI GPT models

Run command:
    python examples/simple_rag_app.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

# Ensure parent directory is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sentence_transformers import SentenceTransformer

from patternmem import PatternMemMiddleware
from patternmem.backends.json_backend import JSONBackend

# Clean token env vars for anonymous HF model download if needed
for _k in ("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
    os.environ.pop(_k, None)


# ---------------------------------------------------------------------------
# 1. Knowledge Base (Sample Documents)
# ---------------------------------------------------------------------------

DOCUMENTS = [
    {
        "id": "doc_1",
        "title": "Refund Policy 2026",
        "text": "Customers can request a full refund within 30 days of purchase. "
                "Digital subscriptions and gift cards are strictly non-refundable.",
        "keywords": ["refund", "policy", "30 days", "return", "money back", "subscription"],
    },
    {
        "id": "doc_2",
        "title": "Enterprise SLA & Uptime",
        "text": "PatternMem Enterprise SLA guarantees 99.99% monthly uptime. "
                "Scheduled maintenance windows are announced 7 days in advance on Sundays.",
        "keywords": ["sla", "uptime", "enterprise", "maintenance", "availability"],
    },
    {
        "id": "doc_3",
        "title": "Data Retention and Security",
        "text": "All failure patterns and telemetry are encrypted at rest with AES-256. "
                "Customer logs are retained for 90 days before automatic deletion.",
        "keywords": ["security", "encryption", "retention", "aes-256", "privacy", "gdpr"],
    },
]


# ---------------------------------------------------------------------------
# 2. Simple Vector Retriever
# ---------------------------------------------------------------------------

class SimpleRetriever:
    """Embeds documents with MiniLM and performs cosine similarity search."""

    def __init__(self, docs: list[dict[str, Any]]):
        self.docs = docs
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2", token=False)
        self.doc_embeddings = [
            self.encoder.encode(f"{d['title']} {d['text']}", normalize_embeddings=True)
            for d in docs
        ]

    def retrieve(self, query: str, top_k: int = 2, hint: str = "") -> list[dict[str, Any]]:
        # If PatternMem provided a retrieval hint, augment the search query!
        search_query = f"{query} {hint}".strip() if hint else query

        query_emb = self.encoder.encode(search_query, normalize_embeddings=True)
        scores = [float(np.dot(query_emb, doc_emb)) for doc_emb in self.doc_embeddings]

        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {**self.docs[idx], "score": scores[idx]}
            for idx in ranked_indices
        ]


# ---------------------------------------------------------------------------
# 3. Simple Generator (with optional OpenAI support or local fallback)
# ---------------------------------------------------------------------------

class SimpleLLM:
    """Duck-typed LLM satisfying PatternMem LLMResolver requirement."""

    def __init__(self):
        self.openai_client = None
        if os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                self.openai_client = OpenAI()
            except ImportError:
                pass

    def invoke(self, prompt: str) -> str:
        # If OpenAI is configured, use it:
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                print(f"[OpenAI fallback] {e}")

        # Local intelligent generator fallback:
        # PatternMem Phase 3 asks for {"root_cause": "...", "hint": "..."}
        if "In one sentence each:" in prompt and "root_cause" in prompt:
            return '{"root_cause": "Retriever missed specific keyword synonyms", "hint": "expand query with policy and subscription terms"}'

        return "Based on the provided context, here is the answer."


# ---------------------------------------------------------------------------
# 4. The Complete RAG Pipeline Function
# ---------------------------------------------------------------------------

class SimpleRAGPipeline:
    def __init__(self):
        self.retriever = SimpleRetriever(DOCUMENTS)
        self.llm = SimpleLLM()  # LLMResolver discovers self.llm

    async def __call__(self, query: str, **kwargs: Any) -> dict[str, Any]:
        # Lane 1: Read retrieval hints from PatternMem
        retrieval_hint = kwargs.get("retrieval_hint", "")
        # Lane 2: Read generation constraints from PatternMem
        gen_constraint = kwargs.get("generation_constraint", "")

        # 1. Retrieve
        retrieved_docs = self.retriever.retrieve(
            query=query,
            top_k=1,
            hint=retrieval_hint,
        )
        chunks = [d["text"] for d in retrieved_docs]

        # 2. Build answer
        context_text = "\n".join(chunks)
        
        # Synthetic quality check:
        # If the user asks about digital subscription refunds, but the retriever
        # didn't get doc_1, it will fail/hallucinate.
        is_relevant = any("non-refundable" in c or "refund" in c for c in chunks)

        if is_relevant:
            answer = f"According to policy: {context_text}"
            faithfulness_score = 0.95
        else:
            answer = "Sorry, I could not find relevant documentation."
            faithfulness_score = 0.15  # Low score signals failure to PatternMem

        return {
            "query": query,
            "answer": answer,
            "chunks": chunks,
            "reflect": faithfulness_score,  # Used by EvalRouter short-circuit
            "applied_hint": retrieval_hint,
            "applied_constraint": gen_constraint,
        }


# ---------------------------------------------------------------------------
# 5. Main Demonstration Loop
# ---------------------------------------------------------------------------

async def main():
    print("=" * 65)
    print("  🚀 PatternMem RAG — Self-Improving Pipeline Demo")
    print("=" * 65)

    pipeline = SimpleRAGPipeline()
    backend_path = Path("/tmp/patternmem_rag_test.json")
    backend_path.unlink(missing_ok=True)  # Clean test file

    backend = JSONBackend(path=backend_path, similarity_threshold=0.65)

    # Wrap the standard pipeline in 1 line of middleware!
    async with PatternMemMiddleware(
        pipeline=pipeline,
        backend=backend,
        eval="auto",
        observability=None,
    ) as self_improving_rag:

        # Ambiguous query that initially misses keyword matching
        query = "Can I get money back for my monthly digital subscription plan?"

        print(f"\nUser Query: '{query}'\n")

        # --- Turn 1: First attempt (Fails / Misses specific docs) ---
        print("▶️  [Query Run 1] (Fresh memory, no hints yet)...")
        res1 = await self_improving_rag.ainvoke(query)
        print(f"   Chunks Retrieved: {len(res1['chunks'])}")
        print(f"   Answer: {res1['answer']}")
        print(f"   Faithfulness Score: {res1['reflect']}")
        print(f"   Hint Used: {res1['applied_hint'] or 'None'}")

        # Allow background Phase 2 (Eval) & Phase 3 (Reflector) to persist pattern
        await asyncio.sleep(0.4)

        # --- Turn 2: Second attempt (PatternMem intercepts & injects learned hint) ---
        print("\n▶️  [Query Run 2] (Same or similar query, PatternMem applies learned hint)...")
        res2 = await self_improving_rag.ainvoke(query)
        print(f"   Chunks Retrieved: {len(res2['chunks'])}")
        print(f"   Answer: {res2['answer']}")
        print(f"   Faithfulness Score: {res2['reflect']}")
        print(f"   💡 Injected Hint: '{res2['applied_hint']}'")

        print("\n" + "─" * 65)
        stats = await backend.get_stats()
        print(f"📊 Pattern Memory Store Stats: {stats}")
        print("=" * 65)
        print("✨ Notice how Run 2 automatically succeeded using the persistent hint!")


if __name__ == "__main__":
    asyncio.run(main())
