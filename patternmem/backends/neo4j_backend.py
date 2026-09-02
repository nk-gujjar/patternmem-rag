"""
patternmem.backends.neo4j_backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Neo4j / AuraDB backend for PatternMem.  Requires the ``[neo4j]`` optional extra::

    pip install patternmem-rag[neo4j]

Graph schema (implement exactly — do not deviate)
--------------------------------------------------
Nodes
~~~~~
- ``FailurePattern {id, query_embedding, failure_type, root_cause, hint_text,
                    score, created_at, decay_weight}``
- ``QueryType {label}``   — values: temporal | entity | multi-hop | factoid
- ``RetrievalHint {text}``

Edges
~~~~~
- ``(:FailurePattern)-[:TRIGGERED_BY]->(:QueryType)``
- ``(:FailurePattern)-[:SUGGESTS]->(:RetrievalHint)``
- ``(:RetrievalHint)-[:SIMILAR_TO {sim_score}]->(:RetrievalHint)``

Vector index
~~~~~~~~~~~~
Named ``'failure_pattern_embeddings'`` on the ``query_embedding`` property of
``FailurePattern`` nodes.  Created automatically on first use.

Lookup Cypher (from spec — do not modify)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    CALL db.index.vector.queryNodes('failure_pattern_embeddings', 5, $query_embedding)
    YIELD node, score
    WHERE score > 0.82
    RETURN node.hint_text, node.failure_type
    ORDER BY score DESC LIMIT 3
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

try:
    from neo4j import AsyncGraphDatabase  # type: ignore[import]
    from neo4j.exceptions import ClientError  # type: ignore[import]

    _NEO4J_AVAILABLE = True
except ImportError:
    _NEO4J_AVAILABLE = False

from patternmem.backend import MemoryBackend
from patternmem.types import FailurePattern, FailureType

# ---------------------------------------------------------------------------
# Cypher statements
# ---------------------------------------------------------------------------

_CREATE_VECTOR_INDEX = """
CREATE VECTOR INDEX failure_pattern_embeddings IF NOT EXISTS
FOR (n:FailurePattern)
ON n.query_embedding
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}
"""

_UPSERT_PATTERN = """
MERGE (p:FailurePattern {id: $id})
SET p.query_embedding = $query_embedding,
    p.failure_type    = $failure_type,
    p.root_cause      = $root_cause,
    p.hint_text       = $hint_text,
    p.score           = $score,
    p.created_at      = $created_at,
    p.decay_weight    = $decay_weight
WITH p
MERGE (h:RetrievalHint {text: $hint_text})
MERGE (p)-[:SUGGESTS]->(h)
"""

_LOOKUP_PATTERNS = """
CALL db.index.vector.queryNodes('failure_pattern_embeddings', 5, $query_embedding)
YIELD node, score
WHERE score > $threshold
RETURN node.id           AS id,
       node.query_embedding AS query_embedding,
       node.failure_type AS failure_type,
       node.root_cause   AS root_cause,
       node.hint_text    AS hint_text,
       node.score        AS score,
       node.created_at   AS created_at,
       node.decay_weight AS decay_weight,
       score             AS sim_score
ORDER BY score DESC LIMIT 3
"""

_COUNT_PATTERNS = "MATCH (p:FailurePattern) RETURN count(p) AS cnt"

_UPDATE_DECAY = """
MATCH (p:FailurePattern {id: $id})
SET p.decay_weight = $decay_weight
RETURN p.id AS id
"""

_DELETE_PATTERN = """
MATCH (p:FailurePattern {id: $id})
DETACH DELETE p
RETURN $id AS id
"""


class Neo4jBackend(MemoryBackend):
    """Neo4j / AuraDB backend with native vector index lookup.

    Parameters
    ----------
    uri:
        Bolt or neo4j+s URI (e.g. ``"bolt://localhost:7687"`` or
        ``"neo4j+s://<aura-id>.databases.neo4j.io"``).
    auth:
        ``(username, password)`` tuple.
    database:
        Target database name.  ``"neo4j"`` for most single-DB setups.
    similarity_threshold:
        Cosine similarity floor for the vector index lookup.  Passed as
        ``$threshold`` to the lookup Cypher.
    """

    def __init__(
        self,
        uri: str,
        auth: tuple[str, str],
        database: str = "neo4j",
        similarity_threshold: float = 0.82,
    ) -> None:
        if not _NEO4J_AVAILABLE:
            raise ImportError(
                "Neo4jBackend requires the 'neo4j' driver. "
                "Install it with: pip install patternmem-rag[neo4j]"
            )
        self._driver = AsyncGraphDatabase.driver(uri, auth=auth)
        self._database = database
        self._threshold = similarity_threshold
        self._index_ready = False

    async def _ensure_index(self) -> None:
        """Create the vector index if it does not already exist."""
        if self._index_ready:
            return
        async with self._driver.session(database=self._database) as session:
            try:
                await session.run(_CREATE_VECTOR_INDEX)
            except ClientError:
                # Index may already exist under a different schema — ignore
                pass
        self._index_ready = True

    async def write_pattern(self, pattern: FailurePattern) -> None:
        await self._ensure_index()
        async with self._driver.session(database=self._database) as session:
            await session.run(
                _UPSERT_PATTERN,
                id=pattern.id,
                query_embedding=pattern.query_embedding,
                failure_type=pattern.failure_type.name,
                root_cause=pattern.root_cause,
                hint_text=pattern.hint_text,
                score=pattern.score,
                created_at=pattern.created_at.isoformat(),
                decay_weight=pattern.decay_weight,
            )

    async def lookup_patterns(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[FailurePattern]:
        await self._ensure_index()
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                _LOOKUP_PATTERNS,
                query_embedding=query_embedding,
                threshold=self._threshold,
            )
            records = await result.data()

        patterns: list[FailurePattern] = []
        for rec in records:
            patterns.append(
                FailurePattern(
                    id=rec["id"],
                    query_embedding=list(rec["query_embedding"]),
                    failure_type=FailureType[rec["failure_type"]],
                    root_cause=rec["root_cause"],
                    hint_text=rec["hint_text"],
                    score=rec["score"],
                    created_at=datetime.fromisoformat(rec["created_at"]).replace(
                        tzinfo=timezone.utc
                    ),
                    decay_weight=rec["decay_weight"],
                )
            )
        return patterns[:top_k]

    async def get_stats(self) -> dict[str, Any]:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(_COUNT_PATTERNS)
            record = await result.single()
        return {
            "count": int(record["cnt"]) if record else 0,
            "database": self._database,
            "backend": "neo4j",
        }

    async def update_pattern(self, pattern_id: str, decay_weight: float) -> None:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                _UPDATE_DECAY, id=pattern_id, decay_weight=decay_weight
            )
            record = await result.single()
            if record is None:
                raise KeyError(
                    f"Pattern {pattern_id!r} not found in Neo4j backend"
                )

    async def delete_pattern(self, pattern_id: str) -> None:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(_DELETE_PATTERN, id=pattern_id)
            summary = await result.consume()
            if summary.counters.nodes_deleted == 0:
                raise KeyError(
                    f"Pattern {pattern_id!r} not found in Neo4j backend"
                )

    async def close(self) -> None:
        """Close the driver connection pool.  Call on teardown."""
        await self._driver.close()
