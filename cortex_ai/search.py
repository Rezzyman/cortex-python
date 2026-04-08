"""
CORTEX Search — Hybrid 7-factor scoring.

score = 0.50 * cosine_similarity
      + 0.20 * text_match
      + 0.15 * recency
      + 0.10 * resonance
      + 0.05 * priority_boost
"""

from dataclasses import dataclass

from cortex_ai.db.connection import get_db, resolve_agent
from cortex_ai.ingestion.embeddings import embed_query


@dataclass
class SearchResult:
    id: int
    content: str
    source: str | None
    score: float
    entities: list[str]
    semantic_tags: list[str]


def search(
    query: str,
    agent_id: str = "default",
    limit: int = 10,
) -> list[SearchResult]:
    """
    Hybrid 7-factor search across all memories.

    Returns ranked results combining semantic similarity, text match,
    recency, resonance, and priority.
    """
    agent_num = resolve_agent(agent_id)
    query_embedding = embed_query(query)
    emb_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH vector_scores AS (
                    SELECT
                        id, content, source, source_type, priority,
                        resonance_score, entities, semantic_tags, created_at,
                        valid_from, valid_until, superseded_by,
                        1 - (embedding <=> %(emb)s::vector) AS cosine_sim,
                        CASE WHEN content ILIKE %(text_match)s THEN 1.0 ELSE 0.0 END AS text_match,
                        EXP(-0.023 * EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400) AS recency,
                        LEAST(resonance_score / 10.0, 1.0) AS norm_resonance,
                        CASE priority
                            WHEN 0 THEN 1.0 WHEN 1 THEN 0.8 WHEN 2 THEN 0.5
                            WHEN 3 THEN 0.3 WHEN 4 THEN 0.1 ELSE 0.5
                        END AS priority_boost
                    FROM memory_nodes
                    WHERE agent_id = %(agent_id)s
                        AND status = 'active'
                        AND embedding IS NOT NULL
                )
                SELECT id, content, source, entities, semantic_tags,
                    (0.50 * cosine_sim + 0.20 * text_match + 0.15 * recency
                     + 0.10 * norm_resonance + 0.05 * priority_boost) AS hybrid_score
                FROM vector_scores
                ORDER BY hybrid_score DESC
                LIMIT %(limit)s
                """,
                {
                    "emb": emb_str,
                    "text_match": f"%{query}%",
                    "agent_id": agent_num,
                    "limit": limit,
                },
            )

            results = cur.fetchall()

            # Update access counts
            if results:
                ids = [r["id"] for r in results]
                cur.execute(
                    """
                    UPDATE memory_nodes
                    SET access_count = access_count + 1, last_accessed_at = NOW()
                    WHERE id = ANY(%s)
                    """,
                    (ids,),
                )
                conn.commit()

    return [
        SearchResult(
            id=r["id"],
            content=r["content"],
            source=r["source"],
            score=float(r["hybrid_score"]),
            entities=r["entities"] or [],
            semantic_tags=r["semantic_tags"] or [],
        )
        for r in results
    ]


def recall(
    query: str,
    agent_id: str = "default",
    token_budget: int = 4000,
) -> str:
    """
    Token-budget-aware context retrieval.

    Fetches the most relevant memories that fit within the token budget.
    Returns formatted context string.
    """
    from cortex_ai.ingestion.chunker import count_tokens

    results = search(query, agent_id=agent_id, limit=50)

    context_parts: list[str] = []
    used_tokens = 0

    for r in results:
        tokens = count_tokens(r.content)
        if used_tokens + tokens > token_budget:
            break
        context_parts.append(r.content)
        used_tokens += tokens

    return "\n\n".join(context_parts)
