"""
Procedural Memory Layer

Stores and retrieves skill/habit/workflow knowledge separately from
episodic memory. Procedural memories:
  - Don't decay with time (skills persist)
  - Strengthen with repeated execution
  - Are retrieved by task context, not just semantic similarity
  - Can be refined/versioned as the agent improves
"""

from cortex_ai.db.connection import get_db
from cortex_ai.ingestion.embeddings import embed_texts, embed_query


def store_procedural(
    agent_id: int,
    name: str,
    description: str,
    procedural_type: str,
    trigger_context: str,
    steps: list[str],
    domain_tags: list[str] | None = None,
    source_memory_ids: list[int] | None = None,
) -> int:
    """Store a new procedural memory (skill, workflow, pattern, preference, heuristic)."""
    text_for_embedding = f"{name}. {trigger_context}. {description}. {'. '.join(steps)}"
    embeddings = embed_texts([text_for_embedding])
    emb_str = "[" + ",".join(str(v) for v in embeddings[0]) + "]"

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO procedural_memories
                (agent_id, name, description, procedural_type, trigger_context,
                 steps, embedding, domain_tags, source_memory_ids,
                 proficiency, execution_count, success_count, success_rate, version, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, %s,
                        'novice', 0, 0, 0, 1, 'active')
                RETURNING id
                """,
                (
                    agent_id, name, description, procedural_type, trigger_context,
                    steps, emb_str, domain_tags or [], source_memory_ids or [],
                ),
            )
            row = cur.fetchone()
        conn.commit()

    print(f'[procedural] Stored: "{name}" ({procedural_type}) -> #{row["id"]}')
    return row["id"]


def retrieve_procedural(
    agent_id: int,
    task_context: str,
    limit: int = 5,
) -> list[dict]:
    """
    Retrieve relevant procedural memories for a task context.
    Uses trigger match + semantic similarity.
    """
    query_embedding = embed_query(task_context)
    emb_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
    trigger_match_text = task_context[:80]

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id, name, description, procedural_type, trigger_context,
                    steps, proficiency, execution_count, success_count, success_rate,
                    domain_tags, version,
                    1 - (embedding <=> %(emb)s::vector) AS cosine_sim,
                    CASE WHEN trigger_context ILIKE %(trigger)s THEN 1.0 ELSE 0.0 END AS trigger_match
                FROM procedural_memories
                WHERE agent_id = %(agent_id)s
                    AND status = 'active'
                    AND embedding IS NOT NULL
                ORDER BY trigger_match DESC, cosine_sim DESC
                LIMIT %(limit)s
                """,
                {
                    "emb": emb_str,
                    "trigger": f"%{trigger_match_text}%",
                    "agent_id": agent_id,
                    "limit": limit,
                },
            )
            results = cur.fetchall()

    return [
        {
            "id": r["id"],
            "name": r["name"],
            "description": r["description"],
            "procedural_type": r["procedural_type"],
            "trigger_context": r["trigger_context"],
            "steps": r["steps"] or [],
            "proficiency": r["proficiency"],
            "execution_count": r["execution_count"],
            "success_rate": float(r["success_rate"] or 0),
            "domain_tags": r["domain_tags"] or [],
            "relevance_score": float(r["cosine_sim"]) if r["trigger_match"] == 0 else 1.0,
            "match_type": "trigger" if r["trigger_match"] > 0 else "semantic",
        }
        for r in results
    ]


def record_execution(procedural_id: int, success: bool) -> dict:
    """Record execution outcome. This is how skills improve."""
    with get_db() as conn:
        with conn.cursor() as cur:
            success_inc = 1 if success else 0
            cur.execute(
                """
                UPDATE procedural_memories
                SET execution_count = execution_count + 1,
                    success_count = success_count + %s,
                    success_rate = (success_count + %s)::real / (execution_count + 1)::real,
                    last_executed_at = NOW(),
                    updated_at = NOW()
                WHERE id = %s
                """,
                (success_inc, success_inc, procedural_id),
            )

            cur.execute(
                "SELECT execution_count, success_count, success_rate, proficiency FROM procedural_memories WHERE id = %s",
                (procedural_id,),
            )
            current = cur.fetchone()

            if not current:
                conn.commit()
                return {"proficiency": "novice", "success_rate": 0}

            new_proficiency = current["proficiency"]
            ec = current["execution_count"]
            sr = float(current["success_rate"] or 0)

            if ec >= 20 and sr >= 0.9:
                new_proficiency = "expert"
            elif ec >= 10 and sr >= 0.8:
                new_proficiency = "proficient"
            elif ec >= 3 and sr >= 0.6:
                new_proficiency = "competent"

            if new_proficiency != current["proficiency"]:
                cur.execute(
                    "UPDATE procedural_memories SET proficiency = %s, updated_at = NOW() WHERE id = %s",
                    (new_proficiency, procedural_id),
                )

        conn.commit()

    return {"proficiency": new_proficiency, "success_rate": sr}
