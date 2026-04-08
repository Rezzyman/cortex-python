"""Memory ingestion pipeline."""

from dataclasses import dataclass
from pathlib import Path

from cortex_ai.db.connection import get_db, resolve_agent
from cortex_ai.ingestion.chunker import chunk_text
from cortex_ai.ingestion.embeddings import embed_texts
from cortex_ai.ingestion.entities import extract_entities, extract_semantic_tags
from cortex_ai.hippocampus import dg_encode, compute_novelty


@dataclass
class IngestResult:
    memory_ids: list[int]
    chunks: int
    agent_id: int


def ingest(
    content: str,
    agent_id: str = "default",
    source: str = "api",
    priority: int = 2,
    hippocampal: bool = True,
) -> IngestResult:
    """
    Ingest text content into CORTEX.

    Chunks, embeds, extracts entities, optionally runs hippocampal encoding.
    """
    agent_num = resolve_agent(agent_id)
    chunks = chunk_text(content)

    if not chunks:
        return IngestResult(memory_ids=[], chunks=0, agent_id=agent_num)

    embeddings = embed_texts([c["text"] for c in chunks])
    inserted_ids: list[int] = []

    with get_db() as conn:
        with conn.cursor() as cur:
            for i, chunk in enumerate(chunks):
                entities = extract_entities(chunk["text"])
                tags = extract_semantic_tags(chunk["text"])

                novelty_score = 0.5
                if hippocampal:
                    try:
                        sparse_code = dg_encode(embeddings[i])
                        result = compute_novelty(agent_num, embeddings[i], sparse_code, priority)
                        novelty_score = result.novelty_score
                        priority = result.adjusted_priority
                    except Exception:
                        pass

                emb_str = "[" + ",".join(str(v) for v in embeddings[i]) + "]"
                cur.execute(
                    """
                    INSERT INTO memory_nodes
                    (agent_id, content, source, source_type, chunk_index,
                     embedding, entities, semantic_tags, priority,
                     resonance_score, novelty_score, status)
                    VALUES (%s, %s, %s, %s, %s, %s::vector, %s, %s, %s, %s, %s, 'active')
                    RETURNING id
                    """,
                    (
                        agent_num,
                        chunk["text"],
                        source,
                        "api",
                        i,
                        emb_str,
                        entities,
                        tags,
                        priority,
                        5.0,
                        novelty_score,
                    ),
                )
                row = cur.fetchone()
                inserted_ids.append(row["id"])

        conn.commit()

    return IngestResult(memory_ids=inserted_ids, chunks=len(chunks), agent_id=agent_num)


def ingest_file(
    path: str | Path,
    agent_id: str = "default",
    priority: int = 2,
) -> IngestResult:
    """Ingest a text/markdown file."""
    content = Path(path).read_text()
    return ingest(content, agent_id=agent_id, source=str(path), priority=priority)
