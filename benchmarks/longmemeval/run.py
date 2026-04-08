"""
LongMemEval Benchmark Runner for CORTEX Python

Same methodology as the TypeScript version:
- Ingest all haystack sessions per question
- Search with the question text
- Check if answer_session_ids appear in top K results
- Score Recall@K, MRR, Hit Rate

No LLM. No tricks. No hand-coded patches.

Usage: python benchmarks/longmemeval/run.py [--limit 50] [--topk 10]
"""

import json
import time
import sys
import os
import argparse
from pathlib import Path
from dataclasses import dataclass, field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cortex_ai.db.connection import init_database, get_db, resolve_agent
from cortex_ai.ingestion.chunker import chunk_text
from cortex_ai.ingestion.embeddings import embed_texts, embed_query
from cortex_ai.ingestion.entities import extract_entities, extract_semantic_tags


@dataclass
class QuestionResult:
    question_id: str
    question: str
    expected_session_ids: list[str]
    retrieved_session_ids: list[str]
    rank: int | None
    hit: bool


def clear_agent_data(agent_id: int):
    """Clear all memories for a benchmark agent."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_nodes WHERE agent_id = %s", (agent_id,))
        conn.commit()


def ingest_session(agent_id: int, session_id: str, content: str):
    """Ingest a conversation session (fast mode, no hippocampal encoding)."""
    chunks = chunk_text(content)
    if not chunks:
        return

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    with get_db() as conn:
        with conn.cursor() as cur:
            for i, chunk in enumerate(chunks):
                entities = extract_entities(chunk["text"])
                tags = extract_semantic_tags(chunk["text"])
                emb_str = "[" + ",".join(str(v) for v in embeddings[i]) + "]"

                cur.execute(
                    """
                    INSERT INTO memory_nodes
                    (agent_id, content, source, source_type, chunk_index,
                     embedding, entities, semantic_tags, priority,
                     resonance_score, novelty_score, status)
                    VALUES (%s, %s, %s, %s, %s, %s::vector, %s, %s, %s, %s, %s, 'active')
                    """,
                    (
                        agent_id,
                        chunk["text"],
                        f"longmemeval/{session_id}",
                        "benchmark",
                        i,
                        emb_str,
                        entities,
                        tags,
                        2,
                        5.0,
                        0.5,
                    ),
                )
        conn.commit()


def search_memories(agent_id: int, query: str, top_k: int = 10) -> list[dict]:
    """Hybrid 7-factor search."""
    query_embedding = embed_query(query)
    emb_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH vector_scores AS (
                    SELECT id, content, source,
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
                SELECT id, content, source,
                    (0.50 * cosine_sim + 0.20 * text_match + 0.15 * recency
                     + 0.10 * norm_resonance + 0.05 * priority_boost) AS hybrid_score
                FROM vector_scores
                ORDER BY hybrid_score DESC
                LIMIT %(limit)s
                """,
                {"emb": emb_str, "text_match": f"%{query}%", "agent_id": agent_id, "limit": top_k},
            )
            return cur.fetchall()


def find_rank(expected: list[str], retrieved: list[str]) -> int | None:
    """Find rank of first correct result."""
    for i, r in enumerate(retrieved):
        if r in expected:
            return i + 1
    return None


def main():
    parser = argparse.ArgumentParser(description="LongMemEval Benchmark for CORTEX Python")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions")
    parser.add_argument("--topk", type=int, default=10, help="Top K for retrieval")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N questions")
    parser.add_argument("--dataset", type=str, default="oracle", help="Dataset variant")
    args = parser.parse_args()

    print("=" * 50)
    print("  CORTEX Python V2.4 -- LongMemEval Benchmark")
    print("=" * 50)
    print(f"Dataset: {args.dataset} | Top-K: {args.topk} | Limit: {args.limit or 'all'} | Skip: {args.skip}")
    print()

    # Load dataset
    data_dir = Path(__file__).parent
    if args.dataset == "oracle":
        data_file = data_dir / "longmemeval_oracle.json"
    else:
        data_file = data_dir / "longmemeval_s.json"

    if not data_file.exists():
        print(f"Dataset not found: {data_file}")
        print("Download: curl -sL https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json -o benchmarks/longmemeval/longmemeval_oracle.json")
        sys.exit(1)

    with open(data_file) as f:
        questions = json.load(f)

    if args.skip > 0:
        questions = questions[args.skip:]
    if args.limit > 0:
        questions = questions[: args.limit]

    print(f"Loaded {len(questions)} questions\n")

    # Initialize
    init_database()

    # Use a separate agent for Python benchmarks
    agent_id = resolve_agent("benchmark-longmemeval-python")
    print(f"Agent ID: {agent_id}")

    results: list[QuestionResult] = []
    type_results: dict[str, list[QuestionResult]] = {}
    total_ingest_ms = 0
    total_search_ms = 0

    for qi, q in enumerate(questions):
        progress = f"[{qi + 1}/{len(questions)}]"

        # Clear previous question
        clear_agent_data(agent_id)

        # Ingest all haystack sessions
        ingest_start = time.time()
        for si, session in enumerate(q["haystack_sessions"]):
            session_id = q["haystack_session_ids"][si]
            session_text = "\n".join(
                f"[{turn['role']}] {turn['content']}" for turn in session
            )
            ingest_session(agent_id, session_id, session_text)
        ingest_ms = int((time.time() - ingest_start) * 1000)
        total_ingest_ms += ingest_ms

        # Search
        search_start = time.time()
        search_results = search_memories(agent_id, q["question"], args.topk)
        search_ms = int((time.time() - search_start) * 1000)
        total_search_ms += search_ms

        # Extract session IDs from results
        retrieved = []
        seen = set()
        for r in search_results:
            src = (r["source"] or "").split("/")[-1]
            if src and src not in seen:
                retrieved.append(src)
                seen.add(src)

        # Score
        rank = find_rank(q["answer_session_ids"], retrieved)
        hit = rank is not None

        result = QuestionResult(
            question_id=q["question_id"],
            question=q["question"],
            expected_session_ids=q["answer_session_ids"],
            retrieved_session_ids=retrieved,
            rank=rank,
            hit=hit,
        )
        results.append(result)

        qtype = q["question_type"]
        if qtype not in type_results:
            type_results[qtype] = []
        type_results[qtype].append(result)

        status = "HIT" if hit else "MISS"
        print(
            f"{progress} {status} (rank: {rank or '-'}) | {qtype} | "
            f"ingest: {ingest_ms}ms | search: {search_ms}ms | "
            f"sessions: {len(q['haystack_sessions'])}"
        )

    # Cleanup
    clear_agent_data(agent_id)

    # Score
    total = len(results)
    r1 = sum(1 for r in results if find_rank(r.expected_session_ids, r.retrieved_session_ids[:1]) is not None)
    r3 = sum(1 for r in results if find_rank(r.expected_session_ids, r.retrieved_session_ids[:3]) is not None)
    r5 = sum(1 for r in results if find_rank(r.expected_session_ids, r.retrieved_session_ids[:5]) is not None)
    r10 = sum(1 for r in results if find_rank(r.expected_session_ids, r.retrieved_session_ids[:10]) is not None)
    mrr = sum(1 / r.rank for r in results if r.rank is not None) / total if total > 0 else 0
    hits = sum(1 for r in results if r.hit)

    print("\n" + "=" * 50)
    print("  RESULTS (Python Implementation)")
    print("=" * 50)
    print(f"\nTotal Questions: {total}")
    print(f"| Recall@1  | {r1/total*100:.1f}% |")
    print(f"| Recall@3  | {r3/total*100:.1f}% |")
    print(f"| Recall@5  | {r5/total*100:.1f}% |")
    print(f"| Recall@10 | {r10/total*100:.1f}% |")
    print(f"| MRR       | {mrr*100:.1f}% |")
    print(f"| Hit Rate  | {hits/total*100:.1f}% |")
    print(f"| Misses    | {total - hits} |")

    print("\n--- By Question Type ---")
    for qtype, qresults in sorted(type_results.items()):
        t = len(qresults)
        tr5 = sum(1 for r in qresults if find_rank(r.expected_session_ids, r.retrieved_session_ids[:5]) is not None)
        tr10 = sum(1 for r in qresults if find_rank(r.expected_session_ids, r.retrieved_session_ids[:10]) is not None)
        tmrr = sum(1 / r.rank for r in qresults if r.rank is not None) / t if t > 0 else 0
        print(f"  {qtype} ({t}): R@5={tr5/t*100:.1f}% | R@10={tr10/t*100:.1f}% | MRR={tmrr*100:.1f}%")

    print(f"\n--- Timing ---")
    print(f"Total ingest: {total_ingest_ms/1000:.1f}s (avg: {total_ingest_ms/total/1000:.2f}s/q)")
    print(f"Total search: {total_search_ms/1000:.1f}s (avg: {total_search_ms/total:.0f}ms/q)")

    # Save results
    output_path = data_dir / f"results-python-{args.dataset}-top{args.topk}.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "benchmark": "LongMemEval",
                "implementation": "Python",
                "system": "CORTEX Python V2.4",
                "dataset": args.dataset,
                "topK": args.topk,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "results": {
                    "total": total,
                    "recall_at_1": r1 / total if total else 0,
                    "recall_at_5": r5 / total if total else 0,
                    "recall_at_10": r10 / total if total else 0,
                    "mrr": mrr,
                    "hit_rate": hits / total if total else 0,
                    "misses": total - hits,
                },
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
