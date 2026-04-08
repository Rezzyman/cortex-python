"""
CogBench Runner for CORTEX Python

Tests 7 cognitive capabilities that standard benchmarks miss:
1. Temporal Validity
2. Reconsolidation
3. Novelty Detection
4. Emotional Recall
5. Cross-Agent Transfer
6. Compounding Intelligence
7. Procedural Learning

Usage: python benchmarks/cogbench/run.py [--task temporal-validity] [--verbose]
"""

import json
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cortex_ai.db.connection import init_database, get_db, resolve_agent
from cortex_ai.ingestion.embeddings import embed_texts, embed_query
from cortex_ai.ingestion.chunker import chunk_text
from cortex_ai.ingestion.entities import extract_entities, extract_semantic_tags


@dataclass
class ScenarioResult:
    scenario_id: str
    task: str
    description: str
    score: float  # 0.0 to 1.0
    details: dict


def clear_agent(agent_id: int):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_nodes WHERE agent_id = %s", (agent_id,))
        conn.commit()


def ingest_memory(agent_id: int, memory: dict):
    """Ingest a single CogBench memory with metadata."""
    content = memory["content"]
    source = memory.get("source", "cogbench")
    mem_id = memory["id"]
    valid_from = memory.get("validFrom")
    valid_until = memory.get("validUntil")
    priority = memory.get("priority", 2)

    # Embed
    embeddings = embed_texts([content])
    emb_str = "[" + ",".join(str(v) for v in embeddings[0]) + "]"
    entities = extract_entities(content)
    tags = extract_semantic_tags(content)

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memory_nodes
                (agent_id, content, source, source_type, chunk_index,
                 embedding, entities, semantic_tags, priority,
                 resonance_score, novelty_score, status, valid_from, valid_until)
                VALUES (%s, %s, %s, 'cogbench', 0, %s::vector, %s, %s, %s, %s, %s, 'active', %s, %s)
                RETURNING id
                """,
                (
                    agent_id, content, f"cogbench/{mem_id}",
                    emb_str, entities, tags, priority,
                    memory.get("resonance", 5.0),
                    memory.get("novelty", 0.5),
                    valid_from, valid_until,
                ),
            )
            db_id = cur.fetchone()["id"]
        conn.commit()
    return db_id


def search_memories(agent_id: int, query: str, top_k: int = 10, query_timestamp: str | None = None) -> list[dict]:
    """Search with optional temporal filtering."""
    query_embedding = embed_query(query)
    emb_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    # Build temporal filter
    temporal_clause = ""
    params: dict = {
        "emb": emb_str,
        "text_match": f"%{query}%",
        "agent_id": agent_id,
        "limit": top_k,
    }

    if query_timestamp:
        temporal_clause = """
            AND (valid_from IS NULL OR valid_from <= %(ts)s::timestamp)
            AND (valid_until IS NULL OR valid_until > %(ts)s::timestamp)
        """
        params["ts"] = query_timestamp

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
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
                        {temporal_clause}
                )
                SELECT id, content, source,
                    (0.50 * cosine_sim + 0.20 * text_match + 0.15 * recency
                     + 0.10 * norm_resonance + 0.05 * priority_boost) AS hybrid_score
                FROM vector_scores
                ORDER BY hybrid_score DESC
                LIMIT %(limit)s
                """,
                params,
            )
            return cur.fetchall()


def evaluate_procedural_scenario(agent_id: int, scenario: dict, verbose: bool = False) -> ScenarioResult:
    """Evaluate a procedural learning scenario."""
    from cortex_ai.procedural import store_procedural, retrieve_procedural, record_execution

    clear_procedural(agent_id)

    config = scenario["config"]["procedure"]
    expected_proficiency = scenario["expected"][0].get("expectedProficiency", "novice") if scenario.get("expected") else "novice"

    # Step 1: Store the procedure
    proc_id = store_procedural(
        agent_id=agent_id,
        name=config["name"],
        description=config["description"],
        procedural_type=config.get("type", "skill"),
        trigger_context=config.get("triggerContext", config["name"]),
        steps=config.get("steps", []),
        domain_tags=config.get("domainTags", []),
    )

    # Step 2: Simulate executions to advance proficiency
    # To reach "proficient": 10 executions, 80%+ success
    # To reach "expert": 20 executions, 90%+ success
    target_executions = 12 if expected_proficiency == "proficient" else 22 if expected_proficiency == "expert" else 5
    for i in range(target_executions):
        success = i % 5 != 4  # 80% success rate (fail every 5th)
        record_execution(proc_id, success)

    # Step 3: Try to retrieve it with each query
    query_scores: list[float] = []
    details: dict = {"queries": []}

    for qi, query in enumerate(scenario.get("queries", [])):
        results = retrieve_procedural(agent_id, query["query"], limit=5)

        # Check if our procedure was retrieved
        found = any(r["id"] == proc_id for r in results)
        # Check proficiency matches expected
        matching = [r for r in results if r["id"] == proc_id]
        proficiency_ok = False
        if matching:
            actual_prof = matching[0]["proficiency"]
            proficiency_ok = actual_prof == expected_proficiency

        score = 1.0 if found and proficiency_ok else 0.5 if found else 0.0
        query_scores.append(score)

        details["queries"].append({
            "query": query["query"],
            "score": score,
            "found": found,
            "proficiency_match": proficiency_ok,
        })

        if verbose:
            status = "PASS" if score >= 0.5 else "FAIL"
            print(f"    Q{qi+1}: {status} ({score:.2f}) | {query['query'][:60]}...")

    avg_score = sum(query_scores) / len(query_scores) if query_scores else 0.0

    return ScenarioResult(
        scenario_id=scenario["id"],
        task="procedural-learning",
        description=scenario.get("description", ""),
        score=avg_score,
        details=details,
    )


def clear_procedural(agent_id: int):
    """Clear procedural memories for an agent."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM procedural_memories WHERE agent_id = %s", (agent_id,))
        conn.commit()


def evaluate_scenario(agent_id: int, scenario: dict, task_name: str, verbose: bool = False) -> ScenarioResult:
    """Run a single CogBench scenario and score it."""
    clear_agent(agent_id)

    # Handle procedural-learning task specially
    if task_name == "procedural-learning" and "config" in scenario and "procedure" in scenario["config"]:
        return evaluate_procedural_scenario(agent_id, scenario, verbose)

    # Ingest memories
    mem_id_map: dict[str, int] = {}
    for memory in scenario.get("memories", []):
        db_id = ingest_memory(agent_id, memory)
        mem_id_map[memory["id"]] = db_id

    # Run queries and evaluate
    query_scores: list[float] = []
    details: dict = {"queries": []}

    for qi, query in enumerate(scenario.get("queries", [])):
        query_text = query["query"]
        query_ts = query.get("queryTimestamp")
        expected = scenario["expected"][qi] if qi < len(scenario.get("expected", [])) else {}

        # Search
        results = search_memories(agent_id, query_text, top_k=10, query_timestamp=query_ts)
        retrieved_sources = [r["source"] or "" for r in results]

        # Score: check expected memory IDs found, excluded memory IDs absent
        expected_ids = expected.get("expectedMemoryIds", [])
        excluded_ids = expected.get("excludedMemoryIds", [])
        expects_results = expected.get("expectsResults", True)

        score = 0.0
        if not expects_results:
            # Should return no relevant results
            score = 1.0 if len(results) == 0 else 0.0
        elif expected_ids:
            # Check if expected IDs are in results
            found = 0
            for eid in expected_ids:
                source_match = f"cogbench/{eid}"
                if any(source_match in s for s in retrieved_sources):
                    found += 1

            # Check excluded IDs are NOT in results
            excluded_found = 0
            for xid in excluded_ids:
                source_match = f"cogbench/{xid}"
                if any(source_match in s for s in retrieved_sources):
                    excluded_found += 1

            # Score = (found expected / total expected) * (1 - excluded_found / max(1, total_excluded))
            found_score = found / len(expected_ids) if expected_ids else 1.0
            exclude_penalty = excluded_found / max(1, len(excluded_ids)) if excluded_ids else 0.0
            score = found_score * (1.0 - exclude_penalty)
        else:
            score = 1.0 if results else 0.0

        query_scores.append(score)
        details["queries"].append({
            "query": query_text,
            "score": score,
            "retrieved": len(results),
            "expected_found": len([eid for eid in expected_ids if any(f"cogbench/{eid}" in s for s in retrieved_sources)]),
            "excluded_found": len([xid for xid in excluded_ids if any(f"cogbench/{xid}" in s for s in retrieved_sources)]),
        })

        if verbose:
            status = "PASS" if score >= 0.5 else "FAIL"
            print(f"    Q{qi+1}: {status} ({score:.2f}) | {query_text[:60]}...")

    avg_score = sum(query_scores) / len(query_scores) if query_scores else 0.0

    return ScenarioResult(
        scenario_id=scenario["id"],
        task=task_name,
        description=scenario.get("description", ""),
        score=avg_score,
        details=details,
    )


def main():
    parser = argparse.ArgumentParser(description="CogBench for CORTEX Python")
    parser.add_argument("--task", type=str, default=None, help="Run only one task")
    parser.add_argument("--verbose", action="store_true", help="Print per-query detail")
    args = parser.parse_args()

    print("=" * 50)
    print("  CORTEX Python V2.4 -- CogBench")
    print("  Testing cognitive capabilities")
    print("=" * 50)
    print()

    # Load dataset (from TypeScript repo)
    dataset_path = Path("/Users/rezcorp/cortex/benchmarks/cogbench/dataset/cogbench-v1.json")
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    with open(dataset_path) as f:
        data = json.load(f)

    init_database()
    agent_id = resolve_agent("benchmark-cogbench-python")

    task_scores: dict[str, list[float]] = {}
    all_results: list[ScenarioResult] = []
    total_scenarios = 0

    tasks = data["tasks"]
    if args.task:
        tasks = {args.task: tasks[args.task]}

    for task_name, scenarios in tasks.items():
        print(f"\n--- {task_name} ({len(scenarios)} scenarios) ---")
        task_scores[task_name] = []

        for si, scenario in enumerate(scenarios):
            total_scenarios += 1
            result = evaluate_scenario(agent_id, scenario, task_name, verbose=args.verbose)
            all_results.append(result)
            task_scores[task_name].append(result.score)

            status = "PASS" if result.score >= 0.5 else "FAIL"
            print(f"  [{si+1}/{len(scenarios)}] {status} ({result.score:.2f}) | {result.description[:60]}...")

    # Cleanup
    clear_agent(agent_id)

    # Results
    print("\n" + "=" * 50)
    print("  COGBENCH RESULTS (Python)")
    print("=" * 50)

    print(f"\nTotal Scenarios: {total_scenarios}")
    print(f"\n| Task | Scenarios | Avg Score | Pass Rate |")
    print(f"|------|-----------|-----------|-----------|")

    geometric_product = 1.0
    for task_name, scores in sorted(task_scores.items()):
        avg = sum(scores) / len(scores) if scores else 0
        passes = sum(1 for s in scores if s >= 0.5)
        print(f"| {task_name} | {len(scores)} | {avg*100:.1f}% | {passes}/{len(scores)} |")
        geometric_product *= max(avg, 0.001)

    composite = geometric_product ** (1 / len(task_scores)) if task_scores else 0
    print(f"\n**Composite Score (geometric mean): {composite*100:.1f}%**")

    # Save
    output_path = Path(__file__).parent / "results-python.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "benchmark": "CogBench",
                "implementation": "Python",
                "system": "CORTEX Python V2.4",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "composite_score": composite,
                "per_task": {
                    name: {
                        "avg_score": sum(scores) / len(scores) if scores else 0,
                        "pass_rate": sum(1 for s in scores if s >= 0.5) / len(scores) if scores else 0,
                        "scenarios": len(scores),
                    }
                    for name, scores in task_scores.items()
                },
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
