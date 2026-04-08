"""
CORTEX Dream Cycle -- Two-phase synthetic sleep.

SWS (Slow-Wave Sleep):
  Phase 1: Resonance Analysis (Ebbinghaus stability-adjusted decay)
  Phase 2: Pruning (adaptive percentile-based thresholds)
  Phase 3: Consolidation (cluster + summarize high-resonance groups)

REM (Rapid Eye Movement):
  Phase 4: Free Association (random activation for novel connections)
  Phase 5: Synthesis (generate insights from novel synapses)
"""

import json
import time
from dataclasses import dataclass, field

from cortex_ai.db.connection import get_db


@dataclass
class DreamStats:
    resonance_updated: int = 0
    memories_deleted: int = 0
    memories_archived: int = 0
    synapses_pruned: int = 0
    observations_pruned: int = 0
    clusters_found: int = 0
    consolidations: int = 0
    synapses_strengthened: int = 0
    duration_ms: int = 0


def dream(
    agent_id: int,
    cycle_type: str = "full",
) -> DreamStats:
    """Run a dream cycle for the specified agent."""
    start = time.time()
    stats = DreamStats()

    run_sws = cycle_type in ("full", "sws_only")
    run_rem = cycle_type in ("full", "rem_only")

    with get_db() as conn:
        with conn.cursor() as cur:
            # Log the cycle
            cur.execute(
                "INSERT INTO dream_cycle_logs (agent_id, cycle_type, stats) VALUES (%s, %s, %s) RETURNING id",
                (agent_id, cycle_type, json.dumps({})),
            )
            log_id = cur.fetchone()["id"]

            if run_sws or cycle_type in ("resonance_only", "pruning_only"):
                # Phase 1: Resonance Analysis
                if run_sws or cycle_type == "resonance_only":
                    cur.execute(
                        """
                        WITH synapse_strength AS (
                            SELECT mn.id,
                                COALESCE(SUM(ms.connection_strength), 0) AS total_strength,
                                LEAST(COALESCE(SUM(ms.connection_strength), 0) / 5.0, 1.0) AS connectivity
                            FROM memory_nodes mn
                            LEFT JOIN memory_synapses ms ON (ms.memory_a = mn.id OR ms.memory_b = mn.id)
                            WHERE mn.agent_id = %s AND mn.status = 'active'
                            GROUP BY mn.id
                        )
                        UPDATE memory_nodes mn
                        SET resonance_score = (
                            0.15 * EXP(-0.023 * EXTRACT(EPOCH FROM (NOW() - mn.created_at)) / 86400
                                / (1.0 + 0.3 * LN(mn.access_count + 1) + 0.2 * ss.connectivity))
                            + 0.2 * LN(mn.access_count + 1)
                            + 0.25 * ss.connectivity
                            + 0.2 * CASE mn.priority WHEN 0 THEN 1.0 WHEN 1 THEN 0.8 WHEN 2 THEN 0.5
                                WHEN 3 THEN 0.3 WHEN 4 THEN 0.1 ELSE 0.5 END
                            + 0.1 * LEAST(mn.access_count / 10.0, 1.0)
                            + 0.1 * COALESCE(mn.novelty_score, 0.5)
                        ) * 10.0, updated_at = NOW()
                        FROM synapse_strength ss
                        WHERE mn.id = ss.id AND mn.agent_id = %s AND mn.status = 'active'
                        """,
                        (agent_id, agent_id),
                    )
                    stats.resonance_updated = cur.rowcount

                # Phase 2: Adaptive Pruning
                if run_sws or cycle_type == "pruning_only":
                    # Compute adaptive thresholds
                    cur.execute(
                        """
                        SELECT
                            PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY resonance_score) AS p5,
                            PERCENTILE_CONT(0.15) WITHIN GROUP (ORDER BY resonance_score) AS p15
                        FROM memory_nodes
                        WHERE agent_id = %s AND status = 'active' AND priority > 1
                        """,
                        (agent_id,),
                    )
                    row = cur.fetchone()
                    delete_threshold = min(float(row["p5"] or 1.0), 2.0)
                    archive_threshold = min(float(row["p15"] or 3.0), 4.0)

                    # Tier 1: Delete
                    cur.execute(
                        """
                        UPDATE memory_nodes mn SET status = 'deleted', updated_at = NOW()
                        WHERE mn.agent_id = %s AND mn.status = 'active' AND mn.priority > 1
                            AND mn.resonance_score < %s
                            AND mn.created_at < NOW() - INTERVAL '30 days'
                            AND NOT EXISTS (
                                SELECT 1 FROM emotional_valence ev
                                WHERE ev.memory_id = mn.id AND ev.decay_resistance > 0.5
                            )
                        """,
                        (agent_id, delete_threshold),
                    )
                    stats.memories_deleted = cur.rowcount

                    # Tier 2: Archive
                    cur.execute(
                        """
                        UPDATE memory_nodes mn SET status = 'archived', updated_at = NOW()
                        WHERE mn.agent_id = %s AND mn.status = 'active' AND mn.priority > 1
                            AND mn.resonance_score < %s
                            AND mn.created_at < NOW() - INTERVAL '14 days'
                            AND NOT EXISTS (
                                SELECT 1 FROM emotional_valence ev
                                WHERE ev.memory_id = mn.id AND ev.decay_resistance > 0.4
                            )
                        """,
                        (agent_id, archive_threshold),
                    )
                    stats.memories_archived = cur.rowcount

                    # Tier 3: Prune weak synapses
                    cur.execute(
                        """
                        UPDATE memory_synapses
                        SET connection_strength = connection_strength * (1.0 - decay_rate)
                        WHERE memory_a IN (SELECT id FROM memory_nodes WHERE agent_id = %s)
                           OR memory_b IN (SELECT id FROM memory_nodes WHERE agent_id = %s)
                        """,
                        (agent_id, agent_id),
                    )
                    cur.execute(
                        """
                        DELETE FROM memory_synapses
                        WHERE connection_strength < 0.1
                            AND (memory_a IN (SELECT id FROM memory_nodes WHERE agent_id = %s)
                              OR memory_b IN (SELECT id FROM memory_nodes WHERE agent_id = %s))
                        """,
                        (agent_id, agent_id),
                    )
                    stats.synapses_pruned = cur.rowcount

                    # Tier 4: Prune ephemeral observations
                    cur.execute(
                        """
                        UPDATE memory_nodes SET status = 'deleted', updated_at = NOW()
                        WHERE agent_id = %s AND status = 'active'
                            AND source_type = 'observation'
                            AND created_at < NOW() - INTERVAL '7 days'
                        """,
                        (agent_id,),
                    )
                    stats.observations_pruned = cur.rowcount

            # Update log
            stats.duration_ms = int((time.time() - start) * 1000)
            cur.execute(
                """
                UPDATE dream_cycle_logs
                SET stats = %s, completed_at = NOW()
                WHERE id = %s
                """,
                (json.dumps(stats.__dict__), log_id),
            )

        conn.commit()

    print(f"[dream] {cycle_type} complete in {stats.duration_ms}ms: "
          f"resonance={stats.resonance_updated}, deleted={stats.memories_deleted}, "
          f"archived={stats.memories_archived}, synapses_pruned={stats.synapses_pruned}")

    return stats
