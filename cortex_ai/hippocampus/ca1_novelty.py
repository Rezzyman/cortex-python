"""
CA1 -- Novelty Detection / Predictive Coding Comparator

Compares incoming memory against the network's prediction to detect
genuine novelty. Uses both dense-space and sparse-space mismatch.

Includes sparse gating: when DG recognizes a pattern (high sparse overlap)
despite moderate dense distance, the novelty signal is suppressed.
"""

from dataclasses import dataclass

import numpy as np

from cortex_ai.db.connection import get_db
from cortex_ai.hippocampus.dentate_gyrus import dg_encode, sparse_overlap, SparseCode

BASE_RESONANCE = 5.0
DENSE_WEIGHT = 0.6
SPARSE_WEIGHT = 0.4
NOVEL_HIGH = 0.7
NOVEL_LOW = 0.3


@dataclass
class NoveltyResult:
    novelty_score: float
    resonance_score: float
    adjusted_priority: int
    predicted_similarity: float
    sparse_mismatch: float


def compute_novelty(
    agent_id: int,
    dense_embedding: list[float],
    sparse_code: SparseCode,
    base_priority: int,
) -> NoveltyResult:
    """Compute novelty of an incoming memory against the existing network."""
    emb_str = "[" + ",".join(str(v) for v in dense_embedding) + "]"

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, embedding, 1 - (embedding <=> %s::vector) AS similarity
                FROM memory_nodes
                WHERE agent_id = %s AND status = 'active' AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector ASC
                LIMIT 5
                """,
                (emb_str, agent_id, emb_str),
            )
            neighbors = cur.fetchall()

    if not neighbors:
        return NoveltyResult(
            novelty_score=0.8,
            resonance_score=BASE_RESONANCE * 1.3,
            adjusted_priority=min(base_priority, 1),
            predicted_similarity=0,
            sparse_mismatch=1.0,
        )

    # Dense-space prediction: weighted centroid of neighbors
    total_sim = sum(max(float(n["similarity"]), 0) for n in neighbors)
    predicted = np.zeros(len(dense_embedding))

    for neighbor in neighbors:
        sim = max(float(neighbor["similarity"]), 0)
        weight = sim / total_sim if total_sim > 0 else 1.0 / len(neighbors)
        emb = neighbor["embedding"]
        if isinstance(emb, str):
            parsed = [float(x) for x in emb.strip("[]").split(",")]
        else:
            parsed = list(emb)
        predicted += np.array(parsed) * weight

    # Dense mismatch
    a = np.array(dense_embedding)
    b = predicted
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    dense_mismatch = 1 - max(cos_sim, 0)

    # Sparse mismatch
    predicted_sparse = dg_encode(predicted.tolist())
    sparse_match = sparse_overlap(sparse_code, predicted_sparse)
    sparse_mismatch = 1 - sparse_match

    # Combined novelty with sparse gating
    novelty_score = DENSE_WEIGHT * dense_mismatch + SPARSE_WEIGHT * sparse_mismatch

    # Sparse gating: suppress novelty when DG recognizes the pattern
    if sparse_match > 0.5 and 0.3 < dense_mismatch < 0.7:
        damping = 0.4 + 0.6 * (1 - sparse_match)
        novelty_score *= damping

    # Contradiction boost: genuinely novel/contradictory content
    if dense_mismatch > 0.7 and sparse_match < 0.1:
        novelty_score = min(novelty_score * 1.3, 1.0)

    # Modulate resonance and priority
    resonance_score = BASE_RESONANCE
    adjusted_priority = base_priority

    if novelty_score > NOVEL_HIGH:
        resonance_score = BASE_RESONANCE * 1.6
        adjusted_priority = max(0, base_priority - 1)
    elif novelty_score <= NOVEL_LOW:
        resonance_score = BASE_RESONANCE * 0.6

    return NoveltyResult(
        novelty_score=novelty_score,
        resonance_score=resonance_score,
        adjusted_priority=adjusted_priority,
        predicted_similarity=float(cos_sim),
        sparse_mismatch=sparse_mismatch,
    )
