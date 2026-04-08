"""
Dentate Gyrus — Pattern Separation via Sparse Coding

Implements the computational equivalent of DG granule cell expansion:

  1. Random projection: 1024-dim dense -> 4096-dim expanded (4x expansion
     ratio, matching biological ~5x DG granule cell expansion from EC).
  2. ReLU nonlinearity: enforce non-negative activations.
  3. k-Winners-Take-All: keep only top 5% of activations (~204 out of 4096).
  4. L2-normalize: so dot product = cosine similarity for downstream use.

References:
  - Rolls (2013) "The mechanisms for pattern completion and pattern separation"
  - Knierim & Neunzig (2016) "Tracking the flow of hippocampal computation"
"""

from dataclasses import dataclass
from math import sqrt, log, cos, sin, pi

import numpy as np

INPUT_DIM = 1024
EXPANDED_DIM = 4096
SPARSITY_RATIO = 0.05
K = int(SPARSITY_RATIO * EXPANDED_DIM)  # 204 active neurons
DG_SEED = 314159265

_projection_matrix: np.ndarray | None = None


@dataclass
class SparseCode:
    """Sparse hippocampal representation."""
    indices: list[int]
    values: list[float]
    dim: int = EXPANDED_DIM


def _mulberry32(seed: int):
    """Seeded PRNG matching the TypeScript implementation."""
    s = seed & 0xFFFFFFFF
    while True:
        s = (s + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((s ^ (s >> 15)) * (1 | s)) & 0xFFFFFFFF
        t = (t + ((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF) ^ t
        yield ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296


def _generate_projection_matrix() -> np.ndarray:
    """Generate deterministic projection matrix W (INPUT_DIM x EXPANDED_DIM)."""
    rng = _mulberry32(DG_SEED)
    matrix = np.zeros((INPUT_DIM, EXPANDED_DIM), dtype=np.float32)
    scale = 1.0 / sqrt(INPUT_DIM)

    # Box-Muller transform for Gaussian
    flat = matrix.ravel()
    for i in range(0, len(flat), 2):
        u1 = next(rng)
        u2 = next(rng)
        r = sqrt(-2.0 * log(max(u1, 1e-10)))
        theta = 2.0 * pi * u2
        flat[i] = r * cos(theta) * scale
        if i + 1 < len(flat):
            flat[i + 1] = r * sin(theta) * scale

    return matrix


def _get_projection_matrix() -> np.ndarray:
    """Lazy singleton for the projection matrix."""
    global _projection_matrix
    if _projection_matrix is None:
        _projection_matrix = _generate_projection_matrix()
    return _projection_matrix


def dg_encode(dense_embedding: list[float] | np.ndarray) -> SparseCode:
    """
    Encode a 1024-dim dense embedding into a sparse 4096-dim DG representation.

    Args:
        dense_embedding: 1024-dimensional embedding vector

    Returns:
        SparseCode with ~204 active indices and L2-normalized values
    """
    emb = np.array(dense_embedding, dtype=np.float32)
    if emb.shape[0] != INPUT_DIM:
        raise ValueError(f"DG encode: expected {INPUT_DIM}-dim input, got {emb.shape[0]}")

    W = _get_projection_matrix()

    # Step 1: Random projection z = W^T * x
    z = emb @ W

    # Step 2: ReLU
    z = np.maximum(z, 0)

    # Step 3: k-Winners-Take-All
    top_k_idx = np.argpartition(z, -K)[-K:]
    top_k_idx = top_k_idx[np.argsort(-z[top_k_idx])]

    # Step 4: L2-normalize
    values = z[top_k_idx]
    norm = np.linalg.norm(values)
    if norm < 1e-10:
        norm = 1e-10
    values = values / norm

    return SparseCode(
        indices=top_k_idx.tolist(),
        values=values.tolist(),
        dim=EXPANDED_DIM,
    )


def sparse_overlap(a: SparseCode, b: SparseCode) -> float:
    """
    Sparse overlap between two DG codes.
    Sum of min(a[i], b[i]) for shared active indices.
    """
    smaller, larger = (a, b) if len(a.indices) <= len(b.indices) else (b, a)
    index_map = dict(zip(smaller.indices, smaller.values))

    overlap = 0.0
    for i, idx in enumerate(larger.indices):
        val = index_map.get(idx)
        if val is not None:
            overlap += min(val, larger.values[i])

    return overlap


def sparse_jaccard(a: SparseCode, b: SparseCode) -> float:
    """Jaccard index of active neuron sets."""
    set_a = set(a.indices)
    set_b = set(b.indices)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0
