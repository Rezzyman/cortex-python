"""Tests for Dentate Gyrus pattern separation."""

import numpy as np
import pytest

from cortex_ai.hippocampus.dentate_gyrus import (
    dg_encode,
    sparse_overlap,
    sparse_jaccard,
    INPUT_DIM,
    EXPANDED_DIM,
    K,
    SPARSITY_RATIO,
)


def make_embedding(seed: int) -> list[float]:
    return [float(np.sin(seed * (i + 1) * 0.01) * 0.5) for i in range(INPUT_DIM)]


class TestDGEncode:
    def test_output_dimensions(self):
        code = dg_encode(make_embedding(42))
        assert code.dim == EXPANDED_DIM
        assert len(code.indices) == K
        assert len(code.values) == K

    def test_l2_normalized(self):
        code = dg_encode(make_embedding(42))
        norm = sum(v**2 for v in code.values)
        assert abs(norm - 1.0) < 0.01

    def test_deterministic(self):
        emb = make_embedding(42)
        c1 = dg_encode(emb)
        c2 = dg_encode(emb)
        assert c1.indices == c2.indices
        assert c1.values == c2.values

    def test_wrong_dimension_raises(self):
        with pytest.raises(ValueError, match="expected 1024"):
            dg_encode([1.0, 2.0, 3.0])

    def test_sparsity(self):
        code = dg_encode(make_embedding(42))
        ratio = len(code.indices) / code.dim
        assert abs(ratio - SPARSITY_RATIO) < 0.01

    def test_non_negative_values(self):
        code = dg_encode(make_embedding(42))
        assert all(v >= 0 for v in code.values)


class TestSparseOverlap:
    def test_self_overlap_is_maximum(self):
        code = dg_encode(make_embedding(42))
        self_overlap = sparse_overlap(code, code)
        other = dg_encode(make_embedding(999))
        assert sparse_overlap(code, other) < self_overlap

    def test_symmetric(self):
        c1 = dg_encode(make_embedding(1))
        c2 = dg_encode(make_embedding(2))
        assert abs(sparse_overlap(c1, c2) - sparse_overlap(c2, c1)) < 1e-6


class TestSparseJaccard:
    def test_identical_codes(self):
        code = dg_encode(make_embedding(42))
        assert sparse_jaccard(code, code) == 1.0

    def test_bounded_zero_one(self):
        c1 = dg_encode(make_embedding(1))
        c2 = dg_encode(make_embedding(2))
        j = sparse_jaccard(c1, c2)
        assert 0 <= j <= 1


class TestPatternSeparation:
    def test_different_inputs_different_codes(self):
        c1 = dg_encode(make_embedding(1))
        c2 = dg_encode(make_embedding(100))
        assert sparse_jaccard(c1, c2) < 0.5

    def test_identical_inputs_identical_codes(self):
        emb = make_embedding(42)
        c1 = dg_encode(emb)
        c2 = dg_encode(emb)
        assert sparse_jaccard(c1, c2) == 1.0
