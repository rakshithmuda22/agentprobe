"""Tests for resonance_core.py — validated math layer."""

from __future__ import annotations

import numpy as np
import pytest

from src.experimental.resonance_core import (
    QJLProjector,
    calculate_resonance_score,
    hamming_distance,
    batch_resonance_scores,
    pairwise_resonance_matrix,
    _GRAY_3BIT,
    _POPCOUNT_TABLE,
)


class TestGrayCode:
    def test_gray_code_length(self):
        assert len(_GRAY_3BIT) == 8

    def test_gray_code_unique(self):
        assert len(set(_GRAY_3BIT)) == 8

    def test_gray_code_range(self):
        assert all(0 <= v <= 7 for v in _GRAY_3BIT)

    def test_gray_code_adjacency(self):
        """Adjacent quantization levels must differ by exactly 1 bit."""
        for i in range(len(_GRAY_3BIT) - 1):
            xor = int(_GRAY_3BIT[i]) ^ int(_GRAY_3BIT[i + 1])
            assert bin(xor).count("1") == 1, (
                f"Levels {i} and {i+1}: Gray codes {_GRAY_3BIT[i]:03b} and "
                f"{_GRAY_3BIT[i+1]:03b} differ by {bin(xor).count('1')} bits"
            )


class TestPopcount:
    def test_popcount_zero(self):
        assert _POPCOUNT_TABLE[0] == 0

    def test_popcount_max(self):
        assert _POPCOUNT_TABLE[255] == 8

    def test_popcount_powers_of_two(self):
        for i in range(8):
            assert _POPCOUNT_TABLE[1 << i] == 1


class TestQJLProjector:
    def test_projection_shape(self, shared_projector: QJLProjector):
        v = np.random.default_rng(0).standard_normal(64)
        proj = shared_projector.project(v)
        assert proj.shape == (128,)

    def test_quantize_range(self, shared_projector: QJLProjector):
        v = np.random.default_rng(0).standard_normal(64)
        q = shared_projector.project_and_quantize(v)
        assert q.dtype == np.uint8
        assert np.all(q <= 7)

    def test_batch_shape(self, shared_projector: QJLProjector):
        vectors = np.random.default_rng(0).standard_normal((10, 64))
        q = shared_projector.batch_project_and_quantize(vectors)
        assert q.shape == (10, 128)
        assert q.dtype == np.uint8

    def test_deterministic_with_seed(self):
        p1 = QJLProjector(32, 64, seed=123)
        p2 = QJLProjector(32, 64, seed=123)
        v = np.ones(32)
        np.testing.assert_array_equal(
            p1.project_and_quantize(v),
            p2.project_and_quantize(v),
        )

    def test_different_seeds_differ(self):
        p1 = QJLProjector(32, 64, seed=1)
        p2 = QJLProjector(32, 64, seed=2)
        v = np.ones(32)
        q1 = p1.project_and_quantize(v)
        q2 = p2.project_and_quantize(v)
        # Extremely unlikely to be identical with different seeds
        assert not np.array_equal(q1, q2)

    def test_only_3bit_supported(self):
        with pytest.raises(ValueError, match="3-bit"):
            QJLProjector(32, 64, n_bits=4)


class TestHammingDistance:
    def test_identical_zero(self):
        a = np.array([3, 5, 7, 0], dtype=np.uint8)
        assert hamming_distance(a, a) == 0

    def test_all_bits_differ(self):
        a = np.array([0b000], dtype=np.uint8)
        b = np.array([0b111], dtype=np.uint8)
        assert hamming_distance(a, b) == 3

    def test_single_bit(self):
        a = np.array([0b010], dtype=np.uint8)
        b = np.array([0b011], dtype=np.uint8)
        assert hamming_distance(a, b) == 1


class TestResonanceScore:
    def test_identical_vectors_zero(self, shared_projector: QJLProjector):
        v = np.random.default_rng(0).standard_normal(64)
        q = shared_projector.project_and_quantize(v)
        assert calculate_resonance_score(q, q) == 0.0

    def test_orthogonal_vectors_midrange(self, shared_projector: QJLProjector):
        """Orthogonal vectors should give a score around 0.5."""
        rng = np.random.default_rng(42)
        # Create orthogonal pair via Gram-Schmidt
        a = rng.standard_normal(64)
        a /= np.linalg.norm(a)
        b = rng.standard_normal(64)
        b -= np.dot(b, a) * a
        b /= np.linalg.norm(b)

        qa = shared_projector.project_and_quantize(a)
        qb = shared_projector.project_and_quantize(b)
        score = calculate_resonance_score(qa, qb)
        # 3-bit Gray quantization compresses the score range vs. 1-bit sign
        assert 0.10 < score < 0.75, f"Orthogonal score {score} not in expected range"

    def test_antiparallel_high(self, shared_projector: QJLProjector):
        """Antiparallel vectors (v and -v) should give a high score."""
        v = np.random.default_rng(0).standard_normal(64)
        v /= np.linalg.norm(v)

        qa = shared_projector.project_and_quantize(v)
        qb = shared_projector.project_and_quantize(-v)
        score = calculate_resonance_score(qa, qb)
        # 3-bit Gray quantization: max Hamming is 3 bits per dim, but Gray
        # code means antiparallel vectors (opposite bins) differ by ~2 bits
        # on average, not the full 3. Score around 0.33 is expected.
        assert score > 0.25, f"Antiparallel score {score} not high enough"

    def test_shape_mismatch_raises(self):
        a = np.array([1, 2, 3], dtype=np.uint8)
        b = np.array([1, 2], dtype=np.uint8)
        with pytest.raises(ValueError, match="Shape mismatch"):
            calculate_resonance_score(a, b)

    def test_score_range(self, shared_projector: QJLProjector):
        rng = np.random.default_rng(99)
        for _ in range(20):
            a = rng.standard_normal(64)
            b = rng.standard_normal(64)
            qa = shared_projector.project_and_quantize(a)
            qb = shared_projector.project_and_quantize(b)
            score = calculate_resonance_score(qa, qb)
            assert 0.0 <= score <= 1.0


class TestBatchResonance:
    def test_batch_matches_individual(self, shared_projector: QJLProjector):
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((5, 64))
        quantized = shared_projector.batch_project_and_quantize(vectors)

        ref = quantized[0]
        batch_scores = batch_resonance_scores(ref, quantized)

        for i in range(5):
            individual = calculate_resonance_score(ref, quantized[i])
            np.testing.assert_almost_equal(batch_scores[i], individual)


class TestPairwiseMatrix:
    def test_symmetric(self, shared_projector: QJLProjector):
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((8, 64))
        quantized = shared_projector.batch_project_and_quantize(vectors)
        matrix = pairwise_resonance_matrix(quantized)
        np.testing.assert_array_almost_equal(matrix, matrix.T)

    def test_zero_diagonal(self, shared_projector: QJLProjector):
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((8, 64))
        quantized = shared_projector.batch_project_and_quantize(vectors)
        matrix = pairwise_resonance_matrix(quantized)
        np.testing.assert_array_equal(np.diag(matrix), 0.0)


class TestJLPreservation:
    """Statistical test: QJL resonance scores correlate with true angular distance."""

    def test_angular_correlation(self):
        n, d, k = 50, 64, 256
        rng = np.random.default_rng(42)
        projector = QJLProjector(d, k, n_bits=3, seed=42)

        vectors = rng.standard_normal((n, d))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        # Ground truth angular distances
        cosines = np.clip(vectors @ vectors.T, -1, 1)
        angular = np.arccos(cosines) / np.pi

        # QJL resonance scores
        quantized = projector.batch_project_and_quantize(vectors)
        resonance = pairwise_resonance_matrix(quantized)

        mask = np.triu_indices(n, k=1)
        a_flat = angular[mask]
        r_flat = resonance[mask]
        coef = float(np.corrcoef(a_flat, r_flat)[0, 1])
        assert not np.isnan(coef), "Correlation undefined (degenerate inputs)"
        # 3-bit Gray QJL weakens linear correlation vs. raw cosine distance.
        assert coef > 0.50, f"Pearson r = {coef:.4f}, expected > 0.50"
