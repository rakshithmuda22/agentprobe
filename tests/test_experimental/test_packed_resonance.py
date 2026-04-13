"""Tests for packed uint64 bitwise acceleration in resonance_core.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.experimental.resonance_core import (
    QJLProjector,
    ResonanceBackend,
    batch_resonance_scores,
    batch_resonance_scores_packed,
    detect_backend,
    pack_quantized,
    pack_quantized_batch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_quantized(k: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 8, size=k, dtype=np.uint8)


def _random_quantized_batch(n: int, k: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 8, size=(n, k), dtype=np.uint8)


# ---------------------------------------------------------------------------
# pack_quantized
# ---------------------------------------------------------------------------

class TestPackQuantized:
    def test_shape_k64(self):
        q = _random_quantized(64)
        packed = pack_quantized(q, n_bits=3)
        # 64 * 3 = 192 bits → 3 uint64 words
        assert packed.dtype == np.uint64
        assert packed.shape == (3,)

    def test_shape_k512(self):
        q = _random_quantized(512)
        packed = pack_quantized(q, n_bits=3)
        # 512 * 3 = 1536 bits → 24 uint64 words
        assert packed.shape == (24,)

    def test_shape_k128(self):
        q = _random_quantized(128)
        packed = pack_quantized(q, n_bits=3)
        # 128 * 3 = 384 bits → 6 uint64 words
        assert packed.shape == (6,)

    def test_identical_vectors_same_packed(self):
        q = _random_quantized(128)
        p1 = pack_quantized(q)
        p2 = pack_quantized(q)
        np.testing.assert_array_equal(p1, p2)

    def test_different_vectors_different_packed(self):
        q1 = _random_quantized(128, seed=0)
        q2 = _random_quantized(128, seed=1)
        # Almost certainly differ
        assert not np.array_equal(pack_quantized(q1), pack_quantized(q2))

    def test_output_dtype(self):
        q = _random_quantized(64)
        assert pack_quantized(q).dtype == np.uint64


# ---------------------------------------------------------------------------
# pack_quantized_batch
# ---------------------------------------------------------------------------

class TestPackQuantizedBatch:
    def test_shape_n10_k512(self):
        batch = _random_quantized_batch(10, 512)
        packed = pack_quantized_batch(batch)
        assert packed.shape == (10, 24)
        assert packed.dtype == np.uint64

    def test_shape_n1_matches_single(self):
        q = _random_quantized(128)
        single = pack_quantized(q)
        batch = pack_quantized_batch(q[np.newaxis, :])
        np.testing.assert_array_equal(single, batch[0])

    def test_each_row_matches_individual(self):
        rng = np.random.default_rng(42)
        batch = rng.integers(0, 8, size=(5, 64), dtype=np.uint8)
        packed_batch = pack_quantized_batch(batch)
        for i in range(5):
            packed_single = pack_quantized(batch[i])
            np.testing.assert_array_equal(packed_batch[i], packed_single)


# ---------------------------------------------------------------------------
# batch_resonance_scores_packed vs uint8 path
# ---------------------------------------------------------------------------

class TestPackedResonanceScores:
    """Packed scores must match uint8 scores exactly (same math, different layout)."""

    def _run_comparison(self, k: int, n: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        ref = rng.integers(0, 8, size=k, dtype=np.uint8)
        batch = rng.integers(0, 8, size=(n, k), dtype=np.uint8)

        scores_u8 = batch_resonance_scores(ref, batch, n_bits=3)

        ref_packed = pack_quantized(ref)
        batch_packed = pack_quantized_batch(batch)
        scores_packed = batch_resonance_scores_packed(
            ref_packed, batch_packed, n_dims=k, n_bits=3
        )

        np.testing.assert_allclose(scores_packed, scores_u8, atol=1e-10)

    def test_k128_n50(self):
        self._run_comparison(k=128, n=50)

    def test_k512_n100(self):
        self._run_comparison(k=512, n=100)

    def test_k64_n10(self):
        self._run_comparison(k=64, n=10)

    def test_identical_ref_and_batch_zero(self):
        q = _random_quantized(128)
        ref_p = pack_quantized(q)
        batch_p = pack_quantized_batch(q[np.newaxis, :])
        scores = batch_resonance_scores_packed(ref_p, batch_p, n_dims=128)
        assert scores[0] == pytest.approx(0.0)

    def test_scores_in_unit_range(self):
        k, n = 128, 50
        rng = np.random.default_rng(7)
        ref = rng.integers(0, 8, size=k, dtype=np.uint8)
        batch = rng.integers(0, 8, size=(n, k), dtype=np.uint8)
        ref_p = pack_quantized(ref)
        batch_p = pack_quantized_batch(batch)
        scores = batch_resonance_scores_packed(ref_p, batch_p, n_dims=k)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_real_projections_match(self):
        """Verify with real QJL-projected vectors, not random uint8."""
        projector = QJLProjector(input_dim=64, projection_dim=128, seed=42)
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((20, 64))

        quantized = projector.batch_project_and_quantize(vectors)
        ref = quantized[0]
        batch = quantized[1:]

        scores_u8 = batch_resonance_scores(ref, batch)
        ref_p = pack_quantized(ref)
        batch_p = pack_quantized_batch(batch)
        scores_packed = batch_resonance_scores_packed(ref_p, batch_p, n_dims=128)

        np.testing.assert_allclose(scores_packed, scores_u8, atol=1e-10)


# ---------------------------------------------------------------------------
# detect_backend
# ---------------------------------------------------------------------------

class TestDetectBackend:
    def test_returns_valid_enum(self):
        backend = detect_backend()
        assert isinstance(backend, ResonanceBackend)

    def test_numpy_packed_available(self):
        # numpy >= 1.26 is installed; packed backend should be returned
        # unless a torch backend is available (CUDA, MPS, or CPU)
        backend = detect_backend()
        assert backend in (
            ResonanceBackend.NUMPY_PACKED,
            ResonanceBackend.TORCH_CUDA,
            ResonanceBackend.TORCH_MPS,
            ResonanceBackend.TORCH_CPU,
        )
