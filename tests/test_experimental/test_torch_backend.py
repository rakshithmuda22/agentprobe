"""
Tests for the PyTorch resonance scoring backend.

All tests are skipped if torch is not installed — the torch backend
is an optional acceleration path, not a hard dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip entire module if torch is not installed
torch = pytest.importorskip("torch", reason="PyTorch not installed")

from src.experimental.torch_backend import (  # noqa: E402
    TorchResonanceBackend,
    TorchProjector,
    detect_torch_device,
    numpy_to_torch,
    torch_to_numpy,
)
from src.experimental.resonance_core import (  # noqa: E402
    QJLProjector,
    ResonanceBackend,
    batch_resonance_scores,
    pairwise_resonance_matrix,
    detect_backend,
)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

class TestDeviceDetection:
    def test_detect_returns_string(self):
        device = detect_torch_device()
        assert device in ("cpu", "mps", "cuda")

    def test_detect_backend_returns_torch(self):
        backend = detect_backend()
        assert backend in (
            ResonanceBackend.TORCH_CUDA,
            ResonanceBackend.TORCH_MPS,
            ResonanceBackend.TORCH_CPU,
        )

    def test_new_enum_values_exist(self):
        assert ResonanceBackend.TORCH_MPS is not None
        assert ResonanceBackend.TORCH_CPU is not None


# ---------------------------------------------------------------------------
# Conversion utilities
# ---------------------------------------------------------------------------

class TestConversions:
    def test_numpy_to_torch_roundtrip(self):
        arr = np.random.default_rng(42).random((10, 5))
        t = numpy_to_torch(arr)
        back = torch_to_numpy(t)
        np.testing.assert_array_almost_equal(arr, back)

    def test_device_placement(self):
        arr = np.array([1.0, 2.0, 3.0])
        t = numpy_to_torch(arr, device="cpu")
        assert t.device.type == "cpu"


# ---------------------------------------------------------------------------
# TorchResonanceBackend
# ---------------------------------------------------------------------------

class TestTorchBackend:
    @pytest.fixture
    def backend(self):
        return TorchResonanceBackend(device="cpu")

    @pytest.fixture
    def projector(self):
        return QJLProjector(input_dim=32, projection_dim=64, seed=42)

    def test_init_device(self, backend):
        assert backend.device == "cpu"

    def test_batch_scores_identical_is_zero(self, backend):
        ref = torch.tensor([0, 1, 3, 2, 6, 7, 5, 4], dtype=torch.uint8)
        batch = ref.unsqueeze(0)  # (1, k)
        scores = backend.batch_resonance_scores(ref, batch, n_bits=3)
        assert scores.shape == (1,)
        assert scores[0].item() == pytest.approx(0.0)

    def test_batch_scores_range(self, backend):
        rng = np.random.default_rng(42)
        ref = numpy_to_torch(
            rng.integers(0, 8, size=64, dtype=np.uint8), device="cpu"
        )
        batch = numpy_to_torch(
            rng.integers(0, 8, size=(20, 64), dtype=np.uint8), device="cpu"
        )
        scores = backend.batch_resonance_scores(ref, batch, n_bits=3)
        assert scores.shape == (20,)
        assert (scores >= 0.0).all()
        assert (scores <= 1.0).all()

    def test_pairwise_diagonal_zero(self, backend):
        rng = np.random.default_rng(42)
        states = numpy_to_torch(
            rng.integers(0, 8, size=(10, 32), dtype=np.uint8), device="cpu"
        )
        matrix = backend.pairwise_resonance_matrix(states, n_bits=3)
        assert matrix.shape == (10, 10)
        diag = torch.diag(matrix)
        np.testing.assert_array_almost_equal(
            torch_to_numpy(diag), np.zeros(10)
        )

    def test_pairwise_symmetric(self, backend):
        rng = np.random.default_rng(42)
        states = numpy_to_torch(
            rng.integers(0, 8, size=(15, 32), dtype=np.uint8), device="cpu"
        )
        matrix = backend.pairwise_resonance_matrix(states, n_bits=3)
        np.testing.assert_array_almost_equal(
            torch_to_numpy(matrix), torch_to_numpy(matrix.T)
        )

    def test_pairwise_matches_numpy(self, projector, backend):
        """Torch pairwise matrix must match NumPy pairwise matrix exactly."""
        rng = np.random.default_rng(99)
        vectors = rng.standard_normal((20, 32))
        # NumPy path
        states_np = projector.batch_project_and_quantize(vectors)
        matrix_np = pairwise_resonance_matrix(states_np)
        # Torch path
        states_torch = numpy_to_torch(states_np, device="cpu")
        matrix_torch = torch_to_numpy(
            backend.pairwise_resonance_matrix(states_torch, n_bits=3)
        )
        np.testing.assert_array_almost_equal(matrix_np, matrix_torch, decimal=10)


# ---------------------------------------------------------------------------
# TorchProjector
# ---------------------------------------------------------------------------

class TestTorchProjector:
    @pytest.fixture
    def projector(self):
        return QJLProjector(input_dim=32, projection_dim=64, seed=42)

    @pytest.fixture
    def torch_proj(self, projector):
        return TorchProjector(projector, device="cpu")

    def test_properties(self, torch_proj):
        assert torch_proj.device == "cpu"
        assert torch_proj.input_dim == 32
        assert torch_proj.projection_dim == 64

    def test_project_and_quantize_shape(self, torch_proj):
        vec = torch.randn(32, dtype=torch.float64)
        result = torch_proj.project_and_quantize(vec)
        assert result.shape == (64,)
        assert result.dtype == torch.uint8

    def test_batch_project_shape(self, torch_proj):
        vecs = torch.randn(10, 32, dtype=torch.float64)
        result = torch_proj.project_and_quantize(vecs)
        assert result.shape == (10, 64)

    def test_numpy_input_accepted(self, torch_proj):
        vec = np.random.default_rng(42).standard_normal(32)
        result = torch_proj.project_and_quantize(vec)
        assert result.shape == (64,)

    def test_matches_numpy_projector(self, projector, torch_proj):
        """Torch projection must produce same output as NumPy projection."""
        rng = np.random.default_rng(42)
        vec = rng.standard_normal(32)
        # NumPy path
        np_result = projector.project_and_quantize(vec)
        # Torch path
        torch_result = torch_to_numpy(torch_proj.project_and_quantize(vec))
        np.testing.assert_array_equal(np_result, torch_result)

    def test_batch_matches_numpy(self, projector, torch_proj):
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((15, 32))
        np_result = projector.batch_project_and_quantize(vecs)
        torch_result = torch_to_numpy(torch_proj.project_and_quantize(vecs))
        np.testing.assert_array_equal(np_result, torch_result)

    def test_pairwise_via_projector(self, projector, torch_proj):
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((10, 32))
        # NumPy full path
        np_states = projector.batch_project_and_quantize(vecs)
        np_matrix = pairwise_resonance_matrix(np_states)
        # Torch full path
        torch_states = torch_proj.project_and_quantize(vecs)
        torch_matrix = torch_to_numpy(
            torch_proj.pairwise_resonance_matrix(torch_states)
        )
        np.testing.assert_array_almost_equal(np_matrix, torch_matrix, decimal=10)
