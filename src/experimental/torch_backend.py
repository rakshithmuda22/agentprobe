"""
PyTorch Backend for QJL Resonance Scoring
=========================================

Provides a PyTorch-based resonance scoring engine that works across
multiple devices: CPU, MPS (Apple Silicon), and CUDA (NVIDIA GPU).

VALIDATED: All operations are standard linear algebra and bitwise ops.
No speculative extensions. The math is identical to the NumPy path in
resonance_core.py; the only difference is execution on a different backend.

Honest speedup expectations:
  - PyTorch CPU vs NumPy packed: ~1-3x. Main gain is from fully vectorized
    pairwise_resonance_matrix (eliminates the Python for-loop in the
    NumPy version). For single-pair scoring, PyTorch overhead may dominate.
  - MPS (Apple Silicon) vs CPU: ~2-5x on projection matmul for N > 1000.
    Uncertain for XOR+popcount — bitwise ops are not Metal's strength.
  - CUDA (NVIDIA): 10-30x plausible. 50-100x requires a custom Triton
    fused kernel which is NOT implemented here (Triton requires NVIDIA GPU).
  - Feasible agent count on M2: ~5,000-10,000 for pairwise matrix
    (N² memory bound). Not 100K without approximate nearest-neighbor.

References:
  - torch.bitwise_xor: standard PyTorch op
  - torch.bitwise_count: added in PyTorch 2.1 (maps to hardware POPCNT)
  - torch.searchsorted: equivalent to np.searchsorted
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import torch

# Lazy import flag — avoids hard dependency on torch
_HAS_TORCH = None


def _check_torch() -> bool:
    """Lazily check if torch is importable."""
    global _HAS_TORCH
    if _HAS_TORCH is None:
        try:
            import torch  # noqa: F401
            _HAS_TORCH = True
        except ImportError:
            _HAS_TORCH = False
    return _HAS_TORCH


def detect_torch_device() -> str:
    """Auto-detect the best available PyTorch device.

    Priority: cuda > mps > cpu.

    Returns:
        Device string suitable for torch.device().

    Raises:
        ImportError: if torch is not installed.
    """
    if not _check_torch():
        raise ImportError("PyTorch is required for the torch backend")

    import torch as _torch

    if _torch.cuda.is_available():
        return "cuda"
    if hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def numpy_to_torch(
    arr: NDArray, device: str = "cpu", dtype: "torch.dtype | None" = None
) -> "torch.Tensor":
    """Convert a NumPy array to a torch tensor on the specified device."""
    import torch as _torch

    t = _torch.from_numpy(np.ascontiguousarray(arr))
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t.to(device)


def torch_to_numpy(tensor: "torch.Tensor") -> NDArray:
    """Convert a torch tensor to a NumPy array (always on CPU)."""
    return tensor.detach().cpu().numpy()


# Precomputed popcount table for byte-level fallback (same as resonance_core)
_POPCOUNT_TABLE_NP = np.array(
    [bin(i).count("1") for i in range(256)], dtype=np.int32
)


class TorchResonanceBackend:
    """PyTorch-based resonance scoring engine.

    Provides the same scoring semantics as the NumPy path in
    resonance_core.py, but executes on a PyTorch device (CPU, MPS, CUDA).

    The key advantage over NumPy is the fully vectorized pairwise_resonance_matrix
    which uses broadcasting to eliminate the Python for-loop:
        states.unsqueeze(0) ^ states.unsqueeze(1)  → (N, N, k) XOR in one op
    """

    def __init__(self, device: str | None = None) -> None:
        """Initialize backend on the specified device.

        Args:
            device: "cpu", "mps", "cuda", or None for auto-detect.
        """
        if not _check_torch():
            raise ImportError(
                "PyTorch is required for TorchResonanceBackend. "
                "Install with: pip install 'torch>=2.1.0'"
            )
        import torch as _torch

        self._torch = _torch
        self._device = device or detect_torch_device()
        # MPS doesn't support float64; fall back to float32
        self._dtype_float = (
            _torch.float32 if self._device == "mps" else _torch.float64
        )
        self._dtype_uint8 = _torch.uint8

        # Check for native bitwise_count (PyTorch 2.1+)
        self._has_bitwise_count = hasattr(_torch, "bitwise_count")

        # Popcount table on device for fallback path
        self._popcount_table = numpy_to_torch(
            _POPCOUNT_TABLE_NP, device=self._device, dtype=_torch.int32
        )

    @property
    def device(self) -> str:
        """The device this backend operates on."""
        return self._device

    @property
    def has_native_popcount(self) -> bool:
        """Whether torch.bitwise_count is available (PyTorch 2.1+)."""
        return self._has_bitwise_count

    def project(
        self, R: "torch.Tensor", vectors: "torch.Tensor"
    ) -> "torch.Tensor":
        """Apply random Gaussian projection: R @ x.

        Args:
            R: projection matrix, shape (k, d) on device.
            vectors: shape (d,) or (n, d) on device.

        Returns:
            Projected values, shape (k,) or (n, k).
        """
        if vectors.ndim == 1:
            return R @ vectors
        return vectors @ R.T

    def quantize(
        self,
        projected: "torch.Tensor",
        boundaries: "torch.Tensor",
        gray: "torch.Tensor",
    ) -> "torch.Tensor":
        """Map projected float values to 3-bit Gray-coded integers.

        Args:
            projected: shape (k,) or (n, k) float tensor.
            boundaries: shape (7,) quantile boundaries.
            gray: shape (8,) Gray code lookup table.

        Returns:
            uint8 tensor of Gray-coded values.
        """
        _torch = self._torch
        bins = _torch.searchsorted(boundaries, projected.contiguous())
        return gray[bins.long()]

    def _popcount_uint8(self, xor_result: "torch.Tensor") -> "torch.Tensor":
        """Count set bits in a uint8 tensor via lookup table.

        Works on all devices. Uses torch.bitwise_count if available (faster),
        otherwise falls back to a 256-entry lookup table.

        Args:
            xor_result: uint8 tensor of any shape.

        Returns:
            int32 tensor of popcount values, same shape.
        """
        _torch = self._torch
        if self._has_bitwise_count:
            # Native hardware popcount (single PTX instruction on CUDA)
            return _torch.bitwise_count(xor_result).to(_torch.int32)
        # Fallback: byte-level lookup table
        return self._popcount_table[xor_result.long()]

    def batch_resonance_scores(
        self,
        ref: "torch.Tensor",
        batch: "torch.Tensor",
        n_bits: int = 3,
    ) -> "torch.Tensor":
        """Compute resonance scores between a reference and a batch of states.

        Args:
            ref: shape (k,) uint8 quantized state.
            batch: shape (n, k) uint8 quantized states.
            n_bits: bits per quantized value (default 3).

        Returns:
            float64 tensor shape (n,) of scores in [0, 1].
        """
        _torch = self._torch
        # XOR: (n, k) ^ (k,) → (n, k)
        xor = _torch.bitwise_xor(batch, ref)
        # Popcount per element
        popcounts = self._popcount_uint8(xor)
        # Sum across projection dimension
        total_bits = popcounts.sum(dim=1).to(self._dtype_float)
        max_hd = batch.shape[1] * n_bits
        return total_bits / max_hd

    def pairwise_resonance_matrix(
        self,
        states: "torch.Tensor",
        n_bits: int = 3,
    ) -> "torch.Tensor":
        """Compute full N×N pairwise resonance matrix.

        This is the key advantage over the NumPy path: fully vectorized
        via broadcasting, no Python for-loop.

        states.unsqueeze(0) ^ states.unsqueeze(1) produces (N, N, k)
        XOR results in a single op. For N=1000, k=128, this is ~128 MB
        of uint8 — feasible on most devices.

        Memory limit: N² × k bytes. For k=128:
          N=1000 → 128 MB, N=5000 → 3.2 GB, N=10000 → 12.8 GB.

        Args:
            states: shape (N, k) uint8 quantized states.
            n_bits: bits per quantized value.

        Returns:
            float64 tensor shape (N, N) of scores in [0, 1].
            Diagonal is 0.0 (self-resonance).
        """
        _torch = self._torch
        n = states.shape[0]
        k = states.shape[1]

        # Memory check: warn if > 2 GB intermediate
        mem_bytes = n * n * k
        if mem_bytes > 2_000_000_000:
            import warnings
            warnings.warn(
                f"Pairwise XOR will allocate ~{mem_bytes / 1e9:.1f} GB. "
                f"Consider reducing N={n} or using approximate methods.",
                ResourceWarning,
                stacklevel=2,
            )

        # Fully vectorized: (1, N, k) ^ (N, 1, k) → (N, N, k)
        xor = _torch.bitwise_xor(
            states.unsqueeze(0),  # (1, N, k)
            states.unsqueeze(1),  # (N, 1, k)
        )

        # Popcount and sum across k
        popcounts = self._popcount_uint8(xor)  # (N, N, k) int32
        total_bits = popcounts.sum(dim=2).to(self._dtype_float)  # (N, N)
        max_hd = k * n_bits
        matrix = total_bits / max_hd

        # Zero diagonal (self-resonance = 0)
        matrix.fill_diagonal_(0.0)
        return matrix


class TorchProjector:
    """PyTorch wrapper around QJLProjector for device-accelerated projection.

    Holds the random matrix R, quantile boundaries, and Gray code table as
    device tensors. Provides project_and_quantize() returning torch tensors.
    """

    def __init__(
        self,
        projector: "QJLProjector",  # noqa: F821 — forward ref
        device: str | None = None,
    ) -> None:
        """Create a TorchProjector from an existing QJLProjector.

        Args:
            projector: the QJLProjector to wrap (shares the same random matrix).
            device: "cpu", "mps", "cuda", or None for auto-detect.
        """
        if not _check_torch():
            raise ImportError("PyTorch is required for TorchProjector")

        import torch as _torch

        self._device = device or detect_torch_device()
        self._n_bits = projector.n_bits
        self._input_dim = projector.input_dim
        self._projection_dim = projector.projection_dim

        # Move data to device
        self._R = numpy_to_torch(
            projector.projection_matrix, device=self._device, dtype=_torch.float64
        )
        self._boundaries = numpy_to_torch(
            projector._boundaries, device=self._device, dtype=_torch.float64
        )
        self._gray = numpy_to_torch(
            projector._gray, device=self._device, dtype=_torch.uint8
        )

        self._backend = TorchResonanceBackend(device=self._device)

    @property
    def device(self) -> str:
        return self._device

    @property
    def projection_dim(self) -> int:
        return self._projection_dim

    @property
    def input_dim(self) -> int:
        return self._input_dim

    def project_and_quantize(
        self, vectors: "torch.Tensor | NDArray"
    ) -> "torch.Tensor":
        """Project and quantize vectors on the torch device.

        Args:
            vectors: shape (d,) or (n, d). NumPy arrays are auto-converted.

        Returns:
            uint8 tensor of Gray-coded 3-bit values.
        """
        import torch as _torch

        if isinstance(vectors, np.ndarray):
            vectors = numpy_to_torch(vectors, device=self._device, dtype=_torch.float64)
        elif vectors.device != _torch.device(self._device):
            vectors = vectors.to(self._device)

        projected = self._backend.project(self._R, vectors)
        return self._backend.quantize(projected, self._boundaries, self._gray)

    def batch_resonance_scores(
        self,
        ref: "torch.Tensor",
        batch: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute resonance scores via the torch backend."""
        return self._backend.batch_resonance_scores(ref, batch, self._n_bits)

    def pairwise_resonance_matrix(
        self,
        states: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute N×N pairwise resonance matrix via the torch backend."""
        return self._backend.pairwise_resonance_matrix(states, self._n_bits)
