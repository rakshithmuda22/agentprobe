"""
EXPERIMENTAL — QJL Resonance Module for AgentProbe
===================================================

Speculative research prototype exploring QJL projection spaces as a
coordination metric for decentralized multi-agent systems.

Real foundations (VALIDATED):
  - QJL: arxiv 2406.03482 (ICLR 2025)
  - PolarQuant: arxiv 2502.02617 (NeurIPS 2025)
  - SimHash angular-Hamming: Charikar 2002
  - PyTorch backend: standard linear algebra + bitwise ops

Speculative extensions (clearly labeled in code):
  - Angular divergence in QJL space as VFE "surprise" proxy
  - Gossip-based convergence via resonance thresholding
  - Manifold dissonance as refactoring trigger (refactor_proposals.py)

Modules:
  resonance_core     — QJL projector, quantization, XOR+popcount scoring
  kinetic_agent      — Agent with belief, trajectory, neighbor prediction
  gossip_governance  — Gossip controller, PID governor
  code_transformer   — Source → features → QJL manifold; ManifoldScanner
  torch_backend      — PyTorch CPU/MPS/CUDA acceleration
  refactor_proposals — SPECULATIVE: manifold-based refactoring suggestions
  telemetry          — Visualization (matplotlib + ASCII fallback)
"""

__version__ = "0.0.2-experimental"

try:
    import numpy as np  # noqa: F401
except ImportError as e:
    raise ImportError(
        "The experimental module requires numpy. "
        "Install with: pip install agentprobe[experimental]"
    ) from e
