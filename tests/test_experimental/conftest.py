from __future__ import annotations

import numpy as np
import pytest

from src.experimental.resonance_core import QJLProjector
from src.experimental.gossip_governance import GossipConfig


@pytest.fixture
def shared_projector() -> QJLProjector:
    return QJLProjector(input_dim=64, projection_dim=128, n_bits=3, seed=42)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(64)
    return v / np.linalg.norm(v)


@pytest.fixture
def default_gossip_config() -> GossipConfig:
    return GossipConfig(seed=42)
