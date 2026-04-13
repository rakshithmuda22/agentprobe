"""
Kinetic Agent — Belief Projection with Predictive Trajectory
=============================================================

An agent that maintains a belief vector, projects it into the QJL
manifold, and tracks first-order trajectory for neighbor prediction.

VALIDATED components:
  - QJL projection preserves angular relationships
  - Linear extrapolation on quantized states is standard prediction

SPECULATIVE components:
  - "Surprise" = resonance score between predicted and actual neighbor
    state, interpreted as variational free energy
  - Agents minimize surprise by adjusting beliefs when dissonance is
    detected — this is the active inference framing
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.experimental.resonance_core import QJLProjector, calculate_resonance_score


# ---------------------------------------------------------------------------
# Trajectory tracker
# ---------------------------------------------------------------------------

@dataclass
class AgentTrajectory:
    """Ring buffer of recent quantized projected states with velocity."""

    max_history: int = 10
    _history: list[NDArray[np.uint8]] = field(default_factory=list, repr=False)

    def append(self, state: NDArray[np.uint8]) -> None:
        """Add state, evict oldest if at capacity."""
        self._history.append(state.copy())
        if len(self._history) > self.max_history:
            self._history.pop(0)

    @property
    def length(self) -> int:
        return len(self._history)

    @property
    def latest(self) -> NDArray[np.uint8] | None:
        return self._history[-1] if self._history else None

    @property
    def velocity(self) -> NDArray[np.int16] | None:
        """First-order derivative of manifold position.

        Computed as signed difference between the two most recent states.
        Uses int16 to handle signed deltas on uint8 values (range -7..+7).

        Returns None if fewer than 2 observations.
        """
        if len(self._history) < 2:
            return None
        curr = self._history[-1].astype(np.int16)
        prev = self._history[-2].astype(np.int16)
        return curr - prev

    @property
    def acceleration(self) -> NDArray[np.int16] | None:
        """Second-order derivative — rate of change of velocity.

        Returns None if fewer than 3 observations.
        """
        if len(self._history) < 3:
            return None
        s2 = self._history[-1].astype(np.int16)
        s1 = self._history[-2].astype(np.int16)
        s0 = self._history[-3].astype(np.int16)
        return (s2 - s1) - (s1 - s0)


# ---------------------------------------------------------------------------
# Kinetic Agent
# ---------------------------------------------------------------------------

# Neutral prior surprise when no prediction is possible
_NEUTRAL_SURPRISE = 0.5


class KineticAgent:
    """Agent with QJL-projected belief state and predictive trajectory.

    Each agent:
    1. Maintains a raw belief vector (continuous, high-dimensional)
    2. Projects it through a shared QJLProjector to get a compact
       quantized representation
    3. Tracks its own trajectory (ring buffer of projected states)
    4. Tracks neighbor trajectories to predict their next state
    5. Computes "surprise" when observing actual neighbor states

    CRITICAL: All agents in a swarm MUST share the same QJLProjector
    instance. Projections from different random matrices are incomparable.
    """

    __slots__ = (
        "_agent_id",
        "_projector",
        "_belief",
        "_projected",
        "_trajectory",
        "_neighbor_trajectories",
        "_neighbor_surprises",
    )

    def __init__(
        self,
        agent_id: str,
        projector: QJLProjector,
        initial_belief: NDArray[np.float64] | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._projector = projector
        self._trajectory = AgentTrajectory()
        self._neighbor_trajectories: dict[str, AgentTrajectory] = {}
        self._neighbor_surprises: dict[str, float] = {}

        dim = projector.input_dim
        if initial_belief is not None:
            if initial_belief.shape != (dim,):
                raise ValueError(
                    f"Belief dim {initial_belief.shape} != projector input_dim ({dim},)"
                )
            self._belief = initial_belief.copy()
        else:
            self._belief = np.zeros(dim, dtype=np.float64)

        # Initial projection
        self._projected = self._projector.project_and_quantize(self._belief)
        self._trajectory.append(self._projected)

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def belief_vector(self) -> NDArray[np.float64]:
        return self._belief

    @property
    def projected_state(self) -> NDArray[np.uint8]:
        """Current quantized projected state — this is what gets shared."""
        return self._projected

    @property
    def trajectory(self) -> AgentTrajectory:
        return self._trajectory

    @property
    def velocity(self) -> NDArray[np.int16] | None:
        """First-order derivative of this agent's manifold position."""
        return self._trajectory.velocity

    def update_belief(self, new_belief: NDArray[np.float64]) -> None:
        """Set new belief, reproject, and update trajectory."""
        self._belief = new_belief.copy()
        self._projected = self._projector.project_and_quantize(self._belief)
        self._trajectory.append(self._projected)

    def predict_neighbor(self, neighbor_id: str) -> NDArray[np.uint8] | None:
        """Predict neighbor's next projected state via linear extrapolation.

        Uses the neighbor's trajectory velocity (first-order derivative)
        to extrapolate: predicted = last_state + velocity.

        Results are clamped to valid quantization range [0, 2^n_bits - 1].

        Returns None if we have fewer than 2 observations of this neighbor.
        """
        traj = self._neighbor_trajectories.get(neighbor_id)
        if traj is None or traj.length < 2:
            return None

        latest = traj.latest
        vel = traj.velocity
        if latest is None or vel is None:
            return None

        # Extrapolate in int16 space to handle overflow, then clamp
        max_val = (1 << self._projector.n_bits) - 1  # 7 for 3-bit
        predicted = latest.astype(np.int16) + vel
        predicted = np.clip(predicted, 0, max_val)
        return predicted.astype(np.uint8)

    def observe_neighbor(
        self, neighbor_id: str, neighbor_state: NDArray[np.uint8]
    ) -> float:
        """Observe a neighbor's actual projected state and compute surprise.

        SPECULATIVE (Active Inference framing):
        Surprise = resonance_score(predicted, actual). A high score means
        the neighbor moved in an unexpected direction on the manifold —
        analogous to high variational free energy.

        Steps:
        1. Predict where the neighbor should be (linear extrapolation)
        2. Compute surprise = resonance_score(predicted, actual)
           If no prediction available, use neutral prior (0.5)
        3. Record actual state in neighbor's trajectory
        4. Store surprise value
        5. Return surprise

        Args:
            neighbor_id: identifier of the observed neighbor
            neighbor_state: their current quantized projected state

        Returns:
            Surprise score in [0.0, 1.0]
        """
        # Ensure neighbor trajectory exists
        if neighbor_id not in self._neighbor_trajectories:
            self._neighbor_trajectories[neighbor_id] = AgentTrajectory()

        # Step 1: Predict
        predicted = self.predict_neighbor(neighbor_id)

        # Step 2: Compute surprise
        if predicted is not None:
            surprise = calculate_resonance_score(
                predicted, neighbor_state, n_bits=self._projector.n_bits
            )
        else:
            surprise = _NEUTRAL_SURPRISE

        # Step 3: Record actual state
        self._neighbor_trajectories[neighbor_id].append(neighbor_state)

        # Step 4: Store
        self._neighbor_surprises[neighbor_id] = surprise

        # Step 5: Return
        return surprise

    def get_surprise(self, neighbor_id: str) -> float:
        """Last surprise for a neighbor, or -1.0 if never observed."""
        return self._neighbor_surprises.get(neighbor_id, -1.0)

    def get_mean_surprise(self) -> float:
        """Mean surprise across all observed neighbors."""
        if not self._neighbor_surprises:
            return _NEUTRAL_SURPRISE
        return float(np.mean(list(self._neighbor_surprises.values())))

    def get_most_dissonant_neighbor(self) -> tuple[str, float] | None:
        """Return (neighbor_id, surprise) for the highest-surprise neighbor."""
        if not self._neighbor_surprises:
            return None
        nid = max(self._neighbor_surprises, key=self._neighbor_surprises.get)  # type: ignore[arg-type]
        return nid, self._neighbor_surprises[nid]
