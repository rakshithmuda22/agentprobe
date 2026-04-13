"""
Gossip Governance — Decentralized Resonance-Based Coordination
===============================================================

A gossip controller where agents share quantized projected states with
random neighbors and only resynchronize when "surprise" exceeds a
threshold. This prevents the "semantic telephone" decay problem where
frequent blind synchronization causes belief drift.

Architecture:
  - No central orchestrator — convergence emerges from local interactions
  - Each round: drift → gossip → selective resync → stats
  - Agents only sync with neighbors whose surprise exceeds the threshold
  - Resync blends raw belief vectors (simulation-level simplification;
    a real P2P system would resync via projected-space negotiation)

VALIDATED: gossip protocols with selective sync are well-established
in distributed systems (epidemic dissemination, CRDT convergence).

SPECULATIVE: using QJL-projected resonance scores as the sync trigger,
interpreting convergence as "mutual predictability" in an active
inference sense.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.experimental.resonance_core import (
    QJLProjector,
    batch_resonance_scores,
    pairwise_resonance_matrix,
)
from src.experimental.kinetic_agent import KineticAgent


# ---------------------------------------------------------------------------
# PID Resonance Governor
# ---------------------------------------------------------------------------

class ResonancePIDGovernor:
    """Entropy-aware PID controller for dynamic resync_strength.

    Solves the stability-plasticity dilemma: high drift causes churn
    with static resync_strength. The PID governor adapts the blend weight
    in response to the swarm's current surprise level.

    Control law:
        e(t) = mean_surprise(t) - target_surprise
        integral = sum of error_window (anti-windup rolling window)
        u(t) = Kp * e(t) + Ki * integral + Kd * (e(t) - e(t-1))
        resync_strength(t) = clamp(base_strength + u(t), output_min, output_max)

    When mean_surprise > target: u > 0 → increased resync (pull together)
    When mean_surprise < target: u < 0 → decreased resync (allow drift)

    Args:
        target_surprise: desired mean_surprise level
        Kp: proportional gain
        Ki: integral gain
        Kd: derivative gain
        base_strength: resync_strength when e(t) = 0 (neutral point)
        integral_window: anti-windup window size in rounds
        output_min: lower clamp for resync_strength
        output_max: upper clamp for resync_strength
    """

    def __init__(
        self,
        target_surprise: float = 0.15,
        Kp: float = 0.5,
        Ki: float = 0.01,
        Kd: float = 0.1,
        base_strength: float = 0.1,
        integral_window: int = 20,
        output_min: float = 0.0,
        output_max: float = 0.5,
    ) -> None:
        self._target = target_surprise
        self._Kp = Kp
        self._Ki = Ki
        self._Kd = Kd
        self._base = base_strength
        self._output_min = output_min
        self._output_max = output_max

        self._error_window: deque[float] = deque(maxlen=integral_window)
        self._prev_error: float = 0.0
        self._last_output: float = base_strength
        self._last_error: float = 0.0

    def update(self, mean_surprise: float) -> float:
        """Feed current mean_surprise, compute and return new resync_strength.

        Args:
            mean_surprise: swarm's mean surprise from the completed round

        Returns:
            New resync_strength in [output_min, output_max]
        """
        e = mean_surprise - self._target

        self._error_window.append(e)
        integral = sum(self._error_window)
        derivative = e - self._prev_error

        u = self._Kp * e + self._Ki * integral + self._Kd * derivative
        output = float(np.clip(self._base + u, self._output_min, self._output_max))

        self._prev_error = e
        self._last_error = e
        self._last_output = output
        return output

    @property
    def last_error(self) -> float:
        """Error from the most recent update call."""
        return self._last_error

    @property
    def last_output(self) -> float:
        """Resync strength from the most recent update call."""
        return self._last_output

    def reset(self) -> None:
        """Clear integral window and error history."""
        self._error_window.clear()
        self._prev_error = 0.0
        self._last_error = 0.0
        self._last_output = self._base


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GossipConfig:
    """Configuration for the gossip controller."""

    n_agents: int = 8
    state_dim: int = 64
    projection_dim: int = 128
    n_bits: int = 3
    fanout: int = 3           # neighbors per gossip round
    dissonance_threshold: float = 0.4  # surprise above this triggers resync
    resync_strength: float = 0.1       # static blend weight (overridden by pid_governor)
    drift_magnitude: float = 0.05      # per-round belief perturbation
    max_rounds: int = 100
    seed: int | None = 42
    pid_governor: ResonancePIDGovernor | None = None  # None = static resync_strength


# ---------------------------------------------------------------------------
# Round statistics
# ---------------------------------------------------------------------------

@dataclass
class RoundStats:
    """Statistics from one gossip round."""

    round_number: int
    mean_surprise: float
    max_surprise: float
    min_surprise: float
    n_resyncs: int
    n_gossip_exchanges: int
    agent_surprises: dict[str, float] = field(default_factory=dict)

    @property
    def is_converged(self) -> bool:
        """Heuristic: converged when max surprise < 0.15."""
        return self.max_surprise < 0.15


# ---------------------------------------------------------------------------
# Gossip Controller
# ---------------------------------------------------------------------------

class GossipController:
    """Decentralized gossip-based resonance coordination.

    Lifecycle:
        controller = GossipController(config)
        controller.initialize()
        stats = controller.run()
        curve = controller.get_convergence_curve()
    """

    def __init__(self, config: GossipConfig | None = None) -> None:
        self._config = config or GossipConfig()
        self._rng = np.random.default_rng(self._config.seed)

        self._projector = QJLProjector(
            input_dim=self._config.state_dim,
            projection_dim=self._config.projection_dim,
            n_bits=self._config.n_bits,
            seed=self._config.seed,
        )

        self._agents: dict[str, KineticAgent] = {}
        self._round_history: list[RoundStats] = []
        self._round_counter = 0
        self._initialized = False
        # Effective resync strength — updated by PID governor if present
        self._effective_resync: float = self._config.resync_strength

    @property
    def config(self) -> GossipConfig:
        return self._config

    @property
    def agents(self) -> dict[str, KineticAgent]:
        return self._agents

    @property
    def projector(self) -> QJLProjector:
        return self._projector

    @property
    def round_history(self) -> list[RoundStats]:
        return self._round_history

    @property
    def effective_resync(self) -> float:
        """Current resync strength (static or PID-adjusted)."""
        return self._effective_resync

    def initialize(self) -> None:
        """Create N agents with random unit belief vectors."""
        self._agents.clear()
        self._round_history.clear()
        self._round_counter = 0
        self._effective_resync = self._config.resync_strength
        if self._config.pid_governor is not None:
            self._config.pid_governor.reset()

        for i in range(self._config.n_agents):
            # Random unit vector as initial belief
            belief = self._rng.standard_normal(self._config.state_dim)
            belief /= np.linalg.norm(belief)

            agent = KineticAgent(
                agent_id=f"agent_{i:04d}",
                projector=self._projector,
                initial_belief=belief,
            )
            self._agents[agent.agent_id] = agent

        self._initialized = True

    def step(self) -> RoundStats:
        """Execute one gossip round: drift → gossip → resync → stats."""
        if not self._initialized:
            self.initialize()

        self._round_counter += 1
        agent_list = list(self._agents.values())
        cfg = self._config

        # --- Phase 1: DRIFT ---
        # Each agent's belief gets a small random perturbation
        for agent in agent_list:
            if cfg.drift_magnitude > 0:
                noise = self._rng.standard_normal(cfg.state_dim)
                noise *= cfg.drift_magnitude
                new_belief = agent.belief_vector + noise
                norm = np.linalg.norm(new_belief)
                if norm > 0:
                    new_belief /= norm
                agent.update_belief(new_belief)

        # --- Phase 2: GOSSIP EXCHANGE ---
        # Each agent selects `fanout` random neighbors and observes their state
        all_surprises: dict[str, float] = {}
        gossip_pairs: list[tuple[KineticAgent, KineticAgent]] = []
        n_exchanges = 0

        for agent in agent_list:
            others = [a for a in agent_list if a.agent_id != agent.agent_id]
            k = min(cfg.fanout, len(others))
            if k == 0:
                all_surprises[agent.agent_id] = 0.0
                continue

            neighbor_indices = self._rng.choice(len(others), size=k, replace=False)
            neighbors = [others[idx] for idx in neighbor_indices]

            surprises = []
            for neighbor in neighbors:
                surprise = agent.observe_neighbor(
                    neighbor.agent_id, neighbor.projected_state
                )
                surprises.append(surprise)
                gossip_pairs.append((agent, neighbor))
                n_exchanges += 1

            all_surprises[agent.agent_id] = float(np.mean(surprises))

        # --- Phase 3: SELECTIVE RESYNCHRONIZATION ---
        # Only sync with neighbors whose surprise exceeds threshold.
        # This prevents "semantic telephone" — blind sync that erodes beliefs.
        n_resyncs = 0

        for agent, neighbor in gossip_pairs:
            surprise = agent.get_surprise(neighbor.agent_id)
            if surprise > cfg.dissonance_threshold:
                # Blend belief toward the dissonant neighbor using current
                # effective resync strength (PID-adjusted or static).
                # NOTE: Simulation-level access to raw beliefs. In a real
                # P2P system, resync would use projected-space negotiation.
                blended = (
                    (1 - self._effective_resync) * agent.belief_vector
                    + self._effective_resync * neighbor.belief_vector
                )
                norm = np.linalg.norm(blended)
                if norm > 0:
                    blended /= norm
                agent.update_belief(blended)
                n_resyncs += 1

        # --- Phase 4: STATISTICS ---
        surprise_values = list(all_surprises.values())
        stats = RoundStats(
            round_number=self._round_counter,
            mean_surprise=float(np.mean(surprise_values)) if surprise_values else 0.0,
            max_surprise=float(np.max(surprise_values)) if surprise_values else 0.0,
            min_surprise=float(np.min(surprise_values)) if surprise_values else 0.0,
            n_resyncs=n_resyncs,
            n_gossip_exchanges=n_exchanges,
            agent_surprises=all_surprises,
        )
        self._round_history.append(stats)

        # --- Phase 4b: PID UPDATE ---
        # Governor adjusts effective_resync for the *next* round based on
        # this round's mean_surprise. Without a governor, effective_resync
        # stays at config.resync_strength throughout.
        if cfg.pid_governor is not None:
            self._effective_resync = cfg.pid_governor.update(stats.mean_surprise)

        return stats

    def run(self, n_rounds: int | None = None) -> list[RoundStats]:
        """Run multiple gossip rounds.

        Args:
            n_rounds: number of rounds (defaults to config.max_rounds)

        Returns:
            List of RoundStats for all rounds
        """
        if not self._initialized:
            self.initialize()

        rounds = n_rounds or self._config.max_rounds
        results = []
        for _ in range(rounds):
            stats = self.step()
            results.append(stats)
        return results

    def get_convergence_curve(self) -> list[float]:
        """Mean surprise per round — the primary convergence metric."""
        return [s.mean_surprise for s in self._round_history]

    def get_resync_curve(self) -> list[int]:
        """Resyncs per round — should decrease as swarm converges."""
        return [s.n_resyncs for s in self._round_history]

    def get_current_resonance_matrix(self) -> NDArray[np.float64]:
        """Full pairwise resonance matrix for current agent states.

        Returns shape (n, n) float64 array.
        """
        agent_list = list(self._agents.values())
        states = np.array([a.projected_state for a in agent_list])
        return pairwise_resonance_matrix(states, n_bits=self._config.n_bits)

    def inject_dissonant_agent(
        self, agent_id: str | None = None, magnitude: float = 5.0
    ) -> str:
        """Inject a "dissonant" agent with a wildly different belief.

        Useful for testing whether the swarm detects and adapts to
        an outlier. The agent gets a belief vector pointing in a
        random direction, scaled by `magnitude`.

        Returns the agent_id of the dissonant agent.
        """
        if agent_id is None:
            agent_id = f"dissonant_{self._round_counter:04d}"

        belief = self._rng.standard_normal(self._config.state_dim)
        belief = belief / np.linalg.norm(belief) * magnitude

        agent = KineticAgent(
            agent_id=agent_id,
            projector=self._projector,
            initial_belief=belief,
        )
        self._agents[agent_id] = agent
        return agent_id
