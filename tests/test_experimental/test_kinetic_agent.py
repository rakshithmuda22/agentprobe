"""Tests for kinetic_agent.py — agent behavior and prediction."""

from __future__ import annotations

import numpy as np
import pytest

from src.experimental.resonance_core import QJLProjector
from src.experimental.kinetic_agent import KineticAgent, AgentTrajectory


class TestAgentTrajectory:
    def test_empty_velocity(self):
        t = AgentTrajectory()
        assert t.velocity is None

    def test_single_entry_no_velocity(self):
        t = AgentTrajectory()
        t.append(np.array([1, 2, 3], dtype=np.uint8))
        assert t.velocity is None

    def test_velocity_computed(self):
        t = AgentTrajectory()
        t.append(np.array([1, 2, 3], dtype=np.uint8))
        t.append(np.array([3, 2, 1], dtype=np.uint8))
        vel = t.velocity
        assert vel is not None
        np.testing.assert_array_equal(vel, [2, 0, -2])

    def test_max_history_eviction(self):
        t = AgentTrajectory(max_history=3)
        for i in range(5):
            t.append(np.array([i], dtype=np.uint8))
        assert t.length == 3
        assert t.latest is not None
        assert t.latest[0] == 4

    def test_acceleration(self):
        t = AgentTrajectory()
        t.append(np.array([0], dtype=np.uint8))
        t.append(np.array([1], dtype=np.uint8))
        t.append(np.array([3], dtype=np.uint8))
        acc = t.acceleration
        assert acc is not None
        # vel1 = 1-0 = 1, vel2 = 3-1 = 2, acc = 2-1 = 1
        np.testing.assert_array_equal(acc, [1])


class TestKineticAgent:
    def test_create_with_belief(self, shared_projector: QJLProjector):
        belief = np.ones(64) / np.sqrt(64)
        agent = KineticAgent("a1", shared_projector, initial_belief=belief)
        assert agent.agent_id == "a1"
        assert agent.projected_state.shape == (128,)
        assert agent.projected_state.dtype == np.uint8

    def test_create_default_belief(self, shared_projector: QJLProjector):
        agent = KineticAgent("a2", shared_projector)
        np.testing.assert_array_equal(agent.belief_vector, np.zeros(64))

    def test_update_belief_changes_projection(self, shared_projector: QJLProjector):
        agent = KineticAgent("a3", shared_projector)
        old_proj = agent.projected_state.copy()
        agent.update_belief(np.ones(64))
        # Very likely to differ
        assert not np.array_equal(agent.projected_state, old_proj)

    def test_trajectory_grows(self, shared_projector: QJLProjector):
        agent = KineticAgent("a4", shared_projector)
        # Initial projection is already in trajectory
        assert agent.trajectory.length == 1
        agent.update_belief(np.ones(64))
        assert agent.trajectory.length == 2

    def test_observe_first_returns_neutral(self, shared_projector: QJLProjector):
        a1 = KineticAgent("a1", shared_projector)
        a2 = KineticAgent("a2", shared_projector, initial_belief=np.ones(64) / 8)
        surprise = a1.observe_neighbor("a2", a2.projected_state)
        # First observation: no prediction available → neutral 0.5
        assert surprise == 0.5

    def test_observe_second_uses_prediction(self, shared_projector: QJLProjector):
        rng = np.random.default_rng(42)
        a1 = KineticAgent("a1", shared_projector, initial_belief=rng.standard_normal(64))
        a2 = KineticAgent("a2", shared_projector, initial_belief=rng.standard_normal(64))

        # First observation → neutral
        a1.observe_neighbor("a2", a2.projected_state)

        # Update a2 slightly
        a2.update_belief(a2.belief_vector + rng.standard_normal(64) * 0.01)

        # Second observation → still neutral (need 2 in neighbor trajectory for velocity)
        a1.observe_neighbor("a2", a2.projected_state)

        # Update a2 again
        a2.update_belief(a2.belief_vector + rng.standard_normal(64) * 0.01)

        # Third observation → now prediction available
        surprise = a1.observe_neighbor("a2", a2.projected_state)
        assert surprise != 0.5  # Should use actual prediction now

    def test_predict_neighbor_none_without_history(self, shared_projector: QJLProjector):
        agent = KineticAgent("a1", shared_projector)
        assert agent.predict_neighbor("unknown") is None

    def test_predict_neighbor_clamps(self, shared_projector: QJLProjector):
        agent = KineticAgent("a1", shared_projector)
        # Manually feed extreme values
        agent._neighbor_trajectories["x"] = AgentTrajectory()
        agent._neighbor_trajectories["x"].append(np.full(128, 5, dtype=np.uint8))
        agent._neighbor_trajectories["x"].append(np.full(128, 7, dtype=np.uint8))
        # velocity = 2, so predicted = 7 + 2 = 9 → clamped to 7
        pred = agent.predict_neighbor("x")
        assert pred is not None
        assert np.all(pred <= 7)

    def test_get_surprise_unknown(self, shared_projector: QJLProjector):
        agent = KineticAgent("a1", shared_projector)
        assert agent.get_surprise("nonexistent") == -1.0

    def test_mean_surprise_empty(self, shared_projector: QJLProjector):
        agent = KineticAgent("a1", shared_projector)
        assert agent.get_mean_surprise() == 0.5

    def test_most_dissonant_empty(self, shared_projector: QJLProjector):
        agent = KineticAgent("a1", shared_projector)
        assert agent.get_most_dissonant_neighbor() is None

    def test_dimension_mismatch_raises(self, shared_projector: QJLProjector):
        with pytest.raises(ValueError):
            KineticAgent("bad", shared_projector, initial_belief=np.ones(32))
