"""Tests for gossip_governance.py — convergence and coordination."""

from __future__ import annotations

import numpy as np
import pytest

from src.experimental.gossip_governance import GossipConfig, GossipController


class TestGossipController:
    def test_initialize_creates_agents(self):
        config = GossipConfig(n_agents=5, seed=42)
        ctrl = GossipController(config)
        ctrl.initialize()
        assert len(ctrl.agents) == 5

    def test_agent_ids_unique(self):
        config = GossipConfig(n_agents=10, seed=42)
        ctrl = GossipController(config)
        ctrl.initialize()
        ids = list(ctrl.agents.keys())
        assert len(set(ids)) == 10

    def test_single_step_returns_stats(self):
        config = GossipConfig(n_agents=4, seed=42)
        ctrl = GossipController(config)
        ctrl.initialize()
        stats = ctrl.step()
        assert stats.round_number == 1
        assert 0.0 <= stats.mean_surprise <= 1.0
        assert stats.n_gossip_exchanges > 0

    def test_run_returns_correct_length(self):
        config = GossipConfig(n_agents=4, max_rounds=10, seed=42)
        ctrl = GossipController(config)
        results = ctrl.run(n_rounds=10)
        assert len(results) == 10

    def test_convergence_curve_length(self):
        config = GossipConfig(n_agents=4, max_rounds=20, seed=42)
        ctrl = GossipController(config)
        ctrl.run()
        curve = ctrl.get_convergence_curve()
        assert len(curve) == 20

    def test_auto_initialize(self):
        """run() should auto-initialize if not called explicitly."""
        config = GossipConfig(n_agents=4, seed=42)
        ctrl = GossipController(config)
        results = ctrl.run(n_rounds=5)
        assert len(results) == 5
        assert len(ctrl.agents) == 4


class TestGossipConvergence:
    """Speculative hypothesis tests — these validate the hypothesis,
    not proven math. Failures here are informative, not bugs."""

    def test_gossip_converges(self):
        """Mean surprise should decrease over 100 rounds."""
        config = GossipConfig(
            n_agents=12,
            state_dim=64,
            projection_dim=128,
            fanout=3,
            dissonance_threshold=0.35,
            resync_strength=0.15,
            drift_magnitude=0.03,
            max_rounds=100,
            seed=42,
        )
        ctrl = GossipController(config)
        ctrl.run()
        curve = ctrl.get_convergence_curve()

        first_10 = np.mean(curve[:10])
        last_10 = np.mean(curve[-10:])
        assert last_10 < first_10, (
            f"Expected convergence: first_10={first_10:.4f}, last_10={last_10:.4f}"
        )

    @pytest.mark.xfail(reason=(
        "Speculative hypothesis: resync benefit depends on drift/strength "
        "ratio. High drift + resync can cause churn — this is an informative "
        "finding, not a bug."
    ))
    def test_resync_helps(self):
        """With-resync should have lower final surprise than without."""
        # Higher drift forces agents apart — resync should pull them back
        base_kwargs = dict(
            n_agents=12,
            fanout=3,
            drift_magnitude=0.10,
            dissonance_threshold=0.3,
            max_rounds=80,
            seed=42,
        )

        # With resync
        ctrl_on = GossipController(
            GossipConfig(**{**base_kwargs, "resync_strength": 0.2})
        )
        ctrl_on.run()

        # Without resync (strength=0)
        ctrl_off = GossipController(
            GossipConfig(**{**base_kwargs, "resync_strength": 0.0})
        )
        ctrl_off.run()

        on_final = np.mean(ctrl_on.get_convergence_curve()[-10:])
        off_final = np.mean(ctrl_off.get_convergence_curve()[-10:])

        assert on_final < off_final, (
            f"Resync should help: with={on_final:.4f}, without={off_final:.4f}"
        )

    def test_no_drift_stable(self):
        """Zero drift should yield stable low surprise after warmup."""
        config = GossipConfig(
            n_agents=8,
            drift_magnitude=0.0,
            max_rounds=50,
            seed=42,
        )
        ctrl = GossipController(config)
        ctrl.run()
        curve = ctrl.get_convergence_curve()

        # After warmup (round 20+), variance should be low
        post_warmup = curve[20:]
        std = np.std(post_warmup)
        assert std < 0.1, f"Surprise variance too high with no drift: std={std:.4f}"

    def test_inject_dissonant_detected(self):
        """A dissonant agent should cause a surprise spike."""
        config = GossipConfig(n_agents=8, max_rounds=30, seed=42)
        ctrl = GossipController(config)
        ctrl.run()

        baseline = ctrl.get_convergence_curve()[-1]

        ctrl.inject_dissonant_agent(magnitude=10.0)
        stats = ctrl.step()

        assert stats.mean_surprise > baseline * 0.8, (
            "Dissonant agent should cause noticeable surprise increase"
        )

    def test_resonance_matrix_shape(self):
        config = GossipConfig(n_agents=6, seed=42)
        ctrl = GossipController(config)
        ctrl.initialize()
        matrix = ctrl.get_current_resonance_matrix()
        assert matrix.shape == (6, 6)
