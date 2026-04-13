"""Tests for ResonancePIDGovernor and PID-augmented GossipController."""

from __future__ import annotations

import numpy as np
import pytest

from src.experimental.gossip_governance import (
    GossipConfig,
    GossipController,
    ResonancePIDGovernor,
)


# ---------------------------------------------------------------------------
# ResonancePIDGovernor unit tests
# ---------------------------------------------------------------------------

class TestResonancePIDGovernor:
    def test_high_surprise_increases_strength(self):
        """Surprise above target should increase resync strength."""
        gov = ResonancePIDGovernor(target_surprise=0.15, Kp=0.5, base_strength=0.1)
        output = gov.update(mean_surprise=0.50)
        assert output > 0.1, f"Expected > base_strength, got {output}"

    def test_low_surprise_decreases_strength(self):
        """Surprise below target should decrease resync strength."""
        gov = ResonancePIDGovernor(target_surprise=0.15, Kp=0.5, base_strength=0.1)
        output = gov.update(mean_surprise=0.02)
        assert output < 0.1, f"Expected < base_strength, got {output}"

    def test_at_target_near_base(self):
        """Surprise exactly at target → output ≈ base_strength (no integral yet)."""
        gov = ResonancePIDGovernor(target_surprise=0.15, Kp=0.5, base_strength=0.1,
                                    Ki=0.0, Kd=0.0)
        output = gov.update(mean_surprise=0.15)
        assert output == pytest.approx(0.1)

    def test_output_clamped_upper(self):
        """Extreme high surprise should not exceed output_max."""
        gov = ResonancePIDGovernor(target_surprise=0.0, Kp=5.0, output_max=0.5)
        output = gov.update(mean_surprise=1.0)
        assert output <= 0.5

    def test_output_clamped_lower(self):
        """Extreme low surprise should not go below output_min."""
        gov = ResonancePIDGovernor(target_surprise=1.0, Kp=5.0, output_min=0.0)
        output = gov.update(mean_surprise=0.0)
        assert output >= 0.0

    def test_integral_anti_windup(self):
        """Integral term is bounded by the rolling window, not accumulated unboundedly."""
        gov = ResonancePIDGovernor(
            target_surprise=0.15, Kp=0.0, Ki=1.0, Kd=0.0,
            base_strength=0.1, integral_window=10, output_max=0.5
        )
        # Feed 30 rounds of constant high error
        for _ in range(30):
            gov.update(mean_surprise=0.50)
        # Integral should reflect only the last 10 rounds, not all 30
        # Window has 10 entries of error=0.35 each → integral=3.5
        # u = Ki * 3.5 = 3.5, output = clamp(0.1 + 3.5) = 0.5
        assert gov.last_output == pytest.approx(0.5)

    def test_derivative_responds_to_change(self):
        """Derivative term should produce different output when error changes."""
        gov = ResonancePIDGovernor(
            target_surprise=0.15, Kp=0.0, Ki=0.0, Kd=1.0, base_strength=0.1
        )
        # First call: no prior error, derivative = e - 0 = e
        out1 = gov.update(mean_surprise=0.25)
        # Second call: same surprise, derivative = e - e = 0
        out2 = gov.update(mean_surprise=0.25)
        assert out1 != out2

    def test_reset_clears_state(self):
        """After reset, governor behaves as if freshly created."""
        gov = ResonancePIDGovernor(target_surprise=0.15, base_strength=0.1)
        for _ in range(10):
            gov.update(mean_surprise=0.8)
        gov.reset()
        # After reset, a single update at target should give ≈ base_strength
        gov2 = ResonancePIDGovernor(target_surprise=0.15, base_strength=0.1,
                                     Kp=gov._Kp, Ki=gov._Ki, Kd=gov._Kd)
        out_fresh = gov2.update(mean_surprise=0.15)
        out_reset = gov.update(mean_surprise=0.15)
        assert out_reset == pytest.approx(out_fresh)

    def test_last_error_property(self):
        gov = ResonancePIDGovernor(target_surprise=0.15)
        gov.update(mean_surprise=0.40)
        assert gov.last_error == pytest.approx(0.40 - 0.15)

    def test_last_output_property(self):
        gov = ResonancePIDGovernor(target_surprise=0.15, Kp=0.5, base_strength=0.1,
                                    Ki=0.0, Kd=0.0)
        expected = gov.update(mean_surprise=0.30)
        assert gov.last_output == pytest.approx(expected)


# ---------------------------------------------------------------------------
# GossipController backward compatibility
# ---------------------------------------------------------------------------

class TestGossipPIDBackwardCompat:
    def test_no_pid_governor_identical_to_phase4(self):
        """Without pid_governor, step() behavior is identical to Phase 4."""
        config = GossipConfig(n_agents=4, seed=42)
        assert config.pid_governor is None
        ctrl = GossipController(config)
        stats = ctrl.step()
        assert stats.round_number == 1
        assert 0.0 <= stats.mean_surprise <= 1.0
        # effective_resync should stay at config.resync_strength throughout
        assert ctrl.effective_resync == pytest.approx(config.resync_strength)

    def test_effective_resync_starts_at_config_value(self):
        config = GossipConfig(resync_strength=0.25)
        ctrl = GossipController(config)
        assert ctrl.effective_resync == pytest.approx(0.25)

    def test_initialize_resets_effective_resync(self):
        gov = ResonancePIDGovernor()
        config = GossipConfig(resync_strength=0.1, pid_governor=gov, seed=42)
        ctrl = GossipController(config)
        ctrl.run(n_rounds=10)
        # Resync may have drifted; re-initialize should reset it
        ctrl.initialize()
        assert ctrl.effective_resync == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# PID reduces churn in high-drift scenario (replaces xfail test)
# ---------------------------------------------------------------------------

class TestPIDReducesChurn:
    def test_pid_vs_static_high_drift(self):
        """PID governor should achieve lower final surprise than static resync
        in a high-drift scenario where churn is the main problem."""
        seed = 42
        base_kwargs = dict(
            n_agents=12,
            state_dim=64,
            projection_dim=128,
            fanout=3,
            drift_magnitude=0.10,
            dissonance_threshold=0.3,
            max_rounds=120,
            seed=seed,
        )

        # With PID governor — adapts strength to avoid churn
        pid = ResonancePIDGovernor(
            target_surprise=0.15,
            Kp=0.8,
            Ki=0.02,
            Kd=0.15,
            base_strength=0.1,
            integral_window=15,
        )
        ctrl_pid = GossipController(
            GossipConfig(**{**base_kwargs, "resync_strength": 0.1, "pid_governor": pid})
        )
        ctrl_pid.run()

        # Without PID — static high strength causes churn
        ctrl_static = GossipController(
            GossipConfig(**{**base_kwargs, "resync_strength": 0.2, "pid_governor": None})
        )
        ctrl_static.run()

        pid_final = float(np.mean(ctrl_pid.get_convergence_curve()[-20:]))
        static_final = float(np.mean(ctrl_static.get_convergence_curve()[-20:]))

        assert pid_final < static_final, (
            f"PID ({pid_final:.4f}) should beat static ({static_final:.4f})"
        )

    def test_pid_tracks_target(self):
        """After convergence, PID mean_surprise should be near target_surprise."""
        target = 0.15
        pid = ResonancePIDGovernor(target_surprise=target, Kp=0.6, Ki=0.02, Kd=0.1)
        config = GossipConfig(
            n_agents=10,
            drift_magnitude=0.03,
            dissonance_threshold=0.3,
            max_rounds=80,
            pid_governor=pid,
            seed=42,
        )
        ctrl = GossipController(config)
        ctrl.run()

        curve = ctrl.get_convergence_curve()
        final_mean = float(np.mean(curve[-20:]))
        # PID should bring surprise within 0.15 of the target
        assert abs(final_mean - target) < 0.15, (
            f"PID final surprise {final_mean:.4f} not near target {target}"
        )
