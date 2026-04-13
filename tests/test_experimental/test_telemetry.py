"""
Tests for telemetry visualization module.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.experimental.telemetry import (
    ResonanceTelemetry,
    ascii_bar,
    ascii_convergence,
    ascii_heatmap,
)


# ---------------------------------------------------------------------------
# ASCII functions (always available, no matplotlib needed)
# ---------------------------------------------------------------------------

class TestAsciiHeatmap:
    def test_basic_output(self):
        matrix = np.eye(5)
        output = ascii_heatmap(matrix, title="Test")
        assert "Test" in output
        assert len(output.splitlines()) >= 5 + 2  # 5 data rows + header

    def test_with_labels(self):
        matrix = np.eye(3)
        output = ascii_heatmap(matrix, labels=["a", "b", "c"])
        assert "a" in output
        assert "b" in output

    def test_large_matrix_sampled(self):
        matrix = np.random.default_rng(42).random((100, 100))
        output = ascii_heatmap(matrix)
        # Should have 50 data rows + header
        lines = output.strip().splitlines()
        # Title + underline + 50 data rows = at minimum
        assert len(lines) >= 50

    def test_zero_matrix(self):
        matrix = np.zeros((4, 4))
        output = ascii_heatmap(matrix)
        assert " " in output  # space = lowest block char


class TestAsciiConvergence:
    def test_basic_output(self):
        surprises = [0.5, 0.4, 0.3, 0.2, 0.1]
        output = ascii_convergence(surprises, title="Conv")
        assert "Conv" in output
        assert "#" in output

    def test_empty_data(self):
        output = ascii_convergence([], title="Empty")
        assert "no data" in output

    def test_single_value(self):
        output = ascii_convergence([0.5], title="Single")
        assert "0.5000" in output


class TestAsciiBar:
    def test_basic_output(self):
        output = ascii_bar([0.3, 0.7], ["low", "high"], title="Test")
        assert "Test" in output
        assert "low" in output
        assert "high" in output

    def test_empty(self):
        output = ascii_bar([], [], title="Empty")
        assert "no data" in output


# ---------------------------------------------------------------------------
# ResonanceTelemetry
# ---------------------------------------------------------------------------

class TestResonanceTelemetry:
    @pytest.fixture
    def telemetry(self):
        return ResonanceTelemetry()

    def test_init(self, telemetry):
        assert isinstance(telemetry.has_matplotlib, bool)

    def test_heatmap_ascii_fallback(self, telemetry, monkeypatch):
        """Force ASCII fallback by pretending matplotlib is absent."""
        monkeypatch.setattr(telemetry, "_has_mpl", False)
        matrix = np.eye(5)
        result = telemetry.plot_resonance_heatmap(matrix, title="Test")
        assert result is not None  # ASCII string returned
        assert "Test" in result

    def test_convergence_ascii_fallback(self, telemetry, monkeypatch):
        monkeypatch.setattr(telemetry, "_has_mpl", False)

        class _FakeStats:
            def __init__(self, mean_surprise):
                self.mean_surprise = mean_surprise

        class _FakeController:
            round_history = [_FakeStats(0.5), _FakeStats(0.3), _FakeStats(0.1)]

        result = telemetry.plot_convergence(_FakeController(), title="Conv")
        assert result is not None
        assert "Conv" in result

    def test_pid_response_ascii_fallback(self, telemetry, monkeypatch):
        monkeypatch.setattr(telemetry, "_has_mpl", False)
        result = telemetry.plot_pid_response(
            [0.5, 0.3, 0.1], [0.1, 0.15, 0.12], title="PID"
        )
        assert result is not None
        assert "PID" in result

    def test_surprise_dist_ascii_fallback(self, telemetry, monkeypatch):
        monkeypatch.setattr(telemetry, "_has_mpl", False)
        result = telemetry.plot_surprise_distribution(
            {"a": 0.3, "b": 0.7, "c": 0.1}, title="Dist"
        )
        assert result is not None
        assert "Dist" in result

    def test_feature_radar_ascii_fallback(self, telemetry, monkeypatch):
        monkeypatch.setattr(telemetry, "_has_mpl", False)
        fv = np.random.default_rng(42).random(15)
        mean = np.full(15, 0.5)
        result = telemetry.plot_feature_radar(fv, mean, title="Radar")
        assert result is not None
        assert "Radar" in result

    def test_empty_history(self, telemetry, monkeypatch):
        monkeypatch.setattr(telemetry, "_has_mpl", False)

        class _FakeController:
            round_history = []

        result = telemetry.plot_convergence(_FakeController())
        assert result is None  # no data → nothing returned

    def test_large_heatmap_downsampled(self, telemetry, monkeypatch):
        monkeypatch.setattr(telemetry, "_has_mpl", False)
        matrix = np.random.default_rng(42).random((200, 200))
        with pytest.warns(match="Downsampling"):
            result = telemetry.plot_resonance_heatmap(matrix)
        assert result is not None
