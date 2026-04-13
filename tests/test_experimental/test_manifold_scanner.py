"""
Tests for ManifoldScanner and CodeSwarm.from_directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from src.experimental.code_transformer import CodeSwarm, ManifoldScanner
from src.experimental.resonance_core import QJLProjector


# The experimental source directory always exists and has multiple .py files
_EXPERIMENTAL_DIR = str(
    Path(__file__).resolve().parent.parent.parent / "src" / "experimental"
)


@pytest.fixture
def projector():
    return QJLProjector(input_dim=64, projection_dim=128, seed=42)


@pytest.fixture
def scanner(projector):
    return ManifoldScanner(projector, state_dim=64)


# ---------------------------------------------------------------------------
# ManifoldScanner
# ---------------------------------------------------------------------------

class TestManifoldScanner:
    def test_scan_finds_multiple_files(self, scanner):
        scanner.scan(_EXPERIMENTAL_DIR)
        assert len(scanner.file_summary) >= 3  # at least resonance_core, code_transformer, etc.

    def test_scan_returns_chunks(self, scanner):
        chunks = scanner.scan(_EXPERIMENTAL_DIR)
        assert len(chunks) > 10  # experimental dir has many functions/classes

    def test_chunks_have_file_path(self, scanner):
        scanner.scan(_EXPERIMENTAL_DIR)
        for chunk in scanner.chunks:
            assert chunk.file_path is not None
            assert chunk.file_path.endswith(".py")

    def test_chunks_have_projected_state(self, scanner):
        scanner.scan(_EXPERIMENTAL_DIR)
        for chunk in scanner.chunks:
            assert chunk.projected_state is not None
            assert chunk.projected_state.dtype == np.uint8

    def test_chunks_have_feature_vector(self, scanner):
        scanner.scan(_EXPERIMENTAL_DIR)
        for chunk in scanner.chunks:
            assert chunk.feature_vector is not None
            assert len(chunk.feature_vector) == 64

    def test_exclude_patterns(self, projector):
        scanner = ManifoldScanner(
            projector, state_dim=64,
            exclude_patterns=["demo", "__init__"],
        )
        scanner.scan(_EXPERIMENTAL_DIR)
        for fpath in scanner.file_summary:
            assert "demo" not in Path(fpath).name
            assert "__init__" not in Path(fpath).name

    def test_empty_directory(self, scanner, tmp_path):
        chunks = scanner.scan(str(tmp_path))
        assert chunks == []
        assert scanner.file_summary == {}

    def test_multiple_files_in_summary(self, scanner):
        scanner.scan(_EXPERIMENTAL_DIR)
        # At least 3 .py files with chunks
        assert len(scanner.file_summary) >= 3


class TestCrossFileDissonance:
    def test_returns_cross_file_pairs(self, scanner):
        scanner.scan(_EXPERIMENTAL_DIR)
        pairs = scanner.find_cross_file_dissonance(threshold=0.0)
        if pairs:
            a, b, score = pairs[0]
            assert a.file_path != b.file_path

    def test_scores_in_range(self, scanner):
        scanner.scan(_EXPERIMENTAL_DIR)
        pairs = scanner.find_cross_file_dissonance(threshold=0.0)
        for _, _, score in pairs[:20]:  # check first 20
            assert 0.0 <= score <= 1.0

    def test_sorted_descending(self, scanner):
        scanner.scan(_EXPERIMENTAL_DIR)
        pairs = scanner.find_cross_file_dissonance(threshold=0.0)
        scores = [s for _, _, s in pairs]
        assert scores == sorted(scores, reverse=True)


class TestFileCohesion:
    def test_returns_scores_for_all_files(self, scanner):
        scanner.scan(_EXPERIMENTAL_DIR)
        cohesion = scanner.get_file_cohesion_scores()
        assert len(cohesion) == len(scanner.file_summary)

    def test_scores_in_range(self, scanner):
        scanner.scan(_EXPERIMENTAL_DIR)
        cohesion = scanner.get_file_cohesion_scores()
        for score in cohesion.values():
            assert 0.0 <= score <= 1.0

    def test_single_chunk_file_is_zero(self, scanner):
        """Files with only 1 chunk should have cohesion score 0.0."""
        scanner.scan(_EXPERIMENTAL_DIR)
        cohesion = scanner.get_file_cohesion_scores()
        for fpath, n_chunks in scanner.file_summary.items():
            if n_chunks == 1:
                assert cohesion[fpath] == 0.0


# ---------------------------------------------------------------------------
# CodeSwarm.from_directory
# ---------------------------------------------------------------------------

class TestCodeSwarmFromDirectory:
    def test_creates_swarm(self):
        swarm = CodeSwarm.from_directory(_EXPERIMENTAL_DIR, n_rounds=5)
        assert swarm.controller is not None

    def test_swarm_run_completes(self):
        swarm = CodeSwarm.from_directory(_EXPERIMENTAL_DIR, n_rounds=5)
        swarm.run()
        assert len(swarm.controller.round_history) == 5

    def test_dissonant_chunks_returned(self):
        swarm = CodeSwarm.from_directory(_EXPERIMENTAL_DIR, n_rounds=10)
        swarm.run()
        dissonant = swarm.get_dissonant_chunks(top_k=5)
        assert len(dissonant) <= 5
        assert len(dissonant) > 0

    def test_raises_on_empty_dir(self, tmp_path):
        with pytest.raises(ValueError, match="No code chunks"):
            CodeSwarm.from_directory(str(tmp_path))

    def test_custom_exclude_patterns(self):
        swarm = CodeSwarm.from_directory(
            _EXPERIMENTAL_DIR,
            n_rounds=3,
            exclude_patterns=["demo", "__init__"],
        )
        swarm.run()
        # Should still have chunks from non-excluded files
        assert len(swarm.controller.agents) > 0
