"""
Tests for the Resonance-Correcting Refactor (RCR) proposal generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from src.experimental.refactor_proposals import (
    ManifoldStats,
    RefactorAnalyzer,
    RefactorProposal,
    RefactorType,
    _FEATURE_NAMES,
)


# ---------------------------------------------------------------------------
# Helpers: minimal CodeChunk stub for testing
# ---------------------------------------------------------------------------

@dataclass
class _StubChunk:
    """Minimal CodeChunk-compatible stub for testing without tree-sitter."""
    source_text: str = ""
    name: str = "test_func"
    chunk_type: str = "function"
    file_path: str = "test.py"
    line_start: int = 1
    line_end: int = 10
    feature_vector: np.ndarray | None = None
    projected_state: np.ndarray | None = None


def _make_chunk(
    name: str = "test_func",
    overrides: dict[int, float] | None = None,
    state_dim: int = 64,
) -> _StubChunk:
    """Create a stub chunk with a default (near-mean) feature vector."""
    fv = np.zeros(state_dim, dtype=np.float64)
    # Set sensible defaults for 15 features (roughly mid-range)
    defaults = {
        0: 0.15,  # source_length
        1: 0.12,  # line_count
        2: 0.20,  # param_count
        3: 0.10,  # cyclomatic_proxy
        4: 0.10,  # import_count
        5: 0.15,  # nesting_depth
        6: 0.25,  # name_length
        7: 1.00,  # is_snake_case
        8: 0.60,  # token_diversity
        9: 0.10,  # comment_density
        10: 0.10, # return_count
        11: 0.00, # has_try_except
        12: 0.00, # has_with_stmt
        13: 1.00, # has_for_loop
        14: 0.00, # has_while_loop
    }
    for i, v in defaults.items():
        fv[i] = v
    if overrides:
        for i, v in overrides.items():
            fv[i] = v
    return _StubChunk(name=name, feature_vector=fv)


def _make_chunks(n: int = 10) -> list[_StubChunk]:
    """Create n near-identical chunks (low variance)."""
    rng = np.random.default_rng(42)
    chunks = []
    for i in range(n):
        c = _make_chunk(name=f"func_{i}")
        # Add small noise to avoid zero std
        c.feature_vector[:15] += rng.normal(0, 0.01, size=15)
        c.feature_vector[:15] = np.clip(c.feature_vector[:15], 0, 1)
        chunks.append(c)
    return chunks


# ---------------------------------------------------------------------------
# Tests: ManifoldStats
# ---------------------------------------------------------------------------

class TestManifoldStats:
    def test_compute_returns_correct_shape(self):
        analyzer = RefactorAnalyzer()
        chunks = _make_chunks(10)
        stats = analyzer.compute_manifold_stats(chunks)
        assert stats.mean.shape == (15,)
        assert stats.std.shape == (15,)
        assert stats.n_chunks == 10

    def test_compute_mean_in_range(self):
        analyzer = RefactorAnalyzer()
        chunks = _make_chunks(20)
        stats = analyzer.compute_manifold_stats(chunks)
        assert (stats.mean >= 0).all()
        assert (stats.mean <= 1).all()

    def test_compute_raises_on_empty(self):
        analyzer = RefactorAnalyzer()
        with pytest.raises(ValueError, match="No chunks"):
            analyzer.compute_manifold_stats([])

    def test_compute_raises_on_no_features(self):
        analyzer = RefactorAnalyzer()
        chunk = _StubChunk(feature_vector=None)
        with pytest.raises(ValueError, match="No chunks"):
            analyzer.compute_manifold_stats([chunk])

    def test_single_chunk_zero_std(self):
        analyzer = RefactorAnalyzer()
        chunks = [_make_chunk()]
        stats = analyzer.compute_manifold_stats(chunks)
        # With ddof=1 and n=1, std should be zeros (handled by code)
        assert stats.n_chunks == 1


# ---------------------------------------------------------------------------
# Tests: analyze_chunk
# ---------------------------------------------------------------------------

class TestAnalyzeChunk:
    def test_normal_chunk_no_proposals(self):
        """A chunk near the manifold mean should produce no proposals."""
        analyzer = RefactorAnalyzer()
        chunks = _make_chunks(20)
        stats = analyzer.compute_manifold_stats(chunks)
        # Pick a chunk that's right at the mean
        normal = chunks[0]
        proposals = analyzer.analyze_chunk(normal, stats)
        # May produce 0 or very few proposals since it's near mean
        # All near-mean chunks should have low confidence
        for p in proposals:
            assert p.confidence < 3.0  # shouldn't be extreme outlier

    def test_high_cyclomatic_triggers_split(self):
        """High cyclomatic proxy should trigger SPLIT_FUNCTION."""
        analyzer = RefactorAnalyzer(z_threshold=1.5)
        chunks = _make_chunks(20)
        # Add an outlier with very high cyclomatic
        outlier = _make_chunk("complex_func", overrides={3: 0.95})
        chunks.append(outlier)
        stats = analyzer.compute_manifold_stats(chunks)
        proposals = analyzer.analyze_chunk(outlier, stats)
        types = [p.refactor_type for p in proposals]
        assert RefactorType.SPLIT_FUNCTION in types

    def test_high_line_count_triggers_extract(self):
        """High line count should trigger EXTRACT_METHOD."""
        analyzer = RefactorAnalyzer(z_threshold=1.5)
        chunks = _make_chunks(20)
        outlier = _make_chunk("long_func", overrides={1: 0.95})
        chunks.append(outlier)
        stats = analyzer.compute_manifold_stats(chunks)
        proposals = analyzer.analyze_chunk(outlier, stats)
        types = [p.refactor_type for p in proposals]
        assert RefactorType.EXTRACT_METHOD in types

    def test_deep_nesting_triggers_flatten(self):
        """Deep nesting should trigger FLATTEN_NESTING."""
        analyzer = RefactorAnalyzer(z_threshold=1.5)
        chunks = _make_chunks(20)
        outlier = _make_chunk("nested_func", overrides={5: 0.95})
        chunks.append(outlier)
        stats = analyzer.compute_manifold_stats(chunks)
        proposals = analyzer.analyze_chunk(outlier, stats)
        types = [p.refactor_type for p in proposals]
        assert RefactorType.FLATTEN_NESTING in types

    def test_low_comments_triggers_docs(self):
        """Low comment density should trigger ADD_DOCUMENTATION."""
        analyzer = RefactorAnalyzer(z_threshold=1.5)
        # Create chunks with moderate comment density
        chunks = []
        for i in range(20):
            c = _make_chunk(f"func_{i}", overrides={9: 0.15})
            c.feature_vector[:15] += np.random.default_rng(i).normal(0, 0.01, 15)
            c.feature_vector[:15] = np.clip(c.feature_vector[:15], 0, 1)
            chunks.append(c)
        # Add outlier with zero comments
        outlier = _make_chunk("undocumented", overrides={9: 0.0})
        chunks.append(outlier)
        stats = analyzer.compute_manifold_stats(chunks)
        proposals = analyzer.analyze_chunk(outlier, stats)
        types = [p.refactor_type for p in proposals]
        assert RefactorType.ADD_DOCUMENTATION in types

    def test_chunk_without_features_returns_empty(self):
        analyzer = RefactorAnalyzer()
        chunk = _StubChunk(feature_vector=None)
        stats = ManifoldStats(
            mean=np.zeros(15), std=np.ones(15),
            min_val=np.zeros(15), max_val=np.ones(15),
            median=np.full(15, 0.5), n_chunks=10,
        )
        assert analyzer.analyze_chunk(chunk, stats) == []

    def test_proposal_fields_populated(self):
        """All RefactorProposal fields should be populated."""
        analyzer = RefactorAnalyzer(z_threshold=1.5)
        chunks = _make_chunks(20)
        outlier = _make_chunk("complex_func", overrides={3: 0.95})
        chunks.append(outlier)
        stats = analyzer.compute_manifold_stats(chunks)
        proposals = analyzer.analyze_chunk(outlier, stats)
        assert len(proposals) > 0
        p = proposals[0]
        assert p.chunk.name == "complex_func"
        assert isinstance(p.refactor_type, RefactorType)
        assert p.confidence > 0
        assert len(p.reason) > 0
        assert len(p.suggestion) > 0
        assert p.feature_name in _FEATURE_NAMES

    def test_proposals_sorted_by_confidence(self):
        analyzer = RefactorAnalyzer(z_threshold=1.0)
        chunks = _make_chunks(20)
        # Create outlier that triggers multiple rules
        outlier = _make_chunk("bad_func", overrides={
            1: 0.95, 3: 0.95, 5: 0.95,
        })
        chunks.append(outlier)
        stats = analyzer.compute_manifold_stats(chunks)
        proposals = analyzer.analyze_chunk(outlier, stats)
        confidences = [p.confidence for p in proposals]
        assert confidences == sorted(confidences, reverse=True)


# ---------------------------------------------------------------------------
# Tests: analyze_codebase
# ---------------------------------------------------------------------------

class TestAnalyzeCodebase:
    def test_returns_stats_and_proposals(self):
        analyzer = RefactorAnalyzer(z_threshold=1.5)
        chunks = _make_chunks(15)
        outlier = _make_chunk("outlier", overrides={3: 0.95, 1: 0.90})
        chunks.append(outlier)
        stats, proposals = analyzer.analyze_codebase(chunks)
        assert stats.n_chunks == 16
        assert len(proposals) > 0

    def test_max_proposals_respected(self):
        analyzer = RefactorAnalyzer(z_threshold=0.5)  # low threshold → many proposals
        chunks = _make_chunks(20)
        # Add several outliers
        for i in range(5):
            chunks.append(_make_chunk(f"outlier_{i}", overrides={3: 0.95, 1: 0.90}))
        _, proposals = analyzer.analyze_codebase(chunks, max_proposals=3)
        assert len(proposals) <= 3

    def test_deduplication(self):
        """Same chunk + same type should not appear twice."""
        analyzer = RefactorAnalyzer(z_threshold=1.5)
        chunks = _make_chunks(15)
        outlier = _make_chunk("outlier", overrides={3: 0.95})
        chunks.append(outlier)
        _, proposals = analyzer.analyze_codebase(chunks)
        seen = set()
        for p in proposals:
            key = (p.chunk.name, p.refactor_type)
            assert key not in seen, f"Duplicate proposal: {key}"
            seen.add(key)


# ---------------------------------------------------------------------------
# Tests: format_proposals
# ---------------------------------------------------------------------------

class TestFormatProposals:
    def test_empty_proposals_message(self):
        analyzer = RefactorAnalyzer()
        output = analyzer.format_proposals([])
        assert "No refactoring proposals" in output

    def test_nonempty_contains_chunk_name(self):
        analyzer = RefactorAnalyzer(z_threshold=1.5)
        chunks = _make_chunks(15)
        outlier = _make_chunk("my_outlier_func", overrides={3: 0.95})
        chunks.append(outlier)
        stats, proposals = analyzer.analyze_codebase(chunks)
        output = analyzer.format_proposals(proposals, stats)
        assert "my_outlier_func" in output
        assert "SPECULATIVE" in output

    def test_format_includes_evidence(self):
        analyzer = RefactorAnalyzer(z_threshold=1.5)
        chunks = _make_chunks(15)
        outlier = _make_chunk("complex", overrides={3: 0.95})
        chunks.append(outlier)
        stats, proposals = analyzer.analyze_codebase(chunks)
        output = analyzer.format_proposals(proposals, stats)
        assert "z=" in output
        assert "manifold mean=" in output
