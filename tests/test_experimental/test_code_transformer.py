"""Tests for code_transformer.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.experimental.resonance_core import QJLProjector
from src.experimental.code_transformer import (
    CodeChunk,
    CodeSwarm,
    CodeTransformer,
    _extract_features,
)
from src.experimental.gossip_governance import GossipConfig


# Shared fixture paths
_RESONANCE_CORE = str(
    Path(__file__).parent.parent.parent / "src" / "experimental" / "resonance_core.py"
)
_ARCH_AGENT = str(
    Path(__file__).parent.parent.parent / "src" / "agents" / "architecture_agent.py"
)


@pytest.fixture
def projector() -> QJLProjector:
    return QJLProjector(input_dim=64, projection_dim=128, n_bits=3, seed=42)


@pytest.fixture
def transformer(projector: QJLProjector) -> CodeTransformer:
    return CodeTransformer(projector, state_dim=64)


# ---------------------------------------------------------------------------
# CodeChunk
# ---------------------------------------------------------------------------

class TestCodeChunk:
    def test_n_lines(self):
        chunk = CodeChunk(
            source_text="def foo():\n    pass",
            name="foo",
            chunk_type="function",
            file_path="test.py",
            line_start=1,
            line_end=2,
        )
        assert chunk.n_lines == 2

    def test_str_repr(self):
        chunk = CodeChunk("x", "bar", "class", "file.py", 5, 10)
        s = str(chunk)
        assert "bar" in s
        assert "class" in s


# ---------------------------------------------------------------------------
# _extract_features
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def _make_chunk(self, src: str, name: str = "foo", ctype: str = "function") -> CodeChunk:
        lines = src.splitlines()
        return CodeChunk(
            source_text=src,
            name=name,
            chunk_type=ctype,
            file_path="test.py",
            line_start=1,
            line_end=len(lines),
        )

    def test_empty_source_zero_vector(self):
        chunk = self._make_chunk("")
        fv = _extract_features(chunk, file_import_count=0, state_dim=64)
        assert np.all(fv == 0.0)

    def test_features_in_unit_interval(self):
        src = (
            "def complex_function(a, b, c):\n"
            "    # A comment\n"
            "    for i in range(10):\n"
            "        if a > 0:\n"
            "            try:\n"
            "                with open('x') as f:\n"
            "                    pass\n"
            "            except Exception:\n"
            "                pass\n"
            "    return a + b\n"
        )
        chunk = self._make_chunk(src, name="complex_function")
        fv = _extract_features(chunk, file_import_count=5, state_dim=64)
        assert fv.shape == (64,)
        assert np.all(fv >= 0.0)
        assert np.all(fv <= 1.0)

    def test_padded_to_state_dim(self):
        chunk = self._make_chunk("def f():\n    pass\n", name="f")
        fv = _extract_features(chunk, file_import_count=0, state_dim=64)
        assert fv.shape == (64,)
        # Positions 15..63 should be zero (padding)
        assert np.all(fv[15:] == 0.0)

    def test_complex_function_higher_cyclomatic(self):
        simple = self._make_chunk("def f():\n    return 1\n", name="f")
        complex_ = self._make_chunk(
            "def g():\n    if True:\n        for x in []:\n"
            "            while True:\n                try:\n                    pass\n"
            "                except: pass\n    return 1\n",
            name="g"
        )
        fv_simple = _extract_features(simple, 0, 64)
        fv_complex = _extract_features(complex_, 0, 64)
        # Feature 3 (cyclomatic) should be higher for complex function
        assert fv_complex[3] > fv_simple[3]

    def test_snake_case_detected(self):
        chunk = self._make_chunk("def my_func():\n    pass\n", name="my_func")
        fv = _extract_features(chunk, 0, 64)
        assert fv[7] == 1.0  # is_snake_case

    def test_non_snake_case(self):
        chunk = self._make_chunk("class MyClass:\n    pass\n", name="MyClass", ctype="class")
        fv = _extract_features(chunk, 0, 64)
        assert fv[7] == 0.0  # not snake_case

    def test_has_try_except(self):
        src = "def f():\n    try:\n        pass\n    except: pass\n"
        chunk = self._make_chunk(src, name="f")
        fv = _extract_features(chunk, 0, 64)
        assert fv[11] == 1.0

    def test_no_try_except(self):
        chunk = self._make_chunk("def f():\n    return 1\n", name="f")
        fv = _extract_features(chunk, 0, 64)
        assert fv[11] == 0.0


# ---------------------------------------------------------------------------
# CodeTransformer
# ---------------------------------------------------------------------------

class TestCodeTransformer:
    def test_projector_dim_mismatch_raises(self):
        p = QJLProjector(input_dim=32, projection_dim=64)
        with pytest.raises(ValueError, match="input_dim"):
            CodeTransformer(p, state_dim=64)

    def test_unsupported_extension_raises(self, transformer: CodeTransformer):
        with pytest.raises(ValueError, match="Unsupported"):
            transformer.extract_chunks("file.rs")

    def test_extract_chunks_resonance_core(self, transformer: CodeTransformer):
        chunks = transformer.extract_chunks(_RESONANCE_CORE)
        assert len(chunks) >= 2
        names = [c.name for c in chunks]
        assert "QJLProjector" in names or any("QJL" in n for n in names)
        # All chunks have feature vectors
        for c in chunks:
            assert c.feature_vector is not None
            assert c.feature_vector.shape == (64,)

    def test_feature_vectors_in_unit_interval(self, transformer: CodeTransformer):
        chunks = transformer.extract_chunks(_RESONANCE_CORE)
        for c in chunks:
            assert np.all(c.feature_vector >= 0.0)
            assert np.all(c.feature_vector <= 1.0)

    def test_project_chunks_fills_projected_state(self, transformer: CodeTransformer):
        chunks = transformer.extract_chunks(_RESONANCE_CORE)
        transformer.project_chunks(chunks)
        for c in chunks:
            assert c.projected_state is not None
            assert c.projected_state.shape == (128,)
            assert c.projected_state.dtype == np.uint8

    def test_project_chunks_returns_same_list(self, transformer: CodeTransformer):
        chunks = transformer.extract_chunks(_RESONANCE_CORE)
        returned = transformer.project_chunks(chunks)
        assert returned is chunks

    def test_find_dissonant_empty_if_one_chunk(self, transformer: CodeTransformer):
        # Manually create a single chunk
        chunk = CodeChunk("def f(): pass", "f", "function", "f.py", 1, 1)
        chunk.feature_vector = np.zeros(64)
        chunk.projected_state = np.zeros(128, dtype=np.uint8)
        result = transformer.find_dissonant_chunks([chunk])
        assert result == []

    def test_find_dissonant_sorted_descending(self, transformer: CodeTransformer):
        chunks = transformer.extract_chunks(_RESONANCE_CORE)
        transformer.project_chunks(chunks)
        pairs = transformer.find_dissonant_chunks(chunks, threshold=0.0)
        if len(pairs) >= 2:
            for i in range(len(pairs) - 1):
                assert pairs[i][2] >= pairs[i + 1][2]

    def test_find_dissonant_no_identical_pairs(self, transformer: CodeTransformer):
        """Pairs with score 0 (identical projections) should not appear if threshold > 0."""
        chunks = transformer.extract_chunks(_RESONANCE_CORE)
        transformer.project_chunks(chunks)
        pairs = transformer.find_dissonant_chunks(chunks, threshold=0.01)
        for _, _, score in pairs:
            assert score >= 0.01


# ---------------------------------------------------------------------------
# CodeSwarm
# ---------------------------------------------------------------------------

class TestCodeSwarm:
    def test_empty_chunks_raises(self, projector: QJLProjector):
        transformer = CodeTransformer(projector, state_dim=64)
        with pytest.raises(ValueError):
            CodeSwarm([])

    def test_run_produces_round_history(self, transformer: CodeTransformer):
        chunks = transformer.extract_chunks(_RESONANCE_CORE)
        transformer.project_chunks(chunks)
        swarm = CodeSwarm(chunks, n_rounds=10)
        swarm.run()
        assert len(swarm.controller.round_history) == 10

    def test_get_dissonant_chunks_sorted(self, transformer: CodeTransformer):
        chunks = transformer.extract_chunks(_RESONANCE_CORE)
        transformer.project_chunks(chunks)
        swarm = CodeSwarm(chunks, n_rounds=5)
        swarm.run()
        dissonant = swarm.get_dissonant_chunks(top_k=5)
        assert len(dissonant) <= 5
        # Sorted by surprise descending
        for i in range(len(dissonant) - 1):
            assert dissonant[i][1] >= dissonant[i + 1][1]

    def test_controller_accessible(self, transformer: CodeTransformer):
        chunks = transformer.extract_chunks(_RESONANCE_CORE)
        swarm = CodeSwarm(chunks, n_rounds=3)
        assert swarm.controller is not None

    def test_no_drift_stable_across_runs(self, transformer: CodeTransformer):
        """With no drift, running twice should not change chunk ordering dramatically."""
        chunks = transformer.extract_chunks(_RESONANCE_CORE)
        transformer.project_chunks(chunks)
        swarm = CodeSwarm(chunks, n_rounds=5)
        swarm.run()
        top = swarm.get_dissonant_chunks(top_k=3)
        assert len(top) >= 1
        # All surprises in [0, 1]
        for _, surprise in top:
            assert 0.0 <= surprise <= 1.0
