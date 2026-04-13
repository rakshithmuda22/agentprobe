"""
CodeTransformer — Source Code Structural Analysis via QJL Manifold
==================================================================

Chunks Python source files into function/class units, extracts 15
deterministic structural features, and projects them into the QJL
manifold for resonance-based similarity analysis.

Features are purely syntactic (no semantics): line counts, keyword
frequencies, nesting depth, token statistics, normalized to [0, 1].
The feature vector is padded with zeros to state_dim.

SPECULATIVE extension:
  - Treating structural feature vectors as "belief vectors" enables
    gossip-based resonance to identify structurally unusual code chunks
    relative to their neighbors in the manifold.

VALIDATED components:
  - Tree-sitter AST extraction (production-grade parser)
  - QJL projection and XOR+popcount resonance (from resonance_core.py)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.experimental.resonance_core import QJLProjector, pairwise_resonance_matrix
from src.experimental.gossip_governance import GossipConfig, GossipController
from src.experimental.kinetic_agent import KineticAgent
from src.parsers.tree_sitter_engine import TreeSitterEngine


# ---------------------------------------------------------------------------
# Feature extraction constants
# ---------------------------------------------------------------------------

# Keywords that increase cyclomatic complexity proxy
_CYCLOMATIC_KEYWORDS = re.compile(
    r"\b(if|elif|else|for|while|try|except|finally|with|and|or)\b"
)
_TOKEN_PATTERN = re.compile(r"\b\w+\b")
_COMMENT_LINE = re.compile(r"^\s*#")


# ---------------------------------------------------------------------------
# CodeChunk dataclass
# ---------------------------------------------------------------------------

@dataclass
class CodeChunk:
    """A single function or class extracted from a source file."""

    source_text: str
    name: str
    chunk_type: str          # "function" | "class"
    file_path: str
    line_start: int
    line_end: int
    feature_vector: NDArray[np.float64] | None = field(default=None, repr=False)
    projected_state: NDArray[np.uint8] | None = field(default=None, repr=False)

    @property
    def n_lines(self) -> int:
        return max(self.line_end - self.line_start + 1, 1)

    def __str__(self) -> str:
        return (
            f"{self.chunk_type}:{self.name} "
            f"[{self.file_path}:{self.line_start}-{self.line_end}]"
        )


# ---------------------------------------------------------------------------
# Feature extraction (15 features, all normalized to [0, 1])
# ---------------------------------------------------------------------------

def _extract_features(
    chunk: CodeChunk,
    file_import_count: int,
    state_dim: int,
) -> NDArray[np.float64]:
    """Compute a 15-element feature vector from a code chunk.

    All values normalized to [0, 1]. Vector is zero-padded to state_dim.

    Features:
        0: source_length        min(len(source) / 2000, 1)
        1: line_count           min(n_lines / 100, 1)
        2: param_count          min(n_params / 10, 1)  [0 for classes]
        3: cyclomatic_proxy     min(keyword_count / 20, 1)
        4: file_import_count    min(n / 30, 1)
        5: max_nesting_depth    min(max_indent_level / 6, 1)
        6: name_length          min(len(name) / 40, 1)
        7: is_snake_case        1.0 if name is snake_case
        8: token_diversity      unique_tokens / total_tokens
        9: comment_density      comment_lines / n_lines
       10: return_count         min(count('return') / 5, 1)
       11: has_try_except       1.0 if 'try:' in source
       12: has_with_stmt        1.0 if 'with ' in source
       13: has_for_loop         1.0 if '\nfor ' or source startswith 'for '
       14: has_while_loop       1.0 if 'while ' in source
    """
    src = chunk.source_text
    if not src.strip():
        return np.zeros(state_dim, dtype=np.float64)

    lines = src.splitlines()
    n_lines = chunk.n_lines

    # --- Feature 0: source length ---
    f0 = min(len(src) / 2000.0, 1.0)

    # --- Feature 1: line count ---
    f1 = min(n_lines / 100.0, 1.0)

    # --- Feature 2: param count (for functions; method count for classes) ---
    if chunk.chunk_type == "function":
        # Count comma-separated params in first line (heuristic)
        first_line = lines[0] if lines else ""
        paren_start = first_line.find("(")
        paren_end = first_line.rfind(")")
        if paren_start != -1 and paren_end > paren_start:
            params_str = first_line[paren_start + 1 : paren_end].strip()
            n_params = len(params_str.split(",")) if params_str else 0
        else:
            n_params = 0
        f2 = min(n_params / 10.0, 1.0)
    else:
        # For classes, use line count as a proxy for complexity
        f2 = min(n_lines / 50.0, 1.0)

    # --- Feature 3: cyclomatic complexity proxy ---
    keyword_count = len(_CYCLOMATIC_KEYWORDS.findall(src))
    f3 = min(keyword_count / 20.0, 1.0)

    # --- Feature 4: file-level import count ---
    f4 = min(file_import_count / 30.0, 1.0)

    # --- Feature 5: max nesting depth (indent level) ---
    max_indent = 0
    for line in lines:
        if line.strip():
            stripped = line.lstrip()
            indent_chars = len(line) - len(stripped)
            indent_level = indent_chars // 4
            max_indent = max(max_indent, indent_level)
    f5 = min(max_indent / 6.0, 1.0)

    # --- Feature 6: name length ---
    f6 = min(len(chunk.name) / 40.0, 1.0)

    # --- Feature 7: is snake_case ---
    name = chunk.name
    f7 = 1.0 if ("_" in name and name == name.lower() and name[0] != "_") else 0.0

    # --- Feature 8: token diversity ---
    tokens = _TOKEN_PATTERN.findall(src)
    if tokens:
        f8 = len(set(tokens)) / len(tokens)
    else:
        f8 = 0.0

    # --- Feature 9: comment density ---
    comment_lines = sum(1 for ln in lines if _COMMENT_LINE.match(ln))
    f9 = min(comment_lines / max(n_lines, 1), 1.0)

    # --- Feature 10: return statement count ---
    return_count = src.count("return ")
    f10 = min(return_count / 5.0, 1.0)

    # --- Feature 11: has try/except ---
    f11 = 1.0 if "try:" in src else 0.0

    # --- Feature 12: has with statement ---
    f12 = 1.0 if "with " in src else 0.0

    # --- Feature 13: has for loop ---
    f13 = 1.0 if ("for " in src) else 0.0

    # --- Feature 14: has while loop ---
    f14 = 1.0 if ("while " in src) else 0.0

    features = np.array(
        [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14],
        dtype=np.float64,
    )

    # Zero-pad to state_dim
    result = np.zeros(state_dim, dtype=np.float64)
    n = min(len(features), state_dim)
    result[:n] = features[:n]
    return result


# ---------------------------------------------------------------------------
# CodeTransformer
# ---------------------------------------------------------------------------

class CodeTransformer:
    """Extract code chunks from source files and project them into QJL manifold.

    Usage:
        projector = QJLProjector(input_dim=64, projection_dim=128, seed=42)
        transformer = CodeTransformer(projector, state_dim=64)
        chunks = transformer.extract_chunks("path/to/file.py")
        transformer.project_chunks(chunks)
        dissonant = transformer.find_dissonant_chunks(chunks, threshold=0.35)
    """

    def __init__(
        self,
        projector: QJLProjector,
        state_dim: int = 64,
    ) -> None:
        if projector.input_dim != state_dim:
            raise ValueError(
                f"projector.input_dim ({projector.input_dim}) must equal "
                f"state_dim ({state_dim})"
            )
        self._projector = projector
        self._state_dim = state_dim
        self._engine = TreeSitterEngine()

    @property
    def projector(self) -> QJLProjector:
        return self._projector

    @property
    def state_dim(self) -> int:
        return self._state_dim

    def extract_chunks(self, file_path: str) -> list[CodeChunk]:
        """Parse file and return CodeChunks with feature vectors.

        Args:
            file_path: path to a Python (or supported language) source file

        Returns:
            List of CodeChunks with feature_vector set; projected_state is None
            until project_chunks() is called.

        Raises:
            ValueError: if file type is unsupported
        """
        language = self._engine.detect_language(file_path)
        if language is None:
            raise ValueError(f"Unsupported file type: {file_path}")

        with open(file_path, encoding="utf-8", errors="replace") as fh:
            source = fh.read()

        source_lines = source.splitlines()

        functions = self._engine.extract_functions(source, language)
        classes = self._engine.extract_classes(source, language)
        imports = self._engine.extract_imports(source, language)
        file_import_count = len(imports)

        chunks: list[CodeChunk] = []

        for fn in functions:
            chunk_lines = source_lines[fn.line_start - 1 : fn.line_end]
            chunk_source = "\n".join(chunk_lines)
            chunk = CodeChunk(
                source_text=chunk_source,
                name=fn.name,
                chunk_type="function",
                file_path=file_path,
                line_start=fn.line_start,
                line_end=fn.line_end,
            )
            chunk.feature_vector = _extract_features(
                chunk, file_import_count, self._state_dim
            ).copy()
            chunks.append(chunk)

        for cls in classes:
            chunk_lines = source_lines[cls.line_start - 1 : cls.line_end]
            chunk_source = "\n".join(chunk_lines)
            chunk = CodeChunk(
                source_text=chunk_source,
                name=cls.name,
                chunk_type="class",
                file_path=file_path,
                line_start=cls.line_start,
                line_end=cls.line_end,
            )
            chunk.feature_vector = _extract_features(
                chunk, file_import_count, self._state_dim
            ).copy()
            chunks.append(chunk)

        return chunks

    def project_chunks(self, chunks: list[CodeChunk]) -> list[CodeChunk]:
        """Fill projected_state for each chunk.

        Args:
            chunks: list from extract_chunks (feature_vector must be set)

        Returns:
            Same list with projected_state filled (mutates in place).
        """
        if not chunks:
            return chunks

        feature_matrix = np.array(
            [c.feature_vector for c in chunks], dtype=np.float64
        )  # (n, state_dim)

        quantized = self._projector.batch_project_and_quantize(feature_matrix)

        for chunk, q in zip(chunks, quantized):
            chunk.projected_state = q

        return chunks

    def find_dissonant_chunks(
        self,
        chunks: list[CodeChunk],
        threshold: float = 0.4,
    ) -> list[tuple[CodeChunk, CodeChunk, float]]:
        """Find structurally dissimilar chunk pairs above the threshold.

        Uses pairwise resonance matrix for efficiency. Only upper-triangle
        pairs (i < j) are returned to avoid duplicates.

        Args:
            chunks: list with projected_state filled
            threshold: resonance score threshold (0=identical, 1=maximally different)

        Returns:
            List of (chunk_a, chunk_b, score) sorted by score descending.
        """
        if len(chunks) < 2:
            return []

        projected = [c.projected_state for c in chunks]
        if any(p is None for p in projected):
            self.project_chunks(chunks)
            projected = [c.projected_state for c in chunks]

        states = np.array(projected, dtype=np.uint8)
        matrix = pairwise_resonance_matrix(states, n_bits=self._projector.n_bits)

        pairs = []
        n = len(chunks)
        for i in range(n):
            for j in range(i + 1, n):
                score = float(matrix[i, j])
                if score >= threshold:
                    pairs.append((chunks[i], chunks[j], score))

        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs


# ---------------------------------------------------------------------------
# CodeSwarm
# ---------------------------------------------------------------------------

class CodeSwarm:
    """Wrap GossipController to govern a set of code chunks as agents.

    Each chunk's feature_vector becomes its agent's initial belief.
    After gossip runs, agents with high mean_surprise represent code
    that is structurally unusual relative to its neighbors.

    SPECULATIVE: this assumes structural feature similarity in QJL space
    correlates with semantic consistency. Not validated beyond the structural
    feature level.
    """

    def __init__(
        self,
        chunks: list[CodeChunk],
        config: GossipConfig | None = None,
        n_rounds: int = 50,
    ) -> None:
        if not chunks:
            raise ValueError("CodeSwarm requires at least one chunk")

        self._chunks = chunks
        self._n_rounds = n_rounds
        state_dim = len(chunks[0].feature_vector) if chunks[0].feature_vector is not None else 64

        # Build config with correct agent count and state_dim
        if config is None:
            cfg = GossipConfig(
                n_agents=len(chunks),
                state_dim=state_dim,
                projection_dim=128,
                fanout=min(3, len(chunks) - 1),
                dissonance_threshold=0.35,
                resync_strength=0.05,
                drift_magnitude=0.0,  # no drift — let structural features speak
                max_rounds=n_rounds,
                seed=42,
            )
        else:
            cfg = config

        self._controller = GossipController(cfg)
        self._controller.initialize()

        # Override each agent's belief with the chunk's feature vector
        for chunk, agent in zip(chunks, self._controller.agents.values()):
            if chunk.feature_vector is not None:
                fv = chunk.feature_vector.copy()
                norm = np.linalg.norm(fv)
                if norm > 0:
                    fv /= norm
                agent.update_belief(fv)

    @property
    def controller(self) -> GossipController:
        return self._controller

    def run(self) -> None:
        """Run gossip for n_rounds."""
        self._controller.run(n_rounds=self._n_rounds)

    def get_dissonant_chunks(
        self, top_k: int = 10
    ) -> list[tuple[CodeChunk, float]]:
        """Return top_k chunks ranked by mean surprise (most unusual first).

        Aggregates agent_surprises across round_history. Chunks whose agents
        consistently surprised their neighbors are structurally unusual.

        Args:
            top_k: number of top results to return

        Returns:
            List of (chunk, mean_surprise) sorted descending.
        """
        history = self._controller.round_history
        if not history:
            return []

        agent_ids = list(self._controller.agents.keys())

        # Average surprise per agent across all rounds
        per_agent: dict[str, list[float]] = {aid: [] for aid in agent_ids}
        for stats in history:
            for aid, surprise in stats.agent_surprises.items():
                if aid in per_agent:
                    per_agent[aid].append(surprise)

        mean_surprises = {
            aid: float(np.mean(vals)) if vals else 0.0
            for aid, vals in per_agent.items()
        }

        # Match agents back to chunks by insertion order
        results = []
        for chunk, aid in zip(self._chunks, agent_ids):
            results.append((chunk, mean_surprises.get(aid, 0.0)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    @classmethod
    def from_directory(
        cls,
        root_dir: str,
        config: GossipConfig | None = None,
        n_rounds: int = 50,
        projector: QJLProjector | None = None,
        state_dim: int = 64,
        exclude_patterns: list[str] | None = None,
    ) -> "CodeSwarm":
        """Create a CodeSwarm from all Python files in a directory.

        Convenience factory that scans a directory tree, extracts chunks
        from all .py files, and builds a CodeSwarm.

        Args:
            root_dir: path to scan recursively for .py files.
            config: optional GossipConfig override.
            n_rounds: gossip rounds (default 50).
            projector: QJLProjector to use; created with default params if None.
            state_dim: state dimension (must match projector.input_dim).
            exclude_patterns: filename substrings to skip (e.g., ["test_", "__pycache__"]).

        Returns:
            CodeSwarm with all extracted chunks.

        Raises:
            ValueError: if no chunks were extracted.
        """
        scanner = ManifoldScanner(
            projector=projector or QJLProjector(input_dim=state_dim, projection_dim=128, seed=42),
            state_dim=state_dim,
            exclude_patterns=exclude_patterns,
        )
        scanner.scan(root_dir)
        if not scanner.chunks:
            raise ValueError(f"No code chunks found in {root_dir}")
        return cls(scanner.chunks, config=config, n_rounds=n_rounds)


# ---------------------------------------------------------------------------
# ManifoldScanner — Multi-file directory scanning
# ---------------------------------------------------------------------------

class ManifoldScanner:
    """Scan a directory tree and extract code chunks into a unified QJL manifold.

    ENGINEERING: no speculative components. Uses existing CodeTransformer
    infrastructure to process multiple files.

    Usage:
        projector = QJLProjector(input_dim=64, projection_dim=128, seed=42)
        scanner = ManifoldScanner(projector, state_dim=64)
        scanner.scan("src/experimental/")
        pairs = scanner.find_cross_file_dissonance(threshold=0.3)
        cohesion = scanner.get_file_cohesion_scores()
    """

    # Directories to always skip
    _SKIP_DIRS = {"__pycache__", ".venv", "node_modules", ".git", ".tox", ".mypy_cache"}

    def __init__(
        self,
        projector: QJLProjector,
        state_dim: int = 64,
        exclude_patterns: list[str] | None = None,
    ) -> None:
        """Initialize the scanner.

        Args:
            projector: shared QJLProjector (all projections must use the same R).
            state_dim: feature vector dimension (must match projector.input_dim).
            exclude_patterns: filename substrings to skip (e.g., ["test_", "conftest"]).
        """
        self._transformer = CodeTransformer(projector, state_dim=state_dim)
        self._exclude_patterns = exclude_patterns or []
        self._chunks: list[CodeChunk] = []
        self._file_summary: dict[str, int] = {}

    @property
    def chunks(self) -> list[CodeChunk]:
        """All extracted chunks across all scanned files."""
        return self._chunks

    @property
    def file_summary(self) -> dict[str, int]:
        """Mapping of file_path → number of chunks extracted."""
        return self._file_summary

    def scan(self, root_dir: str) -> list[CodeChunk]:
        """Walk root_dir recursively, extract chunks from all .py files.

        After scanning, calls project_chunks() so all chunks have
        projected_state set using the shared projector.

        Args:
            root_dir: path to scan.

        Returns:
            All extracted chunks (also stored in self.chunks).
        """
        root = Path(root_dir)
        self._chunks = []
        self._file_summary = {}

        py_files = sorted(root.rglob("*.py"))

        for py_file in py_files:
            # Skip excluded directories
            if any(skip in py_file.parts for skip in self._SKIP_DIRS):
                continue
            # Skip files matching exclude patterns
            if any(pat in py_file.name for pat in self._exclude_patterns):
                continue

            try:
                file_chunks = self._transformer.extract_chunks(str(py_file))
                if file_chunks:
                    self._chunks.extend(file_chunks)
                    self._file_summary[str(py_file)] = len(file_chunks)
            except (ValueError, OSError):
                # Unsupported file type or read error — skip silently
                continue

        # Project all chunks through the shared QJL projector
        if self._chunks:
            self._transformer.project_chunks(self._chunks)

        return self._chunks

    def find_cross_file_dissonance(
        self,
        threshold: float = 0.3,
    ) -> list[tuple[CodeChunk, CodeChunk, float]]:
        """Find structurally dissimilar chunk pairs from DIFFERENT files.

        Only returns pairs where chunk_a.file_path != chunk_b.file_path,
        highlighting cross-file structural inconsistencies.

        Args:
            threshold: minimum resonance score to include.

        Returns:
            List of (chunk_a, chunk_b, score) sorted by score descending.
        """
        all_pairs = self._transformer.find_dissonant_chunks(
            self._chunks, threshold=threshold
        )
        return [
            (a, b, score)
            for a, b, score in all_pairs
            if a.file_path != b.file_path
        ]

    def get_file_cohesion_scores(self) -> dict[str, float]:
        """Compute mean intra-file pairwise resonance for each file.

        Files with high cohesion scores have internally diverse chunks
        (structurally varied functions). Low scores indicate uniform structure.

        A high score is NOT necessarily bad — a utility file with many
        diverse helper functions will naturally score higher than a file
        with many similar getter/setter methods.

        Returns:
            Dict of file_path → mean intra-file resonance score.
        """
        from collections import defaultdict

        file_chunks: dict[str, list[CodeChunk]] = defaultdict(list)
        for chunk in self._chunks:
            file_chunks[chunk.file_path].append(chunk)

        scores: dict[str, float] = {}
        for file_path, chunks in file_chunks.items():
            if len(chunks) < 2:
                scores[file_path] = 0.0
                continue

            projected = np.array(
                [c.projected_state for c in chunks if c.projected_state is not None],
                dtype=np.uint8,
            )
            if len(projected) < 2:
                scores[file_path] = 0.0
                continue

            matrix = pairwise_resonance_matrix(projected)
            # Mean of upper triangle (excluding diagonal)
            n = len(projected)
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            scores[file_path] = float(matrix[mask].mean())

        return scores
