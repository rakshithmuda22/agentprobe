"""
Resonance-Correcting Refactor (RCR) — Proposal Generator
=========================================================

SPECULATIVE: The hypothesis that QJL manifold dissonance reliably identifies
code chunks that benefit from refactoring is UNVALIDATED. The mapping from
"high surprise in projected angular space" to "should be refactored" is the
novel, falsifiable claim.

What IS validated:
  - The 15-feature structural analysis in code_transformer.py captures real
    syntactic properties (line count, cyclomatic proxy, nesting depth, etc.)
  - z-score outlier detection is standard statistics.
  - The refactoring rules (split long functions, flatten nesting, etc.) are
    well-established static analysis heuristics.

What is SPECULATIVE:
  - Using QJL-projected Hamming distance as the trigger for refactoring.
  - The assumption that the codebase "manifold" (mean feature profile)
    represents a healthy target state.
  - The idea that pulling outlier chunks toward the manifold improves code.

This module generates structured RefactorProposal objects with human-readable
suggestions. It does NOT generate git-diffs or modify code directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from src.experimental.code_transformer import CodeChunk


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class RefactorType(Enum):
    """Categories of refactoring suggestions."""
    SPLIT_FUNCTION = auto()     # high cyclomatic complexity
    EXTRACT_METHOD = auto()     # very long function
    FLATTEN_NESTING = auto()    # deep nesting depth
    ADD_DOCUMENTATION = auto()  # low comment density
    REDUCE_PARAMETERS = auto()  # too many parameters
    ADD_ERROR_HANDLING = auto() # no try/except in complex code
    SIMPLIFY_LOGIC = auto()     # high complexity + low token diversity


# Human-readable descriptions for each refactor type
_REFACTOR_DESCRIPTIONS = {
    RefactorType.SPLIT_FUNCTION: (
        "This function has high cyclomatic complexity relative to the codebase. "
        "Consider splitting it into smaller, focused helper functions."
    ),
    RefactorType.EXTRACT_METHOD: (
        "This function is significantly longer than the codebase average. "
        "Consider extracting cohesive blocks into separate methods."
    ),
    RefactorType.FLATTEN_NESTING: (
        "This function has deep nesting relative to the codebase. "
        "Consider using early returns, guard clauses, or extracting nested logic."
    ),
    RefactorType.ADD_DOCUMENTATION: (
        "This function has lower comment density than the codebase average. "
        "Consider adding docstrings or inline comments for complex logic."
    ),
    RefactorType.REDUCE_PARAMETERS: (
        "This function has more parameters than typical in the codebase. "
        "Consider grouping related parameters into a config object or dataclass."
    ),
    RefactorType.ADD_ERROR_HANDLING: (
        "This function has high complexity but no try/except blocks. "
        "Consider adding error handling for failure-prone operations."
    ),
    RefactorType.SIMPLIFY_LOGIC: (
        "This function has high complexity but low token diversity, suggesting "
        "repetitive patterns. Consider refactoring repeated logic into loops or helpers."
    ),
}

# Feature index → RefactorType mapping rules
# Each rule: (feature_index, direction, refactor_type, secondary_condition)
# direction: "high" = z > threshold, "low" = z < -threshold
_REFACTOR_RULES: list[tuple[int, str, RefactorType, tuple[int, str] | None]] = [
    # Feature 3 (cyclomatic proxy) high → SPLIT_FUNCTION
    (3, "high", RefactorType.SPLIT_FUNCTION, None),
    # Feature 1 (line count) high → EXTRACT_METHOD
    (1, "high", RefactorType.EXTRACT_METHOD, None),
    # Feature 5 (nesting depth) high → FLATTEN_NESTING
    (5, "high", RefactorType.FLATTEN_NESTING, None),
    # Feature 9 (comment density) low → ADD_DOCUMENTATION
    (9, "low", RefactorType.ADD_DOCUMENTATION, None),
    # Feature 2 (param count) high → REDUCE_PARAMETERS
    (2, "high", RefactorType.REDUCE_PARAMETERS, None),
    # Feature 3 (cyclomatic) high + Feature 11 (has_try) = 0 → ADD_ERROR_HANDLING
    (3, "high", RefactorType.ADD_ERROR_HANDLING, (11, "low")),
    # Feature 3 (cyclomatic) high + Feature 8 (token diversity) low → SIMPLIFY_LOGIC
    (3, "high", RefactorType.SIMPLIFY_LOGIC, (8, "low")),
]

# Feature names for readable output
_FEATURE_NAMES = [
    "source_length",      # 0
    "line_count",         # 1
    "param_count",        # 2
    "cyclomatic_proxy",   # 3
    "import_count",       # 4
    "nesting_depth",      # 5
    "name_length",        # 6
    "is_snake_case",      # 7
    "token_diversity",    # 8
    "comment_density",    # 9
    "return_count",       # 10
    "has_try_except",     # 11
    "has_with_stmt",      # 12
    "has_for_loop",       # 13
    "has_while_loop",     # 14
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ManifoldStats:
    """Per-feature statistics across all chunks in the codebase.

    These represent the "manifold" — the statistical profile of the
    codebase as a whole. Chunks that deviate significantly from these
    statistics are candidates for refactoring (SPECULATIVE claim).
    """
    mean: NDArray[np.float64]     # shape (n_features,)
    std: NDArray[np.float64]      # shape (n_features,)
    min_val: NDArray[np.float64]  # shape (n_features,)
    max_val: NDArray[np.float64]  # shape (n_features,)
    median: NDArray[np.float64]   # shape (n_features,)
    n_chunks: int = 0


@dataclass
class RefactorProposal:
    """A structured refactoring suggestion for a code chunk.

    This is NOT a git-diff. It provides statistical evidence for why
    a chunk is an outlier and a textual suggestion for improvement.
    """
    chunk: "CodeChunk"
    refactor_type: RefactorType
    confidence: float             # abs(z-score) of the triggering feature
    reason: str                   # human-readable explanation
    feature_index: int            # which of the 15 features triggered this
    feature_name: str             # human-readable feature name
    feature_value: float          # the chunk's value for this feature
    manifold_mean: float          # the codebase average
    suggestion: str               # the refactoring suggestion text


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class RefactorAnalyzer:
    """Analyzes code chunks against the codebase manifold to generate
    refactoring proposals.

    SPECULATIVE: The assumption that manifold outliers benefit from
    refactoring is the novel hypothesis under test.
    """

    def __init__(self, z_threshold: float = 1.5) -> None:
        """Initialize the analyzer.

        Args:
            z_threshold: minimum |z-score| to trigger a proposal.
                1.5 ≈ outside the ~87th percentile in a normal distribution.
        """
        self._z_threshold = z_threshold

    @property
    def z_threshold(self) -> float:
        return self._z_threshold

    def compute_manifold_stats(self, chunks: list["CodeChunk"]) -> ManifoldStats:
        """Compute per-feature statistics across all chunks.

        Args:
            chunks: list of CodeChunks with feature_vector set.

        Returns:
            ManifoldStats with mean, std, min, max, median per feature.

        Raises:
            ValueError: if no chunks have feature vectors.
        """
        vectors = []
        for c in chunks:
            if c.feature_vector is not None:
                vectors.append(c.feature_vector[:15])  # first 15 features only
        if not vectors:
            raise ValueError("No chunks with feature vectors found")

        matrix = np.array(vectors)  # (n, 15)
        return ManifoldStats(
            mean=matrix.mean(axis=0),
            std=matrix.std(axis=0, ddof=1) if len(vectors) > 1 else np.zeros(15),
            min_val=matrix.min(axis=0),
            max_val=matrix.max(axis=0),
            median=np.median(matrix, axis=0),
            n_chunks=len(vectors),
        )

    def analyze_chunk(
        self,
        chunk: "CodeChunk",
        stats: ManifoldStats,
    ) -> list[RefactorProposal]:
        """Analyze a single chunk against the manifold.

        Computes z-scores for each of the 15 features and applies
        rule-based mapping to generate proposals.

        Args:
            chunk: a CodeChunk with feature_vector set.
            stats: manifold statistics from compute_manifold_stats().

        Returns:
            List of RefactorProposal objects, sorted by confidence desc.
        """
        if chunk.feature_vector is None:
            return []

        fv = chunk.feature_vector[:15]
        proposals: list[RefactorProposal] = []

        # Compute z-scores (avoid division by zero with a small epsilon)
        eps = 1e-10
        z_scores = (fv - stats.mean) / (stats.std + eps)

        for feat_idx, direction, refactor_type, secondary in _REFACTOR_RULES:
            if feat_idx >= len(z_scores):
                continue

            z = z_scores[feat_idx]

            # Check primary condition
            if direction == "high" and z < self._z_threshold:
                continue
            if direction == "low" and z > -self._z_threshold:
                continue

            # Check secondary condition if present
            if secondary is not None:
                sec_idx, sec_dir = secondary
                if sec_idx >= len(z_scores):
                    continue
                sec_z = z_scores[sec_idx]
                if sec_dir == "low" and sec_z > -self._z_threshold:
                    # For binary features (like has_try_except), check the
                    # raw value — z-score is unreliable for binary features
                    if sec_idx in (7, 11, 12, 13, 14):
                        if fv[sec_idx] > 0.5:
                            continue
                    else:
                        continue
                if sec_dir == "high" and sec_z < self._z_threshold:
                    continue

            feat_name = (
                _FEATURE_NAMES[feat_idx] if feat_idx < len(_FEATURE_NAMES)
                else f"feature_{feat_idx}"
            )

            reason = (
                f"{chunk.name} has {feat_name}={fv[feat_idx]:.3f} "
                f"(z={z:.2f}, manifold mean={stats.mean[feat_idx]:.3f})"
            )

            proposals.append(RefactorProposal(
                chunk=chunk,
                refactor_type=refactor_type,
                confidence=abs(z),
                reason=reason,
                feature_index=feat_idx,
                feature_name=feat_name,
                feature_value=float(fv[feat_idx]),
                manifold_mean=float(stats.mean[feat_idx]),
                suggestion=_REFACTOR_DESCRIPTIONS[refactor_type],
            ))

        # Sort by confidence descending
        proposals.sort(key=lambda p: p.confidence, reverse=True)
        return proposals

    def analyze_codebase(
        self,
        chunks: list["CodeChunk"],
        max_proposals: int = 20,
    ) -> tuple[ManifoldStats, list[RefactorProposal]]:
        """Analyze all chunks and return the top proposals.

        Args:
            chunks: list of CodeChunks with feature_vector set.
            max_proposals: maximum number of proposals to return.

        Returns:
            Tuple of (ManifoldStats, sorted list of RefactorProposal).
        """
        stats = self.compute_manifold_stats(chunks)
        all_proposals: list[RefactorProposal] = []

        for chunk in chunks:
            all_proposals.extend(self.analyze_chunk(chunk, stats))

        # Deduplicate: keep highest-confidence proposal per (chunk, type)
        seen: set[tuple[str, RefactorType]] = set()
        unique: list[RefactorProposal] = []
        for p in sorted(all_proposals, key=lambda x: x.confidence, reverse=True):
            key = (p.chunk.name, p.refactor_type)
            if key not in seen:
                seen.add(key)
                unique.append(p)

        return stats, unique[:max_proposals]

    def format_proposals(
        self,
        proposals: list[RefactorProposal],
        stats: ManifoldStats | None = None,
    ) -> str:
        """Format proposals as a human-readable report.

        Args:
            proposals: list of RefactorProposal objects.
            stats: optional ManifoldStats for header info.

        Returns:
            Multi-line string report.
        """
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("RESONANCE-CORRECTING REFACTOR PROPOSALS")
        lines.append("(SPECULATIVE: manifold dissonance → refactoring)")
        lines.append("=" * 60)

        if stats is not None:
            lines.append(f"Manifold: {stats.n_chunks} chunks analyzed")
            lines.append("")

        if not proposals:
            lines.append("No refactoring proposals generated.")
            lines.append("All chunks are within normal manifold bounds.")
            return "\n".join(lines)

        for i, p in enumerate(proposals, 1):
            lines.append(f"--- Proposal {i} ---")
            lines.append(f"  Chunk: {p.chunk.name} ({p.chunk.chunk_type})")
            if hasattr(p.chunk, "file_path") and p.chunk.file_path:
                lines.append(f"  File:  {p.chunk.file_path}")
            lines.append(f"  Lines: {p.chunk.line_start}-{p.chunk.line_end}")
            lines.append(f"  Type:  {p.refactor_type.name}")
            lines.append(f"  Confidence: {p.confidence:.2f} (z-score)")
            lines.append(f"  Evidence: {p.reason}")
            lines.append(f"  Suggestion: {p.suggestion}")
            lines.append("")

        return "\n".join(lines)
