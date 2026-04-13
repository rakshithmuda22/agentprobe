"""
Resonance Telemetry — Visualization and Monitoring
===================================================

ENGINEERING: No speculative components. Provides visualization tools
for resonance data — heatmaps, convergence curves, PID response plots,
feature radar charts.

Matplotlib is OPTIONAL. If not installed, all plot_* methods fall back
to ASCII equivalents printed to stdout. No hard crash.

For large agent counts (N > 100), heatmaps are auto-downsampled to
a 100×100 random subsample with a warning.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from src.experimental.gossip_governance import GossipController
    from src.experimental.refactor_proposals import ManifoldStats

# Lazy matplotlib check
_HAS_MPL = None


def _check_matplotlib() -> bool:
    global _HAS_MPL
    if _HAS_MPL is None:
        try:
            import matplotlib  # noqa: F401
            _HAS_MPL = True
        except ImportError:
            _HAS_MPL = False
    return _HAS_MPL


# ---------------------------------------------------------------------------
# ASCII rendering (always available)
# ---------------------------------------------------------------------------

# Unicode block characters for ASCII heatmap intensity
_BLOCKS = " ░▒▓█"


def ascii_heatmap(
    matrix: NDArray[np.float64],
    labels: list[str] | None = None,
    title: str = "Resonance Heatmap",
) -> str:
    """Render a matrix as an ASCII heatmap using unicode blocks.

    Args:
        matrix: 2D float array in [0, 1].
        labels: optional row/column labels (first 30 chars used).
        title: header string.

    Returns:
        Multi-line string.
    """
    n = matrix.shape[0]
    lines = [title, "=" * len(title)]

    if n > 50:
        lines.append(f"(Showing 50×50 sample of {n}×{n} matrix)")
        idx = np.random.default_rng(42).choice(n, 50, replace=False)
        idx.sort()
        matrix = matrix[np.ix_(idx, idx)]
        if labels:
            labels = [labels[i] for i in idx]
        n = 50

    # Compute label width
    if labels:
        label_width = min(max(len(l) for l in labels), 15)
    else:
        label_width = 0

    for i in range(n):
        row_chars = []
        for j in range(n):
            val = np.clip(matrix[i, j], 0.0, 1.0)
            idx = min(int(val * (len(_BLOCKS) - 1)), len(_BLOCKS) - 1)
            row_chars.append(_BLOCKS[idx])
        prefix = ""
        if labels:
            prefix = labels[i][:label_width].ljust(label_width) + " "
        lines.append(prefix + "".join(row_chars))

    return "\n".join(lines)


def ascii_convergence(
    surprises: list[float],
    title: str = "Convergence",
    width: int = 40,
) -> str:
    """Render a surprise curve as ASCII horizontal bars.

    Args:
        surprises: list of mean surprise values per round.
        title: header string.
        width: max bar width in characters.

    Returns:
        Multi-line string.
    """
    if not surprises:
        return f"{title}\n(no data)"

    max_val = max(surprises) or 1.0
    lines = [title, "-" * (width + 10)]

    step = max(1, len(surprises) // 20)  # show ~20 rows
    for i in range(0, len(surprises), step):
        val = surprises[i]
        bar_len = int(val / max_val * width)
        bar = "#" * bar_len + "." * (width - bar_len)
        lines.append(f"  {i:4d} | {val:.4f} | {bar}")

    return "\n".join(lines)


def ascii_bar(
    values: list[float],
    labels: list[str],
    title: str = "Values",
    width: int = 40,
) -> str:
    """Render labeled values as ASCII horizontal bars.

    Args:
        values: numeric values.
        labels: corresponding labels.
        title: header string.
        width: max bar width.

    Returns:
        Multi-line string.
    """
    if not values:
        return f"{title}\n(no data)"

    max_val = max(abs(v) for v in values) or 1.0
    label_width = min(max(len(l) for l in labels), 20)
    lines = [title, "-" * (label_width + width + 15)]

    for label, val in zip(labels, values):
        bar_len = int(abs(val) / max_val * width)
        bar = "#" * bar_len
        lines.append(f"  {label[:label_width].ljust(label_width)} | {val:.4f} | {bar}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Matplotlib rendering (optional)
# ---------------------------------------------------------------------------

def _save_or_show(fig, save_path: str | None) -> None:
    """Save figure to file or show interactively."""
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    else:
        import matplotlib.pyplot as plt
        plt.show()
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# ResonanceTelemetry
# ---------------------------------------------------------------------------

class ResonanceTelemetry:
    """Visualization tools for resonance data.

    All plot_* methods check for matplotlib at call time. If absent,
    they print an ASCII fallback and return without error.
    """

    def __init__(self) -> None:
        self._has_mpl = _check_matplotlib()

    @property
    def has_matplotlib(self) -> bool:
        return self._has_mpl

    # ----- Heatmap -----

    def plot_resonance_heatmap(
        self,
        matrix: NDArray[np.float64],
        labels: list[str] | None = None,
        title: str = "Pairwise Resonance Heatmap",
        save_path: str | None = None,
    ) -> str | None:
        """Plot or print a resonance heatmap.

        For N > 100, auto-downsamples to 100×100.

        Returns:
            ASCII string if matplotlib unavailable, else None.
        """
        n = matrix.shape[0]

        if n > 100:
            import warnings
            warnings.warn(
                f"Downsampling {n}×{n} matrix to 100×100 for display",
                stacklevel=2,
            )
            idx = np.random.default_rng(42).choice(n, 100, replace=False)
            idx.sort()
            matrix = matrix[np.ix_(idx, idx)]
            if labels:
                labels = [labels[i] for i in idx]

        if not self._has_mpl:
            output = ascii_heatmap(matrix, labels, title)
            print(output)
            return output

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=matrix.max() or 1)
        ax.set_title(title)
        fig.colorbar(im, label="Resonance Score")

        if labels and len(labels) <= 30:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=7)

        _save_or_show(fig, save_path)
        return None

    # ----- Convergence -----

    def plot_convergence(
        self,
        controller: "GossipController",
        title: str = "Gossip Convergence",
        save_path: str | None = None,
    ) -> str | None:
        """Plot mean surprise over rounds."""
        history = controller.round_history
        if not history:
            print("  (no round history)")
            return None

        surprises = [s.mean_surprise for s in history]

        if not self._has_mpl:
            output = ascii_convergence(surprises, title)
            print(output)
            return output

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        rounds = list(range(1, len(surprises) + 1))
        ax.plot(rounds, surprises, color="crimson", linewidth=1.5)
        ax.set_xlabel("Round")
        ax.set_ylabel("Mean Surprise")
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

        _save_or_show(fig, save_path)
        return None

    # ----- PID Response -----

    def plot_pid_response(
        self,
        surprises: list[float],
        resync_values: list[float],
        title: str = "PID Response",
        save_path: str | None = None,
    ) -> str | None:
        """Plot surprise + resync_strength on dual axes."""
        if not surprises:
            print("  (no data)")
            return None

        if not self._has_mpl:
            lines = [title, "-" * 40]
            lines.append("  Round | Surprise | Resync")
            step = max(1, len(surprises) // 15)
            for i in range(0, len(surprises), step):
                rs = resync_values[i] if i < len(resync_values) else 0.0
                lines.append(f"  {i:5d} | {surprises[i]:.4f}  | {rs:.4f}")
            output = "\n".join(lines)
            print(output)
            return output

        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(8, 4))
        rounds = list(range(1, len(surprises) + 1))
        ax1.plot(rounds, surprises, color="crimson", label="Mean Surprise")
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Surprise", color="crimson")

        ax2 = ax1.twinx()
        ax2.plot(
            rounds[:len(resync_values)],
            resync_values,
            color="steelblue",
            linestyle="--",
            label="Resync Strength",
        )
        ax2.set_ylabel("Resync Strength", color="steelblue")

        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95))

        _save_or_show(fig, save_path)
        return None

    # ----- Surprise Distribution -----

    def plot_surprise_distribution(
        self,
        agent_surprises: dict[str, float],
        title: str = "Surprise Distribution",
        save_path: str | None = None,
    ) -> str | None:
        """Plot histogram of per-agent surprises."""
        values = list(agent_surprises.values())
        if not values:
            print("  (no data)")
            return None

        if not self._has_mpl:
            # Simple ASCII histogram
            hist, edges = np.histogram(values, bins=10, range=(0, 1))
            lines = [title, "-" * 40]
            for i, count in enumerate(hist):
                bar = "#" * count
                lines.append(f"  {edges[i]:.2f}-{edges[i+1]:.2f} | {bar} ({count})")
            output = "\n".join(lines)
            print(output)
            return output

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(values, bins=20, range=(0, 1), color="coral", edgecolor="black", alpha=0.7)
        ax.set_xlabel("Surprise")
        ax.set_ylabel("Agent Count")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

        _save_or_show(fig, save_path)
        return None

    # ----- Feature Radar -----

    def plot_feature_radar(
        self,
        feature_vector: NDArray[np.float64],
        manifold_mean: NDArray[np.float64] | None = None,
        feature_names: list[str] | None = None,
        title: str = "Feature Radar",
        save_path: str | None = None,
    ) -> str | None:
        """Plot a radar/spider chart of a chunk's 15 features vs manifold mean.

        Args:
            feature_vector: shape (15,) or longer (only first 15 used).
            manifold_mean: optional shape (15,) manifold average to overlay.
            feature_names: optional custom names.
            title: chart title.
            save_path: file path to save (None → show/print).
        """
        n_features = min(15, len(feature_vector))
        fv = feature_vector[:n_features]

        if feature_names is None:
            from src.experimental.refactor_proposals import _FEATURE_NAMES
            feature_names = _FEATURE_NAMES[:n_features]

        if not self._has_mpl:
            lines = [title, "-" * 50]
            lines.append("  Feature              | Value | Mean")
            lines.append("  ---------------------+-------+------")
            for i, (name, val) in enumerate(zip(feature_names, fv)):
                mean_val = f"{manifold_mean[i]:.3f}" if manifold_mean is not None else "  -  "
                lines.append(f"  {name[:20].ljust(20)} | {val:.3f} | {mean_val}")
            output = "\n".join(lines)
            print(output)
            return output

        import matplotlib.pyplot as plt

        angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
        angles += angles[:1]  # close the polygon

        values = fv.tolist() + [fv[0]]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, values, "o-", linewidth=1.5, color="crimson", label="Chunk")
        ax.fill(angles, values, alpha=0.15, color="crimson")

        if manifold_mean is not None:
            mean_vals = manifold_mean[:n_features].tolist() + [manifold_mean[0]]
            ax.plot(angles, mean_vals, "s--", linewidth=1, color="steelblue", label="Manifold Mean")
            ax.fill(angles, mean_vals, alpha=0.1, color="steelblue")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_names, fontsize=7)
        ax.set_title(title, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

        _save_or_show(fig, save_path)
        return None
