"""Boundary configuration loader."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.parsers.import_graph import Boundaries


def load_boundaries(path: str | Path = ".agentprobe/boundaries.yaml") -> Boundaries:
    """Load module boundaries from a YAML file."""
    path = Path(path)
    if not path.exists():
        return Boundaries()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    return Boundaries(
        modules=raw.get("modules", {}),
        layers=raw.get("layers", []),
    )
