"""Import dependency graph builder and boundary checker."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.parsers.tree_sitter_engine import ImportStmt


@dataclass
class Boundaries:
    """Module boundary rules loaded from boundaries.yaml."""
    modules: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    layers: list[dict[str, str | list[str]]] = field(default_factory=list)

    def get_layer_index(self, layer_name: str) -> int | None:
        """Get the position of a layer in the hierarchy."""
        for i, layer in enumerate(self.layers):
            if layer["name"] == layer_name:
                return i
        return None


class ImportGraph:
    """Directed graph of module dependencies with boundary checking."""

    def __init__(self) -> None:
        self.edges: dict[str, set[str]] = {}

    def add_edge(self, source: str, target: str) -> None:
        """Add a dependency edge from source to target module."""
        if source not in self.edges:
            self.edges[source] = set()
        self.edges[source].add(target)

    def add_imports(self, module: str, imports: list[ImportStmt]) -> None:
        """Add all imports from a module as graph edges."""
        for imp in imports:
            target = self._normalize_module(imp.module_path)
            if target:
                self.add_edge(module, target)

    def detect_circular(self) -> list[tuple[str, str]]:
        """Detect circular dependencies. Returns list of (A, B) cycles."""
        cycles = []
        for source, targets in self.edges.items():
            for target in targets:
                if target in self.edges and source in self.edges[target]:
                    pair = tuple(sorted([source, target]))
                    if pair not in cycles:
                        cycles.append(pair)
        return cycles

    def check_boundary(
        self, source_module: str, target_module: str, boundaries: Boundaries
    ) -> list[str]:
        """Check if an import violates any boundary rules.

        Returns a list of violation descriptions (empty = no violations).
        """
        violations = []

        # Check module-level forbidden imports
        source_key = self._extract_module_key(source_module)
        target_key = self._extract_module_key(target_module)

        if source_key in boundaries.modules:
            rules = boundaries.modules[source_key]
            forbidden = rules.get("forbidden_imports", [])
            allowed = rules.get("allowed_imports", [])

            if target_key in forbidden:
                violations.append(
                    f"Module '{source_key}' is forbidden from importing '{target_key}'"
                )
            elif allowed and target_key not in allowed and target_key != source_key:
                violations.append(
                    f"Module '{source_key}' can only import from {allowed}, "
                    f"not '{target_key}'"
                )

        # Check layer-level rules
        source_layer = self._find_layer(source_module, boundaries)
        target_layer = self._find_layer(target_module, boundaries)

        if source_layer and target_layer and source_layer != target_layer:
            for layer in boundaries.layers:
                if layer["name"] == source_layer:
                    can_import = layer.get("can_import", [])
                    if target_layer not in can_import:
                        violations.append(
                            f"Layer '{source_layer}' cannot import from "
                            f"layer '{target_layer}'"
                        )
                    break

        return violations

    def check_import_depth(self, module_path: str, max_depth: int) -> bool:
        """Check if an import path exceeds maximum depth."""
        parts = module_path.replace("/", ".").split(".")
        return len(parts) > max_depth

    @staticmethod
    def _normalize_module(path: str) -> str:
        """Normalize a module path for graph edges."""
        # Remove relative dots
        path = path.lstrip(".")
        # Normalize separators
        return path.replace("/", ".")

    @staticmethod
    def _extract_module_key(module_path: str) -> str:
        """Extract the top-level module name (e.g., 'src.payments.x' -> 'payments')."""
        parts = module_path.replace("/", ".").split(".")
        # Skip 'src' prefix if present
        if parts and parts[0] == "src":
            parts = parts[1:]
        return parts[0] if parts else module_path

    @staticmethod
    def _find_layer(module_path: str, boundaries: Boundaries) -> str | None:
        """Determine which architectural layer a module belongs to."""
        parts = module_path.replace("/", ".").split(".")
        layer_names = {l["name"] for l in boundaries.layers}
        for part in parts:
            if part in layer_names:
                return part
        return None
