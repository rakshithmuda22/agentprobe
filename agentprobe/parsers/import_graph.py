"""Import dependency graph builder and boundary checker."""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from agentprobe.parsers.tree_sitter_engine import ImportStmt

_JS_LANGUAGES = {"typescript", "tsx", "javascript"}
_JS_SUFFIXES = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")


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


@dataclass
class ResolvedImport:
    """An import resolved to a dotted module path relative to the repo root."""
    module: str          # e.g. "src.payments.models"
    relative: bool       # written as a relative import (always first-party)


def module_of_file(file_path: str) -> str:
    """'src/payments/refunds.py' -> 'src.payments.refunds' ('__init__'/'index' -> package)."""
    parts = list(PurePosixPath(file_path).with_suffix("").parts)
    if len(parts) > 1 and parts[-1] in ("__init__", "index"):
        parts = parts[:-1]
    return ".".join(parts)


def resolve_import(module_path: str, file_path: str, language: str) -> ResolvedImport:
    """Resolve an import as written in ``file_path`` to a repo-relative dotted module.

    Python: ``from .models import X`` in src/payments/refunds.py -> src.payments.models
    JS/TS:  ``'../analytics/tracker'`` in src/payments/refunds.ts -> src.analytics.tracker
    Anything else (absolute / package imports) is returned unchanged, dotted.
    """
    if language == "python" and module_path.startswith("."):
        dots = len(module_path) - len(module_path.lstrip("."))
        rest = module_path[dots:]
        package = list(PurePosixPath(file_path).parent.parts)
        up = dots - 1  # one dot = the file's own package
        base = package[: len(package) - up] if up <= len(package) else []
        parts = base + (rest.split(".") if rest else [])
        return ResolvedImport(".".join(p for p in parts if p), relative=True)

    if language in _JS_LANGUAGES and module_path.startswith("."):
        joined = posixpath.normpath(posixpath.join(posixpath.dirname(file_path), module_path))
        for suffix in _JS_SUFFIXES:
            if joined.endswith(suffix):
                joined = joined[: -len(suffix)]
                break
        if joined.endswith("/index"):
            joined = joined[: -len("/index")]
        return ResolvedImport(joined.strip("/").replace("/", "."), relative=True)

    return ResolvedImport(module_path.replace("/", "."), relative=False)


def _canonical(module: str) -> str:
    """Drop a leading 'src.' so 'src.api.x' and 'api.x' compare equal."""
    return module[4:] if module.startswith("src.") else module


def _prefix_to_module(prefix: str) -> str:
    """'src/api/' -> 'api', 'pkg/cli.py' -> 'pkg.cli'."""
    prefix = str(prefix).strip("/")
    for suffix in (".py",) + _JS_SUFFIXES + (".go",):
        if prefix.endswith(suffix):
            prefix = prefix[: -len(suffix)]
            break
    return _canonical(prefix.replace("/", "."))


def _under(module: str, prefix_module: str) -> bool:
    return bool(prefix_module) and (module == prefix_module
                                    or module.startswith(prefix_module + "."))


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

    @classmethod
    def is_internal(
        cls,
        resolved: ResolvedImport,
        boundaries: Boundaries,
        repo_root: str | Path | None = None,
    ) -> bool:
        """Is this import first-party code (vs. stdlib / third-party)?

        First-party if it was written relatively, lives under ``src.``, names a
        module or layer declared in boundaries.yaml, or its top-level package
        exists in the checked-out repo.
        """
        if resolved.relative:
            return True
        module = resolved.module
        if not module:
            return False
        if module.split(".")[0] == "src":
            return True
        if cls.module_key(module, boundaries) in (boundaries.modules or {}):
            return True
        if cls._find_layer(module, boundaries, by_name=False):
            return True
        if repo_root:
            top = module.split(".")[0]
            root = Path(repo_root)
            for base in (root, root / "src"):
                if (base / top).is_dir() or (base / f"{top}.py").is_file():
                    return True
        return False

    def check_boundary(
        self,
        source_module: str,
        target_module: str,
        boundaries: Boundaries,
        internal: bool = True,
    ) -> list[str]:
        """Check if an import violates any boundary rules.

        ``forbidden_imports`` applies to every import (so you can forbid a
        third-party package too). ``allowed_imports`` is an allow-list for
        first-party modules only, so stdlib/third-party imports never trip it.
        Returns a list of violation descriptions (empty = no violations).
        """
        violations = []

        source_key = self.module_key(source_module, boundaries)
        target_key = self.module_key(target_module, boundaries)

        modules = boundaries.modules or {}
        if source_key in modules and target_key and target_key != source_key:
            rules = modules[source_key] or {}
            forbidden = rules.get("forbidden_imports", []) or []
            allowed = rules.get("allowed_imports", []) or []

            if target_key in forbidden:
                violations.append(
                    f"Module '{source_key}' is forbidden from importing '{target_key}'"
                )
            elif internal and allowed and target_key not in allowed:
                violations.append(
                    f"Module '{source_key}' can only import from {allowed}, "
                    f"not '{target_key}'"
                )

        # Layer rules apply to first-party code only.
        if internal:
            source_layer = self._find_layer(source_module, boundaries)
            target_layer = self._find_layer(target_module, boundaries)

            if source_layer and target_layer and source_layer != target_layer:
                for layer in boundaries.layers:
                    if layer["name"] == source_layer:
                        can_import = layer.get("can_import", []) or []
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

    @classmethod
    def module_key(cls, module_path: str, boundaries: Boundaries) -> str:
        """Which boundaries.yaml module a dotted path belongs to.

        A module may declare ``paths`` (e.g. ``[src/myapp/payments]``);
        otherwise the module is the first directory under ``src/`` (or the
        repo root): 'src.payments.x' -> 'payments'.
        """
        canon = _canonical(module_path.replace("/", "."))
        for name, rules in (boundaries.modules or {}).items():
            for prefix in (rules or {}).get("paths", []) or []:
                if _under(canon, _prefix_to_module(prefix)):
                    return name
        return cls._extract_module_key(module_path)

    @staticmethod
    def _extract_module_key(module_path: str) -> str:
        """Extract the top-level module name (e.g., 'src.payments.x' -> 'payments')."""
        parts = [p for p in module_path.replace("/", ".").split(".") if p]
        # Skip 'src' prefix if present
        if parts and parts[0] == "src":
            parts = parts[1:]
        return parts[0] if parts else ""

    @staticmethod
    def _find_layer(module_path: str, boundaries: Boundaries, by_name: bool = True) -> str | None:
        """Determine which architectural layer a module belongs to.

        A layer may list ``paths`` (alias: ``modules``) such as ``src/api``;
        a module belongs to the layer if it lives under one of them. Layers
        without paths fall back to matching the layer name as a path segment
        (e.g. ``src/domain/...`` is in layer ``domain``).
        """
        module = _canonical(module_path.replace("/", "."))
        for layer in boundaries.layers:
            prefixes = layer.get("paths") or layer.get("modules") or []
            for prefix in prefixes:
                if _under(module, _prefix_to_module(prefix)):
                    return layer["name"]
        if not by_name:
            return None
        parts = module.split(".")
        for layer in boundaries.layers:
            if layer.get("paths") or layer.get("modules"):
                continue
            if layer["name"] in parts:
                return layer["name"]
        return None
