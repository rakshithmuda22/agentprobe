"""Architecture Agent — enforces module boundaries and layer rules on PRs."""

from __future__ import annotations

from pathlib import Path

from src.graph.state import AgentProbeState
from src.parsers.diff_parser import parse_diff
from src.parsers.tree_sitter_engine import TreeSitterEngine
from src.parsers.import_graph import ImportGraph
from src.profiles.boundary_loader import load_boundaries
from src.config.loader import load_config


def run(state: AgentProbeState) -> AgentProbeState:
    """Check PR diff for architectural boundary violations.

    - Parses all modified files for import statements
    - Checks each import against module boundaries and layer rules
    - Sets short_circuit=True if any FATAL violation is found
    """
    pr_diff = state.get("pr_diff", "")
    repo_path = state.get("repo_path", ".")

    config = load_config(Path(repo_path) / ".agentprobe" / "config.yaml")
    boundaries = load_boundaries(Path(repo_path) / ".agentprobe" / "boundaries.yaml")

    diff_result = parse_diff(pr_diff)
    engine = TreeSitterEngine()
    graph = ImportGraph()

    violations = []
    max_depth = 3  # default from style profile

    for file_change in diff_result.files:
        language = engine.detect_language(file_change.file_path)
        if language is None:
            continue

        # Extract imports from added lines (new code in the PR)
        added_source = "\n".join(line for _, line in file_change.added_lines)
        if not added_source.strip():
            continue

        try:
            imports = engine.extract_imports(added_source, language)
        except Exception:
            continue

        source_module = file_change.file_path.replace("/", ".").removesuffix(f".{language}")
        if source_module.endswith(".py"):
            source_module = source_module[:-3]

        graph.add_imports(source_module, imports)

        # Check each import against boundaries
        for imp in imports:
            target = ImportGraph._normalize_module(imp.module_path)

            # Check boundary violations
            boundary_violations = graph.check_boundary(source_module, target, boundaries)
            for desc in boundary_violations:
                violations.append({
                    "file": file_change.file_path,
                    "line": imp.line_number,
                    "import": imp.module_path,
                    "severity": "FATAL",
                    "description": desc,
                })

            # Check import depth
            if graph.check_import_depth(imp.module_path, max_depth):
                violations.append({
                    "file": file_change.file_path,
                    "line": imp.line_number,
                    "import": imp.module_path,
                    "severity": "WARN",
                    "description": f"Import depth exceeds maximum of {max_depth}",
                })

    # Determine status
    has_fatal = any(v["severity"] == "FATAL" for v in violations)
    if has_fatal:
        status = "FATAL"
    elif violations:
        status = "WARN"
    else:
        status = "PASS"

    confidence = 1.0 if not violations else 0.95

    state["architecture_report"] = {
        "status": status,
        "violations": violations,
        "confidence": confidence,
    }
    state["short_circuit"] = has_fatal

    return state
