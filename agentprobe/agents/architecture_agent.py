"""Architecture analyzer — enforces module boundaries and layer rules on PRs.

Deterministic: Tree-sitter extracts the imports a PR adds, each import is
resolved to a repo-relative module, and checked against boundaries.yaml.
"""

from __future__ import annotations

from agentprobe.config.loader import DEFAULT_CONFIG_DIR, policy_dir
from agentprobe.graph.state import AgentProbeState
from agentprobe.parsers.diff_parser import parse_diff
from agentprobe.parsers.import_graph import ImportGraph, module_of_file, resolve_import
from agentprobe.parsers.tree_sitter_engine import TreeSitterEngine
from agentprobe.profiles.boundary_loader import load_boundaries
from agentprobe.profiles.style_generator import load_style_profile

POLICY_FILES = ("config.yaml", "boundaries.yaml", "style-profile.yaml")


def run(state: AgentProbeState) -> AgentProbeState:
    """Check PR diff for architectural boundary violations.

    - Extracts import statements from the lines the PR adds
    - Resolves relative imports against the importing file
    - Checks each import against module boundaries and layer rules
    - Sets short_circuit=True if any FATAL violation is found
    """
    pr_diff = state.get("pr_diff", "")
    repo_path = state.get("repo_path")
    policy = policy_dir(state)
    policy_prefix = (state.get("policy_repo_dir") or DEFAULT_CONFIG_DIR).strip("/") + "/"

    boundaries = load_boundaries(policy / "boundaries.yaml")
    style = load_style_profile(policy / "style-profile.yaml")
    max_depth = (style.get("imports") or {}).get("max_depth")

    diff_result = parse_diff(pr_diff)
    engine = TreeSitterEngine()
    graph = ImportGraph()

    violations = []

    for file_change in diff_result.files:
        # A PR that edits the governance policy itself deserves a human look.
        # (The action/CLI judge the PR with the *base* branch's policy.)
        if file_change.file_path.startswith(policy_prefix) and (
            file_change.file_path.rsplit("/", 1)[-1] in POLICY_FILES
        ):
            violations.append({
                "file": file_change.file_path,
                "line": 0,
                "import": "",
                "severity": "WARN",
                "description": "PR changes a governance policy file; it was judged "
                               "with the base branch's policy. Review the rule change.",
            })
            continue

        language = engine.detect_language(file_change.file_path)
        if language is None:
            continue

        view = file_change.code_view(
            ((state.get("file_sources") or {}).get(file_change.file_path) or {}).get("after"))
        if not file_change.added_source().strip():
            continue

        try:
            imports = [imp for imp in engine.extract_imports(view.source, language)
                       if view.is_added(imp.line_number)]
        except Exception:
            continue

        source_module = module_of_file(file_change.file_path)
        graph.add_imports(source_module, imports)

        for imp in imports:
            resolved = resolve_import(imp.module_path, file_change.file_path, language)
            internal = ImportGraph.is_internal(resolved, boundaries, repo_path)
            line = view.to_file_line(imp.line_number)

            for desc in graph.check_boundary(
                source_module, resolved.module, boundaries, internal=internal
            ):
                violations.append({
                    "file": file_change.file_path,
                    "line": line,
                    "import": imp.module_path,
                    "severity": "FATAL",
                    "description": desc,
                })

            # Optional: cap how deep into another package's internals code may reach.
            if max_depth and internal and graph.check_import_depth(resolved.module, max_depth):
                violations.append({
                    "file": file_change.file_path,
                    "line": line,
                    "import": imp.module_path,
                    "severity": "WARN",
                    "description": f"Import depth exceeds maximum of {max_depth}",
                })

    has_fatal = any(v["severity"] == "FATAL" for v in violations)
    if has_fatal:
        status = "FATAL"
    elif violations:
        status = "WARN"
    else:
        status = "PASS"

    state["architecture_report"] = {
        "status": status,
        "violations": violations,
        "confidence": 1.0,  # deterministic
    }
    state["short_circuit"] = has_fatal

    return state
