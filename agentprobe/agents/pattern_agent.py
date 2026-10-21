"""Pattern analyzer — checks PR code against the style-profile.yaml conventions.

Deterministic: Tree-sitter extracts the functions/classes/imports the PR adds;
names, import order and forbidden regexes are checked against the profile.
"""

from __future__ import annotations

from pathlib import Path

from agentprobe.config.loader import policy_dir
from agentprobe.graph.state import AgentProbeState
from agentprobe.parsers.diff_parser import parse_diff
from agentprobe.parsers.tree_sitter_engine import TreeSitterEngine
from agentprobe.profiles.style_generator import (
    check_forbidden_patterns,
    check_import_order,
    check_name_convention,
    detect_case,
    detect_import_category,
    load_style_profile,
)

_NAMING_KINDS = {"functions", "classes", "files", "constants"}


def _naming_for(naming: dict, language: str) -> dict:
    """Naming rules for one language.

    ``naming`` may be flat (``functions: snake_case``) or keyed by language
    (``python: {functions: snake_case}``, ``typescript: {functions: camelCase}``);
    language-specific keys override the flat defaults.
    """
    rules = {k: v for k, v in (naming or {}).items() if k in _NAMING_KINDS}
    specific = (naming or {}).get(language)
    if isinstance(specific, dict):
        rules.update(specific)
    return rules


def _check_filename_convention(file_path: str, expected: str) -> dict | None:
    """Check if a filename follows the expected naming convention."""
    stem = Path(file_path).stem
    # Skip __init__ and similar special files
    if stem.startswith("__") and stem.endswith("__"):
        return None
    if not check_name_convention(stem, expected):
        actual = detect_case(stem)
        return {
            "file_path": file_path,
            "line_number": 0,
            "rule_violated": "naming.files",
            "expected": expected,
            "actual": actual,
            "severity": "WARN",
            "description": f"File name '{Path(file_path).name}' is {actual}, expected {expected}",
        }
    return None


def run(state: AgentProbeState) -> AgentProbeState:
    """Check PR diff against style-profile.yaml conventions.

    Deterministic checks — no LLM calls:
    - Function / class / file naming conventions
    - Import ordering
    - Forbidden patterns (built-in names or custom regexes)
    """
    pr_diff = state.get("pr_diff", "")

    profile = load_style_profile(policy_dir(state) / "style-profile.yaml")
    if not profile:
        state["pattern_report"] = {
            "status": "PASS",
            "violations": [],
            "confidence": 1.0,
            "note": "no style-profile.yaml; pattern checks skipped",
        }
        return state

    import_rules = profile.get("imports") or {}
    forbidden = profile.get("forbidden") or []

    diff_result = parse_diff(pr_diff)
    engine = TreeSitterEngine()
    violations = []

    for file_change in diff_result.files:
        if file_change.status == "deleted":
            continue
        language = engine.detect_language(file_change.file_path)
        if language is None:
            continue
        naming = _naming_for(profile.get("naming") or {}, language)

        # Only judge the file name when the PR creates the file.
        file_convention = naming.get("files")
        if file_convention and file_change.status == "added":
            v = _check_filename_convention(file_change.file_path, file_convention)
            if v:
                violations.append(v)

        added_source = file_change.added_source()
        if not added_source.strip():
            continue
        view = file_change.code_view(
            ((state.get("file_sources") or {}).get(file_change.file_path) or {}).get("after"))
        real_line = view.to_file_line

        for kind, extract in (("functions", engine.extract_functions),
                              ("classes", engine.extract_classes)):
            convention = naming.get(kind)
            if not convention:
                continue
            try:
                defs = [d for d in extract(view.source, language) if view.is_added(d.line_start)]
            except Exception:
                continue
            for d in defs:
                if d.name.startswith("__") and d.name.endswith("__"):
                    continue  # dunder methods are not a style choice
                if not check_name_convention(d.name, convention):
                    actual = detect_case(d.name)
                    violations.append({
                        "file_path": file_change.file_path,
                        "line_number": real_line(d.line_start),
                        "rule_violated": f"naming.{kind}",
                        "expected": convention,
                        "actual": actual,
                        "severity": "WARN",
                        "description": f"{kind[:-1].capitalize()} '{d.name}' is {actual}, "
                                       f"expected {convention}",
                    })

        expected_order = import_rules.get("order", [])
        if expected_order and language == "python":
            try:
                imports = [imp for imp in engine.extract_imports(view.source, language)
                           if view.is_added(imp.line_number) and imp.top_level]
            except Exception:
                imports = []
            if imports:
                categories = [detect_import_category(imp.module_path) for imp in imports]
                for idx in check_import_order(categories, expected_order):
                    imp = imports[idx]
                    violations.append({
                        "file_path": file_change.file_path,
                        "line_number": real_line(imp.line_number),
                        "rule_violated": "imports.order",
                        "expected": " → ".join(expected_order),
                        "actual": f"'{imp.module_path}' ({categories[idx]}) out of order",
                        "severity": "WARN",
                        "description": f"Import '{imp.module_path}' ({categories[idx]}) is out "
                                       f"of order; expected {' → '.join(expected_order)}",
                    })

        if forbidden:
            for match in check_forbidden_patterns(added_source, forbidden, file_change.file_path):
                message = match["message"] or f"Forbidden pattern '{match['pattern']}'"
                violations.append({
                    "file_path": file_change.file_path,
                    "line_number": file_change.added_line_number(match["line_number"]),
                    "rule_violated": f"forbidden.{match['pattern']}",
                    "expected": "Pattern not present",
                    "actual": match["line_content"],
                    "severity": match["severity"],
                    "description": f"{message}: `{match['line_content'][:80]}`",
                })

    if any(v["severity"] == "FATAL" for v in violations):
        status = "FATAL"
    elif violations:
        status = "WARN"
    else:
        status = "PASS"

    state["pattern_report"] = {
        "status": status,
        "violations": violations,
        "confidence": 1.0,  # All checks are deterministic
    }
    return state
