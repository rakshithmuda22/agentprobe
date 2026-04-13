"""Pattern Agent — checks PR code against approved style profile conventions."""

from __future__ import annotations

from pathlib import Path

from src.graph.state import AgentProbeState
from src.parsers.diff_parser import parse_diff
from src.parsers.tree_sitter_engine import TreeSitterEngine
from src.profiles.style_generator import (
    check_forbidden_patterns,
    check_import_order,
    check_name_convention,
    detect_case,
    detect_import_category,
    load_style_profile,
)


def _check_filename_convention(file_path: str, expected: str) -> dict | None:
    """Check if a filename follows the expected naming convention."""
    stem = Path(file_path).stem
    # Skip __init__ and similar special files
    if stem.startswith("__") and stem.endswith("__"):
        return None
    actual = detect_case(stem)
    if actual != expected:
        return {
            "file_path": file_path,
            "line_number": 0,
            "rule_violated": "naming.files",
            "expected": expected,
            "actual": actual,
            "severity": "WARN",
        }
    return None


def run(state: AgentProbeState) -> AgentProbeState:
    """Check PR diff against style-profile.yaml conventions.

    Deterministic checks — no LLM calls:
    - Function naming conventions
    - Class naming conventions
    - File naming conventions
    - Import ordering
    - Forbidden patterns
    """
    pr_diff = state.get("pr_diff", "")
    repo_path = state.get("repo_path", ".")

    profile = load_style_profile(Path(repo_path) / ".agentprobe" / "style-profile.yaml")
    if not profile:
        # No style profile — nothing to check
        state["pattern_report"] = {
            "status": "PASS",
            "violations": [],
            "confidence": 1.0,
        }
        return state

    naming = profile.get("naming", {})
    import_rules = profile.get("imports", {})
    forbidden = profile.get("forbidden", [])

    diff_result = parse_diff(pr_diff)
    engine = TreeSitterEngine()
    violations = []

    for file_change in diff_result.files:
        language = engine.detect_language(file_change.file_path)
        if language is None:
            continue

        # Check filename convention
        file_convention = naming.get("files")
        if file_convention:
            v = _check_filename_convention(file_change.file_path, file_convention)
            if v:
                violations.append(v)

        # Reconstruct added source
        added_source = "\n".join(line for _, line in file_change.added_lines)
        if not added_source.strip():
            continue

        # Check function naming
        func_convention = naming.get("functions")
        if func_convention:
            try:
                functions = engine.extract_functions(added_source, language)
                for func in functions:
                    if not check_name_convention(func.name, func_convention):
                        violations.append({
                            "file_path": file_change.file_path,
                            "line_number": func.line_start,
                            "rule_violated": "naming.functions",
                            "expected": func_convention,
                            "actual": detect_case(func.name),
                            "severity": "WARN",
                        })
            except Exception:
                pass

        # Check class naming
        class_convention = naming.get("classes")
        if class_convention:
            try:
                classes = engine.extract_classes(added_source, language)
                for cls in classes:
                    if not check_name_convention(cls.name, class_convention):
                        violations.append({
                            "file_path": file_change.file_path,
                            "line_number": cls.line_start,
                            "rule_violated": "naming.classes",
                            "expected": class_convention,
                            "actual": detect_case(cls.name),
                            "severity": "WARN",
                        })
            except Exception:
                pass

        # Check import ordering
        expected_order = import_rules.get("order", [])
        if expected_order:
            try:
                imports = engine.extract_imports(added_source, language)
                if imports:
                    categories = [detect_import_category(imp.module_path) for imp in imports]
                    bad_indices = check_import_order(categories, expected_order)
                    for idx in bad_indices:
                        imp = imports[idx]
                        violations.append({
                            "file_path": file_change.file_path,
                            "line_number": imp.line_number,
                            "rule_violated": "imports.order",
                            "expected": " → ".join(expected_order),
                            "actual": f"'{imp.module_path}' ({categories[idx]}) out of order",
                            "severity": "WARN",
                        })
            except Exception:
                pass

        # Check forbidden patterns
        if forbidden:
            matches = check_forbidden_patterns(added_source, forbidden)
            for match in matches:
                violations.append({
                    "file_path": file_change.file_path,
                    "line_number": match["line_number"],
                    "rule_violated": f"forbidden.{match['pattern']}",
                    "expected": "Pattern not present",
                    "actual": match["line_content"],
                    "severity": "WARN",
                })

    status = "WARN" if violations else "PASS"
    confidence = 1.0  # All checks are deterministic

    state["pattern_report"] = {
        "status": status,
        "violations": violations,
        "confidence": confidence,
    }
    return state
