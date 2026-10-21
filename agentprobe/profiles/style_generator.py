"""Style profile utilities — case detection, import categorization, forbidden rules."""

from __future__ import annotations

import fnmatch
import logging
import re
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


# Known Python stdlib modules (subset covering common ones)
_STDLIB_MODULES = {
    "os", "sys", "re", "json", "math", "time", "datetime", "collections",
    "itertools", "functools", "pathlib", "typing", "abc", "io", "copy",
    "hashlib", "hmac", "secrets", "logging", "unittest", "dataclasses",
    "enum", "contextlib", "subprocess", "threading", "multiprocessing",
    "socket", "http", "urllib", "email", "html", "xml", "csv", "sqlite3",
    "pickle", "shelve", "struct", "codecs", "unicodedata", "textwrap",
    "string", "difflib", "pprint", "traceback", "warnings", "inspect",
    "importlib", "pkgutil", "tempfile", "shutil", "glob", "fnmatch",
    "stat", "fileinput", "argparse", "configparser", "signal", "ast",
    "dis", "token", "tokenize", "pdb", "profile", "cProfile", "timeit",
    "random", "statistics", "fractions", "decimal", "numbers",
    "array", "queue", "heapq", "bisect", "weakref", "types",
    "operator", "concurrent", "asyncio", "selectors", "mmap",
    "platform", "ctypes", "ssl", "base64", "binascii", "zlib", "gzip",
    "bz2", "lzma", "zipfile", "tarfile", "uuid",
}


def detect_case(name: str) -> str:
    """Detect the naming convention of an identifier.

    Returns one of: camelCase, snake_case, PascalCase, UPPER_SNAKE_CASE, kebab-case, unknown.
    """
    if not name or name.startswith("_"):
        # Strip leading underscores for detection
        stripped = name.lstrip("_")
        if not stripped:
            return "unknown"
        name = stripped

    if "-" in name:
        return "kebab-case"

    if name.isupper() and "_" in name:
        return "UPPER_SNAKE_CASE"

    if name.isupper() and "_" not in name:
        # Single word all caps like "URL" — ambiguous
        return "UPPER_SNAKE_CASE"

    if "_" in name:
        # Has underscores and is not all-caps
        return "snake_case"

    if name[0].isupper():
        return "PascalCase"

    if name[0].islower() and any(c.isupper() for c in name[1:]):
        return "camelCase"

    # Single lowercase word — could be anything, default to snake_case
    if name.islower():
        return "snake_case"

    return "unknown"


def detect_import_category(module_path: str) -> str:
    """Categorize an import as builtin, external, internal, or relative.

    - relative: starts with '.'
    - builtin: in Python stdlib
    - internal: starts with 'src' or looks like a local module
    - external: everything else
    """
    if module_path.startswith("."):
        return "relative"

    top_level = module_path.split(".")[0]

    if top_level in _STDLIB_MODULES:
        return "builtin"

    if top_level == "src" or module_path.startswith("src/"):
        return "internal"

    # Heuristic: if it could be a stdlib module we missed, check sys.stdlib_module_names
    if hasattr(sys, "stdlib_module_names") and top_level in sys.stdlib_module_names:
        return "builtin"

    return "external"


def load_style_profile(path: str | Path = ".agentprobe/style-profile.yaml") -> dict:
    """Load a style profile from YAML. Returns empty dict if file missing."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def check_name_convention(name: str, expected_convention: str) -> bool:
    """Check if a name follows the expected naming convention.

    A single lowercase word ("run", "utils") is valid snake_case, camelCase
    and kebab-case at the same time, so it satisfies any of the three.
    """
    if detect_case(name) == expected_convention:
        return True
    core = name.strip("_")
    return (
        bool(core)
        and core.isalnum()
        and core.islower()
        and expected_convention in ("snake_case", "camelCase", "kebab-case")
    )


def check_import_order(categories: list[str], expected_order: list[str]) -> list[int]:
    """Check if import categories follow the expected ordering.

    Returns list of line indices where ordering is violated.
    """
    violations = []
    if not expected_order or not categories:
        return violations

    order_map = {cat: i for i, cat in enumerate(expected_order)}

    max_seen_rank = -1
    for i, cat in enumerate(categories):
        rank = order_map.get(cat, len(expected_order))
        if rank < max_seen_rank:
            violations.append(i)
        else:
            max_seen_rank = rank

    return violations


# Built-in forbidden patterns that can be enabled by name in style-profile.yaml.
FORBIDDEN_PATTERNS = {
    "console.log-in-production-code": re.compile(r"\bconsole\s*\.\s*log\s*\("),
    "any-type-in-typescript": re.compile(r":\s*any\b"),
}

_VALID_SEVERITIES = {"WARN", "FATAL"}


def compile_forbidden_rules(forbidden_list: list) -> list[dict]:
    r"""Normalize the ``forbidden`` section of style-profile.yaml into rules.

    Each entry is either the name of a built-in pattern (see FORBIDDEN_PATTERNS)
    or a custom rule::

        - name: no-print
          pattern: '\bprint\('          # Python regex, matched per added line
          message: Use structlog instead of print()
          files: "*.py"                   # optional glob (or list of globs)
          severity: WARN                  # WARN (default) or FATAL (blocks the PR)

    Unknown names and invalid regexes are skipped with a logged warning rather
    than crashing the run.
    """
    rules: list[dict] = []
    for entry in forbidden_list or []:
        if isinstance(entry, str):
            regex = FORBIDDEN_PATTERNS.get(entry)
            if regex is None:
                logger.warning("Unknown built-in forbidden pattern %r (skipped)", entry)
                continue
            rules.append({"name": entry, "regex": regex, "message": "", "files": [],
                          "severity": "WARN"})
        elif isinstance(entry, dict) and entry.get("pattern"):
            name = str(entry.get("name") or entry["pattern"])
            try:
                regex = re.compile(str(entry["pattern"]))
            except re.error as exc:
                logger.warning("Invalid regex for forbidden rule %r: %s (skipped)", name, exc)
                continue
            files = entry.get("files") or []
            if isinstance(files, str):
                files = [files]
            severity = str(entry.get("severity", "WARN")).upper()
            rules.append({
                "name": name,
                "regex": regex,
                "message": str(entry.get("message") or ""),
                "files": [str(f) for f in files],
                "severity": severity if severity in _VALID_SEVERITIES else "WARN",
            })
        else:
            logger.warning("Ignoring malformed forbidden entry: %r", entry)
    return rules


def check_forbidden_patterns(
    source: str, forbidden_list: list, file_path: str | None = None
) -> list[dict]:
    """Scan source code for forbidden patterns.

    Returns list of {pattern, line_number, line_content, message, severity}
    for each match. ``line_number`` is 1-based within ``source``.
    """
    matches = []
    lines = source.split("\n")
    for rule in compile_forbidden_rules(forbidden_list):
        if rule["files"] and file_path is not None:
            name = file_path.rsplit("/", 1)[-1]
            if not any(fnmatch.fnmatch(file_path, g) or fnmatch.fnmatch(name, g)
                       for g in rule["files"]):
                continue
        for i, line in enumerate(lines, 1):
            if rule["regex"].search(line):
                matches.append({
                    "pattern": rule["name"],
                    "line_number": i,
                    "line_content": line.strip(),
                    "message": rule["message"],
                    "severity": rule["severity"],
                })
    return matches
