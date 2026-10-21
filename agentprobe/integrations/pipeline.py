"""Shared glue for the GitHub Action, the local CLI and the webhook server.

All three entry points do the same thing:
  1. get the PR diff,
  2. load the governance policy from the *base* ref (so a PR cannot relax
     the rules it is judged by),
  3. fetch full before/after sources of modified code files (so the
     regression analyzer sees whole functions, not 3-line hunks),
  4. run the LangGraph workflow and return the final state.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Callable

from agentprobe.config.loader import DEFAULT_CONFIG_DIR, load_config
from agentprobe.graph.state import AgentProbeState
from agentprobe.graph.workflow import run_agentprobe
from agentprobe.parsers.diff_parser import filter_diff, parse_diff
from agentprobe.parsers.tree_sitter_engine import EXTENSION_TO_LANGUAGE

logger = logging.getLogger(__name__)

POLICY_FILES = ("config.yaml", "boundaries.yaml", "style-profile.yaml")
MAX_SOURCE_FILES = 50

# fetch(path, ref) -> text or None
Fetcher = Callable[[str, str], "str | None"]


def materialize_policy(
    fetch: Fetcher | None,
    base_ref: str | None,
    policy_repo_dir: str,
    target_dir: Path,
    fallback_dir: Path | None = None,
) -> str:
    """Write the policy files from ``base_ref`` into ``target_dir``.

    Returns where the policy came from: "base", "workspace" (base had no
    policy, e.g. the PR that adopts AgentProbe) or "none".
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    found = False
    if fetch and base_ref:
        for name in POLICY_FILES:
            try:
                text = fetch(f"{policy_repo_dir.strip('/')}/{name}", base_ref)
            except Exception:  # noqa: BLE001 - fall back to the workspace copy
                logger.warning("Could not read %s at %s", name, base_ref, exc_info=True)
                text = None
            if text is not None:
                (target_dir / name).write_text(text)
                found = True
    if found:
        return "base"
    if fallback_dir and any((fallback_dir / n).exists() for n in POLICY_FILES):
        for name in POLICY_FILES:
            src = fallback_dir / name
            if src.exists():
                (target_dir / name).write_text(src.read_text())
        return "workspace"
    return "none"


def collect_file_sources(
    diff: str,
    fetch: Fetcher | None,
    base_ref: str | None,
    head_ref: str | None,
    max_files: int = MAX_SOURCE_FILES,
) -> dict[str, dict[str, str | None]]:
    """Full before/after text of changed code files.

    ``after`` (head) lets the architecture/pattern analyzers parse whole files;
    ``before`` (base, only when lines were removed) lets the regression
    analyzer compare whole functions.
    """
    if not (fetch and base_ref and head_ref):
        return {}
    sources: dict[str, dict[str, str | None]] = {}
    for change in parse_diff(diff).files:
        if len(sources) >= max_files:
            break
        if change.status != "modified":
            continue  # added files: the diff already holds the whole file
        if Path(change.file_path).suffix.lower() not in EXTENSION_TO_LANGUAGE:
            continue
        try:
            before = fetch(change.file_path, base_ref) if change.deleted_lines else None
            after = fetch(change.file_path, head_ref)
        except Exception:  # noqa: BLE001 - regression falls back to hunks
            logger.warning("Could not fetch sources for %s", change.file_path, exc_info=True)
            continue
        sources[change.file_path] = {"before": before, "after": after}
    return sources


def run_pipeline(
    *,
    diff: str,
    repo_full_name: str = "",
    pr_number: int = 0,
    fetch: Fetcher | None = None,
    base_ref: str | None = None,
    head_ref: str | None = None,
    repo_path: str | None = None,
    policy_repo_dir: str = DEFAULT_CONFIG_DIR,
    allow_code_execution: bool = True,
) -> tuple[AgentProbeState, str]:
    """Run the full workflow. Returns (final_state, policy_source)."""
    fallback = Path(repo_path) / policy_repo_dir if repo_path else None
    with tempfile.TemporaryDirectory(prefix="agentprobe-policy-") as tmp:
        policy_source = materialize_policy(fetch, base_ref, policy_repo_dir, Path(tmp),
                                           fallback)
        config = load_config(Path(tmp) / "config.yaml")
        if config.ignore_paths:
            diff = filter_diff(diff, config.is_ignored)
        state = AgentProbeState(
            pr_diff=diff,
            repo_path=repo_path or "",
            pr_number=pr_number,
            repo_full_name=repo_full_name,
            config_dir=tmp,
            policy_repo_dir=policy_repo_dir,
            file_sources=collect_file_sources(diff, fetch, base_ref, head_ref),
            allow_code_execution=allow_code_execution,
            architecture_report=None,
            pattern_report=None,
            regression_report=None,
            short_circuit=False,
            verdict=None,
            cached_functions=[],
            cost_tracker={},
        )
        result = run_agentprobe(state)
    return result, policy_source


def iter_findings(state: AgentProbeState):
    """Yield (severity, file, line, message) for every finding in the final state."""
    for key, label in (("architecture_report", "architecture"),
                       ("pattern_report", "pattern")):
        for v in (state.get(key) or {}).get("violations", []):
            yield (
                v.get("severity", "WARN"),
                v.get("file_path") or v.get("file", ""),
                v.get("line_number") or v.get("line") or 0,
                f"[{label}] {v.get('description') or v.get('rule_violated', '')}",
            )
    for d in (state.get("regression_report") or {}).get("deltas", []):
        text = d.get("summary") or (f"{d.get('field')}: {str(d.get('before'))[:140]} -> "
                                    f"{str(d.get('after'))[:140]}")
        yield (d.get("severity", "MEDIUM"), d.get("file", ""), d.get("line") or 0,
               f"[regression] {d.get('function')}: {text}")
