"""Command-line interface.

    agentprobe check [--repo .] [--base main] [--head HEAD]   # local, no GitHub needed
    agentprobe action                                       # GitHub Action entrypoint

``check`` judges ``git diff <base>...<head>`` (the same merge-base diff a PR
shows) using the policy committed on <base>, prints the verdict and exits
1 on BLOCK, so it also works as a pre-push hook or in any CI system.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from agentprobe.config.loader import DEFAULT_CONFIG_DIR


def _git(repo: str, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", repo, *args], check=True, capture_output=True, text=True
    ).stdout


def _git_fetcher(repo: str):
    def fetch(path: str, ref: str) -> str | None:
        proc = subprocess.run(
            ["git", "-C", repo, "show", f"{ref}:{path}"], capture_output=True, text=True
        )
        return proc.stdout if proc.returncode == 0 else None
    return fetch


def cmd_check(args: argparse.Namespace) -> int:
    from agentprobe.integrations.action_runner import exit_code_for
    from agentprobe.integrations.pipeline import run_pipeline

    repo = str(Path(args.repo).resolve())
    try:
        merge_base = _git(repo, "merge-base", args.base, args.head).strip()
        head = _git(repo, "rev-parse", args.head).strip()
        diff = _git(repo, "diff", "--no-color", "--no-ext-diff", "--unified=3",
                    f"{merge_base}..{head}")
    except subprocess.CalledProcessError as exc:
        print(f"agentprobe: git failed: {exc.stderr.strip()}", file=sys.stderr)
        return 2

    result, policy_source = run_pipeline(
        diff=diff,
        repo_full_name=Path(repo).name,
        fetch=_git_fetcher(repo),
        base_ref=merge_base,
        head_ref=head,
        repo_path=repo,
        policy_repo_dir=args.config_dir,
        allow_code_execution=True,
    )
    verdict = result.get("verdict") or {}
    status = verdict.get("status", "PASS")

    if args.json:
        print(json.dumps({
            "verdict": status,
            "score": verdict.get("score"),
            "analyzer_scores": verdict.get("analyzer_scores"),
            "policy_source": policy_source,
            "architecture": result.get("architecture_report"),
            "pattern": result.get("pattern_report"),
            "regression": result.get("regression_report"),
        }, indent=2, default=str))
    else:
        print(verdict.get("comment", "").replace(
            "<!-- agentprobe-verdict -->\n", ""))
        print(f"\n(policy from {policy_source}; diff {args.base}...{args.head})")
    return exit_code_for(status, args.fail_on)


def cmd_action(_args: argparse.Namespace) -> int:
    from agentprobe.integrations.action_runner import main as action_main

    return action_main()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agentprobe",
        description="Deterministic PR governance: architecture boundaries, naming/forbidden "
                    "patterns and behavior-change detection.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="judge a local git range (no GitHub needed)")
    check.add_argument("--repo", default=".", help="path to the git repository (default: .)")
    check.add_argument("--base", default="main", help="base ref (default: main)")
    check.add_argument("--head", default="HEAD", help="head ref (default: HEAD)")
    check.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR,
                       help="policy directory inside the repo (default: .agentprobe)")
    check.add_argument("--fail-on", choices=("block", "warn", "never"), default="block")
    check.add_argument("--json", action="store_true", help="print machine-readable JSON")
    check.set_defaults(func=cmd_check)

    action = sub.add_parser("action", help="GitHub Action entrypoint (reads GITHUB_* env)")
    action.set_defaults(func=cmd_action)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
