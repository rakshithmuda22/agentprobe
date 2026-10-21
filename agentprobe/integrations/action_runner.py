"""GitHub Action entry point for AgentProbe.

Reads the inputs GitHub passes to a Docker action (``INPUT_*`` env vars), the
``pull_request`` event payload and the standard ``GITHUB_*`` variables, runs
the governance workflow, then:

* prints the verdict and emits ``::error``/``::warning`` annotations (shown
  inline on the PR diff),
* writes the verdict to the job summary and ``verdict``/``score`` outputs,
* creates or updates a single PR comment and the ``agentprobe/governance``
  commit status (skipped with a warning if the token can't write, e.g. fork PRs),
* exits 1 on BLOCK (or on WARN with ``fail_on: warn``) so the check fails.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import structlog

from agentprobe.agents.verdict_orchestrator import COMMENT_MARKER
from agentprobe.config.loader import DEFAULT_CONFIG_DIR
from agentprobe.integrations.github_app import GitHubClient
from agentprobe.integrations.pipeline import iter_findings, run_pipeline

logger = structlog.get_logger(__name__)

# Events whose token/secrets are privileged while the code is attacker-controlled:
# never execute PR code under them, even if property tests are enabled.
UNTRUSTED_CODE_EVENTS = {"pull_request_target", "workflow_run"}


def _input(name: str, default: str = "") -> str:
    """Read an action input the way the runner passes it (INPUT_<NAME>)."""
    value = os.environ.get(f"INPUT_{name.upper()}")
    if value is None:
        value = os.environ.get(f"INPUT_{name.upper().replace('_', '-')}")
    return value.strip() if value and value.strip() else default


def _escape(value: str, *, prop: bool = False) -> str:
    value = str(value).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    if prop:
        value = value.replace(":", "%3A").replace(",", "%2C")
    return value


def emit_annotations(state: dict, limit: int = 50) -> None:
    """Print workflow commands so findings appear inline on the PR's Files tab."""
    for i, (severity, file, line, message) in enumerate(iter_findings(state)):
        if i >= limit:
            break
        level = "error" if severity in ("FATAL", "BLOCK") else "warning"
        props = [f"file={_escape(file, prop=True)}"] if file else []
        if line:
            props.append(f"line={int(line)}")
        props.append("title=AgentProbe")
        print(f"::{level} {','.join(props)}::{_escape(message)}")


def _append_file(env_var: str, text: str) -> None:
    path = os.environ.get(env_var)
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)
    except OSError:
        logger.warning("could_not_write", target=env_var)


def write_outputs(verdict: dict) -> None:
    delimiter = f"agentprobe_{uuid.uuid4().hex}"
    _append_file(
        "GITHUB_OUTPUT",
        f"verdict={verdict.get('status', 'PASS')}\n"
        f"score={verdict.get('score', 0)}\n"
        f"comment<<{delimiter}\n{verdict.get('comment', '')}\n{delimiter}\n",
    )
    _append_file("GITHUB_STEP_SUMMARY", verdict.get("comment", "") + "\n")


def exit_code_for(status: str, fail_on: str) -> int:
    fail_on = (fail_on or "block").lower()
    if fail_on == "never":
        return 0
    if fail_on == "warn":
        return 1 if status in ("WARN", "BLOCK") else 0
    return 1 if status == "BLOCK" else 0


def main() -> int:
    """Run AgentProbe as a GitHub Action. Returns the process exit code."""
    github_token = _input("github_token") or os.environ.get("GITHUB_TOKEN", "")
    if not github_token:
        logger.error("No token: set the github_token input or GITHUB_TOKEN")
        return 1

    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    if not event_path or not Path(event_path).exists():
        logger.error("GITHUB_EVENT_PATH not set or file missing")
        return 1

    repo_full_name = os.environ.get("GITHUB_REPOSITORY", "")
    if not repo_full_name:
        logger.error("GITHUB_REPOSITORY not set")
        return 1

    with open(event_path) as f:
        event = json.load(f)

    pr = event.get("pull_request") or {}
    pr_number = pr.get("number")
    if not pr_number:
        logger.error("No pull_request.number in event payload; run this action on "
                     "pull_request events")
        return 1

    event_name = os.environ.get("GITHUB_EVENT_NAME", "pull_request")
    workspace = os.environ.get("GITHUB_WORKSPACE") or os.getcwd()
    policy_repo_dir = _input("config_dir", DEFAULT_CONFIG_DIR)
    post_comment = _input("post_comment", "true").lower() not in ("false", "0", "no")
    fail_on = _input("fail_on", "block")

    try:
        client = GitHubClient(github_token, base_url=os.environ.get("GITHUB_API_URL"))
        diff = client.get_pr_diff(repo_full_name, pr_number)
        base_sha = (pr.get("base") or {}).get("sha")
        head_sha = (pr.get("head") or {}).get("sha")
        if not base_sha or not head_sha:
            base_sha, head_sha = client.get_pr_refs(repo_full_name, pr_number)
    except Exception:
        logger.exception("Failed to fetch PR data from GitHub")
        return 1

    def fetch(path: str, ref: str) -> str | None:
        return client.get_file_content(repo_full_name, path, ref)

    try:
        result, policy_source = run_pipeline(
            diff=diff,
            repo_full_name=repo_full_name,
            pr_number=pr_number,
            fetch=fetch,
            base_ref=base_sha,
            head_ref=head_sha,
            repo_path=workspace,
            policy_repo_dir=policy_repo_dir,
            allow_code_execution=event_name not in UNTRUSTED_CODE_EVENTS,
        )
    except Exception:
        logger.exception("Pipeline execution failed")
        return 1

    verdict = result.get("verdict") or {}
    status = verdict.get("status", "PASS")
    score = float(verdict.get("score", 0) or 0)
    comment = verdict.get("comment", "AgentProbe analysis complete.")

    print(f"AgentProbe verdict: {status} (score {score:.1f}/100, policy from {policy_source})")
    print("::group::AgentProbe report")
    print(comment)
    print("::endgroup::")
    emit_annotations(result)
    write_outputs(verdict)

    if post_comment:
        try:
            how = client.upsert_pr_comment(repo_full_name, pr_number, comment, COMMENT_MARKER)
            client.set_check_status(
                repo_full_name, head_sha, status,
                f"AgentProbe: {status} (score: {score:.0f})",
            )
            print(f"PR comment {how}; commit status agentprobe/governance set on {head_sha[:7]}")
        except Exception as exc:  # noqa: BLE001 - reporting must not change the verdict
            print(f"::warning title=AgentProbe::Could not post the PR comment/status "
                  f"({type(exc).__name__}). Fork PRs get a read-only token; grant "
                  f"'pull-requests: write' and 'statuses: write' for same-repo PRs.")

    code = exit_code_for(status, fail_on)
    logger.info("action_complete", verdict=status, score=score, exit_code=code)
    return code


if __name__ == "__main__":
    sys.exit(main())
