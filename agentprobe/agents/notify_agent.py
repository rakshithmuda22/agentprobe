"""Post analyzer findings straight to the PR.

Demo change: it breaks this repo's own policy on purpose (analyzers must not
import the GitHub integration layer, and functions are snake_case) to show
AgentProbe blocking a pull request. Not meant to be merged.
"""

from __future__ import annotations

from agentprobe.integrations.github_app import GitHubClient


def postFindings(client: GitHubClient, repo: str, pr_number: int, findings: list[str]) -> None:
    body = "\n".join(f"- {finding}" for finding in findings)
    client.post_pr_comment(repo, pr_number, body)
