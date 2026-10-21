"""GitHub REST client for AgentProbe (token auth via PyGithub)."""

from __future__ import annotations

import logging
import time
from typing import Any

from github import Auth, Github, GithubException

logger = logging.getLogger(__name__)

# Max retries for rate-limited requests
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds
DEFAULT_API_URL = "https://api.github.com"


class GitHubClient:
    """Handles GitHub API interactions for AgentProbe.

    ``base_url`` defaults to $GITHUB_API_URL semantics: api.github.com, or the
    GitHub Enterprise Server API URL the runner provides.
    """

    def __init__(self, token: str, base_url: str | None = None):
        if not token:
            raise ValueError("GitHub token must not be empty")
        self._github = Github(
            auth=Auth.Token(token),
            base_url=(base_url or DEFAULT_API_URL).rstrip("/"),
        )
        self._pulls: dict[tuple[str, int], Any] = {}

    def _pull(self, repo_full_name: str, pr_number: int):
        key = (repo_full_name, pr_number)
        if key not in self._pulls:
            self._pulls[key] = self._github.get_repo(repo_full_name).get_pull(pr_number)
        return self._pulls[key]

    def post_pr_comment(self, repo_full_name: str, pr_number: int, body: str) -> None:
        """Post a new comment on a pull request."""
        self._with_retry(lambda: self._pull(repo_full_name, pr_number).create_issue_comment(body))

    def upsert_pr_comment(
        self, repo_full_name: str, pr_number: int, body: str, marker: str
    ) -> str:
        """Edit the existing AgentProbe comment (found by ``marker``) or create one.

        Returns "updated" or "created". Keeps one comment per PR instead of a
        new one on every push.
        """
        def _do() -> str:
            pr = self._pull(repo_full_name, pr_number)
            for comment in pr.get_issue_comments():
                if marker in (comment.body or ""):
                    comment.edit(body)
                    return "updated"
            pr.create_issue_comment(body)
            return "created"

        return self._with_retry(_do)

    def set_check_status(
        self, repo_full_name: str, sha: str, status: str, summary: str
    ) -> None:
        """Set the ``agentprobe/governance`` commit status (success/failure)."""
        status_map = {"PASS": "success", "WARN": "success", "BLOCK": "failure"}
        gh_status = status_map.get(status, "error")
        self._with_retry(
            lambda: self._set_status(repo_full_name, sha, gh_status, summary)
        )

    def _set_status(
        self, repo_full_name: str, sha: str, state: str, description: str
    ) -> None:
        repo = self._github.get_repo(repo_full_name)
        repo.get_commit(sha).create_status(
            state=state,
            description=description[:140],  # GitHub limit
            context="agentprobe/governance",
        )

    def get_pr_diff(self, repo_full_name: str, pr_number: int) -> str:
        """Fetch the PR's changes as a unified diff built from the files API."""
        pr = self._pull(repo_full_name, pr_number)
        diff_parts = []
        for f in pr.get_files():
            if not f.patch:  # binary or too large for the API
                continue
            name = f.filename
            header = f"diff --git a/{name} b/{name}\n"
            if f.status == "added":
                header += f"new file mode 100644\n--- /dev/null\n+++ b/{name}\n"
            elif f.status == "removed":
                header += f"deleted file mode 100644\n--- a/{name}\n+++ /dev/null\n"
            else:
                header += f"--- a/{name}\n+++ b/{name}\n"
            diff_parts.append(header + f.patch)
        return "\n".join(diff_parts)

    def get_pr_refs(self, repo_full_name: str, pr_number: int) -> tuple[str, str]:
        """(base_sha, head_sha) of a pull request."""
        pr = self._pull(repo_full_name, pr_number)
        return pr.base.sha, pr.head.sha

    def get_pr_head_sha(self, repo_full_name: str, pr_number: int) -> str:
        """Get the HEAD SHA of a pull request."""
        return self._pull(repo_full_name, pr_number).head.sha

    def get_file_content(self, repo_full_name: str, path: str, ref: str) -> str | None:
        """Text content of ``path`` at ``ref``, or None if missing/unreadable."""
        try:
            contents = self._github.get_repo(repo_full_name).get_contents(path, ref=ref)
        except GithubException as e:
            if e.status == 404:
                return None
            raise
        if isinstance(contents, list):  # a directory
            return None
        try:
            return contents.decoded_content.decode("utf-8")
        except (UnicodeDecodeError, AssertionError, GithubException):
            return None

    @staticmethod
    def _with_retry(func: Any, max_retries: int = MAX_RETRIES) -> Any:
        """Execute a function with exponential backoff on rate limit errors."""
        for attempt in range(max_retries):
            try:
                return func()
            except GithubException as e:
                if e.status in (403, 429) and "rate limit" in str(e).lower():
                    wait = RETRY_BACKOFF_BASE ** (attempt + 1)
                    logger.warning(f"Rate limited, retrying in {wait}s (attempt {attempt + 1})")
                    time.sleep(wait)
                else:
                    raise
        raise GithubException(403, {"message": "Rate limit exceeded after retries"}, None)
