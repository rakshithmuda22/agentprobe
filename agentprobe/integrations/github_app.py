"""GitHub App integration — wraps PyGithub for PR interactions."""

from __future__ import annotations

import logging
import time
from typing import Any

from github import Github, GithubException

logger = logging.getLogger(__name__)

# Max retries for rate-limited requests
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds


class GitHubClient:
    """Handles GitHub API interactions for AgentProbe."""

    def __init__(self, token: str):
        if not token:
            raise ValueError("GitHub token must not be empty")
        self._github = Github(token)

    def post_pr_comment(self, repo_full_name: str, pr_number: int, body: str) -> None:
        """Post a comment on a pull request."""
        self._with_retry(lambda: self._post_comment(repo_full_name, pr_number, body))

    def _post_comment(self, repo_full_name: str, pr_number: int, body: str) -> None:
        repo = self._github.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        pr.create_issue_comment(body)

    def set_check_status(
        self, repo_full_name: str, sha: str, status: str, summary: str
    ) -> None:
        """Set a commit check status (success, failure, neutral)."""
        status_map = {"PASS": "success", "WARN": "success", "BLOCK": "failure"}
        gh_status = status_map.get(status, "neutral")
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
        """Fetch the diff for a pull request."""
        repo = self._github.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        # PyGithub doesn't have a direct diff method, use the files
        files = pr.get_files()
        diff_parts = []
        for f in files:
            if f.patch:
                diff_parts.append(
                    f"diff --git a/{f.filename} b/{f.filename}\n"
                    f"--- a/{f.filename}\n"
                    f"+++ b/{f.filename}\n"
                    f"{f.patch}"
                )
        return "\n".join(diff_parts)

    def get_pr_head_sha(self, repo_full_name: str, pr_number: int) -> str:
        """Get the HEAD SHA of a pull request."""
        repo = self._github.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        return pr.head.sha

    @staticmethod
    def _with_retry(func: Any, max_retries: int = MAX_RETRIES) -> Any:
        """Execute a function with exponential backoff on rate limit errors."""
        for attempt in range(max_retries):
            try:
                return func()
            except GithubException as e:
                if e.status == 403 and "rate limit" in str(e).lower():
                    wait = RETRY_BACKOFF_BASE ** (attempt + 1)
                    logger.warning(f"Rate limited, retrying in {wait}s (attempt {attempt + 1})")
                    time.sleep(wait)
                else:
                    raise
        raise GithubException(403, {"message": "Rate limit exceeded after retries"}, None)
