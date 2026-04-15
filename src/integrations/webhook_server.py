"""FastAPI webhook server for GitHub PR events."""

from __future__ import annotations

import hashlib
import hmac
import os

from fastapi import FastAPI, Header, HTTPException, Request, BackgroundTasks
import structlog

from src.graph.state import AgentProbeState
from src.graph.workflow import run_agentprobe
from src.integrations.github_app import GitHubClient

logger = structlog.get_logger(__name__)

app = FastAPI(title="AgentProbe Webhook Server")


def _verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify GitHub webhook X-Hub-Signature-256."""
    if not signature or not secret:
        return False
    expected = "sha256=" + hmac.new(
        secret.encode("utf-8"), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


async def _process_pr(
    repo_full_name: str,
    pr_number: int,
    github_token: str,
) -> None:
    """Process a PR event: run AgentProbe pipeline and post results."""
    try:
        client = GitHubClient(github_token)

        # Get PR diff and HEAD SHA
        pr_diff = client.get_pr_diff(repo_full_name, pr_number)
        head_sha = client.get_pr_head_sha(repo_full_name, pr_number)

        # Build initial state
        state = AgentProbeState(
            pr_diff=pr_diff,
            repo_path=".",
            pr_number=pr_number,
            repo_full_name=repo_full_name,
            architecture_report=None,
            pattern_report=None,
            regression_report=None,
            short_circuit=False,
            verdict=None,
            cached_functions=[],
            cost_tracker={},
        )

        # Run the pipeline
        result = run_agentprobe(state)

        # Post results
        verdict = result.get("verdict", {})
        comment = verdict.get("comment", "AgentProbe analysis complete.")
        client.post_pr_comment(repo_full_name, pr_number, comment)
        client.set_check_status(
            repo_full_name,
            head_sha,
            verdict.get("status", "PASS"),
            f"AgentProbe: {verdict.get('status', 'PASS')} (score: {verdict.get('score', 0):.0f})",
        )

        logger.info(
            "pr_processed",
            repo=repo_full_name,
            pr=pr_number,
            verdict=verdict.get("status"),
            score=verdict.get("score"),
        )
    except Exception:
        logger.exception("pr_processing_failed", repo=repo_full_name, pr=pr_number)


@app.post("/webhook")
async def webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: str | None = Header(None),
    x_github_event: str | None = Header(None),
) -> dict:
    """Handle GitHub webhook events."""
    payload = await request.body()

    # Verify webhook signature — ALWAYS require it for security
    webhook_secret = os.environ.get("GITHUB_WEBHOOK_SECRET", "")
    if not webhook_secret:
        logger.error("GITHUB_WEBHOOK_SECRET not configured — rejecting request")
        raise HTTPException(status_code=500, detail="Webhook secret not configured")
    if not _verify_signature(payload, x_hub_signature_256 or "", webhook_secret):
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Only handle pull_request events
    if x_github_event != "pull_request":
        return {"status": "ignored", "reason": "not a pull_request event"}

    body = await request.json()
    action = body.get("action", "")

    # Only process opened or synchronize (new push)
    if action not in ("opened", "synchronize"):
        return {"status": "ignored", "reason": f"action '{action}' not handled"}

    pr = body.get("pull_request", {})
    pr_number = pr.get("number")
    repo_full_name = body.get("repository", {}).get("full_name", "")

    if not pr_number or not repo_full_name:
        raise HTTPException(status_code=400, detail="Missing PR number or repo")

    github_token = os.environ.get("GITHUB_TOKEN", "")
    if not github_token:
        raise HTTPException(status_code=500, detail="GITHUB_TOKEN not configured")

    # Process async
    background_tasks.add_task(_process_pr, repo_full_name, pr_number, github_token)

    return {"status": "processing", "pr": pr_number, "repo": repo_full_name}


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}
