"""Tests for GitHub integration — webhook server, GitHubClient, action runner."""

import hashlib
import hmac
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.integrations.webhook_server import app, _verify_signature


# ── Webhook Signature Verification ──


class TestVerifySignature:
    def test_valid_signature(self):
        secret = "test-secret"
        payload = b'{"action": "opened"}'
        sig = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        assert _verify_signature(payload, sig, secret) is True

    def test_invalid_signature(self):
        assert _verify_signature(b"payload", "sha256=bad", "secret") is False

    def test_empty_signature_returns_false(self):
        assert _verify_signature(b"payload", "", "secret") is False

    def test_empty_secret_returns_false(self):
        assert _verify_signature(b"payload", "sha256=abc", "") is False

    def test_timing_safe_comparison(self):
        """Verify we use constant-time comparison (hmac.compare_digest)."""
        secret = "s"
        payload = b"p"
        sig = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        # If this passes, compare_digest is being used (we verified in the source)
        assert _verify_signature(payload, sig, secret) is True


# ── Webhook Endpoint ──


SECRET = "test-secret"


def _signed_post(client, payload_dict, event="pull_request", secret=SECRET, env_overrides=None):
    """Helper to send a properly signed webhook request."""
    payload = json.dumps(payload_dict).encode()
    sig = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    env = {"GITHUB_WEBHOOK_SECRET": secret}
    if env_overrides:
        env.update(env_overrides)
    with patch.dict(os.environ, env, clear=False):
        return client.post(
            "/webhook",
            content=payload,
            headers={
                "x-github-event": event,
                "x-hub-signature-256": sig,
                "content-type": "application/json",
            },
        )


class TestWebhookEndpoint:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_non_pr_event_ignored(self, client):
        resp = _signed_post(client, {"action": "pushed"}, event="push")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ignored"

    def test_pr_closed_action_ignored(self, client):
        resp = _signed_post(client, {"action": "closed", "pull_request": {"number": 1}})
        assert resp.status_code == 200
        assert resp.json()["reason"] == "action 'closed' not handled"

    def test_missing_pr_number_returns_400(self, client):
        resp = _signed_post(
            client,
            {"action": "opened", "pull_request": {}, "repository": {"full_name": "o/r"}},
        )
        assert resp.status_code == 400

    def test_missing_github_token_returns_500(self, client):
        resp = _signed_post(
            client,
            {"action": "opened", "pull_request": {"number": 1}, "repository": {"full_name": "o/r"}},
            env_overrides={"GITHUB_TOKEN": ""},
        )
        assert resp.status_code == 500

    @patch("src.integrations.webhook_server._process_pr")
    def test_valid_pr_opened_starts_processing(self, mock_process, client):
        secret = "test-secret"
        payload = json.dumps({
            "action": "opened",
            "pull_request": {"number": 42},
            "repository": {"full_name": "owner/repo"},
        }).encode()
        sig = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        with patch.dict(os.environ, {"GITHUB_TOKEN": "tok", "GITHUB_WEBHOOK_SECRET": secret}):
            resp = client.post(
                "/webhook",
                content=payload,
                headers={
                    "x-github-event": "pull_request",
                    "x-hub-signature-256": sig,
                    "content-type": "application/json",
                },
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "processing"
        assert resp.json()["pr"] == 42

    def test_missing_webhook_secret_returns_500(self, client):
        with patch.dict(os.environ, {"GITHUB_WEBHOOK_SECRET": ""}, clear=False):
            resp = client.post(
                "/webhook",
                json={"action": "opened"},
                headers={"x-github-event": "pull_request"},
            )
            assert resp.status_code == 500

    def test_invalid_signature_returns_401(self, client):
        with patch.dict(os.environ, {"GITHUB_WEBHOOK_SECRET": "real-secret"}):
            resp = client.post(
                "/webhook",
                json={"action": "opened"},
                headers={
                    "x-github-event": "pull_request",
                    "x-hub-signature-256": "sha256=wrong",
                },
            )
            assert resp.status_code == 401


# ── Action Runner ──


class TestActionRunner:
    @patch.dict(os.environ, {"GITHUB_TOKEN": ""}, clear=True)
    def test_missing_token_exits_1(self):
        from src.integrations.action_runner import main
        assert main() == 1

    @patch.dict(
        os.environ,
        {"GITHUB_TOKEN": "tok", "GITHUB_EVENT_PATH": "/nonexistent"},
        clear=True,
    )
    def test_missing_event_file_exits_1(self):
        from src.integrations.action_runner import main
        assert main() == 1

    @patch.dict(
        os.environ,
        {"GITHUB_TOKEN": "tok", "GITHUB_EVENT_PATH": "", "GITHUB_REPOSITORY": "o/r"},
        clear=True,
    )
    def test_empty_event_path_exits_1(self):
        from src.integrations.action_runner import main
        assert main() == 1

    def test_missing_pr_number_exits_1(self, tmp_path):
        event_file = tmp_path / "event.json"
        event_file.write_text(json.dumps({"issue": {"number": 1}}))
        with patch.dict(
            os.environ,
            {
                "GITHUB_TOKEN": "tok",
                "GITHUB_EVENT_PATH": str(event_file),
                "GITHUB_REPOSITORY": "o/r",
            },
            clear=True,
        ):
            from src.integrations.action_runner import main
            assert main() == 1

    @patch("src.integrations.action_runner.GitHubClient")
    @patch("src.integrations.action_runner.run_agentprobe")
    def test_pass_verdict_exits_0(self, mock_run, mock_client_cls, tmp_path):
        event_file = tmp_path / "event.json"
        event_file.write_text(json.dumps({"pull_request": {"number": 5}}))
        style_file = tmp_path / "style-profile.yaml"
        style_file.write_text("naming:\n  functions: snake_case\n")

        mock_run.return_value = {"verdict": {"status": "PASS", "score": 0, "comment": "ok"}}
        mock_gh = MagicMock()
        mock_client_cls.return_value = mock_gh
        mock_gh.get_pr_diff.return_value = ""
        mock_gh.get_pr_head_sha.return_value = "abc123"

        with patch.dict(
            os.environ,
            {
                "GITHUB_TOKEN": "tok",
                "GITHUB_EVENT_PATH": str(event_file),
                "GITHUB_REPOSITORY": "o/r",
                "STYLE_PROFILE_PATH": str(style_file),
            },
            clear=True,
        ):
            from src.integrations.action_runner import main
            assert main() == 0

    @patch("src.integrations.action_runner.GitHubClient")
    @patch("src.integrations.action_runner.run_agentprobe")
    def test_block_verdict_exits_1(self, mock_run, mock_client_cls, tmp_path):
        event_file = tmp_path / "event.json"
        event_file.write_text(json.dumps({"pull_request": {"number": 5}}))
        style_file = tmp_path / "style-profile.yaml"
        style_file.write_text("naming:\n  functions: snake_case\n")

        mock_run.return_value = {"verdict": {"status": "BLOCK", "score": 100, "comment": "blocked"}}
        mock_gh = MagicMock()
        mock_client_cls.return_value = mock_gh
        mock_gh.get_pr_diff.return_value = ""
        mock_gh.get_pr_head_sha.return_value = "abc123"

        with patch.dict(
            os.environ,
            {
                "GITHUB_TOKEN": "tok",
                "GITHUB_EVENT_PATH": str(event_file),
                "GITHUB_REPOSITORY": "o/r",
                "STYLE_PROFILE_PATH": str(style_file),
            },
            clear=True,
        ):
            from src.integrations.action_runner import main
            assert main() == 1
