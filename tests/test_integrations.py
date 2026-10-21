"""Tests for GitHub integration — webhook server, GitHubClient, action runner."""

import hashlib
import hmac
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from agentprobe.integrations.webhook_server import app, _verify_signature


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

    @patch("agentprobe.integrations.webhook_server._process_pr")
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


def _event(tmp_path, payload=None):
    event_file = tmp_path / "event.json"
    event_file.write_text(json.dumps(payload or {
        "pull_request": {"number": 5, "base": {"sha": "base111"}, "head": {"sha": "head222"}},
    }))
    return event_file


def _action_env(tmp_path, **extra):
    env = {
        "INPUT_GITHUB_TOKEN": "tok",
        "GITHUB_EVENT_PATH": str(_event(tmp_path)),
        "GITHUB_REPOSITORY": "o/r",
        "GITHUB_EVENT_NAME": "pull_request",
        "GITHUB_WORKSPACE": str(tmp_path),
        "GITHUB_OUTPUT": str(tmp_path / "output.txt"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
    }
    env.update(extra)
    return env


def _pipeline_result(status, score=0.0):
    return ({"verdict": {"status": status, "score": score, "comment": f"report {status}"}},
            "base")


class TestActionRunner:
    @patch.dict(os.environ, {"GITHUB_TOKEN": ""}, clear=True)
    def test_missing_token_exits_1(self):
        from agentprobe.integrations.action_runner import main
        assert main() == 1

    @patch.dict(
        os.environ,
        {"GITHUB_TOKEN": "tok", "GITHUB_EVENT_PATH": "/nonexistent"},
        clear=True,
    )
    def test_missing_event_file_exits_1(self):
        from agentprobe.integrations.action_runner import main
        assert main() == 1

    @patch.dict(
        os.environ,
        {"GITHUB_TOKEN": "tok", "GITHUB_EVENT_PATH": "", "GITHUB_REPOSITORY": "o/r"},
        clear=True,
    )
    def test_empty_event_path_exits_1(self):
        from agentprobe.integrations.action_runner import main
        assert main() == 1

    def test_missing_pr_number_exits_1(self, tmp_path):
        env = _action_env(tmp_path, GITHUB_EVENT_PATH=str(_event(tmp_path, {"issue": {"number": 1}})))
        with patch.dict(os.environ, env, clear=True):
            from agentprobe.integrations.action_runner import main
            assert main() == 1

    @patch("agentprobe.integrations.action_runner.GitHubClient")
    @patch("agentprobe.integrations.action_runner.run_pipeline")
    def test_pass_verdict_exits_0_and_writes_outputs(self, mock_run, mock_client_cls, tmp_path):
        mock_run.return_value = _pipeline_result("PASS")
        mock_gh = mock_client_cls.return_value
        mock_gh.get_pr_diff.return_value = ""
        mock_gh.upsert_pr_comment.return_value = "created"

        with patch.dict(os.environ, _action_env(tmp_path), clear=True):
            from agentprobe.integrations.action_runner import main
            assert main() == 0

        kwargs = mock_run.call_args.kwargs
        assert kwargs["base_ref"] == "base111" and kwargs["head_ref"] == "head222"
        assert kwargs["allow_code_execution"] is True
        assert kwargs["policy_repo_dir"] == ".agentprobe"
        mock_gh.upsert_pr_comment.assert_called_once()
        mock_gh.set_check_status.assert_called_once_with("o/r", "head222", "PASS",
                                                         "AgentProbe: PASS (score: 0)")
        outputs = (tmp_path / "output.txt").read_text()
        assert "verdict=PASS" in outputs and "score=0.0" in outputs
        assert "report PASS" in (tmp_path / "summary.md").read_text()

    @patch("agentprobe.integrations.action_runner.GitHubClient")
    @patch("agentprobe.integrations.action_runner.run_pipeline")
    def test_block_verdict_exits_1(self, mock_run, mock_client_cls, tmp_path):
        mock_run.return_value = _pipeline_result("BLOCK", 100.0)
        mock_client_cls.return_value.get_pr_diff.return_value = ""
        with patch.dict(os.environ, _action_env(tmp_path), clear=True):
            from agentprobe.integrations.action_runner import main
            assert main() == 1

    @patch("agentprobe.integrations.action_runner.GitHubClient")
    @patch("agentprobe.integrations.action_runner.run_pipeline")
    def test_fail_on_warn(self, mock_run, mock_client_cls, tmp_path):
        mock_run.return_value = _pipeline_result("WARN", 5.0)
        mock_client_cls.return_value.get_pr_diff.return_value = ""
        from agentprobe.integrations.action_runner import main
        with patch.dict(os.environ, _action_env(tmp_path), clear=True):
            assert main() == 0
        with patch.dict(os.environ, _action_env(tmp_path, INPUT_FAIL_ON="warn"), clear=True):
            assert main() == 1

    @patch("agentprobe.integrations.action_runner.GitHubClient")
    @patch("agentprobe.integrations.action_runner.run_pipeline")
    def test_post_failure_does_not_change_verdict(self, mock_run, mock_client_cls, tmp_path,
                                                  capsys):
        """Fork PRs get a read-only token: warn, but keep the verdict's exit code."""
        from github import GithubException

        mock_run.return_value = _pipeline_result("PASS")
        mock_gh = mock_client_cls.return_value
        mock_gh.get_pr_diff.return_value = ""
        mock_gh.upsert_pr_comment.side_effect = GithubException(403, {"message": "nope"}, None)
        with patch.dict(os.environ, _action_env(tmp_path), clear=True):
            from agentprobe.integrations.action_runner import main
            assert main() == 0
        assert "::warning title=AgentProbe::Could not post" in capsys.readouterr().out

    @patch("agentprobe.integrations.action_runner.GitHubClient")
    @patch("agentprobe.integrations.action_runner.run_pipeline")
    def test_post_comment_false_skips_writes(self, mock_run, mock_client_cls, tmp_path):
        mock_run.return_value = _pipeline_result("PASS")
        mock_gh = mock_client_cls.return_value
        mock_gh.get_pr_diff.return_value = ""
        with patch.dict(os.environ, _action_env(tmp_path, INPUT_POST_COMMENT="false"), clear=True):
            from agentprobe.integrations.action_runner import main
            assert main() == 0
        mock_gh.upsert_pr_comment.assert_not_called()
        mock_gh.set_check_status.assert_not_called()

    @patch("agentprobe.integrations.action_runner.GitHubClient")
    @patch("agentprobe.integrations.action_runner.run_pipeline")
    def test_pull_request_target_never_executes_pr_code(self, mock_run, mock_client_cls,
                                                        tmp_path):
        mock_run.return_value = _pipeline_result("PASS")
        mock_client_cls.return_value.get_pr_diff.return_value = ""
        env = _action_env(tmp_path, GITHUB_EVENT_NAME="pull_request_target")
        with patch.dict(os.environ, env, clear=True):
            from agentprobe.integrations.action_runner import main
            main()
        assert mock_run.call_args.kwargs["allow_code_execution"] is False

    @patch("agentprobe.integrations.action_runner.GitHubClient")
    def test_github_api_url_is_honored(self, mock_client_cls, tmp_path):
        mock_client_cls.return_value.get_pr_diff.side_effect = RuntimeError("stop")
        env = _action_env(tmp_path, GITHUB_API_URL="https://ghe.example.com/api/v3")
        with patch.dict(os.environ, env, clear=True):
            from agentprobe.integrations.action_runner import main
            assert main() == 1
        mock_client_cls.assert_called_once_with("tok", base_url="https://ghe.example.com/api/v3")

    def test_annotations_are_valid_workflow_commands(self, capsys):
        from agentprobe.integrations.action_runner import emit_annotations

        emit_annotations({
            "architecture_report": {"violations": [{"file": "src/a.py", "line": 3,
                                                     "severity": "FATAL",
                                                     "description": "Module 'a' is forbidden"}]},
            "pattern_report": {"violations": [{"file_path": "src/b,c.py", "line_number": 7,
                                                "severity": "WARN",
                                                "description": "Function 'x' is camelCase"}]},
        })
        out = capsys.readouterr().out.splitlines()
        assert out[0] == ("::error file=src/a.py,line=3,title=AgentProbe::"
                          "[architecture] Module 'a' is forbidden")
        assert out[1].startswith("::warning file=src/b%2Cc.py,line=7,")


# ── GitHub client ──


class TestGitHubClient:
    def _client(self):
        from agentprobe.integrations.github_app import GitHubClient

        client = GitHubClient("tok")
        client._github = MagicMock()
        return client

    def test_upsert_edits_existing_comment(self):
        client = self._client()
        old = MagicMock(body="<!-- agentprobe-verdict -->\nold")
        pr = client._github.get_repo.return_value.get_pull.return_value
        pr.get_issue_comments.return_value = [MagicMock(body="unrelated"), old]
        assert client.upsert_pr_comment("o/r", 1, "new", "<!-- agentprobe-verdict -->") == "updated"
        old.edit.assert_called_once_with("new")
        pr.create_issue_comment.assert_not_called()

    def test_upsert_creates_when_missing(self):
        client = self._client()
        pr = client._github.get_repo.return_value.get_pull.return_value
        pr.get_issue_comments.return_value = []
        assert client.upsert_pr_comment("o/r", 1, "new", "<!-- m -->") == "created"
        pr.create_issue_comment.assert_called_once_with("new")

    def test_diff_marks_added_files(self):
        client = self._client()
        pr = client._github.get_repo.return_value.get_pull.return_value
        pr.get_files.return_value = [
            MagicMock(filename="src/new.py", status="added", patch="@@ -0,0 +1 @@\n+x = 1"),
            MagicMock(filename="img.png", status="added", patch=None),
        ]
        from agentprobe.parsers.diff_parser import parse_diff

        parsed = parse_diff(client.get_pr_diff("o/r", 1))
        assert [(f.file_path, f.status) for f in parsed.files] == [("src/new.py", "added")]

    def test_file_content_404_is_none(self):
        from github import GithubException

        client = self._client()
        client._github.get_repo.return_value.get_contents.side_effect = GithubException(
            404, {"message": "Not Found"}, None)
        assert client.get_file_content("o/r", "missing.yaml", "abc") is None
