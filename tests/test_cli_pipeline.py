"""End-to-end tests of the local CLI on a real git repository."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agentprobe.cli import main
from agentprobe.integrations.pipeline import materialize_policy

BOUNDARIES = """\
modules:
  payments:
    allowed_imports: [shared]
    forbidden_imports: [analytics]
"""
STYLE = "naming:\n  functions: snake_case\n  classes: PascalCase\n  files: snake_case\n"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True).stdout


def _commit(repo: Path, files: dict[str, str], message: str) -> None:
    for path, text in files.items():
        p = repo / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", message)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "shop"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _commit(r, {
        ".agentprobe/boundaries.yaml": BOUNDARIES,
        ".agentprobe/style-profile.yaml": STYLE,
        "src/payments/charge.py": "def charge(amount):\n    return amount\n",
        "src/analytics/tracker.py": "def track(name):\n    return name\n",
    }, "base")
    return r


def _check(repo: Path, head: str, capsys) -> tuple[int, dict]:
    code = main(["check", "--repo", str(repo), "--base", "main", "--head", head, "--json"])
    return code, json.loads(capsys.readouterr().out)


def test_boundary_violation_blocks(repo, capsys):
    _git(repo, "checkout", "-q", "-b", "bad")
    _commit(repo, {"src/payments/charge.py":
                   "from src.analytics.tracker import track\n\n\n"
                   "def charge(amount):\n    track('charge')\n    return amount\n"}, "bad")
    code, out = _check(repo, "bad", capsys)
    assert code == 1
    assert out["verdict"] == "BLOCK"
    assert out["architecture"]["violations"][0]["line"] == 1


def test_naming_violation_warns_but_exits_0(repo, capsys):
    _git(repo, "checkout", "-q", "-b", "naming")
    _commit(repo, {"src/payments/refund.py": "def refundOrder(order):\n    return order\n"},
            "naming")
    code, out = _check(repo, "naming", capsys)
    assert code == 0
    assert out["verdict"] == "WARN"
    assert out["pattern"]["violations"][0]["rule_violated"] == "naming.functions"


def test_clean_pr_passes(repo, capsys):
    _git(repo, "checkout", "-q", "-b", "clean")
    _commit(repo, {"src/payments/refund.py": "def refund_order(order):\n    return order\n"},
            "clean")
    code, out = _check(repo, "clean", capsys)
    assert (code, out["verdict"], out["score"]) == (0, "PASS", 0.0)


def test_pr_cannot_relax_its_own_policy(repo, capsys):
    """The PR deletes the forbidden rule *and* violates it: base policy still applies."""
    _git(repo, "checkout", "-q", "-b", "sneaky")
    _commit(repo, {
        ".agentprobe/boundaries.yaml": "modules: {}\n",
        "src/payments/charge.py": "from src.analytics.tracker import track\n\n\n"
                                  "def charge(amount):\n    return amount\n",
    }, "sneaky")
    code, out = _check(repo, "sneaky", capsys)
    assert out["policy_source"] == "base"
    assert out["verdict"] == "BLOCK"
    descriptions = [v["description"] for v in out["architecture"]["violations"]]
    assert any("governance policy file" in d for d in descriptions)
    assert any("forbidden from importing 'analytics'" in d for d in descriptions)


def test_body_only_change_is_seen_by_regression_analyzer(repo, capsys):
    _commit(repo, {"src/payments/charge.py":
                   "def charge(amount):\n    if amount <= 0:\n        raise ValueError(amount)\n"
                   + "    x = 1\n" * 10 + "    return amount\n"}, "longer charge")
    _git(repo, "checkout", "-q", "-b", "regress")
    text = (repo / "src/payments/charge.py").read_text().replace(
        "raise ValueError(amount)", "return None")
    _commit(repo, {"src/payments/charge.py": text}, "regress")
    code, out = _check(repo, "regress", capsys)
    deltas = out["regression"]["deltas"]
    assert any(d["function"] == "charge" and d["field"] == "error_paths" for d in deltas)
    assert out["verdict"] == "WARN"


def test_markdown_output(repo, capsys):
    code = main(["check", "--repo", str(repo), "--base", "main", "--head", "main"])
    text = capsys.readouterr().out
    assert code == 0
    assert "AgentProbe Verdict: **PASS**" in text


def test_bad_ref_exits_2(repo, capsys):
    assert main(["check", "--repo", str(repo), "--base", "nope", "--head", "HEAD"]) == 2


def test_materialize_policy_falls_back_to_workspace(tmp_path):
    ws = tmp_path / ".agentprobe"
    ws.mkdir()
    (ws / "boundaries.yaml").write_text("modules: {}\n")
    target = tmp_path / "out"
    assert materialize_policy(lambda p, r: None, "abc", ".agentprobe", target, ws) == "workspace"
    assert (target / "boundaries.yaml").read_text() == "modules: {}\n"
    assert materialize_policy(None, None, ".agentprobe", tmp_path / "o2", None) == "none"


def test_ignore_paths_skip_files(repo, capsys):
    _commit(repo, {".agentprobe/config.yaml": "ignore_paths: ['vendor/*']\n"}, "ignore vendor")
    _git(repo, "checkout", "-q", "-b", "vendored")
    _commit(repo, {"vendor/lib.py": "def badName():\n    return 1\n",
                   "src/payments/ok.py": "def fine():\n    return 1\n"}, "vendored")
    code, out = _check(repo, "vendored", capsys)
    assert (code, out["verdict"]) == (0, "PASS")
