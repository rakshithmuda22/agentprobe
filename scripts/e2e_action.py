#!/usr/bin/env python3
"""End-to-end test of the AgentProbe GitHub Action, without GitHub.

1. Builds a real git repo from examples/acme-shop (main + one branch per PR).
2. Serves the GitHub REST endpoints the action uses (pulls, files, contents,
   issue comments, commit statuses) from a local stub backed by that repo,
   and records every comment/status the action posts.
3. Runs the action once per PR, either as the Docker image exactly the way
   the Actions runner starts Docker actions (``--runner docker``), or as the
   installed ``agentprobe action`` entrypoint (``--runner local``).
4. Checks verdicts and exit codes against examples/acme-shop/README.md.

    python scripts/e2e_action.py --runner docker   # needs: docker build -t agentprobe:local .
    python scripts/e2e_action.py --runner local    # needs: pip install .
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse, parse_qs

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "acme-shop"
OWNER, REPO = "acme", "shop"
FULL = f"{OWNER}/{REPO}"
EXPECTED = {1: ("BLOCK", 1), 2: ("WARN", 0), 3: ("PASS", 0)}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout


def build_repo(dest: Path) -> dict[int, dict]:
    """main = base/, branch pr-N = base/ + prs/N-*/ overlay. Returns PR metadata."""
    shutil.copytree(EXAMPLE / "base", dest)
    git(dest, "init", "-q", "-b", "main")
    ident = ["-c", "user.name=acme-dev", "-c", "user.email=dev@acme.test"]
    git(dest, "add", "-A")
    git(dest, *ident, "commit", "-qm", "Initial shop")
    base_sha = git(dest, "rev-parse", "HEAD").strip()
    prs = {}
    for pr_dir in sorted((EXAMPLE / "prs").iterdir()):
        number = int(pr_dir.name.split("-", 1)[0])
        branch = f"pr-{pr_dir.name}"
        git(dest, "checkout", "-q", "-b", branch, "main")
        shutil.copytree(pr_dir, dest, dirs_exist_ok=True)
        git(dest, "add", "-A")
        git(dest, *ident, "commit", "-qm", pr_dir.name.split("-", 1)[1].replace("-", " "))
        prs[number] = {"branch": branch, "title": pr_dir.name,
                       "base": base_sha, "head": git(dest, "rev-parse", "HEAD").strip()}
    git(dest, "checkout", "-q", "main")
    return prs


class FakeGitHub(BaseHTTPRequestHandler):
    """Just enough of api.github.com for PyGithub + the action."""

    repo: Path
    prs: dict[int, dict]
    base_url: str
    comments: list[dict] = []
    statuses: list[dict] = []
    lock = threading.Lock()

    def log_message(self, *args):  # keep the transcript clean
        pass

    def _send(self, code: int, body) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def _pr_json(self, n: int) -> dict:
        pr = self.prs[n]
        api = f"{self.base_url}/repos/{FULL}"
        return {"number": n, "id": n, "state": "open", "title": pr["title"],
                "url": f"{api}/pulls/{n}", "issue_url": f"{api}/issues/{n}",
                "head": {"sha": pr["head"], "ref": pr["branch"]},
                "base": {"sha": pr["base"], "ref": "main"}}

    def _files(self, n: int) -> list[dict]:
        pr = self.prs[n]
        out = []
        for line in git(self.repo, "diff", "--no-renames", "--name-status",
                        f"{pr['base']}..{pr['head']}").splitlines():
            status, path = line.split("\t", 1)
            patch = git(self.repo, "diff", "--no-renames", f"{pr['base']}..{pr['head']}",
                        "--", path)
            patch = patch[patch.index("@@"):] if "@@" in patch else None
            out.append({"filename": path, "patch": patch, "sha": "0" * 40,
                        "status": {"A": "added", "D": "removed"}.get(status[0], "modified")})
        return out

    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        parts = [unquote(p) for p in url.path.strip("/").split("/")]
        if parts[:3] != ["repos", OWNER, REPO]:
            return self._send(404, {"message": "Not Found"})
        rest = parts[3:]
        if not rest:
            return self._send(200, {"full_name": FULL, "name": REPO, "id": 1,
                                    "owner": {"login": OWNER},
                                    "url": f"{self.base_url}/repos/{FULL}"})
        if rest[0] == "pulls" and len(rest) == 2:
            return self._send(200, self._pr_json(int(rest[1])))
        if rest[0] == "pulls" and rest[2:] == ["files"]:
            return self._send(200, self._files(int(rest[1])))
        if rest[0] == "issues" and rest[2:] == ["comments"]:
            with self.lock:
                return self._send(200, [c for c in self.comments if c["pr"] == int(rest[1])])
        if rest[0] == "contents":
            path = "/".join(rest[1:])
            ref = parse_qs(url.query).get("ref", ["HEAD"])[0]
            proc = subprocess.run(["git", "-C", str(self.repo), "show", f"{ref}:{path}"],
                                  capture_output=True)
            if proc.returncode != 0:
                return self._send(404, {"message": "Not Found"})
            return self._send(200, {"type": "file", "encoding": "base64", "path": path,
                                    "name": path.rsplit("/", 1)[-1], "sha": "0" * 40,
                                    "size": len(proc.stdout),
                                    "content": base64.b64encode(proc.stdout).decode()})
        if rest[0] == "commits" and len(rest) == 2:
            return self._send(200, {"sha": rest[1],
                                    "url": f"{self.base_url}/repos/{FULL}/commits/{rest[1]}"})
        return self._send(404, {"message": "Not Found"})

    def do_POST(self):  # noqa: N802
        parts = urlparse(self.path).path.strip("/").split("/")[3:]
        body = self._body()
        with self.lock:
            if parts[0] == "issues" and parts[2:] == ["comments"]:
                cid = len(self.comments) + 1
                comment = {"id": cid, "pr": int(parts[1]), "body": body["body"],
                           "url": f"{self.base_url}/repos/{FULL}/issues/comments/{cid}"}
                self.comments.append(comment)
                return self._send(201, comment)
            if parts[0] == "statuses":
                self.statuses.append({"sha": parts[1], **body})
                return self._send(201, {"id": len(self.statuses), **body})
        return self._send(404, {"message": "Not Found"})

    def do_PATCH(self):  # noqa: N802
        parts = urlparse(self.path).path.strip("/").split("/")[3:]
        body = self._body()
        with self.lock:
            for c in self.comments:
                if parts[:2] == ["issues", "comments"] and c["id"] == int(parts[2]):
                    c["body"] = body["body"]
                    return self._send(200, c)
        return self._send(404, {"message": "Not Found"})


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def run_action(runner: str, image: str, repo: Path, work: Path, n: int, pr: dict,
               port: int) -> tuple[int, str, dict]:
    """Start the action the way the Actions runner does; return (exit, log, outputs)."""
    git(repo, "checkout", "-q", pr["branch"])  # actions/checkout of the PR
    gh = work / f"pr{n}"
    (gh / "workflow").mkdir(parents=True, exist_ok=True)
    (gh / "file_commands").mkdir(exist_ok=True)
    event = {"action": "opened", "number": n, "pull_request": {
        "number": n, "head": {"sha": pr["head"]}, "base": {"sha": pr["base"]}}}
    (gh / "workflow" / "event.json").write_text(json.dumps(event))
    for name in ("output", "summary"):
        (gh / "file_commands" / name).write_text("")

    if runner == "docker":
        inside = "/github"
        env = {
            "INPUT_GITHUB_TOKEN": "ghs_fake_token_for_e2e", "INPUT_CONFIG_DIR": ".agentprobe",
            "INPUT_POST_COMMENT": "true", "INPUT_FAIL_ON": "block",
            "GITHUB_EVENT_NAME": "pull_request", "GITHUB_REPOSITORY": FULL,
            "GITHUB_EVENT_PATH": f"{inside}/workflow/event.json",
            "GITHUB_WORKSPACE": f"{inside}/workspace",
            "GITHUB_OUTPUT": f"{inside}/file_commands/output",
            "GITHUB_STEP_SUMMARY": f"{inside}/file_commands/summary",
            "GITHUB_API_URL": f"http://host.docker.internal:{port}",
            "GITHUB_ACTIONS": "true", "CI": "true",
        }
        cmd = ["docker", "run", "--rm", "--workdir", f"{inside}/workspace",
               "--add-host", "host.docker.internal:host-gateway",
               "-v", f"{repo}:{inside}/workspace",
               "-v", f"{gh / 'workflow'}:{inside}/workflow",
               "-v", f"{gh / 'file_commands'}:{inside}/file_commands"]
        for k, v in env.items():
            cmd += ["-e", f"{k}={v}"]
        proc = subprocess.run(cmd + [image], capture_output=True, text=True)
    else:
        env = {
            "PATH": os.environ.get("PATH", ""), "HOME": str(gh),
            "INPUT_GITHUB_TOKEN": "ghs_fake_token_for_e2e",
            "GITHUB_EVENT_NAME": "pull_request", "GITHUB_REPOSITORY": FULL,
            "GITHUB_EVENT_PATH": str(gh / "workflow" / "event.json"),
            "GITHUB_WORKSPACE": str(repo),
            "GITHUB_OUTPUT": str(gh / "file_commands" / "output"),
            "GITHUB_STEP_SUMMARY": str(gh / "file_commands" / "summary"),
            "GITHUB_API_URL": f"http://127.0.0.1:{port}",
        }
        exe = shutil.which("agentprobe") or str(Path(sys.executable).with_name("agentprobe"))
        proc = subprocess.run([exe, "action"], cwd=repo, env=env, capture_output=True,
                              text=True)

    outputs = {}
    for line in (gh / "file_commands" / "output").read_text().splitlines():
        if "=" in line and "<<" not in line:
            k, v = line.split("=", 1)
            outputs[k] = v
    return proc.returncode, proc.stdout + proc.stderr, outputs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", choices=("docker", "local"), default="docker")
    ap.add_argument("--image", default="agentprobe:local")
    ap.add_argument("--keep", help="copy the generated repo + logs here")
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="agentprobe-e2e-"))
    repo = work / "acme-shop"
    prs = build_repo(repo)

    port = free_port()
    FakeGitHub.repo, FakeGitHub.prs = repo, prs
    FakeGitHub.base_url = (f"http://host.docker.internal:{port}" if args.runner == "docker"
                           else f"http://127.0.0.1:{port}")
    server = ThreadingHTTPServer(("0.0.0.0", port), FakeGitHub)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    failures = 0
    for n, pr in sorted(prs.items()):
        code, log, outputs = run_action(args.runner, args.image, repo, work, n, pr, port)
        want_verdict, want_code = EXPECTED[n]
        ok = outputs.get("verdict") == want_verdict and code == want_code
        failures += not ok
        print(f"\n=== PR #{n} {pr['title']} ({args.runner}) ===")
        for line in log.splitlines():
            if line.startswith(("AgentProbe verdict", "::error", "::warning", "PR comment")):
                print(line)
            elif line.startswith("Traceback") or "Error" in line.split(":", 1)[0]:
                print(line)
        status = [s for s in FakeGitHub.statuses if s["sha"] == pr["head"]]
        print(f"exit code: {code} | outputs: verdict={outputs.get('verdict')} "
              f"score={outputs.get('score')} | commit status: "
              f"{status[-1]['state'] if status else 'none'} | "
              f"{'OK' if ok else f'MISMATCH (expected {want_verdict}/exit {want_code})'}")
        (work / f"pr{n}.log").write_text(log)

    for c in FakeGitHub.comments:
        (work / f"pr{c['pr']}-comment.md").write_text(c["body"])
    server.shutdown()
    print(f"\n{len(prs) - failures}/{len(prs)} PRs matched the expected verdict and exit "
          f"code. Comments + logs: {args.keep or work}")
    if args.keep:
        shutil.copytree(work, args.keep, dirs_exist_ok=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
