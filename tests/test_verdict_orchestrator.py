"""Tests for the Verdict Orchestrator."""

from pathlib import Path

import pytest

from agentprobe.agents.verdict_orchestrator import run, build_comment, STATUS_SCORES
from agentprobe.graph.state import AgentProbeState


def _make_state(**overrides) -> AgentProbeState:
    base = AgentProbeState(
        pr_diff="",
        repo_path=str(Path(__file__).parent / "fixtures" / "sample_policy"),
        pr_number=1,
        repo_full_name="owner/repo",
        architecture_report=None,
        pattern_report=None,
        regression_report=None,
        short_circuit=False,
        verdict=None,
        cached_functions=[],
        cost_tracker={},
    )
    base.update(overrides)
    return base


class TestVerdictScoring:
    def test_all_pass_verdict(self):
        state = _make_state(
            architecture_report={"status": "PASS", "violations": []},
            pattern_report={"status": "PASS", "violations": []},
            regression_report={"status": "PASS", "deltas": [], "confidence": 1.0},
        )
        result = run(state)
        assert result["verdict"]["status"] == "PASS"
        assert result["verdict"]["score"] == 0.0

    def test_any_fatal_finding_blocks(self):
        """A FATAL finding blocks even without the short-circuit flag."""
        state = _make_state(
            architecture_report={"status": "FATAL", "violations": [{"file_path": "a.py", "line_number": 1, "description": "bad", "severity": "FATAL"}]},
            pattern_report={"status": "WARN", "violations": []},
            regression_report={"status": "WARN", "deltas": [], "confidence": 0.8},
        )
        result = run(state)
        assert result["verdict"]["status"] == "BLOCK"
        assert result["verdict"]["score"] == 100.0

    def test_fatal_forbidden_pattern_blocks(self):
        state = _make_state(
            architecture_report={"status": "PASS", "violations": []},
            pattern_report={"status": "FATAL", "violations": [{"file_path": "a.py", "line_number": 3, "description": "eval", "severity": "FATAL"}]},
        )
        assert run(state)["verdict"]["status"] == "BLOCK"

    def test_short_circuit_forces_block(self):
        state = _make_state(
            architecture_report={"status": "FATAL", "violations": [{"file_path": "a.py", "line_number": 1, "description": "forbidden import", "severity": "FATAL"}]},
            pattern_report=None,
            regression_report=None,
            short_circuit=True,
        )
        result = run(state)
        assert result["verdict"]["status"] == "BLOCK"
        assert result["verdict"]["score"] >= 100.0

    def test_single_finding_is_never_silently_passed(self):
        """One naming violation must surface as WARN (it used to score 12.5 -> PASS)."""
        state = _make_state(
            architecture_report={"status": "PASS", "violations": []},
            pattern_report={"status": "WARN", "violations": [{"file_path": "b.py", "line_number": 5, "description": "wrong naming", "severity": "WARN"}]},
            regression_report={"status": "PASS", "deltas": [], "confidence": 1.0},
        )
        result = run(state)
        # pattern 20 points * 0.25 = 5.0 > warn threshold 0
        assert result["verdict"]["score"] == 5.0
        assert result["verdict"]["status"] == "WARN"

    def test_warn_threshold_is_configurable(self, tmp_path):
        (tmp_path / ".agentprobe").mkdir()
        (tmp_path / ".agentprobe" / "config.yaml").write_text("thresholds:\n  warn: 10\n")
        state = _make_state(
            repo_path=str(tmp_path),
            pattern_report={"status": "WARN", "violations": [{"file_path": "b.py", "line_number": 5, "description": "n", "severity": "WARN"}]},
        )
        assert run(state)["verdict"]["status"] == "PASS"

    def test_many_warnings_escalate_to_block(self):
        many = [{"file_path": "f.py", "line_number": i, "description": "x", "severity": "WARN"} for i in range(5)]
        high = [{"function": "f", "field": "error_paths", "severity": "HIGH"} for _ in range(3)]
        state = _make_state(
            architecture_report={"status": "WARN", "violations": many},
            pattern_report={"status": "WARN", "violations": many},
            regression_report={"status": "WARN", "deltas": high, "confidence": 1.0},
        )
        result = run(state)
        # 100*0.40 + 100*0.25 + 100*0.35 = 100 > 70
        assert result["verdict"]["status"] == "BLOCK"
        assert result["verdict"]["analyzer_scores"] == {"architecture": 100.0, "pattern": 100.0, "regression": 100.0}

    def test_all_fatal_gives_block(self):
        state = _make_state(
            architecture_report={"status": "BLOCK", "violations": []},
            pattern_report={"status": "WARN", "violations": []},
            regression_report={"status": "BLOCK", "deltas": [], "confidence": 0.5},
        )
        result = run(state)
        assert result["verdict"]["status"] == "BLOCK"
        assert result["verdict"]["score"] > 70

    def test_none_reports_default_to_pass(self):
        state = _make_state()
        result = run(state)
        assert result["verdict"]["status"] == "PASS"
        assert result["verdict"]["score"] == 0.0


class TestBuildComment:
    def test_comment_contains_status_and_score(self):
        verdict = {"status": "PASS", "score": 0.0}
        state = _make_state()
        comment = build_comment(verdict, state)
        assert "PASS" in comment
        assert "0.0" in comment

    def test_comment_includes_violations(self):
        state = _make_state(
            architecture_report={
                "status": "FATAL",
                "violations": [{"file_path": "x.py", "line_number": 10, "description": "forbidden", "severity": "FATAL"}],
            },
        )
        verdict = {"status": "BLOCK", "score": 100.0}
        comment = build_comment(verdict, state)
        assert "x.py" in comment
        assert "FATAL" in comment

    def test_comment_includes_regression_deltas(self):
        state = _make_state(
            regression_report={
                "status": "WARN",
                "deltas": [{"function": "process_payment", "field": "error_handling", "severity": "WARN", "before": "raises", "after": "returns None"}],
                "confidence": 0.9,
            },
        )
        verdict = {"status": "WARN", "score": 50.0}
        comment = build_comment(verdict, state)
        assert "process_payment" in comment
        assert "Semantic Changes" in comment

    def test_comment_caps_violations_at_10(self):
        violations = [
            {"file_path": f"file_{i}.py", "line_number": i, "description": f"issue {i}", "severity": "WARN"}
            for i in range(15)
        ]
        state = _make_state(
            architecture_report={"status": "WARN", "violations": violations},
        )
        verdict = {"status": "WARN", "score": 50.0}
        comment = build_comment(verdict, state)
        assert "and 5 more" in comment

    def test_comment_with_cost_tracker(self):
        state = _make_state(
            regression_report={"status": "PASS", "deltas": [], "confidence": 1.0},
            cost_tracker={"regression_llm_calls": 3},
        )
        verdict = {"status": "PASS", "score": 0.0}
        comment = build_comment(verdict, state)
        assert "LLM calls: 3" in comment
