"""End-to-end integration tests — full pipeline from diff to verdict."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.graph.state import AgentProbeState
from src.graph.workflow import run_agentprobe


def _make_state(pr_diff: str, repo_path: str = ".") -> AgentProbeState:
    return AgentProbeState(
        pr_diff=pr_diff,
        repo_path=repo_path,
        pr_number=1,
        repo_full_name="test/repo",
        architecture_report=None,
        pattern_report=None,
        regression_report=None,
        short_circuit=False,
        verdict=None,
        cached_functions=[],
        cost_tracker={},
    )


BOUNDARY_VIOLATION_DIFF = """\
diff --git a/src/payments/checkout.py b/src/payments/checkout.py
--- a/src/payments/checkout.py
+++ b/src/payments/checkout.py
@@ -1,3 +1,5 @@
+from src.analytics.tracker import track_event
+
 def process_checkout(cart):
     total = sum(item.price for item in cart.items)
     return {"total": total}
"""

NAMING_VIOLATION_DIFF = """\
diff --git a/utils/helpers.py b/utils/helpers.py
--- a/utils/helpers.py
+++ b/utils/helpers.py
@@ -1,3 +1,6 @@
+def calculateTotal(items):
+    return sum(items)
+
 def existing_function():
     pass
"""

SEMANTIC_REGRESSION_DIFF = """\
diff --git a/src/payments/process_payment.py b/src/payments/process_payment.py
--- a/src/payments/process_payment.py
+++ b/src/payments/process_payment.py
@@ -1,7 +1,5 @@
-class PaymentError(Exception):
-    pass
-
-def process_payment(amount, currency="USD"):
-    if amount <= 0:
-        raise PaymentError(f"Invalid amount: {amount}")
-    return {"status": "ok", "amount": amount, "currency": currency}
+def process_payment(amount, currency="USD"):
+    if amount <= 0:
+        return None
+    return {"status": "ok", "amount": amount, "currency": currency}
"""

CLEAN_DIFF = """\
diff --git a/src/utils/helpers.py b/src/utils/helpers.py
--- a/src/utils/helpers.py
+++ b/src/utils/helpers.py
@@ -1,3 +1,6 @@
+def format_name(first, last):
+    return f"{first} {last}"
+
 def existing_function():
     pass
"""


class TestEndToEnd:
    def test_boundary_violation_blocks(self, tmp_path):
        """Architecture FATAL should short-circuit to BLOCK."""
        # Set up boundaries that forbid payments -> analytics
        agentprobe_dir = tmp_path / ".agentprobe"
        agentprobe_dir.mkdir()
        (agentprobe_dir / "boundaries.yaml").write_text(
            "modules:\n"
            "  payments:\n"
            "    allowed_imports: [shared, utils]\n"
            "    forbidden_imports: [analytics]\n"
            "layers: []\n"
        )
        (agentprobe_dir / "config.yaml").write_text("")
        (agentprobe_dir / "style-profile.yaml").write_text(
            "naming:\n  functions: snake_case\n  classes: PascalCase\n  files: snake_case\n"
            "imports:\n  order: [builtin, external, internal, relative]\n"
            "forbidden: []\n"
        )

        state = _make_state(BOUNDARY_VIOLATION_DIFF, str(tmp_path))
        result = run_agentprobe(state)

        assert result["verdict"]["status"] == "BLOCK"
        assert result["short_circuit"] is True
        assert result["architecture_report"]["status"] == "FATAL"

    def test_clean_pr_passes(self, tmp_path):
        """A clean diff with no violations should PASS."""
        agentprobe_dir = tmp_path / ".agentprobe"
        agentprobe_dir.mkdir()
        (agentprobe_dir / "boundaries.yaml").write_text(
            "modules: []\nlayers: []\nlayer_rules: {}\n"
        )
        (agentprobe_dir / "config.yaml").write_text("")
        (agentprobe_dir / "style-profile.yaml").write_text(
            "naming:\n  functions: snake_case\n  classes: PascalCase\n  files: snake_case\n"
            "imports:\n  order: [builtin, external, internal, relative]\n"
            "forbidden: []\n"
        )

        state = _make_state(CLEAN_DIFF, str(tmp_path))
        result = run_agentprobe(state)

        assert result["verdict"]["status"] == "PASS"
        assert result["verdict"]["score"] == 0.0

    def test_semantic_regression_warns(self, tmp_path):
        """Semantic change (raise -> return None) should produce WARN."""
        agentprobe_dir = tmp_path / ".agentprobe"
        agentprobe_dir.mkdir()
        (agentprobe_dir / "boundaries.yaml").write_text(
            "modules: []\nlayers: []\nlayer_rules: {}\n"
        )
        (agentprobe_dir / "config.yaml").write_text(
            "regression:\n  critical_dirs: []\n  max_llm_calls_per_pr: 5\n"
        )
        (agentprobe_dir / "style-profile.yaml").write_text(
            "naming:\n  functions: snake_case\n  classes: PascalCase\n  files: snake_case\n"
            "imports:\n  order: [builtin, external, internal, relative]\n"
            "forbidden: []\n"
        )

        state = _make_state(SEMANTIC_REGRESSION_DIFF, str(tmp_path))
        result = run_agentprobe(state)

        # Regression agent should detect the error_paths change
        reg = result["regression_report"]
        assert reg["status"] == "WARN" or len(reg["deltas"]) > 0

    def test_verdict_comment_is_markdown(self, tmp_path):
        """Verdict comment should be valid markdown with emoji badges."""
        agentprobe_dir = tmp_path / ".agentprobe"
        agentprobe_dir.mkdir()
        (agentprobe_dir / "boundaries.yaml").write_text(
            "modules: []\nlayers: []\nlayer_rules: {}\n"
        )
        (agentprobe_dir / "config.yaml").write_text("")
        (agentprobe_dir / "style-profile.yaml").write_text(
            "naming:\n  functions: snake_case\nimports:\n  order: []\nforbidden: []\n"
        )

        state = _make_state(CLEAN_DIFF, str(tmp_path))
        result = run_agentprobe(state)

        comment = result["verdict"]["comment"]
        assert "## " in comment  # has markdown headers
        assert "AgentProbe Verdict" in comment
        assert "PASS" in comment
