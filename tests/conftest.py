"""Shared pytest fixtures for AgentProbe tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.graph.state import AgentProbeState

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def boundary_violation_repo() -> Path:
    return FIXTURES_DIR / "boundary_violation_repo"


@pytest.fixture
def clean_repo() -> Path:
    return FIXTURES_DIR / "clean_repo"


@pytest.fixture
def minimal_state() -> AgentProbeState:
    """Create a minimal AgentProbeState with mock data."""
    return AgentProbeState(
        pr_diff="",
        repo_path=".",
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


@pytest.fixture
def forbidden_import_diff() -> str:
    """A diff where payments imports from analytics (FORBIDDEN)."""
    return """diff --git a/src/payments/processor.py b/src/payments/processor.py
new file mode 100644
--- /dev/null
+++ b/src/payments/processor.py
@@ -0,0 +1,8 @@
+from src.analytics.tracker import track_event
+from src.shared.utils import validate
+
+
+def process_payment(amount, currency):
+    track_event("payment_started")
+    validate(amount)
+    return {"status": "ok"}
"""


@pytest.fixture
def clean_import_diff() -> str:
    """A diff with only allowed imports."""
    return """diff --git a/src/shared/utils.py b/src/shared/utils.py
new file mode 100644
--- /dev/null
+++ b/src/shared/utils.py
@@ -0,0 +1,5 @@
+from src.types.common import Amount
+
+
+def validate(amount):
+    return amount > 0
"""


@pytest.fixture
def deep_import_diff() -> str:
    """A diff with an import that exceeds max depth."""
    return """diff --git a/src/payments/processor.py b/src/payments/processor.py
new file mode 100644
--- /dev/null
+++ b/src/payments/processor.py
@@ -0,0 +1,4 @@
+from src.shared.utils.validators.amount import check
+
+def process():
+    return check(100)
"""


@pytest.fixture
def naming_violation_repo() -> Path:
    return FIXTURES_DIR / "naming_violation_repo"


@pytest.fixture
def semantic_regression_repo() -> Path:
    return FIXTURES_DIR / "semantic_regression_repo"


@pytest.fixture
def naming_violation_diff() -> str:
    """A diff with snake_case functions in a camelCase codebase."""
    return """diff --git a/src/user_service.py b/src/user_service.py
new file mode 100644
--- /dev/null
+++ b/src/user_service.py
@@ -0,0 +1,12 @@
+import os
+from src.models.user import User
+import json
+
+
+def get_user_by_id(user_id):
+    return User.find(user_id)
+
+
+def update_user_profile(user_id, data):
+    user = get_user_by_id(user_id)
+    return user.update(data)
"""


@pytest.fixture
def clean_python_diff() -> str:
    """A diff with proper snake_case naming (matches clean_repo profile)."""
    return """diff --git a/src/utils.py b/src/utils.py
new file mode 100644
--- /dev/null
+++ b/src/utils.py
@@ -0,0 +1,10 @@
+import os
+import json
+
+from src.models.base import BaseModel
+
+
+def validate_input(data):
+    return bool(data)
+
+
"""


@pytest.fixture
def forbidden_pattern_diff() -> str:
    """A diff with console.log (forbidden pattern)."""
    return """diff --git a/src/debug-helper.js b/src/debug-helper.js
new file mode 100644
--- /dev/null
+++ b/src/debug-helper.js
@@ -0,0 +1,6 @@
+function processData(items) {
+    console.log("processing items");
+    return items.map(x => x * 2);
+}
+
+console.log("module loaded");
"""


@pytest.fixture
def import_order_violation_diff() -> str:
    """A diff with imports in wrong order (internal before external)."""
    return """diff --git a/src/handler.py b/src/handler.py
new file mode 100644
--- /dev/null
+++ b/src/handler.py
@@ -0,0 +1,6 @@
+from src.utils.helpers import validate
+import flask
+import os
+
+def handle():
+    pass
"""


@pytest.fixture
def semantic_change_diff() -> str:
    """A diff modifying a function's error handling behavior."""
    return """diff --git a/src/payments/processor.py b/src/payments/processor.py
--- a/src/payments/processor.py
+++ b/src/payments/processor.py
@@ -1,8 +1,7 @@
-def process_payment(amount, currency):
-    if amount <= 0:
-        raise PaymentError("Invalid amount")
-    return {"status": "ok", "amount": amount}
+def process_payment(amount, currency):
+    if amount <= 0:
+        return None
+    return {"status": "ok", "amount": amount}
"""
