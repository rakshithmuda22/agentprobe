"""Tests for Tree-sitter engine, diff parser, and import graph."""

from __future__ import annotations

import pytest

from src.parsers.tree_sitter_engine import TreeSitterEngine
from src.parsers.diff_parser import parse_diff
from src.parsers.import_graph import ImportGraph, Boundaries


# ── Tree-sitter Engine ──────────────────────────────────────────


class TestTreeSitterEngine:
    def setup_method(self):
        self.engine = TreeSitterEngine()

    def test_detect_language(self):
        assert self.engine.detect_language("foo.py") == "python"
        assert self.engine.detect_language("bar.ts") == "typescript"
        assert self.engine.detect_language("baz.js") == "javascript"
        assert self.engine.detect_language("qux.go") == "go"
        assert self.engine.detect_language("readme.md") is None

    def test_extract_python_functions(self):
        source = """
def foo(x, y):
    return x + y

def bar(name: str) -> bool:
    return len(name) > 0

def baz():
    pass
"""
        funcs = self.engine.extract_functions(source, "python")
        names = [f.name for f in funcs]
        assert "foo" in names
        assert "bar" in names
        assert "baz" in names
        assert len(funcs) == 3

        foo = next(f for f in funcs if f.name == "foo")
        assert "x" in foo.params
        assert "y" in foo.params

    def test_extract_python_classes(self):
        source = """
class MyClass:
    def method_a(self):
        pass

    def method_b(self, x):
        return x

class AnotherClass:
    pass
"""
        classes = self.engine.extract_classes(source, "python")
        names = [c.name for c in classes]
        assert "MyClass" in names
        assert "AnotherClass" in names
        assert len(classes) == 2

        my_class = next(c for c in classes if c.name == "MyClass")
        assert "method_a" in my_class.methods
        assert "method_b" in my_class.methods

    def test_extract_python_imports(self):
        source = """
import os
import sys
from pathlib import Path
from src.utils.helpers import validate, transform
from . import local_module
"""
        imports = self.engine.extract_imports(source, "python")
        module_paths = [i.module_path for i in imports]
        assert "os" in module_paths
        assert "sys" in module_paths
        assert "pathlib" in module_paths
        assert len(imports) >= 4

    def test_extract_javascript_imports(self):
        source = """
import React from 'react';
import { useState, useEffect } from 'react';
import { Button } from './components/Button';
"""
        imports = self.engine.extract_imports(source, "javascript")
        module_paths = [i.module_path for i in imports]
        assert "react" in module_paths
        assert "./components/Button" in module_paths

    def test_extract_typescript_functions(self):
        source = """
function greet(name: string): string {
    return "Hello " + name;
}

class Service {
    process(data: any): void {
        console.log(data);
    }
}
"""
        funcs = self.engine.extract_functions(source, "typescript")
        names = [f.name for f in funcs]
        assert "greet" in names


# ── Diff Parser ─────────────────────────────────────────────────


class TestDiffParser:
    def test_parse_single_file_diff(self):
        diff = """diff --git a/src/app.py b/src/app.py
--- a/src/app.py
+++ b/src/app.py
@@ -1,3 +1,4 @@
 import os
+import sys

 def main():
"""
        result = parse_diff(diff)
        assert len(result.files) == 1
        assert result.files[0].file_path == "src/app.py"
        assert result.files[0].status == "modified"
        assert len(result.files[0].added_lines) == 1
        assert result.files[0].added_lines[0][1] == "import sys"

    def test_parse_multi_file_diff(self):
        diff = """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -1,2 +1,3 @@
 x = 1
+y = 2
 z = 3
diff --git a/src/b.py b/src/b.py
new file mode 100644
--- /dev/null
+++ b/src/b.py
@@ -0,0 +1,2 @@
+def hello():
+    pass
diff --git a/src/c.py b/src/c.py
--- a/src/c.py
+++ b/src/c.py
@@ -1,3 +1,2 @@
 a = 1
-b = 2
 c = 3
"""
        result = parse_diff(diff)
        assert len(result.files) == 3
        assert result.modified_files == ["src/a.py", "src/b.py", "src/c.py"]

        # b.py is a new file
        b_file = result.files[1]
        assert b_file.status == "added"
        assert len(b_file.added_lines) == 2

        # c.py has a deletion
        c_file = result.files[2]
        assert len(c_file.deleted_lines) == 1

    def test_parse_empty_diff(self):
        result = parse_diff("")
        assert len(result.files) == 0


# ── Import Graph ────────────────────────────────────────────────


class TestImportGraph:
    def test_add_and_check_edges(self):
        graph = ImportGraph()
        graph.add_edge("payments", "analytics")
        graph.add_edge("payments", "shared")
        graph.add_edge("analytics", "shared")

        assert "analytics" in graph.edges["payments"]
        assert "shared" in graph.edges["payments"]
        assert "shared" in graph.edges["analytics"]

    def test_detect_circular(self):
        graph = ImportGraph()
        graph.add_edge("a", "b")
        graph.add_edge("b", "a")
        graph.add_edge("c", "d")

        cycles = graph.detect_circular()
        assert len(cycles) == 1
        assert tuple(sorted(cycles[0])) == ("a", "b")

    def test_check_boundary_forbidden(self):
        boundaries = Boundaries(
            modules={
                "payments": {
                    "allowed_imports": ["shared", "types"],
                    "forbidden_imports": ["analytics", "marketing"],
                }
            },
            layers=[],
        )
        graph = ImportGraph()
        violations = graph.check_boundary("src.payments.processor", "src.analytics.tracker", boundaries)
        assert len(violations) >= 1
        assert "forbidden" in violations[0].lower()

    def test_check_boundary_allowed(self):
        boundaries = Boundaries(
            modules={
                "payments": {
                    "allowed_imports": ["shared", "types"],
                    "forbidden_imports": ["analytics"],
                }
            },
            layers=[],
        )
        graph = ImportGraph()
        violations = graph.check_boundary("src.payments.processor", "src.shared.utils", boundaries)
        assert len(violations) == 0

    def test_check_layer_violation(self):
        boundaries = Boundaries(
            modules={},
            layers=[
                {"name": "domain", "can_import": []},
                {"name": "application", "can_import": ["domain", "infrastructure"]},
                {"name": "presentation", "can_import": ["application", "shared"]},
            ],
        )
        graph = ImportGraph()
        # domain cannot import anything
        violations = graph.check_boundary("src.domain.model", "src.application.service", boundaries)
        assert len(violations) >= 1
        assert "layer" in violations[0].lower()

    def test_check_import_depth(self):
        graph = ImportGraph()
        assert graph.check_import_depth("a.b.c.d", max_depth=3) is True
        assert graph.check_import_depth("a.b.c", max_depth=3) is False
        assert graph.check_import_depth("a.b", max_depth=3) is False
