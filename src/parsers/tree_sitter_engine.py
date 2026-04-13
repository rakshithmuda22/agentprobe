"""Tree-sitter AST parsing engine for multi-language code analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter_languages


@dataclass
class FunctionDef:
    """Extracted function definition."""
    name: str
    line_start: int
    line_end: int
    params: list[str] = field(default_factory=list)
    return_type: str | None = None


@dataclass
class ClassDef:
    """Extracted class definition."""
    name: str
    line_start: int
    line_end: int
    methods: list[str] = field(default_factory=list)


@dataclass
class ImportStmt:
    """Extracted import statement."""
    module_path: str
    imported_names: list[str] = field(default_factory=list)
    line_number: int = 0


EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
}


class TreeSitterEngine:
    """Parse source files using Tree-sitter and extract structural information."""

    def __init__(self) -> None:
        self._parsers: dict[str, tree_sitter_languages.Language] = {}

    def detect_language(self, file_path: str) -> str | None:
        """Detect programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        return EXTENSION_TO_LANGUAGE.get(ext)

    def _get_parser(self, language: str):
        """Get or create a Tree-sitter parser for the given language."""
        parser = tree_sitter_languages.get_parser(language)
        return parser

    def parse_source(self, source: str, language: str):
        """Parse source code string and return the AST tree."""
        parser = self._get_parser(language)
        return parser.parse(source.encode("utf-8"))

    def parse_file(self, file_path: str):
        """Parse a file and return the AST tree."""
        language = self.detect_language(file_path)
        if language is None:
            raise ValueError(f"Unsupported file type: {file_path}")
        with open(file_path) as f:
            source = f.read()
        return self.parse_source(source, language), language, source

    def extract_functions(self, source: str, language: str) -> list[FunctionDef]:
        """Extract function definitions from source code."""
        tree = self.parse_source(source, language)
        functions = []
        self._walk_for_functions(tree.root_node, source, language, functions)
        return functions

    def _walk_for_functions(self, node, source: str, language: str, results: list[FunctionDef]) -> None:
        """Recursively walk AST to find function definitions."""
        func_types = {
            "python": ["function_definition"],
            "typescript": ["function_declaration", "method_definition", "arrow_function"],
            "tsx": ["function_declaration", "method_definition", "arrow_function"],
            "javascript": ["function_declaration", "method_definition", "arrow_function"],
            "go": ["function_declaration", "method_declaration"],
        }
        target_types = func_types.get(language, [])

        if node.type in target_types:
            name = self._get_function_name(node, language)
            if name:
                params = self._get_function_params(node, language)
                results.append(FunctionDef(
                    name=name,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    params=params,
                ))

        for child in node.children:
            self._walk_for_functions(child, source, language, results)

    def _get_function_name(self, node, language: str) -> str | None:
        """Extract the function name from an AST node."""
        for child in node.children:
            if child.type == "name" or child.type == "identifier":
                return child.text.decode("utf-8")
            if child.type == "property_identifier":
                return child.text.decode("utf-8")
        # For arrow functions assigned to variables, look at the parent
        if node.type == "arrow_function" and node.parent:
            if node.parent.type == "variable_declarator":
                for child in node.parent.children:
                    if child.type in ("name", "identifier"):
                        return child.text.decode("utf-8")
        return None

    def _get_function_params(self, node, language: str) -> list[str]:
        """Extract parameter names from a function node."""
        params = []
        for child in node.children:
            if child.type in ("parameters", "formal_parameters", "parameter_list"):
                for param in child.children:
                    if param.type in ("identifier", "name"):
                        params.append(param.text.decode("utf-8"))
                    # Handle typed parameters (e.g., Python's `name: str`)
                    elif param.type in ("typed_parameter", "required_parameter",
                                         "typed_default_parameter", "default_parameter"):
                        for sub in param.children:
                            if sub.type in ("identifier", "name"):
                                params.append(sub.text.decode("utf-8"))
                                break
        return params

    def extract_classes(self, source: str, language: str) -> list[ClassDef]:
        """Extract class definitions from source code."""
        tree = self.parse_source(source, language)
        classes = []
        self._walk_for_classes(tree.root_node, source, language, classes)
        return classes

    def _walk_for_classes(self, node, source: str, language: str, results: list[ClassDef]) -> None:
        """Recursively walk AST to find class definitions."""
        class_types = {
            "python": ["class_definition"],
            "typescript": ["class_declaration"],
            "tsx": ["class_declaration"],
            "javascript": ["class_declaration"],
            "go": ["type_declaration"],
        }
        target_types = class_types.get(language, [])

        if node.type in target_types:
            name = self._get_class_name(node, language)
            if name:
                methods = self._get_class_methods(node, language)
                results.append(ClassDef(
                    name=name,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    methods=methods,
                ))

        for child in node.children:
            self._walk_for_classes(child, source, language, results)

    def _get_class_name(self, node, language: str) -> str | None:
        """Extract class name from an AST node."""
        for child in node.children:
            if child.type in ("name", "identifier", "type_identifier"):
                return child.text.decode("utf-8")
            if child.type == "type_spec":
                for sub in child.children:
                    if sub.type in ("type_identifier", "identifier"):
                        return sub.text.decode("utf-8")
        return None

    def _get_class_methods(self, node, language: str) -> list[str]:
        """Extract method names from a class node."""
        methods = []
        func_results: list[FunctionDef] = []
        for child in node.children:
            if child.type in ("block", "class_body"):
                self._walk_for_functions(child, "", language, func_results)
        return [f.name for f in func_results]

    def extract_imports(self, source: str, language: str) -> list[ImportStmt]:
        """Extract import statements from source code."""
        tree = self.parse_source(source, language)
        imports = []
        self._walk_for_imports(tree.root_node, language, imports)
        return imports

    def _walk_for_imports(self, node, language: str, results: list[ImportStmt]) -> None:
        """Recursively walk AST to find import statements."""
        if language == "python":
            if node.type == "import_statement":
                self._extract_python_import(node, results)
            elif node.type == "import_from_statement":
                self._extract_python_from_import(node, results)
        elif language in ("typescript", "tsx", "javascript"):
            if node.type == "import_statement":
                self._extract_js_import(node, results)
        elif language == "go":
            if node.type == "import_declaration":
                self._extract_go_import(node, results)

        for child in node.children:
            self._walk_for_imports(child, language, results)

    def _extract_python_import(self, node, results: list[ImportStmt]) -> None:
        """Extract `import x` style Python imports."""
        for child in node.children:
            if child.type == "dotted_name":
                results.append(ImportStmt(
                    module_path=child.text.decode("utf-8"),
                    imported_names=[],
                    line_number=node.start_point[0] + 1,
                ))

    def _extract_python_from_import(self, node, results: list[ImportStmt]) -> None:
        """Extract `from x import y` style Python imports."""
        module_path = ""
        names = []
        # The structure is: from <module> import <names>
        # Find the module path: first dotted_name or relative_import before 'import' keyword
        found_import_keyword = False
        for child in node.children:
            if child.type == "import":
                found_import_keyword = True
                continue
            if not found_import_keyword:
                # Before 'import' keyword — this is the module path
                if child.type in ("dotted_name", "relative_import"):
                    module_path = child.text.decode("utf-8")
            else:
                # After 'import' keyword — these are the imported names
                if child.type in ("dotted_name", "identifier"):
                    names.append(child.text.decode("utf-8"))
                elif child.type in ("import_list", "import_from_as_names"):
                    for sub in child.children:
                        if sub.type in ("identifier", "dotted_name"):
                            names.append(sub.text.decode("utf-8"))
                        elif sub.type == "aliased_import":
                            for n in sub.children:
                                if n.type in ("identifier", "dotted_name"):
                                    names.append(n.text.decode("utf-8"))
                                    break

        if module_path:
            results.append(ImportStmt(
                module_path=module_path,
                imported_names=names,
                line_number=node.start_point[0] + 1,
            ))

    def _extract_js_import(self, node, results: list[ImportStmt]) -> None:
        """Extract JavaScript/TypeScript import statements."""
        module_path = ""
        names = []
        for child in node.children:
            if child.type == "string":
                module_path = child.text.decode("utf-8").strip("'\"")
            elif child.type == "import_clause":
                for sub in child.children:
                    if sub.type == "identifier":
                        names.append(sub.text.decode("utf-8"))
                    elif sub.type == "named_imports":
                        for spec in sub.children:
                            if spec.type == "import_specifier":
                                for n in spec.children:
                                    if n.type == "identifier":
                                        names.append(n.text.decode("utf-8"))
                                        break

        if module_path:
            results.append(ImportStmt(
                module_path=module_path,
                imported_names=names,
                line_number=node.start_point[0] + 1,
            ))

    def _extract_go_import(self, node, results: list[ImportStmt]) -> None:
        """Extract Go import declarations."""
        for child in node.children:
            if child.type == "import_spec":
                for sub in child.children:
                    if sub.type == "interpreted_string_literal":
                        path = sub.text.decode("utf-8").strip('"')
                        results.append(ImportStmt(
                            module_path=path,
                            imported_names=[],
                            line_number=child.start_point[0] + 1,
                        ))
            elif child.type == "import_spec_list":
                for spec in child.children:
                    if spec.type == "import_spec":
                        for sub in spec.children:
                            if sub.type == "interpreted_string_literal":
                                path = sub.text.decode("utf-8").strip('"')
                                results.append(ImportStmt(
                                    module_path=path,
                                    imported_names=[],
                                    line_number=spec.start_point[0] + 1,
                                ))
