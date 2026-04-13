"""GitHub unified diff parser."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class FileChange:
    """Changes to a single file in a diff."""
    file_path: str
    status: str  # "added", "modified", "deleted"
    added_lines: list[tuple[int, str]] = field(default_factory=list)   # (line_number, content)
    deleted_lines: list[tuple[int, str]] = field(default_factory=list)  # (line_number, content)


@dataclass
class DiffResult:
    """Parsed result of a unified diff."""
    files: list[FileChange] = field(default_factory=list)

    @property
    def modified_files(self) -> list[str]:
        """Return list of all file paths in the diff."""
        return [f.file_path for f in self.files]

    @property
    def added_files(self) -> list[str]:
        return [f.file_path for f in self.files if f.status == "added"]


DIFF_HEADER_RE = re.compile(r"^diff --git a/(.*) b/(.*)$")
HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
NEW_FILE_RE = re.compile(r"^new file mode")
DELETED_FILE_RE = re.compile(r"^deleted file mode")


def parse_diff(diff_text: str) -> DiffResult:
    """Parse a GitHub unified diff string into structured FileChange objects."""
    result = DiffResult()
    current_file: FileChange | None = None
    new_line_num = 0
    old_line_num = 0
    is_new_file = False
    is_deleted_file = False

    for line in diff_text.split("\n"):
        # New file header
        header_match = DIFF_HEADER_RE.match(line)
        if header_match:
            # Save previous file
            if current_file is not None:
                result.files.append(current_file)

            file_path = header_match.group(2)
            is_new_file = False
            is_deleted_file = False
            current_file = FileChange(file_path=file_path, status="modified")
            continue

        if current_file is None:
            continue

        # Check for new/deleted file markers
        if NEW_FILE_RE.match(line):
            is_new_file = True
            current_file.status = "added"
            continue
        if DELETED_FILE_RE.match(line):
            is_deleted_file = True
            current_file.status = "deleted"
            continue

        # Hunk header
        hunk_match = HUNK_HEADER_RE.match(line)
        if hunk_match:
            old_line_num = int(hunk_match.group(1))
            new_line_num = int(hunk_match.group(2))
            continue

        # Diff content lines
        if line.startswith("+") and not line.startswith("+++"):
            current_file.added_lines.append((new_line_num, line[1:]))
            new_line_num += 1
        elif line.startswith("-") and not line.startswith("---"):
            current_file.deleted_lines.append((old_line_num, line[1:]))
            old_line_num += 1
        elif line.startswith(" "):
            old_line_num += 1
            new_line_num += 1

    # Don't forget the last file
    if current_file is not None:
        result.files.append(current_file)

    return result
