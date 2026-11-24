"""Deterministic incremental codebase indexer, AST skeletonizer, and dependency/git analyzer."""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass, field
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import subprocess
import tempfile
from typing import Any, Mapping

from .storage import ContextStore
from .tokens import estimate_tokens

_CACHE_SCHEMA_VERSION = 1
_MAX_FILE_BYTES = 512 * 1024  # 512 KB hard limit
_MAX_COCHANGE_COMMIT_FILES = 20

_EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".ctx",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        "dist",
        "build",
        ".next",
        "target",
    }
)

_SECRET_EXACT_NAMES: frozenset[str] = frozenset(
    {
        ".env",
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
    }
)

_SECRET_GLOB_PATTERNS: tuple[str, ...] = (
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa.*",
)

_BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".ico",
        ".bmp",
        ".svgz",
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
        ".eot",
        ".mp3",
        ".mp4",
        ".wav",
        ".ogg",
        ".webm",
        ".mov",
        ".avi",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".a",
        ".o",
        ".obj",
        ".pyc",
        ".pyo",
        ".class",
        ".jar",
        ".war",
        ".wasm",
        ".bin",
        ".dat",
        ".db",
        ".sqlite",
        ".sqlite3",
        ".parquet",
    }
)

_EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".md": "markdown",
    ".mdx": "markdown",
    ".rst": "rst",
    ".json": "json",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".css": "css",
    ".scss": "css",
    ".html": "html",
    ".htm": "html",
    ".txt": "text",
}

VALID_SYMBOL_KINDS: frozenset[str] = frozenset(
    {"class", "function", "method", "interface", "type", "heading"}
)


@dataclass(slots=True)
class SymbolInfo:
    """Extracted structural symbol within a source or documentation file."""

    name: str
    kind: str
    signature: str
    docstring_summary: str = ""
    start_line: int = 1
    end_line: int = 1
    parent: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in VALID_SYMBOL_KINDS:
            raise ValueError(
                f"Invalid SymbolInfo.kind '{self.kind}'; expected one of {sorted(VALID_SYMBOL_KINDS)}."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "signature": self.signature,
            "docstring_summary": self.docstring_summary,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "parent": self.parent,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SymbolInfo:
        return cls(
            name=str(data["name"]),
            kind=str(data["kind"]),
            signature=str(data["signature"]),
            docstring_summary=str(data.get("docstring_summary", "")),
            start_line=int(data.get("start_line", 1)),
            end_line=int(data.get("end_line", 1)),
            parent=data.get("parent"),
        )


@dataclass(slots=True)
class FileRecord:
    """Indexed metadata, symbols, imports, and Tier-2 compressed skeleton for a file."""

    rel_path: str
    language: str
    sha256: str
    mtime_ns: int
    token_count: int
    imports: list[str]
    symbols: list[SymbolInfo]
    skeleton: str
    skeleton_tokens: int
    size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rel_path": self.rel_path,
            "language": self.language,
            "sha256": self.sha256,
            "mtime_ns": self.mtime_ns,
            "size": self.size,
            "token_count": self.token_count,
            "imports": list(self.imports),
            "symbols": [symbol.to_dict() for symbol in self.symbols],
            "skeleton": self.skeleton,
            "skeleton_tokens": self.skeleton_tokens,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FileRecord:
        return cls(
            rel_path=str(data["rel_path"]),
            language=str(data["language"]),
            sha256=str(data["sha256"]),
            mtime_ns=int(data["mtime_ns"]),
            token_count=int(data["token_count"]),
            imports=[str(item) for item in data.get("imports", [])],
            symbols=[SymbolInfo.from_dict(s) for s in data.get("symbols", [])],
            skeleton=str(data["skeleton"]),
            skeleton_tokens=int(data["skeleton_tokens"]),
            size=int(data.get("size", 0)),
        )


@dataclass(slots=True)
class RepoIndex:
    """Complete repository structural index and dependency/co-change graphs."""

    root: str
    files: dict[str, FileRecord] = field(default_factory=dict)
    import_graph: dict[str, list[str]] = field(default_factory=dict)
    reverse_graph: dict[str, list[str]] = field(default_factory=dict)
    cochange_graph: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": _CACHE_SCHEMA_VERSION,
            "root": self.root,
            "files": {
                rel_path: record.to_dict()
                for rel_path, record in sorted(self.files.items())
            },
            "import_graph": {
                k: list(v) for k, v in sorted(self.import_graph.items())
            },
            "reverse_graph": {
                k: list(v) for k, v in sorted(self.reverse_graph.items())
            },
            "cochange_graph": {
                k: dict(sorted(v.items()))
                for k, v in sorted(self.cochange_graph.items())
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RepoIndex:
        raw_files = data.get("files", {})
        files: dict[str, FileRecord] = {}
        if isinstance(raw_files, Mapping):
            for rel_path, record_data in raw_files.items():
                files[str(rel_path)] = FileRecord.from_dict(record_data)

        raw_import = data.get("import_graph", {})
        import_graph = {
            str(k): [str(x) for x in v]
            for k, v in raw_import.items()
        } if isinstance(raw_import, Mapping) else {}

        raw_reverse = data.get("reverse_graph", {})
        reverse_graph = {
            str(k): [str(x) for x in v]
            for k, v in raw_reverse.items()
        } if isinstance(raw_reverse, Mapping) else {}

        raw_cochange = data.get("cochange_graph", {})
        cochange_graph = {
            str(k): {str(peer): int(weight) for peer, weight in peers.items()}
            for k, peers in raw_cochange.items()
        } if isinstance(raw_cochange, Mapping) else {}

        return cls(
            root=str(data["root"]),
            files=files,
            import_graph=import_graph,
            reverse_graph=reverse_graph,
            cochange_graph=cochange_graph,
        )


# ----------------------------------------------------------------------
# .gitignore / .ctxignore Parser
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _IgnoreRule:
    base_dir: str  # "" for root, or POSIX relative dir like "src/sub"
    regex: re.Pattern[str]
    negated: bool
    dir_only: bool


def _glob_segment_to_regex(pattern: str) -> str:
    """Translate a gitignore glob pattern into a regex body."""
    out: list[str] = []
    i = 0
    n = len(pattern)
    while i < n:
        ch = pattern[i]
        if ch == "*":
            if i + 1 < n and pattern[i + 1] == "*":
                # Handle ** globstar.
                prev_is_slash = i == 0 or pattern[i - 1] == "/"
                next_is_slash = i + 2 >= n or pattern[i + 2] == "/"
                if prev_is_slash and next_is_slash:
                    if i + 2 < n and pattern[i + 2] == "/":
                        out.append("(?:.*/)?")
                        i += 3
                        continue
                    out.append(".*")
                    i += 2
                    continue
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
            i += 1
        elif ch == "?":
            out.append("[^/]")
            i += 1
        elif ch == "[":
            j = i + 1
            if j < n and pattern[j] in ("!", "^"):
                j += 1
            if j < n and pattern[j] == "]":
                j += 1
            while j < n and pattern[j] != "]":
                j += 1
            if j >= n:
                out.append(re.escape(ch))
                i += 1
            else:
                inner = pattern[i + 1 : j]
                if inner.startswith("!"):
                    inner = "^" + inner[1:]
                out.append(f"[{inner}]")
                i = j + 1
        elif ch == "\\":
            if i + 1 < n:
                out.append(re.escape(pattern[i + 1]))
                i += 2
            else:
                out.append(re.escape(ch))
                i += 1
        else:
            out.append(re.escape(ch))
            i += 1
    return "".join(out)


def _compile_ignore_line(raw_line: str, base_dir: str) -> _IgnoreRule | None:
    line = raw_line.rstrip("\r\n")
    if not line:
        return None

    # Strip trailing unescaped spaces.
    while line.endswith(" ") and not line.endswith("\\ "):
        line = line[:-1]
    if not line or line.startswith("#"):
        return None

    negated = False
    if line.startswith("!"):
        negated = True
        line = line[1:]
    elif line.startswith("\\#") or line.startswith("\\!"):
        line = line[1:]

    if not line:
        return None

    dir_only = line.endswith("/")
    if dir_only:
        line = line.rstrip("/")
        if not line:
            return None

    anchored = "/" in line
    if line.startswith("/"):
        line = line[1:]
    if not line:
        return None

    regex_body = _glob_segment_to_regex(line)
    if anchored:
        pattern_str = f"^{regex_body}(?:/.*)?$" if dir_only else f"^{regex_body}$"
    else:
        pattern_str = (
            f"(?:^|/){regex_body}(?:/.*)?$"
            if dir_only
            else f"(?:^|/){regex_body}$"
        )

    return _IgnoreRule(
        base_dir=base_dir,
        regex=re.compile(pattern_str),
        negated=negated,
        dir_only=dir_only,
    )


class _IgnoreMatcher:
    """Evaluates stacked `.gitignore` and `.ctxignore` rules."""

    def __init__(self) -> None:
        self._rules: list[_IgnoreRule] = []

    def add_ignore_file(self, path: Path, base_dir: str = "") -> None:
        if not path.is_file():
            return
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        for raw_line in text.splitlines():
            rule = _compile_ignore_line(raw_line, base_dir=base_dir)
            if rule is not None:
                self._rules.append(rule)

    def copy_with_file(self, path: Path, base_dir: str) -> _IgnoreMatcher:
        clone = _IgnoreMatcher()
        clone._rules = list(self._rules)
        clone.add_ignore_file(path, base_dir=base_dir)
        return clone

    def is_ignored(self, rel_path: str, *, is_dir: bool) -> bool:
        ignored = False
        for rule in self._rules:
            if rule.base_dir:
                prefix = f"{rule.base_dir}/"
                if not rel_path.startswith(prefix):
                    continue
                target = rel_path[len(prefix) :]
            else:
                target = rel_path

            if not target:
                continue

            if rule.dir_only and not is_dir:
                # Directory-only rules apply to files only if a parent segment matched.
                parts = target.split("/")[:-1]
                matched = any(
                    rule.regex.search("/".join(parts[: idx + 1])) is not None
                    for idx in range(len(parts))
                )
            else:
                matched = rule.regex.search(target) is not None

            if matched:
                ignored = not rule.negated
        return ignored


def _is_secret_or_binary_name(filename: str) -> bool:
    if filename in _SECRET_EXACT_NAMES:
        return True
    lower = filename.lower()
    for pat in _SECRET_GLOB_PATTERNS:
        if fnmatch.fnmatch(lower, pat):
            return True
    suffix = Path(lower).suffix
    return suffix in _BINARY_EXTENSIONS


def _looks_binary(raw: bytes) -> bool:
    sample = raw[:8192]
    if b"\x00" in sample:
        return True
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def _detect_language(rel_path: str) -> str:
    suffix = Path(rel_path).suffix.lower()
    if suffix in _EXTENSION_TO_LANGUAGE:
        return _EXTENSION_TO_LANGUAGE[suffix]
    name = Path(rel_path).name.lower()
    if name in ("makefile", "gnumakefile"):
        return "makefile"
    if name == "dockerfile":
        return "dockerfile"
    return "text"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=False,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# ----------------------------------------------------------------------
# Language Parsers & Tier-2 Skeletonizers
# ----------------------------------------------------------------------


def _first_doc_line(doc: str | None, max_chars: int = 140) -> str:
    if not doc:
        return ""
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            if len(stripped) > max_chars:
                return stripped[: max_chars - 3].rstrip() + "..."
            return stripped
    return ""


def _finalize_skeleton(rel_path: str, token_count: int, body_lines: list[str]) -> tuple[str, int]:
    """Attach header with converged skeleton token count."""
    body = "\n".join(body_lines).rstrip()
    header_tpl = f"# File: {rel_path} (full: {token_count} tok -> skeleton: {{sk}} tok)"
    if body:
        draft = f"{header_tpl.format(sk=0)}\n{body}"
    else:
        draft = header_tpl.format(sk=0)

    sk_tokens = estimate_tokens(draft)
    for _ in range(2):
        rendered = f"{header_tpl.format(sk=sk_tokens)}\n{body}" if body else header_tpl.format(sk=sk_tokens)
        recalculated = estimate_tokens(rendered)
        if recalculated == sk_tokens:
            return rendered, sk_tokens
        sk_tokens = recalculated

    rendered = f"{header_tpl.format(sk=sk_tokens)}\n{body}" if body else header_tpl.format(sk=sk_tokens)
    return rendered, estimate_tokens(rendered)


def _format_py_arg(arg: ast.arg, default: ast.expr | None) -> str:
    text = arg.arg
    if arg.annotation is not None:
        text += f": {ast.unparse(arg.annotation)}"
    if default is not None:
        text += f" = {ast.unparse(default)}"
    return text


def _format_py_function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    args = node.args
    params: list[str] = []

    all_pos = list(args.posonlyargs) + list(args.args)
    num_no_default = len(all_pos) - len(args.defaults)
    posonly_count = len(args.posonlyargs)

    for idx, arg in enumerate(all_pos):
        default = args.defaults[idx - num_no_default] if idx >= num_no_default else None
        params.append(_format_py_arg(arg, default))
        if posonly_count > 0 and idx == posonly_count - 1:
            params.append("/")

    if args.vararg is not None:
        params.append("*" + _format_py_arg(args.vararg, None))
    elif args.kwonlyargs:
        params.append("*")

    for kw_arg, kw_default in zip(args.kwonlyargs, args.kw_defaults):
        params.append(_format_py_arg(kw_arg, kw_default))

    if args.kwarg is not None:
        params.append("**" + _format_py_arg(args.kwarg, None))

    ret = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix}{node.name}({', '.join(params)}){ret}"


def _format_py_class_signature(node: ast.ClassDef) -> str:
    bases = [ast.unparse(base) for base in node.bases]
    keywords = [
        f"{kw.arg}={ast.unparse(kw.value)}" if kw.arg else f"**{ast.unparse(kw.value)}"
        for kw in node.keywords
    ]
    all_parents = bases + keywords
    if all_parents:
        return f"class {node.name}({', '.join(all_parents)})"
    return f"class {node.name}"


def _collect_py_imports_from_stmt(stmt: ast.stmt, out: list[str]) -> None:
    if isinstance(stmt, ast.Import):
        names = ", ".join(alias.name for alias in stmt.names)
        if names:
            out.append(f"import {names}")
    elif isinstance(stmt, ast.ImportFrom):
        dots = "." * stmt.level
        module = stmt.module or ""
        names = ", ".join(alias.name for alias in stmt.names)
        out.append(f"from {dots}{module} import {names}")
    elif isinstance(stmt, ast.If):
        for child in stmt.body:
            _collect_py_imports_from_stmt(child, out)
        for child in stmt.orelse:
            _collect_py_imports_from_stmt(child, out)
    elif isinstance(stmt, ast.Try):
        for child in stmt.body:
            _collect_py_imports_from_stmt(child, out)
        for handler in stmt.handlers:
            for child in handler.body:
                _collect_py_imports_from_stmt(child, out)
        for child in stmt.orelse:
            _collect_py_imports_from_stmt(child, out)
        for child in stmt.finalbody:
            _collect_py_imports_from_stmt(child, out)


def _parse_python_file(
    rel_path: str,
    source: str,
    token_count: int,
) -> tuple[list[str], list[SymbolInfo], str, int]:
    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError:
        return _parse_python_fallback(rel_path, source, token_count)

    imports: list[str] = []
    symbols: list[SymbolInfo] = []
    skeleton_lines: list[str] = []

    module_doc = _first_doc_line(ast.get_docstring(tree, clean=True))
    if module_doc:
        skeleton_lines.append(f'"""{module_doc}"""')

    for stmt in tree.body:
        _collect_py_imports_from_stmt(stmt, imports)

    # Deduplicate imports preserving order.
    dedup_imports = list(dict.fromkeys(imports))
    skeleton_lines.extend(dedup_imports)

    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig = _format_py_function_signature(stmt)
            doc = _first_doc_line(ast.get_docstring(stmt, clean=True))
            symbols.append(
                SymbolInfo(
                    name=stmt.name,
                    kind="function",
                    signature=sig,
                    docstring_summary=doc,
                    start_line=stmt.lineno,
                    end_line=stmt.end_lineno or stmt.lineno,
                    parent=None,
                )
            )
            for dec in stmt.decorator_list:
                skeleton_lines.append(f"@{ast.unparse(dec)}")
            if doc:
                skeleton_lines.append(f"{sig}:")
                skeleton_lines.append(f'    """{doc}"""')
                skeleton_lines.append("    ...")
            else:
                skeleton_lines.append(f"{sig}: ...")

        elif isinstance(stmt, ast.ClassDef):
            cls_sig = _format_py_class_signature(stmt)
            cls_doc = _first_doc_line(ast.get_docstring(stmt, clean=True))
            symbols.append(
                SymbolInfo(
                    name=stmt.name,
                    kind="class",
                    signature=cls_sig,
                    docstring_summary=cls_doc,
                    start_line=stmt.lineno,
                    end_line=stmt.end_lineno or stmt.lineno,
                    parent=None,
                )
            )
            for dec in stmt.decorator_list:
                skeleton_lines.append(f"@{ast.unparse(dec)}")
            skeleton_lines.append(f"{cls_sig}:")
            if cls_doc:
                skeleton_lines.append(f'    """{cls_doc}"""')

            member_added = False
            for member in stmt.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    member_added = True
                    m_sig = _format_py_function_signature(member)
                    m_doc = _first_doc_line(ast.get_docstring(member, clean=True))
                    symbols.append(
                        SymbolInfo(
                            name=member.name,
                            kind="method",
                            signature=m_sig,
                            docstring_summary=m_doc,
                            start_line=member.lineno,
                            end_line=member.end_lineno or member.lineno,
                            parent=stmt.name,
                        )
                    )
                    for dec in member.decorator_list:
                        skeleton_lines.append(f"    @{ast.unparse(dec)}")
                    if m_doc:
                        skeleton_lines.append(f"    {m_sig}:")
                        skeleton_lines.append(f'        """{m_doc}"""')
                        skeleton_lines.append("        ...")
                    else:
                        skeleton_lines.append(f"    {m_sig}: ...")
                elif isinstance(member, ast.ClassDef):
                    member_added = True
                    sub_sig = _format_py_class_signature(member)
                    sub_doc = _first_doc_line(ast.get_docstring(member, clean=True))
                    symbols.append(
                        SymbolInfo(
                            name=member.name,
                            kind="class",
                            signature=sub_sig,
                            docstring_summary=sub_doc,
                            start_line=member.lineno,
                            end_line=member.end_lineno or member.lineno,
                            parent=stmt.name,
                        )
                    )
                    skeleton_lines.append(f"    {sub_sig}: ...")

            if not member_added and not cls_doc:
                skeleton_lines.append("    ...")

    skeleton, sk_tokens = _finalize_skeleton(rel_path, token_count, skeleton_lines)
    return dedup_imports, symbols, skeleton, sk_tokens


def _parse_python_fallback(
    rel_path: str,
    source: str,
    token_count: int,
) -> tuple[list[str], list[SymbolInfo], str, int]:
    """Line-based fallback when a Python file contains a temporary SyntaxError."""
    imports: list[str] = []
    symbols: list[SymbolInfo] = []
    skeleton_lines: list[str] = []

    for lineno, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            imports.append(stripped)
            skeleton_lines.append(stripped)
        elif stripped.startswith(("class ", "def ", "async def ")):
            indent = line[: len(line) - len(line.lstrip())]
            sig = stripped.rstrip(":").strip()
            kind = "class" if stripped.startswith("class ") else ("method" if indent else "function")
            name_match = re.search(r"(?:class|def)\s+([A-Za-z_][A-Za-z0-9_]*)", sig)
            name = name_match.group(1) if name_match else sig
            symbols.append(
                SymbolInfo(
                    name=name,
                    kind=kind,
                    signature=sig,
                    start_line=lineno,
                    end_line=lineno,
                )
            )
            skeleton_lines.append(f"{indent}{sig}: ...")

    skeleton, sk_tokens = _finalize_skeleton(rel_path, token_count, skeleton_lines)
    return list(dict.fromkeys(imports)), symbols, skeleton, sk_tokens


# ----------------------------------------------------------------------
# C-family Structural Scanner (TS/TSX/JS/JSX/Go/Rust/Java)
# ----------------------------------------------------------------------


def _blank_comments_and_strings(source: str, language: str) -> str:
    """Return same-length text with comments and string literals replaced by spaces.

    Newlines are preserved so character indices and line numbers stay 1:1 with `source`.
    """
    n = len(source)
    chars = list(source)
    i = 0

    def blank_range(start: int, end: int) -> None:
        for idx in range(start, end):
            if chars[idx] != "\n":
                chars[idx] = " "

    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""

        # Line comment: // ...
        if ch == "/" and nxt == "/":
            start = i
            i += 2
            while i < n and source[i] != "\n":
                i += 1
            blank_range(start, i)
            continue

        # Block comment: /* ... */ (supports nesting in Rust)
        if ch == "/" and nxt == "*":
            start = i
            i += 2
            depth = 1
            while i < n and depth > 0:
                if language == "rust" and source[i] == "/" and i + 1 < n and source[i + 1] == "*":
                    depth += 1
                    i += 2
                elif source[i] == "*" and i + 1 < n and source[i + 1] == "/":
                    depth -= 1
                    i += 2
                else:
                    i += 1
            blank_range(start, i)
            continue

        # Double-quoted string literal
        if ch == '"':
            start = i
            i += 1
            while i < n:
                if source[i] == "\\" and i + 1 < n:
                    i += 2
                elif source[i] == '"':
                    i += 1
                    break
                else:
                    i += 1
            blank_range(start, i)
            continue

        # Backtick template / raw string (TS/JS/Go)
        if ch == "`" and language in ("typescript", "javascript", "go"):
            start = i
            i += 1
            while i < n:
                if language != "go" and source[i] == "\\" and i + 1 < n:
                    i += 2
                elif source[i] == "`":
                    i += 1
                    break
                else:
                    i += 1
            blank_range(start, i)
            continue

        # Single-quoted string or char literal (preserve Rust lifetimes like &'a)
        if ch == "'":
            if language == "rust":
                # Only treat as char literal if closed on the same line within a few chars.
                close = source.find("'", i + 1, min(n, i + 6))
                if close != -1 and "\n" not in source[i + 1 : close]:
                    blank_range(i, close + 1)
                    i = close + 1
                    continue
                i += 1
                continue
            start = i
            i += 1
            while i < n:
                if source[i] == "\\" and i + 1 < n:
                    i += 2
                elif source[i] == "'":
                    i += 1
                    break
                else:
                    i += 1
            blank_range(start, i)
            continue

        i += 1

    return "".join(chars)


def _strip_comments_only(source: str, language: str) -> str:
    """Return same-length text with only comments replaced by spaces (strings intact)."""
    n = len(source)
    chars = list(source)
    i = 0

    def blank_range(start: int, end: int) -> None:
        for idx in range(start, end):
            if chars[idx] != "\n":
                chars[idx] = " "

    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            start = i
            i += 2
            while i < n and source[i] != "\n":
                i += 1
            blank_range(start, i)
            continue
        if ch == "/" and nxt == "*":
            start = i
            i += 2
            depth = 1
            while i < n and depth > 0:
                if language == "rust" and source[i] == "/" and i + 1 < n and source[i + 1] == "*":
                    depth += 1
                    i += 2
                elif source[i] == "*" and i + 1 < n and source[i + 1] == "/":
                    depth -= 1
                    i += 2
                else:
                    i += 1
            blank_range(start, i)
            continue
        if ch in ('"', "`") or (ch == "'" and language != "rust"):
            quote = ch
            i += 1
            while i < n:
                if not (quote == "`" and language == "go") and source[i] == "\\" and i + 1 < n:
                    i += 2
                elif source[i] == quote:
                    i += 1
                    break
                else:
                    i += 1
            continue
        i += 1

    return "".join(chars)


def _line_offsets(source: str) -> list[int]:
    offsets = [0]
    for idx, ch in enumerate(source):
        if ch == "\n":
            offsets.append(idx + 1)
    return offsets


def _offset_to_line(offsets: list[int], pos: int) -> int:
    lo, hi = 0, len(offsets) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if offsets[mid] <= pos:
            lo = mid + 1
        else:
            hi = mid - 1
    return hi + 1


def _find_matching_brace(sanitized: str, open_pos: int) -> int:
    depth = 0
    n = len(sanitized)
    for idx in range(open_pos, n):
        ch = sanitized[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return idx
    return n - 1


def _extract_preceding_doc_comment(lines: list[str], decl_line_1based: int) -> str:
    idx = decl_line_1based - 2
    while idx >= 0 and not lines[idx].strip():
        idx -= 1
    if idx < 0:
        return ""

    line = lines[idx].strip()
    if line.startswith("///") or line.startswith("//"):
        collected: list[str] = []
        while idx >= 0 and lines[idx].strip().startswith("//"):
            cleaned = lines[idx].strip().lstrip("/").strip()
            if cleaned:
                collected.append(cleaned)
            idx -= 1
        if collected:
            return _first_doc_line(collected[-1])
    elif line.endswith("*/"):
        collected = []
        while idx >= 0:
            cur = lines[idx].strip()
            cleaned = cur.lstrip("/*").rstrip("*/").strip()
            if cleaned:
                collected.append(cleaned)
            if "/*" in cur:
                break
            idx -= 1
        if collected:
            return _first_doc_line(collected[-1])
    return ""


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_c_style_imports(comment_free: str, language: str) -> list[str]:
    imports: list[str] = []
    if language in ("typescript", "javascript"):
        # Multi-line or single-line ES imports and re-exports
        for match in re.finditer(
            r"""(?:^|[\n;])\s*((?:import|export)\s+(?:type\s+)?(?:[\s\S]*?\s+from\s+)?["'][^"']+["']\s*;?)""",
            comment_free,
        ):
            stmt = _collapse_ws(match.group(1)).rstrip(";")
            imports.append(stmt)
        # require('...')
        for match in re.finditer(r"""\brequire\s*\(\s*["']([^"']+)["']\s*\)""", comment_free):
            imports.append(f"require('{match.group(1)}')")
    elif language == "go":
        for match in re.finditer(r"""\bimport\s*\(([\s\S]*?)\)""", comment_free):
            block = match.group(1)
            for pkg_match in re.finditer(r"""(?:[A-Za-z_][A-Za-z0-9_]*\s+)?["']([^"']+)["']""", block):
                imports.append(f'import "{pkg_match.group(1)}"')
        for match in re.finditer(
            r"""(?:^|\n)\s*import\s+(?:[A-Za-z_][A-Za-z0-9_]*\s+)?["']([^"']+)["']""",
            comment_free,
        ):
            imports.append(f'import "{match.group(1)}"')
    elif language == "rust":
        for match in re.finditer(
            r"""(?:^|\n)\s*((?:pub\s+)?(?:use|mod)\s+[^;{]+;)""",
            comment_free,
        ):
            imports.append(_collapse_ws(match.group(1)))
    elif language == "java":
        for match in re.finditer(
            r"""(?:^|\n)\s*(import\s+(?:static\s+)?[A-Za-z0-9_.*]+\s*;)""",
            comment_free,
        ):
            imports.append(_collapse_ws(match.group(1)))
    return list(dict.fromkeys(imports))


def _scan_decl_end(sanitized: str, start_idx: int) -> tuple[int, str]:
    """Scan from declaration start to top-level `{`, `;`, or line end."""
    paren_depth = 0
    bracket_depth = 0
    angle_depth = 0
    n = len(sanitized)
    i = start_idx
    while i < n:
        ch = sanitized[i]
        if ch == "(":
            paren_depth += 1
        elif ch == ")" and paren_depth > 0:
            paren_depth -= 1
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]" and bracket_depth > 0:
            bracket_depth -= 1
        elif ch == "<":
            # Distinguish generics from comparison operators conservatively.
            if i == 0 or sanitized[i - 1] != " ":
                angle_depth += 1
        elif ch == ">" and angle_depth > 0:
            if i == 0 or sanitized[i - 1] != "=":
                angle_depth -= 1
        elif paren_depth == 0 and bracket_depth == 0:
            if ch == "{":
                return i, "{"
            if ch == ";":
                return i, ";"
        i += 1
    return n, ""


_TS_DECL_RE = re.compile(
    r"""^[ \t]*(?:export\s+(?:default\s+)?)?(?:declare\s+)?(?:abstract\s+)?"""
    r"""(?:(?P<class>class)\s+(?P<class_name>[A-Za-z_$][A-Za-z0-9_$]*)|"""
    r"""(?P<iface>interface)\s+(?P<iface_name>[A-Za-z_$][A-Za-z0-9_$]*)|"""
    r"""(?P<type>type|enum)\s+(?P<type_name>[A-Za-z_$][A-Za-z0-9_$]*)|"""
    r"""(?:async\s+)?(?P<func>function)\s*\*?\s*(?P<func_name>[A-Za-z_$][A-Za-z0-9_$]*)|"""
    r"""(?P<arrow>const|let)\s+(?P<arrow_name>[A-Za-z_$][A-Za-z0-9_$]*)\s*(?::[^=]+)?=\s*(?:async\s+)?(?:\([^)]*\)|[A-Za-z_$][A-Za-z0-9_$]*)\s*(?::[^=]+)?=>)"""
)

_TS_METHOD_RE = re.compile(
    r"""^[ \t]*(?:(?:public|private|protected|readonly|static|override|abstract|async)\s+)*"""
    r"""(?:(get|set)\s+)?(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*(?:<[^>\n]+>)?\s*\("""
)

_TS_KEYWORDS_NOT_METHODS: frozenset[str] = frozenset(
    {"if", "for", "while", "switch", "catch", "return", "throw", "new", "typeof", "await"}
)

_GO_DECL_RE = re.compile(
    r"""^[ \t]*(?:"""
    r"""func\s+(?:\([^)]+\)\s*)?(?P<func_name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]+\])?\s*\(|"""
    r"""type\s+(?P<type_name>[A-Za-z_][A-Za-z0-9_]*)\s+(?P<type_kind>struct|interface)\b"""
    r""")"""
)

_RUST_DECL_RE = re.compile(
    r"""^[ \t]*(?:pub(?:\([^)]+\))?\s+)?(?:unsafe\s+)?(?:async\s+)?"""
    r"""(?:(?P<fn>fn)\s+(?P<fn_name>[A-Za-z_][A-Za-z0-9_]*)|"""
    r"""(?P<struct>struct|enum|type)\s+(?P<struct_name>[A-Za-z_][A-Za-z0-9_]*)|"""
    r"""(?P<trait>trait)\s+(?P<trait_name>[A-Za-z_][A-Za-z0-9_]*)|"""
    r"""(?P<impl>impl)(?:<[^>\n]+>)?\s+(?:[A-Za-z_][A-Za-z0-9_:<>]*\s+for\s+)?(?P<impl_name>[A-Za-z_][A-Za-z0-9_]*))"""
)

_JAVA_TYPE_RE = re.compile(
    r"""^[ \t]*(?:(?:public|protected|private|abstract|final|static|sealed|non-sealed)\s+)*"""
    r"""(?P<kind>class|interface|enum|record)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)"""
)

_JAVA_METHOD_RE = re.compile(
    r"""^[ \t]*(?:(?:public|protected|private|abstract|final|static|synchronized|default|native)\s+)+"""
    r"""(?:<[^>\n]+>\s+)?[A-Za-z_][A-Za-z0-9_<>\[\]., ?]+\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\("""
)


def _parse_c_style_file(
    rel_path: str,
    source: str,
    language: str,
    token_count: int,
) -> tuple[list[str], list[SymbolInfo], str, int]:
    comment_free = _strip_comments_only(source, language)
    sanitized = _blank_comments_and_strings(source, language)
    imports = _extract_c_style_imports(comment_free, language)

    lines = source.splitlines()
    sanitized_lines = sanitized.splitlines()
    offsets = _line_offsets(source)

    symbols: list[SymbolInfo] = []
    skeleton_lines: list[str] = list(imports)

    # Track enclosing class / impl / interface scopes via brace close index.
    containers: list[tuple[str, int]] = []  # (name, closing_brace_offset)
    i_line = 0
    num_lines = len(sanitized_lines)
    last_consumed_offset = -1

    while i_line < num_lines:
        line_start = offsets[i_line]
        if line_start <= last_consumed_offset:
            i_line += 1
            continue

        while containers and line_start > containers[-1][1]:
            containers.pop()

        s_line = sanitized_lines[i_line]
        if not s_line.strip():
            i_line += 1
            continue

        kind: str | None = None
        name: str | None = None
        is_container = False

        if language in ("typescript", "javascript"):
            m = _TS_DECL_RE.match(s_line)
            if m and not containers:
                if m.group("class"):
                    kind, name, is_container = "class", m.group("class_name"), True
                elif m.group("iface"):
                    kind, name, is_container = "interface", m.group("iface_name"), True
                elif m.group("type"):
                    kind, name = "type", m.group("type_name")
                elif m.group("func"):
                    kind, name = "function", m.group("func_name")
                elif m.group("arrow"):
                    kind, name = "function", m.group("arrow_name")
            elif containers:
                mm = _TS_METHOD_RE.match(s_line)
                if mm and mm.group("name") not in _TS_KEYWORDS_NOT_METHODS:
                    kind, name = "method", mm.group("name")

        elif language == "go":
            m = _GO_DECL_RE.match(s_line)
            if m:
                if m.group("func_name"):
                    kind = "method" if s_line.lstrip().startswith("func (") else "function"
                    name = m.group("func_name")
                elif m.group("type_name"):
                    t_kind = m.group("type_kind")
                    kind = "interface" if t_kind == "interface" else "type"
                    name = m.group("type_name")

        elif language == "rust":
            m = _RUST_DECL_RE.match(s_line)
            if m:
                if m.group("fn"):
                    kind = "method" if containers else "function"
                    name = m.group("fn_name")
                elif m.group("struct"):
                    kind = "type"
                    name = m.group("struct_name")
                elif m.group("trait"):
                    kind, name, is_container = "interface", m.group("trait_name"), True
                elif m.group("impl"):
                    kind, name, is_container = "type", m.group("impl_name"), True

        elif language == "java":
            m = _JAVA_TYPE_RE.match(s_line)
            if m:
                j_kind = m.group("kind")
                kind = "interface" if j_kind == "interface" else ("type" if j_kind == "enum" else "class")
                name = m.group("name")
                is_container = True
            elif containers:
                mm = _JAVA_METHOD_RE.match(s_line)
                if mm and mm.group("name") not in _TS_KEYWORDS_NOT_METHODS:
                    kind, name = "method", mm.group("name")

        if kind is None or name is None:
            i_line += 1
            continue

        decl_end, term = _scan_decl_end(sanitized, line_start)
        raw_sig = source[line_start:decl_end]
        sig = _collapse_ws(raw_sig)
        start_line = i_line + 1

        if term == "{":
            close_pos = _find_matching_brace(sanitized, decl_end)
            end_line = _offset_to_line(offsets, close_pos)
            # Advance past declaration signature lines so multi-line params aren't re-scanned.
            sig_end_line = _offset_to_line(offsets, decl_end)
            last_consumed_offset = decl_end if is_container else close_pos
        else:
            close_pos = decl_end
            end_line = _offset_to_line(offsets, decl_end)
            sig_end_line = end_line
            last_consumed_offset = decl_end

        doc = _extract_preceding_doc_comment(lines, start_line)
        parent = containers[-1][0] if (containers and kind == "method") else None

        symbols.append(
            SymbolInfo(
                name=name,
                kind=kind,
                signature=sig,
                docstring_summary=doc,
                start_line=start_line,
                end_line=end_line,
                parent=parent,
            )
        )

        indent = "  " if parent else ""
        if doc:
            skeleton_lines.append(f"{indent}// {doc}")
        if term == "{":
            skeleton_lines.append(f"{indent}{sig} {{ ... }}")
        elif term == ";":
            skeleton_lines.append(f"{indent}{sig};")
        else:
            skeleton_lines.append(f"{indent}{sig}")

        if is_container and term == "{":
            containers.append((name, close_pos))

        i_line = max(i_line + 1, sig_end_line)

    skeleton, sk_tokens = _finalize_skeleton(rel_path, token_count, skeleton_lines)
    return imports, symbols, skeleton, sk_tokens


# ----------------------------------------------------------------------
# Markdown & reStructuredText Parser
# ----------------------------------------------------------------------

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_MD_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_RST_UNDERLINE_RE = re.compile(r"^([=\-~^\"#*+_`])\1{2,}\s*$")


def _parse_markdown_file(
    rel_path: str,
    source: str,
    token_count: int,
) -> tuple[list[str], list[SymbolInfo], str, int]:
    lines = source.splitlines()
    symbols: list[SymbolInfo] = []
    skeleton_lines: list[str] = []
    in_fence = False
    fence_marker = ""
    stack: list[tuple[int, str]] = []  # (level, heading_name)

    for lineno, line in enumerate(lines, start=1):
        fence_match = _MD_FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            continue

        if in_fence:
            continue

        m = _MD_HEADING_RE.match(line)
        if m:
            hashes, title = m.group(1), m.group(2).strip()
            level = len(hashes)
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent = stack[-1][1] if stack else None
            stack.append((level, title))

            symbols.append(
                SymbolInfo(
                    name=title,
                    kind="heading",
                    signature=f"{hashes} {title}",
                    start_line=lineno,
                    end_line=lineno,
                    parent=parent,
                )
            )
            skeleton_lines.append(f"{hashes} {title}")

    skeleton, sk_tokens = _finalize_skeleton(rel_path, token_count, skeleton_lines)
    return [], symbols, skeleton, sk_tokens


def _parse_rst_file(
    rel_path: str,
    source: str,
    token_count: int,
) -> tuple[list[str], list[SymbolInfo], str, int]:
    lines = source.splitlines()
    symbols: list[SymbolInfo] = []
    skeleton_lines: list[str] = []
    char_levels: dict[str, int] = {}
    stack: list[tuple[int, str]] = []

    i = 0
    n = len(lines)
    while i + 1 < n:
        title_line = lines[i].strip()
        underline = lines[i + 1].strip()
        if title_line and _RST_UNDERLINE_RE.match(underline) and len(underline) >= len(title_line):
            adornment = underline[0]
            if adornment not in char_levels:
                char_levels[adornment] = len(char_levels) + 1
            level = char_levels[adornment]
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent = stack[-1][1] if stack else None
            stack.append((level, title_line))

            hashes = "#" * min(level, 6)
            symbols.append(
                SymbolInfo(
                    name=title_line,
                    kind="heading",
                    signature=f"{hashes} {title_line}",
                    start_line=i + 1,
                    end_line=i + 2,
                    parent=parent,
                )
            )
            skeleton_lines.append(f"{hashes} {title_line}")
            i += 2
            continue
        i += 1

    skeleton, sk_tokens = _finalize_skeleton(rel_path, token_count, skeleton_lines)
    return [], symbols, skeleton, sk_tokens


def _parse_generic_file(
    rel_path: str,
    source: str,
    token_count: int,
) -> tuple[list[str], list[SymbolInfo], str, int]:
    non_empty = [line.rstrip() for line in source.splitlines() if line.strip()][:15]
    skeleton, sk_tokens = _finalize_skeleton(rel_path, token_count, non_empty)
    return [], [], skeleton, sk_tokens


def _parse_file_record(
    rel_path: str,
    source: str,
    sha256_short: str,
    mtime_ns: int,
    size: int,
) -> FileRecord:
    """Parse source text into a complete FileRecord."""
    language = _detect_language(rel_path)
    token_count = estimate_tokens(source)

    if language == "python":
        imports, symbols, skeleton, sk_tokens = _parse_python_file(
            rel_path, source, token_count
        )
    elif language in ("typescript", "javascript", "go", "rust", "java"):
        imports, symbols, skeleton, sk_tokens = _parse_c_style_file(
            rel_path, source, language, token_count
        )
    elif language == "markdown":
        imports, symbols, skeleton, sk_tokens = _parse_markdown_file(
            rel_path, source, token_count
        )
    elif language == "rst":
        imports, symbols, skeleton, sk_tokens = _parse_rst_file(
            rel_path, source, token_count
        )
    else:
        imports, symbols, skeleton, sk_tokens = _parse_generic_file(
            rel_path, source, token_count
        )

    return FileRecord(
        rel_path=rel_path,
        language=language,
        sha256=sha256_short,
        mtime_ns=mtime_ns,
        token_count=token_count,
        imports=imports,
        symbols=symbols,
        skeleton=skeleton,
        skeleton_tokens=sk_tokens,
        size=size,
    )


# ----------------------------------------------------------------------
# Local Import Resolver & Git Co-Change Analyzer
# ----------------------------------------------------------------------

_PY_FROM_RE = re.compile(r"^from\s+(\.*)([A-Za-z0-9_.]*)\s+import\s+(.+)$")
_PY_IMPORT_RE = re.compile(r"^import\s+(.+)$")
_JS_SPEC_RE = re.compile(
    r"""(?:from\s+|import\s+|require\s*\(\s*|import\s*\(\s*)["']([^"']+)["']"""
)
_GO_IMPORT_RE = re.compile(r"""["']([^"']+)["']""")
_RUST_MOD_RE = re.compile(r"^(?:pub\s+)?mod\s+([A-Za-z0-9_]+)\s*;")
_RUST_USE_RE = re.compile(r"^(?:pub\s+)?use\s+([A-Za-z0-9_:{}*,\s]+)\s*;")
_JAVA_IMPORT_RE = re.compile(r"^import\s+(?:static\s+)?([A-Za-z0-9_.]+)\s*;")

_JS_EXTENSIONS: tuple[str, ...] = (
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mts",
    ".cts",
    ".mjs",
    ".cjs",
    ".json",
)


def _resolve_py_module_candidates(
    module_path: str,
    known_files: set[str],
    src_roots: tuple[str, ...],
) -> list[str]:
    if not module_path:
        return []
    slash_path = module_path.replace(".", "/")
    candidates: list[str] = []
    prefixes = ("",) + src_roots
    for prefix in prefixes:
        base = f"{prefix}/{slash_path}" if prefix else slash_path
        for suffix in (".py", ".pyi", "/__init__.py", "/__init__.pyi"):
            cand = base + suffix
            if cand in known_files:
                candidates.append(cand)
    return candidates


def _resolve_python_import(
    source_rel_path: str,
    import_stmt: str,
    known_files: set[str],
    src_roots: tuple[str, ...],
) -> list[str]:
    resolved: list[str] = []
    stmt = import_stmt.strip()

    from_match = _PY_FROM_RE.match(stmt)
    if from_match:
        dots = len(from_match.group(1))
        module = from_match.group(2)
        names_raw = from_match.group(3)
        imported_names = [
            part.strip().split(" as ")[0].strip()
            for part in names_raw.strip("() ").split(",")
            if part.strip()
        ]

        if dots > 0:
            cur_dir = PurePosixPath(source_rel_path).parent
            parts = () if str(cur_dir) == "." else cur_dir.parts
            up = dots - 1
            if up > len(parts):
                return []
            base_parts = parts[: len(parts) - up]
            base_prefix = "/".join(base_parts)

            if module:
                mod_rel = module.replace(".", "/")
                target_base = f"{base_prefix}/{mod_rel}" if base_prefix else mod_rel
                for suffix in (".py", ".pyi", "/__init__.py", "/__init__.pyi"):
                    cand = target_base + suffix
                    if cand in known_files:
                        resolved.append(cand)
                for name in imported_names:
                    if name == "*":
                        continue
                    sub_base = f"{target_base}/{name}"
                    for suffix in (".py", ".pyi", "/__init__.py"):
                        cand = sub_base + suffix
                        if cand in known_files:
                            resolved.append(cand)
            else:
                for name in imported_names:
                    if name == "*":
                        continue
                    sub_base = f"{base_prefix}/{name}" if base_prefix else name
                    found_sub = False
                    for suffix in (".py", ".pyi", "/__init__.py"):
                        cand = sub_base + suffix
                        if cand in known_files:
                            resolved.append(cand)
                            found_sub = True
                    if not found_sub:
                        pkg_init = (
                            f"{base_prefix}/__init__.py"
                            if base_prefix
                            else "__init__.py"
                        )
                        if pkg_init in known_files:
                            resolved.append(pkg_init)
        else:
            direct = _resolve_py_module_candidates(module, known_files, src_roots)
            resolved.extend(direct)
            for name in imported_names:
                if name == "*":
                    continue
                sub_mod = f"{module}.{name}" if module else name
                resolved.extend(
                    _resolve_py_module_candidates(sub_mod, known_files, src_roots)
                )
        return resolved

    imp_match = _PY_IMPORT_RE.match(stmt)
    if imp_match:
        for item in imp_match.group(1).split(","):
            mod_name = item.strip().split(" as ")[0].strip()
            resolved.extend(
                _resolve_py_module_candidates(mod_name, known_files, src_roots)
            )
        return resolved

    # Bare dotted module path fallback
    if re.match(r"^[A-Za-z_][A-Za-z0-9_.]*$", stmt):
        resolved.extend(_resolve_py_module_candidates(stmt, known_files, src_roots))
    return resolved


def _resolve_js_import(
    source_rel_path: str,
    import_stmt: str,
    known_files: set[str],
) -> list[str]:
    m = _JS_SPEC_RE.search(import_stmt)
    spec = m.group(1) if m else import_stmt.strip().strip("'\"")
    if not spec:
        return []

    candidate_bases: list[str] = []
    if spec.startswith(("./", "../")):
        cur_dir = posixpath.dirname(source_rel_path)
        joined = posixpath.normpath(posixpath.join(cur_dir, spec))
        if not joined.startswith(".."):
            candidate_bases.append(joined)
    elif spec.startswith(("@/", "~/")):
        stripped = spec[2:]
        candidate_bases.append(posixpath.normpath(stripped))
        candidate_bases.append(posixpath.normpath(f"src/{stripped}"))
    elif "/" in spec:
        candidate_bases.append(posixpath.normpath(spec))
        candidate_bases.append(posixpath.normpath(f"src/{spec}"))
    else:
        return []

    resolved: list[str] = []
    for base in candidate_bases:
        if base in known_files:
            resolved.append(base)
            continue

        # Handle TypeScript ESM convention importing `.js` / `.jsx` for `.ts` / `.tsx` sources.
        if base.endswith(".js"):
            stem = base[:-3]
            for ext in (".ts", ".tsx", ".js", ".jsx"):
                if stem + ext in known_files:
                    resolved.append(stem + ext)
                    break
        elif base.endswith(".jsx"):
            stem = base[:-4]
            for ext in (".tsx", ".jsx"):
                if stem + ext in known_files:
                    resolved.append(stem + ext)
                    break

        for ext in _JS_EXTENSIONS:
            if base + ext in known_files:
                resolved.append(base + ext)
        for ext in _JS_EXTENSIONS:
            idx_path = f"{base}/index{ext}"
            if idx_path in known_files:
                resolved.append(idx_path)

    return resolved


def _resolve_other_import(
    source_rel_path: str,
    language: str,
    import_stmt: str,
    known_files: set[str],
    go_dir_map: dict[str, list[str]],
) -> list[str]:
    resolved: list[str] = []
    if language == "go":
        m = _GO_IMPORT_RE.search(import_stmt)
        if not m:
            return []
        pkg_path = m.group(1).strip("/")
        for go_dir, files in go_dir_map.items():
            if pkg_path == go_dir or pkg_path.endswith(f"/{go_dir}"):
                resolved.extend(files)
    elif language == "rust":
        cur_dir = posixpath.dirname(source_rel_path)
        mod_m = _RUST_MOD_RE.match(import_stmt)
        if mod_m:
            mod_name = mod_m.group(1)
            for cand in (
                posixpath.normpath(posixpath.join(cur_dir, f"{mod_name}.rs")),
                posixpath.normpath(posixpath.join(cur_dir, mod_name, "mod.rs")),
            ):
                if cand in known_files:
                    resolved.append(cand)
            return resolved

        use_m = _RUST_USE_RE.match(import_stmt)
        if use_m:
            raw_path = use_m.group(1).split("{")[0].strip().rstrip(":")
            parts = [p.strip() for p in raw_path.split("::") if p.strip()]
            if not parts:
                return []
            if parts[0] == "crate":
                rel_parts = parts[1:]
                base_dir = "src"
            elif parts[0] == "super":
                rel_parts = parts[1:]
                base_dir = posixpath.normpath(posixpath.join(cur_dir, ".."))
            elif parts[0] == "self":
                rel_parts = parts[1:]
                base_dir = cur_dir
            else:
                return []
            for depth in (len(rel_parts), max(0, len(rel_parts) - 1)):
                if depth == 0:
                    continue
                sub = "/".join(rel_parts[:depth])
                for cand in (
                    posixpath.normpath(posixpath.join(base_dir, f"{sub}.rs")),
                    posixpath.normpath(posixpath.join(base_dir, sub, "mod.rs")),
                ):
                    if cand in known_files:
                        resolved.append(cand)
    elif language == "java":
        m = _JAVA_IMPORT_RE.match(import_stmt)
        if m:
            fqcn = m.group(1)
            if fqcn.endswith(".*"):
                pkg_suffix = fqcn[:-2].replace(".", "/") + "/"
                for kf in known_files:
                    if kf.endswith(".java") and pkg_suffix in kf:
                        resolved.append(kf)
            else:
                class_suffix = fqcn.replace(".", "/") + ".java"
                for kf in known_files:
                    if kf == class_suffix or kf.endswith(f"/{class_suffix}"):
                        resolved.append(kf)
    return resolved


def resolve_local_imports(
    files: dict[str, FileRecord],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Resolve import strings in `files` to local `rel_path` entries.

    Returns:
        `(import_graph, reverse_graph)` where keys cover all indexed files.
    """
    known_files = set(files.keys())
    src_roots: tuple[str, ...] = ("src",) if any(p.startswith("src/") for p in known_files) else ()

    go_dir_map: dict[str, list[str]] = defaultdict(list)
    for rel_path, record in files.items():
        if record.language == "go":
            go_dir = posixpath.dirname(rel_path) or "."
            go_dir_map[go_dir].append(rel_path)

    import_graph: dict[str, list[str]] = {rel: [] for rel in files}
    reverse_sets: dict[str, set[str]] = {rel: set() for rel in files}

    for rel_path, record in files.items():
        targets: set[str] = set()
        for imp in record.imports:
            if record.language == "python":
                matched = _resolve_python_import(rel_path, imp, known_files, src_roots)
            elif record.language in ("typescript", "javascript"):
                matched = _resolve_js_import(rel_path, imp, known_files)
            else:
                matched = _resolve_other_import(
                    rel_path, record.language, imp, known_files, go_dir_map
                )

            for target in matched:
                if target != rel_path and target in known_files:
                    targets.add(target)

        sorted_targets = sorted(targets)
        import_graph[rel_path] = sorted_targets
        for target in sorted_targets:
            reverse_sets[target].add(rel_path)

    reverse_graph = {rel: sorted(sources) for rel, sources in reverse_sets.items()}
    return import_graph, reverse_graph


def extract_git_cochange(
    root: Path | str,
    max_commits: int = 100,
    *,
    valid_files: set[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Extract co-change edge weights from recent git commits safely.

    Returns `{}` when `root` is not inside a git repository, has no commits,
    or git is unavailable. Commits touching more than 20 files are skipped as
    bulk/formatting commits.
    """
    if max_commits <= 0:
        return {}

    root_path = Path(root).resolve()
    try:
        probe = subprocess.run(
            ["git", "-C", str(root_path), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
        if probe.returncode != 0 or probe.stdout.strip().lower() != "true":
            return {}

        result = subprocess.run(
            [
                "git",
                "-C",
                str(root_path),
                "log",
                f"-n{max_commits}",
                "--relative",
                "--no-renames",
                "--name-only",
                "--pretty=format:__CTX_COMMIT__%H",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}

    if result.returncode != 0 or not result.stdout:
        return {}

    commits: list[list[str]] = []
    current_files: list[str] = []

    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("__CTX_COMMIT__"):
            if current_files:
                commits.append(current_files)
                current_files = []
            continue
        rel = line.replace("\\", "/")
        while rel.startswith("./"):
            rel = rel[2:]
        if not rel:
            continue
        if valid_files is not None and rel not in valid_files:
            continue
        current_files.append(rel)

    if current_files:
        commits.append(current_files)

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for commit_files in commits:
        unique_files = sorted(set(commit_files))
        if len(unique_files) < 2 or len(unique_files) > _MAX_COCHANGE_COMMIT_FILES:
            continue
        for i, file_a in enumerate(unique_files):
            for file_b in unique_files[i + 1 :]:
                counts[file_a][file_b] += 1
                counts[file_b][file_a] += 1

    return {
        file_a: dict(sorted(peers.items()))
        for file_a, peers in sorted(counts.items())
    }


# ----------------------------------------------------------------------
# Public RepoIndexer
# ----------------------------------------------------------------------


class RepoIndexer:
    """Incremental codebase indexer, AST skeletonizer, and neighbor analyzer."""

    def __init__(self, root: Path | str, store: ContextStore | None = None) -> None:
        self._root = Path(root).resolve()
        self._store = store
        self._index: RepoIndex | None = None

    @property
    def root(self) -> Path:
        return self._root

    @property
    def cache_path(self) -> Path:
        ctx_dir = self._store.ctx_dir if self._store is not None else (self._root / ".ctx")
        return ctx_dir / "cache" / "index.json"

    def _load_cached_index(self) -> RepoIndex | None:
        if not self.cache_path.is_file():
            return None
        try:
            raw = self.cache_path.read_text(encoding="utf-8")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                return None
            if payload.get("version") != _CACHE_SCHEMA_VERSION:
                return None
            if str(payload.get("root", "")) != str(self._root):
                return None
            return RepoIndex.from_dict(payload)
        except (OSError, json.JSONDecodeError, ValueError, TypeError, KeyError):
            return None

    def _save_cached_index(self, index: RepoIndex) -> None:
        _atomic_write_text(
            self.cache_path,
            json.dumps(index.to_dict(), ensure_ascii=False, indent=2) + "\n",
        )

    def _discover_files(self) -> list[tuple[str, Path, int, int]]:
        """Walk repository respecting ignore rules, size limits, and secret/binary filters."""
        root_matcher = _IgnoreMatcher()
        root_matcher.add_ignore_file(self._root / ".gitignore", base_dir="")
        root_matcher.add_ignore_file(self._root / ".ctxignore", base_dir="")

        discovered: list[tuple[str, Path, int, int]] = []
        dir_matchers: dict[Path, _IgnoreMatcher] = {self._root: root_matcher}

        for dirpath_str, dirnames, filenames in os.walk(self._root, followlinks=False):
            current_dir = Path(dirpath_str)
            rel_dir = (
                ""
                if current_dir == self._root
                else current_dir.relative_to(self._root).as_posix()
            )
            matcher = dir_matchers.get(current_dir, root_matcher)

            if current_dir != self._root:
                local_gi = current_dir / ".gitignore"
                local_ci = current_dir / ".ctxignore"
                if local_gi.is_file() or local_ci.is_file():
                    matcher = matcher.copy_with_file(local_gi, base_dir=rel_dir)
                    matcher.add_ignore_file(local_ci, base_dir=rel_dir)

            kept_dirs: list[str] = []
            for dname in sorted(dirnames):
                if dname in _EXCLUDED_DIRS:
                    continue
                sub_path = current_dir / dname
                if sub_path.is_symlink():
                    continue
                rel_sub = f"{rel_dir}/{dname}" if rel_dir else dname
                if matcher.is_ignored(rel_sub, is_dir=True):
                    continue
                kept_dirs.append(dname)
                dir_matchers[sub_path] = matcher
            dirnames[:] = kept_dirs

            for fname in sorted(filenames):
                if _is_secret_or_binary_name(fname):
                    continue
                file_path = current_dir / fname
                try:
                    resolved = file_path.resolve()
                except OSError:
                    continue
                if not resolved.is_relative_to(self._root):
                    continue

                rel_file = f"{rel_dir}/{fname}" if rel_dir else fname
                if matcher.is_ignored(rel_file, is_dir=False):
                    continue

                try:
                    stat = file_path.stat()
                except OSError:
                    continue

                if not file_path.is_file() or stat.st_size > _MAX_FILE_BYTES:
                    continue

                discovered.append((rel_file, file_path, stat.st_mtime_ns, stat.st_size))

        return discovered

    def resolve_local_imports(
        self, index: RepoIndex
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Populate `import_graph` and `reverse_graph` on `index`."""
        import_graph, reverse_graph = resolve_local_imports(index.files)
        index.import_graph = import_graph
        index.reverse_graph = reverse_graph
        return import_graph, reverse_graph

    def extract_git_cochange(
        self,
        max_commits: int = 100,
        *,
        valid_files: set[str] | None = None,
    ) -> dict[str, dict[str, int]]:
        """Extract git co-change matrix for `self.root`."""
        return extract_git_cochange(
            self._root,
            max_commits=max_commits,
            valid_files=valid_files,
        )

    def build_or_update(self, force: bool = False) -> RepoIndex:
        """Incrementally index repository files, resolve graphs, and update cache."""
        cached_index = None if force else self._load_cached_index()
        cached_files = cached_index.files if cached_index is not None else {}

        discovered = self._discover_files()
        new_files: dict[str, FileRecord] = {}

        for rel_path, abs_path, mtime_ns, size in discovered:
            cached_rec = cached_files.get(rel_path)

            # Fast path: identical mtime_ns and byte size -> cache hit.
            if (
                not force
                and cached_rec is not None
                and cached_rec.mtime_ns == mtime_ns
                and cached_rec.size == size
            ):
                new_files[rel_path] = cached_rec
                continue

            try:
                raw_bytes = abs_path.read_bytes()
            except OSError:
                continue

            if _looks_binary(raw_bytes):
                continue

            sha256_short = hashlib.sha256(raw_bytes).hexdigest()[:16]

            # Content hash match after stat/mtime change -> reuse parsed AST/skeleton.
            if (
                not force
                and cached_rec is not None
                and cached_rec.sha256 == sha256_short
                and cached_rec.size == size
            ):
                cached_rec.mtime_ns = mtime_ns
                new_files[rel_path] = cached_rec
                continue

            source_text = raw_bytes.decode("utf-8")
            new_files[rel_path] = _parse_file_record(
                rel_path=rel_path,
                source=source_text,
                sha256_short=sha256_short,
                mtime_ns=mtime_ns,
                size=size,
            )

        index = RepoIndex(root=str(self._root), files=new_files)
        self.resolve_local_imports(index)
        index.cochange_graph = self.extract_git_cochange(
            max_commits=100,
            valid_files=set(new_files.keys()),
        )

        self._save_cached_index(index)
        self._index = index
        return index

    def get_neighbors(
        self,
        seed_files: list[str],
        hops: int = 1,
    ) -> list[tuple[str, float, str]]:
        """Return neighboring files with relevance score and primary reason.

        Reasons are strictly one of: `"imports"`, `"imported_by"`, `"git_cochange"`.
        """
        if hops <= 0 or not seed_files:
            return []

        if self._index is None:
            self.build_or_update()
        assert self._index is not None
        index = self._index

        normalized_seeds: list[str] = []
        for seed in seed_files:
            norm = seed.replace("\\", "/").strip()
            while norm.startswith("./"):
                norm = norm[2:]
            if norm and norm in index.files:
                normalized_seeds.append(norm)

        if not normalized_seeds:
            return []

        seed_set = set(normalized_seeds)
        reason_priority = {"imports": 3, "imported_by": 2, "git_cochange": 1}

        scores: dict[str, float] = {}
        best_contrib: dict[str, tuple[float, int, str]] = {}

        visited: set[str] = set(seed_set)
        frontier: list[str] = list(dict.fromkeys(normalized_seeds))

        for hop in range(1, hops + 1):
            decay = 0.5 ** (hop - 1)
            next_frontier: list[str] = []

            for node in frontier:
                candidates: list[tuple[str, float, str]] = []

                for target in index.import_graph.get(node, []):
                    candidates.append((target, 1.0 * decay, "imports"))

                for caller in index.reverse_graph.get(node, []):
                    candidates.append((caller, 0.85 * decay, "imported_by"))

                for peer, commit_count in index.cochange_graph.get(node, {}).items():
                    weight = min(0.95, 0.35 + 0.12 * commit_count) * decay
                    candidates.append((peer, weight, "git_cochange"))

                for target, edge_score, reason in candidates:
                    if target in seed_set or target not in index.files:
                        continue

                    if target not in scores:
                        scores[target] = edge_score
                    else:
                        scores[target] = min(1.5, scores[target] + 0.25 * edge_score)

                    rank_key = (edge_score, reason_priority[reason], reason)
                    prev = best_contrib.get(target)
                    if prev is None or (rank_key[0], rank_key[1]) > (prev[0], prev[1]):
                        best_contrib[target] = rank_key

                    if target not in visited:
                        visited.add(target)
                        next_frontier.append(target)

            if not next_frontier:
                break
            frontier = next_frontier

        neighbors = [
            (rel_path, round(score, 4), best_contrib[rel_path][2])
            for rel_path, score in scores.items()
        ]
        neighbors.sort(key=lambda item: (-item[1], item[0]))
        return neighbors