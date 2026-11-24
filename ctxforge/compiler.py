"""Token-budgeted Context Compiler with BM25 retrieval and Lost-in-the-Middle packing."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Mapping, Sequence

from .indexer import FileRecord, RepoIndex, RepoIndexer
from .models import (
    CardKind,
    CardStatus,
    KnowledgeCard,
    ProjectMeta,
    SessionLogEntry,
    TaskNode,
)
from .storage import ContextStore
from .tokens import estimate_tokens, truncate_to_tokens

_DEFAULT_K1 = 1.5
_DEFAULT_B = 0.75
_MAX_BM25_TIER1_FALLBACK = 3
_MAX_TIER2_NEIGHBORS = 12
_MAX_GIT_STATUS_LINES = 15

_CYRILLIC_WORD_RE = re.compile(r"[\u0400-\u04FF\u0500-\u052F]+")
_TOKEN_SCAN_RE = re.compile(
    r"[\u0400-\u04FF\u0500-\u052F]+"
    r"|[A-Za-z][A-Za-z0-9_]*"
    r"|[0-9]+"
)
_CAMEL_SPLIT_RE = re.compile(
    r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|[0-9]+|[A-Z]+"
)

# Grammatical stopwords filtered out so BM25 matches domain/code semantics rather than glue words.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "for",
        "with",
        "from",
        "into",
        "onto",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "is",
        "are",
        "be",
        "this",
        "that",
        "it",
        "as",
        "not",
        "no",
        "и",
        "в",
        "во",
        "на",
        "по",
        "к",
        "из",
        "от",
        "до",
        "для",
        "с",
        "со",
        "при",
        "без",
        "под",
        "над",
        "о",
        "об",
        "а",
        "но",
        "или",
        "не",
        "что",
        "как",
        "это",
        "все",
        "всё",
    }
)

# Generic path extensions / folder names excluded from card BM25 matching so unrelated modules never leak.
_CARD_PATH_NOISE_TERMS: frozenset[str] = frozenset(
    {
        "py",
        "pyi",
        "ts",
        "tsx",
        "js",
        "jsx",
        "mjs",
        "cjs",
        "go",
        "rs",
        "java",
        "md",
        "mdx",
        "rst",
        "json",
        "toml",
        "yaml",
        "yml",
        "txt",
        "app",
        "src",
        "lib",
        "test",
        "tests",
    }
)

_GENERIC_ANCHOR_STEMS: frozenset[str] = frozenset(
    {
        "__init__",
        "init",
        "index",
        "mod",
        "main",
        "app",
        "src",
        "lib",
        "test",
        "tests",
        "utils",
        "helpers",
        "types",
        "models",
        "cli",
    }
)

_CTX_PROTOCOL_TEXT = """<ctx_protocol>
Rules:
1. Obey <project_invariants>; never suggest <rejected_approaches>.
2. Flag any invariant conflict or architectural shift before coding.
3. End your response with <ctx_delta> JSON (omit empty arrays):
<ctx_delta>
{"task_id":"...","task_status":"in_progress|done|blocked","decisions_added":[{"kind":"ADR|INV|PITFALL","title":"...","statement":"...","rationale":"...","tags":[],"anchors":["file.py"]}],"decisions_superseded":[{"old_id":"ADR-...","replacement":{...}}],"new_subtasks":[{"title":"...","description":"...","anchors":["..."]}],"next_anchor":"next step"}
</ctx_delta>
</ctx_protocol>"""


def _normalize_rel_path(path: str) -> str:
    norm = path.replace("\\", "/").strip()
    while norm.startswith("./"):
        norm = norm[2:]
    return norm


class BM25Scorer:
    """Zero-dependency Okapi BM25 scorer with CamelCase/snake_case and Cyrillic support."""

    def __init__(self, k1: float = _DEFAULT_K1, b: float = _DEFAULT_B) -> None:
        self.k1 = k1
        self.b = b
        self._doc_ids: list[str] = []
        self._doc_tf: list[Counter[str]] = []
        self._doc_len: list[int] = []
        self._df: Counter[str] = Counter()
        self._avgdl: float = 0.0

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Tokenize Latin/code identifiers (splitting CamelCase & snake_case) and Cyrillic."""
        if not text:
            return []
        tokens: list[str] = []
        for match in _TOKEN_SCAN_RE.finditer(text):
            chunk = match.group()
            if _CYRILLIC_WORD_RE.fullmatch(chunk):
                lowered = chunk.lower()
                if lowered not in _STOPWORDS:
                    tokens.append(lowered)
            elif chunk.isdigit():
                tokens.append(chunk)
            else:
                for segment in chunk.split("_"):
                    if not segment:
                        continue
                    subwords = _CAMEL_SPLIT_RE.findall(segment)
                    if not subwords:
                        lowered = segment.lower()
                        if lowered not in _STOPWORDS:
                            tokens.append(lowered)
                        continue
                    for sub in subwords:
                        lowered = sub.lower()
                        if lowered not in _STOPWORDS:
                            tokens.append(lowered)
        return tokens

    def fit(self, documents: Sequence[tuple[str, str]]) -> None:
        """Index (doc_id, text) pairs for BM25 scoring."""
        self._doc_ids = []
        self._doc_tf = []
        self._doc_len = []
        self._df = Counter()
        for doc_id, text in documents:
            tokens = self.tokenize(text)
            tf = Counter(tokens)
            self._doc_ids.append(doc_id)
            self._doc_tf.append(tf)
            self._doc_len.append(len(tokens))
            for term in tf:
                self._df[term] += 1
        total_len = sum(self._doc_len)
        self._avgdl = (total_len / len(self._doc_len)) if self._doc_len else 0.0

    def _idf(self, term: str) -> float:
        n_docs = len(self._doc_ids)
        if n_docs == 0:
            return 0.0
        df = self._df.get(term, 0)
        # Non-negative Robertson-Sparck Jones variant suitable for small and large corpora.
        return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    def score_all(self, query: str) -> list[tuple[str, float]]:
        """Score all indexed documents against query, sorted descending."""
        query_tokens = list(dict.fromkeys(self.tokenize(query)))
        if not query_tokens or not self._doc_ids or self._avgdl <= 0.0:
            return [(doc_id, 0.0) for doc_id in self._doc_ids]
        idfs = {term: self._idf(term) for term in query_tokens}
        results: list[tuple[str, float]] = []
        for idx, doc_id in enumerate(self._doc_ids):
            tf = self._doc_tf[idx]
            doc_len = self._doc_len[idx]
            if doc_len == 0:
                results.append((doc_id, 0.0))
                continue
            norm_factor = self.k1 * (1.0 - self.b + self.b * (doc_len / self._avgdl))
            score = 0.0
            for term in query_tokens:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                score += idfs[term] * ((freq * (self.k1 + 1.0)) / (freq + norm_factor))
            results.append((doc_id, score))
        results.sort(key=lambda item: (-item[1], item[0]))
        return results

    def rank_files(
        self,
        query: str,
        files: Mapping[str, FileRecord],
        *,
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Rank FileRecord entries by path, symbols, docstrings, and imports."""
        docs: list[tuple[str, str]] = []
        for rel_path, record in sorted(files.items()):
            symbol_parts: list[str] = []
            for sym in record.symbols:
                symbol_parts.append(sym.name)
                symbol_parts.append(sym.signature)
                if sym.docstring_summary:
                    symbol_parts.append(sym.docstring_summary)
            doc_text = "\n".join(
                [
                    rel_path,
                    rel_path,
                    " ".join(symbol_parts),
                    " ".join(record.imports),
                ]
            )
            docs.append((rel_path, doc_text))
        self.fit(docs)
        ranked = self.score_all(query)
        if top_k is not None:
            return ranked[: max(0, top_k)]
        return ranked

    def rank_cards(
        self,
        query: str,
        cards: Sequence[KnowledgeCard],
        *,
        top_k: int | None = None,
    ) -> list[tuple[KnowledgeCard, float]]:
        """Rank KnowledgeCard entries by title, tags, statement, rationale, and anchor symbols."""
        cards_by_id = {card.id: card for card in cards}
        docs: list[tuple[str, str]] = []
        for card in cards:
            tags_text = " ".join(card.tags)
            anchor_terms: list[str] = []
            for anchor in card.anchors:
                if anchor.symbol:
                    anchor_terms.append(anchor.symbol)
                stem = PurePosixPath(_normalize_rel_path(anchor.file_path)).stem
                if stem and stem.lower() not in _GENERIC_ANCHOR_STEMS:
                    anchor_terms.append(stem)
            anchor_text = " ".join(anchor_terms)
            doc_text = "\n".join(
                [
                    card.title,
                    card.title,
                    tags_text,
                    tags_text,
                    card.statement,
                    card.rationale,
                    anchor_text,
                ]
            )
            docs.append((card.id, doc_text))
        self.fit(docs)
        filtered_query = " ".join(
            tok
            for tok in self.tokenize(query)
            if tok not in _CARD_PATH_NOISE_TERMS
        )
        ranked = self.score_all(filtered_query)
        result = [(cards_by_id[card_id], score) for card_id, score in ranked]
        if top_k is not None:
            return result[: max(0, top_k)]
        return result


@dataclass(slots=True)
class CompileRequest:
    """Input specification for compiling a bounded LLM briefing capsule."""

    task_id: str | None = None
    query: str | None = None
    explicit_files: list[str] = field(default_factory=list)
    token_budget: int = 8000
    include_protocol: bool = True
    extra_instruction: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.token_budget, bool) or not isinstance(self.token_budget, int):
            raise ValueError("CompileRequest.token_budget must be an integer.")
        if self.token_budget <= 0:
            raise ValueError("CompileRequest.token_budget must be positive.")


@dataclass(slots=True)
class CompileResult:
    """Compiled prompt capsule, honest baseline comparison, and packing diagnostics."""

    capsule_text: str
    capsule_tokens: int
    full_repo_tokens: int
    savings_percent: float
    stats: dict[str, Any]

    def format_summary_report(self) -> str:
        """Render a concise human-readable compilation report for CLI output."""
        s = self.stats
        lines = [
            "=== ctxforge Context Capsule Summary ===",
            (
                f"Tokens: {self.capsule_tokens} / {s.get('token_budget', '?')} budget "
                f"(full unfiltered repo: {self.full_repo_tokens} tok, "
                f"savings: {self.savings_percent:.1f}%)"
            ),
            (
                f"Code: Tier-1 full={s.get('tier1_count', 0)}, "
                f"Tier-2 skeletons={s.get('tier2_count', 0)}, "
                f"Tier-3 tree={s.get('tier3_count', 0)}"
            ),
            (
                f"Cards (active={s.get('cards_total_active', 0)}): "
                f"INV={s.get('cards_invariants', 0)}, "
                f"ADR={s.get('cards_adrs', 0)}, "
                f"PITFALL={s.get('cards_pitfalls', 0)}"
            ),
            (
                f"Continuity: sessions={s.get('sessions_included', 0)}, "
                f"git={'yes' if s.get('git_included') else 'no'}, "
                f"protocol={'yes' if s.get('protocol_included') else 'no'}"
            ),
        ]
        dropped = s.get("dropped", [])
        if dropped:
            lines.append(f"Dropped by budget ({len(dropped)}): {', '.join(dropped)}")
        warnings = s.get("warnings", [])
        if warnings:
            lines.append("Warnings:")
            for warning in warnings:
                lines.append(f"  - {warning}")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Store & Git Adapter Helpers
# ----------------------------------------------------------------------


def _load_project_compat(store: Any) -> ProjectMeta:
    if hasattr(store, "get_project"):
        return store.get_project()
    return store.load_project()


def _list_sessions_compat(store: Any, limit: int = 3) -> list[SessionLogEntry]:
    if hasattr(store, "list_sessions"):
        return store.list_sessions(limit=limit)
    return store.load_recent_sessions(limit=limit)


def _get_task_with_deps_compat(
    store: Any, task_id: str
) -> tuple[TaskNode, list[TaskNode]]:
    res = store.get_task_with_deps(task_id)
    if hasattr(res, "task") and hasattr(res, "dependencies"):
        return res.task, list(res.dependencies)
    task, deps = res
    return task, list(deps)


def _collect_git_continuity(root: Path) -> str:
    """Safely inspect working tree status and last 3 commits for continuity context."""
    try:
        probe = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
        if probe.returncode != 0 or probe.stdout.strip().lower() != "true":
            return ""

        status_res = subprocess.run(
            ["git", "-C", str(root), "status", "--short"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
        log_res = subprocess.run(
            ["git", "-C", str(root), "log", "-n3", "--oneline"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""

    parts: list[str] = []
    if status_res.returncode == 0:
        status_lines = [
            line.rstrip() for line in status_res.stdout.splitlines() if line.strip()
        ]
        if status_lines:
            shown = status_lines[:_MAX_GIT_STATUS_LINES]
            extra = len(status_lines) - len(shown)
            block = "\n".join(shown)
            if extra > 0:
                block += f"\n... (+{extra} more changed files)"
            parts.append(f"Git status:\n{block}")
        else:
            parts.append("Git status: clean working tree")
    if log_res.returncode == 0:
        log_lines = [
            line.rstrip() for line in log_res.stdout.splitlines() if line.strip()
        ]
        if log_lines:
            parts.append("Recent commits:\n" + "\n".join(log_lines))
    return "\n".join(parts)


# ----------------------------------------------------------------------
# Section Formatting Helpers
# ----------------------------------------------------------------------


def _format_card_line(card: KnowledgeCard) -> str:
    flags: list[str] = []
    if card.immutable:
        flags.append("IMMUTABLE")
    if card.always_include:
        flags.append("ALWAYS")
    flag_str = f" [{' | '.join(flags)}]" if flags else ""
    anchor_str = ""
    if card.anchors:
        formatted_anchors = ", ".join(
            f"{a.file_path}:{a.symbol}" if a.symbol else a.file_path
            for a in card.anchors
        )
        anchor_str = f" (anchors: {formatted_anchors})"
    why_label = "Why rejected" if card.kind is CardKind.PITFALL else "Why"
    lines = [f"- [{card.id}]{flag_str} {card.title}: {card.statement}"]
    if card.rationale.strip():
        lines.append(f"  {why_label}: {card.rationale.strip()}{anchor_str}")
    elif anchor_str:
        lines.append(f"  Scope:{anchor_str}")
    return "\n".join(lines)


def _format_project_overview(project: ProjectMeta) -> str:
    lines = [f"<project_overview>\nProject: {project.name}"]
    if project.vision.strip():
        vision = truncate_to_tokens(project.vision.strip(), 55, marker="...")
        lines.append(f"Vision: {vision}")
    if project.non_goals:
        non_goals = "; ".join(project.non_goals[:5])
        lines.append(f"Non-goals: {truncate_to_tokens(non_goals, 45, marker='...')}")
    if project.tech_stack_notes.strip():
        notes = truncate_to_tokens(project.tech_stack_notes.strip(), 45, marker="...")
        lines.append(f"Stack notes: {notes}")
    lines.append("</project_overview>")
    return "\n".join(lines)


def _format_active_task_section(
    request: CompileRequest,
    target_task: TaskNode | None,
    dep_tasks: list[TaskNode],
) -> str:
    lines = ["<active_task>"]
    if target_task is not None:
        lines.append(
            f"Task: [{target_task.id}] {target_task.title} "
            f"(status={target_task.status.value}, priority={target_task.priority})"
        )
        if target_task.description.strip():
            lines.append(f"Description: {target_task.description.strip()}")
        if target_task.acceptance_criteria:
            lines.append("Acceptance criteria:")
            for criterion in target_task.acceptance_criteria:
                lines.append(f"  - {criterion}")
        if target_task.next_step_hint.strip():
            lines.append(f"Next step hint: {target_task.next_step_hint.strip()}")
        if target_task.anchors:
            anchors_txt = ", ".join(
                f"{a.file_path}:{a.symbol}" if a.symbol else a.file_path
                for a in target_task.anchors
            )
            lines.append(f"Target anchors: {anchors_txt}")
        if dep_tasks:
            deps_summary = "; ".join(
                f"{d.id}({d.status.value}): {d.title}" for d in dep_tasks
            )
            lines.append(f"Prerequisites: {deps_summary}")
    if request.query and request.query.strip():
        lines.append(f"Query: {request.query.strip()}")
    if request.extra_instruction and request.extra_instruction.strip():
        lines.append(f"Extra instruction: {request.extra_instruction.strip()}")
    if target_task is None and not (request.query and request.query.strip()):
        lines.append("Mode: General repository briefing (no specific task selected).")
    lines.append("</active_task>")
    return "\n".join(lines)


def _format_tier1_file_block(root: Path, record: FileRecord) -> str:
    abs_path = root / record.rel_path
    try:
        source = abs_path.read_text(encoding="utf-8")
    except OSError:
        return record.skeleton
    numbered_lines = [
        f"{idx:>4}| {line}"
        for idx, line in enumerate(source.splitlines(), start=1)
    ]
    body = "\n".join(numbered_lines) if numbered_lines else "(empty file)"
    fence_lang = record.language if record.language != "text" else ""
    return (
        f"### {record.rel_path} (Tier 1 full, {record.token_count} tok)\n"
        f"```{fence_lang}\n{body}\n```"
    )


def _format_tier2_file_block(record: FileRecord, reason: str) -> str:
    fence_lang = record.language if record.language != "text" else ""
    return (
        f"### {record.rel_path} (Tier 2 skeleton via {reason}, {record.skeleton_tokens} tok)\n"
        f"```{fence_lang}\n{record.skeleton}\n```"
    )


def _format_tier3_tree(tier3_paths: list[str]) -> str:
    grouped: dict[str, list[str]] = {}
    for rel_path in sorted(tier3_paths):
        if "/" in rel_path:
            folder, name = rel_path.rsplit("/", 1)
            folder_label = f"{folder}/"
        else:
            folder_label = "(root)"
            name = rel_path
        grouped.setdefault(folder_label, []).append(name)
    lines = ["<project_tree_tier3>", "Remaining project files (paths only):"]
    for folder_label, names in sorted(grouped.items()):
        lines.append(f"- {folder_label}: {', '.join(names)}")
    lines.append("</project_tree_tier3>")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# ContextCompiler
# ----------------------------------------------------------------------


class ContextCompiler:
    """Compiles a high-signal, token-bounded prompt capsule for a task or query."""

    def __init__(
        self,
        store: ContextStore | Path | str,
        indexer: RepoIndexer | None = None,
    ) -> None:
        self._store: ContextStore = (
            store if isinstance(store, ContextStore) else ContextStore(store)
        )
        self._indexer: RepoIndexer = (
            indexer
            if indexer is not None
            else RepoIndexer(self._store.root, store=self._store)
        )
        self._scorer = BM25Scorer()

    @property
    def store(self) -> ContextStore:
        return self._store

    @property
    def indexer(self) -> RepoIndexer:
        return self._indexer

    def compile(
        self,
        request: CompileRequest,
        *,
        index: RepoIndex | None = None,
    ) -> CompileResult:
        """Build a prioritized, Lost-in-the-Middle resistant capsule within token budget."""
        repo_index = index if index is not None else self._indexer.build_or_update()
        project = _load_project_compat(self._store)
        warnings: list[str] = []
        dropped: list[str] = []

        # --------------------------------------------------------------
        # 1. Resolve Task & Build Search Queries
        # --------------------------------------------------------------
        target_task: TaskNode | None = None
        dep_tasks: list[TaskNode] = []
        if request.task_id is not None:
            target_task, dep_tasks = _get_task_with_deps_compat(
                self._store, request.task_id
            )

        file_query_parts: list[str] = []
        card_query_parts: list[str] = []
        if target_task is not None:
            semantic_parts = [
                target_task.title,
                target_task.description,
                target_task.next_step_hint,
                " ".join(target_task.acceptance_criteria),
            ]
            file_query_parts.extend(semantic_parts)
            file_query_parts.append(
                " ".join(
                    f"{a.file_path} {a.symbol or ''}" for a in target_task.anchors
                )
            )
            card_query_parts.extend(semantic_parts)
            card_query_parts.append(
                " ".join(a.symbol for a in target_task.anchors if a.symbol)
            )
        if request.query:
            file_query_parts.append(request.query)
            card_query_parts.append(request.query)
        if request.extra_instruction:
            file_query_parts.append(request.extra_instruction)
            card_query_parts.append(request.extra_instruction)
        if request.explicit_files:
            file_query_parts.append(" ".join(request.explicit_files))

        file_search_query = "\n".join(
            part for part in file_query_parts if part and part.strip()
        )
        card_search_query = "\n".join(
            part for part in card_query_parts if part and part.strip()
        )

        # --------------------------------------------------------------
        # 2. Tier-1 Target Files Selection
        # --------------------------------------------------------------
        tier1_files: list[str] = []
        tier1_seen: set[str] = set()

        def add_tier1_candidate(raw_path: str, source_label: str) -> None:
            norm = _normalize_rel_path(raw_path)
            if not norm or norm in tier1_seen:
                return
            if norm not in repo_index.files:
                warnings.append(
                    f"Target file '{norm}' from {source_label} not found in repository index."
                )
                return
            tier1_seen.add(norm)
            tier1_files.append(norm)

        for path in request.explicit_files:
            add_tier1_candidate(path, "explicit_files")
        if target_task is not None:
            for anchor in target_task.anchors:
                add_tier1_candidate(anchor.file_path, f"{target_task.id}.anchors")

        # Supplement with BM25 top-K if no explicit/anchored target files resolved.
        if not tier1_files and file_search_query and repo_index.files:
            ranked_files = self._scorer.rank_files(
                file_search_query,
                repo_index.files,
                top_k=_MAX_BM25_TIER1_FALLBACK,
            )
            for rel_path, score in ranked_files:
                if score > 0.0:
                    add_tier1_candidate(rel_path, "BM25")

        # --------------------------------------------------------------
        # 3. Tier-2 Dependency Neighbor Files (AST skeletons only, never duplicate Tier-1)
        # --------------------------------------------------------------
        tier2_candidates: dict[str, tuple[float, str]] = {}
        if tier1_files:
            if self._indexer._index is None:  # noqa: SLF001
                self._indexer._index = repo_index  # noqa: SLF001
            for rel_path, score, reason in self._indexer.get_neighbors(
                tier1_files, hops=2
            ):
                if rel_path not in tier1_seen and rel_path in repo_index.files:
                    tier2_candidates[rel_path] = (score, reason)

        for dep_task in dep_tasks:
            for anchor in dep_task.anchors:
                norm = _normalize_rel_path(anchor.file_path)
                if norm in repo_index.files and norm not in tier1_seen:
                    prev = tier2_candidates.get(norm)
                    if prev is None or prev[0] < 0.9:
                        tier2_candidates[norm] = (0.9, f"task_dep:{dep_task.id}")

        tier2_ordered: list[tuple[str, float, str]] = sorted(
            ((path, sc, rsn) for path, (sc, rsn) in tier2_candidates.items()),
            key=lambda item: (-item[1], item[0]),
        )[:_MAX_TIER2_NEIGHBORS]
        tier2_set = {path for path, _, _ in tier2_ordered}

        # --------------------------------------------------------------
        # 4. Tier-3 Project Tree (remaining files)
        # --------------------------------------------------------------
        tier3_all = sorted(
            path
            for path in repo_index.files
            if path not in tier1_seen and path not in tier2_set
        )

        # --------------------------------------------------------------
        # 5. Knowledge Cards Filtering (STRICTLY ACTIVE ONLY!)
        # --------------------------------------------------------------
        raw_active = self._store.get_active_cards()
        active_cards = [c for c in raw_active if c.status is CardStatus.ACTIVE]

        scoped_files = tier1_seen | tier2_set
        related_card_ids = (
            set(target_task.related_cards) if target_task is not None else set()
        )
        card_bm25_scores: dict[str, float] = {}
        if card_search_query and active_cards:
            for card, score in self._scorer.rank_cards(card_search_query, active_cards):
                card_bm25_scores[card.id] = score

        selected_cards: list[tuple[KnowledgeCard, tuple[int, float, str]]] = []
        for card in active_cards:
            # Layer A: Always-On
            if card.always_include:
                selected_cards.append((card, (3, 100.0, card.id)))
                continue
            # Layer B: Explicitly linked, anchor-intersecting, or BM25-relevant
            if card.id in related_card_ids:
                selected_cards.append((card, (2, 50.0, card.id)))
                continue
            card_anchor_paths = {_normalize_rel_path(a.file_path) for a in card.anchors}
            if card_anchor_paths & scoped_files:
                selected_cards.append((card, (2, 40.0, card.id)))
                continue
            bm25_score = card_bm25_scores.get(card.id, 0.0)
            if bm25_score > 0.0:
                selected_cards.append((card, (1, bm25_score, card.id)))

        selected_cards.sort(key=lambda item: (-item[1][0], -item[1][1], item[1][2]))
        invariants = [
            card for card, _ in selected_cards if card.kind is CardKind.INVARIANT
        ]
        adrs = [card for card, _ in selected_cards if card.kind is CardKind.ADR]
        pitfalls = [
            card for card, _ in selected_cards if card.kind is CardKind.PITFALL
        ]

        # --------------------------------------------------------------
        # 6. Recent Sessions & Git Continuity
        # --------------------------------------------------------------
        recent_sessions = _list_sessions_compat(self._store, limit=3)
        git_continuity = _collect_git_continuity(self._store.root)

        # --------------------------------------------------------------
        # 7. Honest Full-Repo Baseline Calculation
        # --------------------------------------------------------------
        all_files_tokens = sum(rec.token_count for rec in repo_index.files.values())
        all_cards_tokens = sum(
            estimate_tokens(_format_card_line(card)) for card in active_cards
        )
        full_repo_tokens = all_files_tokens + all_cards_tokens

        # --------------------------------------------------------------
        # 8. Priority Budget Packer (Lost-in-the-Middle Layout)
        # --------------------------------------------------------------
        budget = request.token_budget
        # Small structural reserve for section "\n\n" joins.
        total_budget = max(0, budget - 16)
        remaining = total_budget

        # --- Priority 1: Project Overview + Active Task + Invariants ---
        overview_text = _format_project_overview(project)
        compact_overview = (
            f"<project_overview>\nProject: {project.name}\n</project_overview>"
        )
        task_text = _format_active_task_section(request, target_task, dep_tasks)
        included_invariants: list[KnowledgeCard] = list(invariants)
        invariants_text = ""
        if included_invariants:
            inv_lines = [
                "<project_invariants>",
                "INVARIANTS (MUST OBEY):",
                *(_format_card_line(c) for c in included_invariants),
                "</project_invariants>",
            ]
            invariants_text = "\n".join(inv_lines)

        core_tokens = (
            estimate_tokens(overview_text)
            + estimate_tokens(task_text)
            + (estimate_tokens(invariants_text) if invariants_text else 0)
        )
        if core_tokens > remaining:
            # Keep invariants untouched; compact overview & truncate task if extreme budget.
            overview_text = compact_overview
            avail_for_task = max(
                40,
                remaining
                - estimate_tokens(overview_text)
                - (estimate_tokens(invariants_text) if invariants_text else 0),
            )
            if estimate_tokens(task_text) > avail_for_task:
                task_text = truncate_to_tokens(task_text, avail_for_task)
                warnings.append(
                    "Active task description truncated due to ultra-small token budget."
                )
            core_tokens = (
                estimate_tokens(overview_text)
                + estimate_tokens(task_text)
                + (estimate_tokens(invariants_text) if invariants_text else 0)
            )
        remaining = max(0, remaining - core_tokens)

        # --- Reserve <ctx_protocol> tail before Pitfalls/ADRs/Code so tail survives! ---
        protocol_text = ""
        protocol_included = False
        if request.include_protocol:
            proto_cost = estimate_tokens(_CTX_PROTOCOL_TEXT)
            if proto_cost > remaining and overview_text != compact_overview:
                saved_by_compact = estimate_tokens(overview_text) - estimate_tokens(
                    compact_overview
                )
                if saved_by_compact > 0:
                    overview_text = compact_overview
                    remaining += saved_by_compact
            if proto_cost <= remaining:
                protocol_text = _CTX_PROTOCOL_TEXT
                protocol_included = True
                remaining = max(0, remaining - proto_cost)
            else:
                dropped.append("ctx_protocol")
                warnings.append(
                    "Insufficient token budget to retain <ctx_protocol> tail."
                )

        # --- Priority 2: Rejected & Pitfalls ---
        included_pitfalls: list[KnowledgeCard] = []
        pitfalls_text = ""
        if pitfalls:
            shell_tokens = estimate_tokens(
                "<rejected_approaches>\nREJECTED & PITFALLS (DO NOT SUGGEST / DO NOT REPEAT):\n</rejected_approaches>"
            )
            if remaining <= shell_tokens and overview_text != compact_overview:
                saved_by_compact = estimate_tokens(overview_text) - estimate_tokens(
                    compact_overview
                )
                if saved_by_compact > 0:
                    overview_text = compact_overview
                    remaining += saved_by_compact
            if remaining > shell_tokens:
                kept_lines: list[str] = []
                used = shell_tokens
                for pit in pitfalls:
                    line = _format_card_line(pit)
                    cost = estimate_tokens(line) + 1
                    if used + cost <= remaining:
                        kept_lines.append(line)
                        included_pitfalls.append(pit)
                        used += cost
                    else:
                        dropped.append(f"card:{pit.id}")
                if kept_lines:
                    pitfalls_text = "\n".join(
                        [
                            "<rejected_approaches>",
                            "REJECTED & PITFALLS (DO NOT SUGGEST / DO NOT REPEAT):",
                            *kept_lines,
                            "</rejected_approaches>",
                        ]
                    )
                    remaining = max(0, remaining - estimate_tokens(pitfalls_text))
            else:
                for pit in pitfalls:
                    dropped.append(f"card:{pit.id}")

        # --- Priority 3: Relevant ADRs ---
        included_adrs: list[KnowledgeCard] = []
        adrs_text = ""
        if adrs:
            adr_shell_tokens = estimate_tokens(
                "<active_decisions>\nACTIVE DECISIONS (ADR):\n</active_decisions>"
            )
            if remaining > adr_shell_tokens:
                kept_lines = []
                used = adr_shell_tokens
                for adr in adrs:
                    line = _format_card_line(adr)
                    cost = estimate_tokens(line) + 1
                    if used + cost <= remaining:
                        kept_lines.append(line)
                        included_adrs.append(adr)
                        used += cost
                    else:
                        dropped.append(f"card:{adr.id}")
                if kept_lines:
                    adrs_text = "\n".join(
                        [
                            "<active_decisions>",
                            "ACTIVE DECISIONS (ADR):",
                            *kept_lines,
                            "</active_decisions>",
                        ]
                    )
                    remaining = max(0, remaining - estimate_tokens(adrs_text))
            else:
                for adr in adrs:
                    dropped.append(f"card:{adr.id}")

        # --- Priority 4: Recent Sessions & Git Continuity ---
        continuity_text = ""
        sessions_included = 0
        git_included = False
        if recent_sessions or git_continuity:
            cont_parts: list[str] = []
            if recent_sessions:
                sess_lines = ["Recent sessions:"]
                for ses in recent_sessions:
                    task_ref = f" [{ses.task_id}]" if ses.task_id else ""
                    anchor_ref = (
                        f" | next: {ses.next_anchor}" if ses.next_anchor else ""
                    )
                    sess_lines.append(
                        f"- {ses.session_id}{task_ref}: {ses.summary}{anchor_ref}"
                    )
                cont_parts.append("\n".join(sess_lines))
            if git_continuity:
                cont_parts.append(git_continuity)

            candidate_cont = "\n".join(
                ["<continuity_context>", *cont_parts, "</continuity_context>"]
            )
            cont_cost = estimate_tokens(candidate_cont)
            if cont_cost <= remaining:
                continuity_text = candidate_cont
                sessions_included = len(recent_sessions)
                git_included = bool(git_continuity)
                remaining = max(0, remaining - cont_cost)
            elif recent_sessions:
                sess_only = "\n".join(
                    [
                        "<continuity_context>",
                        cont_parts[0],
                        "</continuity_context>",
                    ]
                )
                sess_cost = estimate_tokens(sess_only)
                if sess_cost <= remaining:
                    continuity_text = sess_only
                    sessions_included = len(recent_sessions)
                    remaining = max(0, remaining - sess_cost)
                    if git_continuity:
                        dropped.append("continuity:git")
                else:
                    dropped.append("continuity")
            else:
                dropped.append("continuity:git")

        # --- Priority 5: Tier-1 Target Files (Full Code, with graceful degradation) ---
        tier1_blocks: list[str] = []
        included_tier1: list[str] = []
        degraded_tier2_blocks: list[str] = []
        degraded_tier2_files: list[str] = []
        t2_shell_tokens = estimate_tokens(
            "<code_tier2_signatures>\n</code_tier2_signatures>"
        )
        t2_shell_reserved = False

        if tier1_files:
            t1_shell_tokens = estimate_tokens(
                "<code_tier1_full>\n</code_tier1_full>"
            )
            if remaining > t1_shell_tokens:
                used_t1 = t1_shell_tokens
                for rel_path in tier1_files:
                    rec = repo_index.files[rel_path]
                    full_block = _format_tier1_file_block(self._store.root, rec)
                    cost = estimate_tokens(full_block) + 2
                    avail = remaining - used_t1
                    if cost <= avail:
                        tier1_blocks.append(full_block)
                        included_tier1.append(rel_path)
                        used_t1 += cost
                    else:
                        full_tok = estimate_tokens(full_block)
                        if avail >= max(220, int(full_tok * 0.45)):
                            truncated_block = truncate_to_tokens(
                                full_block,
                                avail - 2,
                                marker="\n...[TRUNCATED DUE TO TOKEN BUDGET]...\n```",
                            )
                            tier1_blocks.append(truncated_block)
                            included_tier1.append(rel_path)
                            used_t1 += estimate_tokens(truncated_block) + 2
                            warnings.append(
                                f"Tier-1 file '{rel_path}' truncated to fit token budget."
                            )
                        else:
                            skel_block = _format_tier2_file_block(
                                rec, "budget_degraded_tier1"
                            )
                            extra_shell = 0 if t2_shell_reserved else t2_shell_tokens
                            skel_cost = estimate_tokens(skel_block) + 2 + extra_shell
                            if skel_cost <= avail:
                                degraded_tier2_blocks.append(skel_block)
                                degraded_tier2_files.append(rel_path)
                                used_t1 += skel_cost
                                t2_shell_reserved = True
                                warnings.append(
                                    f"Tier-1 file '{rel_path}' degraded to Tier-2 skeleton due to tight token budget."
                                )
                            else:
                                dropped.append(f"tier1:{rel_path}")
                                warnings.append(
                                    f"Tier-1 file '{rel_path}' dropped due to insufficient token budget."
                                )
                if not tier1_blocks:
                    used_t1 -= t1_shell_tokens
                remaining = max(0, remaining - used_t1)
            else:
                for rel_path in tier1_files:
                    dropped.append(f"tier1:{rel_path}")
                    warnings.append(
                        f"Tier-1 file '{rel_path}' dropped due to insufficient token budget."
                    )

        tier1_text = (
            "\n".join(["<code_tier1_full>", *tier1_blocks, "</code_tier1_full>"])
            if tier1_blocks
            else ""
        )

        # --- Priority 6: Tier-2 Dependency Neighbor Skeletons ---
        tier2_blocks: list[str] = list(degraded_tier2_blocks)
        included_tier2: list[str] = list(degraded_tier2_files)
        if tier2_ordered or tier2_blocks:
            used_t2 = 0 if t2_shell_reserved else t2_shell_tokens
            if remaining > used_t2:
                for rel_path, _score, reason in tier2_ordered:
                    if rel_path in included_tier1 or rel_path in included_tier2:
                        continue
                    rec = repo_index.files[rel_path]
                    block = _format_tier2_file_block(rec, reason)
                    cost = estimate_tokens(block) + 2
                    if used_t2 + cost <= remaining:
                        tier2_blocks.append(block)
                        included_tier2.append(rel_path)
                        used_t2 += cost
                    else:
                        dropped.append(f"tier2:{rel_path}")
                if tier2_blocks:
                    remaining = max(0, remaining - used_t2)
            else:
                for rel_path, _, _ in tier2_ordered:
                    dropped.append(f"tier2:{rel_path}")

        tier2_text = (
            "\n".join(
                ["<code_tier2_signatures>", *tier2_blocks, "</code_tier2_signatures>"]
            )
            if tier2_blocks
            else ""
        )

        # --- Priority 7: Tier-3 Project Tree ---
        tier3_text = ""
        tier3_included_count = 0
        if tier3_all:
            candidate_tree = _format_tier3_tree(tier3_all)
            tree_cost = estimate_tokens(candidate_tree)
            if tree_cost <= remaining:
                tier3_text = candidate_tree
                tier3_included_count = len(tier3_all)
                remaining = max(0, remaining - tree_cost)
            elif remaining >= 45:
                tier3_text = truncate_to_tokens(
                    candidate_tree,
                    remaining,
                    marker="\n...[TREE TRUNCATED]...\n</project_tree_tier3>",
                )
                tier3_included_count = len(tier3_all)
                warnings.append("Tier-3 project tree truncated to fit remaining budget.")
            else:
                dropped.append("tier3_project_tree")

        # --------------------------------------------------------------
        # 9. Assemble in Lost-in-the-Middle Layout & Enforce Hard Ceiling
        # --------------------------------------------------------------
        sections_map: dict[str, str] = {
            "overview": overview_text,
            "invariants": invariants_text,
            "pitfalls": pitfalls_text,
            "task": task_text,
            "adrs": adrs_text,
            "continuity": continuity_text,
            "tier1": tier1_text,
            "tier2": tier2_text,
            "tier3": tier3_text,
            "protocol": protocol_text,
        }
        layout_order = [
            "overview",
            "invariants",
            "pitfalls",
            "task",
            "adrs",
            "continuity",
            "tier1",
            "tier2",
            "tier3",
            "protocol",
        ]
        trim_order = [
            "tier3",
            "tier2",
            "tier1",
            "continuity",
            "adrs",
            "pitfalls",
            "protocol",
        ]

        def assemble() -> str:
            active_parts = [sections_map[key] for key in layout_order if sections_map[key]]
            return "\n\n".join(active_parts)

        capsule_text = assemble()
        capsule_tokens = estimate_tokens(capsule_text)

        # Hard guarantee: trim lowest-priority optional sections if join boundaries pushed over budget.
        for key in trim_order:
            if capsule_tokens <= budget:
                break
            sec = sections_map[key]
            if not sec:
                continue
            excess = capsule_tokens - budget
            sec_tokens = estimate_tokens(sec)
            if key == "tier3" and sec_tokens - excess >= 40:
                sections_map[key] = truncate_to_tokens(
                    sec,
                    sec_tokens - excess,
                    marker="\n...[TREE TRUNCATED]...\n</project_tree_tier3>",
                )
            elif key == "tier3":
                sections_map[key] = ""
                tier3_included_count = 0
                dropped.append("tier3_project_tree")
            elif key == "tier2":
                sections_map[key] = ""
                for path in included_tier2:
                    dropped.append(f"tier2:{path}")
                included_tier2 = []
            elif key == "tier1":
                sections_map[key] = ""
                for path in included_tier1:
                    dropped.append(f"tier1:{path}")
                included_tier1 = []
            elif key == "continuity":
                sections_map[key] = ""
                sessions_included = 0
                git_included = False
                dropped.append("continuity")
            elif key == "adrs":
                sections_map[key] = ""
                for card in included_adrs:
                    dropped.append(f"card:{card.id}")
                included_adrs = []
            elif key == "pitfalls":
                sections_map[key] = ""
                for card in included_pitfalls:
                    dropped.append(f"card:{card.id}")
                included_pitfalls = []
            elif key == "protocol":
                sections_map[key] = ""
                protocol_included = False
                dropped.append("ctx_protocol")
            capsule_text = assemble()
            capsule_tokens = estimate_tokens(capsule_text)

        if capsule_tokens > budget:
            capsule_text = truncate_to_tokens(capsule_text, budget)
            capsule_tokens = estimate_tokens(capsule_text)
            warnings.append("Core capsule exceeded tight budget and was hard-truncated.")

        if full_repo_tokens > 0:
            savings_percent = round(
                (1.0 - (capsule_tokens / full_repo_tokens)) * 100.0, 2
            )
        else:
            savings_percent = 0.0

        stats: dict[str, Any] = {
            "token_budget": budget,
            "capsule_tokens": capsule_tokens,
            "full_repo_tokens": full_repo_tokens,
            "tier1_count": len(included_tier1),
            "tier1_files": included_tier1,
            "tier2_count": len(included_tier2),
            "tier2_files": included_tier2,
            "tier3_count": tier3_included_count,
            "cards_invariants": len(included_invariants),
            "cards_adrs": len(included_adrs),
            "cards_pitfalls": len(included_pitfalls),
            "cards_total_active": len(active_cards),
            "sessions_included": sessions_included,
            "git_included": git_included,
            "protocol_included": protocol_included,
            "dropped": list(dict.fromkeys(dropped)),
            "warnings": warnings,
        }
        return CompileResult(
            capsule_text=capsule_text,
            capsule_tokens=capsule_tokens,
            full_repo_tokens=full_repo_tokens,
            savings_percent=savings_percent,
            stats=stats,
        )
    