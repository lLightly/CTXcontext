"""Experimental A/B benchmarking suite: Full Dump vs Long Chat vs ctxforge Capsule."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence

from .compiler import CompileRequest, ContextCompiler
from .indexer import RepoIndex, RepoIndexer
from .llm_adapter import LLMAdapter
from .models import (
    Anchor,
    CardKind,
    CardStatus,
    KnowledgeCard,
    ProjectMeta,
    SessionLogEntry,
    TaskNode,
    TaskStatus,
    utc_now_iso,
)
from .storage import ContextStore, CtxForgeError
from .tokens import estimate_tokens

STRATEGIES: tuple[str, ...] = (
    "baseline_full_dump",
    "baseline_long_chat",
    "ctxforge_compiled",
)

_BENCH_SCHEMA_VERSION = 1
_MAX_LONG_CHAT_FILES = 24
_MAX_LONG_CHAT_PASTED_TOKENS = 18000

_CODE_FENCE_RE = re.compile(r"```(?:[A-Za-z0-9_+-]+)?\n([\s\S]*?)```")
_PATH_REF_RE = re.compile(
    r"(?<![A-Za-z0-9_./:-])([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+\."
    r"(?:py|pyi|ts|tsx|js|jsx|go|rs|java|md|rst|json|toml|yaml|yml))\b"
)
_PY_FROM_IMPORT_RE = re.compile(
    r"^[ \t]*from[ \t]+([A-Za-z0-9_.]+)[ \t]+import[ \t]+(?:\(([^)]+)\)|([A-Za-z0-9_, \t*]+))",
    re.MULTILINE,
)

_NEGATION_CUES: tuple[str, ...] = (
    "not ",
    "never ",
    "avoid ",
    "don't ",
    "do not ",
    "instead of ",
    "rather than ",
    "without ",
    "forbid",
    "prohibit",
    "deprecated",
    "rejected",
    "не ",
    "нельзя ",
    "запрещ",
    "отказ",
    "вместо ",
    "без ",
)

_RULE_EXTRACT_STOPWORDS: frozenset[str] = frozenset(
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
        "to",
        "of",
        "in",
        "on",
        "by",
        "is",
        "are",
        "be",
        "must",
        "should",
        "only",
        "use",
        "using",
        "never",
        "avoid",
        "not",
        "all",
        "any",
        "store",
        "code",
        "files",
        "file",
        "module",
        "modules",
        "core",
        "data",
        "system",
        "approach",
        "rejected",
        "deprecated",
        "old",
        "new",
        "и",
        "в",
        "во",
        "на",
        "по",
        "к",
        "из",
        "от",
        "для",
        "с",
        "при",
        "без",
        "не",
        "запрещено",
        "использовать",
        "только",
        "нужно",
        "должны",
        "хранить",
    }
)


@dataclass(slots=True)
class StrategyScore:
    """Evaluated metrics for one (task_id, strategy) run."""

    task_id: str
    strategy: str
    has_response: bool
    input_tokens: int
    output_tokens: int
    signal_tokens: int
    snr: float
    adherence_score: float
    violations: list[str] = field(default_factory=list)
    relapse_count: int = 0
    relapses: list[str] = field(default_factory=list)
    anchor_accuracy: float = 1.0
    verified_refs: list[str] = field(default_factory=list)
    hallucinated_refs: list[str] = field(default_factory=list)
    exec_passed: bool | None = None
    exec_output: str = ""
    latency_ms: float | None = None
    manual_quality_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "strategy": self.strategy,
            "has_response": self.has_response,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "signal_tokens": self.signal_tokens,
            "snr": self.snr,
            "adherence_score": self.adherence_score,
            "violations": list(self.violations),
            "relapse_count": self.relapse_count,
            "relapses": list(self.relapses),
            "anchor_accuracy": self.anchor_accuracy,
            "verified_refs": list(self.verified_refs),
            "hallucinated_refs": list(self.hallucinated_refs),
            "exec_passed": self.exec_passed,
            "latency_ms": self.latency_ms,
            "manual_quality_score": self.manual_quality_score,
        }


@dataclass(slots=True)
class BenchmarkReport:
    """Aggregated experimental comparison across all 3 context strategies."""

    bench_dir: Path
    scores: list[StrategyScore]
    strategy_aggregates: dict[str, dict[str, Any]]
    report_markdown: str
    report_path: Path

    def format_console_summary(self) -> str:
        """Return a concise terminal summary of benchmark metrics."""
        lines = [
            "=== ctxforge Benchmark Results ===",
            f"{'Strategy':<20} {'InTok':>8} {'OutTok':>8} {'SNR':>7} {'Adh':>7} {'Relapse':>8} {'AnchorAcc':>10}",
            "-" * 74,
        ]
        for strat in STRATEGIES:
            agg = self.strategy_aggregates.get(strat)
            if not agg:
                continue
            lines.append(
                f"{strat:<20} "
                f"{agg['avg_input_tokens']:>8.0f} "
                f"{agg['avg_output_tokens']:>8.0f} "
                f"{agg['avg_snr'] * 100:>6.1f}% "
                f"{agg['avg_adherence'] * 100:>6.1f}% "
                f"{agg['total_relapses']:>8d} "
                f"{agg['avg_anchor_accuracy'] * 100:>9.1f}%"
            )
        lines.append(f"Full report written -> {self.report_path}")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Rule Extraction & Contract Evaluation Helpers
# ----------------------------------------------------------------------


def _extract_backtick_terms(text: str) -> list[str]:
    return [
        m.group(1).strip().lower()
        for m in re.finditer(r"`([^`\n]{2,40})`", text)
        if m.group(1).strip()
    ]


def _extract_distinctive_tokens(text: str) -> list[str]:
    cleaned = re.sub(r"[^A-Za-z0-9_\u0400-\u04FF]+", " ", text.lower())
    tokens: list[str] = []
    for tok in cleaned.split():
        if len(tok) < 3 or tok in _RULE_EXTRACT_STOPWORDS or tok.isdigit():
            continue
        tokens.append(tok)
    return list(dict.fromkeys(tokens))


def _derive_forbidden_patterns(
    card: KnowledgeCard,
    *,
    active_cards_by_id: Mapping[str, KnowledgeCard],
) -> list[str]:
    """Derive regex/term patterns representing forbidden or rejected ideas in card."""
    combined = f"{card.title}. {card.statement}"
    patterns: list[str] = []

    if card.kind is CardKind.INVARIANT:
        lower = combined.lower()
        # Capture explicit "never/not/avoid ..."
        for m in re.finditer(
            r"(?:never\s+use|do\s+not\s+use|don't\s+use|avoid|forbidden|no\s+external|не\s+использовать|запрещено)\s+([^.;\n]+)",
            lower,
        ):
            clause = m.group(1)
            for term in _extract_distinctive_tokens(clause):
                patterns.append(term)
        # Stdlib-only invariant commonly forbids popular third-party packages
        if "stdlib" in lower or "standard library" in lower or "third-party" in lower:
            patterns.extend(["pydantic", "requests", "fastapi", "flask", "sqlalchemy"])
        if "integer" in lower and (
            "cent" in lower or "money" in lower or "currency" in lower
        ):
            patterns.append("float")
        for bt in _extract_backtick_terms(combined):
            if bt not in _RULE_EXTRACT_STOPWORDS:
                patterns.append(bt)

    elif card.kind is CardKind.PITFALL:
        for bt in _extract_backtick_terms(combined):
            if bt not in _RULE_EXTRACT_STOPWORDS:
                patterns.append(bt)
        for m in re.finditer(
            r"(?:never\s+use|do\s+not\s+use|do\s+not|never|avoid|не\s+использовать|не\s+хранить)\s+([^.;\n]+)",
            combined.lower(),
        ):
            for term in _extract_distinctive_tokens(m.group(1)):
                patterns.append(term)
        if not patterns:
            patterns.extend(_extract_distinctive_tokens(card.title)[:4])

    elif card.status is CardStatus.SUPERSEDED:
        # Tokens present in the old superseded decision but absent from its active replacement.
        old_tokens = set(_extract_distinctive_tokens(combined))
        rep_tokens: set[str] = set()
        if card.superseded_by and card.superseded_by in active_cards_by_id:
            rep = active_cards_by_id[card.superseded_by]
            rep_tokens = set(_extract_distinctive_tokens(f"{rep.title}. {rep.statement}"))
        diff_tokens = sorted(old_tokens - rep_tokens)
        patterns.extend(diff_tokens[:6])

    # Normalize & deduplicate preserving order
    unique: list[str] = []
    for pat in patterns:
        norm = pat.strip().lower()
        if len(norm) >= 3 and norm not in unique:
            unique.append(norm)
    return unique


def _strip_code_comments(code_text: str) -> str:
    lines: list[str] = []
    for line in code_text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("#", "//")):
            continue
        lines.append(line)
    return "\n".join(lines)


def _split_code_and_prose(response_text: str) -> tuple[str, str]:
    code_blocks = [
        _strip_code_comments(m.group(1))
        for m in _CODE_FENCE_RE.finditer(response_text)
    ]
    prose = _CODE_FENCE_RE.sub(" ", response_text)
    return "\n".join(code_blocks), prose


def _matches_affirmative_prose(prose: str, term: str) -> bool:
    """Return True if `term` appears in `prose` without preceding negation cues."""
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", re.IGNORECASE)
    for match in pattern.finditer(prose):
        start = match.start()
        window_start = max(0, start - 65)
        prefix = prose[window_start:start]
        # Restrict negation check to the current sentence/clause.
        for sep in (".", "\n", ";", "!"):
            sep_idx = prefix.rfind(sep)
            if sep_idx != -1:
                prefix = prefix[sep_idx + 1 :]
        prefix_lower = prefix.lower()
        if any(cue in prefix_lower for cue in _NEGATION_CUES):
            continue
        return True
    return False


def _matches_code_usage(code_text: str, term: str) -> bool:
    if not code_text.strip():
        return False
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", re.IGNORECASE)
    return pattern.search(code_text) is not None


def _evaluate_anchor_accuracy(
    response_text: str,
    valid_files: set[str],
    symbols_by_file: Mapping[str, set[str]],
) -> tuple[float, list[str], list[str]]:
    """Verify file paths and imported/qualified symbols mentioned in `response_text`."""
    cleaned = re.sub(r"https?://\S+", " ", response_text)
    verified: list[str] = []
    hallucinated: list[str] = []

    for m in _PATH_REF_RE.finditer(cleaned):
        rel = m.group(1).strip()
        while rel.startswith("./"):
            rel = rel[2:]
        if rel == "path/file.py":
            continue
        if rel in valid_files:
            verified.append(rel)
        else:
            hallucinated.append(rel)

    code_text, _ = _split_code_and_prose(cleaned)
    for m in _PY_FROM_IMPORT_RE.finditer(code_text):
        mod_dotted = m.group(1).strip()
        names_raw = m.group(2) if m.group(2) is not None else (m.group(3) or "")
        mod_path = mod_dotted.replace(".", "/") + ".py"
        if mod_path not in valid_files:
            alt = mod_dotted.replace(".", "/") + "/__init__.py"
            mod_path = alt if alt in valid_files else ""
        if not mod_path:
            continue
        known_syms = symbols_by_file.get(mod_path, set())
        if not known_syms:
            continue
        for item in names_raw.split(","):
            sym = item.strip().split(" as ")[0].strip()
            if not sym or sym == "*":
                continue
            ref = f"{mod_path}:{sym}"
            if sym in known_syms:
                verified.append(ref)
            else:
                hallucinated.append(ref)

    verified = list(dict.fromkeys(verified))
    hallucinated = list(dict.fromkeys(hallucinated))
    total = len(verified) + len(hallucinated)
    if total == 0:
        return 1.0, verified, hallucinated
    return round(len(verified) / total, 4), verified, hallucinated


# ----------------------------------------------------------------------
# Prompt Builders for the 3 Experimental Strategies
# ----------------------------------------------------------------------


def _format_task_brief(task: TaskNode) -> str:
    lines = [
        f"Task [{task.id}]: {task.title}",
        f"Description: {task.description}",
    ]
    if task.acceptance_criteria:
        lines.append("Acceptance Criteria:")
        lines.extend(f"- {c}" for c in task.acceptance_criteria)
    if task.anchors:
        anchors_str = ", ".join(
            f"{a.file_path}:{a.symbol}" if a.symbol else a.file_path
            for a in task.anchors
        )
        lines.append(f"Target files: {anchors_str}")
    lines.append(
        "Provide the concrete implementation/code changes and cite exact file paths and symbols."
    )
    return "\n".join(lines)


def _build_full_dump_prompt(
    store: ContextStore,
    index: RepoIndex,
    task: TaskNode,
    signal_tokens: int,
) -> tuple[str, int, int, float]:
    all_cards = store.list_cards()
    all_tasks = store.list_tasks()
    all_sessions = store.list_sessions()
    project = store.get_project()

    parts: list[str] = [
        f"Project: {project.name}\nVision: {project.vision}",
        "=== ALL STORED CARDS (UNFILTERED) ===",
    ]
    for card in all_cards:
        parts.append(
            f"[{card.id}] ({card.kind.value}, {card.status.value}) {card.title}: "
            f"{card.statement} | Rationale: {card.rationale}"
        )
    parts.append("=== ALL TASKS & SESSIONS ===")
    for t in all_tasks:
        parts.append(f"[{t.id}] ({t.status.value}) {t.title}: {t.description}")
    for ses in all_sessions:
        parts.append(f"[{ses.session_id}] {ses.summary} | next: {ses.next_anchor}")

    parts.append("=== FULL REPOSITORY FILES ===")
    for rel_path in sorted(index.files):
        abs_path = store.root / rel_path
        try:
            body = abs_path.read_text(encoding="utf-8")
        except OSError:
            body = index.files[rel_path].skeleton
        parts.append(f"--- FILE: {rel_path} ---\n{body}")

    parts.append("=== CURRENT TASK ===\n" + _format_task_brief(task))
    prompt = "\n\n".join(parts)
    total_tokens = estimate_tokens(prompt)
    snr = round(min(1.0, signal_tokens / total_tokens), 4) if total_tokens > 0 else 0.0
    return prompt, total_tokens, min(signal_tokens, total_tokens), snr


def _build_long_chat_prompt(
    store: ContextStore,
    index: RepoIndex,
    task: TaskNode,
    signal_tokens: int,
) -> tuple[str, int, int, float]:
    """Simulate a noisy multi-turn chat history containing stale/rejected ideas and chatter."""
    all_cards = store.list_cards()
    superseded_cards = [c for c in all_cards if c.status is CardStatus.SUPERSEDED]
    pitfall_cards = [c for c in all_cards if c.kind is CardKind.PITFALL]
    other_tasks = [t for t in store.list_tasks() if t.id != task.id]
    sessions = store.list_sessions()

    turns: list[str] = [
        "=== ACCUMULATED CONVERSATION HISTORY ===",
        "User (Turn 1): Let's brainstorm how to build this project fast without overthinking.",
        (
            "Assistant (Turn 1): Great idea! Let's sketch out several prototypes quickly, "
            "try multiple storage formats, inspect unrelated modules, and iterate conversationally. "
            "We can keep optional helper libraries in mind, draft quick exploratory wrappers, "
            "and refine everything later once all endpoints and background workers are wired up."
        ),
    ]
    for old_card in superseded_cards:
        turns.append(
            f"User (Earlier Turn): Earlier we agreed on {old_card.id}: {old_card.title}.\n"
            f"Assistant (Earlier Turn): Yes, we prototyped: {old_card.statement} "
            f"({old_card.rationale}). Here is the earlier draft structure and helper configuration "
            "we discussed in detail so we can reference that pattern whenever convenient."
        )
    for pit in pitfall_cards:
        turns.append(
            f"User (Earlier Turn): What about the approach from {pit.title}?\n"
            f"Assistant (Earlier Turn): Note from long chat: {pit.statement} ({pit.rationale}). "
            "We kept the exploratory snippet in our notes from earlier troubleshooting turns."
        )
    for prev_task in other_tasks[:6]:
        turns.append(
            f"User (Previous Task {prev_task.id}): Work on {prev_task.title} — {prev_task.description}\n"
            f"Assistant (Previous Task {prev_task.id}): Completed verbose walkthrough of {prev_task.title}. "
            "We reviewed input validation, formatting helpers, logging boilerplate, error paths, "
            "edge-case fixtures, and step-by-step manual verification output across auxiliary modules."
        )
    for ses in sessions:
        turns.append(f"Assistant (Handoff Note {ses.session_id}): {ses.summary}")

    # Include noisy mix of unrelated + relevant files pasted across older turns, bounded safely.
    pasted_paths = sorted(index.files.keys())
    pasted_tokens = 0
    for rel_path in pasted_paths[:_MAX_LONG_CHAT_FILES]:
        abs_path = store.root / rel_path
        try:
            body = abs_path.read_text(encoding="utf-8")
        except OSError:
            body = index.files[rel_path].skeleton
        turn_block = f"User (Pasted earlier {rel_path}):\n```\n{body}\n```"
        block_tok = estimate_tokens(turn_block)
        if pasted_tokens > 0 and pasted_tokens + block_tok > _MAX_LONG_CHAT_PASTED_TOKENS:
            break
        turns.append(turn_block)
        pasted_tokens += block_tok

    turns.append("=== NEW USER TURN ===\n" + _format_task_brief(task))
    prompt = "\n\n".join(turns)
    total_tokens = estimate_tokens(prompt)
    snr = round(min(1.0, signal_tokens / total_tokens), 4) if total_tokens > 0 else 0.0
    return prompt, total_tokens, min(signal_tokens, total_tokens), snr


# ----------------------------------------------------------------------
# Sample Sandbox Project Generator (`ctx bench init-sample`)
# ----------------------------------------------------------------------


def create_sample_project(target_dir: Path | str | None = None) -> Path:
    """Create a multi-file calibration sandbox project ready for A/B benchmarking."""
    if target_dir is None:
        root = Path(tempfile.mkdtemp(prefix="ctxforge-bench-sample-")).resolve()
    else:
        root = Path(target_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {
        "app/__init__.py": "",
        "app/crypto.py": '''"""Stdlib cryptographic token signing utilities."""

import hashlib
import hmac


def sign_payload(payload: str, secret: str) -> str:
    """Sign payload using HMAC-SHA256."""
    return hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_signature(payload: str, signature: str, secret: str) -> bool:
    """Verify HMAC-SHA256 signature in constant time."""
    expected = sign_payload(payload, secret)
    return hmac.compare_digest(expected, signature)
''',
        "app/db.py": '''"""In-memory user repository."""

from dataclasses import dataclass


@dataclass(slots=True)
class UserRecord:
    """Persisted user entity."""

    user_id: str
    email: str
    active: bool = True


class UserRepository:
    """Repository managing user records."""

    def __init__(self) -> None:
        self._users: dict[str, UserRecord] = {
            "alice@example.com": UserRecord(user_id="u-1", email="alice@example.com")
        }

    def get_by_email(self, email: str) -> UserRecord | None:
        """Return user matching normalized email or None."""
        return self._users.get(email.strip().lower())
''',
        "app/auth.py": '''"""Stateless authentication service."""

from app.crypto import sign_payload, verify_signature
from app.db import UserRepository


class AuthService:
    """Issues and validates stateless HMAC-SHA256 tokens."""

    def __init__(self, repo: UserRepository, secret: str) -> None:
        self._repo = repo
        self._secret = secret

    def issue_token(self, email: str) -> str:
        """Issue signed token for an active user."""
        user = self._repo.get_by_email(email)
        if user is None or not user.active:
            raise PermissionError("Invalid or inactive user")
        sig = sign_payload(user.user_id, self._secret)
        return f"{user.user_id}.{sig}"

    def authenticate_token(self, token: str) -> bool:
        """Verify token signature."""
        if "." not in token:
            return False
        user_id, sig = token.split(".", 1)
        return verify_signature(user_id, sig, self._secret)
''',
        "app/billing.py": '''"""Integer-cents invoice calculator."""


class InvoiceCalculator:
    """Computes subscription totals without floating point drift."""

    def calculate_total_cents(self, subtotal_cents: int, tax_bps: int) -> int:
        """Compute total in integer cents given tax in basis points."""
        if subtotal_cents < 0 or tax_bps < 0:
            raise ValueError("Amounts must be non-negative")
        tax_cents = (subtotal_cents * tax_bps) // 10_000
        return subtotal_cents + tax_cents
''',
        "app/analytics.py": '''"""Event aggregation and funnel window metrics."""

from collections import Counter
from dataclasses import dataclass, field


@dataclass(slots=True)
class FunnelSnapshot:
    """Aggregated conversion summary across tracked steps."""

    total_events: int
    unique_names: int
    top_events: list[tuple[str, int]] = field(default_factory=list)


def count_events(events: list[str]) -> dict[str, int]:
    """Count occurrences of each normalized event name."""
    totals: dict[str, int] = {}
    for event in events:
        cleaned = event.strip().lower()
        if not cleaned:
            continue
        totals[cleaned] = totals.get(cleaned, 0) + 1
    return totals


def build_funnel_snapshot(events: list[str], *, top_n: int = 5) -> FunnelSnapshot:
    """Summarize raw event stream into ordered top-N step counts."""
    counts = Counter(e.strip().lower() for e in events if e.strip())
    ranked = counts.most_common(max(1, top_n))
    return FunnelSnapshot(
        total_events=sum(counts.values()),
        unique_names=len(counts),
        top_events=ranked,
    )


def compute_retention_bps(cohort_size: int, retained_count: int) -> int:
    """Compute cohort retention ratio strictly in integer basis points."""
    if cohort_size <= 0 or retained_count < 0:
        return 0
    capped = min(cohort_size, retained_count)
    return (capped * 10_000) // cohort_size
''',
        "app/reports.py": '''"""Deterministic CSV and summary table formatter."""

from typing import Iterable


def _escape_csv_cell(value: str) -> str:
    if any(ch in value for ch in (",", '"', "\n", "\r")):
        escaped = value.replace('"', '""')
        return f'"{escaped}"'
    return value


def render_csv_rows(rows: list[tuple[str, int]]) -> str:
    """Format name/value pairs as RFC-4180 compliant CSV text."""
    lines = ["name,value"]
    for name, value in rows:
        lines.append(f"{_escape_csv_cell(name)},{value}")
    return "\n".join(lines)


def render_grouped_summary(
    sections: Iterable[tuple[str, list[tuple[str, int]]]],
) -> str:
    """Render multi-section metric summary report."""
    blocks: list[str] = []
    for title, items in sections:
        section_lines = [f"[{title}]"]
        subtotal = 0
        for label, amount in items:
            subtotal += amount
            section_lines.append(f"  {label}: {amount}")
        section_lines.append(f"  TOTAL: {subtotal}")
        blocks.append("\n".join(section_lines))
    return "\n\n".join(blocks)
''',
        "app/notifications.py": '''"""Outbound notification templating and queue buffering."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class NotificationMessage:
    """Prepared outbound alert envelope."""

    recipient: str
    subject: str
    body: str
    priority: int = 2
    tags: list[str] = field(default_factory=list)


def format_alert(recipient: str, subject: str) -> str:
    """Format compact single-line notification payload."""
    return f"TO={recipient.strip()} | SUBJECT={subject.strip()}"


class NotificationQueue:
    """In-memory bounded outbound alert buffer."""

    def __init__(self, max_pending: int = 256) -> None:
        self._max_pending = max(1, max_pending)
        self._pending: list[NotificationMessage] = []

    def enqueue(self, message: NotificationMessage) -> bool:
        """Append message if queue has remaining capacity."""
        if len(self._pending) >= self._max_pending:
            return False
        if not message.recipient.strip() or not message.subject.strip():
            raise ValueError("Recipient and subject must be non-empty")
        self._pending.append(message)
        return True

    def drain(self, limit: int = 20) -> list[NotificationMessage]:
        """Pop up to `limit` highest-priority messages."""
        if limit <= 0 or not self._pending:
            return []
        self._pending.sort(key=lambda msg: msg.priority)
        batch = self._pending[:limit]
        self._pending = self._pending[limit:]
        return batch
''',
        "app/cache.py": '''"""Bounded in-memory LRU-like key-value cache."""

from collections import OrderedDict


class MemoryCache:
    """Simple key-value cache with explicit capacity and hit metrics."""

    def __init__(self, max_items: int = 128) -> None:
        self._max_items = max(1, max_items)
        self._data: OrderedDict[str, str] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> str | None:
        """Fetch key and promote recency on cache hit."""
        if key not in self._data:
            self._misses += 1
            return None
        self._data.move_to_end(key)
        self._hits += 1
        return self._data[key]

    def put(self, key: str, value: str) -> None:
        """Insert or update key while evicting oldest item when full."""
        if key in self._data:
            self._data.move_to_end(key)
        elif len(self._data) >= self._max_items:
            self._data.popitem(last=False)
        self._data[key] = value

    def stats(self) -> dict[str, int]:
        """Return cache occupancy and hit/miss counters."""
        return {
            "size": len(self._data),
            "capacity": self._max_items,
            "hits": self._hits,
            "misses": self._misses,
        }
''',
        "app/audit.py": '''"""Structured append-only operational audit trail."""

from dataclasses import dataclass
import json


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """Single immutable audit event."""

    sequence: int
    actor: str
    action: str
    resource: str
    outcome: str = "ALLOW"

    def to_json_line(self) -> str:
        """Serialize record as deterministic single-line JSON."""
        return json.dumps(
            {
                "seq": self.sequence,
                "actor": self.actor,
                "action": self.action,
                "resource": self.resource,
                "outcome": self.outcome,
            },
            sort_keys=True,
        )


class AuditBuffer:
    """Collects bounded audit records before batch persistence."""

    def __init__(self, max_records: int = 500) -> None:
        self._max_records = max(1, max_records)
        self._seq = 0
        self._records: list[AuditRecord] = []

    def record(self, actor: str, action: str, resource: str, outcome: str = "ALLOW") -> AuditRecord:
        """Append a new sequential audit entry."""
        self._seq += 1
        entry = AuditRecord(
            sequence=self._seq,
            actor=actor.strip(),
            action=action.strip(),
            resource=resource.strip(),
            outcome=outcome.strip(),
        )
        if len(self._records) >= self._max_records:
            self._records.pop(0)
        self._records.append(entry)
        return entry

    def export_jsonl(self) -> str:
        """Render buffered entries as JSONL text."""
        return "\n".join(rec.to_json_line() for rec in self._records)
''',
        "app/exports.py": '''"""Batch payload exporter for tabular diagnostic dumps."""

from typing import Mapping, Sequence


def normalize_export_headers(columns: Sequence[str]) -> list[str]:
    """Normalize and deduplicate export column headers."""
    seen: set[str] = set()
    headers: list[str] = []
    for col in columns:
        cleaned = "_".join(col.strip().lower().split())
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            headers.append(cleaned)
    return headers


def build_tsv_export(
    columns: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> str:
    """Serialize rows into tab-separated text escaping tabs and newlines."""
    headers = normalize_export_headers(columns)
    if not headers:
        return ""
    lines = ["\t".join(headers)]
    for row in rows:
        cells: list[str] = []
        for col in headers:
            raw = str(row.get(col, ""))
            cleaned = raw.replace("\t", " ").replace("\n", " ").replace("\r", " ")
            cells.append(cleaned)
        lines.append("\t".join(cells))
    return "\n".join(lines)
''',
        "app/webhooks.py": '''"""Outbound webhook retry schedule and payload validation."""

from dataclasses import dataclass


@dataclass(slots=True)
class WebhookDelivery:
    """Tracks retry state for an outbound webhook event."""

    delivery_id: str
    endpoint_url: str
    event_type: str
    attempts: int = 0
    max_attempts: int = 5
    delivered: bool = False


def compute_backoff_seconds(attempt: int, *, base_seconds: int = 5, cap_seconds: int = 900) -> int:
    """Compute bounded exponential retry delay in integer seconds."""
    if attempt <= 0:
        return 0
    delay = base_seconds * (2 ** min(attempt - 1, 10))
    return min(cap_seconds, delay)


def schedule_next_attempt(delivery: WebhookDelivery) -> int | None:
    """Advance attempt counter and return next delay, or None if exhausted."""
    if delivery.delivered or delivery.attempts >= delivery.max_attempts:
        return None
    delivery.attempts += 1
    return compute_backoff_seconds(delivery.attempts)
''',
        "tests/test_sample_app.py": '''"""Unit tests for sample calibration project."""

import unittest

from app.auth import AuthService
from app.billing import InvoiceCalculator
from app.db import UserRepository


class TestSampleApp(unittest.TestCase):
    def test_auth_and_billing(self) -> None:
        auth = AuthService(UserRepository(), "dev-secret")
        tok = auth.issue_token("alice@example.com")
        self.assertTrue(auth.authenticate_token(tok))
        calc = InvoiceCalculator()
        self.assertEqual(calc.calculate_total_cents(1000, 1900), 1190)


if __name__ == "__main__":
    unittest.main()
''',
        "README.md": "# Calibration Sample\n\n## Overview\n\nSample multi-module service.\n",
    }

    for rel_path, content in files.items():
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    store = ContextStore(root)
    store.init(
        ProjectMeta(
            name="bench-calibration-sample",
            vision="Zero-dependency authentication and integer-cents billing microservice.",
            non_goals=["Third-party ORM/validation frameworks", "XML session files"],
            tech_stack_notes="Python 3.11+ stdlib only; constant-time HMAC verification required.",
        ),
        force=True,
    )
    indexer = RepoIndexer(root, store=store)
    index = indexer.build_or_update(force=True)

    store.add_card(
        KnowledgeCard(
            id="INV-001",
            kind=CardKind.INVARIANT,
            title="Stdlib only — never use pydantic or requests",
            statement="Core modules must use only Python 3.11+ standard library; never use pydantic or requests.",
            rationale="Zero external dependency footprint.",
            immutable=True,
            always_include=True,
            tags=["architecture", "stdlib"],
            anchors=[
                Anchor(
                    file_path="app/auth.py",
                    symbol="AuthService",
                    file_hash=index.files["app/auth.py"].sha256,
                )
            ],
        )
    )
    store.add_card(
        KnowledgeCard(
            id="INV-002",
            kind=CardKind.INVARIANT,
            title="Integer cents only for currency",
            statement="Represent all money in integer cents; never use float for currency math.",
            rationale="Avoids IEEE-754 rounding drift.",
            immutable=True,
            always_include=True,
            tags=["billing", "money"],
            anchors=[
                Anchor(
                    file_path="app/billing.py",
                    symbol="InvoiceCalculator",
                    file_hash=index.files["app/billing.py"].sha256,
                )
            ],
        )
    )
    store.add_card(
        KnowledgeCard(
            id="ADR-001",
            kind=CardKind.ADR,
            title="Store sessions in XML cookies",
            statement="Store opaque session IDs in server XML cookies.",
            rationale="Initial throwaway prototype.",
            tags=["auth", "session"],
            anchors=[Anchor(file_path="app/auth.py")],
        )
    )
    store.supersede_card(
        "ADR-001",
        KnowledgeCard(
            id="ADR-002",
            kind=CardKind.ADR,
            title="Stateless HMAC-SHA256 tokens",
            statement="Issue and verify stateless HMAC-SHA256 tokens via AuthService and sign_payload.",
            rationale="Stateless verification without server session files.",
            tags=["auth", "crypto"],
            anchors=[
                Anchor(
                    file_path="app/crypto.py",
                    symbol="sign_payload",
                    file_hash=index.files["app/crypto.py"].sha256,
                )
            ],
        ),
    )
    store.add_card(
        KnowledgeCard(
            id="PIT-001",
            kind=CardKind.PITFALL,
            title="Do not use hashlib.md5 for token signing",
            statement="Never use `md5` or `hashlib.md5` for authentication tokens or password hashes.",
            rationale="Cryptographically broken hash function.",
            always_include=False,
            tags=["auth", "security"],
            anchors=[Anchor(file_path="app/crypto.py")],
        )
    )
    store.add_task(
        TaskNode(
            id="TASK-001",
            title="Add expiry timestamp to AuthService tokens",
            description="Extend AuthService.issue_token and authenticate_token to validate expiration.",
            status=TaskStatus.IN_PROGRESS,
            priority=1,
            anchors=[
                Anchor(file_path="app/auth.py", symbol="AuthService"),
                Anchor(file_path="app/crypto.py", symbol="sign_payload"),
            ],
            related_cards=["INV-001", "ADR-002", "PIT-001"],
            acceptance_criteria=[
                "Expired tokens fail authenticate_token cleanly",
                "Uses only stdlib HMAC-SHA256",
            ],
            next_step_hint="Include expiry epoch in signed payload inside app/auth.py",
        )
    )
    store.add_task(
        TaskNode(
            id="TASK-002",
            title="Add discount support in InvoiceCalculator",
            description="Support basis-point discount before tax inside InvoiceCalculator.calculate_total_cents.",
            status=TaskStatus.TODO,
            priority=2,
            anchors=[Anchor(file_path="app/billing.py", symbol="InvoiceCalculator")],
            related_cards=["INV-002"],
            acceptance_criteria=["All arithmetic remains strictly integer cents"],
        )
    )
    store.append_session(
        SessionLogEntry(
            session_id="SES-001",
            task_id="TASK-001",
            summary="Migrated from XML cookies to stateless HMAC-SHA256 tokens.",
            next_anchor="Extend AuthService.issue_token with expiry check.",
        )
    )
    return root


# ----------------------------------------------------------------------
# BenchmarkSuite
# ----------------------------------------------------------------------


class BenchmarkSuite:
    """Runs offline (semi-automated) or live A/B context benchmarks."""

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
        self._compiler = ContextCompiler(self._store, indexer=self._indexer)

    @property
    def store(self) -> ContextStore:
        return self._store

    @property
    def indexer(self) -> RepoIndexer:
        return self._indexer

    def prepare(
        self,
        *,
        task_ids: Sequence[str] | None = None,
        bench_dir: Path | str | None = None,
        token_budget: int = 8000,
        test_command: str | None = None,
    ) -> dict[str, Any]:
        """Prepare prompts for all 3 strategies plus `rubric.json` for A/B testing."""
        out_dir = (
            Path(bench_dir).resolve()
            if bench_dir is not None
            else (self._store.ctx_dir / "bench")
        )
        prompts_dir = out_dir / "prompts"
        responses_dir = out_dir / "responses"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        responses_dir.mkdir(parents=True, exist_ok=True)

        index = self._indexer.build_or_update()
        all_tasks = self._store.list_tasks()
        if not all_tasks:
            raise CtxForgeError("No tasks found in store. Add at least one task before running bench.")

        if task_ids:
            selected_tasks = [self._store.require_task(tid) for tid in task_ids]
        else:
            open_tasks = [
                t
                for t in all_tasks
                if t.status in (TaskStatus.IN_PROGRESS, TaskStatus.TODO, TaskStatus.BLOCKED)
            ]
            selected_tasks = (open_tasks or all_tasks)[:3]

        all_cards = self._store.list_cards()
        active_cards = [c for c in all_cards if c.status is CardStatus.ACTIVE]
        active_by_id = {c.id: c for c in active_cards}

        invariants_spec = [
            {
                "id": c.id,
                "title": c.title,
                "statement": c.statement,
                "forbidden_patterns": _derive_forbidden_patterns(
                    c, active_cards_by_id=active_by_id
                ),
            }
            for c in active_cards
            if c.kind is CardKind.INVARIANT
        ]
        adrs_spec = [
            {
                "id": c.id,
                "title": c.title,
                "statement": c.statement,
                "forbidden_patterns": _derive_forbidden_patterns(
                    c, active_cards_by_id=active_by_id
                ),
            }
            for c in active_cards
            if c.kind is CardKind.ADR
        ]
        rejected_spec = [
            {
                "id": c.id,
                "kind": c.kind.value,
                "status": c.status.value,
                "title": c.title,
                "statement": c.statement,
                "relapse_patterns": _derive_forbidden_patterns(
                    c, active_cards_by_id=active_by_id
                ),
            }
            for c in all_cards
            if c.kind is CardKind.PITFALL or c.status is CardStatus.SUPERSEDED
        ]

        rubric_tasks: list[dict[str, Any]] = []
        for task in selected_tasks:
            compiled = self._compiler.compile(
                CompileRequest(task_id=task.id, token_budget=token_budget),
                index=index,
            )
            # Calculate intrinsic signal tokens for this task from Tier-1, Tier-2, cards, and task spec.
            tier1_tok = sum(
                index.files[p].token_count
                for p in compiled.stats.get("tier1_files", [])
                if p in index.files
            )
            tier2_tok = sum(
                index.files[p].skeleton_tokens
                for p in compiled.stats.get("tier2_files", [])
                if p in index.files
            )
            task_core_tok = estimate_tokens(_format_task_brief(task)) + 60
            # Signal tokens represent useful task + targeted code + governance cards.
            baseline_signal_tok = max(
                task_core_tok,
                min(
                    compiled.capsule_tokens,
                    task_core_tok
                    + tier1_tok
                    + tier2_tok
                    + compiled.stats.get("cards_invariants", 0) * 35
                    + compiled.stats.get("cards_adrs", 0) * 30
                    + compiled.stats.get("cards_pitfalls", 0) * 30,
                ),
            )

            full_prompt, full_tok, full_sig, full_snr = _build_full_dump_prompt(
                self._store, index, task, baseline_signal_tok
            )
            chat_prompt, chat_tok, chat_sig, chat_snr = _build_long_chat_prompt(
                self._store, index, task, baseline_signal_tok
            )

            ctx_tok = compiled.capsule_tokens
            # Tier-3 path tree is lightweight orientation; subtract its minor share from pure signal.
            tier3_count = compiled.stats.get("tier3_count", 0)
            ctx_noise = min(ctx_tok // 6, tier3_count * 6)
            ctx_sig = max(1, ctx_tok - ctx_noise)
            ctx_snr = round(min(1.0, ctx_sig / ctx_tok), 4) if ctx_tok > 0 else 0.0

            strategy_payloads = {
                "baseline_full_dump": (full_prompt, full_tok, full_sig, full_snr),
                "baseline_long_chat": (chat_prompt, chat_tok, chat_sig, chat_snr),
                "ctxforge_compiled": (compiled.capsule_text, ctx_tok, ctx_sig, ctx_snr),
            }

            entries: dict[str, dict[str, Any]] = {}
            for strat, (p_text, in_tok, sig_tok, snr_val) in strategy_payloads.items():
                p_rel = f"prompts/{task.id}__{strat}.md"
                r_rel = f"responses/{task.id}__{strat}.md"
                (out_dir / p_rel).write_text(p_text + "\n", encoding="utf-8")
                entries[strat] = {
                    "prompt_file": p_rel,
                    "response_file": r_rel,
                    "response_text": "",
                    "input_tokens": in_tok,
                    "signal_tokens": sig_tok,
                    "snr": snr_val,
                    "latency_ms": None,
                    "manual_quality_score": None,
                }

            rubric_tasks.append(
                {
                    "task_id": task.id,
                    "title": task.title,
                    "target_files": list(compiled.stats.get("tier1_files", [])),
                    "entries": entries,
                }
            )

        rubric: dict[str, Any] = {
            "version": _BENCH_SCHEMA_VERSION,
            "project_root": str(self._store.root),
            "created_at": utc_now_iso(),
            "token_budget": token_budget,
            "test_command": test_command,
            "invariants": invariants_spec,
            "adrs": adrs_spec,
            "rejected_cards": rejected_spec,
            "tasks": rubric_tasks,
        }
        (out_dir / "rubric.json").write_text(
            json.dumps(rubric, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return rubric

    @staticmethod
    def format_prepare_table(rubric: Mapping[str, Any]) -> str:
        """Render Input Token Footprint & Signal-to-Noise table from prepared rubric."""
        lines = [
            "=== ctxforge Benchmark Prepared ===",
            f"{'Task':<10} {'Strategy':<20} {'InputTok':>9} {'SignalTok':>10} {'SNR':>8}",
            "-" * 61,
        ]
        for task in rubric.get("tasks", []):
            tid = str(task.get("task_id", ""))
            entries = task.get("entries", {})
            for strat in STRATEGIES:
                entry = entries.get(strat)
                if not entry:
                    continue
                lines.append(
                    f"{tid:<10} {strat:<20} "
                    f"{entry['input_tokens']:>9d} "
                    f"{entry['signal_tokens']:>10d} "
                    f"{entry['snr'] * 100:>7.1f}%"
                )
        return "\n".join(lines)

    def _load_rubric(self, bench_dir: Path) -> dict[str, Any]:
        rubric_path = bench_dir / "rubric.json"
        if not rubric_path.is_file():
            raise CtxForgeError(
                f"Benchmark rubric not found at '{rubric_path}'. Run `ctx bench prepare` first."
            )
        try:
            data = json.loads(rubric_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CtxForgeError(f"Failed to parse '{rubric_path}': {exc}") from exc
        if not isinstance(data, dict):
            raise CtxForgeError(f"Invalid rubric structure in '{rubric_path}'.")
        return data

    def score(
        self,
        *,
        bench_dir: Path | str | None = None,
        test_command: str | None = None,
    ) -> BenchmarkReport:
        """Score saved LLM responses in `bench_dir` and generate `benchmark_report.md`."""
        out_dir = (
            Path(bench_dir).resolve()
            if bench_dir is not None
            else (self._store.ctx_dir / "bench")
        )
        rubric = self._load_rubric(out_dir)
        index = self._indexer.build_or_update()
        valid_files = set(index.files.keys())
        symbols_by_file: dict[str, set[str]] = {}
        for rel_path, record in index.files.items():
            syms = {s.name for s in record.symbols}
            syms.update(f"{s.parent}.{s.name}" for s in record.symbols if s.parent)
            symbols_by_file[rel_path] = syms

        effective_test_cmd = test_command or rubric.get("test_command")
        exec_passed: bool | None = None
        exec_output = ""
        if isinstance(effective_test_cmd, str) and effective_test_cmd.strip():
            try:
                proc = subprocess.run(
                    effective_test_cmd.strip(),
                    shell=True,
                    cwd=str(self._store.root),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=60,
                    check=False,
                )
                exec_passed = proc.returncode == 0
                exec_output = (proc.stdout + "\n" + proc.stderr).strip()[:400]
            except (OSError, subprocess.SubprocessError) as exc:
                exec_passed = False
                exec_output = str(exc)

        invariants_spec = rubric.get("invariants", [])
        adrs_spec = rubric.get("adrs", [])
        rejected_spec = rubric.get("rejected_cards", [])

        scores: list[StrategyScore] = []
        for task_spec in rubric.get("tasks", []):
            task_id = str(task_spec["task_id"])
            entries = task_spec.get("entries", {})
            for strat in STRATEGIES:
                entry = entries.get(strat)
                if not entry:
                    continue
                resp_text = str(entry.get("response_text", "") or "").strip()
                if not resp_text:
                    resp_file = out_dir / str(entry.get("response_file", ""))
                    if resp_file.is_file():
                        resp_text = resp_file.read_text(encoding="utf-8").strip()

                input_tokens = int(entry.get("input_tokens", 0))
                signal_tokens = int(entry.get("signal_tokens", 0))
                snr = float(entry.get("snr", 0.0))
                latency_ms = entry.get("latency_ms")
                manual_score = entry.get("manual_quality_score")

                if not resp_text:
                    scores.append(
                        StrategyScore(
                            task_id=task_id,
                            strategy=strat,
                            has_response=False,
                            input_tokens=input_tokens,
                            output_tokens=0,
                            signal_tokens=signal_tokens,
                            snr=snr,
                            adherence_score=0.0,
                            relapse_count=0,
                            anchor_accuracy=0.0,
                            exec_passed=exec_passed,
                            exec_output=exec_output,
                            latency_ms=latency_ms,
                            manual_quality_score=manual_score,
                        )
                    )
                    continue

                output_tokens = estimate_tokens(resp_text)
                code_text, prose_text = _split_code_and_prose(resp_text)

                # 1. Invariant & ADR Adherence
                violations: list[str] = []
                penalty = 0.0
                for inv in invariants_spec:
                    for pat in inv.get("forbidden_patterns", []):
                        if _matches_code_usage(code_text, pat) or _matches_affirmative_prose(
                            prose_text, pat
                        ):
                            violations.append(
                                f"{inv['id']} ({inv['title']}): matched forbidden '{pat}'"
                            )
                            penalty += 0.35
                            break
                for adr in adrs_spec:
                    for pat in adr.get("forbidden_patterns", []):
                        if _matches_code_usage(code_text, pat) or _matches_affirmative_prose(
                            prose_text, pat
                        ):
                            violations.append(
                                f"{adr['id']} ({adr['title']}): violated '{pat}'"
                            )
                            penalty += 0.25
                            break
                adherence = round(max(0.0, 1.0 - penalty), 4)

                # 2. Rejected-Idea Relapses (PITFALL & SUPERSEDED)
                relapses: list[str] = []
                for rej in rejected_spec:
                    for pat in rej.get("relapse_patterns", []):
                        if _matches_code_usage(code_text, pat) or _matches_affirmative_prose(
                            prose_text, pat
                        ):
                            relapses.append(
                                f"{rej['id']} ({rej['kind']}/{rej['status']}): relapsed to '{pat}'"
                            )
                            break

                # 3. Anchor Accuracy
                anchor_acc, verified_refs, hallucinated_refs = _evaluate_anchor_accuracy(
                    resp_text, valid_files, symbols_by_file
                )

                scores.append(
                    StrategyScore(
                        task_id=task_id,
                        strategy=strat,
                        has_response=True,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        signal_tokens=signal_tokens,
                        snr=snr,
                        adherence_score=adherence,
                        violations=violations,
                        relapse_count=len(relapses),
                        relapses=relapses,
                        anchor_accuracy=anchor_acc,
                        verified_refs=verified_refs,
                        hallucinated_refs=hallucinated_refs,
                        exec_passed=exec_passed,
                        exec_output=exec_output,
                        latency_ms=latency_ms,
                        manual_quality_score=manual_score,
                    )
                )

        aggregates = _compute_aggregates(scores)
        report_md = _render_markdown_report(
            rubric=rubric,
            scores=scores,
            aggregates=aggregates,
            exec_passed=exec_passed,
            exec_output=exec_output,
        )
        report_path = out_dir / "benchmark_report.md"
        report_path.write_text(report_md, encoding="utf-8")
        return BenchmarkReport(
            bench_dir=out_dir,
            scores=scores,
            strategy_aggregates=aggregates,
            report_markdown=report_md,
            report_path=report_path,
        )

    def run(
        self,
        adapter: LLMAdapter,
        *,
        task_ids: Sequence[str] | None = None,
        bench_dir: Path | str | None = None,
        token_budget: int = 8000,
        test_command: str | None = None,
    ) -> BenchmarkReport:
        """Automated Mode: execute all 3 strategies through adapter, then score."""
        out_dir = (
            Path(bench_dir).resolve()
            if bench_dir is not None
            else (self._store.ctx_dir / "bench")
        )
        rubric = self.prepare(
            task_ids=task_ids,
            bench_dir=out_dir,
            token_budget=token_budget,
            test_command=test_command,
        )
        for task_spec in rubric["tasks"]:
            for strat in STRATEGIES:
                entry = task_spec["entries"][strat]
                prompt_text = (out_dir / entry["prompt_file"]).read_text(encoding="utf-8")
                t0 = time.perf_counter()
                response_text = adapter.complete(prompt_text)
                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                entry["response_text"] = response_text
                entry["latency_ms"] = elapsed_ms
                (out_dir / entry["response_file"]).write_text(
                    response_text.rstrip() + "\n",
                    encoding="utf-8",
                )

        (out_dir / "rubric.json").write_text(
            json.dumps(rubric, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return self.score(bench_dir=out_dir, test_command=test_command)


def _compute_aggregates(scores: list[StrategyScore]) -> dict[str, dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}
    for strat in STRATEGIES:
        rows = [s for s in scores if s.strategy == strat]
        if not rows:
            continue
        answered = [s for s in rows if s.has_response]
        n = len(rows)
        n_ans = len(answered)
        latencies = [s.latency_ms for s in answered if s.latency_ms is not None]
        aggregates[strat] = {
            "tasks_count": n,
            "answered_count": n_ans,
            "avg_input_tokens": sum(s.input_tokens for s in rows) / n,
            "avg_output_tokens": (
                sum(s.output_tokens for s in answered) / n_ans if n_ans else 0.0
            ),
            "avg_snr": sum(s.snr for s in rows) / n,
            "avg_adherence": (
                sum(s.adherence_score for s in answered) / n_ans if n_ans else 0.0
            ),
            "total_violations": sum(len(s.violations) for s in answered),
            "total_relapses": sum(s.relapse_count for s in answered),
            "avg_anchor_accuracy": (
                sum(s.anchor_accuracy for s in answered) / n_ans if n_ans else 0.0
            ),
            "avg_latency_ms": (
                sum(latencies) / len(latencies) if latencies else None
            ),
        }
    return aggregates


def _ascii_bar(value: float, max_value: float, width: int = 24) -> str:
    if max_value <= 0:
        return "." * width
    filled = max(1 if value > 0 else 0, round((value / max_value) * width))
    filled = min(width, filled)
    return "#" * filled + "." * (width - filled)


def _render_markdown_report(
    *,
    rubric: Mapping[str, Any],
    scores: list[StrategyScore],
    aggregates: Mapping[str, Mapping[str, Any]],
    exec_passed: bool | None,
    exec_output: str,
) -> str:
    lines: list[str] = [
        "# ctxforge A/B Benchmark Report",
        "",
        f"- Generated at: {utc_now_iso()}",
        f"- Project root: `{rubric.get('project_root', '')}`",
        f"- Tasks evaluated: {len(rubric.get('tasks', []))}",
        "",
        "## 1. Summary Comparison",
        "",
        "| Strategy | Avg Input Tok | Avg Output Tok | Avg SNR | Adherence | Relapses | Anchor Acc | Avg Latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strat in STRATEGIES:
        agg = aggregates.get(strat)
        if not agg:
            continue
        lat_str = (
            f"{agg['avg_latency_ms']:.1f} ms"
            if agg["avg_latency_ms"] is not None
            else "manual"
        )
        lines.append(
            f"| `{strat}` | "
            f"{agg['avg_input_tokens']:.0f} | "
            f"{agg['avg_output_tokens']:.0f} | "
            f"{agg['avg_snr'] * 100:.1f}% | "
            f"{agg['avg_adherence'] * 100:.1f}% | "
            f"{agg['total_relapses']} | "
            f"{agg['avg_anchor_accuracy'] * 100:.1f}% | "
            f"{lat_str} |"
        )

    # ASCII Visualization
    max_tok = max(
        (float(aggregates[s]["avg_input_tokens"]) for s in STRATEGIES if s in aggregates),
        default=1.0,
    )
    lines.extend(
        [
            "",
            "## 2. ASCII Comparison (Token Footprint & Quality)",
            "",
            "```text",
            "Input Token Footprint (lower is cheaper):",
        ]
    )
    for strat in STRATEGIES:
        if strat not in aggregates:
            continue
        tok = float(aggregates[strat]["avg_input_tokens"])
        snr = float(aggregates[strat]["avg_snr"]) * 100.0
        adh = float(aggregates[strat]["avg_adherence"]) * 100.0
        rel = int(aggregates[strat]["total_relapses"])
        lines.append(
            f"{strat:<20} | {_ascii_bar(tok, max_tok)} {tok:>6.0f} tok | "
            f"SNR {snr:>5.1f}% | Adh {adh:>5.1f}% | Relapses {rel}"
        )
    lines.append("```")

    # Per-Task Breakdown
    lines.extend(
        [
            "",
            "## 3. Per-Task Breakdown",
            "",
            "| Task | Strategy | Answered | Input Tok | Output Tok | Adherence | Relapses | Anchor Acc |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for sc in scores:
        lines.append(
            f"| `{sc.task_id}` | `{sc.strategy}` | "
            f"{'yes' if sc.has_response else 'no'} | "
            f"{sc.input_tokens} | {sc.output_tokens} | "
            f"{sc.adherence_score * 100:.1f}% | {sc.relapse_count} | "
            f"{sc.anchor_accuracy * 100:.1f}% |"
        )

    # Detailed Violations & Relapses
    findings: list[str] = []
    for sc in scores:
        for v in sc.violations:
            findings.append(f"- Violation [{sc.task_id}/{sc.strategy}]: {v}")
        for r in sc.relapses:
            findings.append(f"- Relapse [{sc.task_id}/{sc.strategy}]: {r}")
        for h in sc.hallucinated_refs:
            findings.append(
                f"- Hallucinated Anchor [{sc.task_id}/{sc.strategy}]: {h}"
            )
    if findings:
        lines.extend(["", "## 4. Detected Violations, Relapses & Hallucinations", "", *findings])

    if exec_passed is not None:
        lines.extend(
            [
                "",
                "## 5. Executable Verification",
                "",
                f"- Test command status: {'PASSED' if exec_passed else 'FAILED'}",
                f"- Output excerpt: `{exec_output}`" if exec_output else "",
            ]
        )

    # Hypotheses H1-H3 Conclusions
    full_agg = aggregates.get("baseline_full_dump", {})
    chat_agg = aggregates.get("baseline_long_chat", {})
    ctx_agg = aggregates.get("ctxforge_compiled", {})
    lines.extend(["", "## 6. Experimental Conclusions (Hypotheses H1-H3)", ""])

    if ctx_agg and full_agg:
        ctx_tok = float(ctx_agg["avg_input_tokens"])
        full_tok = float(full_agg["avg_input_tokens"])
        chat_tok = float(chat_agg.get("avg_input_tokens", 0.0))
        ctx_snr = float(ctx_agg["avg_snr"])
        full_snr = float(full_agg["avg_snr"])
        tok_reduction = ((1.0 - (ctx_tok / full_tok)) * 100.0) if full_tok > 0 else 0.0
        h1_ok = ctx_tok < full_tok and ctx_snr >= full_snr
        lines.append(
            f"- H1 (Token Footprint & Signal-to-Noise): "
            f"{'CONFIRMED' if h1_ok else 'INCONCLUSIVE'} — `ctxforge_compiled` uses "
            f"{ctx_tok:.0f} avg input tokens vs {full_tok:.0f} (`baseline_full_dump`, "
            f"{tok_reduction:.1f}% reduction) and {chat_tok:.0f} (`baseline_long_chat`), "
            f"with SNR {ctx_snr * 100:.1f}% vs {full_snr * 100:.1f}%."
        )
    if ctx_agg and chat_agg:
        ctx_rel = int(ctx_agg["total_relapses"])
        chat_rel = int(chat_agg["total_relapses"])
        ctx_adh = float(ctx_agg["avg_adherence"])
        chat_adh = float(chat_agg["avg_adherence"])
        answered_both = (
            int(ctx_agg["answered_count"]) > 0 and int(chat_agg["answered_count"]) > 0
        )
        h2_ok = answered_both and (ctx_rel <= chat_rel and ctx_adh >= chat_adh)
        lines.append(
            f"- H2 (Truth Governance & Relapse Prevention): "
            f"{'CONFIRMED' if h2_ok else ('PENDING RESPONSES' if not answered_both else 'INCONCLUSIVE')} — "
            f"`ctxforge_compiled` recorded {ctx_rel} relapses ({ctx_adh * 100:.1f}% adherence) "
            f"vs {chat_rel} relapses ({chat_adh * 100:.1f}% adherence) in `baseline_long_chat`."
        )
    if ctx_agg:
        ctx_acc = float(ctx_agg["avg_anchor_accuracy"])
        chat_acc = float(chat_agg.get("avg_anchor_accuracy", 0.0))
        answered_ctx = int(ctx_agg["answered_count"]) > 0
        h3_ok = answered_ctx and ctx_acc >= chat_acc
        lines.append(
            f"- H3 (Grounding & Anchor Accuracy): "
            f"{'CONFIRMED' if h3_ok else ('PENDING RESPONSES' if not answered_ctx else 'INCONCLUSIVE')} — "
            f"`ctxforge_compiled` achieved {ctx_acc * 100:.1f}% anchor accuracy "
            f"vs {chat_acc * 100:.1f}% for `baseline_long_chat`."
        )

    return "\n".join(line for line in lines if line is not None).rstrip() + "\n"
