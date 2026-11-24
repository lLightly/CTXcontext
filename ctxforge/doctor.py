"""Ground-truth Reality-Drift Doctor: zero-LLM health checks for .ctx/ memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import StrEnum
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

from .indexer import RepoIndex, RepoIndexer
from .models import Anchor, CardKind, CardStatus, KnowledgeCard, TaskStatus, utc_now_iso
from .storage import ContextStore, CtxForgeError
from .tokens import estimate_tokens

_MAX_ALWAYS_INCLUDE_COUNT = 10
_MAX_ALWAYS_INCLUDE_TOKENS = 1500
_SIMILARITY_RATIO_THRESHOLD = 0.82
_JACCARD_THRESHOLD = 0.75

_TITLE_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "for",
        "in",
        "on",
        "to",
        "of",
        "with",
        "use",
        "do",
        "not",
        "и",
        "в",
        "на",
        "для",
        "не",
        "с",
        "по",
    }
)


class IssueKind(StrEnum):
    """Diagnostic issue classification."""

    BROKEN_ANCHOR = "BROKEN_ANCHOR"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    ALWAYS_INCLUDE_BLOAT = "ALWAYS_INCLUDE_BLOAT"
    TASK_DAG_ERROR = "TASK_DAG_ERROR"
    DUPLICATE_CARD = "DUPLICATE_CARD"


class IssueSeverity(StrEnum):
    """Severity level of a diagnostic issue."""

    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass(slots=True)
class HealthIssue:
    """Single detected inconsistency between `.ctx/` memory and codebase reality."""

    kind: IssueKind
    severity: IssueSeverity
    entity_id: str
    message: str
    file_path: str | None = None
    symbol: str | None = None


@dataclass(slots=True)
class HealthReport:
    """Aggregated diagnostics report produced by `ContextDoctor.diagnose()`."""

    issues: list[HealthIssue] = field(default_factory=list)
    always_include_count: int = 0
    always_include_tokens: int = 0
    cards_needing_review: list[str] = field(default_factory=list)
    fixed_hashes: int = 0

    @property
    def errors(self) -> list[HealthIssue]:
        return [issue for issue in self.issues if issue.severity is IssueSeverity.ERROR]

    @property
    def warnings(self) -> list[HealthIssue]:
        return [issue for issue in self.issues if issue.severity is IssueSeverity.WARNING]

    @property
    def broken_anchors(self) -> list[HealthIssue]:
        return [issue for issue in self.issues if issue.kind is IssueKind.BROKEN_ANCHOR]

    @property
    def needs_review(self) -> list[HealthIssue]:
        return [issue for issue in self.issues if issue.kind is IssueKind.NEEDS_REVIEW]

    @property
    def is_healthy(self) -> bool:
        return len(self.issues) == 0

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def format_report(self) -> str:
        """Render a clear terminal-friendly health report."""
        lines = [
            "=== ctxforge Context Doctor Report ===",
            (
                f"Status: {'HEALTHY' if self.is_healthy else 'ISSUES FOUND'} | "
                f"errors={len(self.errors)}, warnings={len(self.warnings)}, "
                f"always_include={self.always_include_count} ({self.always_include_tokens} tok)"
            ),
        ]
        if self.fixed_hashes > 0:
            lines.append(f"Auto-fixed anchor hashes: {self.fixed_hashes}")

        if not self.issues:
            lines.append("All anchors, hashes, DAG links, and hygiene checks passed.")
            return "\n".join(lines)

        for issue in self.issues:
            loc = f" @ {issue.file_path}" if issue.file_path else ""
            sym = f":{issue.symbol}" if issue.symbol else ""
            lines.append(
                f"[{issue.severity.value}] {issue.kind.value} [{issue.entity_id}{loc}{sym}]: "
                f"{issue.message}"
            )
        return "\n".join(lines)


def _normalize_title(title: str) -> str:
    cleaned = re.sub(r"[^\w\s]", " ", title.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _title_tokens(norm_title: str) -> set[str]:
    return {
        tok
        for tok in norm_title.split()
        if tok and tok not in _TITLE_STOPWORDS
    }


def _titles_look_duplicate(title_a: str, title_b: str) -> bool:
    norm_a = _normalize_title(title_a)
    norm_b = _normalize_title(title_b)
    if not norm_a or not norm_b:
        return False
    if norm_a == norm_b:
        return True
    if min(len(norm_a), len(norm_b)) >= 8:
        if SequenceMatcher(None, norm_a, norm_b).ratio() >= _SIMILARITY_RATIO_THRESHOLD:
            return True
    toks_a = _title_tokens(norm_a)
    toks_b = _title_tokens(norm_b)
    if toks_a and toks_b:
        overlap = toks_a & toks_b
        union = toks_a | toks_b
        if len(overlap) >= 2 and (len(overlap) / len(union)) >= _JACCARD_THRESHOLD:
            return True
    return False


def _hashes_differ(recorded_hash: str, current_hash: str) -> bool:
    prefix_len = min(len(recorded_hash), len(current_hash))
    if prefix_len < 6:
        return recorded_hash.lower() != current_hash.lower()
    return recorded_hash[:prefix_len].lower() != current_hash[:prefix_len].lower()


def _compute_disk_sha256_short(path: Path, length: int = 16) -> str | None:
    if not path.is_file():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()[:length]


class ContextDoctor:
    """Zero-LLM consistency checker for anchors, hash drift, DAG, and noise hygiene."""

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

    @property
    def store(self) -> ContextStore:
        return self._store

    @property
    def indexer(self) -> RepoIndexer:
        return self._indexer

    def _resolve_file_state(
        self,
        anchor: Anchor,
        index: RepoIndex,
    ) -> tuple[bool, str | None, set[str] | None]:
        """Return `(exists, current_sha256_short, known_symbols_or_none)`."""
        file_rec = index.files.get(anchor.file_path)
        if file_rec is not None:
            known_symbols: set[str] | None = None
            if file_rec.language in (
                "python",
                "typescript",
                "javascript",
                "go",
                "rust",
                "java",
                "markdown",
                "rst",
            ):
                known_symbols = {
                    sym.name for sym in file_rec.symbols
                } | {
                    f"{sym.parent}.{sym.name}"
                    for sym in file_rec.symbols
                    if sym.parent
                }
            return True, file_rec.sha256, known_symbols

        candidate = (self._store.root / anchor.file_path).resolve()
        if not candidate.is_relative_to(self._store.root):
            return False, None, None
        sha = _compute_disk_sha256_short(candidate)
        if sha is None:
            return False, None, None
        return True, sha, None

    def _persist_cards_with_updated_hashes(
        self, updated_active_by_id: dict[str, KnowledgeCard]
    ) -> None:
        all_cards = self._store.list_cards()
        merged: list[KnowledgeCard] = [
            updated_active_by_id.get(card.id, card) for card in all_cards
        ]
        if hasattr(self._store, "_save_cards"):
            self._store._save_cards(merged)  # noqa: SLF001
            return

        from .storage import _atomic_write_text

        lines = [json.dumps(c.to_dict(), ensure_ascii=False) for c in merged]
        payload = ("\n".join(lines) + "\n") if lines else ""
        _atomic_write_text(self._store.cards_file, payload)

    def _check_anchors_for_entity(
        self,
        entity_id: str,
        anchors: Iterable[Anchor],
        index: RepoIndex,
        *,
        check_hash_drift: bool,
        fix_hashes: bool,
        issues: list[HealthIssue],
        cards_needing_review: set[str],
    ) -> bool:
        """Validate anchors on a card or task; returns True if any hash was updated."""
        modified = False
        for anchor in anchors:
            exists, current_sha, known_symbols = self._resolve_file_state(anchor, index)
            if not exists:
                issues.append(
                    HealthIssue(
                        kind=IssueKind.BROKEN_ANCHOR,
                        severity=IssueSeverity.ERROR,
                        entity_id=entity_id,
                        message=f"Anchored file '{anchor.file_path}' does not exist on disk.",
                        file_path=anchor.file_path,
                        symbol=anchor.symbol,
                    )
                )
                continue

            if anchor.symbol and known_symbols is not None:
                if anchor.symbol not in known_symbols:
                    issues.append(
                        HealthIssue(
                            kind=IssueKind.BROKEN_ANCHOR,
                            severity=IssueSeverity.ERROR,
                            entity_id=entity_id,
                            message=(
                                f"Symbol '{anchor.symbol}' not found in '{anchor.file_path}'."
                            ),
                            file_path=anchor.file_path,
                            symbol=anchor.symbol,
                        )
                    )
                    continue

            if check_hash_drift and anchor.file_hash and current_sha:
                if _hashes_differ(anchor.file_hash, current_sha):
                    if fix_hashes:
                        anchor.file_hash = current_sha
                        modified = True
                    else:
                        cards_needing_review.add(entity_id)
                        issues.append(
                            HealthIssue(
                                kind=IssueKind.NEEDS_REVIEW,
                                severity=IssueSeverity.WARNING,
                                entity_id=entity_id,
                                message=(
                                    f"Reality drift: '{anchor.file_path}' changed "
                                    f"(recorded={anchor.file_hash}, current={current_sha}); "
                                    f"NEEDS_REVIEW."
                                ),
                                file_path=anchor.file_path,
                                symbol=anchor.symbol,
                            )
                        )
        return modified

    def diagnose(self, fix_hashes: bool = False) -> HealthReport:
        """Run all zero-LLM health checks on `.ctx/` against disk state."""
        index = self._indexer.build_or_update()
        issues: list[HealthIssue] = []
        cards_needing_review: set[str] = set()
        fixed_hashes = 0

        active_cards = self._store.get_active_cards()
        updated_cards: dict[str, KnowledgeCard] = {}

        # 1. Active KnowledgeCard anchors & reality drift
        for card in active_cards:
            check_drift = card.kind in (CardKind.ADR, CardKind.INVARIANT, CardKind.PITFALL)
            before_hashes = [a.file_hash for a in card.anchors]
            changed = self._check_anchors_for_entity(
                card.id,
                card.anchors,
                index,
                check_hash_drift=check_drift,
                fix_hashes=fix_hashes,
                issues=issues,
                cards_needing_review=cards_needing_review,
            )
            if changed:
                fixed_hashes += sum(
                    1
                    for old_h, anchor in zip(before_hashes, card.anchors)
                    if old_h != anchor.file_hash
                )
                card.updated_at = utc_now_iso()
                updated_cards[card.id] = card

        if fix_hashes and updated_cards:
            self._persist_cards_with_updated_hashes(updated_cards)

        # 2. Open TaskNode anchors
        open_statuses = {TaskStatus.TODO, TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED}
        tasks = self._store.list_tasks()
        for task in tasks:
            if task.status not in open_statuses:
                continue
            self._check_anchors_for_entity(
                task.id,
                task.anchors,
                index,
                check_hash_drift=False,
                fix_hashes=False,
                issues=issues,
                cards_needing_review=cards_needing_review,
            )

        # 3. Context hygiene: always_include bloat (>10 cards or >1500 tokens)
        always_cards = [c for c in active_cards if c.always_include]
        always_count = len(always_cards)
        always_tokens = sum(
            estimate_tokens(f"{c.id} {c.title}: {c.statement}\n{c.rationale}")
            for c in always_cards
        )
        if (
            always_count > _MAX_ALWAYS_INCLUDE_COUNT
            or always_tokens > _MAX_ALWAYS_INCLUDE_TOKENS
        ):
            issues.append(
                HealthIssue(
                    kind=IssueKind.ALWAYS_INCLUDE_BLOAT,
                    severity=IssueSeverity.WARNING,
                    entity_id="always_include",
                    message=(
                        f"Too many always_include cards ({always_count} cards, "
                        f"{always_tokens} tokens; recommended <= {_MAX_ALWAYS_INCLUDE_COUNT} "
                        f"cards and <= {_MAX_ALWAYS_INCLUDE_TOKENS} tokens) — risks baseline noise."
                    ),
                )
            )

        # 4. Task DAG integrity (cycles / dangling depends_on)
        try:
            self._store.validate_task_dag(tasks)
        except CtxForgeError as exc:
            issues.append(
                HealthIssue(
                    kind=IssueKind.TASK_DAG_ERROR,
                    severity=IssueSeverity.ERROR,
                    entity_id="tasks.json",
                    message=str(exc),
                )
            )

        # 5. Duplicate or highly similar active card titles
        for i, card_a in enumerate(active_cards):
            for card_b in active_cards[i + 1 :]:
                if _titles_look_duplicate(card_a.title, card_b.title):
                    issues.append(
                        HealthIssue(
                            kind=IssueKind.DUPLICATE_CARD,
                            severity=IssueSeverity.WARNING,
                            entity_id=f"{card_a.id},{card_b.id}",
                            message=(
                                f"Similar active card titles: '{card_a.title}' ({card_a.id}) "
                                f"and '{card_b.title}' ({card_b.id})."
                            ),
                        )
                    )

        return HealthReport(
            issues=issues,
            always_include_count=always_count,
            always_include_tokens=always_tokens,
            cards_needing_review=sorted(cards_needing_review),
            fixed_hashes=fixed_hashes,
        )