"""Delta-Ingest Engine: tolerant extraction, anchor/immutability validation, and atomic apply."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .indexer import RepoIndex
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


class DeltaParseError(CtxForgeError):
    """Raised when `<ctx_delta>` / `<ctx_bootstrap>` or raw JSON cannot be parsed."""

    def __init__(self, message: str, raw_excerpt: str = "") -> None:
        self.raw_excerpt = raw_excerpt
        suffix = f"\nExcerpt:\n{raw_excerpt}" if raw_excerpt else ""
        super().__init__(f"{message}{suffix}")


class OpKind(StrEnum):
    """Atomic operation types produced by Delta-Ingest planning."""

    CREATE_CARD = "CREATE_CARD"
    SUPERSEDE_CARD = "SUPERSEDE_CARD"
    UPDATE_TASK = "UPDATE_TASK"
    CREATE_TASK = "CREATE_TASK"
    LOG_SESSION = "LOG_SESSION"


class OpStatus(StrEnum):
    """Validation status of a planned operation."""

    READY = "READY"
    BLOCKED_BY_IMMUTABLE = "BLOCKED_BY_IMMUTABLE"
    SKIPPED = "SKIPPED"


@dataclass(slots=True)
class ChangeOp:
    """Single planned Truth-Governed state mutation."""

    kind: OpKind
    target_id: str
    summary: str
    payload: dict[str, Any]
    status: OpStatus = OpStatus.READY
    reason: str = ""
    old_id: str | None = None
    new_id: str | None = None


@dataclass(slots=True)
class DeltaPlan:
    """Validated preview of operations extracted from an LLM `<ctx_delta>` response."""

    operations: list[ChangeOp] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_delta: dict[str, Any] = field(default_factory=dict)

    @property
    def blocked_operations(self) -> list[ChangeOp]:
        return [op for op in self.operations if op.status is OpStatus.BLOCKED_BY_IMMUTABLE]

    @property
    def ready_operations(self) -> list[ChangeOp]:
        return [op for op in self.operations if op.status is OpStatus.READY]

    def format_plan(self) -> str:
        """Render a human-readable diff preview before user confirmation."""
        if not self.operations and not self.warnings:
            return "No state changes detected in delta."

        lines: list[str] = ["=== ctxforge Delta Plan ==="]
        for op in self.operations:
            if op.status is OpStatus.BLOCKED_BY_IMMUTABLE:
                lines.append(
                    f"[!] BLOCKED {op.old_id or op.target_id} (immutable) -> "
                    f"{op.new_id or '?'}: {op.summary}"
                )
            elif op.status is OpStatus.SKIPPED:
                lines.append(f"[-] SKIP {op.target_id}: {op.reason or op.summary}")
            elif op.kind is OpKind.CREATE_CARD:
                lines.append(f"[+] {op.target_id}: {op.summary}")
            elif op.kind is OpKind.SUPERSEDE_CARD:
                lines.append(
                    f"[~] SUPERSEDE {op.old_id} -> {op.new_id}: {op.summary}"
                )
            elif op.kind is OpKind.UPDATE_TASK:
                lines.append(f"[*] UPDATE {op.target_id}: {op.summary}")
            elif op.kind is OpKind.CREATE_TASK:
                lines.append(f"[+] {op.target_id}: {op.summary}")
            elif op.kind is OpKind.LOG_SESSION:
                lines.append(f"[S] LOG {op.target_id}: {op.summary}")

        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")

        return "\n".join(lines)


@dataclass(slots=True)
class IngestSummary:
    """Execution summary after applying a delta plan or bootstrap payload."""

    cards_created: int = 0
    cards_superseded: int = 0
    tasks_created: int = 0
    tasks_updated: int = 0
    sessions_logged: int = 0
    blocked_ops: list[str] = field(default_factory=list)
    skipped_ops: list[str] = field(default_factory=list)
    created_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def format_summary(self) -> str:
        lines = [
            "=== ctxforge Ingest Summary ===",
            (
                f"Cards: +{self.cards_created} created, "
                f"~{self.cards_superseded} superseded | "
                f"Tasks: +{self.tasks_created} created, "
                f"*{self.tasks_updated} updated | "
                f"Sessions: +{self.sessions_logged}"
            ),
        ]
        if self.created_ids:
            lines.append(f"Created IDs: {', '.join(self.created_ids)}")
        if self.blocked_ops:
            lines.append(f"Blocked (immutable): {', '.join(self.blocked_ops)}")
        if self.skipped_ops:
            lines.append(f"Skipped: {', '.join(self.skipped_ops)}")
        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Tolerant JSON Extraction Helpers
# ----------------------------------------------------------------------


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    fence_match = re.match(
        r"^```(?:[A-Za-z0-9_-]+)?\s*\n?([\s\S]*?)\n?```\s*$",
        stripped,
    )
    if fence_match:
        return fence_match.group(1).strip()
    # Also strip leading/trailing orphan fence lines if present.
    stripped = re.sub(r"^```(?:[A-Za-z0-9_-]+)?\s*\n", "", stripped)
    stripped = re.sub(r"\n```\s*$", "", stripped)
    return stripped.strip()


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before `}` or `]` outside JSON string literals."""
    out: list[str] = []
    n = len(text)
    i = 0
    in_string = False

    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue

        if ch == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in ("}", "]"):
                # Drop trailing comma.
                i += 1
                continue

        out.append(ch)
        i += 1

    return "".join(out)


def _excerpt(text: str, limit: int = 240) -> str:
    compact = text.strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def extract_tagged_json(raw_text: str, tag: str = "ctx_delta") -> dict[str, Any]:
    """Extract and tolerantly parse a JSON object from `<tag>...</tag>` or raw JSON.

    Handles surrounding markdown prose/code, takes the last `<tag>...</tag>` block
    if repeated, strips inner markdown code fences, and removes trailing commas
    before `}` / `]` outside string literals.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise DeltaParseError(f"Empty input while extracting <{tag}> JSON.")

    pattern = re.compile(
        rf"<{re.escape(tag)}\b[^>]*>([\s\S]*?)</{re.escape(tag)}>",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(raw_text))

    if matches:
        candidate = matches[-1].group(1)
    else:
        # Check for unclosed opening tag before falling back to raw JSON.
        open_match = re.search(
            rf"<{re.escape(tag)}\b[^>]*>([\s\S]*)$",
            raw_text,
            re.IGNORECASE,
        )
        candidate = open_match.group(1) if open_match else raw_text

    cleaned = _strip_markdown_fences(candidate)
    cleaned = _strip_trailing_commas(cleaned)

    # Fast path: direct JSON object parse.
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        raise DeltaParseError(
            f"Expected JSON object inside <{tag}>, got {type(parsed).__name__}.",
            _excerpt(cleaned),
        )
    except json.JSONDecodeError as first_err:
        # Fallback: locate the first JSON object `{...}` if prose wrapped raw JSON.
        first_brace = cleaned.find("{")
        if first_brace != -1 and not matches:
            decoder = json.JSONDecoder()
            try:
                parsed, _ = decoder.raw_decode(cleaned[first_brace:])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        raise DeltaParseError(
            f"Failed to parse JSON for <{tag}>: {first_err}",
            _excerpt(cleaned),
        ) from first_err


# ----------------------------------------------------------------------
# Entity Normalization & Validation Helpers
# ----------------------------------------------------------------------


def _parse_card_kind(raw_kind: Any) -> CardKind:
    if isinstance(raw_kind, CardKind):
        return raw_kind
    if not isinstance(raw_kind, str):
        raise DeltaParseError(f"Card kind must be a string, got {type(raw_kind).__name__}.")
    norm = raw_kind.strip().upper()
    if norm in ("INV", "INVARIANT"):
        return CardKind.INVARIANT
    if norm == "ADR":
        return CardKind.ADR
    if norm in ("PIT", "PITFALL"):
        return CardKind.PITFALL
    raise DeltaParseError(f"Unsupported card kind '{raw_kind}'.")


def _parse_task_status(raw_status: Any) -> TaskStatus:
    if isinstance(raw_status, TaskStatus):
        return raw_status
    if not isinstance(raw_status, str):
        raise DeltaParseError(
            f"Task status must be a string, got {type(raw_status).__name__}."
        )
    norm = raw_status.strip().upper()
    try:
        return TaskStatus(norm)
    except ValueError as exc:
        raise DeltaParseError(f"Unsupported task status '{raw_status}'.") from exc


def _normalize_rel_path(path: str) -> str:
    norm = path.replace("\\", "/").strip()
    while norm.startswith("./"):
        norm = norm[2:]
    return norm


def _parse_anchor_item(raw_anchor: Any) -> Anchor | None:
    if isinstance(raw_anchor, Anchor):
        return raw_anchor
    if isinstance(raw_anchor, Mapping):
        raw_path = raw_anchor.get("file_path") or raw_anchor.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return None
        symbol = raw_anchor.get("symbol")
        file_hash = raw_anchor.get("file_hash")
        return Anchor(
            file_path=_normalize_rel_path(raw_path),
            symbol=str(symbol).strip() if isinstance(symbol, str) and symbol.strip() else None,
            file_hash=str(file_hash).strip() if isinstance(file_hash, str) and file_hash.strip() else None,
        )
    if isinstance(raw_anchor, str):
        text = raw_anchor.strip()
        if not text:
            return None
        symbol: str | None = None
        path_part = text
        if ":" in text:
            left, right = text.rsplit(":", 1)
            if left.strip() and right.strip() and "/" not in right and "\\" not in right:
                path_part = left
                symbol = right.strip()
        return Anchor(file_path=_normalize_rel_path(path_part), symbol=symbol)
    return None


def _validate_and_hash_anchors(
    raw_anchors: Any,
    *,
    entity_label: str,
    store: ContextStore,
    index: RepoIndex | None,
    warnings: list[str],
) -> list[Anchor]:
    if raw_anchors is None:
        return []
    if not isinstance(raw_anchors, list):
        warnings.append(f"{entity_label}: ignored non-list anchors field.")
        return []

    validated: list[Anchor] = []
    for item in raw_anchors:
        anchor = _parse_anchor_item(item)
        if anchor is None:
            continue

        if index is not None:
            file_rec = index.files.get(anchor.file_path)
            if file_rec is not None:
                anchor.file_hash = file_rec.sha256
                if anchor.symbol and file_rec.symbols:
                    known_symbols = {
                        sym.name for sym in file_rec.symbols
                    } | {
                        f"{sym.parent}.{sym.name}"
                        for sym in file_rec.symbols
                        if sym.parent
                    }
                    if anchor.symbol not in known_symbols:
                        warnings.append(
                            f"{entity_label}: Symbol '{anchor.symbol}' not found in '{anchor.file_path}'."
                        )
            else:
                # Also allow files on disk not yet in index, otherwise warn on hallucinated path.
                disk_hash = store.compute_anchor_hash(anchor.file_path, digest_length=16)
                if disk_hash is not None:
                    anchor.file_hash = disk_hash
                else:
                    warnings.append(
                        f"Unknown anchor path: '{anchor.file_path}' (referenced by {entity_label})."
                    )
        else:
            disk_hash = store.compute_anchor_hash(anchor.file_path, digest_length=16)
            if disk_hash is not None:
                anchor.file_hash = disk_hash
            else:
                warnings.append(
                    f"Unknown anchor path: '{anchor.file_path}' (referenced by {entity_label})."
                )

        validated.append(anchor)

    return validated


class _SequentialIdAllocator:
    """Tracks sequential IDs across multiple planned operations without collisions."""

    def __init__(self, store: ContextStore) -> None:
        self._max: dict[str, int] = {"INV": 0, "ADR": 0, "PIT": 0, "TASK": 0, "SES": 0}
        if store.is_initialized():
            for card in store.list_cards():
                self._observe(card.id)
            for task in store.list_tasks():
                self._observe(task.id)
            for session in store.list_sessions():
                self._observe(session.session_id)

    def _observe(self, item_id: str) -> None:
        m = re.match(r"^([A-Z]+)-(\d+)$", item_id)
        if m:
            prefix, num = m.group(1), int(m.group(2))
            self._max[prefix] = max(self._max.get(prefix, 0), num)

    def next_card_id(self, kind: CardKind) -> str:
        prefix = kind.id_prefix
        nxt = self._max.get(prefix, 0) + 1
        self._max[prefix] = nxt
        return f"{prefix}-{nxt:03d}"

    def next_task_id(self) -> str:
        nxt = self._max.get("TASK", 0) + 1
        self._max["TASK"] = nxt
        return f"TASK-{nxt:03d}"

    def next_session_id(self) -> str:
        nxt = self._max.get("SES", 0) + 1
        self._max["SES"] = nxt
        return f"SES-{nxt:03d}"


# ----------------------------------------------------------------------
# DeltaIngestor
# ----------------------------------------------------------------------


class DeltaIngestor:
    """Validates and applies `<ctx_delta>` and `<ctx_bootstrap>` outputs safely."""

    def __init__(self, store: ContextStore | Path | str) -> None:
        self._store: ContextStore = (
            store if isinstance(store, ContextStore) else ContextStore(store)
        )

    @property
    def store(self) -> ContextStore:
        return self._store

    def build_delta_plan(
        self,
        raw_text: str,
        index: RepoIndex | None = None,
    ) -> DeltaPlan:
        """Parse `<ctx_delta>` and construct a validated `DeltaPlan` without mutating disk."""
        delta = extract_tagged_json(raw_text, tag="ctx_delta")
        warnings: list[str] = []
        operations: list[ChangeOp] = []
        allocator = _SequentialIdAllocator(self._store)

        # 1. Task status update (if concrete task_id provided)
        raw_task_id = delta.get("task_id")
        task_id = (
            raw_task_id.strip()
            if isinstance(raw_task_id, str)
            and raw_task_id.strip()
            and raw_task_id.strip() != "..."
            else None
        )
        raw_task_status = delta.get("task_status")
        next_anchor_raw = delta.get("next_anchor")
        next_anchor = (
            next_anchor_raw.strip()
            if isinstance(next_anchor_raw, str)
            and next_anchor_raw.strip()
            and next_anchor_raw.strip() != "..."
            else ""
        )

        if task_id is not None and isinstance(raw_task_status, str) and raw_task_status.strip():
            new_status = _parse_task_status(raw_task_status)
            existing_task = self._store.get_task(task_id)
            if existing_task is None:
                warnings.append(f"Task '{task_id}' referenced in delta does not exist.")
                operations.append(
                    ChangeOp(
                        kind=OpKind.UPDATE_TASK,
                        target_id=task_id,
                        summary=f"status -> {new_status.value}",
                        payload={"task_id": task_id, "status": new_status.value},
                        status=OpStatus.SKIPPED,
                        reason=f"Task '{task_id}' not found",
                    )
                )
            else:
                operations.append(
                    ChangeOp(
                        kind=OpKind.UPDATE_TASK,
                        target_id=task_id,
                        summary=f"{existing_task.status.value} -> {new_status.value}",
                        payload={
                            "task_id": task_id,
                            "status": new_status.value,
                            "next_step_hint": next_anchor or existing_task.next_step_hint,
                        },
                    )
                )

        # 2. Decisions added
        raw_added = delta.get("decisions_added", [])
        if raw_added is None:
            raw_added = []
        if not isinstance(raw_added, list):
            raise DeltaParseError("'decisions_added' must be a list.")

        for idx, item in enumerate(raw_added):
            if not isinstance(item, Mapping):
                warnings.append(f"decisions_added[{idx}]: expected object, skipped.")
                continue
            kind = _parse_card_kind(item.get("kind", "ADR"))
            title = str(item.get("title", "")).strip()
            statement = str(item.get("statement", "")).strip()
            rationale = str(item.get("rationale", "")).strip()
            if not title or not statement:
                warnings.append(
                    f"decisions_added[{idx}]: missing title or statement, skipped."
                )
                continue

            card_id = allocator.next_card_id(kind)
            tags = [str(t).strip() for t in item.get("tags", []) if str(t).strip()]
            anchors = _validate_and_hash_anchors(
                item.get("anchors", []),
                entity_label=card_id,
                store=self._store,
                index=index,
                warnings=warnings,
            )
            card = KnowledgeCard(
                id=card_id,
                kind=kind,
                title=title,
                statement=statement,
                rationale=rationale,
                immutable=bool(item.get("immutable", False)),
                always_include=bool(item.get("always_include", False)),
                tags=tags,
                anchors=anchors,
            )
            operations.append(
                ChangeOp(
                    kind=OpKind.CREATE_CARD,
                    target_id=card_id,
                    summary=f"{card.title}",
                    payload={"card": card.to_dict()},
                    new_id=card_id,
                )
            )

        # 3. Decisions superseded (Truth Governance + immutability check)
        raw_superseded = delta.get("decisions_superseded", [])
        if raw_superseded is None:
            raw_superseded = []
        if not isinstance(raw_superseded, list):
            raise DeltaParseError("'decisions_superseded' must be a list.")

        for idx, item in enumerate(raw_superseded):
            if not isinstance(item, Mapping):
                warnings.append(f"decisions_superseded[{idx}]: expected object, skipped.")
                continue
            old_id = str(item.get("old_id", "")).strip()
            if not old_id:
                warnings.append(f"decisions_superseded[{idx}]: missing 'old_id', skipped.")
                continue

            replacement_raw = item.get("replacement")
            if not isinstance(replacement_raw, Mapping):
                replacement_raw = item

            old_card = self._store.get_card(old_id)
            if old_card is None:
                warnings.append(f"Cannot supersede missing card '{old_id}'.")
                operations.append(
                    ChangeOp(
                        kind=OpKind.SUPERSEDE_CARD,
                        target_id=old_id,
                        summary=str(replacement_raw.get("title", old_id)),
                        payload={},
                        status=OpStatus.SKIPPED,
                        reason=f"Card '{old_id}' not found",
                        old_id=old_id,
                    )
                )
                continue

            rep_kind = (
                _parse_card_kind(replacement_raw["kind"])
                if replacement_raw.get("kind") is not None
                else old_card.kind
            )
            rep_title = str(replacement_raw.get("title", "")).strip()
            rep_stmt = str(replacement_raw.get("statement", "")).strip()
            rep_rat = str(replacement_raw.get("rationale", "")).strip()
            if not rep_title or not rep_stmt:
                warnings.append(
                    f"decisions_superseded[{idx}] for '{old_id}': missing replacement title/statement, skipped."
                )
                continue

            new_id = allocator.next_card_id(rep_kind)
            tags = [
                str(t).strip()
                for t in replacement_raw.get("tags", old_card.tags)
                if str(t).strip()
            ]
            anchors = _validate_and_hash_anchors(
                replacement_raw.get("anchors", []),
                entity_label=new_id,
                store=self._store,
                index=index,
                warnings=warnings,
            )
            new_card = KnowledgeCard(
                id=new_id,
                kind=rep_kind,
                title=rep_title,
                statement=rep_stmt,
                rationale=rep_rat,
                immutable=bool(replacement_raw.get("immutable", False)),
                always_include=bool(
                    replacement_raw.get("always_include", old_card.always_include)
                ),
                tags=tags,
                anchors=anchors,
            )

            if old_card.status is CardStatus.SUPERSEDED:
                warnings.append(
                    f"Card '{old_id}' is already superseded by '{old_card.superseded_by}'."
                )
                operations.append(
                    ChangeOp(
                        kind=OpKind.SUPERSEDE_CARD,
                        target_id=f"{old_id} -> {new_id}",
                        summary=rep_title,
                        payload={"old_id": old_id, "new_card": new_card.to_dict()},
                        status=OpStatus.SKIPPED,
                        reason=f"Already superseded by {old_card.superseded_by}",
                        old_id=old_id,
                        new_id=new_id,
                    )
                )
                continue

            if old_card.immutable:
                warnings.append(
                    f"Blocked attempt to supersede immutable card '{old_id}' without force_immutable=True."
                )
                operations.append(
                    ChangeOp(
                        kind=OpKind.SUPERSEDE_CARD,
                        target_id=f"{old_id} -> {new_id}",
                        summary=rep_title,
                        payload={"old_id": old_id, "new_card": new_card.to_dict()},
                        status=OpStatus.BLOCKED_BY_IMMUTABLE,
                        reason=f"Card '{old_id}' is immutable",
                        old_id=old_id,
                        new_id=new_id,
                    )
                )
                continue

            operations.append(
                ChangeOp(
                    kind=OpKind.SUPERSEDE_CARD,
                    target_id=f"{old_id} -> {new_id}",
                    summary=rep_title,
                    payload={"old_id": old_id, "new_card": new_card.to_dict()},
                    old_id=old_id,
                    new_id=new_id,
                )
            )

        # 4. New subtasks
        raw_subtasks = delta.get("new_subtasks", [])
        if raw_subtasks is None:
            raw_subtasks = []
        if not isinstance(raw_subtasks, list):
            raise DeltaParseError("'new_subtasks' must be a list.")

        for idx, item in enumerate(raw_subtasks):
            if not isinstance(item, Mapping):
                warnings.append(f"new_subtasks[{idx}]: expected object, skipped.")
                continue
            title = str(item.get("title", "")).strip()
            if not title:
                warnings.append(f"new_subtasks[{idx}]: missing title, skipped.")
                continue
            description = str(item.get("description", title)).strip()
            new_task_id = allocator.next_task_id()
            anchors = _validate_and_hash_anchors(
                item.get("anchors", []),
                entity_label=new_task_id,
                store=self._store,
                index=index,
                warnings=warnings,
            )
            raw_deps = item.get("depends_on", [])
            deps = (
                [str(d).strip() for d in raw_deps if str(d).strip()]
                if isinstance(raw_deps, list)
                else []
            )
            raw_priority = item.get("priority", 2)
            priority = raw_priority if raw_priority in (1, 2, 3) else 2
            criteria = (
                [str(c).strip() for c in item.get("acceptance_criteria", []) if str(c).strip()]
                if isinstance(item.get("acceptance_criteria"), list)
                else []
            )
            task_node = TaskNode(
                id=new_task_id,
                title=title,
                description=description,
                priority=priority,
                depends_on=deps,
                anchors=anchors,
                acceptance_criteria=criteria,
                next_step_hint=str(item.get("next_step_hint", "")).strip(),
            )
            operations.append(
                ChangeOp(
                    kind=OpKind.CREATE_TASK,
                    target_id=new_task_id,
                    summary=title,
                    payload={"task": task_node.to_dict()},
                    new_id=new_task_id,
                )
            )

        # 5. Session log entry (always record continuity when delta carries any signal)
        summary_text = str(delta.get("summary", "")).strip()
        if not summary_text:
            active_count = sum(
                1 for op in operations if op.status is OpStatus.READY
            )
            summary_text = (
                f"Applied delta ({active_count} operations)"
                if active_count
                else (next_anchor or "Delta reviewed with no store mutations")
            )

        session_id = allocator.next_session_id()
        session_entry = SessionLogEntry(
            session_id=session_id,
            task_id=task_id,
            summary=summary_text,
            next_anchor=next_anchor,
        )
        operations.append(
            ChangeOp(
                kind=OpKind.LOG_SESSION,
                target_id=session_id,
                summary=session_entry.summary,
                payload={"session": session_entry.to_dict()},
                new_id=session_id,
            )
        )

        return DeltaPlan(
            operations=operations,
            warnings=warnings,
            raw_delta=delta,
        )

    def apply_plan(
        self,
        plan: DeltaPlan,
        force_immutable: bool = False,
    ) -> IngestSummary:
        """Apply validated operations from `plan` to `ContextStore`."""
        summary = IngestSummary(warnings=list(plan.warnings))
        session_id_for_cards: str | None = None
        for op in plan.operations:
            if op.kind is OpKind.LOG_SESSION:
                session_id_for_cards = op.new_id
                break

        for op in plan.operations:
            if op.status is OpStatus.SKIPPED:
                summary.skipped_ops.append(op.target_id)
                continue

            if op.status is OpStatus.BLOCKED_BY_IMMUTABLE and not force_immutable:
                summary.blocked_ops.append(op.old_id or op.target_id)
                continue

            if op.kind is OpKind.CREATE_CARD:
                card = KnowledgeCard.from_dict(op.payload["card"])
                card.source_session = session_id_for_cards
                if self._store.get_card(card.id) is not None:
                    card.id = self._store.next_card_id(card.kind)
                self._store.add_card(card)
                summary.cards_created += 1
                summary.created_ids.append(card.id)

            elif op.kind is OpKind.SUPERSEDE_CARD:
                old_id = str(op.payload["old_id"])
                new_card = KnowledgeCard.from_dict(op.payload["new_card"])
                new_card.source_session = session_id_for_cards
                if self._store.get_card(new_card.id) is not None:
                    new_card.id = self._store.next_card_id(new_card.kind)
                self._store.supersede_card(
                    old_id,
                    new_card,
                    force_immutable=force_immutable,
                )
                summary.cards_superseded += 1
                summary.created_ids.append(new_card.id)

            elif op.kind is OpKind.UPDATE_TASK:
                task_id = str(op.payload["task_id"])
                task = self._store.require_task(task_id)
                task.status = TaskStatus(op.payload["status"])
                task.next_step_hint = str(op.payload.get("next_step_hint", ""))
                self._store.update_task(task)
                summary.tasks_updated += 1

            elif op.kind is OpKind.CREATE_TASK:
                task = TaskNode.from_dict(op.payload["task"])
                if self._store.get_task(task.id) is not None:
                    task.id = self._store.next_task_id()
                self._store.add_task(task)
                summary.tasks_created += 1
                summary.created_ids.append(task.id)

            elif op.kind is OpKind.LOG_SESSION:
                entry = SessionLogEntry.from_dict(op.payload["session"])
                self._store.append_session(entry)
                summary.sessions_logged += 1

        return summary

    def apply_bootstrap(
        self,
        raw_text: str,
        index: RepoIndex | None = None,
    ) -> IngestSummary:
        """Parse `<ctx_bootstrap>` and initialize ProjectMeta, cards, and task DAG."""
        payload = extract_tagged_json(raw_text, tag="ctx_bootstrap")
        warnings: list[str] = []

        raw_project = payload.get("project")
        if not isinstance(raw_project, Mapping):
            raise DeltaParseError(
                "Bootstrap JSON must include a 'project' object.",
                _excerpt(json.dumps(payload, ensure_ascii=False)),
            )

        proj_name = str(raw_project.get("name", "")).strip()
        proj_vision = str(raw_project.get("vision", "")).strip()
        if not proj_name:
            raise DeltaParseError("Bootstrap 'project.name' must be non-empty.")

        non_goals_raw = raw_project.get("non_goals", [])
        non_goals = (
            [str(g).strip() for g in non_goals_raw if str(g).strip()]
            if isinstance(non_goals_raw, list)
            else []
        )
        tech_notes = str(raw_project.get("tech_stack_notes", "")).strip()

        project_meta = ProjectMeta(
            name=proj_name,
            vision=proj_vision,
            non_goals=non_goals,
            tech_stack_notes=tech_notes,
        )

        if self._store.is_initialized():
            existing_meta = self._store.get_project()
            project_meta.created_at = existing_meta.created_at
            self._store.save_project(project_meta)
        else:
            self._store.init(project_meta)

        summary = IngestSummary()
        session_id = self._store.next_session_id()

        # 1. Cards
        raw_cards = payload.get("cards", [])
        if not isinstance(raw_cards, list):
            raise DeltaParseError("Bootstrap 'cards' must be a list.")

        for idx, item in enumerate(raw_cards):
            if not isinstance(item, Mapping):
                warnings.append(f"bootstrap.cards[{idx}]: expected object, skipped.")
                continue
            kind = _parse_card_kind(item.get("kind", "ADR"))
            title = str(item.get("title", "")).strip()
            statement = str(item.get("statement", "")).strip()
            rationale = str(item.get("rationale", "")).strip()
            if not title or not statement:
                warnings.append(
                    f"bootstrap.cards[{idx}]: missing title or statement, skipped."
                )
                continue

            card_id = self._store.next_card_id(kind)
            tags = [str(t).strip() for t in item.get("tags", []) if str(t).strip()]
            anchors = _validate_and_hash_anchors(
                item.get("anchors", []),
                entity_label=card_id,
                store=self._store,
                index=index,
                warnings=warnings,
            )
            card = KnowledgeCard(
                id=card_id,
                kind=kind,
                title=title,
                statement=statement,
                rationale=rationale,
                immutable=bool(item.get("immutable", False)),
                always_include=bool(item.get("always_include", False)),
                tags=tags,
                anchors=anchors,
                source_session=session_id,
            )
            self._store.add_card(card)
            summary.cards_created += 1
            summary.created_ids.append(card.id)

        # 2. Tasks (remap temporary LLM IDs like T1/T2 to canonical TASK-001/TASK-002)
        raw_tasks = payload.get("tasks", [])
        if not isinstance(raw_tasks, list):
            raise DeltaParseError("Bootstrap 'tasks' must be a list.")

        allocator = _SequentialIdAllocator(self._store)
        temp_id_map: dict[str, str] = {}
        staged_tasks: list[tuple[dict[str, Any], str]] = []

        for idx, item in enumerate(raw_tasks):
            if not isinstance(item, Mapping):
                warnings.append(f"bootstrap.tasks[{idx}]: expected object, skipped.")
                continue
            title = str(item.get("title", "")).strip()
            if not title:
                warnings.append(f"bootstrap.tasks[{idx}]: missing title, skipped.")
                continue
            canonical_id = allocator.next_task_id()
            temp_id = str(item.get("id", "")).strip()
            if temp_id:
                temp_id_map[temp_id] = canonical_id
            staged_tasks.append((dict(item), canonical_id))

        new_task_nodes: list[TaskNode] = []
        existing_task_ids = {t.id for t in self._store.list_tasks()}
        all_known_task_ids = existing_task_ids | set(temp_id_map.values()) | {
            cid for _, cid in staged_tasks
        }

        for item, canonical_id in staged_tasks:
            raw_deps = item.get("depends_on", [])
            remapped_deps: list[str] = []
            if isinstance(raw_deps, list):
                for dep in raw_deps:
                    dep_str = str(dep).strip()
                    if not dep_str:
                        continue
                    resolved_dep = temp_id_map.get(dep_str, dep_str)
                    if resolved_dep in all_known_task_ids:
                        remapped_deps.append(resolved_dep)
                    else:
                        warnings.append(
                            f"Bootstrap task '{canonical_id}': dropped unknown dependency '{dep_str}'."
                        )

            anchors = _validate_and_hash_anchors(
                item.get("anchors", []),
                entity_label=canonical_id,
                store=self._store,
                index=index,
                warnings=warnings,
            )
            raw_priority = item.get("priority", 2)
            priority = raw_priority if raw_priority in (1, 2, 3) else 2
            criteria = (
                [
                    str(c).strip()
                    for c in item.get("acceptance_criteria", [])
                    if str(c).strip()
                ]
                if isinstance(item.get("acceptance_criteria"), list)
                else []
            )
            raw_related = item.get("related_cards", [])
            related = (
                [str(r).strip() for r in raw_related if str(r).strip()]
                if isinstance(raw_related, list)
                else []
            )
            task_node = TaskNode(
                id=canonical_id,
                title=str(item["title"]).strip(),
                description=str(item.get("description", item["title"])).strip(),
                status=_parse_task_status(item.get("status", "TODO")),
                priority=priority,
                depends_on=remapped_deps,
                anchors=anchors,
                related_cards=related,
                acceptance_criteria=criteria,
                next_step_hint=str(item.get("next_step_hint", "")).strip(),
            )
            new_task_nodes.append(task_node)

        if new_task_nodes:
            combined_tasks = self._store.list_tasks() + new_task_nodes
            self._store.save_tasks(combined_tasks, validate_dag=True)
            summary.tasks_created = len(new_task_nodes)
            summary.created_ids.extend(t.id for t in new_task_nodes)

        # 3. Initial bootstrap session log
        first_task = new_task_nodes[0].id if new_task_nodes else None
        first_hint = (
            f"Start {new_task_nodes[0].id}: {new_task_nodes[0].title}"
            if new_task_nodes
            else "Review bootstrapped project cards"
        )
        self._store.append_session(
            SessionLogEntry(
                session_id=session_id,
                task_id=first_task,
                summary=(
                    f"Bootstrap initialized '{project_meta.name}' with "
                    f"{summary.cards_created} cards and {summary.tasks_created} tasks."
                ),
                next_anchor=first_hint,
            )
        )
        summary.sessions_logged = 1
        summary.warnings = warnings
        return summary