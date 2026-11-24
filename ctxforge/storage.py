"""Persistent .ctx/ storage engine with atomic writes and Truth Governance."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

from .models import (
    CardKind,
    CardStatus,
    KnowledgeCard,
    ProjectMeta,
    SessionLogEntry,
    TaskNode,
    TaskStatus,
    utc_now_iso,
)

_GITIGNORE_CONTENT = "cache/\nout/\n"
_TASKS_SCHEMA_VERSION = 1


class CtxForgeError(Exception):
    """Base exception for all ctxforge storage and governance errors."""


class StoreNotInitializedError(CtxForgeError):
    """Raised when attempting to access an uninitialized .ctx/ directory."""


class StoreAlreadyInitializedError(CtxForgeError):
    """Raised when init() is called on an already initialized .ctx/ store."""


class CorruptedStoreError(CtxForgeError):
    """Raised when persisted JSON/JSONL files cannot be parsed or validated."""


class TruthGovernanceError(CtxForgeError):
    """Raised when an operation violates Truth Governance invariants."""


class ImmutableViolationError(TruthGovernanceError):
    """Raised when trying to modify or supersede an immutable card without force."""

    def __init__(self, card_id: str, action: str = "supersede") -> None:
        self.card_id = card_id
        self.action = action
        super().__init__(
            f"Card '{card_id}' is marked immutable and cannot be {action}d "
            f"without force_immutable=True."
        )


class DuplicateIdError(CtxForgeError):
    """Raised when creating an entity with an already existing ID."""


class NotFoundError(CtxForgeError):
    """Base exception when a requested card or task does not exist."""


class CardNotFoundError(NotFoundError):
    """Raised when a KnowledgeCard ID does not exist in storage."""


class TaskNotFoundError(NotFoundError):
    """Raised when a TaskNode ID does not exist in storage."""


class DagError(CtxForgeError):
    """Base exception for task dependency graph validation failures."""


class DagCycleError(DagError):
    """Raised when a cyclic dependency is detected in the task DAG."""

    def __init__(self, cycle: list[str]) -> None:
        self.cycle = list(cycle)
        chain = " -> ".join(self.cycle)
        super().__init__(f"Cyclic dependency detected in task DAG: {chain}")


class MissingDependencyError(DagError):
    """Raised when a task depends on a task ID that does not exist."""

    def __init__(self, task_id: str, missing_dependency_id: str) -> None:
        self.task_id = task_id
        self.missing_dependency_id = missing_dependency_id
        super().__init__(
            f"Task '{task_id}' depends on non-existent task '{missing_dependency_id}'."
        )


class TaskWithDependencies(NamedTuple):
    """Target task paired with its transitive prerequisites in topological order."""

    task: TaskNode
    dependencies: list[TaskNode]


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically write UTF-8 text to `path` using a same-directory temp file."""
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


def _next_prefixed_id(existing_ids: list[str], prefix: str) -> str:
    """Generate the next zero-padded sequential ID such as ADR-001 or TASK-001."""
    pattern = re.compile(rf"^{re.escape(prefix)}-(\d+)$")
    max_num = 0
    for item_id in existing_ids:
        match = pattern.match(item_id)
        if match:
            max_num = max(max_num, int(match.group(1)))
    return f"{prefix}-{max_num + 1:03d}"


class ContextStore:
    """Manages persistent project context inside `<root>/.ctx/`."""

    def __init__(self, project_root: Path | str | None = None) -> None:
        self._root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path.cwd().resolve()
        )
        self._ctx_dir = self._root / ".ctx"

    @property
    def root(self) -> Path:
        """Project root directory."""
        return self._root

    @property
    def ctx_dir(self) -> Path:
        """Path to the `.ctx` metadata directory."""
        return self._ctx_dir

    @property
    def project_file(self) -> Path:
        return self._ctx_dir / "project.json"

    @property
    def cards_file(self) -> Path:
        return self._ctx_dir / "cards.jsonl"

    @property
    def tasks_file(self) -> Path:
        return self._ctx_dir / "tasks.json"

    @property
    def sessions_file(self) -> Path:
        return self._ctx_dir / "sessions.jsonl"

    @property
    def out_dir(self) -> Path:
        return self._ctx_dir / "out"

    @property
    def gitignore_file(self) -> Path:
        return self._ctx_dir / ".gitignore"

    def is_initialized(self) -> bool:
        """Return True if `.ctx/` and `project.json` exist."""
        return self._ctx_dir.is_dir() and self.project_file.is_file()

    def _require_initialized(self) -> None:
        if not self.is_initialized():
            raise StoreNotInitializedError(
                f"ContextStore is not initialized at '{self._ctx_dir}'. Call init() first."
            )

    def init(self, project: ProjectMeta, *, force: bool = False) -> ProjectMeta:
        """Initialize the `.ctx/` directory structure and metadata files."""
        if self.is_initialized() and not force:
            raise StoreAlreadyInitializedError(
                f"ContextStore is already initialized at '{self._ctx_dir}'."
            )

        self._ctx_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        _atomic_write_text(self.gitignore_file, _GITIGNORE_CONTENT)
        _atomic_write_text(
            self.project_file,
            json.dumps(project.to_dict(), ensure_ascii=False, indent=2) + "\n",
        )
        _atomic_write_text(self.cards_file, "")
        _atomic_write_text(
            self.tasks_file,
            json.dumps(
                {"version": _TASKS_SCHEMA_VERSION, "tasks": []},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
        _atomic_write_text(self.sessions_file, "")
        return project

    # ------------------------------------------------------------------
    # Project Metadata
    # ------------------------------------------------------------------

    def get_project(self) -> ProjectMeta:
        """Load project metadata from `.ctx/project.json`."""
        self._require_initialized()
        try:
            raw = self.project_file.read_text(encoding="utf-8")
            data = json.loads(raw)
            return ProjectMeta.from_dict(data)
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            raise CorruptedStoreError(
                f"Failed to read project metadata from '{self.project_file}': {exc}"
            ) from exc

    def save_project(self, project: ProjectMeta) -> ProjectMeta:
        """Persist updated project metadata atomically."""
        self._require_initialized()
        updated = ProjectMeta(
            name=project.name,
            vision=project.vision,
            non_goals=list(project.non_goals),
            tech_stack_notes=project.tech_stack_notes,
            created_at=project.created_at,
            updated_at=utc_now_iso(),
        )
        _atomic_write_text(
            self.project_file,
            json.dumps(updated.to_dict(), ensure_ascii=False, indent=2) + "\n",
        )
        return updated

    # ------------------------------------------------------------------
    # Knowledge Cards & Truth Governance
    # ------------------------------------------------------------------

    def _load_cards(self) -> list[KnowledgeCard]:
        self._require_initialized()
        if not self.cards_file.exists():
            return []

        cards: list[KnowledgeCard] = []
        try:
            content = self.cards_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise CorruptedStoreError(
                f"Failed to read cards file '{self.cards_file}': {exc}"
            ) from exc

        for line_no, raw_line in enumerate(content.splitlines(), start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
                cards.append(KnowledgeCard.from_dict(payload))
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                raise CorruptedStoreError(
                    f"Invalid card record in '{self.cards_file}' at line {line_no}: {exc}"
                ) from exc
        return cards

    def _save_cards(self, cards: list[KnowledgeCard]) -> None:
        lines = [
            json.dumps(card.to_dict(), ensure_ascii=False)
            for card in cards
        ]
        payload = ("\n".join(lines) + "\n") if lines else ""
        _atomic_write_text(self.cards_file, payload)

    def list_cards(
        self,
        *,
        kind: CardKind | None = None,
        status: CardStatus | None = None,
    ) -> list[KnowledgeCard]:
        """Return all stored cards (including historical), optionally filtered."""
        cards = self._load_cards()
        if kind is not None:
            cards = [c for c in cards if c.kind is kind]
        if status is not None:
            cards = [c for c in cards if c.status is status]
        return cards

    def get_card(self, card_id: str) -> KnowledgeCard | None:
        """Return a card by ID, or None if not found."""
        for card in self._load_cards():
            if card.id == card_id:
                return card
        return None

    def require_card(self, card_id: str) -> KnowledgeCard:
        """Return a card by ID or raise CardNotFoundError."""
        card = self.get_card(card_id)
        if card is None:
            raise CardNotFoundError(f"Card '{card_id}' not found.")
        return card

    def get_active_cards(self, kind: CardKind | None = None) -> list[KnowledgeCard]:
        """Return ONLY ACTIVE cards so superseded/stale decisions never enter prompts."""
        return self.list_cards(kind=kind, status=CardStatus.ACTIVE)

    def next_card_id(self, kind: CardKind) -> str:
        """Generate the next sequential card ID for `kind` (never reusing old IDs)."""
        cards = self._load_cards()
        prefix = kind.id_prefix
        return _next_prefixed_id([card.id for card in cards], prefix)

    def add_card(self, card: KnowledgeCard) -> KnowledgeCard:
        """Add a new knowledge card after validating ID uniqueness."""
        cards = self._load_cards()
        if any(existing.id == card.id for existing in cards):
            raise DuplicateIdError(f"Card ID '{card.id}' already exists.")
        cards.append(card)
        self._save_cards(cards)
        return card

    def supersede_card(
        self,
        old_id: str,
        new_card: KnowledgeCard,
        force_immutable: bool = False,
    ) -> None:
        """Replace `old_id` with `new_card` while preserving full decision history.

        Raises:
            CardNotFoundError: If `old_id` does not exist.
            ImmutableViolationError: If `old_id` is immutable and `force_immutable` is False.
            TruthGovernanceError: If `old_id` is already superseded.
            DuplicateIdError: If `new_card.id` already exists in the store.
        """
        cards = self._load_cards()
        old_index: int | None = None
        for idx, card in enumerate(cards):
            if card.id == old_id:
                old_index = idx
                break

        if old_index is None:
            raise CardNotFoundError(f"Cannot supersede missing card '{old_id}'.")

        old_card = cards[old_index]
        if old_card.immutable and not force_immutable:
            raise ImmutableViolationError(old_id, action="supersede")

        if old_card.status is CardStatus.SUPERSEDED:
            raise TruthGovernanceError(
                f"Card '{old_id}' is already superseded by '{old_card.superseded_by}'. "
                f"Supersede the active successor instead to avoid branching history."
            )

        if new_card.id == old_id or any(card.id == new_card.id for card in cards):
            raise DuplicateIdError(
                f"Replacement card ID '{new_card.id}' already exists in the store."
            )

        old_card.status = CardStatus.SUPERSEDED
        old_card.superseded_by = new_card.id
        old_card.updated_at = utc_now_iso()

        cards.append(new_card)
        self._save_cards(cards)

    def set_card_status(
        self,
        card_id: str,
        status: CardStatus,
        *,
        force_immutable: bool = False,
    ) -> KnowledgeCard:
        """Transition a card to DEPRECATED, ARCHIVED, or ACTIVE with immutability guard."""
        if status is CardStatus.SUPERSEDED:
            raise TruthGovernanceError(
                "Use supersede_card() to transition a card to SUPERSEDED."
            )

        cards = self._load_cards()
        target: KnowledgeCard | None = None
        for card in cards:
            if card.id == card_id:
                target = card
                break

        if target is None:
            raise CardNotFoundError(f"Card '{card_id}' not found.")

        if target.immutable and status is not target.status and not force_immutable:
            raise ImmutableViolationError(card_id, action=f"change status to {status.value}")

        target.status = status
        if status is CardStatus.ACTIVE:
            target.superseded_by = None
        target.updated_at = utc_now_iso()
        self._save_cards(cards)
        return target

    def get_supersession_chain(self, card_id: str) -> list[KnowledgeCard]:
        """Follow `superseded_by` links starting from `card_id`."""
        cards_by_id = {card.id: card for card in self._load_cards()}
        if card_id not in cards_by_id:
            raise CardNotFoundError(f"Card '{card_id}' not found.")

        chain: list[KnowledgeCard] = []
        visited: set[str] = set()
        current_id: str | None = card_id
        while current_id is not None:
            if current_id in visited:
                raise CorruptedStoreError(
                    f"Cyclic supersession reference detected at card '{current_id}'."
                )
            visited.add(current_id)
            card = cards_by_id.get(current_id)
            if card is None:
                raise CorruptedStoreError(
                    f"Broken supersession link: referenced card '{current_id}' is missing."
                )
            chain.append(card)
            current_id = card.superseded_by
        return chain

    # ------------------------------------------------------------------
    # Tasks & DAG Validation
    # ------------------------------------------------------------------

    def _load_tasks(self) -> list[TaskNode]:
        self._require_initialized()
        if not self.tasks_file.exists():
            return []

        try:
            raw = self.tasks_file.read_text(encoding="utf-8")
            payload = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise CorruptedStoreError(
                f"Failed to read tasks file '{self.tasks_file}': {exc}"
            ) from exc

        if isinstance(payload, dict):
            raw_tasks = payload.get("tasks")
        elif isinstance(payload, list):
            raw_tasks = payload
        else:
            raise CorruptedStoreError(
                f"Invalid tasks schema in '{self.tasks_file}': expected object or list."
            )

        if not isinstance(raw_tasks, list):
            raise CorruptedStoreError(
                f"Invalid 'tasks' field in '{self.tasks_file}': expected list."
            )

        tasks: list[TaskNode] = []
        for idx, item in enumerate(raw_tasks):
            try:
                tasks.append(TaskNode.from_dict(item))
            except (ValueError, TypeError) as exc:
                raise CorruptedStoreError(
                    f"Invalid task entry at index {idx} in '{self.tasks_file}': {exc}"
                ) from exc
        return tasks

    def _save_tasks(self, tasks: list[TaskNode]) -> None:
        payload: dict[str, Any] = {
            "version": _TASKS_SCHEMA_VERSION,
            "tasks": [task.to_dict() for task in tasks],
        }
        _atomic_write_text(
            self.tasks_file,
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        )

    def list_tasks(self, *, status: TaskStatus | None = None) -> list[TaskNode]:
        """Return all stored tasks, optionally filtered by status."""
        tasks = self._load_tasks()
        if status is not None:
            tasks = [t for t in tasks if t.status is status]
        return tasks

    def get_task(self, task_id: str) -> TaskNode | None:
        """Return a task by ID, or None if not found."""
        for task in self._load_tasks():
            if task.id == task_id:
                return task
        return None

    def require_task(self, task_id: str) -> TaskNode:
        """Return a task by ID or raise TaskNotFoundError."""
        task = self.get_task(task_id)
        if task is None:
            raise TaskNotFoundError(f"Task '{task_id}' not found.")
        return task

    def next_task_id(self) -> str:
        """Generate the next sequential task ID (e.g. TASK-001)."""
        tasks = self._load_tasks()
        return _next_prefixed_id([task.id for task in tasks], "TASK")

    def add_task(self, task: TaskNode, *, validate_dag: bool = True) -> TaskNode:
        """Add a task node and optionally validate DAG integrity before saving."""
        tasks = self._load_tasks()
        if any(existing.id == task.id for existing in tasks):
            raise DuplicateIdError(f"Task ID '{task.id}' already exists.")
        updated = [*tasks, task]
        if validate_dag:
            self.validate_task_dag(updated)
        self._save_tasks(updated)
        return task

    def update_task(self, task: TaskNode, *, validate_dag: bool = True) -> TaskNode:
        """Replace an existing task by ID and optionally validate DAG integrity."""
        tasks = self._load_tasks()
        updated: list[TaskNode] = []
        found = False
        for existing in tasks:
            if existing.id == task.id:
                task.updated_at = utc_now_iso()
                updated.append(task)
                found = True
            else:
                updated.append(existing)
        if not found:
            raise TaskNotFoundError(f"Task '{task.id}' not found.")
        if validate_dag:
            self.validate_task_dag(updated)
        self._save_tasks(updated)
        return task

    def save_tasks(self, tasks: list[TaskNode], *, validate_dag: bool = True) -> None:
        """Persist a complete task list, validating DAG integrity by default."""
        self._require_initialized()
        if validate_dag:
            self.validate_task_dag(tasks)
        self._save_tasks(tasks)

    def validate_task_dag(self, tasks: list[TaskNode] | None = None) -> list[str]:
        """Validate task DAG for duplicate IDs, missing dependencies, and cycles.

        Returns:
            Topologically ordered task IDs where dependencies precede dependents.

        Raises:
            DuplicateIdError: If duplicate task IDs exist.
            MissingDependencyError: If any `depends_on` points to a missing task ID.
            DagCycleError: If any cycle (including self-dependency) exists.
        """
        task_list = self._load_tasks() if tasks is None else tasks

        task_map: dict[str, TaskNode] = {}
        for task in task_list:
            if task.id in task_map:
                raise DuplicateIdError(f"Duplicate task ID '{task.id}' in task list.")
            task_map[task.id] = task

        for task in task_list:
            for dep_id in task.depends_on:
                if dep_id not in task_map:
                    raise MissingDependencyError(task.id, dep_id)

        unvisited, visiting, visited = 0, 1, 2
        state: dict[str, int] = {task.id: unvisited for task in task_list}
        topo_order: list[str] = []

        for root_task in task_list:
            if state[root_task.id] != unvisited:
                continue

            stack: list[tuple[str, int]] = [(root_task.id, 0)]
            state[root_task.id] = visiting
            path: list[str] = [root_task.id]

            while stack:
                current_id, dep_idx = stack[-1]
                deps = task_map[current_id].depends_on
                if dep_idx < len(deps):
                    stack[-1] = (current_id, dep_idx + 1)
                    dep_id = deps[dep_idx]
                    dep_state = state[dep_id]
                    if dep_state == visiting:
                        cycle_start = path.index(dep_id)
                        raise DagCycleError(path[cycle_start:] + [dep_id])
                    if dep_state == unvisited:
                        state[dep_id] = visiting
                        path.append(dep_id)
                        stack.append((dep_id, 0))
                else:
                    stack.pop()
                    path.pop()
                    state[current_id] = visited
                    topo_order.append(current_id)

        return topo_order

    def get_task_with_deps(self, task_id: str) -> TaskWithDependencies:
        """Return `task_id` along with all transitive dependencies in topological order."""
        tasks = self._load_tasks()
        self.validate_task_dag(tasks)
        task_map = {task.id: task for task in tasks}

        target = task_map.get(task_id)
        if target is None:
            raise TaskNotFoundError(f"Task '{task_id}' not found.")

        visited: set[str] = set()
        ordered_deps: list[TaskNode] = []

        for direct_dep_id in target.depends_on:
            if direct_dep_id in visited:
                continue
            visited.add(direct_dep_id)
            stack: list[tuple[str, int]] = [(direct_dep_id, 0)]
            while stack:
                curr_id, idx = stack[-1]
                curr_task = task_map[curr_id]
                if idx < len(curr_task.depends_on):
                    stack[-1] = (curr_id, idx + 1)
                    parent_id = curr_task.depends_on[idx]
                    if parent_id not in visited:
                        visited.add(parent_id)
                        stack.append((parent_id, 0))
                else:
                    stack.pop()
                    ordered_deps.append(curr_task)

        return TaskWithDependencies(task=target, dependencies=ordered_deps)

    # ------------------------------------------------------------------
    # Session Handoff Log
    # ------------------------------------------------------------------

    def _load_sessions(self) -> list[SessionLogEntry]:
        self._require_initialized()
        if not self.sessions_file.exists():
            return []

        try:
            content = self.sessions_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise CorruptedStoreError(
                f"Failed to read sessions file '{self.sessions_file}': {exc}"
            ) from exc

        entries: list[SessionLogEntry] = []
        for line_no, raw_line in enumerate(content.splitlines(), start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
                entries.append(SessionLogEntry.from_dict(payload))
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                raise CorruptedStoreError(
                    f"Invalid session record in '{self.sessions_file}' at line {line_no}: {exc}"
                ) from exc
        return entries

    def append_session(self, entry: SessionLogEntry) -> SessionLogEntry:
        """Append a session handoff entry atomically to `.ctx/sessions.jsonl`."""
        entries = self._load_sessions()
        entries.append(entry)
        lines = [json.dumps(item.to_dict(), ensure_ascii=False) for item in entries]
        _atomic_write_text(self.sessions_file, "\n".join(lines) + "\n")
        return entry

    def list_sessions(
        self,
        *,
        task_id: str | None = None,
        limit: int | None = None,
    ) -> list[SessionLogEntry]:
        """Return chronological session log entries, optionally filtered by task or limit."""
        entries = self._load_sessions()
        if task_id is not None:
            entries = [e for e in entries if e.task_id == task_id]
        if limit is not None:
            if limit <= 0:
                return []
            entries = entries[-limit:]
        return entries

    def next_session_id(self) -> str:
        """Generate the next sequential session ID (e.g. SES-001)."""
        entries = self._load_sessions()
        return _next_prefixed_id([e.session_id for e in entries], "SES")

    # ------------------------------------------------------------------
    # Anchor Verification Helper
    # ------------------------------------------------------------------

    def compute_anchor_hash(
        self,
        relative_posix_path: str,
        *,
        digest_length: int = 12,
    ) -> str | None:
        """Compute a short SHA-256 hash of a file within `self.root`, or None if absent."""
        if digest_length <= 0:
            raise ValueError("digest_length must be positive.")

        candidate = (self._root / relative_posix_path).resolve()
        if not candidate.is_relative_to(self._root):
            raise ValueError(
                f"Anchor path '{relative_posix_path}' escapes project root '{self._root}'."
            )
        if not candidate.is_file():
            return None

        hasher = hashlib.sha256()
        with candidate.open("rb") as handle:
            while chunk := handle.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()[:digest_length]