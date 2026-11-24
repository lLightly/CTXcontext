"""Strict stdlib dataclasses and enums for ctxforge domain entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _require_mapping(data: Any, entity_name: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise TypeError(f"{entity_name}.from_dict expects a mapping, got {type(data).__name__}.")
    return data


def _require_non_empty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _require_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string, got {type(value).__name__}.")
    return value


def _optional_str(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or None, got {type(value).__name__}.")
    return value


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a bool, got {type(value).__name__}.")
    return value


def _require_str_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings.")
    return list(value)


class CardKind(StrEnum):
    """Classification of project knowledge cards."""

    INVARIANT = "INV"
    ADR = "ADR"
    PITFALL = "PITFALL"

    @property
    def id_prefix(self) -> str:
        """Return the canonical ID prefix for this card kind."""
        if self is CardKind.PITFALL:
            return "PIT"
        return self.value


class CardStatus(StrEnum):
    """Lifecycle status of a knowledge card under Truth Governance."""

    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    DEPRECATED = "DEPRECATED"
    ARCHIVED = "ARCHIVED"


class TaskStatus(StrEnum):
    """Execution status of a task node in the project DAG."""

    TODO = "TODO"
    IN_PROGRESS = "IN_PROGRESS"
    BLOCKED = "BLOCKED"
    DONE = "DONE"
    CANCELLED = "CANCELLED"


@dataclass(slots=True)
class Anchor:
    """Reference to a specific codebase file and optional symbol/hash."""

    file_path: str
    symbol: str | None = None
    file_hash: str | None = None

    def __post_init__(self) -> None:
        raw_path = _require_non_empty_str(self.file_path, "Anchor.file_path")
        normalized = raw_path.replace("\\", "/").strip()
        while normalized.startswith("./"):
            normalized = normalized[2:]
        if not normalized:
            raise ValueError("Anchor.file_path must not be empty after normalization.")
        self.file_path = normalized
        self.symbol = _optional_str(self.symbol, "Anchor.symbol")
        self.file_hash = _optional_str(self.file_hash, "Anchor.file_hash")

    def to_dict(self) -> dict[str, Any]:
        """Serialize anchor to a JSON-compatible dictionary."""
        return {
            "file_path": self.file_path,
            "symbol": self.symbol,
            "file_hash": self.file_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Anchor:
        """Deserialize an Anchor from a mapping."""
        mapping = _require_mapping(data, "Anchor")
        if "file_path" not in mapping:
            raise ValueError("Anchor dictionary is missing required key 'file_path'.")
        return cls(
            file_path=mapping["file_path"],
            symbol=mapping.get("symbol"),
            file_hash=mapping.get("file_hash"),
        )


def _coerce_anchors(raw_anchors: Any, field_name: str) -> list[Anchor]:
    if not isinstance(raw_anchors, list):
        raise ValueError(f"{field_name} must be a list of Anchor instances or dicts.")
    result: list[Anchor] = []
    for item in raw_anchors:
        if isinstance(item, Anchor):
            result.append(item)
        elif isinstance(item, Mapping):
            result.append(Anchor.from_dict(item))
        else:
            raise ValueError(
                f"{field_name} items must be Anchor or dict, got {type(item).__name__}."
            )
    return result


@dataclass(slots=True)
class KnowledgeCard:
    """Immutable-capable architectural decision, invariant, or pitfall record."""

    id: str
    kind: CardKind
    title: str
    statement: str
    rationale: str
    status: CardStatus = CardStatus.ACTIVE
    immutable: bool = False
    always_include: bool = False
    tags: list[str] = field(default_factory=list)
    anchors: list[Anchor] = field(default_factory=list)
    superseded_by: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    source_session: str | None = None

    def __post_init__(self) -> None:
        self.id = _require_non_empty_str(self.id, "KnowledgeCard.id").strip()
        if not isinstance(self.kind, CardKind):
            self.kind = CardKind(self.kind)
        self.title = _require_non_empty_str(self.title, "KnowledgeCard.title")
        self.statement = _require_non_empty_str(self.statement, "KnowledgeCard.statement")
        self.rationale = _require_str(self.rationale, "KnowledgeCard.rationale")
        if not isinstance(self.status, CardStatus):
            self.status = CardStatus(self.status)
        self.immutable = _require_bool(self.immutable, "KnowledgeCard.immutable")
        self.always_include = _require_bool(
            self.always_include, "KnowledgeCard.always_include"
        )
        self.tags = _require_str_list(self.tags, "KnowledgeCard.tags")
        self.anchors = _coerce_anchors(self.anchors, "KnowledgeCard.anchors")
        self.superseded_by = _optional_str(
            self.superseded_by, "KnowledgeCard.superseded_by"
        )
        self.created_at = _require_non_empty_str(
            self.created_at, "KnowledgeCard.created_at"
        )
        self.updated_at = _require_non_empty_str(
            self.updated_at, "KnowledgeCard.updated_at"
        )
        self.source_session = _optional_str(
            self.source_session, "KnowledgeCard.source_session"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize card to a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "kind": self.kind.value,
            "title": self.title,
            "statement": self.statement,
            "rationale": self.rationale,
            "status": self.status.value,
            "immutable": self.immutable,
            "always_include": self.always_include,
            "tags": list(self.tags),
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "superseded_by": self.superseded_by,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_session": self.source_session,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> KnowledgeCard:
        """Deserialize a KnowledgeCard from a mapping."""
        mapping = _require_mapping(data, "KnowledgeCard")
        for required_key in ("id", "kind", "title", "statement", "rationale"):
            if required_key not in mapping:
                raise ValueError(
                    f"KnowledgeCard dictionary is missing required key '{required_key}'."
                )

        now = utc_now_iso()
        return cls(
            id=mapping["id"],
            kind=CardKind(mapping["kind"]),
            title=mapping["title"],
            statement=mapping["statement"],
            rationale=mapping["rationale"],
            status=CardStatus(mapping.get("status", CardStatus.ACTIVE.value)),
            immutable=mapping.get("immutable", False),
            always_include=mapping.get("always_include", False),
            tags=mapping.get("tags", []),
            anchors=mapping.get("anchors", []),
            superseded_by=mapping.get("superseded_by"),
            created_at=mapping.get("created_at", now),
            updated_at=mapping.get("updated_at", now),
            source_session=mapping.get("source_session"),
        )


@dataclass(slots=True)
class TaskNode:
    """Task entity within the project dependency DAG."""

    id: str
    title: str
    description: str
    status: TaskStatus = TaskStatus.TODO
    priority: int = 2
    depends_on: list[str] = field(default_factory=list)
    anchors: list[Anchor] = field(default_factory=list)
    related_cards: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    next_step_hint: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        self.id = _require_non_empty_str(self.id, "TaskNode.id").strip()
        self.title = _require_non_empty_str(self.title, "TaskNode.title")
        self.description = _require_str(self.description, "TaskNode.description")
        if not isinstance(self.status, TaskStatus):
            self.status = TaskStatus(self.status)
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ValueError("TaskNode.priority must be an integer (1, 2, or 3).")
        if self.priority not in (1, 2, 3):
            raise ValueError(f"TaskNode.priority must be 1, 2, or 3, got {self.priority}.")
        self.depends_on = _require_str_list(self.depends_on, "TaskNode.depends_on")
        self.anchors = _coerce_anchors(self.anchors, "TaskNode.anchors")
        self.related_cards = _require_str_list(self.related_cards, "TaskNode.related_cards")
        self.acceptance_criteria = _require_str_list(
            self.acceptance_criteria, "TaskNode.acceptance_criteria"
        )
        self.next_step_hint = _require_str(self.next_step_hint, "TaskNode.next_step_hint")
        self.created_at = _require_non_empty_str(self.created_at, "TaskNode.created_at")
        self.updated_at = _require_non_empty_str(self.updated_at, "TaskNode.updated_at")

    def to_dict(self) -> dict[str, Any]:
        """Serialize task node to a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority,
            "depends_on": list(self.depends_on),
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "related_cards": list(self.related_cards),
            "acceptance_criteria": list(self.acceptance_criteria),
            "next_step_hint": self.next_step_hint,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TaskNode:
        """Deserialize a TaskNode from a mapping."""
        mapping = _require_mapping(data, "TaskNode")
        for required_key in ("id", "title", "description"):
            if required_key not in mapping:
                raise ValueError(
                    f"TaskNode dictionary is missing required key '{required_key}'."
                )

        now = utc_now_iso()
        return cls(
            id=mapping["id"],
            title=mapping["title"],
            description=mapping["description"],
            status=TaskStatus(mapping.get("status", TaskStatus.TODO.value)),
            priority=mapping.get("priority", 2),
            depends_on=mapping.get("depends_on", []),
            anchors=mapping.get("anchors", []),
            related_cards=mapping.get("related_cards", []),
            acceptance_criteria=mapping.get("acceptance_criteria", []),
            next_step_hint=mapping.get("next_step_hint", ""),
            created_at=mapping.get("created_at", now),
            updated_at=mapping.get("updated_at", now),
        )


@dataclass(slots=True)
class ProjectMeta:
    """High-level project metadata stored in .ctx/project.json."""

    name: str
    vision: str
    non_goals: list[str] = field(default_factory=list)
    tech_stack_notes: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        self.name = _require_non_empty_str(self.name, "ProjectMeta.name")
        self.vision = _require_str(self.vision, "ProjectMeta.vision")
        self.non_goals = _require_str_list(self.non_goals, "ProjectMeta.non_goals")
        self.tech_stack_notes = _require_str(
            self.tech_stack_notes, "ProjectMeta.tech_stack_notes"
        )
        self.created_at = _require_non_empty_str(self.created_at, "ProjectMeta.created_at")
        self.updated_at = _require_non_empty_str(self.updated_at, "ProjectMeta.updated_at")

    def to_dict(self) -> dict[str, Any]:
        """Serialize project metadata to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "vision": self.vision,
            "non_goals": list(self.non_goals),
            "tech_stack_notes": self.tech_stack_notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProjectMeta:
        """Deserialize ProjectMeta from a mapping."""
        mapping = _require_mapping(data, "ProjectMeta")
        for required_key in ("name", "vision"):
            if required_key not in mapping:
                raise ValueError(
                    f"ProjectMeta dictionary is missing required key '{required_key}'."
                )

        now = utc_now_iso()
        return cls(
            name=mapping["name"],
            vision=mapping["vision"],
            non_goals=mapping.get("non_goals", []),
            tech_stack_notes=mapping.get("tech_stack_notes", ""),
            created_at=mapping.get("created_at", now),
            updated_at=mapping.get("updated_at", now),
        )


@dataclass(slots=True)
class SessionLogEntry:
    """Handoff log entry capturing session progress and next anchor."""

    session_id: str
    task_id: str | None = None
    summary: str = ""
    next_anchor: str = ""
    timestamp: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        self.session_id = _require_non_empty_str(
            self.session_id, "SessionLogEntry.session_id"
        ).strip()
        self.task_id = _optional_str(self.task_id, "SessionLogEntry.task_id")
        self.summary = _require_non_empty_str(self.summary, "SessionLogEntry.summary")
        self.next_anchor = _require_str(
            self.next_anchor, "SessionLogEntry.next_anchor"
        )
        self.timestamp = _require_non_empty_str(
            self.timestamp, "SessionLogEntry.timestamp"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize session log entry to a JSON-compatible dictionary."""
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "summary": self.summary,
            "next_anchor": self.next_anchor,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SessionLogEntry:
        """Deserialize a SessionLogEntry from a mapping."""
        mapping = _require_mapping(data, "SessionLogEntry")
        for required_key in ("session_id", "summary"):
            if required_key not in mapping:
                raise ValueError(
                    f"SessionLogEntry dictionary is missing required key '{required_key}'."
                )

        return cls(
            session_id=mapping["session_id"],
            task_id=mapping.get("task_id"),
            summary=mapping["summary"],
            next_anchor=mapping.get("next_anchor", ""),
            timestamp=mapping.get("timestamp", utc_now_iso()),
        )