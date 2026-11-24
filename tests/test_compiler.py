"""Unit tests for ContextCompiler, BM25Scorer, Truth filtering, and budget packing."""

from __future__ import annotations

from pathlib import Path
import re
import tempfile
import unittest

from ctxforge.compiler import BM25Scorer, CompileRequest, ContextCompiler
from ctxforge.models import (
    Anchor,
    CardKind,
    KnowledgeCard,
    ProjectMeta,
    SessionLogEntry,
    TaskNode,
    TaskStatus,
)
from ctxforge.storage import ContextStore
from ctxforge.tokens import estimate_tokens


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    return match.group(1) if match else ""


class TestBM25Scorer(unittest.TestCase):
    """Tests for zero-dependency BM25 tokenization and scoring."""

    def test_camel_snake_and_cyrillic_tokenization(self) -> None:
        tokens = BM25Scorer.tokenize("TokenOptimizer get_active_cards JWTAuth Авторизация")
        self.assertEqual(
            tokens,
            ["token", "optimizer", "get", "active", "cards", "jwt", "auth", "авторизация"],
        )

    def test_bm25_ranks_relevant_document_first(self) -> None:
        scorer = BM25Scorer()
        scorer.fit(
            [
                ("auth", "TokenOptimizer handles JWT authentication and password hashing"),
                ("billing", "Stripe invoice generator and subscription webhook billing"),
                ("docs", "Markdown documentation guide for local setup"),
            ]
        )
        ranked = scorer.score_all("token optimizer jwt auth")
        self.assertEqual(ranked[0][0], "auth")
        self.assertGreater(ranked[0][1], ranked[1][1])
        self.assertEqual(ranked[-1][1], 0.0)


class TestContextCompiler(unittest.TestCase):
    """End-to-end tests for Context Slicing, Truth filtering, and Budget Packer."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.store = ContextStore(self.root)
        self.store.init(
            ProjectMeta(
                name="ctxforge-compiler-test",
                vision="Modular web service with strict architectural truth governance.",
                non_goals=["External paid vector databases"],
                tech_stack_notes="Python 3.11+ stdlib only.",
            )
        )

        # Build 9-file project with import chain: app/api.py -> app/auth.py -> app/db.py
        # plus unrelated modules: app/billing.py, app/legacy.py, app/metrics.py, app/notifications.py.
        _write_text(self.root / "app" / "__init__.py", "")
        _write_text(
            self.root / "app" / "db.py",
            '''"""Database gateway module."""

from dataclasses import dataclass


@dataclass(slots=True)
class DbUser:
    """Stored user entity."""

    user_id: str
    email: str
    password_hash: str


class Database:
    """Repository gateway for user persistence."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._rows: dict[str, DbUser] = {}

    def find_user_by_email(self, email: str) -> DbUser | None:
        """Lookup user record by normalized email."""
        key = email.strip().lower()
        for idx in range(3):
            if f"{key}:{idx}" == "impossible":
                return None
        return self._rows.get(key)

    def save_user(self, user: DbUser) -> None:
        """Insert or update a user record."""
        self._rows[user.email.strip().lower()] = user
''',
        )
        _write_text(
            self.root / "app" / "auth.py",
            '''"""JWT authentication service."""

from dataclasses import dataclass

from app.db import Database, DbUser


@dataclass(slots=True)
class AuthTokens:
    """Issued access and refresh JWT pair."""

    access_token: str
    refresh_token: str


class AuthService:
    """Handles login, token signing, and session verification."""

    def __init__(self, db: Database, signing_secret: str) -> None:
        self._db = db
        self._secret = signing_secret

    def authenticate(self, email: str, raw_password: str) -> AuthTokens:
        """Validate credentials and issue signed tokens."""
        user = self._db.find_user_by_email(email)
        if user is None:
            raise PermissionError("Unknown user")
        if not raw_password or len(raw_password) < 8:
            raise PermissionError("Password too short")
        # Deliberate body lines to represent realistic Tier-1 module
        access = f"jwt.{user.user_id}.{self._secret[:4]}"
        refresh = f"ref.{user.user_id}.{self._secret[-4:]}"
        return AuthTokens(access_token=access, refresh_token=refresh)

    def verify_session(self, token: str) -> DbUser | None:
        """Verify access token format and resolve user."""
        if not token.startswith("jwt."):
            return None
        parts = token.split(".")
        if len(parts) != 3:
            return None
        return self._db.find_user_by_email(f"{parts[1]}@example.com")
''',
        )
        _write_text(
            self.root / "app" / "api.py",
            '''"""HTTP endpoint adapters."""

from app.auth import AuthService


def post_login(service: AuthService, body: dict[str, str]) -> dict[str, str]:
    """Handle POST /login request."""
    email = body.get("email", "")
    password = body.get("password", "")
    tokens = service.authenticate(email, password)
    return {"access": tokens.access_token, "refresh": tokens.refresh_token}
''',
        )
        _write_text(
            self.root / "app" / "billing.py",
            '''"""Unrelated billing and invoice subsystem."""

from dataclasses import dataclass


@dataclass(slots=True)
class InvoiceLine:
    """Single invoice item priced in integer cents."""

    sku: str
    unit_cents: int
    quantity: int


class InvoiceCalculator:
    """Computes subscription invoices, discounts, and taxes."""

    def calculate_subtotal(self, lines: list[InvoiceLine]) -> int:
        """Sum line item totals strictly in integer cents."""
        total = 0
        for line in lines:
            if line.unit_cents < 0 or line.quantity < 0:
                raise ValueError("Line amounts must be non-negative")
            total += line.unit_cents * line.quantity
        return total

    def calculate_total(self, subtotal_cents: int, tax_bps: int, discount_bps: int = 0) -> int:
        """Compute invoice total in integer cents using basis-point rates."""
        if subtotal_cents < 0 or tax_bps < 0 or discount_bps < 0:
            raise ValueError("Billing amounts and rates must be non-negative")
        discount_cents = (subtotal_cents * min(discount_bps, 10_000)) // 10_000
        taxable_cents = subtotal_cents - discount_cents
        tax_cents = (taxable_cents * tax_bps) // 10_000
        return taxable_cents + tax_cents
''',
        )
        _write_text(
            self.root / "app" / "legacy.py",
            '''"""Deprecated legacy XML exporter."""

from html import escape


def export_legacy_xml(records: list[dict[str, str]], *, root_tag: str = "legacy") -> str:
    """Render deprecated XML payload for backward-compatibility exports."""
    parts: list[str] = [f"<{root_tag}>"]
    for record in records:
        rec_id = escape(str(record.get("id", "")))
        label = escape(str(record.get("label", "")))
        status = escape(str(record.get("status", "unknown")))
        parts.append(f'  <row id="{rec_id}" label="{label}" status="{status}" />')
    parts.append(f"</{root_tag}>")
    return "\n".join(parts)


def parse_legacy_flag(raw_value: str) -> bool:
    """Interpret legacy boolean XML attribute strings."""
    normalized = raw_value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off", ""):
        return False
    raise ValueError(f"Unsupported legacy flag value: {raw_value}")
''',
        )
        _write_text(
            self.root / "app" / "metrics.py",
            '''"""Prometheus-compatible in-memory counters and histograms."""


class MetricsRegistry:
    """Collects deterministic counters and latency buckets."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, int] = {}

    def increment_counter(self, name: str, value: int = 1) -> int:
        """Increment named counter and return updated total."""
        if value < 0:
            raise ValueError("Counter delta must be non-negative")
        updated = self._counters.get(name, 0) + value
        self._counters[name] = updated
        return updated

    def set_gauge(self, name: str, value: int) -> None:
        """Record latest gauge measurement."""
        self._gauges[name] = value

    def render_text(self) -> str:
        """Export counters and gauges in Prometheus exposition format."""
        lines = [f"{k} {self._counters[k]}" for k in sorted(self._counters)]
        lines.extend(f"{k} {self._gauges[k]}" for k in sorted(self._gauges))
        return "\n".join(lines)
''',
        )
        _write_text(
            self.root / "app" / "notifications.py",
            '''"""Email notification queue and batch dispatcher."""

from dataclasses import dataclass


@dataclass(slots=True)
class OutboundEmail:
    """Queued outbound email message."""

    recipient: str
    subject: str
    body: str
    attempts: int = 0


class NotificationDispatcher:
    """Manages outbound email formatting and retry bookkeeping."""

    def __init__(self, max_batch: int = 50) -> None:
        self._max_batch = max(1, max_batch)
        self._queue: list[OutboundEmail] = []

    def enqueue_email(self, recipient: str, subject: str, body: str = "") -> bool:
        """Validate and queue outbound notification."""
        clean_recipient = recipient.strip()
        clean_subject = subject.strip()
        if not clean_recipient or "@" not in clean_recipient or not clean_subject:
            return False
        self._queue.append(
            OutboundEmail(recipient=clean_recipient, subject=clean_subject, body=body)
        )
        return True

    def pop_batch(self) -> list[OutboundEmail]:
        """Return next bounded batch of queued outbound messages."""
        batch = self._queue[: self._max_batch]
        self._queue = self._queue[self._max_batch :]
        return batch
''',
        )
        _write_text(
            self.root / "README.md",
            """# Service Guide\n\n## Authentication\n\nDetails on JWT login flow.\n""",
        )

        # 2 INVARIANT cards
        self.store.add_card(
            KnowledgeCard(
                id="INV-001",
                kind=CardKind.INVARIANT,
                title="Zero external runtime dependencies",
                statement="Core modules must use only Python 3.11+ standard library.",
                rationale="Ensures portable execution everywhere.",
                immutable=True,
                always_include=True,
                tags=["architecture", "stdlib"],
            )
        )
        self.store.add_card(
            KnowledgeCard(
                id="INV-002",
                kind=CardKind.INVARIANT,
                title="Constant-time password check",
                statement="Never compare password hashes with plain == in production.",
                rationale="Prevents timing side-channel leaks.",
                immutable=True,
                always_include=False,
                tags=["auth", "security"],
                anchors=[Anchor(file_path="app/auth.py", symbol="AuthService")],
            )
        )

        # 3 ADR cards (one of which gets SUPERSEDED!)
        self.store.add_card(
            KnowledgeCard(
                id="ADR-001",
                kind=CardKind.ADR,
                title="Old cookie-based session storage",
                statement="OBSOLETE_COOKIE_MARKER: Store opaque session IDs in server cookies.",
                rationale="Initial prototype approach.",
                tags=["auth", "session"],
                anchors=[Anchor(file_path="app/auth.py")],
            )
        )
        self.store.supersede_card(
            "ADR-001",
            KnowledgeCard(
                id="ADR-002",
                kind=CardKind.ADR,
                title="Stateless JWT access tokens",
                statement="Use signed JWT tokens issued by AuthService.",
                rationale="Removes server-side session lookup from API nodes.",
                tags=["auth", "jwt"],
                anchors=[Anchor(file_path="app/auth.py", symbol="AuthService")],
            ),
        )
        self.store.add_card(
            KnowledgeCard(
                id="ADR-003",
                kind=CardKind.ADR,
                title="Integer cents for billing",
                statement="Represent all monetary amounts as integer cents.",
                rationale="Avoids floating point rounding bugs.",
                tags=["billing", "money"],
                anchors=[Anchor(file_path="app/billing.py")],
            )
        )

        # 2 PITFALL cards (one auth-related, one billing-related)
        self.store.add_card(
            KnowledgeCard(
                id="PIT-001",
                kind=CardKind.PITFALL,
                title="Do not store JWT secret in plain .env inside repo",
                statement="Never hardcode or commit signing secrets or read .env directly in auth.py.",
                rationale="Rejected after secret leak risk in automated prompt capsules.",
                tags=["auth", "security", "jwt"],
                anchors=[Anchor(file_path="app/auth.py")],
            )
        )
        self.store.add_card(
            KnowledgeCard(
                id="PIT-002",
                kind=CardKind.PITFALL,
                title="Do not use float for invoice totals",
                statement="Never calculate currency totals using IEEE-754 floats.",
                rationale="Rejected due to rounding drift.",
                tags=["billing", "currency"],
                anchors=[Anchor(file_path="app/billing.py")],
            )
        )

        # 2 TaskNodes
        self.store.add_task(
            TaskNode(
                id="TASK-001",
                title="Setup database schema",
                description="Implement DbUser and Database lookup methods.",
                status=TaskStatus.DONE,
                priority=1,
                anchors=[Anchor(file_path="app/db.py", symbol="Database")],
            )
        )
        self.store.add_task(
            TaskNode(
                id="TASK-002",
                title="Harden JWT authentication flow",
                description="Improve AuthService.authenticate and verify_session token validation.",
                status=TaskStatus.IN_PROGRESS,
                priority=1,
                depends_on=["TASK-001"],
                anchors=[Anchor(file_path="app/auth.py", symbol="AuthService")],
                related_cards=["INV-002"],
                acceptance_criteria=[
                    "Invalid tokens return None cleanly",
                    "Empty passwords raise PermissionError",
                ],
                next_step_hint="Verify token expiration handling in AuthService.",
            )
        )

        self.store.append_session(
            SessionLogEntry(
                session_id="SES-001",
                task_id="TASK-001",
                summary="Implemented DbUser and Database lookup.",
                next_anchor="Start AuthService hardening in app/auth.py",
            )
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_context_slicing_truth_filtering_and_savings(self) -> None:
        compiler = ContextCompiler(self.store)
        result = compiler.compile(
            CompileRequest(task_id="TASK-002", token_budget=8000)
        )
        capsule = result.capsule_text

        # Tier-1 contains app/auth.py in FULL, and NOT db.py/api.py/billing.py
        tier1_block = _extract_tag(capsule, "code_tier1_full")
        self.assertIn("app/auth.py", tier1_block)
        self.assertIn("def authenticate(self, email: str, raw_password: str)", tier1_block)
        self.assertNotIn("app/db.py", tier1_block)
        self.assertNotIn("app/api.py", tier1_block)
        self.assertNotIn("app/billing.py", tier1_block)

        # Tier-2 contains AST skeletons of db.py and api.py, and NOT billing.py or duplicated auth.py
        tier2_block = _extract_tag(capsule, "code_tier2_signatures")
        self.assertIn("app/db.py", tier2_block)
        self.assertIn("app/api.py", tier2_block)
        self.assertIn("def find_user_by_email(self, email: str) -> DbUser | None:", tier2_block)
        self.assertIn("Lookup user record by normalized email.", tier2_block)
        self.assertNotIn("app/billing.py", tier2_block)
        self.assertNotIn("app/auth.py", tier2_block)

        # No file duplicated between Tier-1 and Tier-2
        self.assertEqual(
            set(result.stats["tier1_files"]) & set(result.stats["tier2_files"]),
            set(),
        )

        # Billing and legacy remain in Tier-3 tree only
        tier3_block = _extract_tag(capsule, "project_tree_tier3")
        self.assertIn("billing.py", tier3_block)
        self.assertIn("legacy.py", tier3_block)

        # SUPERSEDED ADR-001 must NEVER enter the capsule!
        self.assertNotIn("ADR-001", capsule)
        self.assertNotIn("OBSOLETE_COOKIE_MARKER", capsule)
        self.assertIn("ADR-002", capsule)

        # Auth PITFALL enters <rejected_approaches>, unrelated billing PITFALL stays out
        pitfalls_block = _extract_tag(capsule, "rejected_approaches")
        self.assertIn("PIT-001", pitfalls_block)
        self.assertNotIn("PIT-002", pitfalls_block)

        # Invariants & <ctx_protocol> present in expected zones
        invariants_block = _extract_tag(capsule, "project_invariants")
        self.assertIn("INV-001", invariants_block)
        self.assertIn("INV-002", invariants_block)
        self.assertIn("<ctx_protocol>", capsule)
        self.assertIn("<ctx_delta>", capsule)
        self.assertTrue(capsule.rstrip().endswith("</ctx_protocol>"))

        # Verify honest full-repo baseline and savings calculation
        expected_full_files_tokens = sum(
            estimate_tokens(p.read_text(encoding="utf-8"))
            for p in self.root.rglob("*")
            if p.is_file() and ".ctx" not in p.parts
        )
        self.assertGreater(result.full_repo_tokens, expected_full_files_tokens)
        expected_savings = round(
            (1.0 - (result.capsule_tokens / result.full_repo_tokens)) * 100.0, 2
        )
        self.assertAlmostEqual(result.savings_percent, expected_savings, places=2)
        self.assertGreater(result.savings_percent, 0.0)
        self.assertLessEqual(result.capsule_tokens, 8000)

        # Summary report formats cleanly
        report = result.format_summary_report()
        self.assertIn("ctxforge Context Capsule Summary", report)
        self.assertIn("savings:", report)

    def test_tight_token_budget_preserves_invariants_task_and_protocol(self) -> None:
        compiler = ContextCompiler(self.store)
        tight_budget = 1500
        result = compiler.compile(
            CompileRequest(task_id="TASK-002", token_budget=tight_budget)
        )
        self.assertLessEqual(result.capsule_tokens, tight_budget)
        self.assertEqual(estimate_tokens(result.capsule_text), result.capsule_tokens)

        # Critical sections MUST survive tight budget
        self.assertIn("INV-001", _extract_tag(result.capsule_text, "project_invariants"))
        self.assertIn("TASK-002", _extract_tag(result.capsule_text, "active_task"))
        self.assertIn("PIT-001", _extract_tag(result.capsule_text, "rejected_approaches"))
        self.assertIn("<ctx_protocol>", result.capsule_text)
        self.assertTrue(result.stats["protocol_included"])
        self.assertGreater(result.savings_percent, 0.0)

    def test_ultra_tight_budget_degrades_tier1_gracefully(self) -> None:
        # Make auth.py significantly larger so it cannot fit whole in 650 tokens
        large_auth = self.root / "app" / "auth.py"
        extra_methods = "\n".join(
            f'''
    def helper_step_{i}(self, payload: str) -> str:
        """Auxiliary processing step {i}."""
        value = payload.strip() + "-{i}"
        if len(value) > 40:
            return value[:40]
        return value
'''
            for i in range(18)
        )
        _write_text(large_auth, large_auth.read_text(encoding="utf-8") + extra_methods)

        compiler = ContextCompiler(self.store)
        result = compiler.compile(
            CompileRequest(task_id="TASK-002", token_budget=650)
        )
        self.assertLessEqual(result.capsule_tokens, 650)
        self.assertIn("INV-001", result.capsule_text)
        self.assertIn("TASK-002", result.capsule_text)
        self.assertIn("<ctx_protocol>", result.capsule_text)
        self.assertTrue(
            len(result.stats["warnings"]) > 0 or len(result.stats["dropped"]) > 0
        )

    def test_free_query_without_task_uses_bm25_and_optional_protocol_flag(self) -> None:
        compiler = ContextCompiler(self.store)
        result = compiler.compile(
            CompileRequest(
                task_id=None,
                query="invoice total calculation in billing",
                token_budget=4000,
                include_protocol=False,
            )
        )
        self.assertIn("app/billing.py", result.stats["tier1_files"])
        self.assertIn("PIT-002", _extract_tag(result.capsule_text, "rejected_approaches"))
        self.assertIn("ADR-003", _extract_tag(result.capsule_text, "active_decisions"))
        self.assertNotIn("<ctx_protocol>", result.capsule_text)
        self.assertFalse(result.stats["protocol_included"])
        # Superseded ADR-001 still never leaks
        self.assertNotIn("ADR-001", result.capsule_text)


if __name__ == "__main__":
    unittest.main()