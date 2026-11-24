"""Unit tests for DeltaIngestor, BootstrapGenerator, and ContextDoctor."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from ctxforge.bootstrap import BootstrapGenerator
from ctxforge.doctor import ContextDoctor, IssueKind
from ctxforge.indexer import RepoIndexer
from ctxforge.ingest import (
    DeltaIngestor,
    DeltaParseError,
    OpKind,
    OpStatus,
    extract_tagged_json,
)
from ctxforge.models import (
    Anchor,
    CardKind,
    CardStatus,
    KnowledgeCard,
    ProjectMeta,
    TaskNode,
    TaskStatus,
)
from ctxforge.storage import ContextStore
from ctxforge.tokens import estimate_tokens


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestIngestAndDoctor(unittest.TestCase):
    """End-to-end tests for delta ingestion, Day-0 bootstrap, and doctor diagnostics."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.store = ContextStore(self.root)
        self.store.init(
            ProjectMeta(
                name="ingest-doctor-demo",
                vision="Test project for delta ingestion and reality drift detection.",
            )
        )
        _write_text(
            self.root / "app" / "auth.py",
            '''"""Authentication module."""


class AuthService:
    """Issues and validates tokens."""

    def login(self, email: str) -> str:
        return f"tok:{email}"
''',
        )
        _write_text(
            self.root / "app" / "db.py",
            '''"""Database repository."""


class Database:
    """In-memory test database."""

    def fetch(self, key: str) -> str | None:
        return None
''',
        )
        self.indexer = RepoIndexer(self.root, store=self.store)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_extract_dirty_llm_response_with_fences_and_trailing_commas(self) -> None:
        dirty_response = """Вот готовое решение и подробный разбор кода.

```python
def example():
    # Декоративный код до блока дельты
    return {"not_delta": True}
```

Если вдруг обсуждалось раньше: <ctx_delta>{"task_id": "OLD"}</ctx_delta>

Итоговое обновление памяти:
<ctx_delta>
```json
{
  "task_id": "TASK-001",
  "task_status": "done",
  "decisions_added": [
    {
      "kind": "ADR",
      "title": "Use signed JWT tokens",
      "statement": "Issue compact JWTs, avoiding {'legacy': 1, } session cookies",
      "rationale": "Stateless horizontal scaling",
      "tags": ["auth", "jwt",],
      "anchors": ["app/auth.py:AuthService", "app/hallucinated.py",],
    },
  ],
  "new_subtasks": [
    {
      "title": "Add refresh endpoint",
      "description": "Implement token refresh flow in AuthService.",
      "anchors": ["app/auth.py"],
    },
  ],
  "next_anchor": "Implement refresh endpoint in app/auth.py",
}
```
</ctx_delta>

Надеюсь, это поможет!
"""
        parsed = extract_tagged_json(dirty_response, tag="ctx_delta")
        self.assertEqual(parsed["task_id"], "TASK-001")
        self.assertEqual(parsed["task_status"], "done")
        # Comma inside string literal must NOT be mangled.
        self.assertIn("{'legacy': 1, }", parsed["decisions_added"][0]["statement"])

        # Direct clean JSON without tags also works.
        raw_json = extract_tagged_json(
            '{"task_id": "TASK-001", "task_status": "in_progress",}'
        )
        self.assertEqual(raw_json["task_status"], "in_progress")

        # Invalid JSON produces DeltaParseError with excerpt.
        with self.assertRaises(DeltaParseError) as err:
            extract_tagged_json("<ctx_delta>{broken: json}</ctx_delta>")
        self.assertIn("broken: json", str(err.exception))

        # Test build_delta_plan + hallucinated anchor warning + hash attachment.
        self.store.add_task(
            TaskNode(
                id="TASK-001",
                title="Implement login",
                description="Create AuthService.login",
                status=TaskStatus.IN_PROGRESS,
            )
        )
        index = self.indexer.build_or_update()
        ingestor = DeltaIngestor(self.store)
        plan = ingestor.build_delta_plan(dirty_response, index=index)

        self.assertTrue(
            any("Unknown anchor path: 'app/hallucinated.py'" in w for w in plan.warnings)
        )
        formatted = plan.format_plan()
        self.assertIn("[*] UPDATE TASK-001", formatted)
        self.assertIn("[+] ADR-001: Use signed JWT tokens", formatted)
        self.assertIn("[+] TASK-002: Add refresh endpoint", formatted)

        summary = ingestor.apply_plan(plan)
        self.assertEqual(summary.cards_created, 1)
        self.assertEqual(summary.tasks_updated, 1)
        self.assertEqual(summary.tasks_created, 1)
        self.assertEqual(summary.sessions_logged, 1)

        added_card = self.store.require_card("ADR-001")
        self.assertEqual(
            added_card.anchors[0].file_hash, index.files["app/auth.py"].sha256
        )
        self.assertIsNone(added_card.anchors[1].file_hash)
        self.assertEqual(self.store.require_task("TASK-001").status, TaskStatus.DONE)
        self.assertEqual(self.store.require_task("TASK-002").title, "Add refresh endpoint")

    def test_delta_blocks_immutable_supersede_unless_forced(self) -> None:
        self.store.add_card(
            KnowledgeCard(
                id="INV-001",
                kind=CardKind.INVARIANT,
                title="Stdlib only",
                statement="No external runtime packages allowed.",
                rationale="Portability.",
                immutable=True,
                always_include=True,
                anchors=[Anchor(file_path="app/auth.py")],
            )
        )
        index = self.indexer.build_or_update()

        llm_response = """
```json
<ctx_delta>
{
  "decisions_superseded": [
    {
      "old_id": "INV-001",
      "replacement": {
        "kind": "INV",
        "title": "Allow external dependencies",
        "statement": "Allow pydantic and fastapi.",
        "rationale": "Faster coding."
      }
    }
  ],
  "next_anchor": "Install dependencies"
}
</ctx_delta>
```
"""
        ingestor = DeltaIngestor(self.store)
        plan = ingestor.build_delta_plan(llm_response, index=index)

        blocked_ops = plan.blocked_operations
        self.assertEqual(len(blocked_ops), 1)
        self.assertEqual(blocked_ops[0].kind, OpKind.SUPERSEDE_CARD)
        self.assertEqual(blocked_ops[0].status, OpStatus.BLOCKED_BY_IMMUTABLE)
        self.assertIn("[!] BLOCKED INV-001 (immutable)", plan.format_plan())

        # Without force_immutable=True, INV-001 remains ACTIVE and unchanged.
        summary = ingestor.apply_plan(plan, force_immutable=False)
        self.assertEqual(summary.cards_superseded, 0)
        self.assertEqual(summary.blocked_ops, ["INV-001"])
        self.assertEqual(self.store.require_card("INV-001").status, CardStatus.ACTIVE)

        # With force_immutable=True, override succeeds deliberately.
        forced_summary = ingestor.apply_plan(plan, force_immutable=True)
        self.assertEqual(forced_summary.cards_superseded, 1)
        self.assertEqual(self.store.require_card("INV-001").status, CardStatus.SUPERSEDED)
        self.assertEqual(self.store.require_card("INV-001").superseded_by, "INV-002")
        self.assertEqual(self.store.require_card("INV-002").status, CardStatus.ACTIVE)

    def test_bootstrap_generator_and_apply_bootstrap_flow(self) -> None:
        fresh_tmp = tempfile.TemporaryDirectory()
        try:
            fresh_root = Path(fresh_tmp.name).resolve()
            _write_text(
                fresh_root / "src" / "main.py",
                '''"""CLI entrypoint."""


def run_app(config_path: str) -> int:
    """Start local application."""
    return 0
''',
            )
            fresh_store = ContextStore(fresh_root)
            fresh_indexer = RepoIndexer(fresh_root, store=fresh_store)
            generator = BootstrapGenerator(fresh_store, fresh_indexer)

            prompt = generator.generate_bootstrap_prompt(
                brief_text="Build a local-first Markdown note indexer in pure Python 3.11.",
                token_budget=6000,
            )
            self.assertLessEqual(estimate_tokens(prompt), 6000)
            self.assertIn("<ctx_bootstrap>", prompt)
            self.assertIn("def run_app(config_path: str) -> int:", prompt)
            self.assertIn("local-first Markdown note indexer", prompt)

            mock_llm_bootstrap = """
Here is the initial structured memory for your project:

<ctx_bootstrap>
```json
{
  "project": {
    "name": "note-forge",
    "vision": "Fast local-first Markdown note indexer with zero external dependencies.",
    "non_goals": ["Cloud sync", "Paid LLM APIs"],
    "tech_stack_notes": "Python 3.11+ stdlib only; must handle large vaults on slow disks."
  },
  "cards": [
    {
      "kind": "INV",
      "title": "Zero external dependencies",
      "statement": "Use only Python 3.11 standard library.",
      "rationale": "Single-command zero-install portability.",
      "immutable": true,
      "always_include": true,
      "tags": ["stdlib"],
      "anchors": ["src/main.py:run_app"]
    },
    {
      "kind": "ADR",
      "title": "Store index as atomic JSON",
      "statement": "Persist vault metadata via temp file + os.replace.",
      "rationale": "Prevents index corruption on interrupted runs.",
      "tags": ["storage"],
      "anchors": ["src/main.py"]
    },
    {
      "kind": "PITFALL",
      "title": "Do not load all markdown bodies into memory",
      "statement": "Never keep full note texts inside index JSON.",
      "rationale": "Rejected to avoid memory bloat on 10k+ notes.",
      "tags": ["performance"],
      "anchors": ["src/nonexistent.py"]
    }
  ],
  "tasks": [
    {
      "id": "T1",
      "title": "Implement vault scanner",
      "description": "Walk markdown files recursively.",
      "priority": 1,
      "depends_on": [],
      "anchors": ["src/main.py:run_app"],
      "acceptance_criteria": ["Skips hidden directories"]
    },
    {
      "id": "T2",
      "title": "Build CLI query command",
      "description": "Query indexed notes from terminal.",
      "priority": 2,
      "depends_on": ["T1"],
      "anchors": ["src/main.py"],
      "acceptance_criteria": ["Returns top matches under 50ms"]
    }
  ]
}
```
</ctx_bootstrap>
"""
            ingestor = DeltaIngestor(fresh_store)
            index = fresh_indexer.build_or_update()
            summary = ingestor.apply_bootstrap(mock_llm_bootstrap, index=index)

            self.assertTrue(fresh_store.is_initialized())
            self.assertEqual(summary.cards_created, 3)
            self.assertEqual(summary.tasks_created, 2)
            self.assertEqual(summary.sessions_logged, 1)
            self.assertTrue(
                any(
                    "Unknown anchor path: 'src/nonexistent.py'" in w
                    for w in summary.warnings
                )
            )

            meta = fresh_store.get_project()
            self.assertEqual(meta.name, "note-forge")
            self.assertEqual(meta.non_goals, ["Cloud sync", "Paid LLM APIs"])

            tasks = fresh_store.list_tasks()
            self.assertEqual([t.id for t in tasks], ["TASK-001", "TASK-002"])
            self.assertEqual(tasks[1].depends_on, ["TASK-001"])
            self.assertEqual(
                fresh_store.require_card("INV-001").anchors[0].file_hash,
                index.files["src/main.py"].sha256,
            )
        finally:
            fresh_tmp.cleanup()

    def test_doctor_detects_reality_drift_broken_anchors_and_noise_bloat(self) -> None:
        index = self.indexer.build_or_update()
        auth_sha = index.files["app/auth.py"].sha256
        db_sha = index.files["app/db.py"].sha256

        self.store.add_card(
            KnowledgeCard(
                id="ADR-001",
                kind=CardKind.ADR,
                title="Auth login contract",
                statement="Use AuthService.login returning prefix tok:.",
                rationale="Initial design.",
                anchors=[
                    Anchor(
                        file_path="app/auth.py",
                        symbol="AuthService.login",
                        file_hash=auth_sha,
                    )
                ],
            )
        )
        self.store.add_card(
            KnowledgeCard(
                id="INV-001",
                kind=CardKind.INVARIANT,
                title="Database fetch contract",
                statement="Database.fetch must return str | None.",
                rationale="Type safety.",
                anchors=[
                    Anchor(
                        file_path="app/db.py",
                        symbol="Database.fetch",
                        file_hash=db_sha,
                    )
                ],
            )
        )
        self.store.add_task(
            TaskNode(
                id="TASK-001",
                title="Review auth",
                description="Verify AuthService.login.",
                anchors=[Anchor(file_path="app/auth.py", symbol="AuthService.login")],
            )
        )

        doctor = ContextDoctor(self.store, self.indexer)
        clean_report = doctor.diagnose()
        self.assertTrue(clean_report.is_healthy)
        self.assertIn("HEALTHY", clean_report.format_report())

        # 1. Modify app/auth.py keeping AuthService.login -> triggers NEEDS_REVIEW drift!
        _write_text(
            self.root / "app" / "auth.py",
            '''"""Authentication module (modified)."""


class AuthService:
    """Issues and validates tokens."""

    def login(self, email: str) -> str:
        return f"jwt:{email}"
''',
        )

        # 2. Remove Database.fetch symbol in app/db.py -> triggers BROKEN_ANCHOR!
        _write_text(
            self.root / "app" / "db.py",
            '''"""Database repository."""


class Database:
    """In-memory test database."""

    def query_record(self, key: str) -> str | None:
        return None
''',
        )

        drift_report = doctor.diagnose()
        self.assertFalse(drift_report.is_healthy)
        self.assertIn("ADR-001", drift_report.cards_needing_review)
        self.assertEqual(len(drift_report.needs_review), 1)
        self.assertTrue(
            any(
                i.kind is IssueKind.BROKEN_ANCHOR and i.entity_id == "INV-001"
                for i in drift_report.issues
            )
        )
        self.assertIn("NEEDS_REVIEW", drift_report.format_report())
        self.assertIn("BROKEN_ANCHOR", drift_report.format_report())

        # fix_hashes=True updates ADR-001's anchor hash after user review.
        fixed_report = doctor.diagnose(fix_hashes=True)
        self.assertEqual(fixed_report.fixed_hashes, 1)
        self.assertEqual(fixed_report.needs_review, [])

        recheck_report = doctor.diagnose()
        self.assertEqual(recheck_report.needs_review, [])
        self.assertNotIn("ADR-001", recheck_report.cards_needing_review)

        # 3. >10 always_include cards triggers ALWAYS_INCLUDE_BLOAT + duplicate check.
        for idx in range(2, 13):
            self.store.add_card(
                KnowledgeCard(
                    id=f"INV-{idx:03d}",
                    kind=CardKind.INVARIANT,
                    title=f"Global hygiene rule number {idx}",
                    statement=f"Mandatory baseline rule {idx}.",
                    rationale="Testing noise threshold.",
                    always_include=True,
                )
            )
        self.store.add_card(
            KnowledgeCard(
                id="ADR-002",
                kind=CardKind.ADR,
                title="Auth login contracts",
                statement="Near duplicate title to test duplicate detection.",
                rationale="Check similarity.",
            )
        )

        bloat_report = doctor.diagnose()
        self.assertGreater(bloat_report.always_include_count, 10)
        self.assertTrue(
            any(i.kind is IssueKind.ALWAYS_INCLUDE_BLOAT for i in bloat_report.issues)
        )
        self.assertTrue(
            any(i.kind is IssueKind.DUPLICATE_CARD for i in bloat_report.issues)
        )


if __name__ == "__main__":
    unittest.main()