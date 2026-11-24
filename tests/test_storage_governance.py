"""Unit tests for ContextStore, Truth Governance, Task DAG validation, and tokens."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ctxforge.models import (
    Anchor,
    CardKind,
    CardStatus,
    KnowledgeCard,
    ProjectMeta,
    SessionLogEntry,
    TaskNode,
    TaskStatus,
)
from ctxforge.storage import (
    CardNotFoundError,
    ContextStore,
    DagCycleError,
    DuplicateIdError,
    ImmutableViolationError,
    MissingDependencyError,
    StoreAlreadyInitializedError,
    StoreNotInitializedError,
    TaskNotFoundError,
    TruthGovernanceError,
)
from ctxforge.tokens import estimate_tokens, truncate_to_tokens


class TestContextStoreAndGovernance(unittest.TestCase):
    """Comprehensive tests for storage persistence, Truth Governance, and DAG checks."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = ContextStore(self.root)
        self.default_project = ProjectMeta(
            name="ctxforge-demo",
            vision="Локальный компилятор контекста для длительной работы с LLM.",
            non_goals=["Платные облачные векторные БД", "Обязательная мультиагентность"],
            tech_stack_notes="Python 3.11+, только stdlib в ядре.",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_init_creates_expected_structure_and_gitignore(self) -> None:
        self.assertFalse(self.store.is_initialized())
        with self.assertRaises(StoreNotInitializedError):
            self.store.get_project()

        self.store.init(self.default_project)

        self.assertTrue(self.store.is_initialized())
        self.assertTrue(self.store.ctx_dir.is_dir())
        self.assertTrue(self.store.out_dir.is_dir())
        self.assertTrue(self.store.project_file.is_file())
        self.assertTrue(self.store.cards_file.is_file())
        self.assertTrue(self.store.tasks_file.is_file())
        self.assertTrue(self.store.sessions_file.is_file())

        gitignore_text = self.store.gitignore_file.read_text(encoding="utf-8")
        self.assertEqual(gitignore_text, "cache/\nout/\n")

        with self.assertRaises(StoreAlreadyInitializedError):
            self.store.init(self.default_project)

        # force=True cleanly re-initializes the store.
        updated_meta = ProjectMeta(name="ctxforge-reinit", vision="Fresh vision")
        self.store.init(updated_meta, force=True)
        self.assertEqual(self.store.get_project().name, "ctxforge-reinit")

    def test_roundtrip_all_entities_with_utf8_cyrillic(self) -> None:
        self.store.init(self.default_project)

        loaded_project = self.store.get_project()
        self.assertEqual(loaded_project.name, self.default_project.name)
        self.assertEqual(loaded_project.vision, self.default_project.vision)
        self.assertEqual(loaded_project.non_goals, self.default_project.non_goals)
        self.assertEqual(
            loaded_project.tech_stack_notes, self.default_project.tech_stack_notes
        )

        # Ensure UTF-8 Cyrillic is stored directly without \\uXXXX escaping.
        raw_project_json = self.store.project_file.read_text(encoding="utf-8")
        self.assertIn("Локальный компилятор контекста", raw_project_json)

        card = KnowledgeCard(
            id="INV-001",
            kind=CardKind.INVARIANT,
            title="Только stdlib в ядре",
            statement="Модули ядра не должны импортировать сторонние пакеты.",
            rationale="Нулевая стоимость установки и стабильность в любой среде.",
            immutable=True,
            always_include=True,
            tags=["architecture", "stdlib"],
            anchors=[
                Anchor(
                    file_path="./ctxforge\\storage.py",
                    symbol="ContextStore",
                    file_hash="abc123def456",
                )
            ],
            source_session="SES-001",
        )
        self.store.add_card(card)

        loaded_card = self.store.require_card("INV-001")
        self.assertEqual(loaded_card.kind, CardKind.INVARIANT)
        self.assertEqual(loaded_card.status, CardStatus.ACTIVE)
        self.assertTrue(loaded_card.immutable)
        self.assertTrue(loaded_card.always_include)
        self.assertEqual(loaded_card.anchors[0].file_path, "ctxforge/storage.py")
        self.assertEqual(loaded_card.anchors[0].symbol, "ContextStore")

        task = TaskNode(
            id="TASK-001",
            title="Реализовать Truth Governance",
            description="Добавить supersede_card и защиту immutable-карт.",
            status=TaskStatus.IN_PROGRESS,
            priority=1,
            depends_on=[],
            anchors=[Anchor(file_path="ctxforge/storage.py", symbol="supersede_card")],
            related_cards=["INV-001"],
            acceptance_criteria=["ImmutableViolationError блокирует перезапись"],
            next_step_hint="Написать тесты на циклы DAG",
        )
        self.store.add_task(task)
        loaded_task = self.store.require_task("TASK-001")
        self.assertEqual(loaded_task.priority, 1)
        self.assertEqual(loaded_task.status, TaskStatus.IN_PROGRESS)
        self.assertEqual(loaded_task.related_cards, ["INV-001"])

        session = SessionLogEntry(
            session_id="SES-001",
            task_id="TASK-001",
            summary="Созданы модели и базовое хранилище.",
            next_anchor="ctxforge/storage.py:ContextStore.supersede_card",
        )
        self.store.append_session(session)
        sessions = self.store.list_sessions(task_id="TASK-001")
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].summary, session.summary)
        self.assertEqual(sessions[0].next_anchor, session.next_anchor)

    def test_supersede_card_preserves_history_and_filters_active(self) -> None:
        self.store.init(self.default_project)

        adr_1 = KnowledgeCard(
            id="ADR-001",
            kind=CardKind.ADR,
            title="Формат карточек JSON",
            statement="Хранить карточки в одном cards.json файле.",
            rationale="Проще на старте.",
        )
        pit_1 = KnowledgeCard(
            id="PIT-001",
            kind=CardKind.PITFALL,
            title="Не хранить кэш в git",
            statement="Исключить cache/ и out/ из индекса git.",
            rationale="Кэш раздувает историю коммитов.",
        )
        self.store.add_card(adr_1)
        self.store.add_card(pit_1)

        adr_2 = KnowledgeCard(
            id="ADR-002",
            kind=CardKind.ADR,
            title="Формат карточек JSONL",
            statement="Хранить карточки в .ctx/cards.jsonl по одной на строку.",
            rationale="Чистые построчные git diff и append-дружественность.",
        )
        self.store.supersede_card("ADR-001", adr_2)

        # Full history remains on disk in cards.jsonl.
        raw_lines = [
            json.loads(line)
            for line in self.store.cards_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(raw_lines), 3)
        by_id = {row["id"]: row for row in raw_lines}
        self.assertEqual(by_id["ADR-001"]["status"], CardStatus.SUPERSEDED.value)
        self.assertEqual(by_id["ADR-001"]["superseded_by"], "ADR-002")
        self.assertEqual(by_id["ADR-002"]["status"], CardStatus.ACTIVE.value)

        # get_active_cards excludes SUPERSEDED cards so stale truth cannot poison prompts.
        active_all = self.store.get_active_cards()
        self.assertEqual([c.id for c in active_all], ["PIT-001", "ADR-002"])

        active_adrs = self.store.get_active_cards(kind=CardKind.ADR)
        self.assertEqual([c.id for c in active_adrs], ["ADR-002"])

        # Supersession chain is fully traceable.
        chain = self.store.get_supersession_chain("ADR-001")
        self.assertEqual([c.id for c in chain], ["ADR-001", "ADR-002"])

        # Branching supersession on an already-superseded card is forbidden.
        adr_3 = KnowledgeCard(
            id="ADR-003",
            kind=CardKind.ADR,
            title="Попытка ветвления",
            statement="Попытка заменить устаревший ADR-001 напрямую.",
            rationale="Проверка защиты цепочки.",
        )
        with self.assertRaises(TruthGovernanceError):
            self.store.supersede_card("ADR-001", adr_3)

        with self.assertRaises(CardNotFoundError):
            self.store.supersede_card("ADR-999", adr_3)

    def test_immutable_violation_blocks_overwrite_unless_forced(self) -> None:
        self.store.init(self.default_project)

        inv = KnowledgeCard(
            id="INV-001",
            kind=CardKind.INVARIANT,
            title="Не изменять схему без миграции",
            statement="Запрещено ломать обратную совместимость .ctx/.",
            rationale="Защита пользовательских репозиториев.",
            immutable=True,
            always_include=True,
        )
        self.store.add_card(inv)

        replacement = KnowledgeCard(
            id="INV-002",
            kind=CardKind.INVARIANT,
            title="Смягчённое правило",
            statement="Разрешить ломающие изменения.",
            rationale="Тест",
        )

        with self.assertRaises(ImmutableViolationError) as ctx:
            self.store.supersede_card("INV-001", replacement)
        self.assertEqual(ctx.exception.card_id, "INV-001")

        # Verify store remained untouched after failed attempt.
        self.assertEqual(self.store.require_card("INV-001").status, CardStatus.ACTIVE)
        self.assertIsNone(self.store.get_card("INV-002"))

        # Status change (archive/deprecate) is also blocked on immutable cards.
        with self.assertRaises(ImmutableViolationError):
            self.store.set_card_status("INV-001", CardStatus.DEPRECATED)

        # Explicit force_immutable=True allows controlled override.
        self.store.supersede_card("INV-001", replacement, force_immutable=True)
        old_after = self.store.require_card("INV-001")
        self.assertEqual(old_after.status, CardStatus.SUPERSEDED)
        self.assertEqual(old_after.superseded_by, "INV-002")
        self.assertEqual([c.id for c in self.store.get_active_cards()], ["INV-002"])

    def test_duplicate_card_id_rejected(self) -> None:
        self.store.init(self.default_project)
        card = KnowledgeCard(
            id="ADR-001",
            kind=CardKind.ADR,
            title="Test",
            statement="Use atomic writes.",
            rationale="Safety.",
        )
        self.store.add_card(card)
        with self.assertRaises(DuplicateIdError):
            self.store.add_card(card)

    def test_task_dag_resolution_and_cycle_detection(self) -> None:
        self.store.init(self.default_project)

        t1 = TaskNode(id="TASK-001", title="Base", description="Core models")
        t2 = TaskNode(
            id="TASK-002",
            title="Store",
            description="Storage layer",
            depends_on=["TASK-001"],
        )
        t3 = TaskNode(
            id="TASK-003",
            title="Tokens",
            description="Token estimator",
            depends_on=["TASK-001"],
        )
        t4 = TaskNode(
            id="TASK-004",
            title="Compiler",
            description="Context compiler",
            depends_on=["TASK-002", "TASK-003"],
        )
        t_unrelated = TaskNode(
            id="TASK-005",
            title="Docs",
            description="Independent docs task",
        )

        for task in (t1, t2, t3, t4, t_unrelated):
            self.store.add_task(task)

        topo = self.store.validate_task_dag()
        self.assertLess(topo.index("TASK-001"), topo.index("TASK-002"))
        self.assertLess(topo.index("TASK-001"), topo.index("TASK-003"))
        self.assertLess(topo.index("TASK-002"), topo.index("TASK-004"))
        self.assertLess(topo.index("TASK-003"), topo.index("TASK-004"))

        target, deps = self.store.get_task_with_deps("TASK-004")
        self.assertEqual(target.id, "TASK-004")
        dep_ids = [d.id for d in deps]
        self.assertEqual(dep_ids[0], "TASK-001")
        self.assertEqual(set(dep_ids), {"TASK-001", "TASK-002", "TASK-003"})
        self.assertEqual(len(dep_ids), 3)  # Diamond dependency deduplicated.

        with self.assertRaises(TaskNotFoundError):
            self.store.get_task_with_deps("TASK-999")

        # Missing dependency is rejected before mutating storage.
        bad_missing = TaskNode(
            id="TASK-006",
            title="Broken",
            description="Missing dep",
            depends_on=["TASK-404"],
        )
        with self.assertRaises(MissingDependencyError):
            self.store.add_task(bad_missing)
        self.assertIsNone(self.store.get_task("TASK-006"))

        # Self-cycle detection.
        self_cycle = TaskNode(
            id="TASK-007",
            title="Self loop",
            description="Depends on itself",
            depends_on=["TASK-007"],
        )
        with self.assertRaises(DagCycleError) as self_err:
            self.store.add_task(self_cycle)
        self.assertEqual(self_err.exception.cycle, ["TASK-007", "TASK-007"])

        # Multi-node cycle detection (TASK-001 -> TASK-004 -> TASK-002 -> TASK-001).
        cyclic_t1 = TaskNode(
            id="TASK-001",
            title="Base",
            description="Introduces cycle",
            depends_on=["TASK-004"],
        )
        with self.assertRaises(DagCycleError) as cycle_err:
            self.store.update_task(cyclic_t1)
        self.assertEqual(cycle_err.exception.cycle[0], cycle_err.exception.cycle[-1])
        self.assertIn("TASK-001", cycle_err.exception.cycle)
        self.assertIn("TASK-004", cycle_err.exception.cycle)

        # Also verify validate_task_dag catches cycles persisted directly.
        self.store.update_task(cyclic_t1, validate_dag=False)
        with self.assertRaises(DagCycleError):
            self.store.validate_task_dag()

    def test_sequential_id_generation_never_reuses_ids(self) -> None:
        self.store.init(self.default_project)

        self.assertEqual(self.store.next_card_id(CardKind.INVARIANT), "INV-001")
        self.assertEqual(self.store.next_card_id(CardKind.ADR), "ADR-001")
        self.assertEqual(self.store.next_card_id(CardKind.PITFALL), "PIT-001")
        self.assertEqual(self.store.next_task_id(), "TASK-001")
        self.assertEqual(self.store.next_session_id(), "SES-001")

        adr1_id = self.store.next_card_id(CardKind.ADR)
        self.store.add_card(
            KnowledgeCard(
                id=adr1_id,
                kind=CardKind.ADR,
                title="First ADR",
                statement="Statement 1",
                rationale="Rationale 1",
            )
        )
        adr2_id = self.store.next_card_id(CardKind.ADR)
        self.assertEqual(adr2_id, "ADR-002")

        # Even after ADR-001 is superseded and ADR-002 is archived, next ID advances.
        self.store.supersede_card(
            "ADR-001",
            KnowledgeCard(
                id=adr2_id,
                kind=CardKind.ADR,
                title="Second ADR",
                statement="Statement 2",
                rationale="Rationale 2",
            ),
        )
        self.store.set_card_status("ADR-002", CardStatus.ARCHIVED)
        self.assertEqual(self.store.next_card_id(CardKind.ADR), "ADR-003")

        # Non-contiguous manual ID jumps forward cleanly.
        self.store.add_card(
            KnowledgeCard(
                id="PIT-005",
                kind=CardKind.PITFALL,
                title="Pitfall 5",
                statement="Avoid global mutable state.",
                rationale="Hard to test.",
            )
        )
        self.assertEqual(self.store.next_card_id(CardKind.PITFALL), "PIT-006")

        self.store.add_task(TaskNode(id="TASK-001", title="T1", description="Desc"))
        self.store.add_task(TaskNode(id="TASK-002", title="T2", description="Desc"))
        self.assertEqual(self.store.next_task_id(), "TASK-003")

    def test_anchor_file_hash_and_traversal_guard(self) -> None:
        self.store.init(self.default_project)
        sample_file = self.root / "src" / "app.py"
        sample_file.parent.mkdir(parents=True, exist_ok=True)
        sample_file.write_text("print('hello')\n", encoding="utf-8")

        digest = self.store.compute_anchor_hash("src/app.py")
        self.assertIsNotNone(digest)
        self.assertEqual(len(digest or ""), 12)
        self.assertIsNone(self.store.compute_anchor_hash("src/missing.py"))

        with self.assertRaises(ValueError):
            self.store.compute_anchor_hash("../outside.txt")


class TestTokenEstimator(unittest.TestCase):
    """Tests for calibrated stdlib token estimation and budget truncation."""

    def test_empty_and_monotonic_estimation(self) -> None:
        self.assertEqual(estimate_tokens(""), 0)
        self.assertGreaterEqual(estimate_tokens("x"), 1)

        short = "ContextStore saves cards."
        longer = short + " Additional architectural context with details."
        self.assertLess(estimate_tokens(short), estimate_tokens(longer))

    def test_english_and_code_ratio_calibration(self) -> None:
        english = (
            "Context engineering is the discipline of curating the exact set of facts, "
            "constraints, architectural decisions, and task specifications that a language "
            "model needs at inference time without exceeding its working memory."
        )
        en_tokens = estimate_tokens(english)
        en_chars_per_token = len(english) / en_tokens
        self.assertGreaterEqual(en_chars_per_token, 3.2)
        self.assertLessEqual(en_chars_per_token, 4.4)

        python_code = (
            "def get_active_cards(self, kind: CardKind | None = None) -> list[KnowledgeCard]:\n"
            "    cards = self._load_cards()\n"
            "    return [\n"
            "        card for card in cards\n"
            "        if card.status is CardStatus.ACTIVE and (kind is None or card.kind is kind)\n"
            "    ]\n"
        )
        code_tokens = estimate_tokens(python_code)
        code_chars_per_token = len(python_code) / code_tokens
        self.assertGreaterEqual(code_chars_per_token, 3.0)
        self.assertLessEqual(code_chars_per_token, 4.3)

    def test_cyrillic_ratio_calibration(self) -> None:
        russian = (
            "Локальный компилятор контекста собирает только активные архитектурные решения, "
            "проверенные инварианты и актуальные задачи, полностью исключая устаревшие записи "
            "из итогового промпта для модели."
        )
        ru_tokens = estimate_tokens(russian)
        ru_chars_per_token = len(russian) / ru_tokens
        self.assertGreaterEqual(ru_chars_per_token, 2.0)
        self.assertLessEqual(ru_chars_per_token, 2.7)

    def test_truncate_to_tokens_respects_budget_and_marker(self) -> None:
        text = " ".join(f"слово{i}" for i in range(120))
        marker = "\n...[TRUNCATED]..."

        self.assertEqual(truncate_to_tokens(text, 0), "")
        self.assertEqual(truncate_to_tokens("", 50), "")
        self.assertEqual(truncate_to_tokens("short text", 50), "short text")

        budget = 40
        truncated = truncate_to_tokens(text, budget, marker=marker)
        self.assertTrue(truncated.endswith(marker))
        self.assertLessEqual(estimate_tokens(truncated), budget)
        self.assertGreater(len(truncated), len(marker))

        # Tiny budget smaller than marker still strictly honors max_tokens.
        tiny = truncate_to_tokens(text, 2, marker=marker)
        self.assertLessEqual(estimate_tokens(tiny), 2)


if __name__ == "__main__":
    unittest.main()