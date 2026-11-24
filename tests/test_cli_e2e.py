"""End-to-end CLI, clipboard, and optional LLM adapter tests for ctxforge."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from ctxforge import cli
from ctxforge.clipboard import copy_to_clipboard, read_from_clipboard
from ctxforge.llm_adapter import (
    GeminiFreeAdapter,
    MissingApiKeyError,
    OllamaAdapter,
    RateLimitError,
    auto_extract_delta_from_raw_chat,
)
from ctxforge.models import CardKind, TaskStatus
from ctxforge.storage import ContextStore
from ctxforge.tokens import estimate_tokens


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestCliEndToEnd(unittest.TestCase):
    """Full workflow test driving cli.main([...]) inside a temp git repository."""

    def setUp(self) -> None:
        self._orig_cwd = Path.cwd()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()

        _write_text(
            self.root / "app" / "__init__.py",
            "",
        )
        _write_text(
            self.root / "app" / "db.py",
            '''"""Database layer."""

class Database:
    """In-memory database stub."""

    def get_user(self, email: str) -> dict[str, str] | None:
        """Fetch user row by email."""
        return {"email": email}
''',
        )
        _write_text(
            self.root / "app" / "auth.py",
            '''"""JWT authentication service."""

from app.db import Database


class AuthService:
    """Handles login and token issuance."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def login(self, email: str, password: str) -> str:
        """Authenticate credentials and issue token."""
        user = self._db.get_user(email)
        if not user or not password:
            raise PermissionError("Invalid login")
        return f"jwt:{email}"
''',
        )
        _write_text(
            self.root / "app" / "billing.py",
            '''"""Billing calculator."""

def compute_total(cents: int) -> int:
    """Return total cents."""
    return cents
''',
        )

        if shutil.which("git") is not None:
            git_env = dict(os.environ)
            git_env["GIT_CONFIG_GLOBAL"] = os.devnull
            git_env["GIT_CONFIG_NOSYSTEM"] = "1"
            subprocess.run(
                ["git", "-C", str(self.root), "init"],
                env=git_env,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(self.root), "config", "user.name", "E2E"],
                env=git_env,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(self.root), "config", "user.email", "e2e@example.com"],
                env=git_env,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(self.root), "config", "commit.gpgsign", "false"],
                env=git_env,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(self.root), "add", "app"],
                env=git_env,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(self.root), "commit", "-m", "feat: initial modules"],
                env=git_env,
                capture_output=True,
                check=True,
            )

        os.chdir(self.root)

    def tearDown(self) -> None:
        os.chdir(self._orig_cwd)
        self._tmp.cleanup()

    def _run_cli(self, argv: list[str], stdin_text: str = "") -> tuple[int, str, str]:
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with (
            contextlib.redirect_stdout(out_buf),
            contextlib.redirect_stderr(err_buf),
            patch("sys.stdin", io.StringIO(stdin_text)),
        ):
            code = cli.main(argv)
        return code, out_buf.getvalue(), err_buf.getvalue()

    def test_full_cli_lifecycle_e2e(self) -> None:
        # 1. ctx init
        code, out, err = self._run_cli(
            ["init", "--name", "e2e-app", "--vision", "Zero-dependency auth service"]
        )
        self.assertEqual(code, 0, msg=err)
        self.assertIn("Initialized ctxforge store", out)
        self.assertTrue((self.root / ".ctx" / "project.json").is_file())

        # 2. ctx bootstrap -> simulated AI Studio <ctx_bootstrap> -> ctx ingest --bootstrap --auto
        brief_path = self.root / "brief.md"
        _write_text(
            brief_path,
            "Build JWT auth service on top of Database using stdlib only.",
        )
        code, out, err = self._run_cli(["bootstrap", str(brief_path), "--no-clip"])
        self.assertEqual(code, 0, msg=err)
        self.assertTrue(
            (self.root / ".ctx" / "out" / "bootstrap_prompt.md").is_file()
        )

        bootstrap_response_path = self.root / "ai_bootstrap_response.txt"
        _write_text(
            bootstrap_response_path,
            """Вот ваш стартовый контекст:
<ctx_bootstrap>
```json
{
  "project": {
    "name": "e2e-app",
    "vision": "Clean stdlib authentication microservice.",
    "non_goals": ["Paid external SaaS"],
    "tech_stack_notes": "Python 3.11+ stdlib only."
  },
  "cards": [
    {
      "kind": "INV",
      "title": "Stdlib only core",
      "statement": "Never add third-party packages to runtime.",
      "rationale": "Zero-install reliability.",
      "immutable": true,
      "always_include": true,
      "anchors": ["app/auth.py:AuthService"]
    }
  ],
  "tasks": [
    {
      "id": "T1",
      "title": "Harden AuthService.login",
      "description": "Validate email and password strength cleanly.",
      "priority": 1,
      "depends_on": [],
      "anchors": ["app/auth.py:AuthService"],
      "acceptance_criteria": ["Empty passwords raise PermissionError"]
    }
  ]
}
</ctx_bootstrap>""",
        )
        code, out, err = self._run_cli(
            ["ingest", "--file", str(bootstrap_response_path), "--bootstrap", "--auto"]
        )
        self.assertEqual(code, 0, msg=err)
        self.assertIn("Cards: +1 created", out)
        self.assertIn("Tasks: +1 created", out)

        # 3. ctx compile TASK-001 --no-clip
        code, out, err = self._run_cli(
            ["compile", "TASK-001", "--no-clip", "--budget", "4000"]
        )
        self.assertEqual(code, 0, msg=err)
        capsule_path = self.root / ".ctx" / "out" / "capsule.md"
        self.assertTrue(capsule_path.is_file())
        capsule_text = capsule_path.read_text(encoding="utf-8")
        self.assertIn("app/auth.py", capsule_text)
        self.assertIn("INV-001", capsule_text)
        self.assertIn("<ctx_protocol>", capsule_text)
        self.assertIn("Full:", out)
        self.assertIn("Capsule:", out)

        # Check compact handoff prompt generation (~80-115 tokens)
        code, out, err = self._run_cli(["handoff", "TASK-001", "--no-clip"])
        self.assertEqual(code, 0, msg=err)
        self.assertIn("<ctx_delta>", out)
        self.assertLessEqual(estimate_tokens(out), 150)

        # 4. Simulated AI Studio <ctx_delta> -> ctx ingest --file ... --auto
        delta_path = self.root / "ai_delta.txt"
        _write_text(
            delta_path,
            """Готово! Изменения внесены.
<ctx_delta>
{
"task_id": "TASK-001",
"task_status": "done",
"decisions_added": [
{
"kind": "ADR",
"title": "Prefix JWT strings with jwt:",
"statement": "Issue tokens formatted as jwt:<email> in prototype stage.",
"rationale": "Simple verification contract.",
"tags": ["auth", "jwt"],
"anchors": ["app/auth.py:AuthService.login"]
},
{
"kind": "PITFALL",
"title": "Do not bypass empty password check",
"statement": "Never return a token when password is empty string.",
"rationale": "Rejected security bypass.",
"tags": ["auth", "security"],
"anchors": ["app/auth.py"]
}
],
"next_anchor": "Add token expiration check"
}
</ctx_delta>""",
        )
        code, out, err = self._run_cli(["ingest", "--file", str(delta_path), "--auto"])
        self.assertEqual(code, 0, msg=err)
        self.assertIn("Cards: +2 created", out)
        self.assertIn("Tasks: +0 created, *1 updated", out)

        store = ContextStore(self.root)
        self.assertEqual(store.require_task("TASK-001").status, TaskStatus.DONE)
        self.assertEqual(store.require_card("ADR-001").kind, CardKind.ADR)
        self.assertEqual(store.require_card("PIT-001").kind, CardKind.PITFALL)

        # Also exercise task/card management commands
        code, out, err = self._run_cli(
            [
                "task",
                "add",
                "--title",
                "Add billing audit",
                "--anchors",
                "app/billing.py",
                "--deps",
                "TASK-001",
                "--ac",
                "Total >= 0",
                "--priority",
                "2",
            ]
        )
        self.assertEqual(code, 0, msg=err)
        self.assertIn("TASK-002", out)

        code, out, err = self._run_cli(["task", "list", "--status", "todo"])
        self.assertEqual(code, 0, msg=err)
        self.assertIn("Tasks (1):", out)
        self.assertIn("[TASK-002] (TODO,", out)
        self.assertNotIn("[TASK-001] (", out)

        code, out, err = self._run_cli(["task", "done", "TASK-002"])
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(store.require_task("TASK-002").status, TaskStatus.DONE)

        code, out, err = self._run_cli(
            [
                "card",
                "add",
                "--kind",
                "ADR",
                "--title",
                "Use integer cents",
                "--statement",
                "Represent money in cents.",
                "--rationale",
                "Avoid float drift.",
                "--anchors",
                "app/billing.py",
                "--tags",
                "billing",
            ]
        )
        self.assertEqual(code, 0, msg=err)
        self.assertIn("ADR-002", out)

        code, out, err = self._run_cli(["card", "list"])
        self.assertEqual(code, 0, msg=err)
        self.assertIn("INV-001", out)
        self.assertIn("ADR-001", out)
        self.assertIn("PIT-001", out)

        # 5. Modify anchored file on disk -> ctx doctor detects reality drift (NEEDS_REVIEW)
        _write_text(
            self.root / "app" / "auth.py",
            '''"""JWT authentication service (modified)."""
from app.db import Database


class AuthService:
    """Handles login and token issuance."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def login(self, email: str, password: str) -> str:
        """Authenticate credentials and issue signed v2 token."""
        user = self._db.get_user(email)
        if not user or len(password) < 4:
            raise PermissionError("Invalid login")
        return f"jwt_v2:{email}"
''',
        )
        code, out, err = self._run_cli(["doctor"])
        self.assertEqual(code, 0, msg=err)
        self.assertIn("NEEDS_REVIEW", out)
        self.assertIn("ADR-001", out)

        # 6. ctx status prints accurate dashboard summary
        code, out, err = self._run_cli(["status"])
        self.assertEqual(code, 0, msg=err)
        self.assertIn("ctxforge status: e2e-app", out)
        self.assertIn("INV=1, ADR=2, PITFALL=1", out)
        self.assertIn("DONE=2", out)
        self.assertIn("Last capsule:", out)
        self.assertIn("NEEDS_REVIEW", out)

        # Verify refresh-hashes clears drift cleanly
        code, out, err = self._run_cli(["doctor", "--refresh-hashes"])
        self.assertEqual(code, 0, msg=err)
        self.assertIn("Auto-fixed anchor hashes", out)

        code, out, err = self._run_cli(["doctor"])
        self.assertEqual(code, 0, msg=err)
        self.assertIn("HEALTHY", out)


class TestClipboardAndOptionalAdapters(unittest.TestCase):
    """Tests for headless-safe clipboard helpers and mocked Gemini/Ollama adapters."""

    def test_clipboard_graceful_fallback_when_no_tools(self) -> None:
        with patch("ctxforge.clipboard.shutil.which", return_value=None):
            self.assertFalse(copy_to_clipboard("hello"))
            self.assertIsNone(read_from_clipboard())

    def test_clipboard_success_when_tool_available(self) -> None:
        mock_proc = MagicMock(returncode=0, stdout="copied-text")
        with (
            patch("ctxforge.clipboard.shutil.which", return_value="/usr/bin/pbcopy"),
            patch("ctxforge.clipboard.subprocess.run", return_value=mock_proc),
        ):
            self.assertTrue(copy_to_clipboard("copied-text"))
            self.assertEqual(read_from_clipboard(), "copied-text")

    def test_gemini_adapter_requires_key_and_handles_429(self) -> None:
        adapter = GeminiFreeAdapter(model="gemini-2.5-flash")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(MissingApiKeyError):
                adapter.complete("hello")

        err_fp = io.BytesIO(b'{"error": {"status": "RESOURCE_EXHAUSTED"}}')
        http_429 = urllib.error.HTTPError(
            url="https://example.com",
            code=429,
            msg="Too Many Requests",
            hdrs={},  # type: ignore[arg-type]
            fp=err_fp,
        )
        try:
            with (
                patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True),
                patch("urllib.request.urlopen", side_effect=http_429),
            ):
                with self.assertRaises(RateLimitError) as ctx:
                    adapter.complete("hello")
                self.assertIn("RESOURCE_EXHAUSTED", str(ctx.exception))
        finally:
            http_429.close()

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "<ctx_delta>{}</ctx_delta>"}]
                        }
                    }
                ]
            }
        ).encode("utf-8")
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_resp
        mock_cm.__exit__.return_value = False
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True),
            patch("urllib.request.urlopen", return_value=mock_cm),
        ):
            result = auto_extract_delta_from_raw_chat(
                "We decided to use JWT.",
                adapter,
            )
            self.assertIn("<ctx_delta>", result)

    def test_ollama_adapter_parses_response(self) -> None:
        adapter = OllamaAdapter(model="qwen3")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"response": '<ctx_delta>{"next_anchor":"Test"}</ctx_delta>'}
        ).encode("utf-8")
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_resp
        mock_cm.__exit__.return_value = False
        with patch("urllib.request.urlopen", return_value=mock_cm):
            out = adapter.complete("Extract delta")
            self.assertIn('"next_anchor":"Test"', out)


if __name__ == "__main__":
    unittest.main()
