"""Unit tests for RepoIndexer, AST skeletonization, ignore rules, and git co-change."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from ctxforge import indexer
from ctxforge.indexer import RepoIndexer, extract_git_cochange
from ctxforge.models import ProjectMeta
from ctxforge.storage import ContextStore


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestRepoIndexer(unittest.TestCase):
    """Comprehensive tests for incremental indexing, skeletons, imports, and co-change."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.store = ContextStore(self.root)
        self.store.init(
            ProjectMeta(name="indexer-demo", vision="Synthetic test repository")
        )

        # Ignore rules (.gitignore + .ctxignore)
        _write_text(
            self.root / ".gitignore",
            "*.log\nignored_dir/\n/root_only.txt\n!important.log\n",
        )
        _write_text(self.root / ".ctxignore", "*.snap\n")

        # Ignored / secret / oversized / binary files that must NOT enter index
        _write_text(self.root / ".env", "API_SECRET=supersecret\n")
        _write_text(self.root / ".env.production", "DB_PASS=secret\n")
        _write_text(self.root / "private.pem", "-----BEGIN RSA PRIVATE KEY-----\n")
        _write_text(self.root / "id_rsa", "SECRET KEY\n")
        _write_text(self.root / "debug.log", "ignored log\n")
        _write_text(self.root / "important.log", "negated ignore rule keeps this\n")
        _write_text(self.root / "ignored_dir" / "junk.py", "def junk(): pass\n")
        _write_text(self.root / "root_only.txt", "ignored at root\n")
        _write_text(self.root / "sub" / "root_only.txt", "kept in subdir\n")
        _write_text(self.root / "ui.snap", "snapshot ignored by .ctxignore\n")
        _write_text(self.root / "node_modules" / "pkg" / "index.js", "export const x = 1;\n")
        _write_text(self.root / "huge.txt", "a" * (513 * 1024))
        (self.root / "blob.bin").write_bytes(b"\x00\x01\x02binarydata")

        # Python package with absolute and relative imports + heavy bodies for >50% skeleton savings
        _write_text(self.root / "app" / "__init__.py", "")
        _write_text(
            self.root / "app" / "db.py",
            '''"""Database abstraction layer."""

from dataclasses import dataclass


@dataclass(slots=True)
class User:
    """Domain user entity."""

    id: int
    email: str
    is_active: bool = True


class Database:
    """Lightweight database gateway with connection pooling."""

    def __init__(self, dsn: str, pool_size: int = 10) -> None:
        self.dsn = dsn
        self.pool_size = pool_size
        self._cache: dict[str, User] = {}

    def get_user(self, email: str) -> User | None:
        """Fetch user by normalized email address."""
        normalized = email.strip().lower()
        if not normalized:
            raise ValueError("email cannot be empty")
        if normalized in self._cache:
            return self._cache[normalized]
        # Simulated lookup logic with multiple branches
        for candidate_id in range(1, 5):
            if f"user{candidate_id}@example.com" == normalized:
                user = User(id=candidate_id, email=normalized)
                self._cache[normalized] = user
                return user
        return None
''',
        )

        _write_text(
            self.root / "app" / "utils.py",
            '''"""Cryptographic and token helpers."""


def hash_password(raw_password: str, *, rounds: int = 12) -> str:
    """Hash a plaintext password deterministically for testing."""
    salt = f"salt-{rounds}"
    acc = raw_password
    for step in range(rounds):
        acc = f"{salt}:{step}:{acc}"
        if len(acc) > 128:
            acc = acc[:128]
    return acc
''',
        )

        _write_text(
            self.root / "app" / "auth.py",
            '''"""Authentication service module."""

from dataclasses import dataclass
from app.db import Database, User
from .utils import hash_password


@dataclass(slots=True)
class TokenPair:
    """Access and refresh JWT pair."""

    access_token: str
    refresh_token: str


class AuthService:
    """Handles JWT auth, login validation, and token refresh."""

    def __init__(self, db: Database, secret: str) -> None:
        self._db = db
        self._secret = secret

    async def login(
        self,
        email: str,
        password: str,
        *,
        remember_me: bool = False,
    ) -> TokenPair:
        """Authenticate a user and return signed tokens."""
        user = self._db.get_user(email)
        if user is None:
            raise PermissionError("Invalid credentials")
        digest = hash_password(password, rounds=10)
        ttl_tag = "long" if remember_me else "short"
        access = f"acc:{user.id}:{ttl_tag}:{digest[:12]}"
        refresh = f"ref:{user.id}:{self._secret[:6]}:{digest[-8:]}"
        if not user.is_active:
            raise PermissionError("User disabled")
        return TokenPair(access_token=access, refresh_token=refresh)

    def verify(self, token: str, *scopes: str, **flags: bool) -> User | None:
        """Verify access token signature and optional scopes."""
        if not token or ":" not in token:
            return None
        parts = token.split(":")
        if len(parts) < 3 or parts[0] != "acc":
            return None
        email = f"user{parts[1]}@example.com"
        if flags.get("strict") and not scopes:
            return None
        return self._db.get_user(email)
''',
        )

        _write_text(
            self.root / "app" / "api.py",
            '''"""HTTP route layer consuming AuthService."""

from app.auth import AuthService


def handle_login(service: AuthService, payload: dict[str, str]) -> dict[str, str]:
    """Dispatch login request."""
    email = payload.get("email", "")
    password = payload.get("password", "")
    if not email or not password:
        return {"error": "missing_fields"}
    return {"status": "ok"}
''',
        )

        # TypeScript modules with relative imports & multi-line signatures
        _write_text(
            self.root / "web" / "types.ts",
            """export interface ApiUser {
  id: string;
  email: string;
}

export type AuthStatus = 'authenticated' | 'anonymous';
""",
        )
        _write_text(
            self.root / "web" / "client.ts",
            """import { ApiUser, AuthStatus } from './types';

/** Client for authentication endpoints. */
export class AuthClient {
  private readonly baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  async fetchProfile(
    userId: string,
    includeRoles: boolean = false,
  ): Promise<ApiUser | null> {
    const url = `${this.baseUrl}/users/${userId}?roles=${includeRoles}`;
    if (!userId) {
      return null;
    }
    return { id: userId, email: url };
  }
}

export const checkStatus = async (status: AuthStatus): Promise<boolean> => {
  return status === 'authenticated';
};
""",
        )

        # Markdown documentation (including # comment inside fenced code block)
        _write_text(
            self.root / "README.md",
            """# Project Overview

Introductory paragraph.

```python
# Not a markdown heading inside code block
def demo():
    pass
```

## Architecture

Details about components.

### Truth Governance

Details about immutable cards.
""",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_security_and_ignore_filtering(self) -> None:
        indexer_inst = RepoIndexer(self.root, store=self.store)
        index = indexer_inst.build_or_update()

        self.assertNotIn(".env", index.files)
        self.assertNotIn(".env.production", index.files)
        self.assertNotIn("private.pem", index.files)
        self.assertNotIn("id_rsa", index.files)
        self.assertNotIn("debug.log", index.files)
        self.assertNotIn("ignored_dir/junk.py", index.files)
        self.assertNotIn("root_only.txt", index.files)
        self.assertNotIn("ui.snap", index.files)
        self.assertNotIn("node_modules/pkg/index.js", index.files)
        self.assertNotIn("huge.txt", index.files)
        self.assertNotIn("blob.bin", index.files)
        self.assertFalse(any(p.startswith(".ctx/") for p in index.files))

        # Negation and anchored root_only rule verification
        self.assertIn("important.log", index.files)
        self.assertIn("sub/root_only.txt", index.files)
        self.assertIn("app/auth.py", index.files)
        self.assertIn("web/client.ts", index.files)
        self.assertIn("README.md", index.files)

    def test_incremental_caching_avoids_reparsing_unchanged_files(self) -> None:
        indexer_inst = RepoIndexer(self.root, store=self.store)

        with patch(
            "ctxforge.indexer._parse_file_record",
            wraps=indexer._parse_file_record,
        ) as parse_spy:
            first_index = indexer_inst.build_or_update()
            first_calls = parse_spy.call_count
            self.assertEqual(first_calls, len(first_index.files))
            self.assertTrue(indexer_inst.cache_path.is_file())

            # Second run with no changes must parse 0 files.
            second_index = indexer_inst.build_or_update()
            self.assertEqual(parse_spy.call_count, first_calls)
            self.assertEqual(set(second_index.files.keys()), set(first_index.files.keys()))

            # Touching a file without changing content updates mtime_ns without re-parsing.
            utils_path = self.root / "app" / "utils.py"
            stat = utils_path.stat()
            os.utime(utils_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
            indexer_inst.build_or_update()
            self.assertEqual(parse_spy.call_count, first_calls)

            # Modifying a single file re-parses ONLY that modified file.
            _write_text(
                utils_path,
                utils_path.read_text(encoding="utf-8")
                + "\ndef verify_hash(raw: str, digest: str) -> bool:\n    return True\n",
            )
            third_index = indexer_inst.build_or_update()
            self.assertEqual(parse_spy.call_count, first_calls + 1)
            self.assertTrue(
                any(sym.name == "verify_hash" for sym in third_index.files["app/utils.py"].symbols)
            )

            # Deleting a file removes it from the index cleanly.
            (self.root / "sub" / "root_only.txt").unlink()
            fourth_index = indexer_inst.build_or_update()
            self.assertNotIn("sub/root_only.txt", fourth_index.files)
            self.assertEqual(parse_spy.call_count, first_calls + 1)

            # force=True re-parses all files.
            indexer_inst.build_or_update(force=True)
            self.assertEqual(
                parse_spy.call_count,
                first_calls + 1 + len(fourth_index.files),
            )

    def test_python_ts_and_markdown_skeletons_and_token_savings(self) -> None:
        indexer_inst = RepoIndexer(self.root, store=self.store)
        index = indexer_inst.build_or_update()

        auth_rec = index.files["app/auth.py"]
        self.assertEqual(auth_rec.language, "python")
        # Significant Tier-2 compression (>50% token savings)
        savings_ratio = 1.0 - (auth_rec.skeleton_tokens / auth_rec.token_count)
        self.assertGreater(savings_ratio, 0.5)

        self.assertIn("from app.db import Database, User", auth_rec.imports)
        self.assertIn("from .utils import hash_password", auth_rec.imports)

        by_name = {(s.parent, s.name): s for s in auth_rec.symbols}
        self.assertEqual(by_name[(None, "TokenPair")].kind, "class")
        self.assertEqual(by_name[(None, "AuthService")].kind, "class")
        self.assertEqual(by_name[("AuthService", "login")].kind, "method")
        self.assertIn(
            "async def login(self, email: str, password: str, *, remember_me: bool = False) -> TokenPair",
            by_name[("AuthService", "login")].signature,
        )
        self.assertIn(
            "def verify(self, token: str, *scopes: str, **flags: bool) -> User | None",
            by_name[("AuthService", "verify")].signature,
        )
        self.assertIn(
            "Handles JWT auth, login validation, and token refresh.",
            auth_rec.skeleton,
        )
        self.assertIn("@dataclass(slots=True)", auth_rec.skeleton)

        # TypeScript structural extraction
        ts_types = index.files["web/types.ts"]
        self.assertEqual(
            [(s.name, s.kind) for s in ts_types.symbols],
            [("ApiUser", "interface"), ("AuthStatus", "type")],
        )

        ts_client = index.files["web/client.ts"]
        self.assertIn("import { ApiUser, AuthStatus } from './types'", ts_client.imports)
        ts_syms = {(s.parent, s.name): s for s in ts_client.symbols}
        self.assertEqual(ts_syms[(None, "AuthClient")].kind, "class")
        self.assertEqual(ts_syms[("AuthClient", "fetchProfile")].kind, "method")
        self.assertIn("Promise<ApiUser | null>", ts_syms[("AuthClient", "fetchProfile")].signature)
        self.assertEqual(ts_syms[(None, "checkStatus")].kind, "function")

        # Markdown headings extraction (ignoring heading-like comments inside code fences)
        md_rec = index.files["README.md"]
        self.assertEqual(md_rec.language, "markdown")
        headings = [(s.name, s.kind, s.parent) for s in md_rec.symbols]
        self.assertEqual(
            headings,
            [
                ("Project Overview", "heading", None),
                ("Architecture", "heading", "Project Overview"),
                ("Truth Governance", "heading", "Architecture"),
            ],
        )
        self.assertNotIn("Not a markdown heading", md_rec.skeleton)

    def test_import_and_reverse_graph_via_get_neighbors(self) -> None:
        indexer_inst = RepoIndexer(self.root, store=self.store)
        index = indexer_inst.build_or_update()

        self.assertEqual(
            sorted(index.import_graph["app/auth.py"]),
            ["app/db.py", "app/utils.py"],
        )
        self.assertEqual(index.import_graph["app/api.py"], ["app/auth.py"])
        self.assertEqual(index.reverse_graph["app/auth.py"], ["app/api.py"])
        self.assertEqual(index.import_graph["web/client.ts"], ["web/types.ts"])
        self.assertEqual(index.reverse_graph["web/types.ts"], ["web/client.ts"])

        # 1-hop neighbors of app/auth.py
        neighbors_1 = indexer_inst.get_neighbors(["app/auth.py"], hops=1)
        by_path_1 = {path: (score, reason) for path, score, reason in neighbors_1}
        self.assertEqual(
            set(by_path_1.keys()),
            {"app/db.py", "app/utils.py", "app/api.py"},
        )
        self.assertEqual(by_path_1["app/db.py"][1], "imports")
        self.assertEqual(by_path_1["app/utils.py"][1], "imports")
        self.assertEqual(by_path_1["app/api.py"][1], "imported_by")
        self.assertGreater(by_path_1["app/db.py"][0], by_path_1["app/api.py"][0])

        # 1-hop neighbors of app/api.py -> only app/auth.py
        api_1hop = indexer_inst.get_neighbors(["app/api.py"], hops=1)
        self.assertEqual([item[0] for item in api_1hop], ["app/auth.py"])

        # 2-hop neighbors of app/api.py -> reaches app/db.py and app/utils.py
        api_2hop = indexer_inst.get_neighbors(["app/api.py"], hops=2)
        self.assertEqual(
            {item[0] for item in api_2hop},
            {"app/auth.py", "app/db.py", "app/utils.py"},
        )

    def test_git_cochange_graceful_without_repo_and_accurate_with_repo(self) -> None:
        # Without .git repo, extract_git_cochange returns {} gracefully.
        self.assertEqual(extract_git_cochange(self.root), {})
        indexer_inst = RepoIndexer(self.root, store=self.store)
        no_git_index = indexer_inst.build_or_update()
        self.assertEqual(no_git_index.cochange_graph, {})

        if shutil.which("git") is None:
            self.skipTest("git binary not installed in environment")

        git_env = dict(os.environ)
        git_env["GIT_CONFIG_GLOBAL"] = os.devnull
        git_env["GIT_CONFIG_NOSYSTEM"] = "1"

        def run_git(*args: str) -> None:
            subprocess.run(
                ["git", "-C", str(self.root), *args],
                env=git_env,
                capture_output=True,
                text=True,
                check=True,
            )

        run_git("init")
        run_git("config", "user.name", "Ctx Test")
        run_git("config", "user.email", "ctx@example.com")
        run_git("config", "commit.gpgsign", "false")

        # Commit 1: app/auth.py + web/client.ts (unrelated by imports, linked by co-change)
        run_git("add", "app/auth.py", "web/client.ts")
        run_git("commit", "-m", "feat: auth and client sync")

        # Commit 2: modify both again to strengthen co-change weight
        _write_text(
            self.root / "app" / "auth.py",
            (self.root / "app" / "auth.py").read_text(encoding="utf-8") + "\n# touch\n",
        )
        _write_text(
            self.root / "web" / "client.ts",
            (self.root / "web" / "client.ts").read_text(encoding="utf-8") + "\n// touch\n",
        )
        run_git("add", "app/auth.py", "web/client.ts")
        run_git("commit", "-m", "fix: align client payload with auth")

        # Commit 3: giant commit (>20 files) must be excluded from cochange_graph
        bulk_files: list[str] = []
        for idx in range(22):
            rel = f"bulk/file_{idx:02d}.py"
            _write_text(self.root / rel, f"VALUE = {idx}\n")
            bulk_files.append(rel)
        run_git("add", *bulk_files)
        run_git("commit", "-m", "chore: bulk generated files")

        git_index = indexer_inst.build_or_update()
        self.assertEqual(git_index.cochange_graph["app/auth.py"]["web/client.ts"], 2)
        self.assertEqual(git_index.cochange_graph["web/client.ts"]["app/auth.py"], 2)
        self.assertNotIn("bulk/file_00.py", git_index.cochange_graph)

        neighbors = indexer_inst.get_neighbors(["app/auth.py"], hops=1)
        by_path = {path: reason for path, _, reason in neighbors}
        self.assertEqual(by_path.get("web/client.ts"), "git_cochange")

if __name__ == "__main__":
    unittest.main()