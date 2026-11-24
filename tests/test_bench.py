"""Unit and integration tests for BenchmarkSuite, CLI `ctx bench`, and reporting."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ctxforge import cli
from ctxforge.bench import BenchmarkSuite, STRATEGIES, create_sample_project
from ctxforge.llm_adapter import LLMAdapter
from ctxforge.storage import ContextStore


class _FakeBenchAdapter(LLMAdapter):
    """Deterministic mock LLM adapter returning degraded vs compliant outputs by prompt."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete(self, prompt: str, system_prompt: str = "") -> str:
        del system_prompt
        self.calls.append(prompt)

        # Differentiate responses cleanly based on prompt strategy markers.
        if "<project_invariants>" in prompt and "<rejected_approaches>" in prompt:
            # Clean ctxforge_compiled response: obeys INV, avoids MD5/XML cookies, cites real files.
            return """We keep zero external packages (INV-001), avoid MD5 and XML cookies, and extend AuthService in app/auth.py using sign_payload from app/crypto.py:
```python
from app.crypto import sign_payload, verify_signature
from app.db import UserRepository

def issue_expiring_token(user_id: str, expires_at: int, secret: str) -> str:
    payload = f"{user_id}:{expires_at}"
    return f"{payload}.{sign_payload(payload, secret)}"
```"""

        if "=== ACCUMULATED CONVERSATION HISTORY ===" in prompt:
            # Noisy long-chat response: relapses into MD5 pitfall, XML cookies, pydantic, and fake path.
            return """Based on our earlier discussion, let's store sessions in XML cookies using app/legacy_cookies.py:
```python
import hashlib
import pydantic

def make_legacy_cookie(email: str) -> str:
    return hashlib.md5(email.encode()).hexdigest()
```"""

        # baseline_full_dump: obeys mostly, slightly verbose or partial reference.
        return """Update app/auth.py (AuthService.issue_token) and app/crypto.py (sign_payload):
```python
from app.crypto import sign_payload

def make_token(user_id: str, secret: str) -> str:
    return f"{user_id}.{sign_payload(user_id, secret)}"
```"""


class TestBenchmarkSuite(unittest.TestCase):
    """Tests for prepare, score, automated run, sample sandbox, and CLI bench commands."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.sample_dir = Path(self._tmp.name).resolve() / "sandbox"
        create_sample_project(self.sample_dir)

        # Add extra modules so full dump and long chat realistically exceed ctxforge by >4x.
        for idx in range(6):
            extra = self.sample_dir / "app" / f"extra_service_{idx}.py"
            body = "\n".join(
                f'''def process_batch_{idx}_{j}(items: list[str]) -> list[str]:
    """Process batch {j} of unrelated subsystem {idx}."""
    cleaned = [item.strip().lower() for item in items if item.strip()]
    return [f"{idx}:{j}:{{item}}" for item in cleaned]
'''
                for j in range(7)
            )
            extra.write_text(
                f'"""Unrelated service {idx}."""\n\n{body}',
                encoding="utf-8",
            )

        self.store = ContextStore(self.sample_dir)
        self.suite = BenchmarkSuite(self.store)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_prepare_and_score_manual_ab_with_relapses_vs_ctxforge(self) -> None:
        bench_dir = self.sample_dir / ".ctx" / "bench"
        rubric = self.suite.prepare(
            task_ids=["TASK-001"],
            bench_dir=bench_dir,
            token_budget=1800,
            test_command="python -m unittest discover -s tests",
        )
        self.assertTrue((bench_dir / "rubric.json").is_file())
        entries = rubric["tasks"][0]["entries"]
        for strat in STRATEGIES:
            self.assertIn(strat, entries)
            self.assertTrue((bench_dir / entries[strat]["prompt_file"]).is_file())

        full_tok = entries["baseline_full_dump"]["input_tokens"]
        chat_tok = entries["baseline_long_chat"]["input_tokens"]
        ctx_tok = entries["ctxforge_compiled"]["input_tokens"]

        # ctxforge capsule uses < 25% of full dump / long chat tokens and achieves higher SNR.
        self.assertLess(ctx_tok, full_tok * 0.25)
        self.assertLess(ctx_tok, chat_tok * 0.25)
        self.assertGreater(
            entries["ctxforge_compiled"]["snr"],
            entries["baseline_full_dump"]["snr"],
        )
        self.assertGreater(
            entries["ctxforge_compiled"]["snr"],
            entries["baseline_long_chat"]["snr"],
        )

        # Simulate manual AI Studio answers saved into responses/.
        adapter = _FakeBenchAdapter()
        for strat in STRATEGIES:
            prompt_text = (bench_dir / entries[strat]["prompt_file"]).read_text(
                encoding="utf-8"
            )
            resp = adapter.complete(prompt_text)
            (bench_dir / entries[strat]["response_file"]).write_text(
                resp, encoding="utf-8"
            )

        report = self.suite.score(bench_dir=bench_dir)
        self.assertTrue(report.report_path.is_file())
        by_strat = {score.strategy: score for score in report.scores}

        # baseline_long_chat relapsed into rejected MD5 & XML cookies, violated INV-001,
        # and hallucinated a path.
        chat_score = by_strat["baseline_long_chat"]
        self.assertGreaterEqual(chat_score.relapse_count, 2)
        self.assertLess(chat_score.adherence_score, 1.0)
        self.assertLess(chat_score.anchor_accuracy, 1.0)
        self.assertIn("app/legacy_cookies.py", chat_score.hallucinated_refs)

        # ctxforge_compiled passed every check cleanly.
        ctx_score = by_strat["ctxforge_compiled"]
        self.assertEqual(ctx_score.relapse_count, 0)
        self.assertEqual(ctx_score.violations, [])
        self.assertEqual(ctx_score.adherence_score, 1.0)
        self.assertEqual(ctx_score.anchor_accuracy, 1.0)
        self.assertTrue(ctx_score.exec_passed)

        md = report.report_markdown
        self.assertIn("# ctxforge A/B Benchmark Report", md)
        self.assertIn("Input Token Footprint", md)
        self.assertIn("H1 (Token Footprint & Signal-to-Noise): CONFIRMED", md)
        self.assertIn("H2 (Truth Governance & Relapse Prevention): CONFIRMED", md)
        self.assertIn("H3 (Grounding & Anchor Accuracy): CONFIRMED", md)

    def test_automated_run_mode_with_mock_adapter(self) -> None:
        bench_dir = self.sample_dir / ".ctx" / "bench_auto"
        adapter = _FakeBenchAdapter()
        report = self.suite.run(
            adapter,
            task_ids=["TASK-001"],
            bench_dir=bench_dir,
            token_budget=2500,
        )
        self.assertEqual(len(adapter.calls), 3)
        self.assertTrue((bench_dir / "benchmark_report.md").is_file())
        rubric_data = json.loads(
            (bench_dir / "rubric.json").read_text(encoding="utf-8")
        )
        for strat in STRATEGIES:
            entry = rubric_data["tasks"][0]["entries"][strat]
            self.assertIsNotNone(entry["latency_ms"])
            self.assertTrue(entry["response_text"])

        self.assertEqual(
            report.strategy_aggregates["ctxforge_compiled"]["total_relapses"], 0
        )
        self.assertGreater(
            report.strategy_aggregates["baseline_long_chat"]["total_relapses"], 0
        )

    def test_cli_bench_subcommands_end_to_end(self) -> None:
        cli_sandbox = Path(self._tmp.name).resolve() / "cli_sample"

        def run_cli(argv: list[str]) -> tuple[int, str, str]:
            out_buf = io.StringIO()
            err_buf = io.StringIO()
            with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(
                err_buf
            ):
                code = cli.main(argv)
            return code, out_buf.getvalue(), err_buf.getvalue()

        # 1. ctx bench init-sample
        code, out, err = run_cli(["bench", "init-sample", "--dir", str(cli_sandbox)])
        self.assertEqual(code, 0, msg=err)
        self.assertIn("Created calibration benchmark sandbox", out)
        self.assertTrue((cli_sandbox / ".ctx" / "project.json").is_file())

        # 2. ctx --root <sandbox> bench prepare
        code, out, err = run_cli(
            ["--root", str(cli_sandbox), "bench", "prepare", "--tasks", "TASK-001"]
        )
        self.assertEqual(code, 0, msg=err)
        self.assertIn("ctxforge Benchmark Prepared", out)
        self.assertIn("ctxforge_compiled", out)

        # 3. ctx --root <sandbox> bench run --provider ollama (mocked adapter)
        with patch(
            "ctxforge.cli._build_llm_adapter", return_value=_FakeBenchAdapter()
        ):
            code, out, err = run_cli(
                [
                    "--root",
                    str(cli_sandbox),
                    "bench",
                    "run",
                    "--provider",
                    "ollama",
                    "--tasks",
                    "TASK-001",
                ]
            )
        self.assertEqual(code, 0, msg=err)
        self.assertIn("ctxforge Benchmark Results", out)
        self.assertTrue(
            (cli_sandbox / ".ctx" / "bench" / "benchmark_report.md").is_file()
        )

        # 4. ctx --root <sandbox> bench score
        code, out, err = run_cli(
            [
                "--root",
                str(cli_sandbox),
                "bench",
                "score",
                "--test-cmd",
                "python -m unittest discover -s tests",
            ]
        )
        self.assertEqual(code, 0, msg=err)
        self.assertIn("ctxforge Benchmark Results", out)
        report_txt = (
            cli_sandbox / ".ctx" / "bench" / "benchmark_report.md"
        ).read_text(encoding="utf-8")
        self.assertIn("Test command status: PASSED", report_txt)


if __name__ == "__main__":
    unittest.main()
