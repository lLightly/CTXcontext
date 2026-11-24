"""Command-line interface (`ctx`) for ctxforge."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence, TextIO

from .bench import BenchmarkSuite, create_sample_project
from .bootstrap import BootstrapGenerator
from .clipboard import copy_to_clipboard, read_from_clipboard
from .compiler import CompileRequest, CompileResult, ContextCompiler
from .doctor import ContextDoctor, HealthReport
from .indexer import RepoIndexer
from .ingest import DeltaIngestor, DeltaPlan
from .llm_adapter import (
    GeminiFreeAdapter,
    LLMAdapter,
    OllamaAdapter,
    auto_extract_delta_from_raw_chat,
)
from .models import (
    Anchor,
    CardKind,
    KnowledgeCard,
    ProjectMeta,
    TaskNode,
    TaskStatus,
    utc_now_iso,
)
from .storage import ContextStore, CtxForgeError
from .tokens import estimate_tokens


class _Console:
    """Minimal ANSI-aware output helper that automatically disables color in non-TTY."""

    def __init__(
        self,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
    ) -> None:
        self.stdout: TextIO = sys.stdout if stdout is None else stdout
        self.stderr: TextIO = sys.stderr if stderr is None else stderr
        isatty = hasattr(self.stdout, "isatty") and self.stdout.isatty()
        no_color = bool(os.environ.get("NO_COLOR")) or os.environ.get("TERM") == "dumb"
        self.use_color = bool(isatty and not no_color)

    def _style(self, text: str, code: str) -> str:
        if not self.use_color:
            return text
        return f"\033[{code}m{text}\033[0m"

    def bold(self, text: str) -> str:
        return self._style(text, "1")

    def dim(self, text: str) -> str:
        return self._style(text, "2")

    def green(self, text: str) -> str:
        return self._style(text, "32")

    def yellow(self, text: str) -> str:
        return self._style(text, "33")

    def red(self, text: str) -> str:
        return self._style(text, "31")

    def cyan(self, text: str) -> str:
        return self._style(text, "36")

    def out(self, text: str = "") -> None:
        self.stdout.write(text + "\n")

    def err(self, text: str) -> None:
        self.stderr.write(self.red("Error: ") + text + "\n")


def _split_csv_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    result: list[str] = []
    for raw in values:
        for item in raw.split(","):
            cleaned = item.strip()
            if cleaned:
                result.append(cleaned)
    return result


def _parse_anchors_arg(
    values: list[str] | None,
    store: ContextStore,
) -> list[Anchor]:
    anchors: list[Anchor] = []
    for raw in _split_csv_list(values):
        symbol: str | None = None
        path_part = raw
        if ":" in raw:
            left, right = raw.rsplit(":", 1)
            if left.strip() and right.strip() and "/" not in right and "\\" not in right:
                path_part = left
                symbol = right.strip()
        anchor = Anchor(file_path=path_part, symbol=symbol)
        if store.is_initialized():
            try:
                anchor.file_hash = store.compute_anchor_hash(
                    anchor.file_path, digest_length=16
                )
            except ValueError:
                pass
        anchors.append(anchor)
    return anchors


def _parse_card_kind_arg(raw_kind: str) -> CardKind:
    norm = raw_kind.strip().upper()
    if norm in ("INV", "INVARIANT"):
        return CardKind.INVARIANT
    if norm == "ADR":
        return CardKind.ADR
    if norm in ("PIT", "PITFALL"):
        return CardKind.PITFALL
    raise ValueError(f"Unsupported card kind '{raw_kind}'. Expected INV, ADR, or PITFALL.")


def _colorize_plan(plan: DeltaPlan, console: _Console) -> str:
    raw = plan.format_plan()
    if not console.use_color:
        return raw
    colored_lines: list[str] = []
    for line in raw.splitlines():
        if line.startswith("[+]"):
            colored_lines.append(console.green(line))
        elif line.startswith("[") and " SUPERSEDE " in line:
            colored_lines.append(console.cyan(line))
        elif line.startswith("[!]"):
            colored_lines.append(console.red(line))
        elif line.startswith("[*]"):
            colored_lines.append(console.yellow(line))
        elif line.startswith("[-]"):
            colored_lines.append(console.dim(line))
        elif line.startswith("==="):
            colored_lines.append(console.bold(line))
        else:
            colored_lines.append(line)
    return "\n".join(colored_lines)


def _colorize_report(report: HealthReport, console: _Console) -> str:
    raw = report.format_report()
    if not console.use_color:
        return raw
    colored_lines: list[str] = []
    for line in raw.splitlines():
        if line.startswith("[ERROR]"):
            colored_lines.append(console.red(line))
        elif line.startswith("[WARNING]"):
            colored_lines.append(console.yellow(line))
        elif "HEALTHY" in line:
            colored_lines.append(console.green(line))
        elif line.startswith("==="):
            colored_lines.append(console.bold(line))
        else:
            colored_lines.append(line)
    return "\n".join(colored_lines)


def _last_compile_meta_path(store: ContextStore) -> Path:
    return store.ctx_dir / "cache" / "last_compile.json"


def _save_last_compile_meta(
    store: ContextStore,
    request: CompileRequest,
    result: CompileResult,
) -> None:
    path = _last_compile_meta_path(store)
    payload = {
        "task_id": request.task_id,
        "capsule_tokens": result.capsule_tokens,
        "full_repo_tokens": result.full_repo_tokens,
        "savings_percent": result.savings_percent,
        "token_budget": request.token_budget,
        "timestamp": utc_now_iso(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def _load_last_compile_meta(store: ContextStore) -> dict[str, Any] | None:
    path = _last_compile_meta_path(store)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


# ----------------------------------------------------------------------
# Command Handlers
# ----------------------------------------------------------------------


def _cmd_init(args: argparse.Namespace, store: ContextStore, console: _Console) -> int:
    name = (args.name or store.root.name or "ctxforge-project").strip()
    vision = (args.vision or f"Software project {name}.").strip()
    meta = ProjectMeta(name=name, vision=vision)
    store.init(meta, force=args.force)

    indexer = RepoIndexer(store.root, store=store)
    index = indexer.build_or_update(force=True)
    total_tokens = sum(rec.token_count for rec in index.files.values())
    lang_counts = Counter(rec.language for rec in index.files.values())
    top_langs = ", ".join(f"{lang}={cnt}" for lang, cnt in lang_counts.most_common(5))

    console.out(console.green(f"Initialized ctxforge store at {store.ctx_dir}"))
    console.out(f"Project: {meta.name} | Indexed files: {len(index.files)} ({total_tokens} tok)")
    if top_langs:
        console.out(f"Languages: {top_langs}")
    return 0


def _cmd_bootstrap(args: argparse.Namespace, store: ContextStore, console: _Console) -> int:
    if args.brief == "-":
        brief_text = sys.stdin.read()
    else:
        brief_path = Path(args.brief)
        if not brief_path.is_absolute():
            brief_path = store.root / brief_path
        if not brief_path.is_file():
            console.err(f"Brief file not found: {brief_path}")
            return 1
        brief_text = brief_path.read_text(encoding="utf-8")

    if not brief_text.strip():
        console.err("Project brief text is empty.")
        return 1

    indexer = RepoIndexer(store.root, store=store)
    generator = BootstrapGenerator(store, indexer)
    prompt = generator.generate_bootstrap_prompt(
        brief_text=brief_text,
        token_budget=args.budget,
    )

    out_path = Path(args.out) if args.out else (store.out_dir / "bootstrap_prompt.md")
    if not out_path.is_absolute():
        out_path = store.root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(prompt, encoding="utf-8")

    prompt_tokens = estimate_tokens(prompt)
    copied = False if args.no_clip else copy_to_clipboard(prompt)
    console.out(
        console.green(f"Bootstrap prompt generated ({prompt_tokens} tok) -> {out_path}")
    )
    if not args.no_clip:
        if copied:
            console.out(
                "Copied to clipboard (paste into Google AI Studio, then run `ctx ingest --bootstrap`)."
            )
        else:
            console.out(
                console.yellow("Clipboard unavailable; copy the prompt from: ") + str(out_path)
            )
    return 0


def _cmd_task_add(args: argparse.Namespace, store: ContextStore, console: _Console) -> int:
    task_id = store.next_task_id()
    anchors = _parse_anchors_arg(args.anchors, store)
    deps = _split_csv_list(args.deps)
    criteria = [c.strip() for c in (args.ac or []) if c.strip()]
    node = TaskNode(
        id=task_id,
        title=args.title.strip(),
        description=(args.desc or "").strip(),
        priority=args.priority,
        depends_on=deps,
        anchors=anchors,
        acceptance_criteria=criteria,
    )
    store.add_task(node)
    console.out(console.green(f"Added {node.id}: {node.title} (priority={node.priority})"))
    return 0


def _cmd_task_list(args: argparse.Namespace, store: ContextStore, console: _Console) -> int:
    status_filter: TaskStatus | None = None
    if args.status and args.status.lower() != "all":
        status_filter = TaskStatus(args.status.upper())
    tasks = store.list_tasks(status=status_filter)
    if not tasks:
        console.out("No tasks found.")
        return 0
    console.out(console.bold(f"Tasks ({len(tasks)}):"))
    for task in tasks:
        deps = f" deps=[{','.join(task.depends_on)}]" if task.depends_on else ""
        line = f"  [{task.id}] ({task.status.value}, P{task.priority}) {task.title}{deps}"
        if task.status is TaskStatus.DONE:
            console.out(console.dim(line))
        elif task.status is TaskStatus.IN_PROGRESS:
            console.out(console.cyan(line))
        elif task.status is TaskStatus.BLOCKED:
            console.out(console.red(line))
        else:
            console.out(line)
    return 0


def _cmd_task_done(args: argparse.Namespace, store: ContextStore, console: _Console) -> int:
    task = store.require_task(args.task_id)
    task.status = TaskStatus.DONE
    store.update_task(task)
    console.out(console.green(f"Marked {task.id} as DONE: {task.title}"))
    return 0


def _cmd_card_add(args: argparse.Namespace, store: ContextStore, console: _Console) -> int:
    kind = _parse_card_kind_arg(args.kind)
    card_id = store.next_card_id(kind)
    anchors = _parse_anchors_arg(args.anchors, store)
    tags = _split_csv_list(args.tags)
    card = KnowledgeCard(
        id=card_id,
        kind=kind,
        title=args.title.strip(),
        statement=args.statement.strip(),
        rationale=args.rationale.strip(),
        immutable=bool(args.immutable),
        always_include=bool(args.always),
        tags=tags,
        anchors=anchors,
    )
    store.add_card(card)
    flags = []
    if card.immutable:
        flags.append("immutable")
    if card.always_include:
        flags.append("always")
    flag_txt = f" [{', '.join(flags)}]" if flags else ""
    console.out(console.green(f"Added {card.id} ({card.kind.value}){flag_txt}: {card.title}"))
    return 0


def _cmd_card_list(args: argparse.Namespace, store: ContextStore, console: _Console) -> int:
    cards = store.list_cards() if args.all else store.get_active_cards()
    if not cards:
        console.out("No knowledge cards found.")
        return 0
    label = "All cards" if args.all else "Active cards"
    console.out(console.bold(f"{label} ({len(cards)}):"))
    for card in cards:
        flags = []
        if card.immutable:
            flags.append("IMMUTABLE")
        if card.always_include:
            flags.append("ALWAYS")
        flag_str = f" [{','.join(flags)}]" if flags else ""
        sup = f" -> {card.superseded_by}" if card.superseded_by else ""
        line = (
            f"  [{card.id}] ({card.kind.value}, {card.status.value}{sup}){flag_str} "
            f"{card.title}: {card.statement}"
        )
        if card.status.value != "ACTIVE":
            console.out(console.dim(line))
        elif card.kind is CardKind.INVARIANT:
            console.out(console.cyan(line))
        elif card.kind is CardKind.PITFALL:
            console.out(console.yellow(line))
        else:
            console.out(line)
    return 0


def _cmd_compile(args: argparse.Namespace, store: ContextStore, console: _Console) -> int:
    indexer = RepoIndexer(store.root, store=store)
    compiler = ContextCompiler(store, indexer=indexer)
    request = CompileRequest(
        task_id=args.task_id,
        query=args.query,
        explicit_files=_split_csv_list(args.files),
        token_budget=args.budget,
        include_protocol=not args.no_protocol,
    )
    result = compiler.compile(request)
    out_path = Path(args.out) if args.out else (store.out_dir / "capsule.md")
    if not out_path.is_absolute():
        out_path = store.root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.capsule_text, encoding="utf-8")
    _save_last_compile_meta(store, request, result)

    copied = False if args.no_clip else copy_to_clipboard(result.capsule_text)
    savings_sign = (
        f"-{result.savings_percent:.1f}%"
        if result.savings_percent >= 0
        else f"+{abs(result.savings_percent):.1f}%"
    )
    console.out(
        console.green(
            f"Full: {result.full_repo_tokens} tok -> Capsule: {result.capsule_tokens} tok ({savings_sign})"
        )
    )
    console.out(f"Saved capsule -> {out_path}")
    if not args.no_clip:
        if copied:
            console.out("Copied capsule to clipboard (paste into Google AI Studio).")
        else:
            console.out(
                console.yellow("Clipboard unavailable; read capsule from: ") + str(out_path)
            )
    console.out(result.format_summary_report())
    s = result.stats
    if s.get("tier1_files"):
        console.out("Tier-1 full files: " + ", ".join(s["tier1_files"]))
    if s.get("tier2_files"):
        console.out("Tier-2 skeletons: " + ", ".join(s["tier2_files"]))
    return 0


def _cmd_handoff(args: argparse.Namespace, store: ContextStore, console: _Console) -> int:
    task_id = args.task_id or "..."
    if args.task_id and store.is_initialized():
        store.require_task(args.task_id)

    handoff_prompt = (
        "Handoff summary. Return ONLY valid JSON inside <ctx_delta>...</ctx_delta>:\n"
        f'{{"task_id":"{task_id}","task_status":"in_progress|done|blocked",'
        '"decisions_added":[{"kind":"ADR|INV|PITFALL","title":"...",'
        '"statement":"...","rationale":"...","tags":[],"anchors":["file.py"]}],'
        '"decisions_superseded":[],"new_subtasks":[],"next_anchor":"next step"}}'
    )
    if store.is_initialized():
        out_path = store.out_dir / "handoff_prompt.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(handoff_prompt + "\n", encoding="utf-8")

    tokens = estimate_tokens(handoff_prompt)
    copied = False if args.no_clip else copy_to_clipboard(handoff_prompt)
    console.out(console.bold(f"Handoff prompt (~{tokens} tok):"))
    console.out(handoff_prompt)
    if not args.no_clip:
        if copied:
            console.out(console.green("Copied handoff prompt to clipboard."))
        else:
            console.out(console.yellow("Clipboard unavailable; copy the prompt above."))
    return 0


def _build_llm_adapter(provider: str, model: str | None) -> LLMAdapter:
    norm = provider.strip().lower()
    if norm == "gemini":
        return GeminiFreeAdapter(model=model or "gemini-2.5-flash")
    if norm == "ollama":
        return OllamaAdapter(model=model or "qwen3")
    raise ValueError(f"Unsupported provider '{provider}'. Expected 'gemini' or 'ollama'.")


def _cmd_ingest(args: argparse.Namespace, store: ContextStore, console: _Console) -> int:
    raw_text: str | None = None
    if args.file:
        if args.file == "-":
            raw_text = sys.stdin.read()
        else:
            in_path = Path(args.file)
            if not in_path.is_absolute():
                in_path = store.root / in_path
            if not in_path.is_file():
                console.err(f"Input file not found: {in_path}")
                return 1
            raw_text = in_path.read_text(encoding="utf-8")
    else:
        raw_text = read_from_clipboard()
        if (raw_text is None or not raw_text.strip()) and not sys.stdin.isatty():
            raw_text = sys.stdin.read()

    if raw_text is None or not raw_text.strip():
        console.err(
            "No LLM response provided (clipboard is empty/unavailable and no --file specified)."
        )
        return 1

    if args.extract_via:
        adapter = _build_llm_adapter(args.extract_via, args.model)
        console.out(console.dim(f"Extracting <ctx_delta> via {args.extract_via}..."))
        raw_text = auto_extract_delta_from_raw_chat(raw_text, adapter)

    indexer = RepoIndexer(store.root, store=store)
    index = indexer.build_or_update()
    ingestor = DeltaIngestor(store)

    if args.bootstrap:
        if not args.auto:
            console.out(console.bold("Applying <ctx_bootstrap> to initialize project memory."))
            try:
                answer = input("Proceed with bootstrap import? [y/N]: ").strip().lower()
            except EOFError:
                answer = ""
            if answer not in ("y", "yes", "д", "да"):
                console.out("Aborted.")
                return 0
        summary = ingestor.apply_bootstrap(raw_text, index=index)
        console.out(console.green(summary.format_summary()))
        return 0

    plan = ingestor.build_delta_plan(raw_text, index=index)
    console.out(_colorize_plan(plan, console))
    if not args.auto:
        try:
            answer = input("Apply these changes? [y/N]: ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes", "д", "да"):
            console.out("Aborted.")
            return 0
    summary = ingestor.apply_plan(plan, force_immutable=args.force_immutable)
    console.out(console.green(summary.format_summary()))
    return 0


def _cmd_doctor(args: argparse.Namespace, store: ContextStore, console: _Console) -> int:
    indexer = RepoIndexer(store.root, store=store)
    doctor = ContextDoctor(store, indexer)
    report = doctor.diagnose(fix_hashes=args.refresh_hashes)
    console.out(_colorize_report(report, console))
    return 1 if report.has_errors else 0


def _cmd_status(args: argparse.Namespace, store: ContextStore, console: _Console) -> int:
    del args
    project = store.get_project()
    all_cards = store.list_cards()
    active_cards = [c for c in all_cards if c.status.value == "ACTIVE"]
    card_counts = Counter(c.kind for c in active_cards)
    tasks = store.list_tasks()
    task_counts = Counter(t.status for t in tasks)

    indexer = RepoIndexer(store.root, store=store)
    report = ContextDoctor(store, indexer).diagnose(fix_hashes=False)
    last_compile = _load_last_compile_meta(store)

    console.out(console.bold(f"=== ctxforge status: {project.name} ==="))
    console.out(f"Vision: {project.vision}")
    console.out(
        "Active cards: "
        f"INV={card_counts[CardKind.INVARIANT]}, "
        f"ADR={card_counts[CardKind.ADR]}, "
        f"PITFALL={card_counts[CardKind.PITFALL]} "
        f"(total stored={len(all_cards)})"
    )
    console.out(
        "Tasks: "
        f"TODO={task_counts[TaskStatus.TODO]}, "
        f"IN_PROGRESS={task_counts[TaskStatus.IN_PROGRESS]}, "
        f"BLOCKED={task_counts[TaskStatus.BLOCKED]}, "
        f"DONE={task_counts[TaskStatus.DONE]}"
    )
    open_tasks = [
        t
        for t in tasks
        if t.status in (TaskStatus.IN_PROGRESS, TaskStatus.TODO, TaskStatus.BLOCKED)
    ]
    if open_tasks:
        preview = ", ".join(f"{t.id} ({t.status.value})" for t in open_tasks[:5])
        console.out(f"Open queue: {preview}")
    if last_compile is not None:
        console.out(
            "Last capsule: "
            f"{last_compile.get('capsule_tokens')} / {last_compile.get('token_budget')} tok "
            f"(full={last_compile.get('full_repo_tokens')} tok, "
            f"savings={last_compile.get('savings_percent')}%)"
        )
    else:
        console.out("Last capsule: none compiled yet")

    if report.is_healthy:
        console.out(console.green("Doctor: HEALTHY (0 errors, 0 warnings)"))
    else:
        console.out(
            console.yellow(
                f"Doctor: {len(report.errors)} error(s), {len(report.warnings)} warning(s)"
            )
        )
        for issue in report.issues[:5]:
            console.out(f"  - [{issue.kind.value}] {issue.entity_id}: {issue.message}")
    return 0


def _resolve_bench_dir(store: ContextStore, raw_dir: str | None) -> Path:
    if not raw_dir:
        return store.ctx_dir / "bench"
    candidate = Path(raw_dir)
    if not candidate.is_absolute():
        candidate = store.root / candidate
    return candidate.resolve()


def _cmd_bench_init_sample(args: argparse.Namespace, console: _Console) -> int:
    target = Path(args.dir).resolve() if args.dir else None
    sample_root = create_sample_project(target)
    console.out(console.green(f"Created calibration benchmark sandbox at: {sample_root}"))
    console.out("Next steps:")
    console.out(f"  1. ctx --root {sample_root} bench prepare --tasks TASK-001,TASK-002")
    console.out(
        f"  2. Paste generated prompts from {sample_root / '.ctx' / 'bench' / 'prompts'} "
        "into Google AI Studio -> save responses into responses/ -> run:"
    )
    console.out(
        f'     ctx --root {sample_root} bench score --test-cmd "python -m unittest discover -s tests"'
    )
    console.out(
        f"     (Or automated mode: ctx --root {sample_root} bench run --provider ollama)"
    )
    return 0


def _cmd_bench_prepare(
    args: argparse.Namespace, store: ContextStore, console: _Console
) -> int:
    suite = BenchmarkSuite(store, RepoIndexer(store.root, store=store))
    out_dir = _resolve_bench_dir(store, args.out_dir)
    task_ids = _split_csv_list(args.tasks) or None
    rubric = suite.prepare(
        task_ids=task_ids,
        bench_dir=out_dir,
        token_budget=args.budget,
        test_command=args.test_cmd,
    )
    console.out(BenchmarkSuite.format_prepare_table(rubric))
    console.out(f"Saved prompts -> {out_dir / 'prompts'}")
    console.out(f"Saved rubric  -> {out_dir / 'rubric.json'}")
    console.out(
        f"Paste AI Studio responses into {out_dir / 'responses'}/<TASK>__<strategy>.md "
        f"then run: ctx bench score --dir {out_dir}"
    )
    return 0


def _cmd_bench_score(
    args: argparse.Namespace, store: ContextStore, console: _Console
) -> int:
    suite = BenchmarkSuite(store, RepoIndexer(store.root, store=store))
    bench_dir = _resolve_bench_dir(store, args.dir)
    report = suite.score(bench_dir=bench_dir, test_command=args.test_cmd)
    console.out(report.format_console_summary())
    return 0


def _cmd_bench_run(
    args: argparse.Namespace, store: ContextStore, console: _Console
) -> int:
    adapter = _build_llm_adapter(args.provider, args.model)
    suite = BenchmarkSuite(store, RepoIndexer(store.root, store=store))
    out_dir = _resolve_bench_dir(store, args.out_dir)
    task_ids = _split_csv_list(args.tasks) or None
    console.out(
        console.dim(
            f"Running live A/B benchmark via provider={args.provider} "
            f"(model={args.model or 'default'})..."
        )
    )
    report = suite.run(
        adapter,
        task_ids=task_ids,
        bench_dir=out_dir,
        token_budget=args.budget,
        test_command=args.test_cmd,
    )
    console.out(report.format_console_summary())
    return 0


# ----------------------------------------------------------------------
# Argument Parser & Entry Point
# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the `ctx` CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="ctx",
        description="ctxforge — local context compiler and Truth Governance CLI for LLMs",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Project root directory (defaults to current working directory)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = subparsers.add_parser("init", help="Initialize .ctx/ and index repository")
    p_init.add_argument("--name", default=None, help="Project name")
    p_init.add_argument("--vision", default=None, help="Short project vision")
    p_init.add_argument("--force", action="store_true", help="Re-initialize existing .ctx/")

    # bootstrap
    p_boot = subparsers.add_parser(
        "bootstrap", help="Generate Day-0 <ctx_bootstrap> prompt for AI Studio"
    )
    p_boot.add_argument("brief", help="Path to brief file or '-' to read from stdin")
    p_boot.add_argument("--budget", type=int, default=12000, help="Prompt token budget")
    p_boot.add_argument("--no-clip", action="store_true", help="Do not copy to clipboard")
    p_boot.add_argument("--out", default=None, help="Custom output file path")

    # task
    p_task = subparsers.add_parser("task", help="Manage task DAG nodes")
    task_sub = p_task.add_subparsers(dest="task_command", required=True)

    p_task_add = task_sub.add_parser("add", help="Add a new task")
    p_task_add.add_argument("--title", required=True, help="Task title")
    p_task_add.add_argument("--desc", default="", help="Task description")
    p_task_add.add_argument(
        "--anchors", action="append", default=None, help="Comma-separated file[:symbol]"
    )
    p_task_add.add_argument(
        "--deps", action="append", default=None, help="Comma-separated dependency task IDs"
    )
    p_task_add.add_argument(
        "--ac", action="append", default=None, help="Acceptance criterion (repeatable)"
    )
    p_task_add.add_argument(
        "--priority", type=int, choices=[1, 2, 3], default=2, help="Priority 1=high..3=low"
    )

    p_task_list = task_sub.add_parser("list", help="List tasks")
    p_task_list.add_argument(
        "--status",
        choices=["todo", "in_progress", "blocked", "done", "cancelled", "all"],
        default="all",
        help="Filter by task status",
    )

    p_task_done = task_sub.add_parser("done", help="Mark a task as DONE")
    p_task_done.add_argument("task_id", help="Task ID (e.g. TASK-001)")

    # card
    p_card = subparsers.add_parser("card", help="Manage KnowledgeCards")
    card_sub = p_card.add_subparsers(dest="card_command", required=True)

    p_card_add = card_sub.add_parser("add", help="Add a KnowledgeCard")
    p_card_add.add_argument("--kind", required=True, help="INV | ADR | PITFALL")
    p_card_add.add_argument("--title", required=True, help="Card title")
    p_card_add.add_argument("--statement", required=True, help="Concise rule/decision/prohibition")
    p_card_add.add_argument("--rationale", required=True, help="Why or why rejected")
    p_card_add.add_argument(
        "--anchors", action="append", default=None, help="Comma-separated file[:symbol]"
    )
    p_card_add.add_argument("--tags", action="append", default=None, help="Comma-separated tags")
    p_card_add.add_argument("--immutable", action="store_true", help="Protect against overwrite")
    p_card_add.add_argument(
        "--always", action="store_true", help="Include in every compiled capsule"
    )

    p_card_list = card_sub.add_parser("list", help="List KnowledgeCards")
    p_card_list.add_argument(
        "--all", action="store_true", help="Include SUPERSEDED/DEPRECATED/ARCHIVED"
    )

    # compile
    p_comp = subparsers.add_parser("compile", help="Compile a token-budgeted context capsule")
    p_comp.add_argument("task_id", nargs="?", default=None, help="Optional Task ID")
    p_comp.add_argument("--query", default=None, help="Free-text task/query description")
    p_comp.add_argument(
        "--files", action="append", default=None, help="Comma-separated explicit target files"
    )
    p_comp.add_argument("--budget", type=int, default=8000, help="Hard token budget ceiling")
    p_comp.add_argument("--no-protocol", action="store_true", help="Omit <ctx_protocol> tail")
    p_comp.add_argument("--no-clip", action="store_true", help="Do not copy capsule to clipboard")
    p_comp.add_argument("--out", default=None, help="Custom output file path")

    # handoff
    p_hand = subparsers.add_parser(
        "handoff", help="Generate ultra-short prompt to extract <ctx_delta> before closing chat"
    )
    p_hand.add_argument("task_id", nargs="?", default=None, help="Optional Task ID")
    p_hand.add_argument("--no-clip", action="store_true", help="Do not copy prompt to clipboard")

    # ingest
    p_ing = subparsers.add_parser("ingest", help="Ingest <ctx_delta> or <ctx_bootstrap> from LLM")
    p_ing.add_argument("--file", default=None, help="Read response from file (or '-' for stdin)")
    p_ing.add_argument(
        "--bootstrap", action="store_true", help="Apply <ctx_bootstrap> project setup"
    )
    p_ing.add_argument(
        "--auto", action="store_true", help="Skip interactive [y/N] confirmation"
    )
    p_ing.add_argument(
        "--force-immutable",
        action="store_true",
        help="Allow superseding immutable cards",
    )
    p_ing.add_argument(
        "--extract-via",
        choices=["ollama", "gemini"],
        default=None,
        help="Optional backend to convert raw chat into <ctx_delta> before ingest",
    )
    p_ing.add_argument("--model", default=None, help="Model name for --extract-via")

    # doctor
    p_doc = subparsers.add_parser("doctor", help="Check .ctx/ for broken anchors, drift, and bloat")
    p_doc.add_argument(
        "--refresh-hashes",
        action="store_true",
        help="Update stored anchor hashes after manual review",
    )

    # status
    subparsers.add_parser("status", help="Show concise project memory and health dashboard")

    # bench
    p_bench = subparsers.add_parser(
        "bench", help="Run offline or live A/B benchmarks comparing context strategies"
    )
    bench_sub = p_bench.add_subparsers(dest="bench_command", required=True)

    p_b_sample = bench_sub.add_parser(
        "init-sample", help="Create an isolated calibration sandbox project in a temp/custom dir"
    )
    p_b_sample.add_argument(
        "--dir", default=None, help="Target directory for calibration sandbox (defaults to temp dir)"
    )

    p_b_prep = bench_sub.add_parser(
        "prepare", help="Prepare baseline & ctxforge prompts + rubric.json for manual AI Studio A/B"
    )
    p_b_prep.add_argument(
        "--tasks", action="append", default=None, help="Comma-separated Task IDs (e.g. TASK-001,TASK-002)"
    )
    p_b_prep.add_argument("--budget", type=int, default=8000, help="ctxforge token budget")
    p_b_prep.add_argument("--out-dir", default=None, help="Output directory (defaults to .ctx/bench)")
    p_b_prep.add_argument("--test-cmd", default=None, help="Optional verification test command")

    p_b_score = bench_sub.add_parser(
        "score", help="Score saved responses and generate benchmark_report.md"
    )
    p_b_score.add_argument("--dir", default=None, help="Benchmark directory (defaults to .ctx/bench)")
    p_b_score.add_argument("--test-cmd", default=None, help="Optional verification test command")

    p_b_run = bench_sub.add_parser(
        "run", help="Run all 3 strategies automatically via Gemini Free or local Ollama"
    )
    p_b_run.add_argument("--provider", required=True, choices=["gemini", "ollama"], help="LLM provider")
    p_b_run.add_argument("--model", default=None, help="Model override")
    p_b_run.add_argument(
        "--tasks", action="append", default=None, help="Comma-separated Task IDs"
    )
    p_b_run.add_argument("--budget", type=int, default=8000, help="ctxforge token budget")
    p_b_run.add_argument("--out-dir", default=None, help="Output directory (defaults to .ctx/bench)")
    p_b_run.add_argument("--test-cmd", default=None, help="Optional verification test command")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the `ctx` CLI."""
    console = _Console()
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    project_root = Path(args.root).resolve() if args.root else Path.cwd().resolve()
    store = ContextStore(project_root)

    try:
        if args.command == "init":
            return _cmd_init(args, store, console)
        if args.command == "bootstrap":
            return _cmd_bootstrap(args, store, console)
        if args.command == "handoff":
            return _cmd_handoff(args, store, console)
        if args.command == "ingest":
            return _cmd_ingest(args, store, console)
        if args.command == "bench":
            if args.bench_command == "init-sample":
                return _cmd_bench_init_sample(args, console)
            if args.bench_command == "prepare":
                return _cmd_bench_prepare(args, store, console)
            if args.bench_command == "score":
                return _cmd_bench_score(args, store, console)
            if args.bench_command == "run":
                return _cmd_bench_run(args, store, console)

        # Commands below require an initialized .ctx/ store.
        if args.command == "task":
            if args.task_command == "add":
                return _cmd_task_add(args, store, console)
            if args.task_command == "list":
                return _cmd_task_list(args, store, console)
            if args.task_command == "done":
                return _cmd_task_done(args, store, console)
        if args.command == "card":
            if args.card_command == "add":
                return _cmd_card_add(args, store, console)
            if args.card_command == "list":
                return _cmd_card_list(args, store, console)
        if args.command == "compile":
            return _cmd_compile(args, store, console)
        if args.command == "doctor":
            return _cmd_doctor(args, store, console)
        if args.command == "status":
            return _cmd_status(args, store, console)

        parser.print_help()
        return 1
    except (CtxForgeError, ValueError) as exc:
        console.err(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())