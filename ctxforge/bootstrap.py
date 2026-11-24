"""Day-0 Bootstrap prompt generator for single-pass project onboarding."""

from __future__ import annotations

from pathlib import Path

from .indexer import FileRecord, RepoIndex, RepoIndexer
from .storage import ContextStore
from .tokens import estimate_tokens, truncate_to_tokens

_BOOTSTRAP_HEADER = """You are initializing a persistent Truth-Governed project memory (`ctxforge`) for a software project.
Analyze the project brief below (and the repository code skeletons, if present) and extract a crisp, zero-fluff project foundation in ONE pass.

CRITICAL RULES:
1. `project.tech_stack_notes`: Record ONLY non-obvious architectural constraints, runtime quirks, version traps, or hard platform boundaries. Do NOT summarize `package.json`, `pyproject.toml`, or list obvious standard libraries.
2. `cards`:
   - `INV` (Invariant): Unbreakable rules or constraints. Set `immutable: true` and `always_include: true` ONLY for truly global invariants that apply to almost every task (keep `always_include` small, ideally <= 5 cards).
   - `ADR` (Architectural Decision): Concrete technical decisions with clear `rationale` ("why").
   - `PITFALL` (Pitfall / Rejected Approach): Dead-ends, forbidden patterns, or traps to avoid and why.
   - `anchors`: Attach real existing relative POSIX paths from the codebase if applicable (e.g., `["app/auth.py"]`), or `[]` for greenfield projects. NEVER invent file paths.
3. `tasks`:
   - Decompose the work into a clean dependency DAG (`id`: `"T1"`, `"T2"`, ...; `depends_on`: `["T1"]`).
   - Give each task concrete `acceptance_criteria` and a short `next_step_hint`.
4. Output ONLY valid JSON wrapped strictly inside `<ctx_bootstrap>...</ctx_bootstrap>` (no trailing commas).

REQUIRED OUTPUT SCHEMA:
<ctx_bootstrap>
{
  "project": {
    "name": "short-project-name",
    "vision": "1-2 sentence core goal",
    "non_goals": ["Explicit out-of-scope item"],
    "tech_stack_notes": "Only non-obvious constraints and nuances"
  },
  "cards": [
    {
      "kind": "INV | ADR | PITFALL",
      "title": "Short descriptive title",
      "statement": "Concise rule, decision, or prohibition",
      "rationale": "Why this rule/decision exists or why approach was rejected",
      "immutable": false,
      "always_include": false,
      "tags": ["domain"],
      "anchors": ["relative/path.py"]
    }
  ],
  "tasks": [
    {
      "id": "T1",
      "title": "Task title",
      "description": "Concrete implementation scope",
      "priority": 1,
      "depends_on": [],
      "anchors": ["relative/path.py"],
      "acceptance_criteria": ["Testable criterion"],
      "next_step_hint": "First concrete action"
    }
  ]
}
</ctx_bootstrap>"""


def _rank_files_for_bootstrap(index: RepoIndex) -> list[FileRecord]:
    """Order repository files so central/symbol-rich code skeletons come first."""
    scored: list[tuple[int, int, str, FileRecord]] = []
    for rel_path, record in index.files.items():
        degree = len(index.import_graph.get(rel_path, [])) + len(
            index.reverse_graph.get(rel_path, [])
        )
        is_code = 1 if record.language in (
            "python",
            "typescript",
            "javascript",
            "go",
            "rust",
            "java",
        ) else 0
        scored.append((is_code, degree + len(record.symbols), rel_path, record))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [item[3] for item in scored]


class BootstrapGenerator:
    """Generates a token-budgeted Day-0 bootstrap prompt from brief + codebase AST."""

    def __init__(
        self,
        store: ContextStore | Path | str,
        indexer: RepoIndexer | None = None,
    ) -> None:
        self._store: ContextStore = (
            store if isinstance(store, ContextStore) else ContextStore(store)
        )
        self._indexer: RepoIndexer = (
            indexer
            if indexer is not None
            else RepoIndexer(self._store.root, store=self._store)
        )

    @property
    def store(self) -> ContextStore:
        return self._store

    @property
    def indexer(self) -> RepoIndexer:
        return self._indexer

    def generate_bootstrap_prompt(
        self,
        brief_text: str,
        token_budget: int = 12000,
    ) -> str:
        """Build a Day-0 prompt combining `brief_text` and compact repo AST skeletons."""
        if not isinstance(brief_text, str) or not brief_text.strip():
            raise ValueError("brief_text must be a non-empty string.")
        if isinstance(token_budget, bool) or not isinstance(token_budget, int) or token_budget <= 0:
            raise ValueError("token_budget must be a positive integer.")

        index = self._indexer.build_or_update()

        brief_section = f"<project_brief>\n{brief_text.strip()}\n</project_brief>"
        base_tokens = estimate_tokens(_BOOTSTRAP_HEADER) + estimate_tokens(brief_section) + 20

        if base_tokens > token_budget:
            avail_brief = max(80, token_budget - estimate_tokens(_BOOTSTRAP_HEADER) - 32)
            brief_section = (
                "<project_brief>\n"
                + truncate_to_tokens(brief_text.strip(), avail_brief)
                + "\n</project_brief>"
            )
            return f"{_BOOTSTRAP_HEADER}\n\n{brief_section}"

        remaining = token_budget - base_tokens
        sections: list[str] = [_BOOTSTRAP_HEADER, brief_section]

        if index.files and remaining > 80:
            ranked_records = _rank_files_for_bootstrap(index)
            skeleton_blocks: list[str] = []
            omitted_paths: list[str] = []
            used = estimate_tokens("<existing_code_skeletons>\n</existing_code_skeletons>")

            for record in ranked_records:
                block = f"### {record.rel_path} ({record.language})\n{record.skeleton}"
                cost = estimate_tokens(block) + 2
                if used + cost <= remaining:
                    skeleton_blocks.append(block)
                    used += cost
                else:
                    omitted_paths.append(record.rel_path)

            if skeleton_blocks:
                code_section = "\n".join(
                    [
                        "<existing_code_skeletons>",
                        *skeleton_blocks,
                        "</existing_code_skeletons>",
                    ]
                )
                sections.append(code_section)
                remaining = max(0, remaining - estimate_tokens(code_section))

            if omitted_paths and remaining >= 30:
                paths_list = "\n".join(f"- {p}" for p in sorted(omitted_paths))
                remaining_block = (
                    "<additional_repo_files>\n"
                    f"{paths_list}\n"
                    "</additional_repo_files>"
                )
                if estimate_tokens(remaining_block) <= remaining:
                    sections.append(remaining_block)
                else:
                    sections.append(
                        truncate_to_tokens(
                            remaining_block,
                            remaining,
                            marker="\n...[MORE FILES]...\n</additional_repo_files>",
                        )
                    )

        prompt = "\n\n".join(sections)
        if estimate_tokens(prompt) > token_budget:
            prompt = truncate_to_tokens(prompt, token_budget)
        return prompt