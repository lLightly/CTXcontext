"""Calibrated zero-dependency token estimator and token-budget truncator."""

from __future__ import annotations

import importlib
import math
import re

# Master scanner partitioning arbitrary Unicode text into non-overlapping chunks.
_TOKEN_CHUNK_RE = re.compile(
    r"(?P<cyrillic>[\u0400-\u04FF\u0500-\u052F]+)"
    r"|(?P<ident>[A-Za-z][A-Za-z0-9_]*)"
    r"|(?P<digits>[0-9]+)"
    r"|(?P<newline>\r?\n)"
    r"|(?P<space>[ \t\f\v]+)"
    r"|(?P<punct>[^\w\s]+)"
    r"|(?P<other>.)",
    re.DOTALL,
)

# Splits CamelCase / acronym transitions and digit runs inside identifiers.
_SUBWORD_RE = re.compile(
    r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|[0-9]+|[A-Z]+"
)


def _score_latin_identifier(token: str) -> float:
    """Estimate subword token cost for English words and code identifiers."""
    if "_" not in token:
        segments = [token]
        underscore_cost = 0.0
    else:
        parts = token.split("_")
        underscores = len(parts) - 1
        underscore_cost = underscores * 0.45
        segments = [p for p in parts if p]

    total = underscore_cost
    for segment in segments:
        subwords = _SUBWORD_RE.findall(segment)
        if not subwords:
            total += max(1.0, len(segment) / 3.5)
            continue
        for sub in subwords:
            length = len(sub)
            if sub.isdigit():
                total += 1.0 + max(0, length - 2) / 2.2
            elif length >= 3 and sub.isupper():
                # Uppercase constants (e.g. SUPERSEDED, INVARIANT) split more aggressively in BPE.
                total += 1.0 + max(0, length - 3) / 2.6
            elif length <= 4:
                total += 1.0
            else:
                total += 1.0 + (length - 4) / 3.4
    return total


def _score_cyrillic_word(word: str) -> float:
    """Estimate BPE/SentencePiece cost for Cyrillic text (~2.2-2.5 chars/token)."""
    length = len(word)
    if length == 1:
        return 0.95
    return length / 2.2


def _estimate_tokens_heuristic(text: str) -> int:
    """Deterministic character/word-category heuristic token estimator."""
    if not text:
        return 0

    estimate = 0.0
    for match in _TOKEN_CHUNK_RE.finditer(text):
        kind = match.lastgroup
        chunk = match.group()

        if kind == "cyrillic":
            estimate += _score_cyrillic_word(chunk)
        elif kind == "ident":
            estimate += _score_latin_identifier(chunk)
        elif kind == "digits":
            estimate += 1.0 + max(0, len(chunk) - 2) / 2.2
        elif kind == "newline":
            estimate += 0.85
        elif kind == "space":
            if chunk == " ":
                estimate += 0.12
            else:
                spaces = chunk.count(" ")
                tabs = chunk.count("\t")
                others = len(chunk) - spaces - tabs
                estimate += spaces / 3.8 + tabs * 0.85 + others * 0.25
        elif kind == "punct":
            estimate += 0.55 + (len(chunk) - 1) * 0.35
        else:
            if chunk == "_":
                estimate += 0.5
            else:
                # CJK, emojis, and standalone symbols typically map to 1-2 tokens.
                estimate += 1.2

    return max(1, math.ceil(estimate))


def estimate_tokens(
    text: str,
    *,
    use_tiktoken: bool = False,
    encoding_name: str = "cl100k_base",
) -> int:
    """Estimate token count for `text`.

    Uses a deterministic category-based heuristic by default (and always in a
    clean stdlib-only environment). Optionally delegates to ``tiktoken`` if
    installed and ``use_tiktoken=True`` is explicitly requested.
    """
    if not text:
        return 0

    if use_tiktoken:
        try:
            tiktoken = importlib.import_module("tiktoken")
            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(text, disallowed_special=()))
        except Exception:
            pass

    return _estimate_tokens_heuristic(text)


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    marker: str = "\n...[TRUNCATED]...",
) -> str:
    """Truncate `text` so that its estimated token count does not exceed `max_tokens`.

    When truncation occurs and `marker` fits within `max_tokens`, appends
    `marker` to the kept prefix while attempting to snap cleanly to a nearby
    whitespace/line boundary. If `max_tokens <= 0` or `text` is empty, returns
    an empty string. If `marker` itself consumes >= `max_tokens`, truncates
    `text` directly without `marker` to strictly satisfy the budget.
    """
    if max_tokens <= 0 or not text:
        return ""

    if estimate_tokens(text) <= max_tokens:
        return text

    marker_tokens = estimate_tokens(marker) if marker else 0

    if not marker or marker_tokens >= max_tokens:
        lo, hi, best = 0, len(text), 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if estimate_tokens(text[:mid]) <= max_tokens:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return text[:best]

    lo, hi, best = 0, len(text), 0
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = f"{text[:mid]}{marker}"
        if estimate_tokens(candidate) <= max_tokens:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    prefix = text[:best]
    if 0 < best < len(text) and not text[best].isspace():
        # Prefer snapping back to a nearby newline or space instead of slicing a word.
        min_keep = int(best * 0.85)
        newline_pos = prefix.rfind("\n", min_keep)
        space_pos = prefix.rfind(" ", min_keep)
        snap_pos = max(newline_pos, space_pos)
        if snap_pos > min_keep:
            snapped = prefix[:snap_pos]
            if estimate_tokens(f"{snapped}{marker}") <= max_tokens:
                prefix = snapped

    if marker.startswith("\n"):
        stripped = prefix.rstrip(" \t\r\n")
        if estimate_tokens(f"{stripped}{marker}") <= max_tokens:
            prefix = stripped

    return f"{prefix}{marker}"