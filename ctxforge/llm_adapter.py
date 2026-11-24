"""Strictly optional zero-dependency adapters for free Gemini Flash and local Ollama."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
import urllib.error
import urllib.request
from typing import Any

from .storage import CtxForgeError
from .tokens import truncate_to_tokens

_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_OLLAMA_MODEL = "qwen3"
_MAX_RAW_CHAT_TOKENS = 20000


class LLMAdapterError(CtxForgeError):
    """Base exception for optional LLM adapter failures."""


class MissingApiKeyError(LLMAdapterError):
    """Raised when GEMINI_API_KEY is missing from environment variables."""


class RateLimitError(LLMAdapterError):
    """Raised on HTTP 429 / RESOURCE_EXHAUSTED free-tier quota responses."""


class LLMAdapter(ABC):
    """Abstract interface for optional LLM completion backends."""

    @abstractmethod
    def complete(self, prompt: str, system_prompt: str = "") -> str:
        """Generate a text completion for `prompt` and optional `system_prompt`."""


class GeminiFreeAdapter(LLMAdapter):
    """Stdlib REST client for Google AI Studio Gemini free-tier models."""

    def __init__(
        self,
        model: str = _DEFAULT_GEMINI_MODEL,
        *,
        temperature: float = 0.1,
        timeout: float = 60.0,
    ) -> None:
        self.model = model.strip() or _DEFAULT_GEMINI_MODEL
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, prompt: str, system_prompt: str = "") -> str:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise MissingApiKeyError(
                "GEMINI_API_KEY environment variable is not set. "
                "Export your free Google AI Studio key or use copy-paste / Ollama."
            )

        url = f"{_GEMINI_API_BASE}/{self.model}:generateContent"
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature},
        }
        if system_prompt.strip():
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw_response = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as err:
            err_body = ""
            try:
                err_body = err.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            finally:
                err.close()
            if err.code == 429 or "RESOURCE_EXHAUSTED" in err_body:
                raise RateLimitError(
                    f"Gemini API free-tier quota exhausted (HTTP 429 / RESOURCE_EXHAUSTED) "
                    f"for model '{self.model}'. Free tier limits RPM/TPM/RPD: wait ~60s, "
                    f"reduce input size, switch to a local Ollama model, or paste manually "
                    f"into Google AI Studio. Details: {err_body[:240]}"
                ) from err
            raise LLMAdapterError(
                f"Gemini API request failed (HTTP {err.code}): {err_body[:300]}"
            ) from err
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            raise LLMAdapterError(
                f"Failed to reach Gemini API endpoint: {err}"
            ) from err

        try:
            data = json.loads(raw_response)
            candidates = data.get("candidates", [])
            if not candidates:
                raise LLMAdapterError(
                    f"Gemini API returned no candidates: {raw_response[:300]}"
                )
            parts = candidates[0].get("content", {}).get("parts", [])
            texts = [str(part.get("text", "")) for part in parts if "text" in part]
            joined = "".join(texts).strip()
            if not joined:
                raise LLMAdapterError("Gemini API returned empty text content.")
            return joined
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as err:
            raise LLMAdapterError(
                f"Unexpected Gemini API response format: {err}"
            ) from err


class OllamaAdapter(LLMAdapter):
    """Stdlib REST client for local Ollama (http://localhost:11434/api/generate)."""

    def __init__(
        self,
        model: str = _DEFAULT_OLLAMA_MODEL,
        *,
        base_url: str = _DEFAULT_OLLAMA_URL,
        timeout: float = 120.0,
    ) -> None:
        self.model = model.strip() or _DEFAULT_OLLAMA_MODEL
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete(self, prompt: str, system_prompt: str = "") -> str:
        url = f"{self.base_url}/api/generate"
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt.strip():
            payload["system"] = system_prompt

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw_response = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as err:
            err_body = ""
            try:
                err_body = err.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            finally:
                err.close()
            raise LLMAdapterError(
                f"Ollama API request failed (HTTP {err.code}) for model '{self.model}': "
                f"{err_body[:300]}"
            ) from err
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            raise LLMAdapterError(
                f"Could not connect to Ollama at {self.base_url}. "
                f"Ensure `ollama serve` is running and model '{self.model}' is pulled. "
                f"Details: {err}"
            ) from err

        try:
            data = json.loads(raw_response)
            text = str(data.get("response", "")).strip()
            if not text:
                raise LLMAdapterError(
                    f"Ollama returned an empty response for model '{self.model}'."
                )
            return text
        except (json.JSONDecodeError, TypeError) as err:
            raise LLMAdapterError(
                f"Unexpected Ollama response format: {err}"
            ) from err


_AUTO_EXTRACT_SYSTEM = (
    "You are a precise memory-delta extractor for ctxforge. "
    "Extract ONLY facts, decisions, task updates, and pitfalls explicitly stated "
    "in the transcript. Never invent file paths or decisions."
)

_AUTO_EXTRACT_TEMPLATE = """Convert the following raw LLM chat transcript into a single valid <ctx_delta>...</ctx_delta> JSON block.
Omit empty arrays to keep output compact. Do not add prose outside <ctx_delta>.

Required JSON schema inside <ctx_delta>:
{{
  "task_id": "TASK-001 or null",
  "task_status": "in_progress | done | blocked",
  "decisions_added": [{{"kind": "ADR|INV|PITFALL", "title": "...", "statement": "...", "rationale": "...", "tags": [], "anchors": ["path/file.py"]}}],
  "decisions_superseded": [{{"old_id": "ADR-...", "replacement": {{...}}}}],
  "new_subtasks": [{{"title": "...", "description": "...", "anchors": ["..."]}}],
  "next_anchor": "Where to resume the next session"
}}

Transcript:
{transcript}"""


def auto_extract_delta_from_raw_chat(
    raw_chat_text: str,
    adapter: LLMAdapter,
) -> str:
    """Compress a raw chat transcript into <ctx_delta> using an optional LLMAdapter."""
    if not isinstance(raw_chat_text, str) or not raw_chat_text.strip():
        raise LLMAdapterError("Cannot extract delta from an empty chat transcript.")
    bounded_chat = truncate_to_tokens(
        raw_chat_text.strip(),
        _MAX_RAW_CHAT_TOKENS,
        marker="\n...[EARLIER CHAT TRUNCATED]...",
    )
    prompt = _AUTO_EXTRACT_TEMPLATE.format(transcript=bounded_chat)
    return adapter.complete(prompt, system_prompt=_AUTO_EXTRACT_SYSTEM)