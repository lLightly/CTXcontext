"""Zero-dependency cross-platform clipboard utilities with safe headless fallback."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

_CLIPBOARD_TIMEOUT_SEC = 3.0


def _copy_commands() -> list[list[str]]:
    if sys.platform == "darwin":
        return [["pbcopy"]]
    if sys.platform == "win32" or os.name == "nt":
        return [
            ["clip.exe"],
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "[Console]::InputEncoding = [System.Text.Encoding]::UTF8; "
                "$input | Out-String | Set-Clipboard",
            ],
        ]

    commands: list[list[str]] = []
    if os.environ.get("WAYLAND_DISPLAY"):
        commands.append(["wl-copy"])
    commands.extend(
        [
            ["xclip", "-selection", "clipboard"],
            ["xsel", "--clipboard", "--input"],
            ["wl-copy"],
        ]
    )
    if os.environ.get("WSL_DISTRO_NAME"):
        commands.append(["clip.exe"])
    return commands


def _paste_commands() -> list[list[str]]:
    if sys.platform == "darwin":
        return [["pbpaste"]]
    if sys.platform == "win32" or os.name == "nt":
        return [
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
                "Get-Clipboard -Raw",
            ],
            ["powershell", "-NoProfile", "-Command", "Get-Clipboard"],
        ]

    commands: list[list[str]] = []
    if os.environ.get("WAYLAND_DISPLAY"):
        commands.append(["wl-paste", "--no-newline"])
    commands.extend(
        [
            ["xclip", "-selection", "clipboard", "-o"],
            ["xsel", "--clipboard", "--output"],
            ["wl-paste", "--no-newline"],
        ]
    )
    if os.environ.get("WSL_DISTRO_NAME"):
        commands.append(
            ["powershell.exe", "-NoProfile", "-Command", "Get-Clipboard"]
        )
    return commands


def copy_to_clipboard(text: str) -> bool:
    """Copy `text` to the system clipboard.

    Returns True on success, or False gracefully when no clipboard utility or
    graphical session is available (e.g., CI/Docker/headless SSH).
    """
    if not isinstance(text, str):
        return False

    seen: set[tuple[str, ...]] = set()
    for cmd in _copy_commands():
        key = tuple(cmd)
        if key in seen:
            continue
        seen.add(key)

        if shutil.which(cmd[0]) is None:
            continue

        try:
            proc = subprocess.run(
                cmd,
                input=text,
                text=True,
                encoding="utf-8",
                capture_output=True,
                timeout=_CLIPBOARD_TIMEOUT_SEC,
                check=False,
            )
            if proc.returncode == 0:
                return True
        except (OSError, subprocess.SubprocessError, ValueError):
            continue

    return False


def read_from_clipboard() -> str | None:
    """Read text from the system clipboard, or return None if unavailable."""
    seen: set[tuple[str, ...]] = set()
    for cmd in _paste_commands():
        key = tuple(cmd)
        if key in seen:
            continue
        seen.add(key)

        if shutil.which(cmd[0]) is None:
            continue

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=_CLIPBOARD_TIMEOUT_SEC,
                check=False,
            )
            if proc.returncode == 0:
                output = proc.stdout
                if sys.platform == "win32" or os.name == "nt":
                    output = output.replace("\r\n", "\n")
                return output
        except (OSError, subprocess.SubprocessError, ValueError):
            continue

    return None