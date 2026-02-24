"""Shared utilities for the pipeline engine."""

import os
import re
import subprocess
from collections import deque
from pathlib import Path

from forge.core.session import SESSIONS_BASE as _CORE_BASE

DEFAULT_DEV_PORT = 8080

_VALID_PACKAGE_RE = re.compile(r"^[a-zA-Z0-9_./@-]+$")


def repo_root() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "."


SESSIONS_BASE = _CORE_BASE / "executor"


def session_prefix() -> str:
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    match = re.search(os.environ.get("FORGE_TICKET_PATTERN", r"[A-Z]+-\d+"), branch)
    return match.group(0) if match else re.sub(r"[/:]", "-", branch)


def find_active_session() -> Path | None:
    if not SESSIONS_BASE.is_dir():
        return None
    prefix = session_prefix()
    candidates = [
        d for d in SESSIONS_BASE.iterdir()
        if d.is_dir() and d.name.startswith(prefix + "_") and (d / "agent-state.json").is_file()
    ]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def validate_package_name(pkg: str) -> None:
    if not pkg or not _VALID_PACKAGE_RE.match(pkg):
        raise ValueError(f"Invalid package name: {pkg!r} — only [a-zA-Z0-9_./@-] allowed")


def is_bazel_repo(root: str | None = None) -> bool:
    """Return True if the repo root contains a WORKSPACE or WORKSPACE.bazel file."""
    root = root or repo_root()
    return os.path.isfile(os.path.join(root, "WORKSPACE")) or os.path.isfile(os.path.join(root, "WORKSPACE.bazel"))


def transitive_dependents(step_name: str, graph: dict[str, list[str]]) -> list[str]:
    reverse: dict[str, list[str]] = {}
    for s, deps in graph.items():
        for d in deps:
            reverse.setdefault(d, []).append(s)
    visited: set[str] = set()
    queue = deque(reverse.get(step_name, []))
    while queue:
        dep = queue.popleft()
        if dep in visited:
            continue
        visited.add(dep)
        queue.extend(reverse.get(dep, []))
    return sorted(visited)
