# Copyright 2026. Structured event logging and session registry.

import enum
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from forge.core.session import SESSIONS_BASE

SCHEMA_VERSION = 1
REGISTRY_MAX_BYTES = 512 * 1024  # 512KB


class EventType(str, enum.Enum):
    PIPELINE_STARTED = "pipeline_started"
    PIPELINE_COMPLETED = "pipeline_completed"
    PIPELINE_FAILED = "pipeline_failed"
    PIPELINE_KILLED = "pipeline_killed"
    STEP_STARTED = "step_started"
    STEP_PASSED = "step_passed"
    STEP_FAILED = "step_failed"
    STEP_RESET = "step_reset"
    STEP_SKIPPED = "step_skipped"
    PHASE_STARTED = "phase_started"
    PHASE_COMPLETED = "phase_completed"
    PHASE_FAILED = "phase_failed"
    PLANNER_STARTED = "planner_started"
    PLANNER_COMPLETED = "planner_completed"
    PLANNER_FAILED = "planner_failed"
    PLANNER_KILLED = "planner_killed"
    JUDGE_VERDICT = "judge_verdict"


_LIFECYCLE_EVENTS = {
    EventType.PIPELINE_STARTED, EventType.PIPELINE_COMPLETED,
    EventType.PIPELINE_FAILED, EventType.PIPELINE_KILLED,
    EventType.PLANNER_STARTED, EventType.PLANNER_COMPLETED,
    EventType.PLANNER_FAILED, EventType.PLANNER_KILLED,
}


@dataclass
class Event:
    v: int
    event: str
    ts: str
    session: str
    subsystem: str
    pipeline: str = ""
    preset: str = ""
    step: str = ""
    phase: str = ""
    error: str = ""
    retries: int = 0
    passed: bool | None = None
    pass_count: int = 0
    total: int = 0
    checkpoint: str = ""


_ALWAYS_KEEP = {"v", "event", "ts", "session", "subsystem"}
_SKIP_VALUES = {"", 0, None, False}


def _event_to_json(event: Event) -> str:
    d = {}
    for k, val in asdict(event).items():
        if k in _ALWAYS_KEEP or val not in _SKIP_VALUES:
            d[k] = val
    return json.dumps(d, separators=(",", ":"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_event(session_dir: str, event: Event) -> None:
    if not session_dir:
        return
    path = os.path.join(session_dir, "events.jsonl")
    try:
        with open(path, "a") as f:
            f.write(_event_to_json(event) + "\n")
    except OSError:
        pass


def _registry_path() -> Path:
    return SESSIONS_BASE / "registry.jsonl"


def _maybe_rotate_registry() -> None:
    path = _registry_path()
    try:
        if path.is_file() and path.stat().st_size >= REGISTRY_MAX_BYTES:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            rotated = path.with_name(f"registry.{ts}.jsonl")
            os.rename(str(path), str(rotated))
    except OSError:
        pass


def log_registry(event: Event) -> None:
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _maybe_rotate_registry()
    line = (_event_to_json(event) + "\n").encode("utf-8")
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
    except OSError:
        pass


def emit(session_dir: str, event_type: EventType, subsystem: str, **kwargs) -> None:
    ev = Event(
        v=SCHEMA_VERSION,
        event=event_type.value,
        ts=_now_iso(),
        session=session_dir,
        subsystem=subsystem,
        **kwargs,
    )
    log_event(session_dir, ev)
    if event_type in _LIFECYCLE_EVENTS:
        log_registry(ev)


# -- Status command ------------------------------------------------------------

_STATE_FILES = {"executor": "agent-state.json", "planner": ".planner-state.json"}


def _summarize_state(subsystem: str, name: str, data: dict) -> dict:
    summary = {"name": name, "subsystem": subsystem}
    if subsystem == "executor":
        summary["pipeline"] = data.get("pipeline", "")
        summary["preset"] = data.get("preset", "")
        summary["session_dir"] = data.get("session_dir", "")
        killed = data.get("killed", False)
        steps = data.get("steps", {})
        if killed:
            summary["status"] = "killed"
        elif any(s.get("status") == "failed" for s in steps.values()):
            summary["status"] = "failed"
        elif all(s.get("status") == "complete" for s in steps.values()):
            summary["status"] = "completed"
        elif any(s.get("status") == "in_progress" for s in steps.values()):
            summary["status"] = "running"
        else:
            summary["status"] = "pending"
        summary["steps"] = {
            n: s.get("status", "pending") for n, s in steps.items()
        }
    else:
        summary["preset"] = data.get("preset", "")
        summary["session_dir"] = data.get("session_dir", "")
        killed = data.get("killed", False)
        phases = data.get("phases", {})
        if killed:
            summary["status"] = "killed"
        elif all(p.get("status") in ("complete", "skipped") for p in phases.values()):
            summary["status"] = "completed"
        elif any(p.get("status") == "failed" for p in phases.values()):
            summary["status"] = "failed"
        elif any(p.get("status") == "in_progress" for p in phases.values()):
            summary["status"] = "running"
        else:
            summary["status"] = "pending"
    summary["created_at"] = data.get("created_at", "")
    summary["updated_at"] = data.get("updated_at", "")
    return summary


def cmd_status(args) -> int:
    limit = getattr(args, "limit", 20)
    active_only = getattr(args, "active", False)

    sessions: list[dict] = []
    for subsystem, state_file in _STATE_FILES.items():
        base = SESSIONS_BASE / subsystem
        if not base.is_dir():
            continue
        for entry in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime,
                            reverse=True):
            sf = entry / state_file
            if not sf.is_file():
                continue
            try:
                data = json.loads(sf.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            sessions.append(_summarize_state(subsystem, entry.name, data))

    sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)

    if active_only:
        sessions = [s for s in sessions if s["status"] == "running"]

    print(json.dumps(sessions[:limit], indent=2))
    return 0
