
"""Pipeline operations library — shared state logic for CLI and driver.

Provides direct function calls for pipeline state management, eliminating the
need for subprocess spawning when operating within the same Python process.
"""

import json
import os
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .state import StateManager, PipelineState, StepState, StepStatus
from .registry import load_preset, get_model, Preset, PipelineDefinition
from .evidence import EvidenceChecker, EvidenceResult
from .checkpoint import write_checkpoint, verify_checkpoint, verify_all_checkpoints, clear_checkpoints
from .runner import execute_command, build_context, StepResult
from .utils import DEFAULT_DEV_PORT, repo_root, find_active_session

MAX_RETRIES = 3


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Path resolution ─────────────────────────────────────────────────────────

def _repo_root() -> str:
    return repo_root()


def _active_session_dir() -> Path | None:
    return find_active_session()


def state_file_path() -> Path:
    sd = _active_session_dir()
    if sd is None:
        raise RuntimeError("No active pipeline session found.")
    return sd / "agent-state.json"


def checkpoint_dir(session_dir: str | None = None) -> str:
    if session_dir is None:
        sd = _active_session_dir()
        if sd is None:
            raise RuntimeError("No active pipeline session found.")
        session_dir = str(sd)
    return os.path.join(session_dir, "checkpoints")


def presets_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "presets"


def state_mgr() -> StateManager:
    return StateManager(state_file_path())


def require_state() -> PipelineState:
    mgr = state_mgr()
    if not mgr.exists():
        raise RuntimeError("No active pipeline. Run: pipeline_cli.py init <full|lightweight>")
    return mgr.load()


def load_preset_for_state(state: PipelineState) -> Preset:
    return load_preset(presets_dir() / (state.preset or "hz-web"))


# ── Core operations ─────────────────────────────────────────────────────────

def is_permanently_failed(step_state: StepState) -> bool:
    return step_state.status == StepStatus.FAILED and step_state.retries >= MAX_RETRIES


def dep_is_satisfied(step_state: StepState) -> bool:
    return step_state.status == StepStatus.COMPLETE


def check_dependencies(state: PipelineState, target_name: str) -> list[str]:
    """Return list of incomplete dependency names. Empty = ready to run."""
    if state.dependency_graph:
        deps = state.dependency_graph.get(target_name, [])
        return [f"{d} ({state.steps[d].status.value})" for d in deps if not dep_is_satisfied(state.steps[d])]

    step_names = state.step_names_ordered()
    idx = step_names.index(target_name)
    return [f"{p} ({state.steps[p].status.value})" for p in step_names[:idx] if not dep_is_satisfied(state.steps[p])]


def get_next_steps(state: PipelineState, preset: Preset) -> dict:
    """Return runnable, in_progress, and blocked steps."""
    runnable = []
    in_progress = []
    blocked = []

    for name in state.step_names_ordered():
        ss = state.steps[name]
        step_def = preset.steps.get(name)
        if ss.status == StepStatus.COMPLETE:
            continue
        if is_permanently_failed(ss):
            continue
        if ss.status == StepStatus.IN_PROGRESS:
            in_progress.append({"step": name})
            continue

        if state.dependency_graph:
            deps = state.dependency_graph.get(name, [])
            deps_met = all(dep_is_satisfied(state.steps[d]) for d in deps)
            has_failed_deps = any(is_permanently_failed(state.steps[d]) for d in deps)
        else:
            step_names = state.step_names_ordered()
            idx = step_names.index(name)
            deps_met = all(dep_is_satisfied(state.steps[p]) for p in step_names[:idx])
            has_failed_deps = any(is_permanently_failed(state.steps[p]) for p in step_names[:idx])

        if deps_met:
            entry = {
                "step": name,
                "type": step_def.step_type if step_def else "unknown",
                "status": ss.status.value,
                "retries": ss.retries,
            }
            if has_failed_deps:
                entry["has_failed_deps"] = True
            runnable.append(entry)
        else:
            blocked.append(name)

    if not runnable and not in_progress:
        if blocked:
            return {"runnable": [], "in_progress": [], "blocked": blocked}
        return {"step": None, "message": "All steps complete"}

    return {"runnable": runnable, "in_progress": in_progress, "blocked": blocked}


def mark_running(step_name: str, skip_deps: bool = False) -> PipelineState:
    """Mark a step as IN_PROGRESS."""
    state = require_state()
    if step_name not in state.steps:
        raise ValueError(f"Unknown step: {step_name}")
    if not skip_deps:
        incomplete = check_dependencies(state, step_name)
        if incomplete:
            raise RuntimeError(f"Dependencies not complete: {', '.join(incomplete)}")

    # Archive old artifacts on retry
    retries = state.steps[step_name].retries
    if retries > 0 and state.session_dir:
        preset = load_preset_for_state(state)
        step_def = preset.steps.get(step_name)
        if step_def and step_def.evidence:
            archive_step_artifacts(step_def, state.session_dir, retries)

    def mutate(s: PipelineState):
        s.steps[step_name].status = StepStatus.IN_PROGRESS
        s.steps[step_name].started_at = _now_iso()
        s.steps[step_name].completed_at = ""
        s.current_step = step_name
    return state_mgr().update(mutate)


def mark_passed(step_name: str) -> tuple[PipelineState, str]:
    """Mark a step as COMPLETE. Returns (state, checkpoint_path)."""
    state = require_state()
    if step_name not in state.steps:
        raise ValueError(f"Unknown step: {step_name}")

    chk_dir = checkpoint_dir(state.session_dir)
    preset = load_preset_for_state(state)
    step_def = preset.steps.get(step_name)
    evidence_result = None
    if step_def and step_def.evidence:
        checker = EvidenceChecker(
            state.session_dir, chk_dir, state.steps, state.dependency_graph or None,
        )
        evidence_result = checker.check(step_def)
        if not evidence_result.passed:
            raise RuntimeError(f"Evidence failed: {evidence_result.message}")

    def mutate(s: PipelineState):
        s.steps[step_name].status = StepStatus.COMPLETE
        s.steps[step_name].completed_at = _now_iso()
    updated = state_mgr().update(mutate)

    cp_path = write_checkpoint(chk_dir, step_name, state.pipeline, evidence_result)

    # Auto-revalidation
    if state.steps[step_name].retries > 0:
        pipeline_def = preset.pipelines.get(state.pipeline)
        if pipeline_def and pipeline_def.revalidation_targets:
            targets = [t for t in pipeline_def.revalidation_targets
                       if t != step_name and t in state.steps and state.steps[t].status == StepStatus.COMPLETE]
            if targets:
                def reset_targets(s: PipelineState):
                    for t in targets:
                        s.steps[t].status = StepStatus.PENDING
                        s.steps[t].retries = 0
                        s.steps[t].last_error = ""
                state_mgr().update(reset_targets)
                for t in targets:
                    cp = os.path.join(chk_dir, f"{t}.passed")
                    if os.path.isfile(cp):
                        os.unlink(cp)

    return updated, cp_path


def mark_failed(step_name: str, error: str = "") -> dict:
    """Mark a step as FAILED. Returns result dict."""
    state = require_state()
    if step_name not in state.steps:
        raise ValueError(f"Unknown step: {step_name}")

    def mutate(s: PipelineState):
        s.steps[step_name].retries += 1
        s.steps[step_name].status = StepStatus.FAILED
        s.steps[step_name].last_error = error
        s.steps[step_name].completed_at = _now_iso()
    updated = state_mgr().update(mutate)
    retries = updated.steps[step_name].retries
    result = {"step": step_name, "status": "failed", "retries": retries, "max_retries": MAX_RETRIES}
    if retries >= MAX_RETRIES:
        result["pipeline_exhausted"] = True
    return result


def reset_step(step_name: str, no_retry_inc: bool = False) -> None:
    """Reset a step to PENDING."""
    state = require_state()
    if step_name not in state.steps:
        raise ValueError(f"Unknown step: {step_name}")

    def mutate(s: PipelineState):
        s.steps[step_name].status = StepStatus.PENDING
        if not no_retry_inc:
            s.steps[step_name].retries += 1
        s.steps[step_name].last_error = ""
        s.steps[step_name].started_at = ""
        s.steps[step_name].completed_at = ""
    state_mgr().update(mutate)

    cp_path = os.path.join(checkpoint_dir(state.session_dir), f"{step_name}.passed")
    if os.path.isfile(cp_path):
        os.unlink(cp_path)


def skip_step(step_name: str) -> None:
    """Skip a step — mark as COMPLETE without evidence checks."""
    state = require_state()
    if step_name not in state.steps:
        raise ValueError(f"Unknown step: {step_name}")

    def mutate(s: PipelineState):
        s.steps[step_name].status = StepStatus.COMPLETE
        s.steps[step_name].completed_at = _now_iso()
    state_mgr().update(mutate)

    # Write a checkpoint with manual_skip flag
    chk_dir = checkpoint_dir(state.session_dir)
    os.makedirs(chk_dir, exist_ok=True)
    cp_path = os.path.join(chk_dir, f"{step_name}.passed")
    ts = _now_iso()
    lines = [f"step={step_name}", f"passed_at={ts}", f"pipeline={state.pipeline}", "manual_skip=true"]
    Path(cp_path).write_text("\n".join(lines) + "\n")


def resume_from(step_name: str) -> list[str]:
    """Mark all steps before target as COMPLETE, reset target to PENDING.

    Returns list of steps marked complete.
    """
    state = require_state()
    if step_name not in state.steps:
        raise ValueError(f"Unknown step: {step_name}")

    step_order = state.step_names_ordered()
    idx = step_order.index(step_name)
    prior_steps = step_order[:idx]

    marked = []
    def mutate(s: PipelineState):
        for prior in prior_steps:
            if s.steps[prior].status != StepStatus.COMPLETE:
                s.steps[prior].status = StepStatus.COMPLETE
                s.steps[prior].completed_at = _now_iso()
                marked.append(prior)
        s.steps[step_name].status = StepStatus.PENDING
        s.steps[step_name].retries = 0
        s.steps[step_name].last_error = ""
        s.steps[step_name].started_at = ""
        s.steps[step_name].completed_at = ""
    state_mgr().update(mutate)

    # Write skip checkpoints for prior steps
    chk_dir = checkpoint_dir(state.session_dir)
    for prior in marked:
        skip_step_checkpoint(prior, state.pipeline, chk_dir)

    # Clear checkpoint for target step
    cp_path = os.path.join(chk_dir, f"{step_name}.passed")
    if os.path.isfile(cp_path):
        os.unlink(cp_path)

    return marked


def skip_step_checkpoint(step_name: str, pipeline: str, chk_dir: str) -> None:
    """Write a manual-skip checkpoint for a step."""
    os.makedirs(chk_dir, exist_ok=True)
    cp_path = os.path.join(chk_dir, f"{step_name}.passed")
    if os.path.isfile(cp_path):
        return  # Already has a checkpoint
    ts = _now_iso()
    lines = [f"step={step_name}", f"passed_at={ts}", f"pipeline={pipeline}", "manual_skip=true"]
    Path(cp_path).write_text("\n".join(lines) + "\n")


def kill_pipeline(reason: str = "") -> None:
    """Set the killed flag, kill dev server, signal driver."""
    import signal

    state = require_state()

    def mutate(s: PipelineState):
        s.killed = True
        s.kill_reason = reason
    state_mgr().update(mutate)

    # Kill dev server if PID is recorded
    if state.dev_server_pid > 0:
        try:
            os.kill(state.dev_server_pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    # Courtesy signal to driver process (triggers its _shutdown)
    if state.driver_pid > 0:
        try:
            os.kill(state.driver_pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass  # Stale PID — flag check is authoritative


def get_summary(state: PipelineState, as_json: bool = False) -> str | dict:
    """Return pipeline summary as string or dict."""
    if as_json:
        now = datetime.now(timezone.utc)
        steps_detail = {}
        for name in state.step_names_ordered():
            ss = state.steps[name]
            entry: dict = {"status": ss.status.value, "retries": ss.retries}
            if ss.started_at:
                entry["started_at"] = ss.started_at
                start = datetime.fromisoformat(ss.started_at.replace("Z", "+00:00"))
                if ss.completed_at:
                    end = datetime.fromisoformat(ss.completed_at.replace("Z", "+00:00"))
                    entry["duration_s"] = round((end - start).total_seconds())
                elif ss.status == StepStatus.IN_PROGRESS:
                    entry["elapsed_s"] = round((now - start).total_seconds())
            if ss.last_error:
                entry["last_error"] = ss.last_error
            steps_detail[name] = entry

        pipeline_elapsed = None
        if state.created_at:
            created = datetime.fromisoformat(state.created_at.replace("Z", "+00:00"))
            pipeline_elapsed = round((now - created).total_seconds())

        return {
            "pipeline": state.pipeline, "preset": state.preset,
            "packages": state.affected_packages,
            "elapsed_s": pipeline_elapsed,
            "driver_pid": state.driver_pid,
            "killed": state.killed,
            "steps": steps_detail,
            "session_dir": state.session_dir,
        }

    lines = [f"Pipeline: {state.pipeline} | Preset: {state.preset}"]
    lines.append(f"Packages: {', '.join(state.affected_packages) or '(none)'}")
    complete, failed, pending, in_prog = [], [], [], []
    for name in state.step_names_ordered():
        ss = state.steps[name]
        if ss.status == StepStatus.COMPLETE:
            complete.append(name)
        elif ss.status == StepStatus.FAILED:
            failed.append(f"{name}(retry {ss.retries})")
        elif ss.status == StepStatus.IN_PROGRESS:
            in_prog.append(name)
        else:
            pending.append(name)
    if complete:
        lines.append(f"DONE: {', '.join(complete)}")
    if in_prog:
        lines.append(f"RUNNING: {', '.join(in_prog)}")
    if failed:
        lines.append(f"FAILED: {', '.join(failed)}")
    if pending:
        lines.append(f"PENDING: {', '.join(pending)}")
    return "\n".join(lines)


# ── Dev server ────────────────────────────────────────────────────────────

PORT_RANGE_END = DEFAULT_DEV_PORT + 10


def _is_port_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def allocate_dev_server() -> dict:
    state = require_state()
    if state.dev_server_port and _is_port_listening(state.dev_server_port):
        return {"port": state.dev_server_port, "running": True, "reused": True}

    for port in range(DEFAULT_DEV_PORT, PORT_RANGE_END):
        if not _is_port_listening(port):
            def mutate(s: PipelineState):
                s.dev_server_port = port
            state_mgr().update(mutate)
            return {"port": port, "running": False, "reused": False}

    raise RuntimeError(f"No free port in range {DEFAULT_DEV_PORT}-{PORT_RANGE_END - 1}")


def dev_server_status() -> dict:
    state = require_state()
    port = state.dev_server_port or DEFAULT_DEV_PORT
    return {"port": port, "running": _is_port_listening(port), "allocated": state.dev_server_port > 0}


# ── Artifact archival ────────────────────────────────────────────────────────

def archive_step_artifacts(step_def, session_dir: str, attempt: int):
    """Move evidence artifacts from previous attempt to _retries/ subdirectory."""
    import glob as globmod
    archive_dir = os.path.join(session_dir, "_retries", f"{step_def.name}_attempt_{attempt}")
    moved = []
    for rule in step_def.evidence:
        if rule.file_glob:
            for f in globmod.glob(os.path.join(session_dir, rule.file_glob)):
                if not moved:
                    os.makedirs(archive_dir, exist_ok=True)
                dest = os.path.join(archive_dir, os.path.basename(f))
                os.rename(f, dest)
                moved.append(os.path.basename(f))
    _archive_output_section(session_dir, step_def.name, archive_dir)
    return moved


def _archive_output_section(session_dir: str, step_name: str, archive_dir: str):
    """Move previous ## step_name section from pipeline-output.md to archive."""
    path = os.path.join(session_dir, "pipeline-output.md")
    if not os.path.isfile(path):
        return
    text = Path(path).read_text()
    pattern = rf'(\n## {re.escape(step_name)}\n.*?)(?=\n## |\Z)'
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return
    os.makedirs(archive_dir, exist_ok=True)
    Path(os.path.join(archive_dir, "output-section.md")).write_text(match.group(1))
    cleaned = text[:match.start()] + text[match.end():]
    Path(path).write_text(cleaned)
