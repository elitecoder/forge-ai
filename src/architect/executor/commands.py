"""Pipeline v2 CLI — state management, direct command execution, AI dispatch."""

import argparse
import json
import os
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from architect.executor.engine.state import StateManager, PipelineState, StepState, StepStatus, _state_to_dict
from architect.executor.engine.registry import load_preset, get_model, Preset, PipelineDefinition
from architect.executor.engine.evidence import EvidenceChecker
from architect.executor.engine.checkpoint import (
    write_checkpoint, verify_checkpoint, verify_all_checkpoints, clear_checkpoints,
)
from architect.executor.engine.runner import (
    execute_command, generate_fix_prompt, generate_ai_prompt, generate_ai_fix_prompt,
    build_context, StepResult,
)
from architect.core.session import (
    list_sessions as _core_list_sessions,
    cleanup_sessions as _core_cleanup_sessions,
)
from architect.executor.engine.utils import DEFAULT_DEV_PORT, repo_root, find_active_session, SESSIONS_BASE


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Timeout / turn limits ───────────────────────────────────────────────────

STALE_THRESHOLD_SEC = 3600

STEP_LIMITS: dict[str, dict] = {
    "fix":          {"max_turns": 0, "timeout": {"sonnet": 3600000, "opus": 3600000}},
    "code":         {"max_turns": 0, "timeout": {"sonnet": 3600000, "opus": 3600000}},
    "code_review":  {"max_turns": 0, "timeout": {"sonnet": 3600000, "opus": 3600000}},
    "visual_test":  {"max_turns": 0, "timeout": {"sonnet": 3600000, "opus": 3600000}},
    "create_pr":    {"max_turns": 0, "timeout": {"sonnet": 3600000, "opus": 3600000}},
}


def _resolve_timeout(step_name: str, model: str, is_fix: bool = False) -> int:
    key = "fix" if is_fix else step_name
    limits = STEP_LIMITS.get(key, STEP_LIMITS["fix"])
    return limits.get("timeout", {}).get(model, 3600000)


def _resolve_max_turns(step_name: str, is_fix: bool = False) -> int:
    key = "fix" if is_fix else step_name
    return STEP_LIMITS.get(key, STEP_LIMITS["fix"]).get("max_turns", 0)


REPO_ROOT = repo_root()

def _last_activity_age(session_dir: str) -> float | None:
    log_path = os.path.join(session_dir, "pipeline-activity.log")
    if not os.path.isfile(log_path):
        return None
    try:
        return datetime.now(timezone.utc).timestamp() - os.path.getmtime(log_path)
    except OSError:
        return None


PRESETS_DIR = Path(__file__).resolve().parent.parent / "presets"


def _active_session_dir() -> Path | None:
    return find_active_session()


def _state_mgr(session_dir: Path | str | None = None) -> StateManager:
    if session_dir is None:
        session_dir = _active_session_dir()
    if session_dir is None:
        print("ERROR: No active pipeline session found. Run: architect execute init <full|lightweight>", file=sys.stderr)
        sys.exit(1)
    return StateManager(Path(session_dir) / "agent-state.json")


def _checkpoint_dir(session_dir: str | None = None) -> str:
    if session_dir is None:
        sd = _active_session_dir()
        if sd is None:
            print("ERROR: No active pipeline session found.", file=sys.stderr)
            sys.exit(1)
        session_dir = str(sd)
    return os.path.join(session_dir, "checkpoints")


def _require_state() -> PipelineState:
    mgr = _state_mgr()
    if not mgr.exists():
        print("ERROR: No active pipeline. Run: architect execute init <full|lightweight>", file=sys.stderr)
        sys.exit(1)
    return mgr.load()


def _load_preset_for_state(state: PipelineState) -> Preset:
    return load_preset(PRESETS_DIR / (state.preset or "hz-web"))


def _is_worktree() -> tuple[bool, str]:
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"], capture_output=True, text=True,
        ).stdout.strip()
        gitdir = subprocess.run(
            ["git", "rev-parse", "--git-dir"], capture_output=True, text=True,
        ).stdout.strip()
        if common and gitdir and common != gitdir:
            return True, os.getcwd()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return False, ""


def _session_name(slug: str = "") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%SZ")
    # Always prefer ticket pattern from branch so find_active_session() can
    # locate the session on --resume and agent pass/fail calls.
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        branch = "unknown"
    ticket_pattern = os.environ.get("ARCHITECT_TICKET_PATTERN", r"[A-Z]+-\d+")
    match = re.search(ticket_pattern, branch)
    if match:
        return f"{match.group(0)}_{ts}"
    # Always use branch-derived prefix so find_active_session() can locate it
    name = re.sub(r"[/:]", "-", branch)
    return f"{name}_{ts}"


MAX_RETRIES = 3


def _is_permanently_failed(step_state) -> bool:
    return step_state.status == StepStatus.FAILED and step_state.retries >= MAX_RETRIES


def _dep_is_satisfied(step_state) -> bool:
    return step_state.status == StepStatus.COMPLETE


def _check_dependencies(state: PipelineState, target_name: str) -> list[str]:
    if state.dependency_graph:
        deps = state.dependency_graph.get(target_name, [])
        return [f"{dep} ({state.steps[dep].status.value})" for dep in deps
                if not _dep_is_satisfied(state.steps[dep])]

    step_names = state.step_names_ordered()
    idx = step_names.index(target_name)
    return [f"{prev} ({state.steps[prev].status.value})" for prev in step_names[:idx]
            if not _dep_is_satisfied(state.steps[prev])]


def _require_dependencies(state: PipelineState, target_name: str):
    incomplete = _check_dependencies(state, target_name)
    if incomplete:
        print(f"ERROR: Cannot advance to '{target_name}': dependencies not complete: {', '.join(incomplete)}",
              file=sys.stderr)
        sys.exit(1)


# ── Artifact archival ───────────────────────────────────────────────────────

def _archive_step_artifacts(step_def, session_dir: str, attempt: int):
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
    if moved:
        print(f"Archived {len(moved)} artifact(s) to _retries/{step_def.name}_attempt_{attempt}/")


def _archive_output_section(session_dir: str, step_name: str, archive_dir: str):
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


# ── Commands ────────────────────────────────────────────────────────────────

def cmd_init(args):
    preset = load_preset(PRESETS_DIR / args.preset)

    if args.pipeline not in preset.pipelines:
        print(f"ERROR: Unknown pipeline '{args.pipeline}'. Available: {', '.join(preset.pipelines)}", file=sys.stderr)
        sys.exit(1)

    is_wt, wt_path = _is_worktree()
    if not is_wt:
        if not getattr(args, "no_worktree", False):
            print("ERROR: Not in a git worktree. Use --no-worktree to override.", file=sys.stderr)
            sys.exit(1)
        print("WARNING: Not in a git worktree. Running pipeline in main working directory.")
        print()

    session_dir = str(SESSIONS_BASE / _session_name(slug=getattr(args, "slug", "")))
    os.makedirs(session_dir, exist_ok=True)

    clear_checkpoints(_checkpoint_dir(session_dir))

    pipeline_def = preset.pipelines[args.pipeline]
    step_names = pipeline_def.steps
    steps = {name: StepState() for name in step_names}

    state = PipelineState(
        phase="execution",
        pipeline=args.pipeline,
        preset=args.preset,
        model_profile="",
        plan_file=getattr(args, "plan", "") or "",
        current_step=step_names[0],
        steps=steps,
        step_order=list(step_names),
        dependency_graph=pipeline_def.dependencies,
        affected_packages=args.packages or [],
        session_dir=session_dir,
        is_worktree=is_wt,
        worktree_path=wt_path,
        created_at=_now_iso(),
        updated_at=_now_iso(),
    )

    _state_mgr(session_dir).save(state)

    mode = "DAG" if pipeline_def.dependencies else "linear"
    print(f"Pipeline initialized: {args.pipeline} ({len(step_names)} steps, {mode} mode)")
    print(f"Models: fix={preset.models.get('fix', 'sonnet')}, "
          f"code_review={preset.models.get('code_review', 'sonnet')}, "
          f"visual_test={preset.models.get('visual_test', 'sonnet')}")
    print(f"Session directory: {session_dir}")


def cmd_status(args):
    state = _require_state()
    print(json.dumps(_state_to_dict(state), indent=2))


def cmd_next(args):
    state = _require_state()
    preset = _load_preset_for_state(state)

    runnable = []
    in_progress = []
    blocked = []

    for name in state.step_names_ordered():
        ss = state.steps[name]
        step_def = preset.steps.get(name)
        if ss.status == StepStatus.COMPLETE:
            continue
        if _is_permanently_failed(ss):
            continue
        if ss.status == StepStatus.IN_PROGRESS:
            entry: dict = {"step": name, "stale": False}
            age = _last_activity_age(state.session_dir)
            if age is not None and age > STALE_THRESHOLD_SEC:
                entry["stale"] = True
                entry["idle_seconds"] = int(age)
            in_progress.append(entry)
            continue

        if state.dependency_graph:
            deps = state.dependency_graph.get(name, [])
            deps_met = all(_dep_is_satisfied(state.steps[d]) for d in deps)
            has_failed_deps = any(_is_permanently_failed(state.steps[d]) for d in deps)
        else:
            step_names = state.step_names_ordered()
            idx = step_names.index(name)
            deps_met = all(_dep_is_satisfied(state.steps[p]) for p in step_names[:idx])
            has_failed_deps = any(_is_permanently_failed(state.steps[p]) for p in step_names[:idx])

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
        print(json.dumps({"step": None, "message": "All steps complete"}))
        return

    result: dict = {"runnable": runnable, "in_progress": in_progress, "blocked": blocked}
    print(json.dumps(result))


def cmd_execute(args):
    """Run a command step directly. No AI dispatch."""
    state = _require_state()
    preset = _load_preset_for_state(state)
    name = args.step

    if name not in state.steps:
        print(f"ERROR: Unknown step: {name}", file=sys.stderr)
        sys.exit(1)

    step_def = preset.steps.get(name)
    if not step_def or step_def.step_type != "command":
        print(f"ERROR: Step '{name}' is not a command step (type: {step_def.step_type if step_def else '?'}). "
              f"Use 'dispatch' for AI steps.", file=sys.stderr)
        sys.exit(1)

    _require_dependencies(state, name)

    def set_running(s: PipelineState):
        s.steps[name].status = StepStatus.IN_PROGRESS
        s.steps[name].started_at = _now_iso()
        s.current_step = name
    _state_mgr().update(set_running)

    print(f"Executing: {name}...")
    start_time = datetime.now(timezone.utc)
    result = execute_command(step_def, state, preset)
    duration = (datetime.now(timezone.utc) - start_time).total_seconds()

    output_file = os.path.join(state.session_dir, "pipeline-output.md")
    try:
        with open(output_file, "a") as f:
            f.write(f"\n## {name}\n\n")
            f.write(f"- **Result:** {'passed' if result.passed else 'failed'}\n")
            f.write(f"- **Command:** `{step_def.run_command}`\n")
            f.write(f"- **Duration:** {duration:.1f}s\n")
            if not result.passed and result.output:
                snippet = result.output[:500].strip()
                f.write(f"- **Error snippet:**\n```\n{snippet}\n```\n")
    except OSError:
        pass

    if result.passed:
        evidence_result = None
        if step_def.evidence:
            chk_dir = _checkpoint_dir(state.session_dir)
            checker = EvidenceChecker(
                state.session_dir, chk_dir, state.steps, state.dependency_graph or None,
            )
            evidence_result = checker.check(step_def)
            if not evidence_result.passed:
                def set_failed(s: PipelineState):
                    s.steps[name].status = StepStatus.FAILED
                    s.steps[name].last_error = evidence_result.message
                    s.steps[name].completed_at = _now_iso()
                _state_mgr().update(set_failed)
                print(json.dumps({"step": name, "result": "failed", "reason": evidence_result.message}))
                return

        def set_complete(s: PipelineState):
            s.steps[name].status = StepStatus.COMPLETE
            s.steps[name].completed_at = _now_iso()
        _state_mgr().update(set_complete)
        write_checkpoint(_checkpoint_dir(state.session_dir), name, state.pipeline, evidence_result)
        print(json.dumps({"step": name, "result": "passed"}))
    else:
        def set_failed(s: PipelineState):
            s.steps[name].retries += 1
            s.steps[name].status = StepStatus.FAILED
            s.steps[name].last_error = result.output[:500] if result.output else "Command failed"
            s.steps[name].completed_at = _now_iso()
        _state_mgr().update(set_failed)

        out = {"step": name, "result": "failed"}
        if result.failed_packages:
            out["failed_packages"] = result.failed_packages
            out["error_files"] = result.error_files
        elif result.error_file:
            out["error_file"] = result.error_file
        print(json.dumps(out))


def cmd_dispatch(args):
    """Get dispatch config for AI steps or fix prompts for command steps."""
    state = _require_state()
    preset = _load_preset_for_state(state)
    name = args.step
    phase = args.phase

    if name not in state.steps:
        print(f"ERROR: Unknown step: {name}", file=sys.stderr)
        sys.exit(1)

    step_def = preset.steps.get(name)
    if not step_def:
        print(f"ERROR: No definition for step: {name}", file=sys.stderr)
        sys.exit(1)

    if step_def.step_type == "inline":
        print(json.dumps({
            "step": name, "type": "inline", "description": step_def.description,
        }))
        return

    if step_def.step_type == "command":
        if phase != "fix":
            print(f"ERROR: Step '{name}' is a command step. Use 'execute' for run phase, "
                  f"'dispatch {name} fix' for fix prompt.", file=sys.stderr)
            sys.exit(1)
        failed_packages = args.failed_packages.split(",") if args.failed_packages else None
        prompt = generate_fix_prompt(step_def, state, preset, failed_packages)
        model = get_model(preset, name, is_fix=True)
        print(json.dumps({
            "step": name, "phase": "fix", "model": model,
            "subagent_type": "general-purpose", "prompt": prompt,
            "max_turns": _resolve_max_turns(name, is_fix=True),
            "timeout_ms": _resolve_timeout(name, model, is_fix=True),
        }, indent=2))
        return

    if step_def.step_type == "ai":
        is_fix = phase == "fix"
        model = get_model(preset, name, is_fix=is_fix)
        if is_fix:
            prompt = generate_ai_fix_prompt(step_def, state, preset)
        else:
            prompt = generate_ai_prompt(step_def, state, preset)
        print(json.dumps({
            "step": name, "phase": phase, "model": model,
            "subagent_type": step_def.subagent_type, "prompt": prompt,
            "max_turns": _resolve_max_turns(name, is_fix=is_fix),
            "timeout_ms": _resolve_timeout(name, model, is_fix=is_fix),
        }, indent=2))
        return


def cmd_run(args):
    state = _require_state()
    name = args.step
    if name not in state.steps:
        print(f"ERROR: Unknown step: {name}", file=sys.stderr)
        sys.exit(1)
    if not getattr(args, "skip_deps", False):
        _require_dependencies(state, name)

    retries = state.steps[name].retries
    if retries > 0 and state.session_dir:
        preset = _load_preset_for_state(state)
        step_def = preset.steps.get(name)
        if step_def and step_def.evidence:
            _archive_step_artifacts(step_def, state.session_dir, retries)

    def mutate(s: PipelineState):
        s.steps[name].status = StepStatus.IN_PROGRESS
        s.steps[name].started_at = _now_iso()
        s.steps[name].completed_at = ""
        s.current_step = name
    _state_mgr().update(mutate)
    print(f"Step '{name}' → in_progress")


def cmd_pass(args):
    state = _require_state()
    name = args.step
    if name not in state.steps:
        print(f"ERROR: Unknown step: {name}", file=sys.stderr)
        sys.exit(1)
    _require_dependencies(state, name)

    preset = _load_preset_for_state(state)
    step_def = preset.steps.get(name)
    chk_dir = _checkpoint_dir(state.session_dir)
    evidence_result = None
    if step_def and step_def.evidence:
        checker = EvidenceChecker(
            state.session_dir, chk_dir, state.steps, state.dependency_graph or None,
        )
        evidence_result = checker.check(step_def)
        if not evidence_result.passed:
            print(f"ERROR: Cannot mark '{name}' as complete: {evidence_result.message}", file=sys.stderr)
            sys.exit(1)
        print(f"Evidence OK: {evidence_result.message}")

    def mutate(s: PipelineState):
        s.steps[name].status = StepStatus.COMPLETE
        s.steps[name].completed_at = _now_iso()
    _state_mgr().update(mutate)

    cp_path = write_checkpoint(chk_dir, name, state.pipeline, evidence_result)
    print(f"Step '{name}' → complete")
    print(f"Checkpoint written: {cp_path}")

    if state.steps[name].retries > 0:
        pipeline_def = preset.pipelines.get(state.pipeline)
        if pipeline_def and pipeline_def.revalidation_targets:
            targets = [t for t in pipeline_def.revalidation_targets
                       if t != name and t in state.steps and state.steps[t].status == StepStatus.COMPLETE]
            if targets:
                def reset_targets(s: PipelineState):
                    for t in targets:
                        s.steps[t].status = StepStatus.PENDING
                        s.steps[t].retries = 0
                        s.steps[t].last_error = ""
                _state_mgr().update(reset_targets)
                for t in targets:
                    cp = os.path.join(chk_dir, f"{t}.passed")
                    if os.path.isfile(cp):
                        os.unlink(cp)
                print(f"Revalidation triggered: {', '.join(targets)} reset to pending")


def cmd_fail(args):
    state = _require_state()
    name = args.step
    if name not in state.steps:
        print(f"ERROR: Unknown step: {name}", file=sys.stderr)
        sys.exit(1)

    def mutate(s: PipelineState):
        s.steps[name].retries += 1
        s.steps[name].status = StepStatus.FAILED
        s.steps[name].last_error = args.error or ""
        s.steps[name].completed_at = _now_iso()
    updated = _state_mgr().update(mutate)
    retries = updated.steps[name].retries
    result = {"step": name, "status": "failed", "retries": retries, "max_retries": MAX_RETRIES}

    if retries >= MAX_RETRIES:
        result["pipeline_exhausted"] = True

    print(json.dumps(result))


def cmd_reset(args):
    state = _require_state()
    name = args.step
    if name not in state.steps:
        print(f"ERROR: Unknown step: {name}", file=sys.stderr)
        sys.exit(1)

    no_retry_inc = getattr(args, "no_retry_inc", False)

    def mutate(s: PipelineState):
        s.steps[name].status = StepStatus.PENDING
        if not no_retry_inc:
            s.steps[name].retries += 1
        s.steps[name].last_error = ""
        s.steps[name].started_at = ""
        s.steps[name].completed_at = ""
    _state_mgr().update(mutate)

    cp_path = os.path.join(_checkpoint_dir(state.session_dir), f"{name}.passed")
    if os.path.isfile(cp_path):
        os.unlink(cp_path)
    print(f"Step '{name}' → reset to pending")


def cmd_skip(args):
    """Mark a step as complete (manual override), skip evidence checks."""
    from architect.executor.engine.pipeline_ops import skip_step
    try:
        skip_step(args.step)
        print(f"Step '{args.step}' → skipped (marked complete)")
    except (ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_resume_from(args):
    """Reset target step to pending, mark all prior as complete."""
    from architect.executor.engine.pipeline_ops import resume_from
    try:
        marked = resume_from(args.step)
        if marked:
            print(f"Marked complete: {', '.join(marked)}")
        print(f"Step '{args.step}' → ready to run")
    except (ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_kill(args):
    """Graceful pipeline termination."""
    from architect.executor.engine.pipeline_ops import kill_pipeline
    reason = getattr(args, "reason", "") or ""
    try:
        kill_pipeline(reason)
        print(f"Pipeline killed{': ' + reason if reason else ''}")
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_summary(args):
    """Compact pipeline status for orchestrator context management."""
    from architect.executor.engine.pipeline_ops import get_summary
    state = _require_state()
    if args.json:
        print(json.dumps(get_summary(state, as_json=True)))
    else:
        print(get_summary(state))


def cmd_add_packages(args):
    _require_state()
    def mutate(s: PipelineState):
        s.affected_packages = sorted(set(s.affected_packages) | set(args.packages))
    updated = _state_mgr().update(mutate)
    print(f"Affected packages updated: {', '.join(updated.affected_packages)}")


def cmd_verify(args):
    state = _require_state()
    all_valid, present, missing = verify_all_checkpoints(
        _checkpoint_dir(state.session_dir), state.step_names_ordered(), exclude={"create_pr"},
    )
    if all_valid:
        print(f"PIPELINE COMPLETE — all {len(present)} steps verified. Safe to create PR / push.")
    else:
        print(f"CHECKPOINT GATE FAILED — missing/invalid checkpoints for: {', '.join(missing)}")
        if present:
            print(f"Present: {', '.join(present)}")
        sys.exit(1)


def cmd_model(args):
    state = _require_state()
    preset = _load_preset_for_state(state)
    model = get_model(preset, args.step, is_fix=(args.phase == "fix"))
    print(model)


def cmd_cleanup(args):
    removed = _core_cleanup_sessions(SESSIONS_BASE, args.older_than)
    if removed:
        print(f"Removed {len(removed)} session(s):")
        for p in removed:
            print(f"  {p}")
    else:
        print("No sessions to clean up.")


def cmd_sessions(args):
    sessions = _core_list_sessions(SESSIONS_BASE)
    if not sessions:
        print("No pipeline sessions found.")
        return
    for s in sessions:
        print(f"  {s['name']}  ({s['age_days']}d old)  {s['path']}")


# ── Dev server port management ──────────────────────────────────────────────

PORT_RANGE_END = DEFAULT_DEV_PORT + 10


def _is_port_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _find_free_port() -> int:
    for port in range(DEFAULT_DEV_PORT, PORT_RANGE_END):
        if not _is_port_listening(port):
            return port
    raise RuntimeError(f"No free port in range {DEFAULT_DEV_PORT}-{PORT_RANGE_END - 1}")


def cmd_dev_server(args):
    """Allocate or check dev server port."""
    state = _require_state()

    if args.action == "allocate":
        if state.dev_server_port and _is_port_listening(state.dev_server_port):
            print(json.dumps({
                "port": state.dev_server_port, "running": True, "reused": True,
            }))
            return

        port = _find_free_port()

        def mutate(s: PipelineState):
            s.dev_server_port = port
        _state_mgr().update(mutate)

        print(json.dumps({
            "port": port, "running": False, "reused": False,
        }))

    elif args.action == "status":
        port = state.dev_server_port or DEFAULT_DEV_PORT
        print(json.dumps({
            "port": port,
            "running": _is_port_listening(port),
            "allocated": state.dev_server_port > 0,
        }))


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(prog="architect-executor",
                                     description="Pipeline v2 — multi-step orchestrator")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("init")
    p.add_argument("pipeline")
    p.add_argument("--preset", default="hz-web")
    p.add_argument("--plan", default="", help="Path to plan file for the code step")
    p.add_argument("--no-worktree", action="store_true", help="Allow running outside a git worktree")
    p.add_argument("packages", nargs="*")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("status")
    p.add_argument("--json", action="store_true", default=True)
    p.set_defaults(func=cmd_status)

    sub.add_parser("next").set_defaults(func=cmd_next)

    p = sub.add_parser("execute", help="Run a command step directly (no AI)")
    p.add_argument("step")
    p.set_defaults(func=cmd_execute)

    p = sub.add_parser("dispatch", help="Get AI dispatch config or fix prompt")
    p.add_argument("step")
    p.add_argument("phase", nargs="?", default="run", choices=["run", "fix"])
    p.add_argument("--failed-packages", default="", help="Comma-separated failed packages for fix")
    p.set_defaults(func=cmd_dispatch)

    p = sub.add_parser("run", help="Mark step as in_progress")
    p.add_argument("step")
    p.add_argument("--skip-deps", action="store_true", help="Skip dependency check (internal fix loop use)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("pass", help="Mark step as complete")
    p.add_argument("step")
    p.set_defaults(func=cmd_pass)

    p = sub.add_parser("fail")
    p.add_argument("step")
    p.add_argument("error", nargs="?", default="")
    p.set_defaults(func=cmd_fail)

    p = sub.add_parser("reset")
    p.add_argument("step")
    p.add_argument("--no-retry-inc", action="store_true",
                   help="Reset without incrementing retries (used by internal fix loops)")
    p.set_defaults(func=cmd_reset)

    p = sub.add_parser("skip", help="Mark step as complete (manual override, no evidence)")
    p.add_argument("step")
    p.set_defaults(func=cmd_skip)

    p = sub.add_parser("resume-from", help="Reset step to pending, mark all prior as complete")
    p.add_argument("step")
    p.set_defaults(func=cmd_resume_from)

    p = sub.add_parser("kill", help="Graceful pipeline termination")
    p.add_argument("--reason", default="", help="Reason for killing the pipeline")
    p.set_defaults(func=cmd_kill)

    p = sub.add_parser("summary", help="Compact pipeline status for context management")
    p.add_argument("--json", action="store_true", default=False)
    p.set_defaults(func=cmd_summary)

    p = sub.add_parser("add-packages")
    p.add_argument("packages", nargs="+")
    p.set_defaults(func=cmd_add_packages)

    sub.add_parser("verify").set_defaults(func=cmd_verify)

    p = sub.add_parser("model")
    p.add_argument("step")
    p.add_argument("phase", nargs="?", default="run", choices=["run", "fix"])
    p.set_defaults(func=cmd_model)

    p = sub.add_parser("cleanup")
    p.add_argument("--older-than", type=int, default=7)
    p.set_defaults(func=cmd_cleanup)

    p = sub.add_parser("dev-server", help="Allocate or check dev server port")
    p.add_argument("action", choices=["allocate", "status"])
    p.set_defaults(func=cmd_dev_server)

    sub.add_parser("sessions").set_defaults(func=cmd_sessions)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
