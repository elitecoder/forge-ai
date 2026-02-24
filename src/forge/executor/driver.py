"""Pipeline Driver — deterministic orchestrator, zero LLM context.

Accepts a plan file, spawns fresh agents per step via Provider.
No context rot. State lives on disk, not in LLM memory.
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from forge.core.events import emit, EventType
from forge.core.logging import utc_timestamp, log_activity, StatusWriter
from forge.core.runner import AgentRunner
from forge.executor.engine import pipeline_ops
from forge.executor.engine.state import StateManager, PipelineState, StepStatus, _state_to_dict, _dict_to_state
from forge.executor.engine.registry import load_preset, get_model
from forge.executor.engine.runner import build_context
from forge.executor.engine.templates import render
from forge.executor.engine.utils import SESSIONS_BASE, is_bazel_repo
from forge.core.session import generate_slug


PIPELINE_TIMEOUT_S = 21600  # 6 hour wall-clock limit
MAX_STEP_RETRIES = 3
MAX_DEV_SERVER_PRECHECK_FAILURES = 2

_fast_mode = False


def _resolve_model(model: str) -> str:
    return "haiku" if _fast_mode else model


def _set_plugin_dir() -> None:
    """Set FORGE_PLUGIN_DIR env var so ClaudeProvider loads hooks."""
    skin = Path(__file__).resolve().parent.parent.parent.parent / "skins" / "claude-code"
    if skin.is_dir() and (skin / "hooks").is_dir():
        os.environ["FORGE_PLUGIN_DIR"] = str(skin)


def _preflight_hooks() -> list[str]:
    """Check that lint/format tools are available. Returns list of issues."""
    issues: list[str] = []

    eslint = shutil.which("eslint")
    if not eslint:
        issues.append("eslint not found — lint hook will not run. Fix: npm install -g eslint")

    # Check if eslint can find a config in the current repo
    if eslint and not os.environ.get("FORGE_ESLINT_CONFIG"):
        try:
            result = subprocess.run(
                ["eslint", "--print-config", "x.ts"],
                capture_output=True, text=True, timeout=5,
            )
            # Exit code 1+ means config not found. Use more robust pattern matching.
            if result.returncode != 0:
                stderr_str = result.stderr if isinstance(result.stderr, str) else ""
                stdout_str = result.stdout if isinstance(result.stdout, str) else ""
                output = stderr_str + stdout_str
                # Match common eslint config error patterns
                if re.search(r"(eslint\.config|config.*not found|no.*config)", output, re.IGNORECASE):
                    issues.append(
                        "eslint cannot find config for this repo. "
                        "Add \"eslint_config\" to your preset manifest"
                    )
        except Exception:
            pass

    prettier = shutil.which("prettier")
    if not prettier:
        issues.append("prettier not found — format hook will not run. Fix: npm install -g prettier")

    if not issues:
        _safe_print(f"  {GREEN}Hooks:{RESET} eslint + prettier ready")
    else:
        _safe_print(f"  {YELLOW}Hooks:{RESET}")
        for issue in issues:
            _safe_print(f"    {YELLOW}!{RESET} {issue}")
        _safe_print()

    return issues


def _set_hook_build_cmd(state: PipelineState, preset) -> None:
    """Set FORGE_BUILD_CMD env var for Claude Code stop hooks."""
    cmd_template = preset.build_command
    if preset.bazel_build_command and is_bazel_repo():
        cmd_template = preset.bazel_build_command
    if not cmd_template:
        return
    ctx = build_context(state, preset)
    packages = state.affected_packages
    if packages:
        ctx["BUILD_TARGETS"] = " ".join(f"//{p}/..." for p in packages)
    elif "{{BUILD_TARGETS}}" in cmd_template:
        # No packages known yet and template requires them — skip until detected
        return
    os.environ["FORGE_BUILD_CMD"] = render(cmd_template, ctx)


def _set_hook_eslint_config(preset) -> None:
    """Set FORGE_ESLINT_CONFIG env var from preset if configured."""
    if not preset.eslint_config:
        return
    config_path = Path(preset.eslint_config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    if config_path.is_file():
        os.environ["FORGE_ESLINT_CONFIG"] = str(config_path)


# ── Terminal UI ────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"


_recent_log: deque[str] = deque(maxlen=20)
_activity_log_path: str | None = None
_session_dir: str | None = None
_dev_server_proc: subprocess.Popen | None = None
_shutting_down: bool = False


def _kill_child_processes():
    my_pid = os.getpid()
    try:
        subprocess.run(["pkill", "-TERM", "-P", str(my_pid)], capture_output=True, timeout=3)
    except Exception:
        pass
    time.sleep(0.5)
    try:
        subprocess.run(["pkill", "-KILL", "-P", str(my_pid)], capture_output=True, timeout=3)
    except Exception:
        pass


def _ts():
    return utc_timestamp()


def _safe_print(*args, **kwargs):
    """Print to stdout, silently ignoring BrokenPipeError."""
    try:
        print(*args, **kwargs)
    except BrokenPipeError:
        pass


def log(step, msg, style=""):
    ts = _ts()
    _safe_print(f"{DIM}[{ts}]{RESET} {step:<20s} {style}{msg}{RESET}", flush=True)
    clean = re.sub(r'\033\[[0-9;]*m', '', f"[{ts}] {step:<20s} {msg}")
    _recent_log.append(clean)
    if _activity_log_path:
        try:
            with open(_activity_log_path, "a") as f:
                f.write(clean + "\n")
        except OSError:
            pass


def log_ok(step, msg, duration_s=None):
    dur = f"  {DIM}{duration_s:.0f}s{RESET}" if duration_s else ""
    log(step, f"{GREEN}✓{RESET} {msg}{dur}")


def log_fail(step, msg):
    log(step, f"{RED}✗{RESET} {msg}")


def log_info(step, msg):
    log(step, f"⣾ {msg}", style=CYAN)


def log_revalidation(steps):
    _safe_print(f"{DIM}[{_ts()}]{RESET} {'':20s} {YELLOW}↳ revalidation: {', '.join(steps)} reset{RESET}")


def log_headline(msg):
    _safe_print(f"\n{BOLD}{msg}{RESET}\n")


# ── Status file ───────────────────────────────────────────────────────────

_ANSI_RE = re.compile(r'\033\[[0-9;]*m')
_STATUS_ICONS = {"complete": "done", "in_progress": "RUNNING", "failed": "FAILED",
                 "pending": "pending"}


def _write_status(session_dir: str):
    try:
        state = pipeline_ops.require_state()
        status = pipeline_ops.get_summary(state, as_json=True)
    except Exception:
        return
    if not isinstance(status, dict):
        return
    try:
        _write_status_impl(session_dir, status)
    except Exception:
        pass


def _write_status_impl(session_dir: str, status: dict):
    now = datetime.now(timezone.utc)

    steps = status.get("steps", {})
    order = list(steps.keys())
    done_count = sum(1 for s in steps.values() if s.get("status") == "complete")
    elapsed_s = status.get("elapsed_s")
    elapsed = f"{elapsed_s // 60}m {elapsed_s % 60:02d}s" if elapsed_s else "—"

    pid = status.get("driver_pid", 0)
    pid_label = f" | PID: {pid}" if pid else ""
    killed = status.get("killed", False)
    killed_label = " | **KILLED**" if killed else ""

    lines = [
        f"# Pipeline Status",
        f"**{status.get('pipeline', '?')}** | {status.get('preset', '?')}"
        f" | Elapsed: {elapsed} | {done_count}/{len(steps)} complete"
        f"{pid_label}{killed_label}",
        "",
        "| Step | Status | Duration | Attempt | Detail |",
        "|------|--------|----------|---------|--------|",
    ]

    for name, ss in steps.items():
        st = ss.get("status", "pending")
        label = _STATUS_ICONS.get(st, st)
        retries = ss.get("retries", 0)
        attempt = str(retries + 1) if st in ("in_progress", "failed") else ""

        dur = "—"
        started = ss.get("started_at", "")
        completed = ss.get("completed_at", "")
        if started:
            try:
                s = datetime.fromisoformat(started.replace("Z", "+00:00"))
                e = datetime.fromisoformat(completed.replace("Z", "+00:00")) if completed else now
                d = int((e - s).total_seconds())
                dur = f"{d // 60}m {d % 60:02d}s" if d >= 60 else f"{d}s"
            except (ValueError, TypeError):
                pass

        detail = ""
        if st == "failed" and ss.get("last_error"):
            detail = ss["last_error"][:60]

        lines.append(f"| {name} | {label} | {dur} | {attempt} | {detail} |")

    if _recent_log:
        lines.append("")
        lines.append("## Recent Activity")
        for entry in _recent_log:
            lines.append(entry)

    lines.append("")
    content = "\n".join(lines)

    from forge.core.logging import atomic_write_file
    atomic_write_file(os.path.join(session_dir, "pipeline-status.md"), content)


# ── Periodic status updater ────────────────────────────────────────────────

_status_writer: StatusWriter | None = None


def _start_status_updater(session_dir: str):
    global _status_writer
    _status_writer = StatusWriter(lambda: _write_status(session_dir), interval=15.0)
    _status_writer.start()


def _stop_status_updater():
    global _status_writer
    if _status_writer:
        _status_writer.stop()
    _status_writer = None


# ── Plan discovery ─────────────────────────────────────────────────────────

def _read_planner_slug(plan_dir: str | None) -> str:
    """Read slug from a planner session's state file, avoiding a second LLM call."""
    if not plan_dir:
        return ""
    state_file = Path(plan_dir).expanduser().resolve() / ".planner-state.json"
    if not state_file.is_file():
        return ""
    try:
        data = json.loads(state_file.read_text())
        return data.get("slug", "")
    except (json.JSONDecodeError, OSError):
        return ""


def discover_plan(plan_path: str | None, plan_dir: str | None) -> str:
    if plan_path:
        p = Path(plan_path).expanduser().resolve()
        if not p.is_file():
            print(f"ERROR: Plan file not found: {p}", file=sys.stderr)
            sys.exit(1)
        return str(p)

    if plan_dir:
        d = Path(plan_dir).expanduser().resolve()
        if not d.is_dir():
            print(f"ERROR: Plan directory not found: {d}", file=sys.stderr)
            sys.exit(1)
        for name in ["final-plan.md", "plan.md"]:
            candidate = d / name
            if candidate.is_file():
                return str(candidate)
        mds = sorted(d.glob("*.md"))
        if mds:
            return str(mds[0])
        print(f"ERROR: No plan file found in {d}", file=sys.stderr)
        sys.exit(1)

    print("ERROR: Provide --plan <file> or --plan-dir <directory>", file=sys.stderr)
    sys.exit(1)


# ── Step execution ─────────────────────────────────────────────────────────

def execute_command_step(step_name: str) -> dict:
    """Run a command step via direct pipeline_ops calls."""
    from forge.executor.engine.runner import execute_command
    from forge.executor.engine.evidence import EvidenceChecker
    from forge.executor.engine.checkpoint import write_checkpoint

    state = pipeline_ops.require_state()
    preset = pipeline_ops.load_preset_for_state(state)
    step_def = preset.steps.get(step_name)

    if not step_def or step_def.step_type != "command":
        return {"step": step_name, "result": "failed"}

    pipeline_ops.mark_running(step_name)
    result = execute_command(step_def, state, preset)

    if result.passed:
        try:
            pipeline_ops.mark_passed(step_name)
        except RuntimeError:
            return {"step": step_name, "result": "failed", "reason": "evidence failed"}
        return {"step": step_name, "result": "passed"}
    else:
        pipeline_ops.mark_failed(step_name, result.output[:500] if result.output else "Command failed")
        out = {"step": step_name, "result": "failed"}
        if result.failed_packages:
            out["failed_packages"] = result.failed_packages
            out["error_files"] = result.error_files
        return out


def _snapshot_worktree(cwd: str) -> set[str]:
    files: set[str] = set()
    for cmd in (
        ["git", "diff", "--name-only"],
        ["git", "diff", "--name-only", "--cached"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    ):
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                if line.strip():
                    files.add(line.strip())
    return files


def _enforce_file_allowlist(step_name: str, cwd: str,
                            before: set[str]) -> list[str]:
    try:
        preset = pipeline_ops.load_preset_for_state(pipeline_ops.require_state())
    except Exception:
        return []
    step_def = preset.steps.get(step_name)
    if not step_def or step_def.allowed_file_patterns is None:
        return []

    after = _snapshot_worktree(cwd)
    new_files = after - before
    if not new_files:
        return []

    from fnmatch import fnmatch
    allowed = step_def.allowed_file_patterns
    violations = []
    for path in sorted(new_files):
        if not allowed or not any(fnmatch(path, pat) for pat in allowed):
            violations.append(path)

    if violations:
        subprocess.run(
            ["git", "checkout", "--"] + violations,
            capture_output=True, cwd=cwd,
        )

    return violations


def _get_dispatch_config(step_name: str, phase: str,
                         failed_packages: list[str] | None = None) -> dict:
    """Get dispatch configuration for an AI step. Direct function call replacement for cli("dispatch", ...)."""
    from forge.executor.engine.runner import (
        generate_fix_prompt, generate_ai_prompt, generate_ai_fix_prompt,
    )
    from forge.executor.commands import _resolve_timeout, _resolve_max_turns

    state = pipeline_ops.require_state()
    preset = pipeline_ops.load_preset_for_state(state)
    step_def = preset.steps.get(step_name)

    if not step_def:
        raise RuntimeError(f"No definition for step: {step_name}")

    if step_def.step_type == "inline":
        return {"step": step_name, "type": "inline", "description": step_def.description}

    if step_def.step_type == "command":
        prompt = generate_fix_prompt(step_def, state, preset, failed_packages)
        model = get_model(preset, step_name, is_fix=True)
        return {
            "step": step_name, "phase": "fix", "model": model,
            "subagent_type": "general-purpose", "prompt": prompt,
            "max_turns": _resolve_max_turns(step_name, is_fix=True),
            "timeout_ms": _resolve_timeout(step_name, model, is_fix=True),
        }

    if step_def.step_type == "ai":
        is_fix = phase == "fix"
        model = get_model(preset, step_name, is_fix=is_fix)
        if is_fix:
            prompt = generate_ai_fix_prompt(step_def, state, preset)
        else:
            prompt = generate_ai_prompt(step_def, state, preset)
        return {
            "step": step_name, "phase": phase, "model": model,
            "subagent_type": step_def.subagent_type, "prompt": prompt,
            "max_turns": _resolve_max_turns(step_name, is_fix=is_fix),
            "timeout_ms": _resolve_timeout(step_name, model, is_fix=is_fix),
        }

    raise RuntimeError(f"Unknown step type: {step_def.step_type}")


def dispatch_ai_step(step_name: str, phase: str, cwd: str,
                     failed_packages: list[str] | None = None,
                     skip_deps: bool = False,
                     attempt: int = 0) -> bool:
    """Dispatch an AI step or fix agent. Returns True if step passed."""
    try:
        config = _get_dispatch_config(step_name, phase, failed_packages)
    except RuntimeError as e:
        log_fail(step_name, str(e))
        return False

    model = _resolve_model(config.get("model", "sonnet"))
    prompt = config.get("prompt", "")
    max_turns = config.get("max_turns", 0)
    timeout_ms = config.get("timeout_ms", 600000)

    if not prompt:
        log_fail(step_name, "empty prompt from dispatch")
        return False

    pipeline_ops.mark_running(step_name, skip_deps=skip_deps)

    label = f"{phase} ({model})" if phase == "fix" else f"({model})"
    log_info(step_name, f"agent {label}")

    if _session_dir:
        _write_status(_session_dir)

    worktree_before = _snapshot_worktree(cwd)
    base = step_name if phase == "run" else f"{step_name}_{phase}"
    transcript_step = f"{base}_attempt{attempt}" if attempt > 0 else base
    runner = AgentRunner(_session_dir, transcript_step, _activity_log_path)
    runner.run(prompt, model, max_turns, cwd, timeout_s=timeout_ms // 1000)

    violations = _enforce_file_allowlist(step_name, cwd, worktree_before)
    if violations:
        log_fail(step_name, f"reverted unauthorized file changes: {', '.join(violations[:5])}")
        pipeline_ops.mark_failed(step_name, f"Modified disallowed files: {', '.join(violations[:5])}")
        return False

    # Check if agent called pass/fail
    state = pipeline_ops.require_state()
    step_status = state.steps.get(step_name)
    if step_status and step_status.status == StepStatus.COMPLETE:
        return True
    if step_status and step_status.status == StepStatus.FAILED:
        return False

    # Agent didn't call pass/fail — check for evidence artifacts as fallback
    if step_status and step_status.status == StepStatus.IN_PROGRESS and _check_step_artifacts(step_name):
        log_info(step_name, "agent forgot pass/fail but artifacts exist — auto-passing")
        try:
            pipeline_ops.mark_passed(step_name)
        except RuntimeError:
            pass
        return True

    pipeline_ops.mark_failed(step_name, "Agent completed without reporting pass/fail status")
    return False


def _check_step_artifacts(step_name: str) -> bool:
    if not _session_dir:
        return False
    artifacts = {
        "code_review": "code-review-verdict.json",
        "code": "code-checklist.json",
        "visual_test": "visual-test-dashboard.html",
    }
    artifact = artifacts.get(step_name)
    return bool(artifact and os.path.isfile(os.path.join(_session_dir, artifact)))


def dispatch_with_judge(step_name: str, phase: str, cwd: str,
                        session_dir: str, judge_config, state) -> bool:
    """Dispatch AI step with judge verification loop."""
    from forge.executor.engine.judge import load_criteria, spawn_judge, save_judge_feedback

    criteria = load_criteria(session_dir, step_name, judge_config, state)

    for attempt in range(judge_config.max_retries):
        ok = dispatch_ai_step(step_name, phase, cwd, attempt=attempt)
        if not ok:
            if attempt < judge_config.max_retries - 1:
                pipeline_ops.reset_step(step_name, no_retry_inc=True)
            continue

        checklist_path = os.path.join(session_dir, f"{step_name}-checklist.json")
        try:
            checklist = json.loads(Path(checklist_path).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            log_fail(step_name, "no structured checklist — agent must produce checklist JSON")
            if attempt < judge_config.max_retries - 1:
                pipeline_ops.reset_step(step_name, no_retry_inc=True)
            continue

        tracked_diff = subprocess.run(
            ["git", "diff", "HEAD"], capture_output=True, text=True, cwd=cwd,
        ).stdout
        untracked_files = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, cwd=cwd,
        ).stdout.strip().splitlines()
        untracked_diff_parts = []
        for uf in untracked_files:
            uf_path = os.path.join(cwd, uf)
            if os.path.isfile(uf_path) and not uf.startswith("."):
                try:
                    content = Path(uf_path).read_text()
                    untracked_diff_parts.append(
                        f"diff --git a/{uf} b/{uf}\n"
                        f"new file mode 100644\n"
                        f"--- /dev/null\n"
                        f"+++ b/{uf}\n"
                        + "\n".join(f"+{line}" for line in content.splitlines())
                    )
                except (OSError, UnicodeDecodeError):
                    pass
        diff = tracked_diff + "\n".join(untracked_diff_parts)

        verdict = spawn_judge(step_name, session_dir, criteria, checklist, diff,
                              _resolve_model(judge_config.model),
                              activity_log_path=_activity_log_path or "",
                              attempt=attempt)
        passed_count = len([i for i in verdict.items if i.get("verdict") == "pass"])
        emit(session_dir, EventType.JUDGE_VERDICT, "executor", step=step_name,
             passed=verdict.passed, pass_count=passed_count, total=len(verdict.items))
        log_info(step_name, f"judge: {passed_count}/{len(verdict.items)} items passed")

        if verdict.passed:
            return True

        feedback_path = save_judge_feedback(session_dir, step_name, attempt, verdict)
        log_info(step_name, f"judge failed — feedback at {feedback_path}")
        if attempt < judge_config.max_retries - 1:
            pipeline_ops.reset_step(step_name, no_retry_inc=True)

    log_fail(step_name, f"judge rejected after {judge_config.max_retries} attempts")
    return False


# ── Step handlers ──────────────────────────────────────────────────────────

def _handle_command_step(name: str, retries: int, cwd: str) -> tuple[str, bool]:
    from forge.executor.engine.agents.fix_agent import run as run_fix_agent

    log_info(name, "running")
    start = time.time()
    result = execute_command_step(name)

    if result.get("result") == "passed":
        log_ok(name, "passed", time.time() - start)
        return name, True

    for attempt in range(MAX_STEP_RETRIES):
        failed_pkgs = result.get("failed_packages", [])
        log_fail(name, f"failed → fix agent (attempt {attempt + 1}/{MAX_STEP_RETRIES})")

        try:
            state = pipeline_ops.require_state()
            preset = pipeline_ops.load_preset_for_state(state)
            step_def = preset.steps.get(name)
        except Exception as exc:
            log_fail(name, f"failed to load state/preset: {exc}")
            return name, False

        if not step_def:
            log_fail(name, f"no step definition for {name}")
            return name, False

        model = _resolve_model(get_model(preset, name, is_fix=True))
        session_dir = state.session_dir or ""
        activity_log = os.path.join(session_dir, "pipeline-activity.log") if session_dir else ""

        pipeline_ops.mark_running(name)
        log_info(name, f"fix agent ({model})")
        if _session_dir:
            _write_status(_session_dir)

        outcome = run_fix_agent(
            step=step_def, state=state, preset=preset,
            cwd=cwd, session_dir=session_dir, activity_log_path=activity_log,
            failed_packages=failed_pkgs or None,
            model=model, max_turns=0, timeout_s=3600,
            attempt=attempt,
        )

        if outcome.passed:
            log_ok(name, f"fixed (attempt {attempt + 1})", time.time() - start)
            return name, True

    log_fail(name, "exhausted retries")
    return name, False


def _handle_code_step(cwd: str, session_dir: str) -> tuple[str, bool]:
    from forge.executor.engine.agents.code_agent import run as run_code_agent

    start = time.time()
    name = "code"

    try:
        state = pipeline_ops.require_state()
        preset = pipeline_ops.load_preset_for_state(state)
    except Exception as exc:
        log_fail(name, f"failed to load state/preset: {exc}")
        return name, False

    model = _resolve_model(get_model(preset, name))
    pipeline_ops.mark_running(name)
    log_info(name, f"agent ({model})")
    if _session_dir:
        _write_status(_session_dir)

    activity_log = os.path.join(session_dir, "pipeline-activity.log")

    outcome = run_code_agent(
        state=state, preset=preset, cwd=cwd,
        session_dir=session_dir, activity_log_path=activity_log,
        model=model, max_turns=0, timeout_s=3600,
    )

    if outcome.detected_packages:
        try:
            def _set_pkgs(s):
                s.affected_packages = outcome.detected_packages
            pipeline_ops.state_mgr().update(_set_pkgs)
            log_info(name, f"detected packages: {', '.join(outcome.detected_packages)}")
            # Refresh hook build command with actual packages
            updated_state = pipeline_ops.require_state()
            _set_hook_build_cmd(updated_state, preset)
        except Exception as exc:
            log_info(name, f"warning: could not update packages in state: {exc}")

    elapsed = time.time() - start
    if outcome.passed:
        log_ok(name, "passed", elapsed)
    else:
        log_fail(name, outcome.reason)
    return name, outcome.passed


def _build_visual_test_config(preset):
    """Build a VisualTestConfig from the preset's visual_test_config dict."""
    from forge.executor.engine.agents.visual_test_agent import VisualTestConfig
    raw = preset.visual_test_config
    if not raw:
        return None
    # Resolve ${PRESET_DIR} in path values
    preset_dir = str(preset.preset_dir)
    def resolve(val):
        if isinstance(val, str):
            return val.replace("${PRESET_DIR}", preset_dir)
        return val
    return VisualTestConfig(
        skill_dir=resolve(raw.get("skill_dir", "")),
        template_path=resolve(raw.get("template_path", "")),
        dashboard_template_path=resolve(raw.get("dashboard_template_path", "")),
        quirks_path=resolve(raw.get("quirks_path", "")),
        playwright_runner_dir=resolve(raw.get("playwright_runner_dir", "")),
        credentials_path=resolve(raw.get("credentials_path", "")),
        fixture_patterns=raw.get("fixture_patterns", []),
        credential_env_vars=raw.get("credential_env_vars", ["EMAIL", "PASSWORD"]),
    )


def _handle_visual_test_step(cwd: str, session_dir: str) -> tuple[str, bool]:
    from forge.executor.engine.agents.visual_test_agent import run as run_visual_test

    start = time.time()
    name = "visual_test"

    try:
        state = pipeline_ops.require_state()
        preset = pipeline_ops.load_preset_for_state(state)
    except Exception as exc:
        log_fail(name, f"failed to load state/preset: {exc}")
        return name, False

    model = _resolve_model(get_model(preset, name))
    pipeline_ops.mark_running(name)
    log_info(name, f"agent ({model})")
    if _session_dir:
        _write_status(_session_dir)

    activity_log = os.path.join(session_dir, "pipeline-activity.log")
    vt_config = _build_visual_test_config(preset)

    outcome = run_visual_test(
        state=state, preset=preset, cwd=cwd,
        session_dir=session_dir, activity_log_path=activity_log,
        model=model, max_turns=0, timeout_s=3600,
        config=vt_config,
    )

    elapsed = time.time() - start
    if outcome.passed:
        log_ok(name, "passed", elapsed)
    else:
        log_fail(name, outcome.reason)
        # Code issues can't be fixed by retrying visual_test — fail permanently
        if outcome.reason and outcome.reason.startswith("Code issue:"):
            from forge.executor.engine.pipeline_ops import MAX_RETRIES
            pipeline_ops.mark_failed(name, outcome.reason)
            _sm = pipeline_ops.state_mgr()
            _sm.update(lambda s: setattr(s.steps[name], "retries", MAX_RETRIES))
    return name, outcome.passed


def _handle_ai_step(name: str, cwd: str, session_dir: str) -> tuple[str, bool]:
    start = time.time()

    if name == "code":
        return _handle_code_step(cwd, session_dir)

    if name == "visual_test":
        return _handle_visual_test_step(cwd, session_dir)

    if name == "code_review":
        ok = run_code_review(cwd, session_dir)
        elapsed = time.time() - start
        if ok:
            log_ok(name, "passed", elapsed)
        else:
            log_fail(name, "failed")
        return name, ok

    try:
        state = pipeline_ops.require_state()
        preset = pipeline_ops.load_preset_for_state(state)
        step_def = preset.steps.get(name)
        if step_def and step_def.judge:
            ok = dispatch_with_judge(name, "run", cwd, session_dir, step_def.judge, state)
            if ok:
                log_ok(name, "passed", time.time() - start)
            else:
                log_fail(name, "failed (judge rejected)")
            return name, ok
    except Exception as exc:
        log_fail(name, f"judge loop error: {exc}")
        return name, False

    for attempt in range(MAX_STEP_RETRIES):
        ok = dispatch_ai_step(name, "run", cwd, attempt=attempt)
        if ok:
            log_ok(name, "passed", time.time() - start)
            return name, True
        if attempt < MAX_STEP_RETRIES - 1:
            log_info(name, f"retry {attempt + 2}/{MAX_STEP_RETRIES}")
            pipeline_ops.reset_step(name)

    log_fail(name, "exhausted retries")
    return name, False


def _handle_inline_step(name: str, session_dir: str) -> tuple[str, bool]:
    if name == "report":
        handle_report_step(session_dir)
        return name, True
    log_info(name, "inline step — no handler")
    return name, False


def _dispatch_one_step(
    step_info: dict, cwd: str, session_dir: str,
    *, dev_server_failure_counter: list | None = None,
) -> tuple[str, bool]:
    name = step_info["step"]
    step_type = step_info["type"]
    retries = step_info.get("retries", 0)

    if name == "visual_test":
        if not ensure_dev_server(cwd):
            if dev_server_failure_counter is not None:
                dev_server_failure_counter[0] += 1
                if dev_server_failure_counter[0] >= MAX_DEV_SERVER_PRECHECK_FAILURES:
                    log_fail("visual_test", f"dev server failed {dev_server_failure_counter[0]} times — giving up")
                    pipeline_ops.mark_failed("visual_test", "Dev server failed to start (budget exhausted)")
                    return name, False
            pipeline_ops.mark_failed("visual_test", "Dev server failed to start")
            return name, False

    if step_type == "command":
        result_name, success = _handle_command_step(name, retries, cwd)
    elif step_type == "ai":
        result_name, success = _handle_ai_step(name, cwd, session_dir)
    elif step_type == "inline":
        result_name, success = _handle_inline_step(name, session_dir)
    else:
        log_fail(name, f"unknown step type: {step_type}")
        return name, False

    if not success:
        state = pipeline_ops.require_state()
        if state.steps.get(name) and state.steps[name].status != StepStatus.FAILED:
            pipeline_ops.mark_failed(name, "step failed")

    if session_dir:
        try:
            from forge.executor.engine.step_summary import write_step_summary
            write_step_summary(session_dir, name, success)
        except Exception:
            pass

    return result_name, success


# ── Report step ────────────────────────────────────────────────────────────

def handle_report_step(session_dir: str):
    pipeline_ops.mark_running("report")

    state = pipeline_ops.require_state()
    status = pipeline_ops.get_summary(state, as_json=True)
    report_path = os.path.join(session_dir, "pipeline-report.md")
    with open(report_path, "w") as f:
        f.write(f"# Pipeline Report\n\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
        if isinstance(status, dict):
            f.write(f"**Pipeline:** {status.get('pipeline', '?')}\n")
            f.write(f"**Preset:** {status.get('preset', '?')}\n\n")
            f.write("## Step Status\n\n")
            for step_name, step_info in status.get("steps", {}).items():
                step_status = step_info.get("status", "unknown").upper()
                label = f"**{step_status}**" if step_status in ("COMPLETE", "FAILED") else step_status
                f.write(f"- {step_name}: {label}\n")

    dashboard = os.path.join(session_dir, "visual-test-dashboard.html")
    if os.path.isfile(dashboard):
        subprocess.run(["open", dashboard], capture_output=True)
        log_ok("report", "dashboard opened")
    else:
        log_ok("report", "report generated")

    try:
        pipeline_ops.mark_passed("report")
    except RuntimeError:
        pass


# ── Two-phase code review ─────────────────────────────────────────────────

def run_code_review(cwd: str, session_dir: str) -> bool:
    from forge.executor.engine.judge import load_criteria, spawn_judge, save_judge_feedback

    dispatch_ok = dispatch_ai_step("code_review", "run", cwd)

    verdict_path = os.path.join(session_dir, "code-review-verdict.json")
    try:
        verdict = json.loads(Path(verdict_path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        if not dispatch_ok:
            return False
        log_fail("code_review", "no verdict JSON found — treating as failure")
        return False

    if verdict.get("verdict") == "CLEAN":
        if not dispatch_ok:
            try:
                pipeline_ops.mark_passed("code_review")
            except RuntimeError:
                pass
        return True

    if verdict.get("verdict") != "HAS_ISSUES":
        log_fail("code_review", f"unknown verdict: {verdict.get('verdict')}")
        return False

    issue_count = verdict.get("issue_count", "?")
    log_info("code_review", f"has {issue_count} issues → fix+judge loop")

    try:
        state = pipeline_ops.require_state()
        preset = pipeline_ops.load_preset_for_state(state)
        step_def = preset.steps.get("code_review")
        judge_config = step_def.judge if step_def else None
    except Exception:
        judge_config = None

    max_retries = judge_config.max_retries if judge_config else 3
    findings = _load_review_findings(session_dir)

    for attempt in range(max_retries):
        pipeline_ops.reset_step("code_review", no_retry_inc=True)
        ok = dispatch_ai_step("code_review", "fix", cwd, skip_deps=True, attempt=attempt)
        if not ok:
            continue

        if not judge_config or not findings:
            return True

        checklist_path = os.path.join(session_dir, "code_review-checklist.json")
        try:
            checklist = json.loads(Path(checklist_path).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            log_info("code_review", "no fix checklist — skipping judge")
            return True

        diff = subprocess.run(
            ["git", "diff", "HEAD"], capture_output=True, text=True, cwd=cwd,
        ).stdout

        judge_verdict = spawn_judge("code_review", session_dir, findings, checklist, diff,
                                    _resolve_model(judge_config.model),
                                    activity_log_path=_activity_log_path or "",
                                    attempt=attempt)
        passed_count = len([i for i in judge_verdict.items if i.get("verdict") == "pass"])
        log_info("code_review", f"judge: {passed_count}/{len(judge_verdict.items)} items passed")

        if judge_verdict.passed:
            return True

        save_judge_feedback(session_dir, "code_review", attempt, judge_verdict)
        failed_items = [i for i in judge_verdict.items if i.get("verdict") == "fail"]
        log_info("code_review", f"judge: {len(failed_items)} findings still unresolved")

    log_fail("code_review", "fix+judge loop exhausted")
    return False


def _load_review_findings(session_dir: str) -> list[dict]:
    checklist_path = Path(session_dir) / "code_review-checklist.json"
    if checklist_path.is_file():
        try:
            data = json.loads(checklist_path.read_text())
            return [
                {"id": item["id"], "criteria": item["criteria"]}
                for item in data.get("checklist", [])
            ]
        except (json.JSONDecodeError, KeyError):
            pass
    return []


# ── Dev server ─────────────────────────────────────────────────────────────

def ensure_dev_server(cwd: str) -> bool:
    """Ensure dev server is running. Reads config from preset if available."""
    info = pipeline_ops.allocate_dev_server()
    if not isinstance(info, dict):
        log_fail("visual_test", "dev-server allocate failed")
        return False

    port = info["port"]
    if info.get("running"):
        return True

    # Read dev server config from preset
    dev_cmd = None
    health_url = None
    startup_timeout = 180
    try:
        state = pipeline_ops.require_state()
        preset = pipeline_ops.load_preset_for_state(state)
        dev_config = getattr(preset, "dev_server", None)
        if dev_config:
            repo = cwd
            if is_bazel_repo(cwd) and dev_config.get("bazel_command"):
                dev_cmd = dev_config["bazel_command"].replace("${PORT}", str(port)).replace("${REPO_ROOT}", repo)
            else:
                dev_cmd = dev_config.get("command", "").replace("${PORT}", str(port)).replace("${REPO_ROOT}", repo)
            health_url = dev_config.get("health_url", "").replace("${PORT}", str(port))
            startup_timeout = dev_config.get("startup_timeout", 180)
    except Exception:
        pass

    if not dev_cmd:
        log_fail("visual_test", "no dev_server.command in preset manifest")
        return False

    # VPN connectivity check — fail fast if corporate network is unreachable
    vpn_host = dev_config.get("vpn_check_host") if dev_config else None
    if vpn_host:
        import socket
        try:
            sock = socket.create_connection((vpn_host, 443), timeout=5)
            sock.close()
        except OSError:
            log_fail("visual_test", f"VPN not connected — cannot reach {vpn_host}")
            return False

    # Verify bazel target exists; if not, search subpackages for the target name
    if is_bazel_repo(cwd):
        import re as _re
        m = _re.search(r'(//[\w/.-]+):([\w.-]+)', dev_cmd)
        if m:
            target = f"{m.group(1)}:{m.group(2)}"
            check = subprocess.run(
                ["bazel", "query", target], capture_output=True, text=True, cwd=cwd, timeout=30,
            )
            if check.returncode != 0:
                pkg_base = m.group(1).lstrip("/")
                fallback = subprocess.run(
                    ["bazel", "query", f'filter("{m.group(2)}$", //{pkg_base}/...)'],
                    capture_output=True, text=True, cwd=cwd, timeout=30,
                )
                alt = fallback.stdout.strip().splitlines()
                if alt:
                    dev_cmd = dev_cmd.replace(target, alt[0])
                    log_info("visual_test", f"target {target} not found, using {alt[0]}")

    log_info("visual_test", f"starting dev server on port {port}")
    proc = subprocess.Popen(
        dev_cmd, shell=True, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )

    global _dev_server_proc
    _dev_server_proc = proc

    if _session_dir:
        try:
            sm = StateManager(Path(_session_dir) / "agent-state.json")
            if sm.exists():
                sm.update(lambda s: setattr(s, "dev_server_pid", proc.pid))
        except Exception:
            pass

    if not health_url:
        health_url = f"http://localhost:{port}/"

    polls = startup_timeout // 5
    for i in range(polls):
        if proc.poll() is not None:
            output = proc.stdout.read().decode(errors="replace")[-500:]
            log_fail("visual_test", f"dev server crashed (exit {proc.returncode}): {output}")
            return False
        try:
            result = subprocess.run(
                ["curl", "-sk", health_url], capture_output=True, timeout=5,
            )
            if result.returncode == 0:
                return True
        except subprocess.TimeoutExpired:
            pass
        time.sleep(5)

    log_fail("visual_test", f"dev server failed to start on port {port}")
    return False


# ── Pre-PR gate ────────────────────────────────────────────────────────────

def run_pre_pr_gate() -> bool:
    from forge.executor.pre_pr_gate import run_gate
    try:
        return run_gate()
    except Exception:
        return False


# ── Main dispatch loop ─────────────────────────────────────────────────────

def dispatch_loop(cwd: str, session_dir: str, pipeline_start: float) -> bool:
    dev_server_failures = 0

    # Set env vars for Claude Code hooks
    _set_plugin_dir()
    os.environ["FORGE_SESSION_DIR"] = session_dir
    try:
        init_state = pipeline_ops.require_state()
        init_preset = pipeline_ops.load_preset_for_state(init_state)
        _set_hook_build_cmd(init_state, init_preset)
        _set_hook_eslint_config(init_preset)
    except Exception as e:
        log_activity(_activity_log_path, "driver", f"warning: failed to setup hooks: {e}")

    while True:
        if time.time() - pipeline_start > PIPELINE_TIMEOUT_S:
            log_fail("driver", f"pipeline timeout ({PIPELINE_TIMEOUT_S}s)")
            return False

        try:
            st = pipeline_ops.require_state()
            if st.killed:
                log_fail("driver", f"pipeline killed: {st.kill_reason or 'no reason given'}")
                return False
        except Exception:
            pass

        state = pipeline_ops.require_state()
        preset = pipeline_ops.load_preset_for_state(state)
        status = pipeline_ops.get_next_steps(state, preset)

        if isinstance(status, dict) and "step" in status and status["step"] is None:
            if status.get("blocked"):
                log_fail("driver", f"pipeline blocked: {', '.join(status['blocked'])} waiting on permanently failed dependencies")
                return False
            return True

        runnable = status.get("runnable", [])
        in_progress = status.get("in_progress", [])
        blocked = status.get("blocked", [])

        if not runnable and not in_progress:
            if blocked:
                log_fail("driver", f"deadlocked: {', '.join(blocked)} blocked")
                return False
            return True

        if not runnable:
            time.sleep(5)
            continue

        for si in runnable:
            if si.get("has_failed_deps"):
                log_info(si["step"], "running despite failed upstream deps (permanently failed)")

        dev_failure_counter = [dev_server_failures]
        if len(runnable) == 1:
            step_info = runnable[0]
            name = step_info["step"]

            if name == "create_pr":
                if not run_pre_pr_gate():
                    log_fail("create_pr", "pre-PR gate failed")
                    pipeline_ops.mark_failed("create_pr", "Pre-PR gate failed")
                    continue

            _dispatch_one_step(step_info, cwd, session_dir, dev_server_failure_counter=dev_failure_counter)
        else:
            filtered = []
            for si in runnable:
                name = si["step"]
                if name == "create_pr" and not run_pre_pr_gate():
                    log_fail("create_pr", "pre-PR gate failed")
                    pipeline_ops.mark_failed("create_pr", "Pre-PR gate failed")
                    continue
                filtered.append(si)

            if filtered:
                with ThreadPoolExecutor(max_workers=len(filtered)) as pool:
                    futures = {
                        pool.submit(
                            _dispatch_one_step, si, cwd, session_dir,
                            dev_server_failure_counter=dev_failure_counter,
                        ): si["step"]
                        for si in filtered
                    }
                    for future in as_completed(futures):
                        step_name = futures[future]
                        try:
                            _, success = future.result()
                        except Exception as exc:
                            log_fail(step_name, f"exception: {exc}")

        dev_server_failures = dev_failure_counter[0]

        _write_status(session_dir)

        # Check for revalidation
        post_state = pipeline_ops.require_state()
        post_preset = pipeline_ops.load_preset_for_state(post_state)
        post_status = pipeline_ops.get_next_steps(post_state, post_preset)
        if isinstance(post_status, dict):
            new_runnable = post_status.get("runnable", [])
            prev_names = {r["step"] for r in runnable}
            revalidated = [s["step"] for s in new_runnable if s["step"] not in prev_names]
            if revalidated:
                log_revalidation(revalidated)
                _write_status(session_dir)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="forge-driver",
        description="Deterministic pipeline driver — plan in, PR out.",
    )
    parser.add_argument("--plan", help="Path to plan file (markdown)")
    parser.add_argument("--plan-dir", help="Path to directory containing plan artifacts")
    parser.add_argument("--packages", nargs="+", default=[],
                        help="Hint packages (optional; auto-detected from git after code step)")
    parser.add_argument("--pipeline", default="full", help="Pipeline type (default: full)")
    parser.add_argument("--preset", default="", help="Preset name (e.g., npm-ts, python-uv)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing pipeline session")
    parser.add_argument("--skip", nargs="+", default=[], metavar="STEP",
                        help="Skip these steps on resume (mark complete, no evidence)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without executing")
    parser.add_argument("--fast", action="store_true", help="Use haiku for all AI steps (for pipeline testing)")
    parser.add_argument("--no-worktree", action="store_true", help="Allow running outside a git worktree")

    args = parser.parse_args()

    global _fast_mode
    if args.fast:
        _fast_mode = True

    if not args.resume and not args.preset:
        parser.error("--preset is required when starting a new pipeline")

    if args.resume:
        try:
            state = pipeline_ops.require_state()
            status = pipeline_ops.get_summary(state, as_json=True)
        except RuntimeError as e:
            print(f"ERROR: Cannot resume — {e}", file=sys.stderr)
            sys.exit(1)

        # Reset orphaned IN_PROGRESS steps (left behind after SIGTERM)
        for name, ss in state.steps.items():
            if ss.status == StepStatus.IN_PROGRESS:
                pipeline_ops.reset_step(name, no_retry_inc=True)
                _safe_print(f"  Reset orphaned step: {name}")
        state = pipeline_ops.require_state()
        status = pipeline_ops.get_summary(state, as_json=True)

        if args.skip:
            for step_name in args.skip:
                try:
                    pipeline_ops.skip_step(step_name)
                    _safe_print(f"Skipped: {step_name}")
                except (ValueError, RuntimeError) as e:
                    print(f"WARNING: Could not skip {step_name}: {e}", file=sys.stderr)
            state = pipeline_ops.require_state()
            status = pipeline_ops.get_summary(state, as_json=True)

        session_dir = status["session_dir"]
        cwd = state.worktree_path or os.getcwd()
        fast_label = " | FAST (haiku)" if _fast_mode else ""
        log_headline(f"Pipeline: {status['pipeline']} | Resuming{fast_label}")
        log_headline(f"Session: {session_dir}")

    else:
        if not args.plan and not args.plan_dir:
            parser.error("Provide --plan <file> or --plan-dir <directory> (or --resume)")

        plan_file = discover_plan(args.plan, args.plan_dir)
        plan_path = Path(plan_file)
        word_count = len(plan_path.read_text().split())

        fast_label = " | FAST (haiku)" if _fast_mode else ""
        log_headline(f"Pipeline: {args.pipeline} | Preset: {args.preset}{fast_label}")
        _safe_print(f"Plan: {plan_path.name} ({word_count} words)")
        if args.packages:
            _safe_print(f"Packages: {', '.join(args.packages)}")
        _safe_print()

        if args.dry_run:
            _safe_print(f"Plan file: {plan_file}")
            _safe_print(f"Would run: forge execute init {args.pipeline} --preset {args.preset} --plan {plan_file}")
            _safe_print(f"Packages: {args.packages}")
            return

        # Initialize pipeline using commands module
        from forge.executor.commands import cmd_init
        import types
        slug = _read_planner_slug(args.plan_dir) if args.plan_dir else ""
        if not slug:
            slug = generate_slug(plan_path.read_text()[:500], fallback="")
        init_args = types.SimpleNamespace(
            pipeline=args.pipeline, preset=args.preset, plan=plan_file,
            no_worktree=args.no_worktree, packages=args.packages,
            slug=slug,
        )
        cmd_init(init_args)

        state = pipeline_ops.require_state()
        status = pipeline_ops.get_summary(state, as_json=True)
        session_dir = status["session_dir"]
        cwd = os.getcwd()
        emit(session_dir, EventType.PIPELINE_STARTED, "executor",
             pipeline=status.get("pipeline", ""), preset=status.get("preset", ""))

        output_file = os.path.join(session_dir, "pipeline-output.md")
        activity_log = os.path.join(session_dir, "pipeline-activity.log")
        with open(output_file, "w") as f:
            f.write(f"# Pipeline Output\n\nStarted: {datetime.now(timezone.utc).isoformat()}\n")
            f.write(f"Plan: {plan_path.name}\n")
        with open(activity_log, "w") as f:
            f.write(f"[{_ts()}] Pipeline initialized — driver mode\n")

    def _is_pid_alive(pid: int) -> bool:
        """Check if a process with the given PID is still running."""
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def _set_driver_pid():
        try:
            state = pipeline_ops.state_mgr().load() if pipeline_ops.state_mgr().exists() else None
            if state and state.driver_pid > 0 and state.driver_pid != os.getpid():
                if _is_pid_alive(state.driver_pid):
                    raise RuntimeError(
                        f"Another driver is already running (PID {state.driver_pid}). "
                        f"Kill it first or use 'forge execute kill'."
                    )
            def mutate(s: PipelineState):
                s.driver_pid = os.getpid()
                s.killed = False
                s.kill_reason = ""
            pipeline_ops.state_mgr().update(mutate)
        except RuntimeError:
            raise
        except Exception:
            pass

    def _shutdown(reason: str):
        global _shutting_down
        if _shutting_down:
            return
        _shutting_down = True

        _stop_status_updater()

        try:
            def mutate(s: PipelineState):
                s.killed = True
                s.kill_reason = reason
                s.driver_pid = 0
                s.dev_server_pid = 0
            pipeline_ops.state_mgr().update(mutate)
            emit(session_dir, EventType.PIPELINE_KILLED, "executor", error=reason)
        except Exception:
            pass

        if _dev_server_proc and _dev_server_proc.poll() is None:
            try:
                _dev_server_proc.terminate()
                _dev_server_proc.wait(timeout=5)
            except Exception:
                try:
                    _dev_server_proc.kill()
                except Exception:
                    pass

        _kill_child_processes()
        _write_status(session_dir)

    def _clear_driver_pid():
        try:
            def mutate(s: PipelineState):
                s.driver_pid = 0
                s.dev_server_pid = 0
            pipeline_ops.state_mgr().update(mutate)
        except Exception:
            pass

    def _handle_signal(sig, frame):
        reason = "SIGTERM" if sig == signal.SIGTERM else "SIGINT"
        _shutdown(reason)
        _safe_print(f"\n{YELLOW}{reason}. Pipeline state preserved in session directory.{RESET}")
        _safe_print(f"Resume with: forge execute --resume")
        sys.exit(130 if sig == signal.SIGINT else 143)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    global _activity_log_path, _session_dir
    _session_dir = session_dir
    _activity_log_path = os.path.join(session_dir, "pipeline-activity.log")

    _set_driver_pid()
    _write_status(session_dir)

    # Set eslint config from preset before preflight checks
    try:
        _init_preset = pipeline_ops.load_preset_for_state(pipeline_ops.require_state())
        _set_hook_eslint_config(_init_preset)
    except Exception:
        pass

    hook_issues = _preflight_hooks()
    if hook_issues:
        for issue in hook_issues:
            log_activity(_activity_log_path, "hooks", issue)
    _start_status_updater(session_dir)

    pipeline_start = time.time()
    success = False
    try:
        success = dispatch_loop(cwd, session_dir, pipeline_start)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BrokenPipeError:
        # Stdout pipe closed (parent process gone) — not a pipeline failure.
        # Clean up without marking pipeline as killed so resume works cleanly.
        _stop_status_updater()
        _clear_driver_pid()
    except Exception as exc:
        _shutdown(f"exception: {type(exc).__name__}: {exc}")
        raise
    finally:
        elapsed = time.time() - pipeline_start
        if not _shutting_down:
            _stop_status_updater()
            _clear_driver_pid()
            if _dev_server_proc and _dev_server_proc.poll() is None:
                _dev_server_proc.terminate()
                try:
                    _dev_server_proc.wait(timeout=5)
                except Exception:
                    _dev_server_proc.kill()

    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    _safe_print()
    if success:
        emit(session_dir, EventType.PIPELINE_COMPLETED, "executor")
        log_headline(f"{GREEN}✓ Pipeline complete in {minutes}m {seconds}s{RESET}")
    else:
        if not _shutting_down:
            emit(session_dir, EventType.PIPELINE_FAILED, "executor")
        log_headline(f"{RED}✗ Pipeline stopped after {minutes}m {seconds}s{RESET}")
        _safe_print(f"Resume with: forge execute --resume")

    _safe_print(f"Session: {session_dir}")
    _write_status(session_dir)

    if not success:
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
