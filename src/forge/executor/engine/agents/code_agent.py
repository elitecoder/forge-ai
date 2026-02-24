
"""Code step wrapper — generates focused prompt, spawns agent, verifies outcome."""

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from forge.core.runner import AgentRunner, AgentResult
from ..runner import _judge_feedback_section, build_context, PIPELINE_CLI
from ..state import PipelineState, StepState
from ..registry import Preset
from ..templates import render
from ..utils import is_bazel_repo


@dataclass
class CodeAgentOutcome:
    passed: bool
    reason: str
    checklist: dict | None = None
    detected_packages: list[str] | None = None


def generate_prompt(state: PipelineState, preset: Preset) -> str:
    """Generate a focused code prompt — task only, no pipeline protocol."""
    ctx = build_context(state, preset)

    plan_path = Path(state.plan_file).expanduser() if state.plan_file else None
    if not plan_path or not plan_path.is_file():
        raise FileNotFoundError(
            f"Plan file not found: {state.plan_file or '(none)'}. "
            f"Use --plan when initializing the pipeline."
        )

    plan_content = plan_path.read_text()

    # Companion files
    plan_dir = plan_path.parent
    brief_section = ""
    codebase_brief = plan_dir / "codebase-brief.md"
    if codebase_brief.is_file():
        brief_section = f"\n## Codebase Brief\n\n{codebase_brief.read_text()}\n"

    # Copy visual test plan to session dir if it exists
    for vtp in plan_dir.glob("*visual-test-plan*"):
        dest = Path(state.session_dir) / vtp.name
        if not dest.exists():
            dest.write_text(vtp.read_text())

    # Build error context from previous retry
    build_error_section = ""
    code_retries = state.steps.get("code", StepState()).retries
    if code_retries > 0 and state.session_dir:
        build_errors_path = Path(state.session_dir) / "build-errors.txt"
        if build_errors_path.is_file():
            try:
                raw = build_errors_path.read_text()
                # Strip ANSI codes for cleaner prompt
                clean = re.sub(r'\033\[[0-9;]*m', '', raw)
                # Extract just the TypeScript errors (most useful part)
                ts_errors = [l for l in clean.splitlines() if "error TS" in l]
                if ts_errors:
                    build_error_section = (
                        "\n## Previous Build Errors (MUST FIX)\n\n"
                        f"The previous attempt (retry {code_retries}) failed with these build errors. "
                        "Fix these errors as your top priority:\n\n```\n"
                        + "\n".join(ts_errors[:20]) + "\n```\n"
                    )
            except OSError:
                pass

    # Judge feedback from previous attempt
    feedback_section = ""
    judge_section = _judge_feedback_section(state.session_dir, "code")
    if judge_section:
        checklist_path = Path(state.session_dir) / "code-checklist.json"
        existing_files_note = ""
        if checklist_path.is_file():
            try:
                cl = json.loads(checklist_path.read_text())
                files = cl.get("files_created", []) + cl.get("files_modified", [])
                if files:
                    existing_files_note = (
                        "\n## Existing Implementation\n\n"
                        "A previous agent already wrote code for this plan. "
                        "The files below already exist on disk. Read them and make "
                        "targeted fixes based on the judge feedback — do NOT rewrite from scratch.\n\n"
                        + "\n".join(f"- `{f}`" for f in files) + "\n"
                    )
            except (json.JSONDecodeError, OSError):
                pass
        feedback_section = existing_files_note + judge_section

    if brief_section:
        explore_instruction = (
            "Use the Codebase Brief below as your primary reference for file locations, "
            "existing patterns, and code structure. Only read specific files when you need "
            "exact implementation details not covered in the brief."
        )
    else:
        explore_instruction = (
            "Explore the codebase to understand existing patterns, then make the changes."
        )

    # Build verification command from preset
    cmd_template = preset.build_command
    if preset.bazel_build_command and is_bazel_repo():
        cmd_template = preset.bazel_build_command

    if cmd_template:
        packages = [p.strip() for p in ctx["AFFECTED_PACKAGES"].split(",") if p.strip()]
        if packages and packages != ["(none)"]:
            ctx["BUILD_TARGETS"] = " ".join(f"//{p}/..." for p in packages)
        elif "{{BUILD_TARGETS}}" in cmd_template:
            build_cmd = f"cd {ctx['REPO_ROOT']} && echo 'No build targets — run build manually after identifying affected packages'"
            cmd_template = ""  # skip render
        if cmd_template:
            build_cmd = render(cmd_template, ctx)
    else:
        build_cmd = f"cd {ctx['REPO_ROOT']} && echo 'No build verification configured'"

    return f"""## Task

Implement the code changes described in the plan below. Read the plan carefully.
{explore_instruction}

- Repository: `{ctx['REPO_ROOT']}`
- Affected packages: {ctx['AFFECTED_PACKAGES']}

**Scope:** Only create or modify files called for in the plan. Do NOT create E2E tests, \
Playwright tests, or integration test files unless the plan explicitly requests them.
{brief_section}{build_error_section}{feedback_section}
## Plan

{plan_content}

## After Implementing

1. **Verify mathematical invariants**: If your code computes ratios, durations, or derived values, prove the formula is correct with concrete examples. For each formula, substitute sample values and verify the output matches the expected direction (e.g., "speed up" produces ratio > 1, "slow down" produces ratio < 1). Document this verification as a code comment if the formula is non-obvious.
2. Run: `{build_cmd}`
3. Fix any compilation errors and repeat until the build succeeds.
"""


def _has_code_changes(cwd: str) -> bool:
    """Check if there are tracked or untracked code changes."""
    tracked = subprocess.run(
        ["git", "diff", "--stat", "HEAD"],
        capture_output=True, text=True, cwd=cwd,
    )
    if tracked.stdout.strip():
        return True

    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        capture_output=True, text=True, cwd=cwd,
    )
    return bool(untracked.stdout.strip())


def _detect_packages(cwd: str) -> list[str]:
    """Detect Bazel packages from uncommitted changes in the worktree.

    Only considers uncommitted modifications, staged files, and untracked files
    (i.e. what the code agent actually produced). Does NOT diff against
    merge-base, which would pick up all historical branch changes and
    inflate the package list dramatically.
    """
    files: set[str] = set()

    # Uncommitted: modified, staged, and untracked
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=cwd,
    )
    for line in status.stdout.splitlines():
        # porcelain format: XY filename  (or XY orig -> renamed)
        parts = line[3:].split(" -> ")
        files.add(parts[-1].strip())

    # Walk up to nearest BUILD.bazel for each file
    root = Path(cwd).resolve()
    packages: set[str] = set()
    for f in files:
        if not f:
            continue
        p = root / f
        d = p.parent if p.is_file() else p
        while d != root and d != d.parent:
            if (d / "BUILD.bazel").exists() or (d / "BUILD").exists():
                packages.add(str(d.relative_to(root)))
                break
            d = d.parent

    return sorted(packages)



def _verify_build(state: PipelineState, preset: Preset, cwd: str, session_dir: str) -> dict:
    """Verify the build succeeds. Returns {"passed": bool, "reason": str}."""
    # Build the build command (same logic as generate_prompt)
    ctx = build_context(state, preset)
    cmd_template = preset.build_command
    if preset.bazel_build_command and is_bazel_repo():
        cmd_template = preset.bazel_build_command

    if not cmd_template:
        # No build command configured, skip verification
        return {"passed": True, "reason": "No build verification configured"}

    packages = [p.strip() for p in ctx["AFFECTED_PACKAGES"].split(",") if p.strip()]
    if packages and packages != ["(none)"]:
        ctx["BUILD_TARGETS"] = " ".join(f"//{p}/..." for p in packages)
    elif "{{BUILD_TARGETS}}" in cmd_template:
        # State doesn't have packages yet — detect from working tree
        detected = _detect_packages(cwd)
        if detected:
            ctx["BUILD_TARGETS"] = " ".join(f"//{p}/..." for p in detected)
        elif _has_code_changes(cwd):
            return {"passed": False, "reason": "Code changes exist but no build targets could be determined"}
        else:
            return {"passed": True, "reason": "No code changes to verify"}

    build_cmd = render(cmd_template, ctx)

    # Run the build command and capture output
    try:
        result = subprocess.run(
            build_cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=300,  # 5 minute timeout for build
        )
    except subprocess.TimeoutExpired:
        build_errors_path = Path(session_dir) / "build-errors.txt"
        build_errors_path.write_text("Build timed out after 5 minutes")
        return {"passed": False, "reason": "Build timed out"}
    except Exception as e:
        build_errors_path = Path(session_dir) / "build-errors.txt"
        build_errors_path.write_text(f"Build execution failed: {str(e)}")
        return {"passed": False, "reason": f"Build execution failed: {str(e)}"}

    # Check if build succeeded
    if result.returncode == 0:
        return {"passed": True, "reason": "Build verification passed"}

    # Build failed — save errors for next retry
    build_output = result.stdout + result.stderr
    build_errors_path = Path(session_dir) / "build-errors.txt"
    build_errors_path.write_text(build_output)

    # Extract error summary for the reason
    error_lines = [l for l in build_output.splitlines() if "error" in l.lower()]
    error_summary = " | ".join(error_lines[:3]) if error_lines else "Build failed with unknown error"
    if len(error_summary) > 200:
        error_summary = error_summary[:200] + "..."

    return {"passed": False, "reason": f"Build failed: {error_summary}"}


def _generate_checklist(cwd: str) -> dict:
    """Generate a checklist from git diff summary."""
    diff_stat = subprocess.run(
        ["git", "diff", "--stat", "HEAD"],
        capture_output=True, text=True, cwd=cwd,
    ).stdout.strip()

    diff_files = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        capture_output=True, text=True, cwd=cwd,
    ).stdout.strip().splitlines()

    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        capture_output=True, text=True, cwd=cwd,
    ).stdout.strip().splitlines()

    items = []
    for f in diff_files:
        items.append({
            "id": f"file-{len(items) + 1}",
            "criteria": f"Modified {f}",
            "status": "done",
            "evidence": f"{f}",
            "files_touched": [f],
        })
    for f in untracked:
        items.append({
            "id": f"file-{len(items) + 1}",
            "criteria": f"Created {f}",
            "status": "done",
            "evidence": f"{f} (new file)",
            "files_touched": [f],
        })

    return {
        "step": "code",
        "summary": diff_stat,
        "files_modified": diff_files,
        "files_created": untracked,
        "checklist": items,
    }


def run(
    state: PipelineState,
    preset: Preset,
    cwd: str,
    session_dir: str,
    activity_log_path: str,
    model: str = "sonnet",
    max_turns: int = 0,
    timeout_s: int = 3600,
    agent_command: str | None = None,
) -> CodeAgentOutcome:
    """Run the code step: generate prompt, spawn agent, verify, report."""
    prompt = generate_prompt(state, preset)

    runner = AgentRunner(
        session_dir=session_dir,
        step_name="code",
        activity_log_path=activity_log_path,
        agent_command=agent_command,
    )

    result = runner.run(
        prompt=prompt,
        model=model,
        max_turns=max_turns,
        cwd=cwd,
        timeout_s=timeout_s,
    )

    # Verify outcome deterministically
    if not _has_code_changes(cwd):
        _report_fail(session_dir, "No code changes produced")
        return CodeAgentOutcome(passed=False, reason="No code changes produced")

    # Verify build succeeds before reporting pass
    build_status = _verify_build(state, preset, cwd, session_dir)
    if not build_status["passed"]:
        _report_fail(session_dir, build_status["reason"])
        return CodeAgentOutcome(passed=False, reason=build_status["reason"])

    packages = _detect_packages(cwd)

    checklist = _generate_checklist(cwd)
    checklist_path = Path(session_dir) / "code-checklist.json"
    checklist_path.write_text(json.dumps(checklist, indent=2))

    _report_pass(session_dir)
    return CodeAgentOutcome(passed=True, reason="Code changes verified",
                            checklist=checklist, detected_packages=packages)


def _report_pass(session_dir: str) -> None:
    """Call pipeline_cli.py pass code."""
    subprocess.run(
        [sys.executable, PIPELINE_CLI, "pass", "code"],
        capture_output=True, text=True,
    )


def _report_fail(session_dir: str, reason: str) -> None:
    """Call pipeline_cli.py fail code <reason>."""
    subprocess.run(
        [sys.executable, PIPELINE_CLI, "fail", "code", reason],
        capture_output=True, text=True,
    )
