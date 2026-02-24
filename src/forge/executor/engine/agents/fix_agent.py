
"""Fix step wrapper — generic fixer for command step failures (lint, test, build)."""

import concurrent.futures
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from forge.core.runner import AgentRunner
from ..runner import build_context, _error_reference, _select_command, PIPELINE_CLI
from ..registry import StepDefinition, Preset
from ..state import PipelineState
from ..templates import render


@dataclass
class FixAgentOutcome:
    passed: bool
    reason: str


def generate_prompt(
    step: StepDefinition,
    state: PipelineState,
    preset: Preset,
    failed_packages: list[str] | None = None,
) -> str:
    """Generate a focused fix prompt — error references + verify command, no protocol."""
    ctx = build_context(state, preset)

    if step.per_package and failed_packages:
        parts = ["## Task\n\nFix the errors and verify each fix.\n"]
        for pkg in failed_packages:
            pkg_ctx = {**ctx, "PACKAGE": pkg, "PACKAGE_SLUG": pkg.replace("/", "-")}
            verify_cmd = render(_select_command(step), pkg_ctx)
            parts.append(f"### Package: `{pkg}`\n")
            parts.append(f"Verify: `{verify_cmd}`\n")
            if step.error_file:
                error_path = Path(state.session_dir) / render(step.error_file, pkg_ctx)
                if error_path.is_file():
                    parts.append(_error_reference(error_path))
        if step.fix_hints:
            parts.append(f"\n**Hint:** {step.fix_hints}")
        return "\n".join(parts)

    verify_cmd = render(_select_command(step), ctx)
    parts = [f"## Task\n\nFix the errors. Verify by running: `{verify_cmd}`\n"]
    if step.fix_hints:
        parts.append(f"**Hint:** {step.fix_hints}")
    if step.error_file:
        error_path = Path(state.session_dir) / render(step.error_file, ctx)
        if error_path.is_file():
            parts.append(_error_reference(error_path))
    return "\n".join(parts)


MAX_PARALLEL_REVERIFY = 4  # Match runner.py cap; Bazel serializes through one server


def _re_execute_command(step: StepDefinition, state: PipelineState, preset: Preset,
                        cwd: str, failed_packages: list[str] | None = None) -> tuple[bool, str]:
    """Re-run the verify command after the fix agent. Returns (passed, output)."""
    ctx = build_context(state, preset)

    if step.per_package and failed_packages:
        def _verify_one(pkg: str) -> tuple[str, bool, str]:
            pkg_ctx = {**ctx, "PACKAGE": pkg, "PACKAGE_SLUG": pkg.replace("/", "-")}
            cmd = render(_select_command(step), pkg_ctx)
            ok, output = _run_command(cmd, cwd, step.timeout or 600)
            if not ok and step.error_file:
                error_path = Path(state.session_dir) / render(step.error_file, pkg_ctx)
                error_path.parent.mkdir(parents=True, exist_ok=True)
                error_path.write_text(output)
            return pkg, ok, output

        all_passed = True
        combined_output = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(failed_packages), MAX_PARALLEL_REVERIFY),
        ) as pool:
            futures = {pool.submit(_verify_one, pkg): pkg for pkg in failed_packages}
            for future in concurrent.futures.as_completed(futures):
                pkg, ok, output = future.result()
                combined_output.append(f"=== {pkg} ===\n{output}")
                if not ok:
                    all_passed = False
        return all_passed, "\n".join(combined_output)

    cmd = render(_select_command(step), ctx)
    ok, output = _run_command(cmd, cwd, step.timeout or 600)
    if not ok and step.error_file:
        error_path = Path(state.session_dir) / render(step.error_file, ctx)
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(output)
    return ok, output


_SHELL_OPERATORS_RE = __import__("re").compile(r'[|&;<>(){}\$`!]')
_SHELL_BUILTINS = {"cd", "exit", "export", "source", ".", "eval", "exec", "set", "unset", "alias"}


def _needs_shell(cmd: str) -> bool:
    if _SHELL_OPERATORS_RE.search(cmd):
        return True
    first_word = cmd.split()[0] if cmd.strip() else ""
    return first_word in _SHELL_BUILTINS


def _run_command(cmd: str, cwd: str, timeout: int) -> tuple[bool, str]:
    """Run a shell command. Returns (passed, combined_output)."""
    use_shell = _needs_shell(cmd)
    try:
        result = subprocess.run(
            cmd if use_shell else shlex.split(cmd),
            shell=use_shell, capture_output=True, timeout=timeout, cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        return False, f"Command timed out after {timeout}s"

    combined = result.stdout.decode(errors="replace") + result.stderr.decode(errors="replace")
    return result.returncode == 0, combined


def run(
    step: StepDefinition,
    state: PipelineState,
    preset: Preset,
    cwd: str,
    session_dir: str,
    activity_log_path: str,
    failed_packages: list[str] | None = None,
    model: str = "sonnet",
    max_turns: int = 0,
    timeout_s: int = 3600,
    agent_command: str | None = None,
    attempt: int = 0,
) -> FixAgentOutcome:
    """Run the fix step: generate prompt, spawn agent, re-verify, report."""
    prompt = generate_prompt(step, state, preset, failed_packages)

    transcript_name = f"{step.name}_attempt{attempt}" if attempt > 0 else step.name
    runner = AgentRunner(
        session_dir=session_dir,
        step_name=transcript_name,
        activity_log_path=activity_log_path,
        agent_command=agent_command,
    )

    runner.run(
        prompt=prompt,
        model=model,
        max_turns=max_turns,
        cwd=cwd,
        timeout_s=timeout_s,
    )

    # Re-run the verify command deterministically
    verify_ok, verify_output = _re_execute_command(step, state, preset, cwd, failed_packages)

    if verify_ok:
        _report_pass(step.name)
        return FixAgentOutcome(passed=True, reason="Fix verified")

    _report_fail(step.name, "Verify command still fails after fix attempt")
    return FixAgentOutcome(passed=False, reason="Verify command still fails after fix attempt")


def _report_pass(step_name: str) -> None:
    subprocess.run(
        [sys.executable, PIPELINE_CLI, "pass", step_name],
        capture_output=True, text=True,
    )


def _report_fail(step_name: str, reason: str) -> None:
    subprocess.run(
        [sys.executable, PIPELINE_CLI, "fail", step_name, reason],
        capture_output=True, text=True,
    )
