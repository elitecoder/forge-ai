
"""Step execution: direct command runner, AI prompt builder, fix prompt generator."""

import concurrent.futures
import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .registry import StepDefinition, Preset
from .state import PipelineState
from .templates import render
from .utils import DEFAULT_DEV_PORT, repo_root, validate_package_name, is_bazel_repo

_SHELL_OPERATORS = re.compile(r'[|&;<>(){}\$`!]')
_SHELL_BUILTINS = {"cd", "exit", "export", "source", ".", "eval", "exec", "set", "unset", "alias"}
_BAZEL_TARGET_RE = re.compile(r'//[\w/.-]+:[\w.-]+')


def _needs_shell(cmd: str) -> bool:
    """Return True if command requires shell interpretation."""
    if _SHELL_OPERATORS.search(cmd):
        return True
    first_word = cmd.split()[0] if cmd.strip() else ""
    return first_word in _SHELL_BUILTINS



PIPELINE_PROTOCOL = """## Pipeline Protocol

You are a pipeline agent for step `{step_name}`.
Session directory: `{session_dir}`

### Reporting (do these during your work)
- Log progress: `echo "[$(date +%H:%M)] {step_name}: <msg>" >> {session_dir}/pipeline-activity.log`
- Append results to `{session_dir}/pipeline-output.md` under `## {step_name}`
- Include `**Agent Model:** <your model identity>` in your output section

### Final actions (do these as your LAST two actions before stopping)

**Action 1:** Write `{session_dir}/{step_name}-checklist.json`
**Action 2:** Run exactly one of these bash commands:
```
python3 {pipeline_cli} pass {step_name}
python3 {pipeline_cli} fail {step_name} "reason"
```

**WARNING:** If you do not run the pass/fail bash command, ALL your work is discarded and the step is retried from scratch. The pass/fail command is mandatory — writing the checklist alone is not enough.
"""

PIPELINE_PROTOCOL_REMINDER = """
---
## MANDATORY: Final Step

Your work is NOT saved until you run one of these commands as your LAST action:
```
python3 {pipeline_cli} pass {step_name}
```
Or if something failed:
```
python3 {pipeline_cli} fail {step_name} "reason"
```
If you do not run this command, ALL your work is discarded.
"""

PIPELINE_CLI = str(Path(__file__).resolve().parent.parent / "pipeline_cli.py")


@dataclass
class StepResult:
    passed: bool
    output: str = ""
    error_file: str = ""
    failed_packages: list[str] = field(default_factory=list)
    error_files: dict[str, str] = field(default_factory=dict)  # pkg -> error_file_path


def _changed_files(base_ref: str = "main") -> str:
    try:
        return subprocess.run(
            ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
            capture_output=True, text=True,
        ).stdout.strip() or "(unable to determine)"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "(unable to determine)"


def build_context(state: PipelineState, preset: Preset, package: str = "") -> dict[str, str]:
    """Build template variables dict for command rendering."""
    port = state.dev_server_port or DEFAULT_DEV_PORT
    ctx = {
        "REPO_ROOT": repo_root(),
        "PIPELINE_CLI": PIPELINE_CLI,
        "PRESET_DIR": str(preset.preset_dir),
        "AFFECTED_PACKAGES": ", ".join(state.affected_packages) or "(none)",
        "CHANGED_FILES": _changed_files(preset.base_ref),
        "SESSION_DIR": state.session_dir,
        "DEV_SERVER_PORT": str(port),
        "DEV_SERVER_URL": f"http://localhost:{port}",
    }
    if package:
        validate_package_name(package)
        ctx["PACKAGE"] = package
        ctx["PACKAGE_SLUG"] = package.replace("/", "-").replace("\\", "-")
    return ctx


# -- Command execution -------------------------------------------------------

def _select_command(step: StepDefinition) -> str:
    """Select bazel_run_command when in a Bazel repo, otherwise run_command."""
    if step.bazel_run_command and is_bazel_repo():
        return step.bazel_run_command
    return step.run_command


def execute_command(step: StepDefinition, state: PipelineState, preset: Preset) -> StepResult:
    """Run a command step directly. No AI. Returns StepResult."""
    if step.per_package:
        return _execute_per_package(step, state, preset)

    ctx = build_context(state, preset)
    cmd = render(_select_command(step), ctx)
    use_shell = _needs_shell(cmd)

    try:
        result = subprocess.run(
            cmd if use_shell else shlex.split(cmd),
            shell=use_shell, capture_output=True,
            timeout=step.timeout or 600, cwd=ctx["REPO_ROOT"],
        )
    except subprocess.TimeoutExpired:
        return StepResult(passed=False, output=f"Command timed out after {step.timeout}s")

    combined = result.stdout.decode(errors="replace") + result.stderr.decode(errors="replace")

    if result.returncode == 0:
        return StepResult(passed=True, output=combined)

    if step.error_file:
        error_path = Path(state.session_dir) / render(step.error_file, ctx)
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(combined)
        return StepResult(passed=False, output=combined, error_file=str(error_path))

    return StepResult(passed=False, output=combined)


def _bazel_target_exists(cmd: str, cwd: str) -> bool | None:
    """Check if the bazel target in a rendered command exists. Returns None if undetermined."""
    match = _BAZEL_TARGET_RE.search(cmd)
    if not match:
        return None
    try:
        result = subprocess.run(
            ["bazel", "query", match.group(0)],
            capture_output=True, text=True, cwd=cwd, timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _run_one_package(step: StepDefinition, state: PipelineState, preset: Preset,
                     pkg: str) -> tuple[str, StepResult]:
    """Run command for a single package."""
    ctx = build_context(state, preset, package=pkg)
    cmd = render(_select_command(step), ctx)

    # Skip if bazel target doesn't exist in this package
    if is_bazel_repo() and step.bazel_run_command:
        exists = _bazel_target_exists(cmd, ctx["REPO_ROOT"])
        if exists is False:
            return pkg, StepResult(passed=True, output=f"Skipped: no matching target in {pkg}")

    use_shell = _needs_shell(cmd)

    try:
        result = subprocess.run(
            cmd if use_shell else shlex.split(cmd),
            shell=use_shell, capture_output=True,
            timeout=step.timeout or 600, cwd=ctx["REPO_ROOT"],
        )
    except subprocess.TimeoutExpired:
        return pkg, StepResult(passed=False, output=f"Timed out after {step.timeout}s")

    combined = result.stdout.decode(errors="replace") + result.stderr.decode(errors="replace")

    if result.returncode == 0:
        return pkg, StepResult(passed=True, output=combined)

    error_file_path = ""
    if step.error_file:
        error_path = Path(state.session_dir) / render(step.error_file, ctx)
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(combined)
        error_file_path = str(error_path)

    return pkg, StepResult(passed=False, output=combined, error_file=error_file_path)


MAX_PARALLEL_PACKAGES = 4  # Bazel serializes through one server; more threads just create contention


def _execute_per_package(step: StepDefinition, state: PipelineState, preset: Preset) -> StepResult:
    """Run command for each affected package, optionally in parallel."""
    packages = state.affected_packages
    if not packages:
        return StepResult(passed=False, output="No affected packages")

    pkg_results: dict[str, StepResult] = {}

    if step.parallel and len(packages) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(packages), MAX_PARALLEL_PACKAGES)) as pool:
            futures = {
                pool.submit(_run_one_package, step, state, preset, pkg): pkg
                for pkg in packages
            }
            for future in concurrent.futures.as_completed(futures):
                pkg, result = future.result()
                pkg_results[pkg] = result
    else:
        for pkg in packages:
            _, result = _run_one_package(step, state, preset, pkg)
            pkg_results[pkg] = result

    failed = [pkg for pkg, r in pkg_results.items() if not r.passed]
    error_files = {pkg: r.error_file for pkg, r in pkg_results.items() if r.error_file}

    if not failed:
        return StepResult(passed=True)

    return StepResult(
        passed=False,
        failed_packages=failed,
        error_files=error_files,
    )


# -- Fix prompt generation ---------------------------------------------------

def _error_reference(error_path: Path) -> str:
    """Return a file reference instruction instead of inline content."""
    line_count = error_path.read_text().count("\n")
    return (
        f"**Error log** ({line_count} lines): `{error_path}`\n"
        f"Read this file to understand the errors. Start with the last 50 lines "
        f"(often the most relevant), then read more if needed."
    )


def generate_fix_prompt(step: StepDefinition, state: PipelineState, preset: Preset,
                        failed_packages: list[str] | None = None) -> str:
    """Generate a fix prompt from error file references + verify command."""
    ctx = build_context(state, preset)
    protocol = PIPELINE_PROTOCOL.format(
        step_name=step.name, session_dir=state.session_dir, pipeline_cli=PIPELINE_CLI,
    )

    if step.per_package and failed_packages:
        parts = [protocol, "## Task\n\nFix the errors and verify each fix.\n"]
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
    parts = [protocol, f"## Task\n\nFix the errors. Verify by running: `{verify_cmd}`\n"]
    if step.fix_hints:
        parts.append(f"**Hint:** {step.fix_hints}")
    if step.error_file:
        error_path = Path(state.session_dir) / render(step.error_file, ctx)
        if error_path.is_file():
            parts.append(_error_reference(error_path))
    return "\n".join(parts)


# -- AI prompt generation ----------------------------------------------------

def _generate_code_prompt(state: PipelineState, preset: Preset) -> str:
    """Generate prompt for the code step by injecting plan content."""
    ctx = build_context(state, preset)
    protocol = PIPELINE_PROTOCOL.format(
        step_name="code", session_dir=state.session_dir, pipeline_cli=PIPELINE_CLI,
    )

    plan_path = Path(state.plan_file).expanduser() if state.plan_file else None
    if not plan_path or not plan_path.is_file():
        raise FileNotFoundError(
            f"Plan file not found: {state.plan_file or '(none)'}. "
            f"Use --plan when initializing the pipeline."
        )

    plan_content = plan_path.read_text()

    # Check for companion files in the same directory
    plan_dir = plan_path.parent
    codebase_brief = plan_dir / "codebase-brief.md"
    brief_section = ""
    if codebase_brief.is_file():
        brief_section = f"\n## Codebase Brief\n\n{codebase_brief.read_text()}\n"

    # Copy visual test plan to session dir if it exists
    for vtp in plan_dir.glob("*visual-test-plan*"):
        dest = Path(state.session_dir) / vtp.name
        if not dest.exists():
            dest.write_text(vtp.read_text())

    # Judge feedback: if a previous attempt was rejected, tell the agent what to fix
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

    # Build verification command from affected packages
    packages = [p.strip() for p in ctx['AFFECTED_PACKAGES'].split(',') if p.strip()]
    if packages:
        build_targets = " ".join(f"//{p}/..." for p in packages)
        build_cmd = f"cd {ctx['REPO_ROOT']} && bazel build {build_targets}"
    else:
        build_cmd = f"cd {ctx['REPO_ROOT']} && bazel build :tsc"

    return f"""{protocol}

## Task

Implement the code changes described in the plan below. Read the plan carefully.
{explore_instruction}

- Repository: `{ctx['REPO_ROOT']}`
- Affected packages: {ctx['AFFECTED_PACKAGES']}
- Session directory: `{state.session_dir}`
{brief_section}{feedback_section}
## Plan

{plan_content}

## After Implementing the Plan — Do These Steps Now

You have finished reading the plan. After implementing the changes above:

1. Run: `{build_cmd}`
2. Fix any compilation errors and repeat until the build succeeds.
3. Write `{state.session_dir}/code-checklist.json` with a summary of changes.
4. Run this bash command: `python3 {PIPELINE_CLI} pass code`

Step 4 is mandatory. If you do not run it, all your work is discarded.
"""


def _checklist_schema_section(step_name: str, session_dir: str) -> str:
    """Generate the structured checklist output requirement for AI prompts."""
    return f"""
## Required: Structured Output

Before calling pass/fail, write `{session_dir}/{step_name}-checklist.json`:

```json
{{
  "step": "{step_name}",
  "checklist": [
    {{
      "id": "<matches criteria item>",
      "criteria": "<what was asked>",
      "status": "done|skipped|blocked",
      "evidence": "<file:line references proving completion>",
      "files_touched": ["<paths>"]
    }}
  ]
}}
```

Your checklist MUST have one entry for EACH criteria item. An external judge will verify your claims against the actual git diff. Do not claim work you did not do.
"""


def _judge_feedback_section(session_dir: str, step_name: str) -> str:
    """Load latest judge feedback if available."""
    judge_dir = Path(session_dir) / "_judge"
    if not judge_dir.is_dir():
        return ""

    feedback_files = sorted(judge_dir.glob(f"{step_name}_attempt_*.json"))
    if not feedback_files:
        return ""

    try:
        latest = json.loads(feedback_files[-1].read_text())
    except (json.JSONDecodeError, OSError):
        return ""

    failed_items = [i for i in latest.get("items", []) if i.get("verdict") == "fail"]
    if not failed_items:
        return ""

    lines = ["\n## Judge Feedback (previous attempt failed)\n"]
    for item in failed_items:
        lines.append(f"- **{item.get('id', '?')}**: {item.get('reason', 'no reason')}")
    lines.append("\nFix these specific items. The judge will re-verify.\n")
    return "\n".join(lines)


def _load_claude_md_for_review() -> str:
    """Load project or global CLAUDE.md for review context. Filter to relevant sections."""
    repo = repo_root()
    candidates = [
        Path(repo) / ".claude" / "CLAUDE.md",
        Path.home() / ".claude" / "CLAUDE.md",
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                text = candidate.read_text()
            except OSError:
                continue
            # Filter to code-relevant sections
            relevant_headers = {"rules", "conventions", "copyright", "code", "style", "testing", "imports"}
            sections = []
            current_section = []
            include = False
            for line in text.splitlines():
                if line.startswith("## "):
                    if include and current_section:
                        sections.extend(current_section)
                    current_section = [line]
                    header_lower = line.lstrip("#").strip().lower()
                    include = any(kw in header_lower for kw in relevant_headers)
                else:
                    current_section.append(line)
            if include and current_section:
                sections.extend(current_section)
            if sections:
                return "\n## Project Conventions (from CLAUDE.md)\n\n" + "\n".join(sections) + "\n"
    return ""


def generate_ai_prompt(step: StepDefinition, state: PipelineState, preset: Preset) -> str:
    """Generate prompt for AI steps by injecting skill content directly."""
    if step.name == "code":
        return _generate_code_prompt(state, preset)

    ctx = build_context(state, preset)
    protocol = PIPELINE_PROTOCOL.format(
        step_name=step.name, session_dir=state.session_dir, pipeline_cli=PIPELINE_CLI,
    )

    skill_raw = step.skill.replace("${PRESET_DIR}", str(preset.preset_dir))
    skill_path = Path(skill_raw).expanduser()
    if not skill_path.is_file():
        raise FileNotFoundError(f"Skill file not found: {skill_path}")

    skill_content = skill_path.read_text()

    context_section = f"""## Context

- Repository: `{ctx['REPO_ROOT']}`
- Affected packages: {ctx['AFFECTED_PACKAGES']}
- Session directory: `{state.session_dir}`
- Dev server port: {ctx['DEV_SERVER_PORT']}
- Dev server URL: `{ctx['DEV_SERVER_URL']}`
- Changed files:
{ctx['CHANGED_FILES']}"""

    # Structured checklist requirement
    checklist_section = _checklist_schema_section(step.name, state.session_dir)

    # Judge feedback from previous attempt
    feedback_section = _judge_feedback_section(state.session_dir, step.name)

    # Code review verdict instruction
    verdict_section = ""
    if step.name == "code_review" and step.two_phase:
        verdict_section = f"""

## Verdict Output

If code is CLEAN:
  echo '{{"verdict": "CLEAN"}}' > {state.session_dir}/code-review-verdict.json
  python3 {PIPELINE_CLI} pass {step.name}

If issues were found:
  echo '{{"verdict": "HAS_ISSUES", "issue_count": N}}' > {state.session_dir}/code-review-verdict.json
  python3 {PIPELINE_CLI} fail {step.name} "HAS_ISSUES: N issues found"

Your review findings in the checklist JSON will be used as criteria for the fix agent's judge."""

    # CLAUDE.md conventions for code review (Fix 9)
    conventions_section = ""
    if step.name == "code_review":
        conventions_section = _load_claude_md_for_review()

    reminder = PIPELINE_PROTOCOL_REMINDER.format(
        pipeline_cli=PIPELINE_CLI, step_name=step.name,
    )
    parts = [protocol, context_section, checklist_section, feedback_section,
             skill_content, conventions_section, verdict_section, reminder]
    return "\n".join(p for p in parts if p)


def generate_ai_fix_prompt(step: StepDefinition, state: PipelineState, preset: Preset) -> str:
    """Generate fix prompt for AI steps (e.g., code_review phase 2)."""
    ctx = build_context(state, preset)
    protocol = PIPELINE_PROTOCOL.format(
        step_name=step.name, session_dir=state.session_dir, pipeline_cli=PIPELINE_CLI,
    )

    parts = [protocol]
    parts.append("## Task\n")
    parts.append(f"Read the findings from the previous review phase in "
                 f"`{state.session_dir}/pipeline-output.md` and fix ALL issues found.")
    parts.append(f"\nAfter fixing, verify: build and tests still pass.")
    parts.append(f"\n## Context\n")
    parts.append(f"- Repository: `{ctx['REPO_ROOT']}`")
    parts.append(f"- Session directory: `{state.session_dir}`")

    # Structured checklist requirement
    parts.append(_checklist_schema_section(step.name, state.session_dir))

    # Judge feedback from previous attempt
    feedback = _judge_feedback_section(state.session_dir, step.name)
    if feedback:
        parts.append(feedback)

    if step.skill:
        skill_raw = step.skill.replace("${PRESET_DIR}", str(preset.preset_dir))
        skill_path = Path(skill_raw).expanduser()
        if skill_path.is_file():
            parts.append(f"\n## Reference\n\n{skill_path.read_text()}")

    reminder = PIPELINE_PROTOCOL_REMINDER.format(
        pipeline_cli=PIPELINE_CLI, step_name=step.name,
    )
    parts.append(reminder)

    return "\n".join(parts)
