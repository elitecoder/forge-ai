
"""Visual test step wrapper -- Python orchestration with focused AI calls.

Replaces the monolithic SKILL.md prompt with deterministic Python orchestration.
AI agents handle only the creative parts (script generation, judging, triage, fixing).
"""

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from architect.core.runner import AgentRunner, AgentResult
from ..runner import build_context, PIPELINE_CLI
from ..state import PipelineState
from ..registry import Preset

MAX_SCRIPT_FIX_ATTEMPTS = 3


@dataclass
class VisualTestConfig:
    """Configurable paths for the visual test agent (skill dir, templates, credentials, etc.)."""
    skill_dir: str = ""
    template_path: str = ""
    dashboard_template_path: str = ""
    quirks_path: str = ""
    playwright_runner_dir: str = ""
    credentials_path: str = ""
    fixture_patterns: list[str] = field(default_factory=list)
    credential_env_vars: list[str] = field(default_factory=lambda: ["EMAIL", "PASSWORD"])


_DEFAULT_CONFIG = VisualTestConfig()


@dataclass
class VisualTestOutcome:
    passed: bool
    reason: str
    test_results_path: str = ""
    dashboard_path: str = ""


def _resolve_paths(config: VisualTestConfig) -> dict[str, Path]:
    """Resolve config strings into Path objects, expanding user."""
    skill_dir = Path(config.skill_dir).expanduser() if config.skill_dir else Path()
    return {
        "skill_dir": skill_dir,
        "template_path": Path(config.template_path).expanduser() if config.template_path else skill_dir / "scripts" / "template.js",
        "dashboard_template_path": Path(config.dashboard_template_path).expanduser() if config.dashboard_template_path else skill_dir / "scripts" / "dashboard-template.html",
        "quirks_path": Path(config.quirks_path).expanduser() if config.quirks_path else skill_dir / "references" / "quirks.md",
        "playwright_runner_dir": Path(config.playwright_runner_dir).expanduser() if config.playwright_runner_dir else Path(),
        "credentials_path": Path(config.credentials_path).expanduser() if config.credentials_path else Path(),
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
    config: VisualTestConfig | None = None,
) -> VisualTestOutcome:
    """Run the visual_test step: context -> generate -> execute -> judge -> triage/fix loop."""
    cfg = config or _DEFAULT_CONFIG
    paths = _resolve_paths(cfg)

    ctx = build_context(state, preset)
    feature_name = _detect_feature_name(session_dir)
    vt_dir = Path(session_dir) / "visual-test"
    vt_dir.mkdir(parents=True, exist_ok=True)

    context_path = _build_context_file(state, preset, cwd, session_dir, feature_name, cfg)

    script_path = Path(session_dir) / f"playwright-test-{feature_name}.js"
    results_path = Path(session_dir) / f"{feature_name}-test-results.json"

    for attempt in range(1, MAX_SCRIPT_FIX_ATTEMPTS + 1):
        attempt_dir = vt_dir / f"attempt-{attempt}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        runner = AgentRunner(
            session_dir=session_dir,
            step_name="visual_test",
            activity_log_path=activity_log_path,
            agent_command=agent_command,
        )

        if attempt == 1 and not script_path.is_file():
            gen_prompt = _script_generation_prompt(
                context_path, script_path, feature_name, session_dir,
            )
        else:
            triage_path = attempt_dir.parent / f"attempt-{attempt - 1}" / "triage.json" if attempt > 1 else None
            triage_reason = ""
            if triage_path and triage_path.is_file():
                try:
                    triage_data = json.loads(triage_path.read_text())
                    triage_reason = triage_data.get("fix_hints", triage_data.get("reason", ""))
                except (json.JSONDecodeError, OSError):
                    pass
            gen_prompt = _script_fix_prompt(script_path, triage_reason, session_dir)

        runner.run(
            prompt=gen_prompt,
            model=model,
            max_turns=max_turns,
            cwd=cwd,
            timeout_s=timeout_s,
        )

        if not script_path.is_file():
            _report_fail(session_dir, "Script generation failed — no script file produced")
            return VisualTestOutcome(passed=False, reason="Script generation failed")

        shutil.copy2(script_path, attempt_dir / script_path.name)

        exec_ok, exec_output = _execute_script(script_path, session_dir, cwd, cfg)

        if not results_path.is_file():
            (attempt_dir / "exec-output.txt").write_text(exec_output)
            if attempt < MAX_SCRIPT_FIX_ATTEMPTS:
                _write_triage(attempt_dir, "script_issue",
                              "Script crashed before writing results JSON",
                              f"Execution output:\n{exec_output[-2000:]}")
                continue
            _report_fail(session_dir, "Script crashed — no results JSON after max attempts")
            return VisualTestOutcome(passed=False, reason="No results JSON produced")

        try:
            test_results = json.loads(results_path.read_text())
        except json.JSONDecodeError as e:
            _report_fail(session_dir, f"Malformed results JSON: {e}")
            return VisualTestOutcome(passed=False, reason=f"Malformed results JSON: {e}")

        shutil.copy2(results_path, attempt_dir / results_path.name)

        screenshots_dir = attempt_dir / "screenshots"
        screenshots_dir.mkdir(exist_ok=True)
        for png in Path(session_dir).glob("verify-*.png"):
            shutil.copy2(png, screenshots_dir / png.name)

        plan_path = _find_visual_test_plan(session_dir)
        screenshot_paths = sorted(str(p) for p in Path(session_dir).glob("verify-*.png"))
        verdict_path = attempt_dir / "judge-verdict.json"

        judge_prompt = _judge_prompt(
            plan_path, script_path, results_path, screenshot_paths, verdict_path,
        )
        judge_runner = AgentRunner(
            session_dir=session_dir,
            step_name="visual_test_judge",
            activity_log_path=activity_log_path,
            agent_command=agent_command,
        )
        judge_runner.run(
            prompt=judge_prompt,
            model="opus",
            max_turns=0,
            cwd=cwd,
            timeout_s=600,
        )

        verdict = _parse_judge_verdict(verdict_path)
        if verdict is None:
            if attempt < MAX_SCRIPT_FIX_ATTEMPTS:
                _write_triage(attempt_dir, "script_issue",
                              "Judge agent failed to produce verdict",
                              "Re-examine the script and results for correctness")
                continue
            _report_fail(session_dir, "Judge failed to produce verdict after max attempts")
            return VisualTestOutcome(passed=False, reason="Judge failed to produce verdict")

        all_pass = verdict.get("verdict") == "PASS"
        if all_pass:
            dashboard_path = _generate_dashboard(session_dir, feature_name, test_results, verdict)
            _report_pass(session_dir)
            return VisualTestOutcome(
                passed=True,
                reason="All scenarios passed",
                test_results_path=str(results_path),
                dashboard_path=dashboard_path,
            )

        triage_path = attempt_dir / "triage.json"
        triage_prompt = _triage_prompt(verdict_path, results_path, script_path, triage_path)
        triage_runner = AgentRunner(
            session_dir=session_dir,
            step_name="visual_test_triage",
            activity_log_path=activity_log_path,
            agent_command=agent_command,
        )
        triage_runner.run(
            prompt=triage_prompt,
            model=model,
            max_turns=0,
            cwd=cwd,
            timeout_s=300,
        )

        triage = _parse_triage(triage_path)
        if triage is None:
            triage = {"diagnosis": "script_issue", "reason": "Triage agent failed",
                      "fix_hints": "Review script and results manually"}

        if triage.get("diagnosis") == "code_issue":
            dashboard_path = _generate_dashboard(session_dir, feature_name, test_results, verdict)
            _report_fail(session_dir, f"Code issue: {triage.get('reason', 'unknown')}")
            return VisualTestOutcome(
                passed=False,
                reason=f"Code issue: {triage.get('reason', 'unknown')}",
                test_results_path=str(results_path),
                dashboard_path=dashboard_path,
            )

        if attempt >= MAX_SCRIPT_FIX_ATTEMPTS:
            dashboard_path = _generate_dashboard(session_dir, feature_name, test_results, verdict)
            _report_fail(session_dir, f"Script fix exhausted after {MAX_SCRIPT_FIX_ATTEMPTS} attempts")
            return VisualTestOutcome(
                passed=False,
                reason=f"Script fix exhausted after {MAX_SCRIPT_FIX_ATTEMPTS} attempts",
                test_results_path=str(results_path),
                dashboard_path=dashboard_path,
            )

    _report_fail(session_dir, "Visual test loop exited unexpectedly")
    return VisualTestOutcome(passed=False, reason="Unexpected loop exit")


# -- Context file generation (pure Python) -----------------------------------


def _detect_feature_name(session_dir: str) -> str:
    """Extract feature name from the visual test plan filename."""
    for p in Path(session_dir).glob("visual-test-plan*"):
        name = p.stem.replace("visual-test-plan-", "").replace("visual-test-plan", "")
        if name:
            return name.strip("-")
    return "feature"


def _find_visual_test_plan(session_dir: str) -> str:
    """Find the visual test plan file in session dir."""
    for p in Path(session_dir).glob("visual-test-plan*"):
        return str(p)
    return ""


def _build_context_file(
    state: PipelineState,
    preset: Preset,
    cwd: str,
    session_dir: str,
    feature_name: str,
    config: VisualTestConfig | None = None,
) -> str:
    """Write visual-test-context.md with all info the script generator needs."""
    cfg = config or _DEFAULT_CONFIG
    paths = _resolve_paths(cfg)

    ctx = build_context(state, preset)
    parts: list[str] = []

    plan_path = _find_visual_test_plan(session_dir)
    if plan_path:
        parts.append("## Visual Test Plan\n")
        parts.append(Path(plan_path).read_text())
        parts.append("")

    fixture_excerpts = _extract_fixture_excerpts(cwd, cfg.fixture_patterns)
    if fixture_excerpts:
        parts.append("## E2E Fixture Patterns\n")
        parts.append(fixture_excerpts)
        parts.append("")

    template_path = paths["template_path"]
    if template_path.is_file():
        parts.append("## Playwright Script Template\n")
        parts.append(f"Template location: `{template_path}`\n")
        parts.append("```javascript")
        parts.append(template_path.read_text())
        parts.append("```\n")

    quirks_path = paths["quirks_path"]
    if quirks_path.is_file():
        parts.append("## Known Quirks\n")
        parts.append(quirks_path.read_text())
        parts.append("")

    parts.append("## Dev Server\n")
    parts.append(f"- URL: `{ctx['DEV_SERVER_URL']}`")
    parts.append(f"- Port: `{ctx['DEV_SERVER_PORT']}`")
    parts.append("")

    credentials_path = paths["credentials_path"]
    parts.append("## Credentials\n")
    parts.append(f"- Source credentials before running: `source {credentials_path}`")
    env_vars = ", ".join(f"`{v}`" for v in cfg.credential_env_vars)
    parts.append(f"- Env vars: {env_vars}")
    parts.append("")

    playwright_runner_dir = paths["playwright_runner_dir"]
    parts.append("## Execution\n")
    parts.append(f"- Feature name: `{feature_name}`")
    parts.append(f"- Script output: `{session_dir}/playwright-test-{feature_name}.js`")
    parts.append(f"- Results output: `{session_dir}/{feature_name}-test-results.json`")
    parts.append(f"- Screenshot dir: `{session_dir}/`")
    parts.append(f"- Set `PIPELINE_SESSION_DIR={session_dir}` in the script's SCREENSHOT_DIR")
    parts.append(f"- Runner: `cd {playwright_runner_dir} && node run.js <script>`")
    parts.append("")

    context_path = Path(session_dir) / "visual-test-context.md"
    context_path.write_text("\n".join(parts))
    return str(context_path)


def _extract_fixture_excerpts(cwd: str, fixture_patterns: list[str] | None = None) -> str:
    """Extract key methods from E2E fixtures."""
    patterns = fixture_patterns or []
    excerpts: list[str] = []
    for rel_path in patterns:
        full_path = Path(cwd) / rel_path
        if not full_path.is_file():
            continue
        try:
            content = full_path.read_text()
        except OSError:
            continue
        methods = []
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith(("async ", "public ", "get ", "readonly ")):
                methods.append(stripped)
        if methods:
            excerpts.append(f"### `{rel_path}`\n")
            excerpts.append("Key methods/properties:")
            for m in methods[:30]:
                excerpts.append(f"- `{m[:120]}`")
            excerpts.append("")
    return "\n".join(excerpts)


# -- AI prompt templates ------------------------------------------------------


def _script_generation_prompt(
    context_path: str | Path,
    script_path: str | Path,
    feature_name: str,
    session_dir: str,
) -> str:
    return f"""## Task

Generate a Playwright visual test script based on the context file.

1. Read `{context_path}` for the full test plan, E2E fixtures, template, and quirks.
2. Follow the template structure exactly. Use E2E fixture patterns for interactions.
3. Write the script to `{script_path}`.
4. Set FEATURE_NAME to `{feature_name}`.
5. Set SCREENSHOT_DIR to `{session_dir}`.
6. Set DEV_PORT from the context file's Dev Server section.

Do NOT modify any source code files. Only create the Playwright test script.
"""


def _script_fix_prompt(
    script_path: str | Path,
    triage_reason: str,
    session_dir: str,
) -> str:
    reason_section = ""
    if triage_reason:
        reason_section = f"\nThe previous attempt failed because:\n{triage_reason}\n"

    return f"""## Task

Fix the existing Playwright test script.

1. Read the script at `{script_path}`.{reason_section}
2. Fix the identified issues in the script.
3. Write the updated script to the same path: `{script_path}`.
4. Do NOT change expected assertions or remove scenarios.
5. Do NOT modify any source code files.
"""


def _judge_prompt(
    plan_path: str,
    script_path: str | Path,
    results_path: str | Path,
    screenshot_paths: list[str],
    verdict_path: str | Path,
) -> str:
    screenshots_section = ""
    if screenshot_paths:
        screenshots_section = "\n## Screenshots\n\nView these files:\n"
        for p in screenshot_paths:
            screenshots_section += f"- `{p}`\n"

    plan_section = ""
    if plan_path:
        plan_section = f"\n## Test Plan\n\nRead: `{plan_path}`\n"

    return f"""## Task

You are a visual test judge. Evaluate whether the test results satisfy the plan.
{plan_section}
## Evidence

- Script: `{script_path}`
- Results: `{results_path}`
{screenshots_section}
## Output

Read the plan, script, results, and screenshots. Then write your verdict as JSON to `{verdict_path}`:

```json
{{
  "verdict": "PASS" or "FAIL",
  "scenarios": [
    {{"number": 1, "verdict": "PASS" or "FAIL", "notes": "explanation"}}
  ],
  "summary": "Overall assessment"
}}
```

A scenario PASSES only if the test correctly implements the plan and the results confirm the expected behavior.
"""


def _triage_prompt(
    verdict_path: str | Path,
    results_path: str | Path,
    script_path: str | Path,
    triage_output_path: str | Path,
) -> str:
    return f"""## Task

A visual test failed. Diagnose whether this is a script issue or a code issue.

## Evidence

- Judge verdict: `{verdict_path}`
- Test results: `{results_path}`
- Script: `{script_path}`

## Output

Read all evidence files, then write triage JSON to `{triage_output_path}`:

```json
{{
  "diagnosis": "script_issue" or "code_issue",
  "target": "visual_test" or "code",
  "reason": "What went wrong",
  "fix_hints": "Specific guidance on what to fix"
}}
```

- `script_issue`: The test script has bugs (wrong selectors, timing, missing waits, wrong assertions).
- `code_issue`: The source code behavior doesn't match what was planned. The script is correct but the code is wrong.

If in doubt, prefer `script_issue` -- it's safer to retry the script first.
"""


# -- Script execution (Python subprocess) -------------------------------------


def _execute_script(
    script_path: str | Path,
    session_dir: str,
    cwd: str,
    config: VisualTestConfig | None = None,
) -> tuple[bool, str]:
    """Execute Playwright script via node run.js. Returns (passed, output)."""
    cfg = config or _DEFAULT_CONFIG
    paths = _resolve_paths(cfg)
    playwright_runner_dir = paths["playwright_runner_dir"]
    credentials_path = paths["credentials_path"]

    runner_script = playwright_runner_dir / "run.js"
    if not runner_script.is_file():
        return False, f"Playwright runner not found: {runner_script}"

    cmd = f"source {credentials_path} 2>/dev/null; cd {playwright_runner_dir} && node run.js {script_path}"
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            cwd=cwd, timeout=300,
            env={**__import__("os").environ, "PIPELINE_SESSION_DIR": session_dir},
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "Script execution timed out after 300s"


# -- Result parsing -----------------------------------------------------------


def _parse_judge_verdict(verdict_path: Path) -> dict | None:
    """Parse judge verdict JSON. Returns None if missing/malformed."""
    if not verdict_path.is_file():
        return None
    try:
        data = json.loads(verdict_path.read_text())
        if "verdict" not in data:
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def _parse_triage(triage_path: Path) -> dict | None:
    """Parse triage JSON. Returns None if missing/malformed."""
    if not triage_path.is_file():
        return None
    try:
        data = json.loads(triage_path.read_text())
        if "diagnosis" not in data:
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def _write_triage(attempt_dir: Path, diagnosis: str, reason: str, fix_hints: str) -> None:
    """Write a synthetic triage file for the next retry."""
    triage_path = attempt_dir / "triage.json"
    triage_path.write_text(json.dumps({
        "diagnosis": diagnosis,
        "reason": reason,
        "fix_hints": fix_hints,
    }, indent=2))


# -- Dashboard generation (Python) -------------------------------------------


def _generate_dashboard(
    session_dir: str,
    feature_name: str,
    test_results: dict,
    verdict: dict,
    config: VisualTestConfig | None = None,
) -> str:
    """Generate visual test dashboard HTML from template + results data."""
    import re
    from datetime import datetime

    cfg = config or _DEFAULT_CONFIG
    paths = _resolve_paths(cfg)
    dashboard_template_path = paths["dashboard_template_path"]

    dashboard_path = Path(session_dir) / "visual-test-dashboard.html"

    if dashboard_template_path.is_file():
        template = dashboard_template_path.read_text()
    else:
        template = ""

    summary = test_results.get("summary", {})
    total = summary.get("total", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    total_assertions = sum(
        len(r.get("assertions", []))
        for r in test_results.get("results", [])
    )
    screenshot_count = len(list(Path(session_dir).glob("verify-*.png")))

    badge_text = "ALL PASS" if failed == 0 else f"{failed} FAILED"
    badge_class = "all-pass" if failed == 0 else "has-fail"
    stat_class = "all-pass" if failed == 0 else "has-failures"
    summary_line = (
        f"{passed}/{total} scenarios passing &bull; "
        f"{total_assertions} assertions &bull; "
        f"{screenshot_count} screenshots captured"
    )

    scenario_verdicts = {s.get("number"): s for s in verdict.get("scenarios", [])}
    cards: list[str] = []
    for i, result in enumerate(test_results.get("results", [])):
        number = result.get("number", i + 1)
        title = result.get("title", f"Scenario {number}")
        subtitle = result.get("subtitle", "")
        status = result.get("status", "UNKNOWN")
        assertions = result.get("assertions", [])
        screenshots = result.get("screenshots", {})

        sid = f"s{number}"
        open_cls = " open" if i == 0 else ""
        fail_cls = " scenario-fail" if status == "FAIL" else ""
        tag_cls = "tag-pass" if status == "PASS" else "tag-fail"

        assert_html = []
        for a in assertions:
            a_cls = "pass" if a.get("passed") else "fail"
            a_icon = "&#10003;" if a.get("passed") else "&#10007;"
            assert_html.append(
                f'            <div class="assertion {a_cls}">'
                f'<span class="assertion-icon">{a_icon}</span>'
                f'<span class="assertion-label">{a.get("label", "")}</span>'
                f'<span class="assertion-value">{a.get("value", "")}</span>'
                f'</div>'
            )

        before_img = screenshots.get("before", "")
        after_img = screenshots.get("after", "")
        img_html = []
        if before_img:
            img_html.append(
                f'            <div class="comparison-pane">'
                f'<div class="comparison-label before">&#9679; Before</div>'
                f'<img src="{before_img}" alt="Before" onclick="openLightbox(this)"/></div>'
            )
        if after_img:
            img_html.append(
                f'            <div class="comparison-pane">'
                f'<div class="comparison-label after">&#9679; After</div>'
                f'<img src="{after_img}" alt="After" onclick="openLightbox(this)"/></div>'
            )

        sv = scenario_verdicts.get(number, {})
        judge_notes = sv.get("notes", "")

        card = f"""      <div class="scenario{open_cls}{fail_cls}" id="{sid}">
        <div class="scenario-header" onclick="toggle('{sid}')">
          <div class="scenario-title-area">
            <div class="scenario-number">{number}</div>
            <div>
              <div class="scenario-title">{title}</div>
              <div class="scenario-subtitle">{subtitle}</div>
            </div>
          </div>
          <div class="scenario-tags">
            <span class="tag {tag_cls}">{status}</span>
            <span class="chevron">&#9662;</span>
          </div>
        </div>
        <div class="scenario-body">
          <div class="assertions">
{chr(10).join(assert_html)}
          </div>
          <div class="validation-section collapsed">
            <div class="validation-header" onclick="this.parentElement.classList.toggle('collapsed')">
              &#9662; Judge Notes
            </div>
            <div class="validation-checks">
              <div class="validation-check">{judge_notes}</div>
            </div>
          </div>
          <div class="comparison">
{chr(10).join(img_html)}
          </div>
        </div>
      </div>"""
        cards.append(card)

    scenario_cards_html = "\n\n".join(cards)

    if template:
        html = template
        html = html.replace("{{FEATURE_NAME}}", feature_name)
        html = html.replace("{{TIMESTAMP}}", datetime.now().isoformat())
        html = html.replace("{{TOTAL_ASSERTIONS}}", str(total_assertions))
        html = html.replace("{{BADGE_TEXT}}", badge_text)
        html = html.replace("{{BADGE_CLASS}}", badge_class)
        html = html.replace("{{STAT_CLASS}}", stat_class)
        html = html.replace("{{SUMMARY_LINE}}", summary_line)
        html = re.sub(
            r'(<!-- Example scenario card -->).*?(</div>\s*<div class="footer">)',
            scenario_cards_html + r'\n\n      <div class="footer">',
            html,
            flags=re.DOTALL,
        )
    else:
        html = f"""<!DOCTYPE html>
<html><head><title>Visual Test Dashboard -- {feature_name}</title></head>
<body>
<h1>Visual Test Dashboard -- {feature_name}</h1>
<p>{badge_text} | {summary_line}</p>
{scenario_cards_html}
<pre>{json.dumps(test_results, indent=2)}</pre>
</body></html>"""

    dashboard_path.write_text(html)
    return str(dashboard_path)


# -- Reporting ----------------------------------------------------------------


def _report_pass(session_dir: str) -> None:
    subprocess.run(
        [sys.executable, PIPELINE_CLI, "pass", "visual_test"],
        capture_output=True, text=True,
    )


def _report_fail(session_dir: str, reason: str) -> None:
    subprocess.run(
        [sys.executable, PIPELINE_CLI, "fail", "visual_test", reason],
        capture_output=True, text=True,
    )
