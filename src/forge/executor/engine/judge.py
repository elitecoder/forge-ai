
"""Judge engine — verify AI step output via structured checklist + external LLM judge."""

# Copyright 2026 — All rights reserved.

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .registry import JudgeConfig
from .state import PipelineState


@dataclass
class JudgeVerdict:
    passed: bool
    items: list[dict] = field(default_factory=list)  # [{"id": ..., "verdict": "pass|fail", "reason": ...}]
    summary: str = ""


def load_criteria(session_dir: str, step_name: str, config: JudgeConfig,
                  state: PipelineState) -> list[dict]:
    """Load criteria items based on criteria_source."""
    if config.criteria_source == "plan":
        return _load_plan_criteria(state)
    if config.criteria_source == "findings":
        return _load_findings_criteria(session_dir)
    if config.criteria_source == "visual_test_plan":
        return _load_visual_test_criteria(session_dir)
    if config.criteria_source == "changed_files":
        return _load_changed_files_criteria()
    return [{"id": "step-complete", "criteria": f"Complete the {step_name} step"}]


def _load_plan_criteria(state: PipelineState) -> list[dict]:
    """Parse plan file into numbered criteria items."""
    plan_path = Path(state.plan_file).expanduser() if state.plan_file else None
    if not plan_path or not plan_path.is_file():
        return [{"id": "plan-1", "criteria": "Implement all changes from the plan"}]

    items = []
    plan_text = plan_path.read_text()
    idx = 0
    for line in plan_text.splitlines():
        stripped = line.strip()
        # Match numbered items (1. ..., 1) ...) or bullet items (- ..., * ...)
        if stripped and (stripped[0].isdigit() or stripped.startswith(("- ", "* "))):
            idx += 1
            text = stripped.lstrip("0123456789.)- *").strip()
            if text and len(text) > 5:
                items.append({"id": f"plan-{idx}", "criteria": text})

    return items or [{"id": "plan-1", "criteria": "Implement all changes from the plan"}]


def _load_findings_criteria(session_dir: str) -> list[dict]:
    """Load code review findings from the reviewer's checklist JSON."""
    checklist_path = Path(session_dir) / "code_review-checklist.json"
    if checklist_path.is_file():
        try:
            data = json.loads(checklist_path.read_text())
            return [
                {"id": item["id"], "criteria": item["criteria"]}
                for item in data.get("checklist", [])
                if item.get("status") != "skipped"
            ]
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback: try to parse findings from pipeline-output.md
    output_path = Path(session_dir) / "pipeline-output.md"
    if output_path.is_file():
        items = []
        text = output_path.read_text()
        idx = 0
        in_review = False
        for line in text.splitlines():
            if "## code_review" in line:
                in_review = True
                continue
            if in_review and line.startswith("## "):
                break
            if in_review and line.strip().startswith(("- ", "* ", "1.", "2.", "3.", "4.", "5.")):
                idx += 1
                text_item = line.strip().lstrip("0123456789.)- *").strip()
                if text_item and len(text_item) > 5:
                    items.append({"id": f"finding-{idx}", "criteria": text_item})
        if items:
            return items

    return [{"id": "finding-1", "criteria": "Fix all code review findings"}]


def _load_visual_test_criteria(session_dir: str) -> list[dict]:
    """Load test scenarios from visual-test-plan.md."""
    vtp_path = Path(session_dir) / "visual-test-plan.md"
    if not vtp_path.is_file():
        return [{"id": "vt-1", "criteria": "Execute all visual test scenarios"}]

    items = []
    idx = 0
    for line in vtp_path.read_text().splitlines():
        stripped = line.strip()
        if stripped and (stripped[0].isdigit() or stripped.startswith(("- ", "* ", "### "))):
            idx += 1
            text = stripped.lstrip("0123456789.)- *#").strip()
            if text and len(text) > 5:
                items.append({"id": f"vt-{idx}", "criteria": text})

    return items or [{"id": "vt-1", "criteria": "Execute all visual test scenarios"}]


def _load_changed_files_criteria() -> list[dict]:
    """Load changed files as criteria items."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "main...HEAD"],
            capture_output=True, text=True,
        )
        files = [f.strip() for f in result.stdout.strip().splitlines() if f.strip()]
        return [{"id": f"file-{i+1}", "criteria": f"Review file {f}"} for i, f in enumerate(files)]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [{"id": "file-1", "criteria": "Review all changed files"}]


def build_judge_prompt(step_name: str, criteria: list[dict], checklist: dict, diff: str) -> str:
    """Build the judge evaluation prompt."""
    criteria_text = "\n".join(
        f"{i+1}. [{item['id']}] {item['criteria']}" for i, item in enumerate(criteria)
    )

    # Truncate diff if too large (keep first and last portions)
    max_diff = 30000
    if len(diff) > max_diff:
        half = max_diff // 2
        diff = diff[:half] + "\n\n... [diff truncated] ...\n\n" + diff[-half:]

    return f"""You are a pipeline judge. Your job is to verify that an agent completed its assigned work for step `{step_name}`.

## Criteria (what the agent was asked to do)
{criteria_text}

## Agent's Self-Report
{json.dumps(checklist, indent=2)}

## Actual Changes (git diff)
```diff
{diff}
```

For EACH criteria item, determine:
1. Did the agent claim to address it? (check self-report)
2. Does the git diff confirm the claim? (check actual changes)
3. Is the implementation correct and complete?

Output ONLY this JSON (no markdown fences, no extra text):
{{
  "items": [
    {{"id": "item-id", "verdict": "pass", "reason": "explanation"}},
    {{"id": "item-id", "verdict": "fail", "reason": "explanation"}}
  ]
}}

Rules:
- A "skipped" or "blocked" status in the self-report is a FAIL unless the reason is genuinely valid
- If the diff doesn't confirm the agent's evidence claim, it's a FAIL
- Be strict. Partial fixes are FAILs.
- Every criteria item must have a verdict entry"""


def _judge_log(activity_log_path: str, step_name: str, msg: str) -> None:
    """Append a timestamped entry to the activity log."""
    if not activity_log_path:
        return
    try:
        with open(activity_log_path, "a") as f:
            f.write(f"[{datetime.now().strftime('%H:%M')}] {step_name}  {msg}\n")
    except OSError:
        pass


def _save_judge_transcript(path: str, stdout: str, stderr: str = "") -> None:
    """Write judge output to a transcript file."""
    try:
        with open(path, "w") as f:
            if stdout:
                f.write(stdout)
                if not stdout.endswith("\n"):
                    f.write("\n")
            if stderr:
                f.write(f"[stderr] {stderr}\n")
    except OSError:
        pass


def spawn_judge(step_name: str, session_dir: str, criteria: list[dict],
                checklist: dict, diff: str, model: str = "opus",
                activity_log_path: str = "",
                attempt: int = 0) -> JudgeVerdict:
    """Spawn a judge agent and parse its verdict."""
    prompt = build_judge_prompt(step_name, criteria, checklist, diff)
    suffix = f"_attempt{attempt}" if attempt > 0 else ""
    transcript_path = os.path.join(session_dir, f"{step_name}_judge{suffix}-transcript.log")

    _judge_log(activity_log_path, step_name, f"judge ({model})")

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)

    try:
        result = subprocess.run(
            ["claude", "-p", "--dangerously-skip-permissions", "--model", model, "--max-turns", "20"],
            input=prompt, capture_output=True, text=True,
            timeout=120, env=env,
        )
        _save_judge_transcript(transcript_path, result.stdout, result.stderr)
        return _parse_judge_output(result.stdout, criteria)
    except subprocess.TimeoutExpired:
        _save_judge_transcript(transcript_path, "", "Judge timed out")
        _judge_log(activity_log_path, step_name, "judge timed out")
        return JudgeVerdict(
            passed=False,
            items=[{"id": c["id"], "verdict": "fail", "reason": "Judge timed out"} for c in criteria],
            summary="Judge agent timed out",
        )
    except FileNotFoundError:
        _save_judge_transcript(transcript_path, "", "Claude CLI not found")
        _judge_log(activity_log_path, step_name, "judge CLI not found")
        return JudgeVerdict(
            passed=False,
            items=[{"id": c["id"], "verdict": "fail", "reason": "Claude CLI not found"} for c in criteria],
            summary="Claude CLI not available for judge",
        )


def _parse_judge_output(output: str, criteria: list[dict]) -> JudgeVerdict:
    """Parse judge output JSON. Handles common LLM output quirks."""
    # Try to extract JSON from the output
    text = output.strip()

    # Strip markdown code fences if present
    if "```" in text:
        lines = text.splitlines()
        json_lines = []
        in_fence = False
        for line in lines:
            if line.strip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                json_lines.append(line)
        if json_lines:
            text = "\n".join(json_lines)

    # Try to find JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    try:
        data = json.loads(text)
        items = data.get("items", [])
        failed = [i for i in items if i.get("verdict") != "pass"]
        return JudgeVerdict(
            passed=len(failed) == 0 and len(items) > 0,
            items=items,
            summary=f"{len(items) - len(failed)}/{len(items)} items passed",
        )
    except (json.JSONDecodeError, KeyError):
        return JudgeVerdict(
            passed=False,
            items=[{"id": c["id"], "verdict": "fail", "reason": "Could not parse judge output"} for c in criteria],
            summary=f"Judge output unparseable: {text[:200]}",
        )


def save_judge_feedback(session_dir: str, step_name: str, attempt: int,
                        verdict: JudgeVerdict) -> str:
    """Save judge verdict to {session_dir}/_judge/{step_name}_attempt_{N}.json. Returns path."""
    judge_dir = Path(session_dir) / "_judge"
    judge_dir.mkdir(parents=True, exist_ok=True)

    path = judge_dir / f"{step_name}_attempt_{attempt}.json"
    path.write_text(json.dumps({
        "step": step_name,
        "attempt": attempt,
        "passed": verdict.passed,
        "summary": verdict.summary,
        "items": verdict.items,
    }, indent=2))

    return str(path)
