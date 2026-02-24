"""Post-step summary extraction from activity logs."""

import json
import os
import re
from pathlib import Path


def extract_summary(activity_log_path: str, step_name: str, passed: bool) -> dict:
    files_read: list[str] = []
    files_written: list[str] = []
    commands: list[str] = []
    errors: list[str] = []

    if not os.path.isfile(activity_log_path):
        return _build_summary(step_name, passed, files_read, files_written, commands, errors)

    entry_re = re.compile(
        rf"^\[[\d:]+\]\s+{re.escape(step_name)}\s{{2}}(\w+):\s+(.+)$"
    )

    try:
        for line in Path(activity_log_path).read_text().splitlines():
            m = entry_re.match(line)
            if not m:
                continue
            event_type, detail = m.group(1), m.group(2).strip()
            if event_type == "read":
                files_read.append(detail)
            elif event_type == "write":
                files_written.append(detail)
            elif event_type == "bash":
                commands.append(detail)
            elif event_type == "error":
                errors.append(detail)
    except OSError:
        pass

    return _build_summary(step_name, passed, files_read, files_written, commands, errors)


def _build_summary(
    step_name: str, passed: bool,
    files_read: list[str], files_written: list[str],
    commands: list[str], errors: list[str],
) -> dict:
    return {
        "step": step_name,
        "outcome": "pass" if passed else "fail",
        "files_read": sorted(set(files_read)),
        "files_written": sorted(set(files_written)),
        "commands": commands,
        "errors": errors,
    }


def write_step_summary(session_dir: str, step_name: str, passed: bool) -> str | None:
    activity_log = os.path.join(session_dir, "pipeline-activity.log")
    summary = extract_summary(activity_log, step_name, passed)

    if not any(summary.get(k) for k in ("files_read", "files_written", "commands", "errors")):
        return None

    out_path = os.path.join(session_dir, f"{step_name}-actions.json")
    Path(out_path).write_text(json.dumps(summary, indent=2) + "\n")

    _append_to_output(session_dir, step_name, summary)

    return out_path


def _append_to_output(session_dir: str, step_name: str, summary: dict) -> None:
    output_path = os.path.join(session_dir, "pipeline-output.md")
    lines = [f"\n## {step_name} — Actions"]
    lines.append(f"Outcome: **{summary['outcome']}**\n")

    if summary["files_read"]:
        lines.append(f"Files read ({len(summary['files_read'])}):")
        for f in summary["files_read"][:10]:
            lines.append(f"- `{f}`")
        if len(summary["files_read"]) > 10:
            lines.append(f"- ... and {len(summary['files_read']) - 10} more")

    if summary["files_written"]:
        lines.append(f"\nFiles modified ({len(summary['files_written'])}):")
        for f in summary["files_written"][:10]:
            lines.append(f"- `{f}`")
        if len(summary["files_written"]) > 10:
            lines.append(f"- ... and {len(summary['files_written']) - 10} more")

    if summary["commands"]:
        lines.append(f"\nCommands ({len(summary['commands'])}):")
        for c in summary["commands"][:5]:
            lines.append(f"- `{c[:100]}`")
        if len(summary["commands"]) > 5:
            lines.append(f"- ... and {len(summary['commands']) - 5} more")

    if summary["errors"]:
        lines.append(f"\nErrors ({len(summary['errors'])}):")
        for e in summary["errors"][:5]:
            lines.append(f"- {e[:120]}")

    lines.append("")
    try:
        with open(output_path, "a") as fh:
            fh.write("\n".join(lines))
    except OSError:
        pass
