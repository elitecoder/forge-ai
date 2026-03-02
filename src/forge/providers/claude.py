# Copyright 2026. Claude Code provider — streaming subprocess wrapper.

import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from forge.core.logging import log_activity, utc_timestamp
from .protocol import AgentResult


_EVENT_PATTERNS = [
    (re.compile(r"Read(?:ing)?\s+(?:file:?\s*)?[`'\"]?([^\s`'\"]+)", re.IGNORECASE), "read"),
    (re.compile(r"\[Read\]\s*(.+)", re.IGNORECASE), "read"),
    (re.compile(r"(?:Edit|Write|Writing|Editing)\s+(?:file:?\s*)?[`'\"]?([^\s`'\"]+)", re.IGNORECASE), "write"),
    (re.compile(r"\[(?:Edit|Write)\]\s*(.+)", re.IGNORECASE), "write"),
    (re.compile(r"(?:Running|Bash|Executing|run):?\s*[`'\"]?(.+?)(?:[`'\"]?\s*$)", re.IGNORECASE), "bash"),
    (re.compile(r"^\$\s+(.+)$"), "bash"),
    (re.compile(r"(?:Error|FAILED|error:)\s*(.+)", re.IGNORECASE), "error"),
]

NARRATION_SUFFIX = (
    "\n\nThink out loud as you work. Say what file you're reading and why, "
    "what change you're making, and whether the build passed."
)

_MAX_INMEMORY_LINES = 100_000
_STALL_TIMEOUT_S = 900  # Kill subprocess if no output for 15 minutes


def _ts() -> str:
    return utc_timestamp()


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    return env


def _extract_content_text(content) -> list[str]:
    if isinstance(content, str) and content.strip():
        return content.splitlines()
    if isinstance(content, list):
        lines: list[str] = []
        for block in content:
            if isinstance(block, str) and block.strip():
                lines.extend(block.splitlines())
            elif isinstance(block, dict):
                text = block.get("text", "")
                if text and isinstance(text, str):
                    lines.extend(text.splitlines())
        return lines
    return []


class ClaudeProvider:
    """Claude Code CLI provider — spawns `claude -p` subprocesses."""

    def __init__(self, agent_command: str = "claude"):
        self.agent_command = agent_command

    def run_agent(self, prompt: str, model: str, max_turns: int, cwd: str,
                  timeout_s: int = 3600, step_name: str = "",
                  session_dir: str = "", activity_log_path: str = "",
                  system_prompt: str = "", allowed_tools: str = "") -> AgentResult:
        transcript_path = str(
            Path(session_dir) / f"{step_name}-transcript.log"
        ) if session_dir and step_name else ""
        raw_log_path = str(
            Path(session_dir) / f"{step_name}-stream-raw.log"
        ) if session_dir and step_name else ""
        full_prompt = prompt + NARRATION_SUFFIX

        cmd = [
            self.agent_command, "-p",
            "--dangerously-skip-permissions",
            "--verbose",
            "--output-format", "stream-json",
            "--model", model,
        ]
        plugin_dir = os.environ.get("FORGE_PLUGIN_DIR", "")
        if plugin_dir:
            cmd += ["--plugin-dir", plugin_dir]
        if max_turns > 0:
            cmd += ["--max-turns", str(max_turns)]
        if system_prompt:
            cmd += ["--system-prompt", system_prompt]
        if allowed_tools:
            cmd += ["--allowedTools", allowed_tools]

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                env=_child_env(),
                text=True,
            )
        except FileNotFoundError:
            self._log_activity(activity_log_path, step_name, f"ERROR: '{self.agent_command}' not found")
            return AgentResult(
                exit_code=127,
                stdout="",
                transcript_path=transcript_path,
                timed_out=False,
            )

        try:
            proc.stdin.write(full_prompt)
            proc.stdin.close()
        except OSError:
            pass

        lines: list[str] = []
        line_count = 0
        last_activity = time.monotonic()
        activity_lock = threading.Lock()
        stop_event = threading.Event()

        def _touch_activity():
            nonlocal last_activity
            with activity_lock:
                last_activity = time.monotonic()

        def _seconds_since_activity() -> float:
            with activity_lock:
                return time.monotonic() - last_activity

        def _reader():
            nonlocal line_count
            for raw_line in proc.stdout:
                raw = raw_line.rstrip("\n")
                if not raw:
                    continue
                line_count += 1
                _touch_activity()
                if raw_log_path:
                    self._write_log(raw_log_path, raw)
                for readable in self._parse_stream_json(raw):
                    if len(lines) < _MAX_INMEMORY_LINES:
                        lines.append(readable)
                    if transcript_path:
                        self._write_log(transcript_path, readable)
                    if activity_log_path:
                        self._check_key_event(activity_log_path, step_name, readable)

        reader_thread = threading.Thread(target=_reader, daemon=True)
        reader_thread.start()

        heartbeat_thread = self._start_heartbeat(
            stop_event, lambda: line_count, _seconds_since_activity,
            activity_log_path, step_name, proc,
        )

        timed_out = False
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait()
            self._log_activity(activity_log_path, step_name, f"TIMEOUT after {timeout_s}s")

        # Stall-killed processes exit with -9 (SIGKILL); treat as timeout
        if proc.returncode == -9 and not timed_out:
            timed_out = True
            self._log_activity(activity_log_path, step_name, "Process was stall-killed")

        stop_event.set()
        reader_thread.join(timeout=5)
        heartbeat_thread.join(timeout=5)

        return AgentResult(
            exit_code=proc.returncode or 0,
            stdout="\n".join(lines),
            transcript_path=transcript_path,
            timed_out=timed_out,
        )

    def run_judge(self, prompt: str, model: str = "fast",
                  max_turns: int = 20, timeout_s: int = 120) -> str:
        cmd = [self.agent_command, "-p", "--model", model, "--max-turns", str(max_turns)]
        try:
            result = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True,
                timeout=timeout_s, env=_child_env(),
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            return ""

    def format_subagent_type(self, agent_type: str, namespace: str = "") -> str:
        if namespace and agent_type != "general-purpose":
            return f"{namespace}:{agent_type}"
        return agent_type

    def _parse_stream_json(self, raw: str) -> list[str]:
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return [raw] if raw.strip() else []

        if not isinstance(msg, dict):
            return [raw] if raw.strip() else []

        out: list[str] = []
        msg_type = msg.get("type", "")

        if msg_type == "assistant":
            for block in msg.get("message", {}).get("content", []):
                if block.get("type") == "text" and block.get("text", "").strip():
                    out.append(block["text"])
                elif block.get("type") == "tool_use":
                    name = block.get("name", "?")
                    inp = block.get("input", {})
                    if name == "Bash":
                        out.append(f"$ {inp.get('command', '')}")
                    else:
                        detail = inp.get("file_path", inp.get("pattern", inp.get("path", "")))
                        out.append(f"[{name}] {detail}")

        elif msg_type == "tool_result":
            content = None
            if isinstance(msg.get("tool_result"), dict):
                content = msg["tool_result"].get("content")
            if content is None:
                content = msg.get("content")
            if content is None:
                if isinstance(msg.get("tool_result"), dict):
                    content = msg["tool_result"].get("output", "")
                else:
                    content = msg.get("output", "")
            out.extend(_extract_content_text(content))

        elif msg_type == "result":
            r = msg.get("result", "")
            if isinstance(r, str) and r.strip():
                out.append(r)

        elif msg_type == "system":
            text = msg.get("message", msg.get("text", ""))
            subtype = msg.get("subtype", "")
            if isinstance(text, str) and text.strip():
                out.append(f"[system:{subtype}] {text}" if subtype else f"[system] {text}")

        elif msg_type == "error":
            error = msg.get("error", msg.get("message", ""))
            if isinstance(error, dict):
                error = error.get("message", str(error))
            if error:
                out.append(f"[error] {error}")

        elif msg_type == "user":
            for item in msg.get("message", {}).get("content", []):
                if isinstance(item, dict):
                    out.extend(_extract_content_text(item.get("content", "")))

        return out

    def _start_heartbeat(self, stop_event: threading.Event, get_line_count,
                         get_idle_seconds, activity_log_path: str,
                         step_name: str, proc: subprocess.Popen) -> threading.Thread:
        def _beat():
            while not stop_event.wait(30):
                count = get_line_count()
                idle = get_idle_seconds()
                self._log_activity(
                    activity_log_path, step_name,
                    f"agent active ({count} lines, idle {int(idle)}s)",
                )
                if idle >= _STALL_TIMEOUT_S:
                    self._log_activity(
                        activity_log_path, step_name,
                        f"STALL detected: no output for {int(idle)}s, killing subprocess",
                    )
                    try:
                        proc.kill()
                    except OSError:
                        pass
                    return

        t = threading.Thread(target=_beat, daemon=True)
        t.start()
        return t

    def _check_key_event(self, activity_log_path: str, step_name: str, line: str) -> None:
        for pattern, event_type in _EVENT_PATTERNS:
            m = pattern.search(line)
            if m:
                detail = m.group(1)[:120]
                self._log_activity(activity_log_path, step_name, f"{event_type}: {detail}")
                return

    @staticmethod
    def _log_activity(activity_log_path: str, step_name: str, msg: str) -> None:
        log_activity(activity_log_path, step_name, msg)

    @staticmethod
    def _write_log(path: str, line: str) -> None:
        try:
            with open(path, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass
