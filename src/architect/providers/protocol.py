# Copyright 2026. Provider protocol — AI tool abstraction layer.

from dataclasses import dataclass
from typing import Protocol


@dataclass
class AgentResult:
    exit_code: int
    stdout: str
    transcript_path: str
    timed_out: bool


class Provider(Protocol):
    def run_agent(self, prompt: str, model: str, max_turns: int, cwd: str,
                  timeout_s: int = 3600, step_name: str = "",
                  session_dir: str = "", activity_log_path: str = "",
                  system_prompt: str = "", allowed_tools: str = "") -> AgentResult: ...

    def run_judge(self, prompt: str, model: str = "fast",
                  max_turns: int = 20, timeout_s: int = 120) -> str: ...

    def format_subagent_type(self, agent_type: str, namespace: str = "") -> str: ...
