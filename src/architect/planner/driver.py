# Copyright 2026. Planner driver — deterministic 6-phase orchestrator.

import json
import os
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from architect.core.logging import log_activity, StatusWriter
from architect.planner.engine.evidence import validate_phase
from architect.planner.engine.state import (
    PHASE_ORDER,
    PhaseState,
    PhaseStatus,
    PlannerState,
    StateManager,
    now_iso,
)
from architect.providers.protocol import AgentResult, Provider


MAX_RETRIES = 2

# Map agent_type to prompt template filename and allowed tools (from frontmatter)
_AGENT_TYPE_PROMPTS: dict[str, tuple[str, str]] = {
    "planner-architect-topdown":  ("architect-topdown.md", "Read Write Glob Grep"),
    "planner-architect-bottomup": ("architect-bottomup.md", "Read Write Glob Grep"),
    "planner-critic":             ("critic.md",            "Read Write"),
    "planner-refiner":            ("refiner.md",           "Read Write"),
    "planner-judge":              ("judge.md",             "Read Write"),
}

PHASE_MODELS = {
    "recon":       {"default": "sonnet", "fast": "sonnet"},
    "architects":  {"default": "opus",   "fast": "sonnet"},
    "critics":     {"default": "opus",   "fast": "sonnet"},
    "refiners":    {"default": "sonnet", "fast": "sonnet"},
    "judge":       {"default": "opus",   "fast": "sonnet"},
    "enrichment":  {"default": "sonnet", "fast": "sonnet"},
}

PHASE_MAX_TURNS = {
    "recon": 200,
    "architects": 200,
    "critics": 200,
    "refiners": 200,
    "judge": 200,
    "enrichment": 200,
}

# Agent definitions per phase: name, output file, agent_type, optional id
PHASE_AGENTS: dict[str, list[dict]] = {
    "recon": [
        {"name": "recon", "output": "codebase-brief.md"},
    ],
    "architects": [
        {"name": "architect-a", "agent_type": "planner-architect-topdown",
         "output": "design-a.md", "id": "a"},
        {"name": "architect-b", "agent_type": "planner-architect-bottomup",
         "output": "design-b.md", "id": "b"},
    ],
    "critics": [
        {"name": "critic-a", "agent_type": "planner-critic",
         "output": "critique-a.md", "id": "a"},
        {"name": "critic-b", "agent_type": "planner-critic",
         "output": "critique-b.md", "id": "b"},
    ],
    "refiners": [
        {"name": "refiner-a", "agent_type": "planner-refiner",
         "output": "refined-a.md", "id": "a"},
        {"name": "refiner-b", "agent_type": "planner-refiner",
         "output": "refined-b.md", "id": "b"},
    ],
    "judge": [
        {"name": "judge", "agent_type": "planner-judge", "output": "final-plan.md"},
    ],
}

DERIVE_CONTEXT_PROMPT = """Given this problem statement, extract:
1. The core tension (the fundamental trade-off)
2. Constraint A (one side of the trade-off to optimize for)
3. Constraint B (the other side of the trade-off to optimize for)

Problem: {problem}

Respond in JSON: {{"tension": "...", "constraint_a": "...", "constraint_b": "..."}}"""


def _kill_child_processes():
    """Kill child processes of this driver. Mirrors executor._kill_child_processes."""
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


def _resolve_model(phase: str, fast: bool) -> str:
    profile = "fast" if fast else "default"
    return PHASE_MODELS.get(phase, {}).get(profile, "sonnet")


def _activity_log(session_dir: str) -> str:
    return os.path.join(session_dir, "planner-activity.log")


def _log_activity(session_dir: str, message: str) -> None:
    log_activity(_activity_log(session_dir), "", message)


def _load_plugins(plugins_file: Path) -> list[dict]:
    if not plugins_file.is_file():
        return []
    try:
        data = json.loads(plugins_file.read_text())
        return data.get("enrichment", [])
    except (json.JSONDecodeError, OSError):
        return []


def _load_system_prompt(prompts_dir: str, template_file: str) -> str:
    """Load a prompt template, stripping YAML frontmatter."""
    path = Path(prompts_dir) / template_file
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    # Strip YAML frontmatter (--- ... ---)
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            text = text[end + 3:].lstrip("\n")
    return text


class PlannerDriver:
    """Deterministic 6-phase planner orchestrator.

    Runs: recon -> architects -> critics -> refiners -> judge -> enrichment
    Each phase: dispatch agents -> wait -> validate evidence -> pass/fail
    """

    def __init__(self, provider: Provider, session_dir: str, prompts_dir: str = "",
                 max_retries: int = MAX_RETRIES, workers: int = 2,
                 repo_dir: str = ""):
        self.provider = provider
        self.session_dir = session_dir
        self.repo_dir = repo_dir or session_dir
        self.prompts_dir = prompts_dir or str(Path(__file__).parent.parent / "prompts")
        self.state_manager = StateManager(Path(session_dir) / ".planner-state.json")
        self.max_retries = max_retries
        self.workers = workers
        self._status_writer: StatusWriter | None = None

    def run(self, problem: str, fast: bool = False,
            tension: str = "", constraint_a: str = "", constraint_b: str = "") -> Path:
        """Run all 6 phases, return path to final-plan.md."""
        self._init_session(problem, fast)
        self._derive_context(problem, tension, constraint_a, constraint_b)
        self._install_signal_handlers()
        self._set_driver_pid()

        try:
            while True:
                state = self.state_manager.load()
                if state.killed:
                    _log_activity(self.session_dir,
                                  f"Stopped: killed ({state.kill_reason or 'no reason'})")
                    break
                phase = state.next_runnable()
                if phase is None:
                    break
                self._dispatch_phase(phase, state)
        finally:
            self._clear_driver_pid()

        return Path(self.session_dir) / "final-plan.md"

    def _install_signal_handlers(self) -> None:
        def _handle(sig, _frame):
            reason = "SIGTERM" if sig == signal.SIGTERM else "SIGINT"
            _log_activity(self.session_dir, f"Signal received: {reason}")
            _kill_child_processes()
            try:
                def mutate(s: PlannerState) -> None:
                    s.killed = True
                    s.kill_reason = reason
                    s.driver_pid = 0
                self.state_manager.update(mutate)
            except Exception:
                pass

        signal.signal(signal.SIGINT, _handle)
        signal.signal(signal.SIGTERM, _handle)

    def _set_driver_pid(self) -> None:
        try:
            def mutate(s: PlannerState) -> None:
                s.driver_pid = os.getpid()
                s.killed = False
                s.kill_reason = ""
            self.state_manager.update(mutate)
        except Exception:
            pass

    def _clear_driver_pid(self) -> None:
        try:
            def mutate(s: PlannerState) -> None:
                s.driver_pid = 0
            self.state_manager.update(mutate)
        except Exception:
            pass

    def _init_session(self, problem: str, fast: bool) -> None:
        os.makedirs(self.session_dir, exist_ok=True)
        slug = Path(self.session_dir).name

        phases = {name: PhaseState() for name in PHASE_ORDER}
        if fast:
            phases["enrichment"] = PhaseState(
                status=PhaseStatus.SKIPPED,
                completed_at=now_iso(),
            )

        state = PlannerState(
            slug=slug,
            session_dir=self.session_dir,
            repo_dir=self.repo_dir,
            fast_mode=fast,
            problem_statement=problem,
            phases=phases,
            created_at=now_iso(),
            updated_at=now_iso(),
        )
        self.state_manager.save(state)
        _log_activity(self.session_dir, f"Session initialized: {slug} (fast={fast})")

    def _derive_context(self, problem: str, tension: str,
                        constraint_a: str, constraint_b: str) -> None:
        if tension and constraint_a and constraint_b:
            def mutate(s: PlannerState) -> None:
                s.core_tension = tension
                s.constraint_a = constraint_a
                s.constraint_b = constraint_b
            self.state_manager.update(mutate)
            return

        prompt = DERIVE_CONTEXT_PROMPT.format(problem=problem)
        raw = self.provider.run_judge(prompt, model="fast")
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            parsed = {}

        derived_tension = tension or parsed.get("tension", "")
        derived_a = constraint_a or parsed.get("constraint_a", "")
        derived_b = constraint_b or parsed.get("constraint_b", "")

        def mutate(s: PlannerState) -> None:
            s.core_tension = derived_tension
            s.constraint_a = derived_a
            s.constraint_b = derived_b

        self.state_manager.update(mutate)
        _log_activity(self.session_dir, "Context derived from problem statement")

    def _dispatch_phase(self, phase: str, state: PlannerState) -> None:
        _log_activity(self.session_dir, f"Phase {phase} -> in_progress")
        phase_start = time.monotonic()

        def mark_running(s: PlannerState) -> None:
            s.phases[phase].status = PhaseStatus.IN_PROGRESS
            s.phases[phase].started_at = now_iso()

        self.state_manager.update(mark_running)

        if phase == "enrichment":
            self._dispatch_enrichment(state)
        else:
            agents = PHASE_AGENTS.get(phase, [])
            if len(agents) > 1 and self.workers > 1:
                with ThreadPoolExecutor(max_workers=self.workers) as pool:
                    futures = {
                        pool.submit(self._run_agent, phase, ad, state): ad
                        for ad in agents
                    }
                    for f in as_completed(futures):
                        f.result()
            else:
                for agent_def in agents:
                    self._run_agent(phase, agent_def, state)

        elapsed = time.monotonic() - phase_start
        elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

        evidence = validate_phase(self.session_dir, phase)
        if evidence.passed:
            def mark_passed(s: PlannerState) -> None:
                s.phases[phase].status = PhaseStatus.COMPLETE
                s.phases[phase].completed_at = now_iso()

            self.state_manager.update(mark_passed)
            _log_activity(self.session_dir,
                          f"Phase {phase} -> complete in {elapsed_str} ({evidence.message})")
        else:
            current = self.state_manager.load()
            retries = current.phases[phase].retries

            def mark_failed(s: PlannerState) -> None:
                s.phases[phase].status = PhaseStatus.FAILED
                s.phases[phase].retries += 1
                s.phases[phase].last_error = evidence.message

            self.state_manager.update(mark_failed)
            _log_activity(self.session_dir,
                          f"Phase {phase} -> failed in {elapsed_str} "
                          f"(retry {retries + 1}/{self.max_retries}): {evidence.message}")

    def _run_agent(self, phase: str, agent_def: dict, state: PlannerState) -> AgentResult:
        model = _resolve_model(phase, state.fast_mode)
        max_turns = PHASE_MAX_TURNS.get(phase, 200)
        prompt = self._build_phase_prompt(phase, agent_def, state)
        step_name = agent_def["name"]
        cwd = state.repo_dir or self.repo_dir

        # Load system prompt and tool restrictions from prompt templates
        agent_type = agent_def.get("agent_type", "")
        system_prompt = ""
        allowed_tools = ""
        if agent_type and agent_type in _AGENT_TYPE_PROMPTS:
            template_file, allowed_tools = _AGENT_TYPE_PROMPTS[agent_type]
            system_prompt = _load_system_prompt(self.prompts_dir, template_file)

        result = self.provider.run_agent(
            prompt=prompt,
            model=model,
            max_turns=max_turns,
            cwd=cwd,
            step_name=step_name,
            session_dir=self.session_dir,
            activity_log_path=_activity_log(self.session_dir),
            system_prompt=system_prompt,
            allowed_tools=allowed_tools,
        )
        if result.timed_out:
            _log_activity(self.session_dir, f"WARNING: agent {step_name} timed out")
        return result

    def _dispatch_enrichment(self, state: PlannerState) -> None:
        plugins_file = Path(__file__).resolve().parent / "config" / "plugins.json"
        plugins = _load_plugins(plugins_file)
        if not plugins:
            return

        model = _resolve_model("enrichment", state.fast_mode)
        max_turns = PHASE_MAX_TURNS.get("enrichment", 200)

        for plugin in plugins:
            skill_name = plugin.get("skill", "")
            output_suffix = plugin.get("output_suffix", skill_name)
            prompt = (
                f"Enrichment task for skill '{skill_name}'.\n"
                f"Session: {self.session_dir}\n"
                f"Problem: {state.problem_statement}\n"
                f"Write output to: {self.session_dir}/{output_suffix}.md"
            )
            self.provider.run_agent(
                prompt=prompt,
                model=model,
                max_turns=max_turns,
                cwd=state.repo_dir or self.repo_dir,
                step_name=f"enrich-{skill_name}",
                session_dir=self.session_dir,
                activity_log_path=_activity_log(self.session_dir),
            )

    def _build_phase_prompt(self, phase: str, agent_def: dict,
                            state: PlannerState) -> str:
        design_id = agent_def.get("id", "")
        sd = self.session_dir

        if phase == "recon":
            return (
                f"## Recon\n\n"
                f"- **Session directory**: {sd}\n"
                f"- **Problem statement**: {state.problem_statement}\n"
                f"- **Write codebase brief to**: {sd}/codebase-brief.md\n\n"
                f"Analyze the codebase and produce a comprehensive brief."
            )

        if phase == "architects":
            constraint = state.constraint_a if design_id == "a" else state.constraint_b
            design_file = f"design-{design_id}.md"
            return (
                f"## Context\n\n"
                f"- **Session directory**: {sd}\n"
                f"- **Problem statement**: {state.problem_statement}\n"
                f"- **Your constraint**: {constraint}\n"
                f"- **Codebase brief**: {sd}/codebase-brief.md\n"
                f"- **Write design to**: {sd}/{design_file}"
            )

        if phase == "critics":
            design_file = f"design-{design_id}.md"
            critique_file = f"critique-{design_id}.md"
            return (
                f"## Context\n\n"
                f"- **Session directory**: {sd}\n"
                f"- **Design file**: {sd}/{design_file}\n"
                f"- **Codebase brief**: {sd}/codebase-brief.md\n"
                f"- **Write critique to**: {sd}/{critique_file}"
            )

        if phase == "refiners":
            design_file = f"design-{design_id}.md"
            critique_file = f"critique-{design_id}.md"
            refined_file = f"refined-{design_id}.md"
            return (
                f"## Context\n\n"
                f"- **Session directory**: {sd}\n"
                f"- **Design file**: {sd}/{design_file}\n"
                f"- **Critique file**: {sd}/{critique_file}\n"
                f"- **Codebase brief**: {sd}/codebase-brief.md\n"
                f"- **Write refined design to**: {sd}/{refined_file}"
            )

        if phase == "judge":
            return (
                f"## Context\n\n"
                f"- **Session directory**: {sd}\n"
                f"- **Problem statement**: {state.problem_statement}\n"
                f"- **Original design A**: {sd}/design-a.md\n"
                f"- **Original design B**: {sd}/design-b.md\n"
                f"- **Refined design A (revisions)**: {sd}/refined-a.md\n"
                f"- **Refined design B (revisions)**: {sd}/refined-b.md\n"
                f"- **Codebase brief**: {sd}/codebase-brief.md\n"
                f"- **Write final plan to**: {sd}/final-plan.md"
            )

        return f"Phase: {phase}\nSession: {sd}\nProblem: {state.problem_statement}"
