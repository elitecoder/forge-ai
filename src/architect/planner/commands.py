# Copyright 2026. Planner CLI — state management, dispatch config, evidence validation.

import argparse
import json
import os
import sys
from pathlib import Path

from architect.planner.engine.state import (
    StateManager, PlannerState, PhaseState, PhaseStatus,
    PHASE_ORDER, TERMINAL_STATUSES, now_iso, state_to_dict,
)
from architect.core.logging import log_activity as _core_log
from architect.core.session import (
    SESSIONS_BASE as _CORE_SESSIONS_BASE,
    create_session,
    find_active_session as _core_find_session,
    generate_slug,
    list_sessions as _core_list_sessions,
    cleanup_sessions as _core_cleanup_sessions,
)
from architect.planner.engine.evidence import validate_phase

SESSIONS_BASE = _CORE_SESSIONS_BASE / "planner"


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
PLUGINS_FILE = Path(__file__).resolve().parent / "config" / "plugins.json"
SKILLS_BASE = Path(os.environ.get("ARCHITECT_SKILLS_BASE", str(Path.home() / ".claude" / "skills")))

MAX_RETRIES = 2

# ── Model resolution ────────────────────────────────────────────────────────

PHASE_MODELS = {
    "recon":       {"default": "sonnet", "fast": "sonnet"},
    "architects":  {"default": "opus",   "fast": "sonnet"},
    "critics":     {"default": "opus",   "fast": "sonnet"},
    "refiners":    {"default": "sonnet", "fast": "sonnet"},
    "judge":       {"default": "opus",   "fast": "sonnet"},
    "enrichment":  {"default": "sonnet", "fast": "sonnet"},
}

PHASE_MODEL_FALLBACKS = {
    "architects": "sonnet",
    "critics": "sonnet",
    "judge": "sonnet",
}

# ── Turn limits ────────────────────────────────────────────────────────────

PHASE_MAX_TURNS = {
    "recon": 200,
    "architects": 200,
    "critics": 200,
    "refiners": 200,
    "judge": 200,
    "enrichment": 200,
}

# ── Agent definitions per phase ─────────────────────────────────────────────

PHASE_AGENTS = {
    "recon": [
        {"name": "recon", "template": "recon.md", "output": "codebase-brief.md"},
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


# ── Helpers ─────────────────────────────────────────────────────────────────

def _state_file(session_dir: str) -> Path:
    return Path(session_dir) / ".planner-state.json"


def _activity_log(session_dir: str) -> str:
    return os.path.join(session_dir, "planner-activity.log")


def _log_activity(session_dir: str, message: str) -> None:
    _core_log(_activity_log(session_dir), "", message)


def _resolve_model(phase: str, fast_mode: bool) -> str:
    profile = "fast" if fast_mode else "default"
    return PHASE_MODELS.get(phase, {}).get(profile, "sonnet")


def _get_fallback_model(phase: str) -> str | None:
    return PHASE_MODEL_FALLBACKS.get(phase)


def _resolve_max_turns(phase: str) -> int:
    return PHASE_MAX_TURNS.get(phase, 15)


def _read_template(template_name: str) -> str:
    path = PROMPTS_DIR / template_name
    if not path.is_file():
        print(f"ERROR: Prompt template not found: {path}", file=sys.stderr)
        sys.exit(1)
    return path.read_text(encoding="utf-8")


def _substitute_prompt(template: str, state: PlannerState, agent: dict) -> str:
    design_id = agent.get("id", "")
    subs = {
        "{{SESSION_DIR}}": state.session_dir,
        "{{PROBLEM_STATEMENT}}": state.problem_statement,
    }
    if design_id == "a":
        subs.update({
            "{{CONSTRAINT}}": state.constraint_a,
            "{{DESIGN_FILE}}": "design-a.md",
            "{{CRITIQUE_FILE}}": "critique-a.md",
            "{{REFINED_FILE}}": "refined-a.md",
            "{{DESIGN_ID}}": "a",
        })
    elif design_id == "b":
        subs.update({
            "{{CONSTRAINT}}": state.constraint_b,
            "{{DESIGN_FILE}}": "design-b.md",
            "{{CRITIQUE_FILE}}": "critique-b.md",
            "{{REFINED_FILE}}": "refined-b.md",
            "{{DESIGN_ID}}": "b",
        })

    result = template
    for key, val in subs.items():
        result = result.replace(key, val)
    return result


# ── Context builders for specialized agents ─────────────────────────────────

def _build_architect_context(state: PlannerState, agent_def: dict) -> str:
    design_id = agent_def["id"]
    constraint = state.constraint_a if design_id == "a" else state.constraint_b
    design_file = f"design-{design_id}.md"
    return f"""## Context

- **Session directory**: {state.session_dir}
- **Problem statement**: {state.problem_statement}
- **Your constraint**: {constraint}
- **Codebase brief**: {state.session_dir}/codebase-brief.md
- **Write design to**: {state.session_dir}/{design_file}"""


def _build_critic_context(state: PlannerState, agent_def: dict) -> str:
    design_id = agent_def["id"]
    design_file = f"design-{design_id}.md"
    critique_file = f"critique-{design_id}.md"
    return f"""## Context

- **Session directory**: {state.session_dir}
- **Design file**: {state.session_dir}/{design_file}
- **Codebase brief**: {state.session_dir}/codebase-brief.md
- **Write critique to**: {state.session_dir}/{critique_file}"""


def _build_refiner_context(state: PlannerState, agent_def: dict) -> str:
    design_id = agent_def["id"]
    design_file = f"design-{design_id}.md"
    critique_file = f"critique-{design_id}.md"
    refined_file = f"refined-{design_id}.md"
    return f"""## Context

- **Session directory**: {state.session_dir}
- **Design file**: {state.session_dir}/{design_file}
- **Critique file**: {state.session_dir}/{critique_file}
- **Codebase brief**: {state.session_dir}/codebase-brief.md
- **Write refined design to**: {state.session_dir}/{refined_file}"""


def _build_judge_context(state: PlannerState, agent_def: dict) -> str:
    return f"""## Context

- **Session directory**: {state.session_dir}
- **Problem statement**: {state.problem_statement}
- **Original design A**: {state.session_dir}/design-a.md
- **Original design B**: {state.session_dir}/design-b.md
- **Refined design A (revisions)**: {state.session_dir}/refined-a.md
- **Refined design B (revisions)**: {state.session_dir}/refined-b.md
- **Codebase brief**: {state.session_dir}/codebase-brief.md
- **Write final plan to**: {state.session_dir}/final-plan.md"""


_CONTEXT_BUILDERS = {
    "planner-architect-topdown": _build_architect_context,
    "planner-architect-bottomup": _build_architect_context,
    "planner-critic": _build_critic_context,
    "planner-refiner": _build_refiner_context,
    "planner-judge": _build_judge_context,
}


def _load_plugins() -> list[dict]:
    if not PLUGINS_FILE.is_file():
        return []
    try:
        data = json.loads(PLUGINS_FILE.read_text())
        return data.get("enrichment", [])
    except (json.JSONDecodeError, OSError):
        return []


def _find_active_session() -> Path | None:
    return _core_find_session(SESSIONS_BASE, ".planner-state.json")


def _require_session() -> tuple[StateManager, PlannerState]:
    session = _find_active_session()
    if not session:
        print("ERROR: No active planner session. Run: architect plan init <slug>", file=sys.stderr)
        sys.exit(1)
    mgr = StateManager(_state_file(str(session)))
    return mgr, mgr.load()


# ── Commands ────────────────────────────────────────────────────────────────

def cmd_init(args):
    slug = args.slug
    session = create_session("planner", slug=slug)
    session_dir = str(session)

    phases = {name: PhaseState() for name in PHASE_ORDER}
    state = PlannerState(
        slug=slug,
        session_dir=session_dir,
        fast_mode=args.fast,
        phases=phases,
        created_at=now_iso(),
        updated_at=now_iso(),
    )

    mgr = StateManager(_state_file(session_dir))
    mgr.save(state)

    _log_activity(session_dir, f"Session initialized: {slug} (fast={args.fast})")

    mode = "fast" if args.fast else "default"
    print(f"Session initialized: {slug} ({mode} mode)")
    print(f"  Architects: {_resolve_model('architects', args.fast)}")
    print(f"  Critics: {_resolve_model('critics', args.fast)}")
    print(f"  Refiners: {_resolve_model('refiners', args.fast)}")
    print(f"  Judge: {_resolve_model('judge', args.fast)}")
    print(f"Session directory: {session_dir}")
    print(session_dir)


def cmd_set_context(args):
    mgr, _ = _require_session()

    def mutate(s: PlannerState):
        if args.problem:
            s.problem_statement = args.problem
        if args.tension:
            s.core_tension = args.tension
        if args.constraint_a:
            s.constraint_a = args.constraint_a
        if args.constraint_b:
            s.constraint_b = args.constraint_b

    state = mgr.update(mutate)
    _log_activity(state.session_dir, "Problem context set")
    print("Context updated:")
    print(f"  Problem: {state.problem_statement[:80]}...")
    print(f"  Tension: {state.core_tension}")
    print(f"  Constraint A: {state.constraint_a[:60]}")
    print(f"  Constraint B: {state.constraint_b[:60]}")


def cmd_status(args):
    _, state = _require_session()
    print(json.dumps(state_to_dict(state), indent=2))


def cmd_next(args):
    _, state = _require_session()
    phase = state.next_runnable()

    if not phase:
        print(json.dumps({"phase": None, "message": "All phases complete"}))
        return

    ps = state.phases.get(phase, PhaseState())
    print(json.dumps({
        "phase": phase,
        "status": ps.status.value,
        "retries": ps.retries,
    }))


def cmd_dispatch(args):
    _, state = _require_session()
    phase = args.phase

    if phase not in PHASE_ORDER:
        print(f"ERROR: Unknown phase: {phase}", file=sys.stderr)
        sys.exit(1)

    model = args.model if hasattr(args, 'model') and args.model else _resolve_model(phase, state.fast_mode)
    fallback_model = _get_fallback_model(phase)
    max_turns = _resolve_max_turns(phase)

    if phase == "enrichment":
        agents = _dispatch_enrichment(state, max_turns)
        if not agents:
            print(json.dumps({"phase": "enrichment", "agents": [], "skip": True,
                             "message": "No enrichment plugins available"}))
            return
        print(json.dumps({"phase": "enrichment", "agents": agents}, indent=2))
        return

    agent_defs = PHASE_AGENTS.get(phase, [])
    agents = []
    for agent_def in agent_defs:
        agent_type = agent_def.get("agent_type")

        if agent_type:
            builder = _CONTEXT_BUILDERS[agent_type]
            prompt = builder(state, agent_def)
        else:
            template = _read_template(agent_def["template"])
            prompt = _substitute_prompt(template, state, agent_def)
            agent_type = "general-purpose"

        # Provider layer handles namespacing — no prefix needed
        prefixed_type = agent_type
        agents.append({
            "name": agent_def["name"],
            "model": model,
            "subagent_type": prefixed_type,
            "prompt": prompt,
            "max_turns": max_turns,
            "output_file": agent_def["output"],
        })

    result = {"phase": phase, "agents": agents}
    if fallback_model:
        result["fallback_model"] = fallback_model
    print(json.dumps(result, indent=2))


def _dispatch_enrichment(state: PlannerState, max_turns: int) -> list[dict]:
    plugins = _load_plugins()
    agents = []

    for plugin in plugins:
        skill_name = plugin.get("skill", "")
        skill_dir = SKILLS_BASE / skill_name
        if not (skill_dir / "SKILL.md").is_file():
            continue

        plugin_model = "sonnet" if state.fast_mode else plugin.get("model", "sonnet")
        output_suffix = plugin.get("output_suffix", skill_name)

        template = _read_template("enrichment.md")
        prompt = _substitute_prompt(template, state, {"id": "", "output": f"{output_suffix}.md"})
        prompt = prompt.replace("{{SKILL_NAME}}", skill_name)
        prompt = prompt.replace("{{OUTPUT_SUFFIX}}", output_suffix)

        agents.append({
            "name": f"enrich-{skill_name}",
            "model": plugin_model,
            "subagent_type": "general-purpose",
            "prompt": prompt,
            "max_turns": max_turns,
            "output_file": f"{output_suffix}.md",
        })

    return agents


def cmd_run(args):
    mgr, state = _require_session()
    phase = args.phase
    if phase not in PHASE_ORDER:
        print(f"ERROR: Unknown phase: {phase}", file=sys.stderr)
        sys.exit(1)

    def mutate(s: PlannerState):
        s.phases[phase].status = PhaseStatus.IN_PROGRESS
        s.phases[phase].started_at = now_iso()

    mgr.update(mutate)
    _log_activity(state.session_dir, f"Phase: {phase} → in_progress")
    print(f"Phase '{phase}' → in_progress")


def cmd_pass(args):
    mgr, state = _require_session()
    phase = args.phase
    if phase not in PHASE_ORDER:
        print(f"ERROR: Unknown phase: {phase}", file=sys.stderr)
        sys.exit(1)

    result = validate_phase(state.session_dir, phase)
    if not result.passed:
        print(f"ERROR: Evidence validation failed for '{phase}': {result.message}", file=sys.stderr)
        sys.exit(1)

    print(f"Evidence OK: {result.message}")

    def mutate(s: PlannerState):
        s.phases[phase].status = PhaseStatus.COMPLETE
        s.phases[phase].completed_at = now_iso()

    mgr.update(mutate)
    _log_activity(state.session_dir, f"Phase: {phase} → complete ({result.message})")
    print(f"Phase '{phase}' → complete")


def cmd_fail(args):
    mgr, state = _require_session()
    phase = args.phase
    if phase not in PHASE_ORDER:
        print(f"ERROR: Unknown phase: {phase}", file=sys.stderr)
        sys.exit(1)

    def mutate(s: PlannerState):
        s.phases[phase].retries += 1
        s.phases[phase].status = PhaseStatus.FAILED
        s.phases[phase].last_error = args.error or ""

    updated = mgr.update(mutate)
    retries = updated.phases[phase].retries
    _log_activity(state.session_dir, f"Phase: {phase} → failed (retry {retries}/{MAX_RETRIES})")
    print(f"Phase '{phase}' → failed (retry {retries}/{MAX_RETRIES})")
    if retries >= MAX_RETRIES:
        print(f"WARNING: Phase '{phase}' has reached max retries ({MAX_RETRIES}). Will not retry.")


def cmd_skip(args):
    mgr, state = _require_session()
    phase = args.phase
    if phase not in PHASE_ORDER:
        print(f"ERROR: Unknown phase: {phase}", file=sys.stderr)
        sys.exit(1)

    def mutate(s: PlannerState):
        s.phases[phase].status = PhaseStatus.SKIPPED
        s.phases[phase].completed_at = now_iso()

    mgr.update(mutate)
    _log_activity(state.session_dir, f"Phase: {phase} → skipped")
    print(f"Phase '{phase}' → skipped")


def cmd_reset(args):
    mgr, state = _require_session()
    phase = args.phase
    if phase not in PHASE_ORDER:
        print(f"ERROR: Unknown phase: {phase}", file=sys.stderr)
        sys.exit(1)

    def mutate(s: PlannerState):
        s.phases[phase] = PhaseState()

    mgr.update(mutate)
    _log_activity(state.session_dir, f"Phase: {phase} → reset")
    print(f"Phase '{phase}' → reset to pending")


def cmd_log(args):
    _, state = _require_session()
    _log_activity(state.session_dir, args.message)
    print(f"Logged: {args.message}")


def cmd_cleanup(args):
    removed = _core_cleanup_sessions(SESSIONS_BASE, args.older_than)
    if removed:
        print(f"Removed {len(removed)} session(s):")
        for p in removed:
            print(f"  {p}")
    else:
        print("No sessions to clean up.")


def cmd_sessions(args):
    sessions = _core_list_sessions(SESSIONS_BASE, state_filename=".planner-state.json")
    if not sessions:
        print("No planner sessions found.")
        return
    for s in sessions:
        status = "active" if s["active"] else "done"
        print(f"  {s['name']}  ({s['age_days']}d old, {status})  {s['path']}")


def cmd_complete(args):
    _, state = _require_session()

    required_phases = ["recon", "architects", "critics", "refiners", "judge"]
    incomplete_phases = []

    for phase in required_phases:
        ps = state.phases.get(phase, PhaseState())
        if ps.status != PhaseStatus.COMPLETE:
            incomplete_phases.append(phase)

    if incomplete_phases:
        result = {
            "complete": False,
            "reason": "incomplete_phases",
            "missing_phases": incomplete_phases
        }
        print(json.dumps(result))
        sys.exit(1)

    final_plan = Path(state.session_dir) / "final-plan.md"
    if not final_plan.exists():
        result = {
            "complete": False,
            "reason": "missing_final_plan",
            "message": "final-plan.md does not exist"
        }
        print(json.dumps(result))
        sys.exit(1)

    file_size = final_plan.stat().st_size
    if file_size < 1000:
        result = {
            "complete": False,
            "reason": "final_plan_too_small",
            "message": f"final-plan.md is only {file_size} bytes (expected >1000)",
            "file_size": file_size
        }
        print(json.dumps(result))
        sys.exit(1)

    result = {
        "complete": True,
        "session_dir": state.session_dir,
        "file_size": file_size
    }
    print(json.dumps(result))
    sys.exit(0)


def cmd_drive(args):
    from architect.planner.driver import PlannerDriver
    from architect.providers.claude import ClaudeProvider

    if args.resume:
        sd = _find_active_session()
        if sd is None:
            print("ERROR: No active planner session to resume.", file=sys.stderr)
            sys.exit(1)
        session_dir = str(sd)
        mgr = StateManager(_state_file(session_dir))
        state = mgr.load()

        def unflag(s: PlannerState) -> None:
            s.killed = False
            s.kill_reason = ""
        mgr.update(unflag)

        provider = ClaudeProvider()
        repo_dir = state.repo_dir or os.getcwd()
        driver = PlannerDriver(
            provider, session_dir, repo_dir=repo_dir,
            workers=args.workers,
        )
        driver.state_manager = mgr

        driver._install_signal_handlers()
        driver._set_driver_pid()
        try:
            while True:
                st = driver.state_manager.load()
                if st.killed:
                    _log_activity(session_dir, f"Stopped: killed ({st.kill_reason})")
                    break
                phase = st.next_runnable()
                if phase is None:
                    break
                driver._dispatch_phase(phase, st)
        finally:
            driver._clear_driver_pid()
        print(f"Session: {session_dir}")
        return

    if not args.problem:
        print("ERROR: --problem is required (or use --resume)", file=sys.stderr)
        sys.exit(1)

    slug = args.slug or generate_slug(args.problem, fallback="plan")
    session = create_session("planner", slug=slug)
    session_dir = str(session)
    repo_dir = args.repo or os.getcwd()

    provider = ClaudeProvider()
    driver = PlannerDriver(
        provider, session_dir, repo_dir=repo_dir,
        workers=args.workers,
    )
    result = driver.run(
        args.problem, fast=args.fast,
        tension=args.tension or "",
        constraint_a=args.constraint_a or "",
        constraint_b=args.constraint_b or "",
    )
    print(f"Plan: {result}")
    print(f"Session: {session_dir}")


def cmd_kill(args):
    sd = _find_active_session()
    if sd is None:
        print("ERROR: No active planner session.", file=sys.stderr)
        sys.exit(1)
    mgr = StateManager(_state_file(str(sd)))
    reason = args.reason or ""

    def mutate(s: PlannerState) -> None:
        s.killed = True
        s.kill_reason = reason
    mgr.update(mutate)
    print(f"Planner killed{': ' + reason if reason else ''}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="architect-planner",
        description="Planner — adversarial dual-architecture design system",
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("init", help="Create a new planner session")
    p.add_argument("slug", help="Semantic slug for the session (kebab-case)")
    p.add_argument("--fast", action="store_true", help="Use Sonnet for all agents (faster, cheaper)")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("set-context", help="Store problem statement and constraints")
    p.add_argument("--problem", help="The full problem statement")
    p.add_argument("--tension", help="Core tension description")
    p.add_argument("--constraint-a", help="Constraint A optimization target")
    p.add_argument("--constraint-b", help="Constraint B optimization target")
    p.set_defaults(func=cmd_set_context)

    p = sub.add_parser("status", help="Print current state as JSON")
    p.set_defaults(func=cmd_status)

    sub.add_parser("next", help="Get next runnable phase").set_defaults(func=cmd_next)

    p = sub.add_parser("dispatch", help="Get dispatch config for a phase")
    p.add_argument("phase", choices=PHASE_ORDER)
    p.add_argument("--model", choices=["sonnet", "opus", "haiku"], help="Override model (for fallback)")
    p.set_defaults(func=cmd_dispatch)

    p = sub.add_parser("run", help="Mark phase as in_progress")
    p.add_argument("phase", choices=PHASE_ORDER)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("pass", help="Validate evidence and mark phase complete")
    p.add_argument("phase", choices=PHASE_ORDER)
    p.set_defaults(func=cmd_pass)

    p = sub.add_parser("fail", help="Mark phase as failed")
    p.add_argument("phase", choices=PHASE_ORDER)
    p.add_argument("error", nargs="?", default="")
    p.set_defaults(func=cmd_fail)

    p = sub.add_parser("skip", help="Mark phase as skipped")
    p.add_argument("phase", choices=PHASE_ORDER)
    p.set_defaults(func=cmd_skip)

    p = sub.add_parser("reset", help="Reset phase to pending")
    p.add_argument("phase", choices=PHASE_ORDER)
    p.set_defaults(func=cmd_reset)

    p = sub.add_parser("log", help="Append to activity log")
    p.add_argument("message")
    p.set_defaults(func=cmd_log)

    p = sub.add_parser("cleanup", help="Remove old sessions")
    p.add_argument("--older-than", type=int, default=30, help="Days threshold (default: 30)")
    p.set_defaults(func=cmd_cleanup)

    sub.add_parser("sessions", help="List all planner sessions").set_defaults(func=cmd_sessions)

    sub.add_parser("complete", help="Verify all phases complete and final-plan.md exists").set_defaults(func=cmd_complete)

    p = sub.add_parser("drive", help="Run the full 6-phase planner (plan → final-plan.md)")
    p.add_argument("--problem", help="The full problem statement")
    p.add_argument("--repo", help="Path to the repository (default: cwd)")
    p.add_argument("--slug", help="Session name slug (kebab-case)")
    p.add_argument("--fast", action="store_true", help="Use Sonnet for all agents")
    p.add_argument("--resume", action="store_true", help="Resume the most recent session")
    p.add_argument("--tension", help="Core tension (optional)")
    p.add_argument("--constraint-a", help="Constraint A (optional)")
    p.add_argument("--constraint-b", help="Constraint B (optional)")
    p.add_argument("--workers", type=int, default=2, help="Parallel workers per phase (default: 2)")
    p.set_defaults(func=cmd_drive)

    p = sub.add_parser("kill", help="Graceful planner termination")
    p.add_argument("--reason", default="", help="Reason for killing")
    p.set_defaults(func=cmd_kill)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)
