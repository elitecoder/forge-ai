# Copyright 2026. Tests for planner driver — deterministic 6-phase orchestrator.

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

from forge.planner.driver import (
    PlannerDriver,
    PHASE_AGENTS,
    _load_enrichment_plugins,
)
from forge.planner.engine.state import (
    PHASE_ORDER,
    PhaseStatus,
    StateManager,
)
from forge.providers.protocol import AgentResult


def _make_provider() -> MagicMock:
    provider = MagicMock()
    provider.run_agent.return_value = AgentResult(
        exit_code=0, stdout="ok", transcript_path="", timed_out=False,
    )
    provider.run_judge.return_value = json.dumps({
        "tension": "speed vs safety",
        "constraint_a": "optimize for speed",
        "constraint_b": "optimize for safety",
    })
    provider.format_subagent_type.return_value = "general-purpose"
    return provider


def _write_evidence(session_dir: str, phase: str) -> None:
    """Write minimal valid evidence files for a given phase."""
    files = {
        "recon": [("codebase-brief.md", 35, None)],
        "architects": [("design-a.md", 25, None), ("design-b.md", 25, None)],
        "critics": [("critique-a.md", 20, None), ("critique-b.md", 20, None)],
        "refiners": [("refined-a.md", 25, None), ("refined-b.md", 25, None)],
        "judge": [("final-plan.md", 35, ["Decision", "Implementation Plan"])],
    }
    for filename, line_count, headings in files.get(phase, []):
        lines = []
        if headings:
            lines.extend(f"# {h}" for h in headings)
        lines.extend(f"content line {i}" for i in range(line_count))
        path = Path(session_dir) / filename
        path.write_text("\n".join(lines))


def _setup_full_evidence(session_dir: str) -> MagicMock:
    """Create a provider that writes evidence files when run_agent is called."""
    provider = _make_provider()
    phase_for_agent = {}
    for phase, agents in PHASE_AGENTS.items():
        for agent_def in agents:
            phase_for_agent[agent_def["name"]] = phase

    def write_on_run(**kwargs):
        step = kwargs.get("step_name", "")
        phase = phase_for_agent.get(step, "")
        if phase:
            _write_evidence(session_dir, phase)
        return AgentResult(exit_code=0, stdout="ok", transcript_path="", timed_out=False)

    provider.run_agent.side_effect = lambda **kwargs: write_on_run(**kwargs)
    return provider


def test_init_session_creates_state(tmp_path):
    session_dir = str(tmp_path / "test-session")
    provider = _make_provider()
    driver = PlannerDriver(provider, session_dir)
    driver._init_session("Add auth", fast=False)

    state_file = Path(session_dir) / ".planner-state.json"
    assert state_file.is_file()

    mgr = StateManager(state_file)
    state = mgr.load()
    assert state.slug == "test-session"
    assert state.problem_statement == "Add auth"
    assert len(state.phases) == len(PHASE_ORDER)
    for phase in PHASE_ORDER:
        assert phase in state.phases


def test_init_session_fast_skips_enrichment(tmp_path):
    session_dir = str(tmp_path / "fast-session")
    provider = _make_provider()
    driver = PlannerDriver(provider, session_dir)
    driver._init_session("Add auth", fast=True)

    mgr = StateManager(Path(session_dir) / ".planner-state.json")
    state = mgr.load()
    assert state.fast_mode is True
    assert state.phases["enrichment"].status == PhaseStatus.SKIPPED
    assert state.phases["recon"].status == PhaseStatus.PENDING


def test_derive_context_uses_overrides(tmp_path):
    session_dir = str(tmp_path / "ctx-session")
    provider = _make_provider()
    driver = PlannerDriver(provider, session_dir)
    driver._init_session("problem", fast=False)

    driver._derive_context("problem", "my tension", "cons a", "cons b")

    provider.run_judge.assert_not_called()
    state = driver.state_manager.load()
    assert state.core_tension == "my tension"
    assert state.constraint_a == "cons a"
    assert state.constraint_b == "cons b"


def test_derive_context_calls_llm_when_no_overrides(tmp_path):
    session_dir = str(tmp_path / "llm-session")
    provider = _make_provider()
    driver = PlannerDriver(provider, session_dir)
    driver._init_session("problem", fast=False)

    driver._derive_context("problem", "", "", "")

    provider.run_judge.assert_called_once()
    state = driver.state_manager.load()
    assert state.core_tension == "speed vs safety"
    assert state.constraint_a == "optimize for speed"


def test_dispatch_phase_calls_correct_agent_count(tmp_path):
    session_dir = str(tmp_path / "dispatch-session")
    provider = _make_provider()
    driver = PlannerDriver(provider, session_dir)
    driver._init_session("problem", fast=False)

    def mutate(s):
        s.core_tension = "t"
        s.constraint_a = "a"
        s.constraint_b = "b"
    driver.state_manager.update(mutate)

    # Write evidence so recon passes
    _write_evidence(session_dir, "recon")
    state = driver.state_manager.load()
    driver._dispatch_phase("recon", state)
    assert provider.run_agent.call_count == 1

    # Mark recon complete so architects can run
    def mark_recon(s):
        s.phases["recon"].status = PhaseStatus.COMPLETE
    driver.state_manager.update(mark_recon)

    provider.run_agent.reset_mock()
    _write_evidence(session_dir, "architects")
    state = driver.state_manager.load()
    driver._dispatch_phase("architects", state)
    assert provider.run_agent.call_count == 2

    provider.run_agent.reset_mock()
    _write_evidence(session_dir, "critics")
    state = driver.state_manager.load()
    driver._dispatch_phase("critics", state)
    assert provider.run_agent.call_count == 2

    provider.run_agent.reset_mock()
    _write_evidence(session_dir, "refiners")
    state = driver.state_manager.load()
    driver._dispatch_phase("refiners", state)
    assert provider.run_agent.call_count == 2

    provider.run_agent.reset_mock()
    _write_evidence(session_dir, "judge")
    state = driver.state_manager.load()
    driver._dispatch_phase("judge", state)
    assert provider.run_agent.call_count == 1


def test_full_six_phase_dispatch(tmp_path):
    session_dir = str(tmp_path / "full-session")
    provider = _setup_full_evidence(session_dir)

    driver = PlannerDriver(provider, session_dir)
    result = driver.run(
        "Add authentication",
        fast=False,
        tension="security vs UX",
        constraint_a="OAuth2",
        constraint_b="magic links",
    )

    assert result == Path(session_dir) / "final-plan.md"

    state = driver.state_manager.load()
    for phase in PHASE_ORDER:
        if phase == "enrichment":
            # No enrichment plugins configured, so enrichment passes trivially
            assert state.phases[phase].status in (
                PhaseStatus.COMPLETE, PhaseStatus.SKIPPED,
            )
        else:
            assert state.phases[phase].status == PhaseStatus.COMPLETE, (
                f"{phase} should be COMPLETE but is {state.phases[phase].status}"
            )

    # Verify correct total agent calls:
    # recon=1, architects=2, critics=2, refiners=2, judge=1 = 8
    assert provider.run_agent.call_count == 8
    provider.run_judge.assert_not_called()


def test_fast_mode_skips_enrichment(tmp_path):
    session_dir = str(tmp_path / "fast-full")
    provider = _setup_full_evidence(session_dir)

    driver = PlannerDriver(provider, session_dir)
    driver.run(
        "problem",
        fast=True,
        tension="t",
        constraint_a="a",
        constraint_b="b",
    )

    state = driver.state_manager.load()
    assert state.phases["enrichment"].status == PhaseStatus.SKIPPED
    assert state.fast_mode is True


def test_retry_on_evidence_failure(tmp_path):
    session_dir = str(tmp_path / "retry-session")
    provider = _make_provider()
    driver = PlannerDriver(provider, session_dir, max_retries=2)
    driver._init_session("problem", fast=False)

    def set_context(s):
        s.core_tension = "t"
        s.constraint_a = "a"
        s.constraint_b = "b"
    driver.state_manager.update(set_context)

    # First dispatch: no evidence files -> fails
    state = driver.state_manager.load()
    driver._dispatch_phase("recon", state)

    state = driver.state_manager.load()
    assert state.phases["recon"].status == PhaseStatus.FAILED
    assert state.phases["recon"].retries == 1

    # Second dispatch: still no evidence -> fails again
    state = driver.state_manager.load()
    driver._dispatch_phase("recon", state)

    state = driver.state_manager.load()
    assert state.phases["recon"].status == PhaseStatus.FAILED
    assert state.phases["recon"].retries == 2


def test_state_transitions_throughout(tmp_path):
    session_dir = str(tmp_path / "transitions")
    provider = _make_provider()
    driver = PlannerDriver(provider, session_dir)
    driver._init_session("problem", fast=False)

    def set_context(s):
        s.core_tension = "t"
        s.constraint_a = "a"
        s.constraint_b = "b"
    driver.state_manager.update(set_context)

    # All phases start as PENDING (except enrichment which has no special treatment here)
    state = driver.state_manager.load()
    for phase in PHASE_ORDER:
        assert state.phases[phase].status == PhaseStatus.PENDING

    # After dispatching recon (with evidence), it goes to COMPLETE
    _write_evidence(session_dir, "recon")
    state = driver.state_manager.load()
    driver._dispatch_phase("recon", state)

    state = driver.state_manager.load()
    assert state.phases["recon"].status == PhaseStatus.COMPLETE
    assert state.phases["recon"].completed_at != ""
    assert state.phases["architects"].status == PhaseStatus.PENDING


def test_build_phase_prompt_includes_context(tmp_path):
    session_dir = str(tmp_path / "prompt-session")
    provider = _make_provider()
    driver = PlannerDriver(provider, session_dir)
    driver._init_session("Add caching layer", fast=False)

    def set_context(s):
        s.core_tension = "t"
        s.constraint_a = "redis"
        s.constraint_b = "memcached"
    driver.state_manager.update(set_context)

    state = driver.state_manager.load()

    # Recon prompt
    prompt = driver._build_phase_prompt("recon", PHASE_AGENTS["recon"][0], state)
    assert "Add caching layer" in prompt
    assert "codebase-brief.md" in prompt

    # Architect A prompt
    prompt = driver._build_phase_prompt(
        "architects", PHASE_AGENTS["architects"][0], state,
    )
    assert "redis" in prompt
    assert "design-a.md" in prompt

    # Architect B prompt
    prompt = driver._build_phase_prompt(
        "architects", PHASE_AGENTS["architects"][1], state,
    )
    assert "memcached" in prompt
    assert "design-b.md" in prompt

    # Judge prompt
    prompt = driver._build_phase_prompt("judge", PHASE_AGENTS["judge"][0], state)
    assert "final-plan.md" in prompt
    assert "refined-a.md" in prompt
    assert "refined-b.md" in prompt


def test_run_agent_passes_correct_kwargs(tmp_path):
    session_dir = str(tmp_path / "kwargs-session")
    provider = _make_provider()
    driver = PlannerDriver(provider, session_dir)
    driver._init_session("problem", fast=False)

    def set_context(s):
        s.core_tension = "t"
        s.constraint_a = "a"
        s.constraint_b = "b"
    driver.state_manager.update(set_context)

    _write_evidence(session_dir, "recon")
    state = driver.state_manager.load()
    agent_def = PHASE_AGENTS["recon"][0]
    driver._run_agent("recon", agent_def, state)

    call_kwargs = provider.run_agent.call_args.kwargs
    assert call_kwargs["model"] == "sonnet"
    assert call_kwargs["max_turns"] == 200
    assert call_kwargs["cwd"] == session_dir
    assert call_kwargs["step_name"] == "recon"
    assert call_kwargs["session_dir"] == session_dir


def test_derive_context_partial_overrides(tmp_path):
    """When only some context values provided, LLM fills the rest."""
    session_dir = str(tmp_path / "partial-session")
    provider = _make_provider()
    driver = PlannerDriver(provider, session_dir)
    driver._init_session("problem", fast=False)

    driver._derive_context("problem", "my tension", "", "")

    provider.run_judge.assert_called_once()
    state = driver.state_manager.load()
    assert state.core_tension == "my tension"
    assert state.constraint_a == "optimize for speed"
    assert state.constraint_b == "optimize for safety"


# ── Enrichment from preset manifest ──────────────────────────────────────


def _create_preset_manifest(preset_dir: Path, enrichment: list[dict]) -> None:
    """Write a minimal preset manifest with enrichment config."""
    preset_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "preset": "test-preset",
        "version": 3,
        "description": "test",
        "enrichment": enrichment,
        "pipelines": {},
        "steps": {},
        "models": {},
    }
    (preset_dir / "manifest.json").write_text(json.dumps(manifest))


def test_load_enrichment_plugins_from_preset(tmp_path):
    preset_dir = tmp_path / "presets" / "hz-web"
    plugins = [
        {"skill": "sq-unit-test-planner", "output_suffix": "unit-test-plan"},
        {"skill": "sq-visual-test-planner", "output_suffix": "visual-test-plan"},
    ]
    _create_preset_manifest(preset_dir, plugins)

    loaded = _load_enrichment_plugins(str(preset_dir))
    assert len(loaded) == 2
    assert loaded[0]["skill"] == "sq-unit-test-planner"
    assert loaded[1]["output_suffix"] == "visual-test-plan"


def test_load_enrichment_plugins_empty_preset_path():
    assert _load_enrichment_plugins("") == []


def test_load_enrichment_plugins_no_manifest(tmp_path):
    assert _load_enrichment_plugins(str(tmp_path / "nonexistent")) == []


def test_load_enrichment_plugins_no_enrichment_key(tmp_path):
    preset_dir = tmp_path / "preset"
    preset_dir.mkdir()
    (preset_dir / "manifest.json").write_text('{"preset": "x"}')
    assert _load_enrichment_plugins(str(preset_dir)) == []


def test_init_session_stores_preset(tmp_path):
    session_dir = str(tmp_path / "preset-session")
    provider = _make_provider()
    driver = PlannerDriver(
        provider, session_dir,
        preset="hz-web", preset_path="/presets/hz-web",
    )
    driver._init_session("problem", fast=False)

    state = driver.state_manager.load()
    assert state.preset == "hz-web"
    assert state.preset_path == "/presets/hz-web"


def test_dispatch_enrichment_with_preset_plugins(tmp_path):
    preset_dir = tmp_path / "presets" / "hz-web"
    plugins = [
        {"skill": "sq-unit-test-planner", "output_suffix": "unit-test-plan"},
        {"skill": "sq-visual-test-planner", "output_suffix": "visual-test-plan"},
    ]
    _create_preset_manifest(preset_dir, plugins)

    session_dir = str(tmp_path / "enrich-session")
    provider = _make_provider()
    driver = PlannerDriver(
        provider, session_dir,
        preset="hz-web", preset_path=str(preset_dir),
    )
    driver._init_session("problem", fast=False)

    # Complete all phases up to enrichment
    def complete_prior(s):
        for phase in PHASE_ORDER:
            if phase == "enrichment":
                break
            s.phases[phase].status = PhaseStatus.COMPLETE
    driver.state_manager.update(complete_prior)

    state = driver.state_manager.load()
    driver._dispatch_enrichment(state)

    assert provider.run_agent.call_count == 2
    call_names = [c.kwargs["step_name"] for c in provider.run_agent.call_args_list]
    assert "enrich-sq-unit-test-planner" in call_names
    assert "enrich-sq-visual-test-planner" in call_names


def test_dispatch_enrichment_without_preset(tmp_path):
    session_dir = str(tmp_path / "no-preset-session")
    provider = _make_provider()
    driver = PlannerDriver(provider, session_dir)
    driver._init_session("problem", fast=False)

    state = driver.state_manager.load()
    driver._dispatch_enrichment(state)

    provider.run_agent.assert_not_called()
