# Copyright 2026. Tests for planner state machine.

import json
from pathlib import Path

from forge.planner.engine.state import (
    PHASE_ORDER,
    PhaseState,
    PhaseStatus,
    PlannerState,
    StateManager,
    dict_to_state,
    state_to_dict,
)


def test_state_round_trip():
    state = PlannerState(
        slug="test-session",
        session_dir="/tmp/test",
        fast_mode=True,
        problem_statement="Add auth",
        core_tension="security vs UX",
        constraint_a="OAuth2",
        constraint_b="magic links",
        phases={
            "recon": PhaseState(status=PhaseStatus.COMPLETE, retries=0),
            "architects": PhaseState(status=PhaseStatus.IN_PROGRESS, retries=1),
        },
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:01:00Z",
    )
    d = state_to_dict(state)
    restored = dict_to_state(d)
    assert restored.slug == "test-session"
    assert restored.fast_mode is True
    assert restored.phases["recon"].status == PhaseStatus.COMPLETE
    assert restored.phases["architects"].retries == 1


def test_state_round_trip_json(tmp_path):
    state = PlannerState(slug="json-test", phases={"recon": PhaseState(status=PhaseStatus.PENDING)})
    d = state_to_dict(state)
    path = tmp_path / "state.json"
    path.write_text(json.dumps(d))
    restored = dict_to_state(json.loads(path.read_text()))
    assert restored.slug == "json-test"


def test_next_runnable_empty():
    state = PlannerState()
    assert state.next_runnable() == "recon"


def test_next_runnable_after_complete():
    state = PlannerState(phases={
        "recon": PhaseState(status=PhaseStatus.COMPLETE),
    })
    assert state.next_runnable() == "architects"


def test_next_runnable_in_progress():
    state = PlannerState(phases={
        "recon": PhaseState(status=PhaseStatus.COMPLETE),
        "architects": PhaseState(status=PhaseStatus.IN_PROGRESS),
    })
    assert state.next_runnable() == "architects"


def test_next_runnable_all_done():
    phases = {p: PhaseState(status=PhaseStatus.COMPLETE) for p in PHASE_ORDER}
    state = PlannerState(phases=phases)
    assert state.next_runnable() is None


def test_next_runnable_failed_retryable():
    state = PlannerState(phases={
        "recon": PhaseState(status=PhaseStatus.COMPLETE),
        "architects": PhaseState(status=PhaseStatus.FAILED, retries=1),
    })
    assert state.next_runnable() == "architects"


def test_dict_to_state_invalid_status():
    d = {"phases": {"recon": {"status": "bogus"}}}
    state = dict_to_state(d)
    assert state.phases["recon"].status == PhaseStatus.FAILED


def test_state_manager_save_load(tmp_path):
    state_file = tmp_path / "state.json"
    mgr = StateManager(state_file)
    assert not mgr.exists()

    state = PlannerState(slug="mgr-test", problem_statement="test problem")
    mgr.save(state)
    assert mgr.exists()

    loaded = mgr.load()
    assert loaded.slug == "mgr-test"
    assert loaded.updated_at != ""


def test_state_manager_update(tmp_path):
    state_file = tmp_path / "state.json"
    mgr = StateManager(state_file)
    mgr.save(PlannerState(slug="update-test"))

    def mutate(s: PlannerState):
        s.phases["recon"] = PhaseState(status=PhaseStatus.COMPLETE)

    updated = mgr.update(mutate)
    assert updated.phases["recon"].status == PhaseStatus.COMPLETE

    reloaded = mgr.load()
    assert reloaded.phases["recon"].status == PhaseStatus.COMPLETE


def test_state_to_dict_optional_fields():
    state = PlannerState(phases={
        "recon": PhaseState(
            status=PhaseStatus.COMPLETE,
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:01:00Z",
            last_error="some error",
        ),
    })
    d = state_to_dict(state)
    phase = d["phases"]["recon"]
    assert phase["started_at"] == "2026-01-01T00:00:00Z"
    assert phase["last_error"] == "some error"


def test_state_to_dict_omits_empty_optional():
    state = PlannerState(phases={
        "recon": PhaseState(status=PhaseStatus.PENDING),
    })
    d = state_to_dict(state)
    phase = d["phases"]["recon"]
    assert "started_at" not in phase
    assert "last_error" not in phase


def test_new_planner_state_fields():
    state = PlannerState(
        repo_dir="/path/to/repo",
        killed=True,
        kill_reason="user requested",
        driver_pid=12345,
    )
    d = state_to_dict(state)
    assert d["repo_dir"] == "/path/to/repo"
    assert d["killed"] is True
    assert d["kill_reason"] == "user requested"
    assert d["driver_pid"] == 12345

    restored = dict_to_state(d)
    assert restored.repo_dir == "/path/to/repo"
    assert restored.killed is True
    assert restored.kill_reason == "user requested"
    assert restored.driver_pid == 12345


def test_new_fields_default_values():
    state = PlannerState()
    assert state.repo_dir == ""
    assert state.killed is False
    assert state.kill_reason == ""
    assert state.driver_pid == 0


def test_preset_fields_round_trip():
    state = PlannerState(
        slug="preset-test",
        preset="hz-web",
        preset_path="/path/to/presets/hz-web",
    )
    d = state_to_dict(state)
    assert d["preset"] == "hz-web"
    assert d["preset_path"] == "/path/to/presets/hz-web"

    restored = dict_to_state(d)
    assert restored.preset == "hz-web"
    assert restored.preset_path == "/path/to/presets/hz-web"


def test_preset_fields_default_empty():
    state = PlannerState()
    assert state.preset == ""
    assert state.preset_path == ""

    d = state_to_dict(state)
    restored = dict_to_state(d)
    assert restored.preset == ""
    assert restored.preset_path == ""
