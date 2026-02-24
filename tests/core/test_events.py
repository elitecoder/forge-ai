# Copyright 2026. Tests for forge.core.events.

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from forge.core.events import (
    REGISTRY_MAX_BYTES,
    SCHEMA_VERSION,
    Event,
    EventType,
    _event_to_json,
    _maybe_rotate_registry,
    _registry_path,
    cmd_status,
    emit,
    log_event,
    log_registry,
)


def _make_event(**overrides) -> Event:
    defaults = dict(
        v=SCHEMA_VERSION, event="step_started", ts="2026-02-23T15:00:00Z",
        session="/tmp/test-session", subsystem="executor",
    )
    defaults.update(overrides)
    return Event(**defaults)


class TestEventToJson:
    def test_includes_required_fields(self):
        ev = _make_event()
        d = json.loads(_event_to_json(ev))
        assert d["v"] == SCHEMA_VERSION
        assert d["event"] == "step_started"
        assert d["ts"] == "2026-02-23T15:00:00Z"
        assert d["session"] == "/tmp/test-session"
        assert d["subsystem"] == "executor"

    def test_strips_empty_fields(self):
        ev = _make_event()
        d = json.loads(_event_to_json(ev))
        assert "step" not in d
        assert "phase" not in d
        assert "error" not in d
        assert "checkpoint" not in d

    def test_keeps_populated_optional_fields(self):
        ev = _make_event(step="code", error="build failed", retries=2)
        d = json.loads(_event_to_json(ev))
        assert d["step"] == "code"
        assert d["error"] == "build failed"
        assert d["retries"] == 2

    def test_keeps_passed_true(self):
        ev = _make_event(event="judge_verdict", passed=True, pass_count=5, total=6)
        d = json.loads(_event_to_json(ev))
        assert d["passed"] is True
        assert d["pass_count"] == 5
        assert d["total"] == 6


class TestLogEvent:
    def test_appends_to_file(self, tmp_path):
        sd = str(tmp_path)
        ev = _make_event(session=sd)
        log_event(sd, ev)
        log_event(sd, ev)
        lines = (tmp_path / "events.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["event"] == "step_started"

    def test_no_crash_on_missing_dir(self):
        ev = _make_event(session="/nonexistent/path")
        log_event("/nonexistent/path", ev)

    def test_no_crash_on_empty_session_dir(self):
        ev = _make_event()
        log_event("", ev)


class TestLogRegistry:
    def test_creates_file(self, tmp_path):
        registry = tmp_path / "registry.jsonl"
        with patch("forge.core.events._registry_path", return_value=registry):
            ev = _make_event()
            log_registry(ev)
        assert registry.is_file()
        d = json.loads(registry.read_text().strip())
        assert d["event"] == "step_started"

    def test_appends_multiple(self, tmp_path):
        registry = tmp_path / "registry.jsonl"
        with patch("forge.core.events._registry_path", return_value=registry):
            log_registry(_make_event(event="pipeline_started"))
            log_registry(_make_event(event="pipeline_completed"))
        lines = registry.read_text().strip().splitlines()
        assert len(lines) == 2


class TestRegistryRotation:
    def test_rotates_on_size_cap(self, tmp_path):
        registry = tmp_path / "registry.jsonl"
        registry.write_text("x" * (REGISTRY_MAX_BYTES + 1))
        with patch("forge.core.events._registry_path", return_value=registry):
            _maybe_rotate_registry()
        assert not registry.exists()
        rotated = list(tmp_path.glob("registry.*.jsonl"))
        assert len(rotated) == 1

    def test_no_rotation_under_cap(self, tmp_path):
        registry = tmp_path / "registry.jsonl"
        registry.write_text("small content")
        with patch("forge.core.events._registry_path", return_value=registry):
            _maybe_rotate_registry()
        assert registry.exists()
        rotated = list(tmp_path.glob("registry.*.jsonl"))
        assert len(rotated) == 0


class TestEmit:
    def test_lifecycle_event_writes_both(self, tmp_path):
        sd = str(tmp_path / "session")
        os.makedirs(sd)
        registry = tmp_path / "registry.jsonl"
        with patch("forge.core.events._registry_path", return_value=registry):
            emit(sd, EventType.PIPELINE_STARTED, "executor", pipeline="full")
        session_log = Path(sd) / "events.jsonl"
        assert session_log.is_file()
        assert registry.is_file()
        ev = json.loads(session_log.read_text().strip())
        assert ev["event"] == "pipeline_started"
        assert ev["pipeline"] == "full"

    def test_step_event_skips_registry(self, tmp_path):
        sd = str(tmp_path / "session")
        os.makedirs(sd)
        registry = tmp_path / "registry.jsonl"
        with patch("forge.core.events._registry_path", return_value=registry):
            emit(sd, EventType.STEP_STARTED, "executor", step="code")
        session_log = Path(sd) / "events.jsonl"
        assert session_log.is_file()
        assert not registry.exists()

    def test_auto_populates_timestamp(self, tmp_path):
        sd = str(tmp_path)
        emit(sd, EventType.STEP_PASSED, "executor", step="build")
        ev = json.loads((tmp_path / "events.jsonl").read_text().strip())
        assert ev["ts"].endswith("Z")
        assert "T" in ev["ts"]


class TestCmdStatus:
    def _make_executor_session(self, base, name, steps, killed=False):
        sd = base / "executor" / name
        sd.mkdir(parents=True)
        state = {
            "pipeline": "full", "preset": "hz-web",
            "session_dir": str(sd), "killed": killed,
            "steps": steps,
            "created_at": "2026-02-23T15:00:00Z",
            "updated_at": "2026-02-23T15:05:00Z",
        }
        (sd / "agent-state.json").write_text(json.dumps(state))
        return sd

    def _make_planner_session(self, base, name, phases, killed=False):
        sd = base / "planner" / name
        sd.mkdir(parents=True)
        state = {
            "preset": "hz-web", "session_dir": str(sd),
            "killed": killed, "phases": phases,
            "created_at": "2026-02-23T14:00:00Z",
            "updated_at": "2026-02-23T14:30:00Z",
        }
        (sd / ".planner-state.json").write_text(json.dumps(state))
        return sd

    def test_lists_executor_sessions(self, tmp_path, capsys):
        self._make_executor_session(tmp_path, "test-session", {
            "code": {"status": "complete"},
            "test": {"status": "in_progress"},
        })
        with patch("forge.core.events.SESSIONS_BASE", tmp_path):
            import types
            args = types.SimpleNamespace(active=False, limit=20)
            cmd_status(args)
        output = json.loads(capsys.readouterr().out)
        assert len(output) == 1
        assert output[0]["subsystem"] == "executor"
        assert output[0]["status"] == "running"

    def test_active_filter(self, tmp_path, capsys):
        self._make_executor_session(tmp_path, "done", {
            "code": {"status": "complete"},
        })
        self._make_executor_session(tmp_path, "running", {
            "code": {"status": "in_progress"},
        })
        with patch("forge.core.events.SESSIONS_BASE", tmp_path):
            import types
            args = types.SimpleNamespace(active=True, limit=20)
            cmd_status(args)
        output = json.loads(capsys.readouterr().out)
        assert len(output) == 1
        assert output[0]["status"] == "running"

    def test_includes_planner_sessions(self, tmp_path, capsys):
        self._make_planner_session(tmp_path, "plan-session", {
            "recon": {"status": "complete"},
            "architects": {"status": "in_progress"},
        })
        with patch("forge.core.events.SESSIONS_BASE", tmp_path):
            import types
            args = types.SimpleNamespace(active=False, limit=20)
            cmd_status(args)
        output = json.loads(capsys.readouterr().out)
        assert len(output) == 1
        assert output[0]["subsystem"] == "planner"
        assert output[0]["status"] == "running"

    def test_killed_status(self, tmp_path, capsys):
        self._make_executor_session(tmp_path, "killed", {
            "code": {"status": "in_progress"},
        }, killed=True)
        with patch("forge.core.events.SESSIONS_BASE", tmp_path):
            import types
            args = types.SimpleNamespace(active=False, limit=20)
            cmd_status(args)
        output = json.loads(capsys.readouterr().out)
        assert output[0]["status"] == "killed"

    def test_limit(self, tmp_path, capsys):
        for i in range(5):
            self._make_executor_session(tmp_path, f"session-{i}", {
                "code": {"status": "complete"},
            })
        with patch("forge.core.events.SESSIONS_BASE", tmp_path):
            import types
            args = types.SimpleNamespace(active=False, limit=2)
            cmd_status(args)
        output = json.loads(capsys.readouterr().out)
        assert len(output) == 2
