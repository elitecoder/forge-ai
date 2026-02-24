"""Tests for state.py — dependency_graph serialization, runnable_steps(), StateManager error handling."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from forge.executor.engine.state import (
    PipelineState, StepState, StepStatus, StateManager,
    _state_to_dict, _dict_to_state,
)


def _make_state(step_order, dependency_graph=None, statuses=None):
    """Helper to build a PipelineState with given steps/deps/statuses."""
    statuses = statuses or {}
    steps = {}
    for name in step_order:
        ss = StepState()
        if name in statuses:
            ss.status = statuses[name]
        steps[name] = ss
    return PipelineState(
        steps=steps,
        step_order=step_order,
        dependency_graph=dependency_graph or {},
    )


class TestDependencyGraphSerialization:
    def test_round_trip(self):
        graph = {"b": ["a"], "c": ["a", "b"]}
        state = _make_state(["a", "b", "c"], dependency_graph=graph)
        d = _state_to_dict(state)
        assert d["dependency_graph"] == graph

        restored = _dict_to_state(d)
        assert restored.dependency_graph == graph

    def test_absent_graph_defaults_empty(self):
        d = {"steps": {}, "step_order": []}
        state = _dict_to_state(d)
        assert state.dependency_graph == {}

    def test_empty_graph_serializes(self):
        state = _make_state(["a", "b"])
        d = _state_to_dict(state)
        assert d["dependency_graph"] == {}


class TestPresetPathSerialization:
    def test_round_trip(self):
        state = _make_state(["a"])
        state.preset_path = "/home/user/.forge/presets/custom"
        d = _state_to_dict(state)
        assert d["preset_path"] == "/home/user/.forge/presets/custom"
        restored = _dict_to_state(d)
        assert restored.preset_path == "/home/user/.forge/presets/custom"

    def test_absent_defaults_empty(self):
        d = {"steps": {}, "step_order": []}
        state = _dict_to_state(d)
        assert state.preset_path == ""

    def test_empty_serializes(self):
        state = _make_state(["a"])
        d = _state_to_dict(state)
        assert d["preset_path"] == ""


class TestRunnableSteps:
    def test_legacy_mode_returns_first_non_complete(self):
        state = _make_state(
            ["a", "b", "c"],
            statuses={"a": StepStatus.COMPLETE},
        )
        assert state.runnable_steps() == ["b"]

    def test_legacy_mode_all_complete(self):
        state = _make_state(
            ["a", "b"],
            statuses={"a": StepStatus.COMPLETE, "b": StepStatus.COMPLETE},
        )
        assert state.runnable_steps() == []

    def test_legacy_mode_in_progress(self):
        state = _make_state(
            ["a", "b", "c"],
            statuses={"a": StepStatus.IN_PROGRESS},
        )
        assert state.runnable_steps() == ["a"]

    def test_dag_roots_runnable_immediately(self):
        state = _make_state(
            ["a", "b", "c"],
            dependency_graph={"b": ["a"], "c": ["a"]},
        )
        assert state.runnable_steps() == ["a"]

    def test_dag_parallel_steps_after_root_complete(self):
        state = _make_state(
            ["a", "b", "c", "d"],
            dependency_graph={"b": ["a"], "c": ["a"], "d": ["b", "c"]},
            statuses={"a": StepStatus.COMPLETE},
        )
        assert state.runnable_steps() == ["b", "c"]

    def test_dag_blocked_until_all_deps_complete(self):
        state = _make_state(
            ["a", "b", "c"],
            dependency_graph={"c": ["a", "b"]},
            statuses={"a": StepStatus.COMPLETE, "b": StepStatus.PENDING},
        )
        # c depends on both a and b; b is still pending
        runnable = state.runnable_steps()
        assert "c" not in runnable
        assert "b" in runnable

    def test_dag_failed_step_is_runnable(self):
        state = _make_state(
            ["a", "b"],
            dependency_graph={"b": ["a"]},
            statuses={"a": StepStatus.COMPLETE, "b": StepStatus.FAILED},
        )
        assert state.runnable_steps() == ["b"]

    def test_dag_in_progress_not_runnable(self):
        state = _make_state(
            ["a", "b", "c"],
            dependency_graph={"b": ["a"], "c": ["a"]},
            statuses={"a": StepStatus.COMPLETE, "b": StepStatus.IN_PROGRESS},
        )
        # b is in_progress (not runnable), c should still be runnable
        assert state.runnable_steps() == ["c"]

    def test_dag_all_complete(self):
        state = _make_state(
            ["a", "b"],
            dependency_graph={"b": ["a"]},
            statuses={"a": StepStatus.COMPLETE, "b": StepStatus.COMPLETE},
        )
        assert state.runnable_steps() == []


class TestTimingFieldsSerialization:
    """Verify started_at / completed_at survive serialization round-trip."""

    def test_round_trip_with_timing(self):
        state = _make_state(["a", "b"])
        state.steps["a"].started_at = "2026-02-17T10:00:00Z"
        state.steps["a"].completed_at = "2026-02-17T10:05:12Z"
        state.steps["b"].started_at = "2026-02-17T10:05:13Z"

        d = _state_to_dict(state)
        assert d["steps"]["a"]["started_at"] == "2026-02-17T10:00:00Z"
        assert d["steps"]["a"]["completed_at"] == "2026-02-17T10:05:12Z"
        assert d["steps"]["b"]["started_at"] == "2026-02-17T10:05:13Z"
        assert "completed_at" not in d["steps"]["b"]

        restored = _dict_to_state(d)
        assert restored.steps["a"].started_at == "2026-02-17T10:00:00Z"
        assert restored.steps["a"].completed_at == "2026-02-17T10:05:12Z"
        assert restored.steps["b"].started_at == "2026-02-17T10:05:13Z"
        assert restored.steps["b"].completed_at == ""

    def test_empty_timing_omitted_from_dict(self):
        state = _make_state(["a"])
        d = _state_to_dict(state)
        assert "started_at" not in d["steps"]["a"]
        assert "completed_at" not in d["steps"]["a"]

    def test_absent_timing_defaults_to_empty(self):
        d = {"steps": {"a": {"status": "pending", "retries": 0}}, "step_order": ["a"]}
        state = _dict_to_state(d)
        assert state.steps["a"].started_at == ""
        assert state.steps["a"].completed_at == ""


class TestStateManagerSaveErrorHandling:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="state_save_err_")
        self.state_file = Path(self.tmp) / ".agent-state.json"
        self.mgr = StateManager(self.state_file)

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_save_cleans_up_temp_on_rename_failure(self):
        state = PipelineState(pipeline="full", steps={"a": StepState()}, step_order=["a"])
        with patch("os.rename", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                self.mgr.save(state)
        # Temp file should be cleaned up
        tmp_files = [f for f in os.listdir(self.tmp) if f.endswith(".tmp")]
        assert tmp_files == []

    def test_save_cleans_up_temp_on_write_failure(self):
        state = PipelineState(pipeline="full", steps={"a": StepState()}, step_order=["a"])
        original_write = os.write

        def failing_write(fd, data):
            # Let first write succeed (the JSON content), fail on second (newline)
            if data == b"\n":
                raise OSError("write failed")
            return original_write(fd, data)

        with patch("os.write", side_effect=failing_write):
            with pytest.raises(OSError, match="write failed"):
                self.mgr.save(state)
        tmp_files = [f for f in os.listdir(self.tmp) if f.endswith(".tmp")]
        assert tmp_files == []


class TestStateManagerUpdateErrorHandling:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="state_upd_err_")
        self.state_file = Path(self.tmp) / ".agent-state.json"
        self.mgr = StateManager(self.state_file)
        # Write initial state so update can load it
        initial = PipelineState(pipeline="full", steps={"a": StepState()}, step_order=["a"])
        self.mgr.save(initial)

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_update_cleans_up_temp_on_rename_failure(self):
        with patch("os.rename", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                self.mgr.update(lambda s: setattr(s, 'pipeline', 'changed'))
        tmp_files = [f for f in os.listdir(self.tmp) if f.endswith(".tmp")]
        assert tmp_files == []
        # Original state file should still be intact
        original = self.mgr.load()
        assert original.pipeline == "full"

    def test_update_cleans_up_temp_on_write_failure(self):
        original_write = os.write

        def failing_write(fd, data):
            if data == b"\n":
                raise OSError("write failed")
            return original_write(fd, data)

        with patch("os.write", side_effect=failing_write):
            with pytest.raises(OSError, match="write failed"):
                self.mgr.update(lambda s: setattr(s, 'pipeline', 'changed'))
        tmp_files = [f for f in os.listdir(self.tmp) if f.endswith(".tmp")]
        assert tmp_files == []


class TestStateManagerSaveCloseFailure:
    """Cover lines 190-191: os.close raises OSError in save() error cleanup."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="state_close_err_")
        self.state_file = Path(self.tmp) / ".agent-state.json"
        self.mgr = StateManager(self.state_file)

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_save_close_fails_during_error_cleanup(self):
        state = PipelineState(pipeline="full", steps={"a": StepState()}, step_order=["a"])
        original_write = os.write
        original_close = os.close
        call_count = {"write": 0}

        def failing_write(fd, data):
            call_count["write"] += 1
            if call_count["write"] == 2:  # fail on newline write
                raise OSError("write failed")
            return original_write(fd, data)

        def failing_close(fd):
            # Let it fail for the temp file fd (not the lock file)
            try:
                original_close(fd)
            except OSError:
                pass
            raise OSError("close failed")

        with patch("os.write", side_effect=failing_write):
            with patch("os.close", side_effect=failing_close):
                with pytest.raises(OSError):
                    self.mgr.save(state)
