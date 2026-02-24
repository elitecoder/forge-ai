
"""Tests for engine.state — CRUD, locking, migration, crash safety."""

import json
import os
import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from forge.executor.engine.state import (
    StateManager, PipelineState, StepState, StepStatus,
    _state_to_dict, _dict_to_state,
)
from tests.executor.helpers import make_state, save_state


class TestPipelineState(unittest.TestCase):
    def test_step_names_ordered_with_step_order(self):
        state = PipelineState(
            steps={"b": StepState(), "a": StepState()},
            step_order=["a", "b"],
        )
        self.assertEqual(state.step_names_ordered(), ["a", "b"])

    def test_step_names_ordered_fallback_to_dict_keys(self):
        state = PipelineState(steps={"x": StepState(), "y": StepState()})
        self.assertEqual(state.step_names_ordered(), ["x", "y"])

    def test_default_model_profile_is_empty(self):
        state = PipelineState()
        self.assertEqual(state.model_profile, "")


class TestSerialization(unittest.TestCase):
    def test_round_trip(self):
        state = make_state(steps=["build", "lint"])
        d = _state_to_dict(state)
        restored = _dict_to_state(d)
        self.assertEqual(restored.pipeline, state.pipeline)
        self.assertEqual(restored.step_names_ordered(), ["build", "lint"])
        self.assertEqual(restored.steps["build"].status, StepStatus.PENDING)

    def test_step_order_serialized(self):
        state = make_state(steps=["lint", "build"])
        d = _state_to_dict(state)
        self.assertEqual(d["step_order"], ["lint", "build"])

    def test_keys_used_as_is(self):
        """v1 compound key migration removed — keys are used verbatim."""
        d = {
            "steps": {"code": {"status": "complete"}, "build": {"status": "pending"}},
            "current_step": "build",
        }
        state = _dict_to_state(d)
        self.assertIn("code", state.steps)
        self.assertIn("build", state.steps)
        self.assertEqual(state.current_step, "build")
        self.assertEqual(state.steps["code"].status, StepStatus.COMPLETE)

    def test_last_error_serialized_only_when_present(self):
        state = make_state(steps=["s"])
        state.steps["s"].last_error = "boom"
        d = _state_to_dict(state)
        self.assertEqual(d["steps"]["s"]["last_error"], "boom")

        state.steps["s"].last_error = ""
        d = _state_to_dict(state)
        self.assertNotIn("last_error", d["steps"]["s"])


class TestStateManager(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="state_test_")
        self.state_file = Path(self.tmp) / ".agent-state.json"

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_exists_false_initially(self):
        mgr = StateManager(self.state_file)
        self.assertFalse(mgr.exists())

    def test_save_and_load(self):
        state = make_state(steps=["a", "b"])
        mgr = save_state(state, self.state_file)
        loaded = mgr.load()
        self.assertEqual(loaded.pipeline, "full")
        self.assertEqual(loaded.step_names_ordered(), ["a", "b"])

    def test_save_creates_file(self):
        mgr = StateManager(self.state_file)
        mgr.save(make_state())
        self.assertTrue(self.state_file.is_file())

    def test_update_mutates_state(self):
        mgr = save_state(make_state(steps=["s"]), self.state_file)
        result = mgr.update(lambda s: setattr(s.steps["s"], "status", StepStatus.COMPLETE))
        self.assertEqual(result.steps["s"].status, StepStatus.COMPLETE)
        reloaded = mgr.load()
        self.assertEqual(reloaded.steps["s"].status, StepStatus.COMPLETE)

    def test_update_sets_updated_at(self):
        mgr = save_state(make_state(), self.state_file)
        result = mgr.update(lambda s: None)
        self.assertTrue(result.updated_at)

    def test_save_atomic_no_partial_write(self):
        mgr = save_state(make_state(steps=["a"]), self.state_file)
        original = self.state_file.read_text()

        def bad_mutator(s):
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            mgr.update(bad_mutator)

        # File should be unchanged
        self.assertEqual(self.state_file.read_text(), original)

    def test_concurrent_updates(self):
        mgr = save_state(make_state(steps=["s"]), self.state_file)
        barrier = threading.Barrier(4)
        errors = []

        def increment(_i):
            try:
                barrier.wait(timeout=5)
                mgr.update(lambda s: setattr(s.steps["s"], "retries", s.steps["s"].retries + 1))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=increment, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(errors, [])
        final = mgr.load()
        self.assertEqual(final.steps["s"].retries, 4)

    def test_save_double_close_safety(self):
        """Verify save() doesn't crash on double-close (Fix 2 regression test)."""
        mgr = StateManager(self.state_file)
        state = make_state()
        # Should not raise
        mgr.save(state)
        self.assertTrue(self.state_file.is_file())

    def test_lockfile_created(self):
        mgr = save_state(make_state(), self.state_file)
        mgr.load()
        lock_file = self.state_file.with_suffix(".json.lock")
        self.assertTrue(lock_file.exists())

    def test_update_uses_lockfile_not_data_file(self):
        """Verify update() uses separate lockfile (Fix 1 regression test)."""
        mgr = save_state(make_state(steps=["s"]), self.state_file)

        # After update, the data file should be valid JSON (not corrupted by lock operations)
        mgr.update(lambda s: setattr(s.steps["s"], "status", StepStatus.IN_PROGRESS))
        data = json.loads(self.state_file.read_text())
        self.assertEqual(data["steps"]["s"]["status"], "in_progress")


if __name__ == "__main__":
    unittest.main()
