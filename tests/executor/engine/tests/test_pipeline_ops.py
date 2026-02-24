"""Tests for forge.executor.engine.pipeline_ops — dep_is_satisfied, get_next_steps."""

import unittest

from forge.executor.engine.registry import StepDefinition
from forge.executor.engine.state import StepState, StepStatus
from forge.executor.engine.pipeline_ops import (
    dep_is_satisfied, is_permanently_failed, get_next_steps, MAX_RETRIES,
)
from tests.executor.helpers import make_state, make_preset


class TestDepIsSatisfied(unittest.TestCase):
    """Bug 5 regression: permanently failed steps must NOT satisfy dependencies."""

    def test_complete_step_satisfies_dep(self):
        ss = StepState(status=StepStatus.COMPLETE)
        self.assertTrue(dep_is_satisfied(ss))

    def test_pending_step_does_not_satisfy_dep(self):
        ss = StepState(status=StepStatus.PENDING)
        self.assertFalse(dep_is_satisfied(ss))

    def test_failed_step_does_not_satisfy_dep(self):
        ss = StepState(status=StepStatus.FAILED, retries=1)
        self.assertFalse(dep_is_satisfied(ss))

    def test_permanently_failed_step_does_not_satisfy_dep(self):
        """Regression: a step that exhausted all retries must NOT satisfy deps."""
        ss = StepState(status=StepStatus.FAILED, retries=MAX_RETRIES)
        self.assertTrue(is_permanently_failed(ss))
        self.assertFalse(dep_is_satisfied(ss))

    def test_in_progress_step_does_not_satisfy_dep(self):
        ss = StepState(status=StepStatus.IN_PROGRESS)
        self.assertFalse(dep_is_satisfied(ss))


class TestGetNextStepsWithPermanentlyFailed(unittest.TestCase):
    """Bug 5 regression: permanently failed dep should block downstream steps."""

    def test_permanently_failed_dep_blocks_downstream(self):
        state = make_state(steps=["build", "lint", "test"])
        state.steps["build"].status = StepStatus.FAILED
        state.steps["build"].retries = MAX_RETRIES

        _step = lambda n: StepDefinition(name=n, run_command="echo ok")
        preset = make_preset(
            steps={"build": _step("build"), "lint": _step("lint"), "test": _step("test")},
            pipelines={"full": ["build", "lint", "test"]},
        )
        result = get_next_steps(state, preset)
        runnable_names = [r["step"] for r in result.get("runnable", [])]
        self.assertNotIn("lint", runnable_names)
        self.assertNotIn("test", runnable_names)


class TestGetNextStepsDeadlockDetection(unittest.TestCase):
    """Bug 8 regression: permanently failed step must not produce 'all complete'."""

    def _make_pipeline(self, step_names):
        _step = lambda n: StepDefinition(name=n, run_command="echo ok")
        return make_preset(
            steps={n: _step(n) for n in step_names},
            pipelines={"full": step_names},
        )

    def test_blocked_steps_not_reported_as_all_complete(self):
        """When upstream is permanently failed, blocked downstream must NOT
        produce the 'all complete' sentinel (step=None)."""
        state = make_state(steps=["lint", "test", "report"])
        state.steps["lint"].status = StepStatus.FAILED
        state.steps["lint"].retries = MAX_RETRIES

        preset = self._make_pipeline(["lint", "test", "report"])
        result = get_next_steps(state, preset)

        # Must NOT return the "all complete" sentinel
        self.assertNotIn("step", result, "Should not return step=None when steps are blocked")
        # Must report blocked steps
        self.assertIn("blocked", result)
        self.assertIn("test", result["blocked"])
        self.assertIn("report", result["blocked"])

    def test_all_complete_when_truly_complete(self):
        """When every step is COMPLETE, return the 'all complete' sentinel."""
        state = make_state(steps=["lint", "test"])
        state.steps["lint"].status = StepStatus.COMPLETE
        state.steps["test"].status = StepStatus.COMPLETE

        preset = self._make_pipeline(["lint", "test"])
        result = get_next_steps(state, preset)

        self.assertEqual(result.get("step"), None)
        self.assertIn("complete", result.get("message", "").lower())

    def test_all_complete_when_some_skipped_and_rest_complete(self):
        """Permanently failed + no blocked downstream = all complete."""
        state = make_state(steps=["lint"])
        state.steps["lint"].status = StepStatus.FAILED
        state.steps["lint"].retries = MAX_RETRIES

        preset = self._make_pipeline(["lint"])
        result = get_next_steps(state, preset)

        # No downstream to block — should be "all complete"
        self.assertEqual(result.get("step"), None)


if __name__ == "__main__":
    unittest.main()
