"""Tests for pre_pr_gate.py — 3-tier gate logic."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from architect.executor.engine.state import StateManager, PipelineState, StepState, StepStatus
from architect.executor.engine.checkpoint import write_checkpoint
from tests.executor.helpers import make_state, save_state

from architect.executor import pre_pr_gate


class TestPrePrGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="gate_test_")
        self.session_dir = Path(self.tmp) / "session"
        self.session_dir.mkdir()
        self.state_file = self.session_dir / "agent-state.json"
        self.cp_dir = str(self.session_dir / "checkpoints")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_tier1_no_pipeline(self):
        """No active session found → PASS."""
        with patch.object(pre_pr_gate, "find_active_session", return_value=None):
            result = pre_pr_gate.main()
        self.assertEqual(result, 0)

    def test_tier2_suspicious_checkpoints(self):
        """Checkpoints exist but no state → FAIL."""
        os.makedirs(self.cp_dir)
        Path(os.path.join(self.cp_dir, "build.passed")).write_text("fake")

        with patch.object(pre_pr_gate, "find_active_session", return_value=self.session_dir):
            result = pre_pr_gate.main()
        self.assertEqual(result, 1)

    def test_tier3_all_valid(self):
        """State + all checkpoints valid → PASS."""
        state = make_state(steps=["build", "lint", "create_pr"])
        for s in ["build", "lint"]:
            state.steps[s].status = StepStatus.COMPLETE
        save_state(state, self.state_file)

        for s in ["build", "lint"]:
            write_checkpoint(self.cp_dir, s, "full")

        with patch.object(pre_pr_gate, "find_active_session", return_value=self.session_dir):
            result = pre_pr_gate.main()
        self.assertEqual(result, 0)

    def test_tier3_missing_checkpoint(self):
        """State exists but checkpoint missing → FAIL."""
        state = make_state(steps=["build", "lint", "create_pr"])
        save_state(state, self.state_file)

        write_checkpoint(self.cp_dir, "build", "full")
        # No lint checkpoint

        with patch.object(pre_pr_gate, "find_active_session", return_value=self.session_dir):
            result = pre_pr_gate.main()
        self.assertEqual(result, 1)

    def test_tier3_deleted_checkpoint(self):
        """Deleted checkpoint file → FAIL."""
        state = make_state(steps=["build", "create_pr"])
        state.steps["build"].status = StepStatus.COMPLETE
        save_state(state, self.state_file)

        path = write_checkpoint(self.cp_dir, "build", "full")
        os.unlink(path)

        with patch.object(pre_pr_gate, "find_active_session", return_value=self.session_dir):
            result = pre_pr_gate.main()
        self.assertEqual(result, 1)


    def test_critical_step_manual_skip_fails_gate(self):
        """Bug 2 regression: critical step (test) with manual_skip=true must FAIL the gate."""
        state = make_state(steps=["build", "test", "create_pr"])
        for s in ["build", "test"]:
            state.steps[s].status = StepStatus.COMPLETE
        save_state(state, self.state_file)

        # build: normal checkpoint; test: manual_skip checkpoint
        write_checkpoint(self.cp_dir, "build", "full")
        os.makedirs(self.cp_dir, exist_ok=True)
        Path(os.path.join(self.cp_dir, "test.passed")).write_text(
            "step=test\npassed_at=2026-01-01T00:00:00Z\npipeline=full\nmanual_skip=true\n"
        )

        with patch.object(pre_pr_gate, "find_active_session", return_value=self.session_dir):
            result = pre_pr_gate.main()
        self.assertEqual(result, 1)

    def test_non_critical_step_manual_skip_passes_gate(self):
        """Bug 2 regression: non-critical step (report) with manual_skip=true passes."""
        state = make_state(steps=["build", "report", "create_pr"])
        for s in ["build", "report"]:
            state.steps[s].status = StepStatus.COMPLETE
        save_state(state, self.state_file)

        write_checkpoint(self.cp_dir, "build", "full")
        os.makedirs(self.cp_dir, exist_ok=True)
        Path(os.path.join(self.cp_dir, "report.passed")).write_text(
            "step=report\npassed_at=2026-01-01T00:00:00Z\npipeline=full\nmanual_skip=true\n"
        )

        with patch.object(pre_pr_gate, "find_active_session", return_value=self.session_dir):
            result = pre_pr_gate.main()
        self.assertEqual(result, 0)

    def test_normal_checkpoint_passes_gate(self):
        """Checkpoint without manual_skip passes the gate normally."""
        state = make_state(steps=["build", "test", "create_pr"])
        for s in ["build", "test"]:
            state.steps[s].status = StepStatus.COMPLETE
        save_state(state, self.state_file)

        for s in ["build", "test"]:
            write_checkpoint(self.cp_dir, s, "full")

        with patch.object(pre_pr_gate, "find_active_session", return_value=self.session_dir):
            result = pre_pr_gate.main()
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
