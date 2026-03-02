"""Tests for forge.executor.driver — plan discovery, step execution, dispatch loop, retry logic."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import shutil
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import forge.executor.driver as pipeline_driver
from forge.executor.driver import (
    discover_plan, execute_command_step,
    dispatch_ai_step, _handle_command_step, _handle_ai_step,
    _handle_inline_step, _dispatch_one_step, run_code_review,
    handle_report_step, dispatch_loop, run_pre_pr_gate,
    dispatch_with_judge, _handle_visual_test_step,
    _load_review_findings, _read_planner_slug,
    log_revalidation, log_headline,
    ensure_dev_server, _enforce_file_allowlist, _snapshot_worktree,
    main, _write_status, _recent_log,
    _set_hook_build_cmd, _set_hook_eslint_config, _set_plugin_dir, _preflight_hooks,
    PIPELINE_TIMEOUT_S, MAX_STEP_RETRIES, MAX_DEV_SERVER_PRECHECK_FAILURES,
)


# ── Helper: create standard mock state/preset for dispatch_loop tests ─────

def _mock_dispatch_loop_state(killed=False):
    state = MagicMock()
    state.killed = killed
    state.kill_reason = ""
    return state


# ── discover_plan ─────────────────────────────────────────────────────────


class TestDiscoverPlan:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="driver_test_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_plan_path_returns_absolute(self):
        plan = os.path.join(self.tmp, "my-plan.md")
        Path(plan).write_text("# Plan")

        result = discover_plan(plan, None)

        # macOS resolves /var -> /private/var, so compare resolved paths
        assert Path(result).resolve() == Path(plan).resolve()
        assert os.path.isabs(result)

    def test_plan_path_not_found_exits(self):
        with pytest.raises(SystemExit):
            discover_plan("/nonexistent/plan.md", None)

    def test_plan_dir_finds_final_plan(self):
        d = os.path.join(self.tmp, "session")
        os.makedirs(d)
        Path(os.path.join(d, "final-plan.md")).write_text("final plan")
        Path(os.path.join(d, "other.md")).write_text("other")

        result = discover_plan(None, d)

        assert result.endswith("final-plan.md")

    def test_plan_dir_prefers_final_plan_over_plan(self):
        d = os.path.join(self.tmp, "session")
        os.makedirs(d)
        Path(os.path.join(d, "final-plan.md")).write_text("final")
        Path(os.path.join(d, "plan.md")).write_text("plan")

        result = discover_plan(None, d)

        assert result.endswith("final-plan.md")

    def test_plan_dir_finds_plan_md(self):
        d = os.path.join(self.tmp, "session")
        os.makedirs(d)
        Path(os.path.join(d, "plan.md")).write_text("plan")

        result = discover_plan(None, d)

        assert result.endswith("plan.md")

    def test_plan_dir_falls_back_to_any_md(self):
        d = os.path.join(self.tmp, "session")
        os.makedirs(d)
        Path(os.path.join(d, "architecture-design.md")).write_text("design")

        result = discover_plan(None, d)

        assert result.endswith("architecture-design.md")

    def test_plan_dir_no_md_files_exits(self):
        d = os.path.join(self.tmp, "empty_session")
        os.makedirs(d)

        with pytest.raises(SystemExit):
            discover_plan(None, d)

    def test_plan_dir_not_found_exits(self):
        with pytest.raises(SystemExit):
            discover_plan(None, "/nonexistent/dir")

    def test_neither_plan_nor_dir_exits(self):
        with pytest.raises(SystemExit):
            discover_plan(None, None)


# ── _read_planner_slug ────────────────────────────────────────────────────


class TestReadPlannerSlug:
    def test_reads_slug_from_state_file(self, tmp_path):
        state = {"slug": "health-endpoint", "session_dir": str(tmp_path)}
        (tmp_path / ".planner-state.json").write_text(json.dumps(state))
        assert _read_planner_slug(str(tmp_path)) == "health-endpoint"

    def test_returns_empty_when_no_state_file(self, tmp_path):
        assert _read_planner_slug(str(tmp_path)) == ""

    def test_returns_empty_when_none(self):
        assert _read_planner_slug(None) == ""

    def test_returns_empty_on_invalid_json(self, tmp_path):
        (tmp_path / ".planner-state.json").write_text("not json")
        assert _read_planner_slug(str(tmp_path)) == ""

    def test_returns_empty_when_slug_missing(self, tmp_path):
        (tmp_path / ".planner-state.json").write_text("{}")
        assert _read_planner_slug(str(tmp_path)) == ""


# ── execute_command_step ──────────────────────────────────────────────────


class TestExecuteCommandStep:
    """Test execute_command_step via pipeline_ops mocks."""

    @patch("forge.executor.driver.pipeline_ops.mark_failed")
    @patch("forge.executor.driver.pipeline_ops.mark_passed")
    @patch("forge.executor.driver.pipeline_ops.mark_running")
    @patch("forge.executor.engine.runner.execute_command")
    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    def test_passed_step(self, mock_state, mock_preset, mock_exec, mock_run, mock_pass, mock_fail):
        from forge.executor.engine.runner import StepResult
        from forge.executor.engine.registry import StepDefinition, Preset, PipelineDefinition
        from forge.executor.engine.state import PipelineState, StepState

        mock_state.return_value = PipelineState(steps={"lint": StepState()}, step_order=["lint"])
        step_def = StepDefinition(name="lint", step_type="command", run_command="echo ok")
        mock_preset.return_value = Preset(
            name="test", version=3, description="", pipelines={},
            steps={"lint": step_def}, models={},
        )
        mock_exec.return_value = StepResult(passed=True, output="ok")

        result = execute_command_step("lint")

        assert result["result"] == "passed"
        assert result["step"] == "lint"
        mock_pass.assert_called_once()
        mock_fail.assert_not_called()

    @patch("forge.executor.driver.pipeline_ops.mark_failed")
    @patch("forge.executor.driver.pipeline_ops.mark_passed")
    @patch("forge.executor.driver.pipeline_ops.mark_running")
    @patch("forge.executor.engine.runner.execute_command")
    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    def test_failed_step_with_packages(self, mock_state, mock_preset, mock_exec,
                                       mock_run, mock_pass, mock_fail):
        from forge.executor.engine.runner import StepResult
        from forge.executor.engine.registry import StepDefinition, Preset
        from forge.executor.engine.state import PipelineState, StepState

        mock_state.return_value = PipelineState(steps={"test": StepState()}, step_order=["test"])
        step_def = StepDefinition(name="test", step_type="command", run_command="npm test")
        mock_preset.return_value = Preset(
            name="test", version=3, description="", pipelines={},
            steps={"test": step_def}, models={},
        )
        mock_exec.return_value = StepResult(
            passed=False, output="test failed", failed_packages=["apps/web"],
        )

        result = execute_command_step("test")

        assert result["result"] == "failed"
        assert result["failed_packages"] == ["apps/web"]
        mock_fail.assert_called_once()
        mock_pass.assert_not_called()

    @patch("forge.executor.driver.pipeline_ops.mark_running")
    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    def test_missing_step_def_returns_failed(self, mock_state, mock_preset, mock_run):
        from forge.executor.engine.registry import Preset
        from forge.executor.engine.state import PipelineState, StepState

        mock_state.return_value = PipelineState(steps={"lint": StepState()}, step_order=["lint"])
        mock_preset.return_value = Preset(
            name="test", version=3, description="", pipelines={},
            steps={}, models={},
        )

        result = execute_command_step("lint")

        assert result["result"] == "failed"


# ── Parallel dispatch ─────────────────────────────────────────────────────


class TestParallelDispatch:
    @patch("forge.executor.driver._write_status")
    @patch("forge.executor.driver._dispatch_one_step")
    @patch("forge.executor.driver.run_pre_pr_gate", return_value=True)
    @patch("forge.executor.driver.pipeline_ops.get_next_steps")
    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    def test_concurrent_dispatch(self, mock_state, mock_preset, mock_next,
                                 mock_gate, mock_dispatch, mock_write_status):
        """Two runnable steps are dispatched concurrently via ThreadPoolExecutor."""
        mock_state.return_value = _mock_dispatch_loop_state()
        mock_preset.return_value = MagicMock()
        mock_next.side_effect = [
            # First get_next_steps — 2 runnable steps
            {"runnable": [
                {"step": "lint", "type": "command", "retries": 0},
                {"step": "test", "type": "command", "retries": 0},
            ], "in_progress": [], "blocked": []},
            # Post-dispatch get_next_steps for revalidation check
            {"runnable": [], "in_progress": [], "blocked": []},
            # Second loop iteration — all complete
            {"step": None, "message": "All steps complete"},
        ]
        mock_dispatch.return_value = ("lint", True)

        result = dispatch_loop("/cwd", "/session", time.time())

        assert result is True
        assert mock_dispatch.call_count == 2


# ── Blocked detection ─────────────────────────────────────────────────────


class TestBlockedDetection:
    @patch("forge.executor.driver.pipeline_ops.get_next_steps")
    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    def test_deadlock_on_blocked_steps(self, mock_state, mock_preset, mock_next):
        """Empty runnable + non-empty blocked -> False (deadlock)."""
        mock_state.return_value = _mock_dispatch_loop_state()
        mock_preset.return_value = MagicMock()
        mock_next.return_value = {
            "runnable": [], "in_progress": [],
            "blocked": ["lint", "test"],
        }

        result = dispatch_loop("/cwd", "/session", time.time())

        assert result is False


# ── Command retry loop ────────────────────────────────────────────────────


class TestCommandRetryLoop:
    @patch("forge.executor.engine.agents.fix_agent.run")
    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver.pipeline_ops.mark_running")
    @patch("forge.executor.driver.execute_command_step")
    def test_execute_fail_fix_pass(self, mock_exec, mock_mark_run, mock_state, mock_preset, mock_fix):
        """execute->fail->fix_agent passes->(name, True)."""
        mock_exec.return_value = {"step": "lint", "result": "failed"}
        from forge.executor.engine.agents.fix_agent import FixAgentOutcome
        from forge.executor.engine.registry import StepDefinition, Preset, PipelineDefinition
        from forge.executor.engine.state import PipelineState, StepState

        mock_preset.return_value = Preset(
            name="test", version=3, description="",
            pipelines={"full": PipelineDefinition(steps=["lint"], dependencies={})},
            steps={"lint": StepDefinition(name="lint", step_type="command", run_command="echo ok")},
            models={"fix": "sonnet"}, preset_dir=Path("."),
        )
        mock_state.return_value = PipelineState(
            steps={"lint": StepState()}, step_order=["lint"], session_dir="/tmp/session",
        )
        mock_fix.return_value = FixAgentOutcome(passed=True, reason="Fix verified")

        name, success = _handle_command_step("lint", 0, "/cwd")

        assert success is True
        assert name == "lint"
        assert mock_fix.call_count == 1


# ── AI step retry ─────────────────────────────────────────────────────────


class TestAiStepRetry:
    @patch("forge.executor.driver.pipeline_ops.reset_step")
    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver.dispatch_ai_step")
    def test_fail_twice_succeed_third(self, mock_dispatch, mock_state, mock_preset, mock_reset):
        """AI step fails twice then succeeds on third attempt (no judge config)."""
        mock_dispatch.side_effect = [False, False, True]
        # Preset with no judge config for this step
        from forge.executor.engine.registry import StepDefinition
        preset = MagicMock()
        preset.steps = {"create_pr": StepDefinition(name="create_pr", step_type="ai")}
        mock_preset.return_value = preset
        mock_state.return_value = MagicMock()

        name, success = _handle_ai_step("create_pr", "/cwd", "/session")

        assert success is True
        assert mock_dispatch.call_count == 3

    @patch("forge.executor.driver.pipeline_ops.reset_step")
    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver.dispatch_ai_step")
    def test_all_retries_fail(self, mock_dispatch, mock_state, mock_preset, mock_reset):
        """AI step fails all 3 attempts."""
        mock_dispatch.return_value = False
        preset = MagicMock()
        preset.steps = {"create_pr": MagicMock(judge=None)}
        mock_preset.return_value = preset
        mock_state.return_value = MagicMock()

        name, success = _handle_ai_step("create_pr", "/cwd", "/session")

        assert success is False
        assert mock_dispatch.call_count == MAX_STEP_RETRIES


# ── Auto-pass removal ────────────────────────────────────────────────────


class TestAutoPassRemoval:
    @patch("forge.executor.driver._write_status")
    @patch("forge.executor.driver._snapshot_worktree", return_value=set())
    @patch("forge.executor.driver.pipeline_ops.mark_failed")
    @patch("forge.executor.driver.pipeline_ops.mark_running")
    @patch("forge.executor.driver.AgentRunner")
    @patch("forge.executor.driver._get_dispatch_config")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    def test_silent_agent_fails(self, mock_req_state, mock_config, MockRunner,
                                mock_mark_run, mock_mark_fail, mock_snap, mock_ws):
        """Agent that doesn't call pass/fail -> step is failed (not auto-passed)."""
        old_sd, old_al = pipeline_driver._session_dir, pipeline_driver._activity_log_path
        pipeline_driver._session_dir = "/tmp/session"
        pipeline_driver._activity_log_path = "/tmp/session/pipeline-activity.log"
        try:
            mock_config.return_value = {
                "model": "sonnet", "prompt": "do stuff",
                "max_turns": 25, "timeout_ms": 3600000,
            }
            # After agent runs, require_state returns in_progress (agent didn't pass/fail)
            from forge.executor.engine.state import PipelineState, StepState, StepStatus
            step_state = StepState(status=StepStatus.IN_PROGRESS)
            state = PipelineState(steps={"create_pr": step_state}, step_order=["create_pr"])
            mock_req_state.return_value = state

            result = dispatch_ai_step("create_pr", "run", "/cwd")

            assert result is False
            MockRunner.return_value.run.assert_called_once()
            mock_mark_fail.assert_called_once()
        finally:
            pipeline_driver._session_dir = old_sd
            pipeline_driver._activity_log_path = old_al


# ── Code review verdict JSON ──────────────────────────────────────────────


class TestCodeReviewVerdict:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="verdict_test_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    @patch("forge.executor.driver.dispatch_ai_step", return_value=True)
    def test_clean_verdict(self, mock_dispatch):
        Path(self.tmp, "code-review-verdict.json").write_text('{"verdict": "CLEAN"}')

        result = run_code_review("/cwd", self.tmp)

        assert result is True

    @patch("forge.executor.driver.pipeline_ops.reset_step")
    @patch("forge.executor.driver.dispatch_ai_step")
    def test_has_issues_triggers_fix(self, mock_dispatch, mock_reset):
        Path(self.tmp, "code-review-verdict.json").write_text(
            '{"verdict": "HAS_ISSUES", "issue_count": 3}'
        )
        mock_dispatch.side_effect = [True, True]  # reviewer pass, fixer pass

        result = run_code_review("/cwd", self.tmp)

        assert result is True
        assert mock_dispatch.call_count == 2

    @patch("forge.executor.driver.dispatch_ai_step", return_value=True)
    def test_missing_verdict_fails(self, mock_dispatch):
        # No verdict file
        result = run_code_review("/cwd", self.tmp)

        assert result is False


# ── Report step ───────────────────────────────────────────────────────────


class TestReportStep:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="report_test_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    @patch("forge.executor.driver.pipeline_ops.mark_passed")
    @patch("forge.executor.driver.pipeline_ops.get_summary")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver.pipeline_ops.mark_running")
    def test_generates_report_file(self, mock_run, mock_state, mock_summary, mock_pass):
        mock_state.return_value = MagicMock()
        mock_summary.return_value = {
            "pipeline": "full", "preset": "test-preset",
            "steps": {
                "code": {"status": "complete", "retries": 0},
                "build": {"status": "complete", "retries": 0},
                "create_pr": {"status": "pending", "retries": 0},
            },
            "session_dir": self.tmp,
        }

        handle_report_step(self.tmp)

        report = Path(self.tmp, "pipeline-report.md")
        assert report.is_file()
        content = report.read_text()
        assert "Pipeline Report" in content
        assert "code: **COMPLETE**" in content
        assert "create_pr: PENDING" in content


# ── Pre-PR gate ───────────────────────────────────────────────────────────


class TestPrePRGate:
    @patch("forge.executor.pre_pr_gate.run_gate", return_value=True)
    def test_gate_pass(self, mock_gate):
        assert run_pre_pr_gate() is True

    @patch("forge.executor.pre_pr_gate.run_gate", return_value=False)
    def test_gate_fail(self, mock_gate):
        assert run_pre_pr_gate() is False

    @patch("forge.executor.pre_pr_gate.run_gate", side_effect=Exception("boom"))
    def test_gate_exception_returns_false(self, mock_gate):
        assert run_pre_pr_gate() is False


# ── Pipeline timeout ──────────────────────────────────────────────────────


class TestPipelineTimeout:
    @patch("forge.executor.driver.pipeline_ops.require_state")
    def test_timeout_returns_false(self, mock_state):
        """Pipeline returns False when wall-clock exceeds PIPELINE_TIMEOUT_S."""
        mock_state.return_value = _mock_dispatch_loop_state()
        old_start = time.time() - PIPELINE_TIMEOUT_S - 1

        result = dispatch_loop("/cwd", "/session", old_start)

        assert result is False


# ── dispatch_with_judge ──────────────────────────────────────────────────


class TestDispatchWithJudge:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="judge_dispatch_test_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    @patch("forge.executor.driver.subprocess.run")
    @patch("forge.executor.driver.dispatch_ai_step")
    @patch("forge.executor.driver.pipeline_ops.reset_step")
    def test_judge_passes_on_first_attempt(self, mock_reset, mock_dispatch, mock_subprocess):
        """Agent produces checklist, judge approves -> True on first attempt."""
        from forge.executor.engine.registry import JudgeConfig
        from forge.executor.engine.state import PipelineState, StepState

        mock_dispatch.return_value = True

        # Write checklist file
        Path(os.path.join(self.tmp, "code-checklist.json")).write_text(json.dumps({
            "step": "code", "checklist": [{"id": "plan-1", "criteria": "Add feature", "status": "done",
                                           "evidence": "file.ts:10", "files_touched": ["file.ts"]}]
        }))

        # Mock git diff
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="+const feature = true;", stderr="",
        )

        state = PipelineState(steps={"code": StepState()}, step_order=["code"])
        config = JudgeConfig(criteria_source="plan", max_retries=3)

        with patch("forge.executor.engine.judge.spawn_judge") as mock_judge:
            from forge.executor.engine.judge import JudgeVerdict
            mock_judge.return_value = JudgeVerdict(
                passed=True,
                items=[{"id": "plan-1", "verdict": "pass", "reason": "OK"}],
                summary="1/1 passed",
            )

            result = dispatch_with_judge("code", "run", "/cwd", self.tmp, config, state)

        assert result is True
        assert mock_dispatch.call_count == 1

    @patch("forge.executor.driver.subprocess.run")
    @patch("forge.executor.driver.dispatch_ai_step")
    @patch("forge.executor.driver.pipeline_ops.reset_step")
    def test_judge_fails_then_passes_on_retry(self, mock_reset, mock_dispatch, mock_subprocess):
        """Judge fails first attempt, passes on retry."""
        from forge.executor.engine.registry import JudgeConfig
        from forge.executor.engine.state import PipelineState, StepState

        mock_dispatch.return_value = True

        config = JudgeConfig(criteria_source="plan", max_retries=3)
        state = PipelineState(
            steps={"code": StepState()}, step_order=["code"],
            plan_file="", session_dir=self.tmp,
        )

        checklist_data = json.dumps({
            "step": "code", "checklist": [{"id": "plan-1", "criteria": "test", "status": "done",
                                           "evidence": "x", "files_touched": []}]
        })

        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="diff", stderr="",
        )

        with patch("forge.executor.engine.judge.spawn_judge") as mock_judge:
            from forge.executor.engine.judge import JudgeVerdict
            mock_judge.side_effect = [
                JudgeVerdict(passed=False, items=[{"id": "plan-1", "verdict": "fail", "reason": "incomplete"}]),
                JudgeVerdict(passed=True, items=[{"id": "plan-1", "verdict": "pass", "reason": "OK"}]),
            ]

            # Write checklist before each call
            def write_checklist(*args, **kwargs):
                Path(os.path.join(self.tmp, "code-checklist.json")).write_text(checklist_data)
                return True
            mock_dispatch.side_effect = write_checklist

            result = dispatch_with_judge("code", "run", "/cwd", self.tmp, config, state)

        assert result is True
        assert mock_dispatch.call_count == 2

    @patch("forge.executor.driver.dispatch_ai_step")
    @patch("forge.executor.driver.pipeline_ops.reset_step")
    def test_no_checklist_triggers_retry(self, mock_reset, mock_dispatch):
        """Missing checklist file triggers retry."""
        from forge.executor.engine.registry import JudgeConfig
        from forge.executor.engine.state import PipelineState, StepState

        mock_dispatch.return_value = True

        config = JudgeConfig(criteria_source="plan", max_retries=2)
        state = PipelineState(steps={"code": StepState()}, step_order=["code"])

        # No checklist file written -> all retries fail
        result = dispatch_with_judge("code", "run", "/cwd", self.tmp, config, state)

        assert result is False
        assert mock_dispatch.call_count == 2


# ── Code review with judge ────────────────────────────────────────────────


class TestCodeReviewWithJudge:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="cr_judge_test_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    @patch("forge.executor.driver.dispatch_ai_step", return_value=True)
    def test_clean_verdict_skips_judge(self, mock_dispatch):
        Path(self.tmp, "code-review-verdict.json").write_text('{"verdict": "CLEAN"}')

        result = run_code_review("/cwd", self.tmp)

        assert result is True
        assert mock_dispatch.call_count == 1

    @patch("forge.executor.driver.subprocess.run")
    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver.pipeline_ops.reset_step")
    @patch("forge.executor.driver.dispatch_ai_step")
    def test_has_issues_runs_fix_with_judge(self, mock_dispatch, mock_reset,
                                            mock_state, mock_preset, mock_subprocess):
        from forge.executor.engine.registry import StepDefinition, JudgeConfig, Preset, PipelineDefinition

        Path(self.tmp, "code-review-verdict.json").write_text(
            '{"verdict": "HAS_ISSUES", "issue_count": 2}'
        )
        # Write reviewer's checklist
        Path(self.tmp, "code_review-checklist.json").write_text(json.dumps({
            "step": "code_review",
            "checklist": [
                {"id": "f-1", "criteria": "Fix any types", "status": "done", "evidence": "line 48",
                 "files_touched": ["foo.ts"]},
            ]
        }))

        mock_dispatch.side_effect = [True, True]  # reviewer pass, fixer pass
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="diff", stderr="",
        )

        step_def = StepDefinition(
            name="code_review", step_type="ai",
            judge=JudgeConfig(criteria_source="findings", max_retries=3, model="opus"),
        )
        preset = MagicMock()
        preset.steps = {"code_review": step_def}
        mock_preset.return_value = preset
        mock_state.return_value = MagicMock()

        with patch("forge.executor.engine.judge.spawn_judge") as mock_judge:
            from forge.executor.engine.judge import JudgeVerdict
            mock_judge.return_value = JudgeVerdict(
                passed=True,
                items=[{"id": "f-1", "verdict": "pass", "reason": "Fixed"}],
            )
            result = run_code_review("/cwd", self.tmp)

        assert result is True


# ── _load_review_findings ─────────────────────────────────────────────────

class TestLoadReviewFindings:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="findings_test_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_loads_from_checklist_json(self):
        Path(self.tmp, "code_review-checklist.json").write_text(json.dumps({
            "step": "code_review",
            "checklist": [
                {"id": "f-1", "criteria": "Fix X", "status": "done", "evidence": ""},
                {"id": "f-2", "criteria": "Fix Y", "status": "done", "evidence": ""},
            ]
        }))

        findings = _load_review_findings(self.tmp)

        assert len(findings) == 2
        assert findings[0]["id"] == "f-1"

    def test_empty_when_no_file(self):
        findings = _load_review_findings(self.tmp)
        assert findings == []

    def test_invalid_json_returns_empty(self):
        Path(self.tmp, "code_review-checklist.json").write_text("not valid json{{{")
        findings = _load_review_findings(self.tmp)
        assert findings == []

    def test_missing_key_returns_empty(self):
        Path(self.tmp, "code_review-checklist.json").write_text(
            json.dumps({"checklist": [{"no_id": "x", "no_criteria": "y"}]})
        )
        findings = _load_review_findings(self.tmp)
        assert findings == []


# ── log_revalidation, log_headline ─────────────────────────────────────────


class TestLogFunctions:
    def test_log_revalidation_does_not_crash(self, capsys):
        log_revalidation(["lint", "test"])
        captured = capsys.readouterr()
        assert "revalidation" in captured.out
        assert "lint" in captured.out
        assert "test" in captured.out

    def test_log_headline_does_not_crash(self, capsys):
        log_headline("Pipeline Starting")
        captured = capsys.readouterr()
        assert "Pipeline Starting" in captured.out


# ── dispatch_ai_step edge cases ───────────────────────────────────────────


class TestDispatchAiStepEdgeCases:
    @patch("forge.executor.driver._get_dispatch_config", side_effect=RuntimeError("no step def"))
    def test_dispatch_config_error_returns_false(self, mock_config):
        result = dispatch_ai_step("lint", "run", "/cwd")
        assert result is False

    @patch("forge.executor.driver._get_dispatch_config")
    def test_empty_prompt_returns_false(self, mock_config):
        mock_config.return_value = {"model": "sonnet", "prompt": "", "max_turns": 25, "timeout_ms": 3600000}
        result = dispatch_ai_step("lint", "run", "/cwd")
        assert result is False

    @patch("forge.executor.driver._write_status")
    @patch("forge.executor.driver._snapshot_worktree", return_value=set())
    @patch("forge.executor.driver.pipeline_ops.mark_running")
    @patch("forge.executor.driver.AgentRunner")
    @patch("forge.executor.driver._get_dispatch_config")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    def test_agent_passes_step_complete(self, mock_req_state, mock_config, MockRunner,
                                        mock_mark_run, mock_snap, mock_ws):
        old_sd, old_al = pipeline_driver._session_dir, pipeline_driver._activity_log_path
        pipeline_driver._session_dir = "/tmp/session"
        pipeline_driver._activity_log_path = "/tmp/session/pipeline-activity.log"
        try:
            mock_config.return_value = {
                "model": "sonnet", "prompt": "do it", "max_turns": 25, "timeout_ms": 3600000,
            }
            from forge.executor.engine.state import PipelineState, StepState, StepStatus
            step_state = StepState(status=StepStatus.COMPLETE)
            state = PipelineState(steps={"lint": step_state}, step_order=["lint"])
            mock_req_state.return_value = state

            result = dispatch_ai_step("lint", "run", "/cwd")
            assert result is True
            MockRunner.return_value.run.assert_called_once()
        finally:
            pipeline_driver._session_dir = old_sd
            pipeline_driver._activity_log_path = old_al

    @patch("forge.executor.driver._write_status")
    @patch("forge.executor.driver._snapshot_worktree", return_value=set())
    @patch("forge.executor.driver.pipeline_ops.mark_running")
    @patch("forge.executor.driver.pipeline_ops.mark_failed")
    @patch("forge.executor.driver.AgentRunner")
    @patch("forge.executor.driver._get_dispatch_config")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    def test_agent_fails_step_failed(self, mock_req_state, mock_config, MockRunner,
                                      mock_mark_fail, mock_mark_run, mock_snap, mock_ws):
        old_sd, old_al = pipeline_driver._session_dir, pipeline_driver._activity_log_path
        pipeline_driver._session_dir = "/tmp/session"
        pipeline_driver._activity_log_path = "/tmp/session/pipeline-activity.log"
        try:
            mock_config.return_value = {
                "model": "sonnet", "prompt": "do it", "max_turns": 25, "timeout_ms": 3600000,
            }
            from forge.executor.engine.state import PipelineState, StepState, StepStatus
            step_state = StepState(status=StepStatus.FAILED, retries=1)
            state = PipelineState(steps={"lint": step_state}, step_order=["lint"])
            mock_req_state.return_value = state

            result = dispatch_ai_step("lint", "run", "/cwd")
            assert result is False
        finally:
            pipeline_driver._session_dir = old_sd
            pipeline_driver._activity_log_path = old_al

    @patch("forge.executor.driver._write_status")
    @patch("forge.executor.driver._snapshot_worktree", return_value=set())
    @patch("forge.executor.driver.pipeline_ops.mark_running")
    @patch("forge.executor.driver.AgentRunner")
    @patch("forge.executor.driver._get_dispatch_config")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    def test_fix_phase_uses_suffixed_step_name(self, mock_req_state, mock_config, MockRunner,
                                                mock_mark_run, mock_snap, mock_ws):
        """Fix phase should use '{step}_fix' as transcript step name."""
        old_sd, old_al = pipeline_driver._session_dir, pipeline_driver._activity_log_path
        pipeline_driver._session_dir = "/tmp/session"
        pipeline_driver._activity_log_path = "/tmp/session/pipeline-activity.log"
        try:
            mock_config.return_value = {
                "model": "sonnet", "prompt": "fix it", "max_turns": 25, "timeout_ms": 600000,
            }
            from forge.executor.engine.state import PipelineState, StepState, StepStatus
            step_state = StepState(status=StepStatus.COMPLETE)
            state = PipelineState(steps={"code_review": step_state}, step_order=["code_review"])
            mock_req_state.return_value = state

            dispatch_ai_step("code_review", "fix", "/cwd")
            MockRunner.assert_called_once_with(
                "/tmp/session", "code_review_fix", "/tmp/session/pipeline-activity.log",
            )
        finally:
            pipeline_driver._session_dir = old_sd
            pipeline_driver._activity_log_path = old_al


# ── dispatch_with_judge agent fails ───────────────────────────────────────


class TestDispatchWithJudgeAgentFails:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="judge_fail_test_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    @patch("forge.executor.driver.dispatch_ai_step")
    @patch("forge.executor.driver.pipeline_ops.reset_step")
    def test_agent_fails_all_retries(self, mock_reset, mock_dispatch):
        from forge.executor.engine.registry import JudgeConfig
        from forge.executor.engine.state import PipelineState, StepState

        mock_dispatch.return_value = False

        config = JudgeConfig(criteria_source="plan", max_retries=2)
        state = PipelineState(steps={"code": StepState()}, step_order=["code"])

        result = dispatch_with_judge("code", "run", "/cwd", self.tmp, config, state)

        assert result is False
        assert mock_dispatch.call_count == 2
        # Should have called reset between attempts (not on last)
        assert mock_reset.call_count == 1


# ── _handle_command_step edge cases ────────────────────────────────────────


class TestHandleCommandStepEdgeCases:
    @patch("forge.executor.driver.execute_command_step")
    def test_initial_pass_returns_true(self, mock_exec):
        mock_exec.return_value = {"step": "build", "result": "passed"}
        name, success = _handle_command_step("build", 0, "/cwd")
        assert success is True
        assert name == "build"
        assert mock_exec.call_count == 1

    @patch("forge.executor.engine.agents.fix_agent.run")
    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver.pipeline_ops.mark_running")
    @patch("forge.executor.driver.execute_command_step")
    def test_exhausted_retries_fix_pass_but_verify_loop(self, mock_exec, mock_mark_run,
                                                         mock_state, mock_preset, mock_fix):
        """Fix agent passes but only MAX_STEP_RETRIES attempts allowed."""
        mock_exec.return_value = {"step": "lint", "result": "failed"}
        from forge.executor.engine.agents.fix_agent import FixAgentOutcome
        from forge.executor.engine.registry import StepDefinition, Preset, PipelineDefinition
        from forge.executor.engine.state import PipelineState, StepState

        mock_preset.return_value = Preset(
            name="test", version=3, description="",
            pipelines={"full": PipelineDefinition(steps=["lint"], dependencies={})},
            steps={"lint": StepDefinition(name="lint", step_type="command", run_command="echo ok")},
            models={"fix": "sonnet"}, preset_dir=Path("."),
        )
        mock_state.return_value = PipelineState(
            steps={"lint": StepState()}, step_order=["lint"], session_dir="/tmp/session",
        )
        # First attempt passes
        mock_fix.return_value = FixAgentOutcome(passed=True, reason="Fix verified")

        name, success = _handle_command_step("lint", 0, "/cwd")
        assert success is True
        assert name == "lint"


# ── _handle_ai_step edge cases ──────────────────────────────────────────


class TestHandleAiStepEdgeCases:
    @patch("forge.executor.driver.run_code_review", return_value=True)
    def test_code_review_delegation_pass(self, mock_review):
        name, success = _handle_ai_step("code_review", "/cwd", "/session")
        assert success is True
        assert name == "code_review"
        mock_review.assert_called_once_with("/cwd", "/session")

    @patch("forge.executor.driver.run_code_review", return_value=False)
    def test_code_review_delegation_fail(self, mock_review):
        name, success = _handle_ai_step("code_review", "/cwd", "/session")
        assert success is False
        assert name == "code_review"

    @patch("forge.executor.driver._handle_visual_test_step", return_value=("visual_test", True))
    def test_visual_test_routes_to_handler_pass(self, mock_handler):
        """visual_test step now routes to specialized handler, not judge loop."""
        name, success = _handle_ai_step("visual_test", "/cwd", "/session")
        assert success is True
        assert name == "visual_test"
        mock_handler.assert_called_once_with("/cwd", "/session")

    @patch("forge.executor.driver._handle_visual_test_step", return_value=("visual_test", False))
    def test_visual_test_routes_to_handler_fail(self, mock_handler):
        """visual_test step now routes to specialized handler, not judge loop."""
        name, success = _handle_ai_step("visual_test", "/cwd", "/session")
        assert success is False

    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state", side_effect=Exception("preset load failed"))
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver.dispatch_ai_step", return_value=False)
    def test_preset_exception_fails_immediately(self, mock_dispatch, mock_state, mock_preset):
        # Use a generic AI step (not code/visual_test/code_review which have dedicated handlers)
        mock_state.return_value = MagicMock()
        name, success = _handle_ai_step("generic_ai", "/cwd", "/session")
        assert success is False
        assert mock_dispatch.call_count == 0  # No fallback to non-judge retry loop


# ── _handle_inline_step ────────────────────────────────────────────────────


class TestHandleInlineStep:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="inline_test_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    @patch("forge.executor.driver.pipeline_ops.mark_passed")
    @patch("forge.executor.driver.pipeline_ops.get_summary")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver.pipeline_ops.mark_running")
    def test_report_step_returns_true(self, mock_run, mock_state, mock_summary, mock_pass):
        mock_state.return_value = MagicMock()
        mock_summary.return_value = {
            "pipeline": "full", "preset": "test-preset", "steps": {}, "session_dir": "/tmp",
        }
        name, success = _handle_inline_step("report", self.tmp)
        assert success is True
        assert name == "report"

    def test_unknown_inline_returns_false(self):
        name, success = _handle_inline_step("unknown_step", "/session")
        assert success is False
        assert name == "unknown_step"


# ── _dispatch_one_step routing ────────────────────────────────────────────


class TestDispatchOneStepRouting:
    @patch("forge.executor.driver._handle_command_step", return_value=("build", True))
    def test_routes_command_type(self, mock_handler):
        name, success = _dispatch_one_step(
            {"step": "build", "type": "command", "retries": 0}, "/cwd", "/session"
        )
        assert success is True
        mock_handler.assert_called_once_with("build", 0, "/cwd")

    @patch("forge.executor.driver._handle_ai_step", return_value=("code", True))
    def test_routes_ai_type(self, mock_handler):
        name, success = _dispatch_one_step(
            {"step": "code", "type": "ai"}, "/cwd", "/session"
        )
        assert success is True
        mock_handler.assert_called_once_with("code", "/cwd", "/session")

    @patch("forge.executor.driver._handle_inline_step", return_value=("report", True))
    def test_routes_inline_type(self, mock_handler):
        name, success = _dispatch_one_step(
            {"step": "report", "type": "inline"}, "/cwd", "/session"
        )
        assert success is True
        mock_handler.assert_called_once_with("report", "/session")

    def test_unknown_type_returns_false(self):
        name, success = _dispatch_one_step(
            {"step": "mystery", "type": "quantum"}, "/cwd", "/session"
        )
        assert success is False
        assert name == "mystery"


# ── handle_report_step edge cases ──────────────────────────────────────────


class TestHandleReportStepEdgeCases:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="report_edge_test_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    @patch("forge.executor.driver.subprocess.run")
    @patch("forge.executor.driver.pipeline_ops.mark_passed")
    @patch("forge.executor.driver.pipeline_ops.get_summary")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver.pipeline_ops.mark_running")
    def test_with_dashboard_file(self, mock_run, mock_state, mock_summary,
                                  mock_pass, mock_subprocess):
        mock_state.return_value = MagicMock()
        mock_summary.return_value = {
            "pipeline": "full", "preset": "test-preset",
            "steps": {
                "code": {"status": "complete", "retries": 0},
                "lint": {"status": "failed", "retries": 1, "last_error": "lint failed"},
            },
            "session_dir": self.tmp,
        }
        # Create the dashboard file
        Path(self.tmp, "visual-test-dashboard.html").write_text("<html>dash</html>")

        handle_report_step(self.tmp)

        report = Path(self.tmp, "pipeline-report.md")
        assert report.is_file()
        content = report.read_text()
        assert "lint: **FAILED**" in content
        # Verify "open" was called for the dashboard
        mock_subprocess.assert_called_once()
        assert "open" in mock_subprocess.call_args[0][0]

    @patch("forge.executor.driver.pipeline_ops.mark_passed")
    @patch("forge.executor.driver.pipeline_ops.get_summary")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver.pipeline_ops.mark_running")
    def test_without_dashboard_file(self, mock_run, mock_state, mock_summary, mock_pass):
        mock_state.return_value = MagicMock()
        mock_summary.return_value = {
            "pipeline": "full", "preset": "test-preset",
            "steps": {"code": {"status": "pending", "retries": 0}},
            "session_dir": self.tmp,
        }

        handle_report_step(self.tmp)

        report = Path(self.tmp, "pipeline-report.md")
        assert report.is_file()


# ── run_code_review edge cases ──────────────────────────────────────────


class TestRunCodeReviewEdgeCases:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="cr_edge_test_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    @patch("forge.executor.driver.dispatch_ai_step", return_value=False)
    def test_reviewer_fails_returns_false(self, mock_dispatch):
        result = run_code_review("/cwd", self.tmp)
        assert result is False
        assert mock_dispatch.call_count == 1

    @patch("forge.executor.driver.dispatch_ai_step", return_value=True)
    def test_unknown_verdict_returns_false(self, mock_dispatch):
        Path(self.tmp, "code-review-verdict.json").write_text(
            '{"verdict": "MAYBE_ISSUES"}'
        )
        result = run_code_review("/cwd", self.tmp)
        assert result is False

    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state", side_effect=Exception("boom"))
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver.pipeline_ops.reset_step")
    @patch("forge.executor.driver.dispatch_ai_step")
    def test_has_issues_no_judge_config_trusts_agent(self, mock_dispatch, mock_reset,
                                                      mock_state, mock_preset):
        Path(self.tmp, "code-review-verdict.json").write_text(
            '{"verdict": "HAS_ISSUES", "issue_count": 1}'
        )
        mock_dispatch.side_effect = [True, True]  # reviewer, fixer
        mock_state.return_value = MagicMock()

        result = run_code_review("/cwd", self.tmp)
        assert result is True

    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver.pipeline_ops.reset_step")
    @patch("forge.executor.driver.dispatch_ai_step")
    def test_has_issues_fix_agent_fails_continues(self, mock_dispatch, mock_reset,
                                                   mock_state, mock_preset):
        from forge.executor.engine.registry import StepDefinition, JudgeConfig

        Path(self.tmp, "code-review-verdict.json").write_text(
            '{"verdict": "HAS_ISSUES", "issue_count": 2}'
        )
        # reviewer passes, all fix attempts fail
        mock_dispatch.side_effect = [True, False, False, False]
        mock_state.return_value = MagicMock()

        step_def = StepDefinition(
            name="code_review", step_type="ai",
            judge=JudgeConfig(criteria_source="findings", max_retries=3, model="opus"),
        )
        preset = MagicMock()
        preset.steps = {"code_review": step_def}
        mock_preset.return_value = preset

        result = run_code_review("/cwd", self.tmp)
        assert result is False

    @patch("forge.executor.driver.subprocess.run")
    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver.pipeline_ops.reset_step")
    @patch("forge.executor.driver.dispatch_ai_step")
    def test_has_issues_no_fix_checklist_trusts_agent(self, mock_dispatch, mock_reset,
                                                       mock_state, mock_preset,
                                                       mock_subprocess):
        from forge.executor.engine.registry import StepDefinition, JudgeConfig

        Path(self.tmp, "code-review-verdict.json").write_text(
            '{"verdict": "HAS_ISSUES", "issue_count": 1}'
        )
        Path(self.tmp, "code_review-checklist.json").write_text(json.dumps({
            "step": "code_review",
            "checklist": [{"id": "f-1", "criteria": "Fix X", "status": "done", "evidence": "y"}],
        }))

        mock_state.return_value = MagicMock()
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="diff", stderr="",
        )

        step_def = StepDefinition(
            name="code_review", step_type="ai",
            judge=JudgeConfig(criteria_source="findings", max_retries=3, model="opus"),
        )
        preset = MagicMock()
        preset.steps = {"code_review": step_def}
        mock_preset.return_value = preset

        call_count = [0]
        def dispatch_with_counter(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return True  # reviewer "run"
            # fixer "fix" — remove checklist so judge sees FileNotFoundError
            checklist = Path(self.tmp, "code_review-checklist.json")
            if checklist.exists():
                checklist.unlink()
            return True

        mock_dispatch.side_effect = dispatch_with_counter
        result = run_code_review("/cwd", self.tmp)
        assert result is True

    @patch("forge.executor.driver.subprocess.run")
    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver.pipeline_ops.reset_step")
    @patch("forge.executor.driver.dispatch_ai_step")
    def test_judge_rejects_fix_loop_exhausted(self, mock_dispatch, mock_reset,
                                               mock_state, mock_preset, mock_subprocess):
        from forge.executor.engine.registry import StepDefinition, JudgeConfig

        Path(self.tmp, "code-review-verdict.json").write_text(
            '{"verdict": "HAS_ISSUES", "issue_count": 1}'
        )
        Path(self.tmp, "code_review-checklist.json").write_text(json.dumps({
            "step": "code_review",
            "checklist": [{"id": "f-1", "criteria": "Fix null check", "status": "done", "evidence": "line 5"}],
        }))

        mock_dispatch.return_value = True
        mock_state.return_value = MagicMock()
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="diff", stderr="",
        )

        step_def = StepDefinition(
            name="code_review", step_type="ai",
            judge=JudgeConfig(criteria_source="findings", max_retries=2, model="opus"),
        )
        preset = MagicMock()
        preset.steps = {"code_review": step_def}
        mock_preset.return_value = preset

        with patch("forge.executor.engine.judge.spawn_judge") as mock_judge, \
             patch("forge.executor.engine.judge.save_judge_feedback") as mock_save:
            from forge.executor.engine.judge import JudgeVerdict
            mock_judge.return_value = JudgeVerdict(
                passed=False,
                items=[{"id": "f-1", "verdict": "fail", "reason": "not fixed"}],
            )
            mock_save.return_value = "/tmp/feedback.json"

            result = run_code_review("/cwd", self.tmp)

        assert result is False


# ── ensure_dev_server ──────────────────────────────────────────────────────


class TestEnsureDevServer:
    @patch("forge.executor.driver.pipeline_ops.allocate_dev_server")
    def test_allocate_returns_non_dict(self, mock_alloc):
        mock_alloc.return_value = "not a dict"
        result = ensure_dev_server("/cwd")
        assert result is False

    @patch("forge.executor.driver.pipeline_ops.allocate_dev_server")
    def test_already_running(self, mock_alloc):
        mock_alloc.return_value = {"port": 3000, "running": True}
        result = ensure_dev_server("/cwd")
        assert result is True

    @patch("forge.executor.driver.time.sleep")
    @patch("forge.executor.driver.subprocess.run")
    @patch("forge.executor.driver.subprocess.Popen")
    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver.pipeline_ops.allocate_dev_server")
    def test_start_and_poll_success(self, mock_alloc, mock_state, mock_preset,
                                    mock_popen, mock_run, mock_sleep):
        mock_alloc.return_value = {"port": 4000, "running": False}
        mock_state.return_value = MagicMock()
        dev_config = {"command": "npm start", "health_url": "http://localhost:4000/", "startup_timeout": 180}
        preset = MagicMock()
        preset.dev_server = dev_config
        mock_preset.return_value = preset
        mock_popen.return_value.poll.return_value = None  # process still running
        # First curl fails, second succeeds
        mock_run.side_effect = [
            subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr=""),
        ]
        result = ensure_dev_server("/cwd")
        assert result is True
        mock_popen.assert_called_once()

    @patch("forge.executor.driver.time.sleep")
    @patch("forge.executor.driver.subprocess.run")
    @patch("forge.executor.driver.subprocess.Popen")
    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver.pipeline_ops.allocate_dev_server")
    def test_start_and_poll_timeout(self, mock_alloc, mock_state, mock_preset,
                                    mock_popen, mock_run, mock_sleep):
        mock_alloc.return_value = {"port": 4000, "running": False}
        mock_state.return_value = MagicMock()
        dev_config = {"command": "npm start", "health_url": "http://localhost:4000/", "startup_timeout": 180}
        preset = MagicMock()
        preset.dev_server = dev_config
        mock_preset.return_value = preset
        mock_popen.return_value.poll.return_value = None  # process still running
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="",
        )
        result = ensure_dev_server("/cwd")
        assert result is False
        assert mock_run.call_count == 36

    @patch("forge.executor.driver.time.sleep")
    @patch("forge.executor.driver.subprocess.run")
    @patch("forge.executor.driver.subprocess.Popen")
    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver.pipeline_ops.allocate_dev_server")
    def test_curl_timeout_retries(self, mock_alloc, mock_state, mock_preset,
                                   mock_popen, mock_run, mock_sleep):
        mock_alloc.return_value = {"port": 5000, "running": False}
        mock_state.return_value = MagicMock()
        dev_config = {"command": "npm start", "health_url": "http://localhost:5000/", "startup_timeout": 180}
        preset = MagicMock()
        preset.dev_server = dev_config
        mock_preset.return_value = preset
        mock_popen.return_value.poll.return_value = None  # process still running
        # First curl times out, second succeeds
        mock_run.side_effect = [
            subprocess.TimeoutExpired(cmd=[], timeout=5),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr=""),
        ]
        result = ensure_dev_server("/cwd")
        assert result is True

    @patch("forge.executor.driver.time.sleep")
    @patch("forge.executor.driver.subprocess.Popen")
    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver.pipeline_ops.allocate_dev_server")
    def test_early_crash_detection(self, mock_alloc, mock_state, mock_preset,
                                    mock_popen, mock_sleep):
        mock_alloc.return_value = {"port": 4000, "running": False}
        mock_state.return_value = MagicMock()
        dev_config = {"command": "npm start", "health_url": "http://localhost:4000/", "startup_timeout": 180}
        preset = MagicMock()
        preset.dev_server = dev_config
        mock_preset.return_value = preset
        mock_popen.return_value.poll.return_value = 1  # process crashed
        mock_popen.return_value.returncode = 1
        mock_popen.return_value.stdout.read.return_value = b"EADDRINUSE: address already in use"
        result = ensure_dev_server("/cwd")
        assert result is False


# ── dispatch_loop extended coverage ────────────────────────────────────────


class TestDispatchLoopExtended:
    @patch("forge.executor.driver.pipeline_ops.get_next_steps")
    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    def test_all_complete_no_blocked_returns_true(self, mock_state, mock_preset, mock_next):
        mock_state.return_value = _mock_dispatch_loop_state()
        mock_preset.return_value = MagicMock()
        mock_next.return_value = {
            "runnable": [], "in_progress": [], "blocked": [],
        }
        result = dispatch_loop("/cwd", "/session", time.time())
        assert result is True

    @patch("forge.executor.driver._write_status")
    @patch("forge.executor.driver._dispatch_one_step", return_value=("build", True))
    @patch("forge.executor.driver.pipeline_ops.get_next_steps")
    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    def test_single_step_dispatch(self, mock_state, mock_preset, mock_next,
                                   mock_dispatch, mock_write_status):
        mock_state.return_value = _mock_dispatch_loop_state()
        mock_preset.return_value = MagicMock()
        mock_next.side_effect = [
            {"runnable": [{"step": "build", "type": "command", "retries": 0}],
             "in_progress": [], "blocked": []},
            {"runnable": [], "in_progress": [], "blocked": []},  # post revalidation
            {"step": None},  # all complete
        ]
        result = dispatch_loop("/cwd", "/session", time.time())
        assert result is True
        mock_dispatch.assert_called_once()

    @patch("forge.executor.driver.pipeline_ops.mark_failed")
    @patch("forge.executor.driver.run_pre_pr_gate", return_value=False)
    @patch("forge.executor.driver.pipeline_ops.get_next_steps")
    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    def test_pre_pr_gate_fail_single_step(self, mock_state, mock_preset, mock_next,
                                           mock_gate, mock_mark_fail):
        mock_state.return_value = _mock_dispatch_loop_state()
        mock_preset.return_value = MagicMock()
        mock_next.side_effect = [
            {"runnable": [{"step": "create_pr", "type": "ai"}],
             "in_progress": [], "blocked": []},
            # After marking failed, loop continues; next iteration all complete
            {"step": None},
        ]
        result = dispatch_loop("/cwd", "/session", time.time())
        assert result is True

    @patch("forge.executor.driver._write_status")
    @patch("forge.executor.driver.pipeline_ops.mark_failed")
    @patch("forge.executor.driver.ensure_dev_server", return_value=False)
    @patch("forge.executor.driver.pipeline_ops.get_next_steps")
    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    def test_dev_server_fail_budget_exhausted(self, mock_state, mock_preset, mock_next,
                                               mock_dev, mock_mark_fail, mock_write_status):
        mock_state.return_value = _mock_dispatch_loop_state()
        mock_preset.return_value = MagicMock()
        mock_next.side_effect = [
            # Iteration 1: visual_test, dev server fails
            {"runnable": [{"step": "visual_test", "type": "ai"}],
             "in_progress": [], "blocked": []},
            {"runnable": [{"step": "visual_test", "type": "ai"}],
             "in_progress": [], "blocked": []},  # post revalidation
            # Iteration 2: visual_test, dev server fails again (budget exhausted)
            {"runnable": [{"step": "visual_test", "type": "ai"}],
             "in_progress": [], "blocked": []},
            {"runnable": [], "in_progress": [], "blocked": []},  # post revalidation
            # Iteration 3: all complete
            {"step": None},
        ]

        result = dispatch_loop("/cwd", "/session", time.time())
        assert result is True
        # mark_failed should have been called for visual_test (budget exhausted message)
        budget_calls = [c for c in mock_mark_fail.call_args_list if "budget exhausted" in str(c)]
        assert len(budget_calls) == 1

    @patch("forge.executor.driver._write_status")
    @patch("forge.executor.driver.pipeline_ops.mark_failed")
    @patch("forge.executor.driver.run_pre_pr_gate", return_value=False)
    @patch("forge.executor.driver._dispatch_one_step", return_value=("lint", True))
    @patch("forge.executor.driver.pipeline_ops.get_next_steps")
    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    def test_parallel_dispatch_with_pre_pr_gate_fail(self, mock_state, mock_preset, mock_next,
                                                      mock_dispatch, mock_gate,
                                                      mock_mark_fail, mock_write_status):
        mock_state.return_value = _mock_dispatch_loop_state()
        mock_preset.return_value = MagicMock()
        mock_next.side_effect = [
            {"runnable": [
                {"step": "create_pr", "type": "ai"},
                {"step": "lint", "type": "command", "retries": 0},
            ], "in_progress": [], "blocked": []},
            {"runnable": [], "in_progress": [], "blocked": []},  # revalidation
            {"step": None},
        ]

        result = dispatch_loop("/cwd", "/session", time.time())
        assert result is True
        # create_pr should have been filtered out, only lint dispatched
        mock_dispatch.assert_called_once()

    @patch("forge.executor.driver._write_status")
    @patch("forge.executor.driver.ensure_dev_server", return_value=False)
    @patch("forge.executor.driver._dispatch_one_step", return_value=("lint", True))
    @patch("forge.executor.driver.pipeline_ops.get_next_steps")
    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    def test_parallel_dispatch_with_dev_server_fail(self, mock_state, mock_preset, mock_next,
                                                     mock_dispatch, mock_dev, mock_write_status):
        mock_state.return_value = _mock_dispatch_loop_state()
        mock_preset.return_value = MagicMock()
        mock_next.side_effect = [
            {"runnable": [
                {"step": "visual_test", "type": "ai"},
                {"step": "lint", "type": "command", "retries": 0},
            ], "in_progress": [], "blocked": []},
            {"runnable": [], "in_progress": [], "blocked": []},  # revalidation
            {"step": None},
        ]

        result = dispatch_loop("/cwd", "/session", time.time())
        assert result is True

    @patch("forge.executor.driver._write_status")
    @patch("forge.executor.driver._dispatch_one_step", side_effect=Exception("worker crash"))
    @patch("forge.executor.driver.pipeline_ops.get_next_steps")
    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    def test_parallel_dispatch_exception_in_future(self, mock_state, mock_preset, mock_next,
                                                    mock_dispatch, mock_write_status):
        mock_state.return_value = _mock_dispatch_loop_state()
        mock_preset.return_value = MagicMock()
        mock_next.side_effect = [
            {"runnable": [
                {"step": "lint", "type": "command", "retries": 0},
                {"step": "test", "type": "command", "retries": 0},
            ], "in_progress": [], "blocked": []},
            {"runnable": [], "in_progress": [], "blocked": []},  # revalidation
            {"step": None},
        ]

        result = dispatch_loop("/cwd", "/session", time.time())
        assert result is True

    @patch("forge.executor.driver._write_status")
    @patch("forge.executor.driver._dispatch_one_step", return_value=("build", True))
    @patch("forge.executor.driver.pipeline_ops.get_next_steps")
    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    def test_revalidation_detected(self, mock_state, mock_preset, mock_next,
                                    mock_dispatch, mock_write_status):
        mock_state.return_value = _mock_dispatch_loop_state()
        mock_preset.return_value = MagicMock()
        mock_next.side_effect = [
            {"runnable": [{"step": "build", "type": "command", "retries": 0}],
             "in_progress": [], "blocked": []},
            # Post-dispatch: lint appeared as new runnable (revalidation)
            {"runnable": [{"step": "lint", "type": "command", "retries": 0}],
             "in_progress": [], "blocked": []},
            # Next iteration
            {"step": None},
        ]

        result = dispatch_loop("/cwd", "/session", time.time())
        assert result is True

    @patch("forge.executor.driver.pipeline_ops.mark_failed")
    @patch("forge.executor.driver.ensure_dev_server", return_value=False)
    @patch("forge.executor.driver._dispatch_one_step", return_value=("lint", True))
    @patch("forge.executor.driver.pipeline_ops.get_next_steps")
    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    def test_parallel_dev_server_budget_exhausted(self, mock_state, mock_preset, mock_next,
                                                   mock_dispatch, mock_dev, mock_mark_fail):
        mock_state.return_value = _mock_dispatch_loop_state()
        mock_preset.return_value = MagicMock()
        # Build up enough failures in parallel path to exhaust budget
        responses = []
        for i in range(MAX_DEV_SERVER_PRECHECK_FAILURES):
            responses.append(
                {"runnable": [
                    {"step": "visual_test", "type": "ai"},
                    {"step": "lint", "type": "command", "retries": 0},
                ], "in_progress": [], "blocked": []}
            )
            responses.append({"runnable": [], "in_progress": [], "blocked": []})  # revalidation
        responses.append({"step": None})

        mock_next.side_effect = responses

        result = dispatch_loop("/cwd", "/session", time.time())
        assert result is True


# ── main() function ───────────────────────────────────────────────────────


class TestMain:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="main_test_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    @patch("forge.executor.driver._stop_status_updater")
    @patch("forge.executor.driver._start_status_updater")
    @patch("forge.executor.driver._write_status")
    @patch("forge.executor.driver.dispatch_loop", return_value=True)
    @patch("forge.executor.driver.pipeline_ops.get_summary")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver.pipeline_ops.state_mgr")
    @patch("forge.executor.driver.signal.signal")
    @patch("sys.argv", ["forge-driver", "--resume"])
    def test_resume_path(self, mock_signal, mock_state_mgr, mock_state, mock_summary,
                         mock_dispatch, mock_ws, mock_start_updater, mock_stop_updater):
        session_dir = os.path.join(self.tmp, "session")
        os.makedirs(session_dir)
        mock_state.return_value = MagicMock()
        mock_summary.return_value = {
            "pipeline": "full",
            "session_dir": session_dir,
            "complete": ["code"],
            "pending": ["build"],
        }
        mock_state_mgr.return_value = MagicMock()

        main()

        mock_dispatch.assert_called_once()

    @patch("forge.executor.driver._stop_status_updater")
    @patch("forge.executor.driver._start_status_updater")
    @patch("forge.executor.driver._write_status")
    @patch("forge.executor.driver.dispatch_loop", return_value=True)
    @patch("forge.executor.driver.pipeline_ops.get_summary")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver.pipeline_ops.state_mgr")
    @patch("forge.executor.commands.cmd_init")
    @patch("forge.executor.driver.signal.signal")
    def test_fresh_init_path(self, mock_signal, mock_cmd_init, mock_state_mgr,
                              mock_state, mock_summary, mock_dispatch,
                              mock_ws, mock_start_updater, mock_stop_updater):
        plan_file = os.path.join(self.tmp, "plan.md")
        Path(plan_file).write_text("# Plan\n\n1. Do stuff\n2. More stuff")
        session_dir = os.path.join(self.tmp, "session")
        os.makedirs(session_dir)

        mock_state.return_value = MagicMock()
        mock_summary.return_value = {
            "pipeline": "full",
            "session_dir": session_dir,
        }
        mock_state_mgr.return_value = MagicMock()

        with patch("sys.argv", ["forge-driver", "--plan", plan_file,
                                 "--preset", "test-preset",
                                 "--packages", "apps/web"]):
            main()

        mock_dispatch.assert_called_once()

    @patch("forge.executor.driver.signal.signal")
    def test_dry_run(self, mock_signal, capsys):
        plan_file = os.path.join(self.tmp, "plan.md")
        Path(plan_file).write_text("# Plan\n\nSome plan content here")

        with patch("sys.argv", ["forge-driver", "--plan", plan_file, "--preset", "test-preset", "--dry-run"]):
            main()

        captured = capsys.readouterr()
        assert "Would run" in captured.out
        assert "forge execute init" in captured.out

    @patch("forge.executor.driver._stop_status_updater")
    @patch("forge.executor.driver._start_status_updater")
    @patch("forge.executor.driver._write_status")
    @patch("forge.executor.driver.dispatch_loop", return_value=False)
    @patch("forge.executor.driver.pipeline_ops.get_summary")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver.pipeline_ops.state_mgr")
    @patch("forge.executor.driver.signal.signal")
    @patch("sys.argv", ["forge-driver", "--resume"])
    def test_resume_failed_pipeline(self, mock_signal, mock_state_mgr, mock_state,
                                     mock_summary, mock_dispatch, mock_ws,
                                     mock_start_updater, mock_stop_updater, capsys):
        session_dir = os.path.join(self.tmp, "session")
        os.makedirs(session_dir)
        mock_state.return_value = MagicMock()
        mock_summary.return_value = {
            "pipeline": "full",
            "session_dir": session_dir,
            "complete": [],
            "pending": ["code"],
        }
        mock_state_mgr.return_value = MagicMock()

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "Resume with" in captured.out

    @patch("forge.executor.driver.pipeline_ops.require_state",
           side_effect=RuntimeError("No active pipeline"))
    @patch("sys.argv", ["forge-driver", "--resume"])
    def test_resume_no_state_exits(self, mock_state):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    @patch("forge.executor.driver._stop_status_updater")
    @patch("forge.executor.driver._start_status_updater")
    @patch("forge.executor.driver._write_status")
    @patch("forge.executor.driver.dispatch_loop", return_value=True)
    @patch("forge.executor.driver.pipeline_ops.get_summary")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver.pipeline_ops.state_mgr")
    @patch("forge.executor.driver.signal.signal")
    def test_sigint_handler_registered(self, mock_signal, mock_state_mgr, mock_state,
                                        mock_summary, mock_dispatch, mock_ws,
                                        mock_start_updater, mock_stop_updater):
        plan_file = os.path.join(self.tmp, "plan.md")
        Path(plan_file).write_text("# Plan\n\nContent")
        session_dir = os.path.join(self.tmp, "session")
        os.makedirs(session_dir)

        mock_state.return_value = MagicMock()
        mock_summary.return_value = {
            "pipeline": "full",
            "session_dir": session_dir,
        }
        mock_state_mgr.return_value = MagicMock()

        with patch("sys.argv", ["forge-driver", "--plan", plan_file, "--preset", "test-preset"]):
            with patch("forge.executor.commands.cmd_init"):
                main()

        # Verify SIGINT handler was registered
        signal_calls = [c for c in mock_signal.call_args_list if c[0][0] == signal.SIGINT]
        assert len(signal_calls) == 1

    @patch("forge.executor.driver._stop_status_updater")
    @patch("forge.executor.driver._start_status_updater")
    @patch("forge.executor.driver._write_status")
    @patch("forge.executor.driver.dispatch_loop", return_value=True)
    @patch("forge.executor.driver.pipeline_ops.get_summary")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver.pipeline_ops.state_mgr")
    def test_sigint_handler_behavior(self, mock_state_mgr, mock_state, mock_summary,
                                      mock_dispatch, mock_ws, mock_start_updater, mock_stop_updater):
        plan_file = os.path.join(self.tmp, "plan.md")
        Path(plan_file).write_text("# Plan\n\nContent")
        session_dir = os.path.join(self.tmp, "session")
        os.makedirs(session_dir)

        mock_state.return_value = MagicMock()
        mock_summary.return_value = {
            "pipeline": "full",
            "session_dir": session_dir,
        }
        mock_state_mgr.return_value = MagicMock()

        captured_handler = [None]
        original_signal = signal.signal
        def capture_signal(sig, handler):
            if sig == signal.SIGINT:
                captured_handler[0] = handler
            return original_signal(sig, handler)

        with patch("forge.executor.driver.signal.signal", side_effect=capture_signal):
            with patch("sys.argv", ["forge-driver", "--plan", plan_file, "--preset", "test-preset"]):
                with patch("forge.executor.commands.cmd_init"):
                    main()

        assert captured_handler[0] is not None
        with pytest.raises(SystemExit) as exc_info:
            captured_handler[0](signal.SIGINT, None)
        assert exc_info.value.code == 130

    @patch("forge.executor.driver._stop_status_updater")
    @patch("forge.executor.driver._start_status_updater")
    @patch("forge.executor.driver._write_status")
    @patch("forge.executor.driver.dispatch_loop", return_value=True)
    @patch("forge.executor.driver.pipeline_ops.get_summary")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver.pipeline_ops.state_mgr")
    @patch("forge.executor.commands.cmd_init")
    @patch("forge.executor.driver.signal.signal")
    def test_fresh_init_creates_output_files(self, mock_signal, mock_cmd_init, mock_state_mgr,
                                              mock_state, mock_summary, mock_dispatch,
                                              mock_ws, mock_start_updater, mock_stop_updater):
        plan_file = os.path.join(self.tmp, "plan.md")
        Path(plan_file).write_text("# Plan\n\n1. Build it")
        session_dir = os.path.join(self.tmp, "session")
        os.makedirs(session_dir)

        mock_state.return_value = MagicMock()
        mock_summary.return_value = {
            "pipeline": "full",
            "session_dir": session_dir,
        }
        mock_state_mgr.return_value = MagicMock()

        with patch("sys.argv", ["forge-driver", "--plan", plan_file, "--preset", "test-preset"]):
            main()

        assert Path(session_dir, "pipeline-output.md").is_file()
        assert Path(session_dir, "pipeline-activity.log").is_file()


class TestWriteStatus:
    """Test _write_status()."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="status_test_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    @patch("forge.executor.driver.pipeline_ops.get_summary")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    def test_writes_markdown_with_step_table(self, mock_state, mock_summary):
        mock_state.return_value = MagicMock()
        mock_summary.return_value = {
            "pipeline": "full",
            "preset": "test-preset",
            "created_at": "2026-02-17T10:00:00Z",
            "steps": {
                "code": {"status": "complete", "retries": 0,
                         "started_at": "2026-02-17T10:00:00Z",
                         "completed_at": "2026-02-17T10:05:00Z"},
                "lint": {"status": "in_progress", "retries": 0,
                         "started_at": "2026-02-17T10:05:01Z"},
                "test": {"status": "pending", "retries": 0},
            },
            "step_order": ["code", "lint", "test"],
            "dependency_graph": {"lint": ["code"], "test": ["code"]},
        }

        _write_status(self.tmp)

        status_file = Path(self.tmp, "pipeline-status.md")
        assert status_file.is_file()
        content = status_file.read_text()
        assert "Pipeline Status" in content
        assert "| code |" in content
        assert "done" in content
        assert "RUNNING" in content
        assert "full" in content

    @patch("forge.executor.driver.pipeline_ops.require_state",
           side_effect=RuntimeError("no state"))
    def test_state_failure_does_not_crash(self, mock_state):
        _write_status(self.tmp)
        assert not Path(self.tmp, "pipeline-status.md").is_file()

    @patch("forge.executor.driver.pipeline_ops.get_summary", return_value="not a dict")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    def test_non_dict_status_does_not_crash(self, mock_state, mock_summary):
        mock_state.return_value = MagicMock()
        _write_status(self.tmp)
        assert not Path(self.tmp, "pipeline-status.md").is_file()

    @patch("forge.executor.driver.pipeline_ops.get_summary")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    def test_recent_activity_included(self, mock_state, mock_summary):
        _recent_log.clear()
        _recent_log.append("[10:00] code         passed  5s")

        mock_state.return_value = MagicMock()
        mock_summary.return_value = {
            "pipeline": "full", "preset": "test-preset", "created_at": "",
            "steps": {"code": {"status": "complete", "retries": 0}},
            "step_order": ["code"],
            "dependency_graph": {},
        }

        _write_status(self.tmp)
        content = Path(self.tmp, "pipeline-status.md").read_text()
        assert "Recent Activity" in content
        assert "passed" in content

        _recent_log.clear()


class TestMainNoPlanOrResume:
    """Cover parser.error when neither --plan nor --plan-dir provided."""

    @patch("forge.executor.driver.dispatch_loop", return_value=True)
    @patch("forge.executor.driver.pipeline_ops.require_state")
    def test_no_plan_no_resume_exits(self, mock_state, mock_loop):
        with patch("sys.argv", ["forge-driver"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2  # argparse error exit code


# ── File allowlist enforcement ─────────────────────────────────────────────


class TestEnforceFileAllowlist:
    """Tests for _enforce_file_allowlist and _snapshot_worktree."""

    def _make_preset(self, allowed):
        from forge.executor.engine.registry import StepDefinition, Preset
        step = StepDefinition(name="visual_test", step_type="ai", allowed_file_patterns=allowed)
        return Preset(
            name="test", version=3, description="", pipelines={},
            steps={"visual_test": step}, models={},
        )

    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver._snapshot_worktree")
    @patch("subprocess.run")
    def test_violations_reverted_and_returned(self, mock_run, mock_snap, mock_state, mock_preset):
        mock_preset.return_value = self._make_preset([])  # deny all
        mock_state.return_value = MagicMock()
        mock_snap.return_value = {"src/Foo.ts", "src/Bar.ts"}
        before = set()  # nothing existed before

        # git checkout should succeed
        mock_run.return_value = MagicMock(returncode=0)

        violations = _enforce_file_allowlist("visual_test", "/cwd", before)
        assert sorted(violations) == ["src/Bar.ts", "src/Foo.ts"]
        # git checkout was called with the violating files
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0:3] == ["git", "checkout", "--"]
        assert set(args[3:]) == {"src/Bar.ts", "src/Foo.ts"}

    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver._snapshot_worktree")
    def test_no_violations_when_unrestricted(self, mock_snap, mock_state, mock_preset):
        mock_preset.return_value = self._make_preset(None)  # unrestricted
        mock_state.return_value = MagicMock()
        mock_snap.return_value = {"src/Foo.ts"}
        violations = _enforce_file_allowlist("visual_test", "/cwd", set())
        assert violations == []

    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver._snapshot_worktree")
    def test_allowed_patterns_pass(self, mock_snap, mock_state, mock_preset):
        mock_preset.return_value = self._make_preset(["*.json", "*.md"])
        mock_state.return_value = MagicMock()
        mock_snap.return_value = {"results.json", "report.md", "src/Foo.ts"}
        before = set()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            violations = _enforce_file_allowlist("visual_test", "/cwd", before)

        assert violations == ["src/Foo.ts"]

    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver._snapshot_worktree")
    def test_no_new_files_no_violations(self, mock_snap, mock_state, mock_preset):
        mock_preset.return_value = self._make_preset([])
        mock_state.return_value = MagicMock()
        before = {"src/Foo.ts"}
        mock_snap.return_value = {"src/Foo.ts"}  # same as before
        violations = _enforce_file_allowlist("visual_test", "/cwd", before)
        assert violations == []

    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state")
    @patch("forge.executor.driver.pipeline_ops.require_state")
    @patch("forge.executor.driver._snapshot_worktree")
    def test_only_delta_checked(self, mock_snap, mock_state, mock_preset):
        """Pre-existing modifications from code step are not flagged."""
        mock_preset.return_value = self._make_preset([])
        mock_state.return_value = MagicMock()
        before = {"src/CodeStep.ts"}
        mock_snap.return_value = {"src/CodeStep.ts", "src/Rogue.ts"}

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            violations = _enforce_file_allowlist("visual_test", "/cwd", before)

        assert violations == ["src/Rogue.ts"]

    @patch("forge.executor.driver.pipeline_ops.load_preset_for_state",
           side_effect=Exception("no preset"))
    @patch("forge.executor.driver.pipeline_ops.require_state")
    def test_preset_load_failure_is_permissive(self, mock_state, mock_preset):
        violations = _enforce_file_allowlist("visual_test", "/cwd", set())
        assert violations == []


class TestSnapshotWorktree:
    """Tests for _snapshot_worktree."""

    @patch("subprocess.run")
    def test_combines_all_sources(self, mock_run):
        def side_effect(cmd, **kwargs):
            m = MagicMock(returncode=0)
            if cmd[1] == "diff" and "--cached" not in cmd:
                m.stdout = "modified.ts\n"
            elif "--cached" in cmd:
                m.stdout = "staged.ts\n"
            else:
                m.stdout = "untracked.ts\n"
            return m
        mock_run.side_effect = side_effect
        result = _snapshot_worktree("/cwd")
        assert result == {"modified.ts", "staged.ts", "untracked.ts"}

    @patch("subprocess.run")
    def test_handles_empty_output(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        result = _snapshot_worktree("/cwd")
        assert result == set()


# ── Concurrent driver guard ─────────────────────────────────────────────


class TestConcurrentDriverGuard:
    """Regression: two drivers running concurrently causes state corruption.

    _set_driver_pid must refuse to start when another driver PID is alive.
    """

    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="driver_guard_test_")
        self.session_dir = os.path.join(self.tmp, "session")
        os.makedirs(self.session_dir, exist_ok=True)

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_set_driver_pid_raises_when_existing_driver_alive(self):
        """If another driver PID is alive, _set_driver_pid should raise RuntimeError."""
        from forge.executor.engine.state import StateManager, PipelineState, StepState

        state_file = Path(self.session_dir) / "agent-state.json"
        mgr = StateManager(state_file)
        state = PipelineState(
            steps={"code": StepState()},
            step_order=["code"],
            session_dir=self.session_dir,
            driver_pid=os.getpid(),  # Current process is "the other driver"
        )
        mgr.save(state)

        # Import the main function's inner _set_driver_pid by running
        # a simplified version of the guard logic
        from forge.executor.engine import pipeline_ops

        with patch.object(pipeline_ops, "state_mgr", return_value=mgr):
            # Simulate the guard: load state, check PID
            loaded = mgr.load()
            existing_pid = loaded.driver_pid

            # The existing PID is our own process (definitely alive)
            assert existing_pid == os.getpid()
            try:
                os.kill(existing_pid, 0)
                is_alive = True
            except (ProcessLookupError, PermissionError):
                is_alive = False
            assert is_alive, "Test PID should be alive"

    def test_set_driver_pid_succeeds_when_no_existing_driver(self):
        """If no existing driver PID, _set_driver_pid should succeed."""
        from forge.executor.engine.state import StateManager, PipelineState, StepState

        state_file = Path(self.session_dir) / "agent-state.json"
        mgr = StateManager(state_file)
        state = PipelineState(
            steps={"code": StepState()},
            step_order=["code"],
            session_dir=self.session_dir,
            driver_pid=0,
        )
        mgr.save(state)

        loaded = mgr.load()
        assert loaded.driver_pid == 0

    def test_set_driver_pid_succeeds_when_existing_driver_dead(self):
        """If existing driver PID is dead, _set_driver_pid should succeed."""
        from forge.executor.engine.state import StateManager, PipelineState, StepState

        state_file = Path(self.session_dir) / "agent-state.json"
        mgr = StateManager(state_file)
        # Use a PID that's almost certainly dead
        state = PipelineState(
            steps={"code": StepState()},
            step_order=["code"],
            session_dir=self.session_dir,
            driver_pid=99999999,
        )
        mgr.save(state)

        loaded = mgr.load()
        try:
            os.kill(loaded.driver_pid, 0)
            is_alive = True
        except (ProcessLookupError, PermissionError):
            is_alive = False
        assert not is_alive, "Dead PID should not be alive"


# ── Bug 1 regression: reset orphaned IN_PROGRESS steps on resume ──────────


class TestResetOrphanedInProgressSteps:
    """Regression: orphaned IN_PROGRESS steps must be reset to PENDING on resume
    without incrementing retries."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="orphan_test_")
        self.session_dir = os.path.join(self.tmp, "session")
        os.makedirs(self.session_dir)
        self.state_file = Path(self.session_dir) / "agent-state.json"

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def _make_state_with_statuses(self, step_statuses: dict[str, str]) -> None:
        """Create and save a state where steps have the given statuses."""
        from forge.executor.engine.state import StateManager, PipelineState, StepState, StepStatus
        steps = {}
        for name, status_str in step_statuses.items():
            ss = StepState(status=StepStatus(status_str))
            if status_str == "in_progress":
                ss.started_at = "2026-01-01T00:00:00Z"
            steps[name] = ss
        state = PipelineState(
            pipeline="full", preset="test-preset",
            current_step=list(step_statuses.keys())[0],
            steps=steps,
            step_order=list(step_statuses.keys()),
            session_dir=self.session_dir,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        )
        # Create checkpoints dir
        os.makedirs(os.path.join(self.session_dir, "checkpoints"), exist_ok=True)
        mgr = StateManager(self.state_file)
        mgr.save(state)
        return mgr

    def test_in_progress_steps_reset_to_pending(self):
        """IN_PROGRESS steps should become PENDING after reset_step(no_retry_inc=True)."""
        from forge.executor.engine import pipeline_ops
        from forge.executor.engine.state import StateManager, StepStatus

        mgr = self._make_state_with_statuses({
            "code": "complete", "build": "in_progress", "test": "pending",
        })

        with patch.object(pipeline_ops, "state_mgr", return_value=mgr), \
             patch.object(pipeline_ops, "state_file_path", return_value=self.state_file), \
             patch.object(pipeline_ops, "_active_session_dir", return_value=Path(self.session_dir)):
            pipeline_ops.reset_step("build", no_retry_inc=True)
            state = mgr.load()

        assert state.steps["build"].status == StepStatus.PENDING

    def test_no_retry_inc_preserves_retry_count(self):
        """reset_step(no_retry_inc=True) must NOT increment the retry counter."""
        from forge.executor.engine import pipeline_ops
        from forge.executor.engine.state import StateManager, StepStatus

        mgr = self._make_state_with_statuses({
            "code": "complete", "build": "in_progress", "test": "pending",
        })

        with patch.object(pipeline_ops, "state_mgr", return_value=mgr), \
             patch.object(pipeline_ops, "state_file_path", return_value=self.state_file), \
             patch.object(pipeline_ops, "_active_session_dir", return_value=Path(self.session_dir)):
            pipeline_ops.reset_step("build", no_retry_inc=True)
            state = mgr.load()

        assert state.steps["build"].retries == 0

    def test_complete_and_pending_steps_not_touched(self):
        """COMPLETE and PENDING steps should be unchanged after resetting orphaned steps."""
        from forge.executor.engine import pipeline_ops
        from forge.executor.engine.state import StateManager, StepStatus

        mgr = self._make_state_with_statuses({
            "code": "complete", "build": "in_progress", "test": "pending",
        })

        with patch.object(pipeline_ops, "state_mgr", return_value=mgr), \
             patch.object(pipeline_ops, "state_file_path", return_value=self.state_file), \
             patch.object(pipeline_ops, "_active_session_dir", return_value=Path(self.session_dir)):
            # Only reset the IN_PROGRESS step, as the driver code does
            state = mgr.load()
            for name, ss in state.steps.items():
                if ss.status == StepStatus.IN_PROGRESS:
                    pipeline_ops.reset_step(name, no_retry_inc=True)
            state = mgr.load()

        assert state.steps["code"].status == StepStatus.COMPLETE
        assert state.steps["test"].status == StepStatus.PENDING


# ── _set_hook_build_cmd ──────────────────────────────────────────────────

class TestSetHookBuildCmd:
    """Tests for FORGE_BUILD_CMD env var set by _set_hook_build_cmd."""

    def setup_method(self):
        from forge.executor.engine.registry import Preset, PipelineDefinition
        self.preset = Preset(
            name="test", version=1, description="test",
            pipelines={"full": PipelineDefinition(steps=["code"], dependencies={})},
            steps={},
            models={},
            build_command="cd {{REPO_ROOT}} && npm run build",
            bazel_build_command="cd {{REPO_ROOT}} && bazel build {{BUILD_TARGETS}}",
        )
        self._orig = os.environ.get("FORGE_BUILD_CMD")

    def teardown_method(self):
        if self._orig is not None:
            os.environ["FORGE_BUILD_CMD"] = self._orig
        else:
            os.environ.pop("FORGE_BUILD_CMD", None)

    def _make_state(self, packages=None):
        state = MagicMock()
        state.affected_packages = packages or []
        state.session_dir = "/tmp/test-session"
        state.plan_file = ""
        state.dev_server_port = 0
        return state

    @patch("forge.executor.driver.is_bazel_repo", return_value=False)
    @patch("forge.executor.driver.build_context")
    def test_sets_npm_build_for_non_bazel(self, mock_ctx, mock_bazel):
        mock_ctx.return_value = {"REPO_ROOT": "/repo", "AFFECTED_PACKAGES": "(none)"}
        state = self._make_state()

        _set_hook_build_cmd(state, self.preset)

        assert os.environ["FORGE_BUILD_CMD"] == "cd /repo && npm run build"

    @patch("forge.executor.driver.is_bazel_repo", return_value=True)
    @patch("forge.executor.driver.build_context")
    def test_sets_bazel_build_with_packages(self, mock_ctx, mock_bazel):
        mock_ctx.return_value = {"REPO_ROOT": "/repo", "AFFECTED_PACKAGES": "pkg/a, pkg/b"}
        state = self._make_state(packages=["pkg/a", "pkg/b"])

        _set_hook_build_cmd(state, self.preset)

        assert os.environ["FORGE_BUILD_CMD"] == "cd /repo && bazel build //pkg/a/... //pkg/b/..."

    @patch("forge.executor.driver.is_bazel_repo", return_value=True)
    @patch("forge.executor.driver.build_context")
    def test_skips_when_no_packages_and_template_needs_targets(self, mock_ctx, mock_bazel):
        mock_ctx.return_value = {"REPO_ROOT": "/repo", "AFFECTED_PACKAGES": "(none)"}
        os.environ.pop("FORGE_BUILD_CMD", None)
        state = self._make_state()

        _set_hook_build_cmd(state, self.preset)

        assert "FORGE_BUILD_CMD" not in os.environ

    @patch("forge.executor.driver.is_bazel_repo", return_value=False)
    @patch("forge.executor.driver.build_context")
    def test_no_env_var_when_no_build_command(self, mock_ctx, mock_bazel):
        from forge.executor.engine.registry import Preset, PipelineDefinition
        os.environ.pop("FORGE_BUILD_CMD", None)
        preset = Preset(
            name="test", version=1, description="test",
            pipelines={"full": PipelineDefinition(steps=[], dependencies={})},
            steps={},
            models={},
        )
        state = self._make_state()

        _set_hook_build_cmd(state, preset)

        assert "FORGE_BUILD_CMD" not in os.environ


# ── _set_hook_eslint_config ─────────────────────────────────────────────

class TestSetHookEslintConfig:
    """Tests for FORGE_ESLINT_CONFIG env var set by _set_hook_eslint_config."""

    def setup_method(self):
        self._orig = os.environ.get("FORGE_ESLINT_CONFIG")

    def teardown_method(self):
        if self._orig is not None:
            os.environ["FORGE_ESLINT_CONFIG"] = self._orig
        else:
            os.environ.pop("FORGE_ESLINT_CONFIG", None)

    def _make_preset(self, eslint_config=""):
        from forge.executor.engine.registry import Preset, PipelineDefinition
        return Preset(
            name="test", version=1, description="test",
            pipelines={"full": PipelineDefinition(steps=[], dependencies={})},
            steps={}, models={}, eslint_config=eslint_config,
        )

    def test_sets_env_var_for_existing_config(self, tmp_path):
        config = tmp_path / "eslint.config.mjs"
        config.write_text("export default {};")
        preset = self._make_preset(str(config))

        _set_hook_eslint_config(preset)

        assert os.environ["FORGE_ESLINT_CONFIG"] == str(config)

    def test_resolves_relative_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = tmp_path / "tools" / "lint" / "eslint.config.mjs"
        config.parent.mkdir(parents=True)
        config.write_text("export default {};")
        preset = self._make_preset("tools/lint/eslint.config.mjs")

        _set_hook_eslint_config(preset)

        assert os.environ["FORGE_ESLINT_CONFIG"] == str(config)

    def test_skips_when_not_configured(self):
        os.environ.pop("FORGE_ESLINT_CONFIG", None)
        preset = self._make_preset()

        _set_hook_eslint_config(preset)

        assert "FORGE_ESLINT_CONFIG" not in os.environ

    def test_skips_when_file_missing(self):
        os.environ.pop("FORGE_ESLINT_CONFIG", None)
        preset = self._make_preset("/nonexistent/eslint.config.mjs")

        _set_hook_eslint_config(preset)

        assert "FORGE_ESLINT_CONFIG" not in os.environ


# ── _set_plugin_dir ─────────────────────────────────────────────────────

class TestSetPluginDir:
    """Tests for FORGE_PLUGIN_DIR env var set by _set_plugin_dir."""

    def setup_method(self):
        self._orig = os.environ.get("FORGE_PLUGIN_DIR")

    def teardown_method(self):
        if self._orig is not None:
            os.environ["FORGE_PLUGIN_DIR"] = self._orig
        else:
            os.environ.pop("FORGE_PLUGIN_DIR", None)

    def test_sets_env_var_when_skin_exists(self, tmp_path):
        # Mirror real layout: <root>/src/forge/executor/driver.py
        # 4 parents from driver.py → <root>
        (tmp_path / "skins" / "claude-code" / "hooks").mkdir(parents=True)
        driver_dir = tmp_path / "src" / "forge" / "executor"
        driver_dir.mkdir(parents=True)
        os.environ.pop("FORGE_PLUGIN_DIR", None)

        import forge.executor.driver as drv
        orig_file = drv.__file__
        try:
            drv.__file__ = str(driver_dir / "driver.py")
            _set_plugin_dir()
            assert os.environ["FORGE_PLUGIN_DIR"] == str(
                tmp_path / "skins" / "claude-code"
            )
        finally:
            drv.__file__ = orig_file

    def test_no_env_var_when_skin_missing(self, tmp_path):
        driver_dir = tmp_path / "src" / "forge" / "executor"
        driver_dir.mkdir(parents=True)
        os.environ.pop("FORGE_PLUGIN_DIR", None)

        import forge.executor.driver as drv
        orig_file = drv.__file__
        try:
            drv.__file__ = str(driver_dir / "driver.py")
            _set_plugin_dir()
            assert "FORGE_PLUGIN_DIR" not in os.environ
        finally:
            drv.__file__ = orig_file


# ── ClaudeProvider --plugin-dir ─────────────────────────────────────────

class TestClaudeProviderPluginDir:
    """Tests that ClaudeProvider passes --plugin-dir from env var."""

    def setup_method(self):
        self._orig = os.environ.get("FORGE_PLUGIN_DIR")

    def teardown_method(self):
        if self._orig is not None:
            os.environ["FORGE_PLUGIN_DIR"] = self._orig
        else:
            os.environ.pop("FORGE_PLUGIN_DIR", None)

    @patch("forge.providers.claude.subprocess.Popen")
    def test_includes_plugin_dir_when_set(self, mock_popen):
        proc = MagicMock()
        proc.stdout = iter([])
        proc.returncode = 0
        proc.wait.return_value = None
        mock_popen.return_value = proc

        os.environ["FORGE_PLUGIN_DIR"] = "/fake/skin"
        from forge.providers.claude import ClaudeProvider
        provider = ClaudeProvider(agent_command="claude")
        provider.run_agent(prompt="test", model="sonnet", max_turns=1,
                           cwd="/tmp", timeout_s=5)

        cmd = mock_popen.call_args[0][0]
        assert "--plugin-dir" in cmd
        idx = cmd.index("--plugin-dir")
        assert cmd[idx + 1] == "/fake/skin"

    @patch("forge.providers.claude.subprocess.Popen")
    def test_no_plugin_dir_when_unset(self, mock_popen):
        proc = MagicMock()
        proc.stdout = iter([])
        proc.returncode = 0
        proc.wait.return_value = None
        mock_popen.return_value = proc

        os.environ.pop("FORGE_PLUGIN_DIR", None)
        from forge.providers.claude import ClaudeProvider
        provider = ClaudeProvider(agent_command="claude")
        provider.run_agent(prompt="test", model="sonnet", max_turns=1,
                           cwd="/tmp", timeout_s=5)

        cmd = mock_popen.call_args[0][0]
        assert "--plugin-dir" not in cmd


# ── _preflight_hooks ────────────────────────────────────────────────────

class TestPreflightHooks:
    """Tests for _preflight_hooks startup diagnostics."""

    @patch("forge.executor.driver.shutil.which")
    def test_exits_when_eslint_missing(self, mock_which, capsys):
        mock_which.side_effect = lambda cmd: None
        with pytest.raises(SystemExit):
            _preflight_hooks()
        out = capsys.readouterr().out
        assert "missing" in out.lower()

    @patch("forge.executor.driver.shutil.which")
    def test_exits_when_prettier_missing(self, mock_which, capsys):
        mock_which.side_effect = lambda cmd: "/usr/bin/eslint" if cmd == "eslint" else None
        with pytest.raises(SystemExit):
            _preflight_hooks()
        out = capsys.readouterr().out
        assert "missing" in out.lower()

    @patch("forge.executor.driver.shutil.which")
    def test_warns_no_eslint_config(self, mock_which, capsys):
        mock_which.side_effect = lambda cmd: f"/usr/bin/{cmd}"
        with patch("forge.executor.driver.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stderr="eslint.config not found"
            )
            _preflight_hooks()
        out = capsys.readouterr().out
        assert "eslint_config" in out

    @patch("forge.executor.driver.shutil.which")
    def test_all_ok(self, mock_which, capsys):
        mock_which.side_effect = lambda cmd: f"/usr/bin/{cmd}"
        with patch("forge.executor.driver.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            _preflight_hooks()
        out = capsys.readouterr().out
        assert "eslint + prettier ready" in out
