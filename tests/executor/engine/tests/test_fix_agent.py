
"""Tests for fix_agent.py — generic fix wrapper for command step failures."""

import json
import os
import tempfile
import shutil
import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from architect.executor.engine.agents.fix_agent import (
    generate_prompt,
    run,
    FixAgentOutcome,
    _re_execute_command,
    _run_command,
    _needs_shell,
)
from architect.core.runner import AgentRunner, AgentResult
from architect.executor.engine.registry import StepDefinition, Preset, PipelineDefinition
from architect.executor.engine.state import PipelineState, StepState


def _make_preset(preset_dir="."):
    return Preset(
        name="test",
        version=3,
        description="Test preset",
        pipelines={"full": PipelineDefinition(steps=["test"], dependencies={})},
        steps={"test": StepDefinition(name="test", step_type="command")},
        models={"fix": "sonnet"},
        preset_dir=Path(preset_dir),
    )


def _make_state(session_dir="", packages=None):
    return PipelineState(
        steps={"test": StepState()},
        step_order=["test"],
        dependency_graph={},
        session_dir=session_dir,
        pipeline="full",
        preset="test",
        affected_packages=packages or [],
    )


def _make_step(**kwargs):
    defaults = dict(
        name="test", step_type="command",
        run_command="bazel test //apps/web/...",
        error_file="test-errors.txt",
        fix_hints="Check type imports",
        timeout=600,
    )
    defaults.update(kwargs)
    return StepDefinition(**defaults)


# -- Prompt generation --------------------------------------------------------

class TestGeneratePrompt:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="fix_agent_prompt_test_")
        self.session_dir = os.path.join(self.tmp, "session")
        os.makedirs(self.session_dir)

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_no_pipeline_protocol(self):
        step = _make_step()
        state = _make_state(session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        prompt = generate_prompt(step, state, preset)

        assert "Pipeline Protocol" not in prompt
        assert "pipeline_cli.py pass" not in prompt
        assert "pipeline_cli.py fail" not in prompt

    def test_verify_command_in_prompt(self):
        step = _make_step(run_command="bazel test //apps/web/...")
        state = _make_state(session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        prompt = generate_prompt(step, state, preset)

        assert "bazel test //apps/web/..." in prompt

    def test_fix_hints_in_prompt(self):
        step = _make_step(fix_hints="Run generate-bazel-build.sh first")
        state = _make_state(session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        prompt = generate_prompt(step, state, preset)

        assert "Run generate-bazel-build.sh first" in prompt

    def test_error_file_reference(self):
        error_path = Path(self.session_dir) / "test-errors.txt"
        error_path.write_text("TypeError: x is not a function\n" * 10)

        step = _make_step(error_file="test-errors.txt")
        state = _make_state(session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        prompt = generate_prompt(step, state, preset)

        assert "Error log" in prompt
        assert str(error_path) in prompt

    def test_missing_error_file_no_reference(self):
        step = _make_step(error_file="test-errors.txt")
        state = _make_state(session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        prompt = generate_prompt(step, state, preset)

        assert "Error log" not in prompt

    def test_per_package_prompt(self):
        error1 = Path(self.session_dir) / "test-errors-apps-web.txt"
        error1.write_text("FAIL: test1\n")

        step = _make_step(
            per_package=True,
            run_command="cd {{REPO_ROOT}}/{{PACKAGE}} && bazel test ...",
            error_file="test-errors-{{PACKAGE_SLUG}}.txt",
        )
        state = _make_state(session_dir=self.session_dir, packages=["apps/web", "libs/common"])
        preset = _make_preset(self.tmp)

        prompt = generate_prompt(step, state, preset, failed_packages=["apps/web", "libs/common"])

        assert "### Package: `apps/web`" in prompt
        assert "### Package: `libs/common`" in prompt
        assert "Error log" in prompt  # apps/web error file exists

    def test_no_hints_when_empty(self):
        step = _make_step(fix_hints="")
        state = _make_state(session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        prompt = generate_prompt(step, state, preset)

        assert "Hint" not in prompt


# -- _needs_shell -------------------------------------------------------------

class TestBazelCommandSelection:
    """Regression: fix_agent must use bazel_run_command in Bazel repos."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="fix_agent_bazel_test_")
        self.session_dir = os.path.join(self.tmp, "session")
        os.makedirs(self.session_dir)

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    @patch("architect.executor.engine.runner.is_bazel_repo", return_value=True)
    def test_prompt_uses_bazel_command(self, mock_bazel):
        step = _make_step(
            run_command="cd {{REPO_ROOT}}/{{PACKAGE}} && npm run lint:fix",
            bazel_run_command="cd {{REPO_ROOT}} && bazel run //{{PACKAGE}}:lint.fix",
            per_package=True,
        )
        state = _make_state(session_dir=self.session_dir, packages=["apps/web"])
        preset = _make_preset(self.tmp)

        prompt = generate_prompt(step, state, preset, failed_packages=["apps/web"])

        assert "bazel run" in prompt
        assert "npm run lint:fix" not in prompt

    @patch("architect.executor.engine.runner.is_bazel_repo", return_value=False)
    def test_prompt_uses_npm_command_in_non_bazel(self, mock_bazel):
        step = _make_step(
            run_command="cd {{REPO_ROOT}}/{{PACKAGE}} && npm run lint:fix",
            bazel_run_command="cd {{REPO_ROOT}} && bazel run //{{PACKAGE}}:lint.fix",
            per_package=True,
        )
        state = _make_state(session_dir=self.session_dir, packages=["apps/web"])
        preset = _make_preset(self.tmp)

        prompt = generate_prompt(step, state, preset, failed_packages=["apps/web"])

        assert "npm run lint:fix" in prompt
        assert "bazel run" not in prompt

    @patch("architect.executor.engine.runner.is_bazel_repo", return_value=True)
    @patch("architect.executor.engine.agents.fix_agent._run_command", return_value=(True, "OK"))
    def test_re_execute_uses_bazel_command(self, mock_run, mock_bazel):
        step = _make_step(
            run_command="cd {{REPO_ROOT}}/{{PACKAGE}} && npm run lint:fix",
            bazel_run_command="cd {{REPO_ROOT}} && bazel run //{{PACKAGE}}:lint.fix",
            per_package=True,
        )
        state = _make_state(session_dir=self.session_dir, packages=["apps/web"])
        preset = _make_preset(self.tmp)

        _re_execute_command(step, state, preset, self.tmp, failed_packages=["apps/web"])

        cmd_arg = mock_run.call_args[0][0]
        assert "bazel run" in cmd_arg
        assert "npm run" not in cmd_arg


class TestNeedsShell:
    def test_pipe_needs_shell(self):
        assert _needs_shell("cat foo | grep bar") is True

    def test_cd_needs_shell(self):
        assert _needs_shell("cd /tmp && ls") is True

    def test_redirect_needs_shell(self):
        assert _needs_shell("echo hello > file.txt") is True

    def test_simple_command_no_shell(self):
        assert _needs_shell("bazel test //apps/web/...") is False


# -- _run_command -------------------------------------------------------------

class TestRunCommand:
    def test_successful_command(self):
        ok, output = _run_command("echo hello", "/tmp", timeout=10)
        assert ok is True
        assert "hello" in output

    def test_failed_command(self):
        ok, output = _run_command("false", "/tmp", timeout=10)
        assert ok is False

    def test_timeout(self):
        ok, output = _run_command("sleep 60", "/tmp", timeout=1)
        assert ok is False
        assert "timed out" in output.lower()


# -- _re_execute_command ------------------------------------------------------

class TestReExecuteCommand:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="re_exec_test_")
        self.session_dir = os.path.join(self.tmp, "session")
        os.makedirs(self.session_dir)

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    @patch("architect.executor.engine.agents.fix_agent._run_command", return_value=(True, "OK"))
    def test_single_command_passes(self, mock_run):
        step = _make_step()
        state = _make_state(session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        ok, output = _re_execute_command(step, state, preset, self.tmp)

        assert ok is True
        mock_run.assert_called_once()

    @patch("architect.executor.engine.agents.fix_agent._run_command", return_value=(False, "FAIL"))
    def test_single_command_fails_writes_error_file(self, mock_run):
        step = _make_step(error_file="test-errors.txt")
        state = _make_state(session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        ok, output = _re_execute_command(step, state, preset, self.tmp)

        assert ok is False
        error_path = Path(self.session_dir) / "test-errors.txt"
        assert error_path.is_file()
        assert "FAIL" in error_path.read_text()

    @patch("architect.executor.engine.agents.fix_agent._run_command")
    def test_per_package_all_pass(self, mock_run):
        mock_run.side_effect = [(True, "OK 1"), (True, "OK 2")]
        step = _make_step(
            per_package=True,
            run_command="cd {{REPO_ROOT}}/{{PACKAGE}} && bazel test ...",
        )
        state = _make_state(session_dir=self.session_dir, packages=["apps/web", "libs/common"])
        preset = _make_preset(self.tmp)

        ok, output = _re_execute_command(step, state, preset, self.tmp,
                                          failed_packages=["apps/web", "libs/common"])

        assert ok is True
        assert mock_run.call_count == 2

    @patch("architect.executor.engine.agents.fix_agent._run_command")
    def test_per_package_one_fails(self, mock_run):
        mock_run.side_effect = [(True, "OK"), (False, "FAIL")]
        step = _make_step(
            per_package=True,
            run_command="cd {{REPO_ROOT}}/{{PACKAGE}} && bazel test ...",
            error_file="test-errors-{{PACKAGE_SLUG}}.txt",
        )
        state = _make_state(session_dir=self.session_dir, packages=["apps/web", "libs/common"])
        preset = _make_preset(self.tmp)

        ok, output = _re_execute_command(step, state, preset, self.tmp,
                                          failed_packages=["apps/web", "libs/common"])

        assert ok is False
        # Error file written for the failed package
        error_path = Path(self.session_dir) / "test-errors-libs-common.txt"
        assert error_path.is_file()


# -- Full run() ---------------------------------------------------------------

class TestFixAgentRun:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="fix_agent_run_test_")
        self.session_dir = os.path.join(self.tmp, "session")
        os.makedirs(self.session_dir)
        self.activity_log = os.path.join(self.session_dir, "pipeline-activity.log")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    @patch("architect.executor.engine.agents.fix_agent._report_pass")
    @patch("architect.executor.engine.agents.fix_agent._re_execute_command", return_value=(True, "OK"))
    @patch.object(AgentRunner, "run", return_value=AgentResult(
        exit_code=0, stdout="fixed", transcript_path="/t.log", timed_out=False,
    ))
    def test_pass_when_verify_succeeds(self, mock_agent, mock_verify, mock_pass):
        step = _make_step()
        state = _make_state(session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        outcome = run(step, state, preset, self.tmp, self.session_dir,
                       self.activity_log, agent_command="/bin/true")

        assert outcome.passed is True
        assert outcome.reason == "Fix verified"
        mock_pass.assert_called_once_with("test")

    @patch("architect.executor.engine.agents.fix_agent._report_fail")
    @patch("architect.executor.engine.agents.fix_agent._re_execute_command", return_value=(False, "FAIL"))
    @patch.object(AgentRunner, "run", return_value=AgentResult(
        exit_code=0, stdout="tried", transcript_path="/t.log", timed_out=False,
    ))
    def test_fail_when_verify_still_fails(self, mock_agent, mock_verify, mock_fail):
        step = _make_step()
        state = _make_state(session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        outcome = run(step, state, preset, self.tmp, self.session_dir,
                       self.activity_log, agent_command="/bin/true")

        assert outcome.passed is False
        assert "still fails" in outcome.reason
        mock_fail.assert_called_once()

    @patch("architect.executor.engine.agents.fix_agent._report_pass")
    @patch("architect.executor.engine.agents.fix_agent._re_execute_command", return_value=(True, "OK"))
    @patch.object(AgentRunner, "run", return_value=AgentResult(
        exit_code=1, stdout="crashed", transcript_path="/t.log", timed_out=False,
    ))
    def test_pass_despite_agent_crash_if_verify_succeeds(self, mock_agent, mock_verify, mock_pass):
        """Agent exit code doesn't matter — only the verify command result counts."""
        step = _make_step()
        state = _make_state(session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        outcome = run(step, state, preset, self.tmp, self.session_dir,
                       self.activity_log, agent_command="/bin/false")

        assert outcome.passed is True

    @patch("architect.executor.engine.agents.fix_agent._report_pass")
    @patch("architect.executor.engine.agents.fix_agent._re_execute_command", return_value=(True, "OK"))
    @patch.object(AgentRunner, "run", return_value=AgentResult(
        exit_code=0, stdout="timeout", transcript_path="/t.log", timed_out=True,
    ))
    def test_pass_despite_timeout_if_verify_succeeds(self, mock_agent, mock_verify, mock_pass):
        step = _make_step()
        state = _make_state(session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        outcome = run(step, state, preset, self.tmp, self.session_dir,
                       self.activity_log, agent_command="/bin/true")

        assert outcome.passed is True

    @patch("architect.executor.engine.agents.fix_agent._report_pass")
    @patch("architect.executor.engine.agents.fix_agent._re_execute_command", return_value=(True, "OK"))
    @patch.object(AgentRunner, "run", return_value=AgentResult(
        exit_code=0, stdout="done", transcript_path="/t.log", timed_out=False,
    ))
    def test_agent_receives_correct_params(self, mock_agent, mock_verify, mock_pass):
        step = _make_step()
        state = _make_state(session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        run(step, state, preset, self.tmp, self.session_dir,
            self.activity_log, model="opus", max_turns=40, timeout_s=1800,
            agent_command="my-stub")

        call_kwargs = mock_agent.call_args
        assert call_kwargs.kwargs["model"] == "opus"
        assert call_kwargs.kwargs["max_turns"] == 40
        assert call_kwargs.kwargs["timeout_s"] == 1800

    @patch("architect.executor.engine.agents.fix_agent._report_pass")
    @patch("architect.executor.engine.agents.fix_agent._re_execute_command", return_value=(True, "OK"))
    @patch.object(AgentRunner, "run", return_value=AgentResult(
        exit_code=0, stdout="done", transcript_path="/t.log", timed_out=False,
    ))
    def test_per_package_failed_packages_forwarded(self, mock_agent, mock_verify, mock_pass):
        step = _make_step(
            per_package=True,
            run_command="cd {{REPO_ROOT}}/{{PACKAGE}} && bazel test ...",
        )
        state = _make_state(session_dir=self.session_dir, packages=["apps/web"])
        preset = _make_preset(self.tmp)

        run(step, state, preset, self.tmp, self.session_dir,
            self.activity_log, failed_packages=["apps/web"], agent_command="/bin/true")

        # Verify command re-executed with failed_packages
        mock_verify.assert_called_once()
        call_args = mock_verify.call_args
        assert call_args[1].get("failed_packages") or call_args[0][4] == ["apps/web"]
