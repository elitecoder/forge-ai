
"""Tests for code_agent.py — code step wrapper with deterministic verification."""

import json
import os
import tempfile
import shutil
import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from architect.executor.engine.agents.code_agent import (
    generate_prompt,
    run,
    CodeAgentOutcome,
    _has_code_changes,
    _generate_checklist,
)
from architect.core.runner import AgentRunner, AgentResult
from architect.executor.engine.registry import StepDefinition, Preset, PipelineDefinition
from architect.executor.engine.state import PipelineState, StepState


def _make_preset(preset_dir=".", pass_command="", bazel_pass_command=""):
    return Preset(
        name="test",
        version=3,
        description="Test preset",
        pipelines={"full": PipelineDefinition(steps=["code"], dependencies={})},
        steps={"code": StepDefinition(
            name="code", step_type="ai",
            pass_command=pass_command,
            bazel_pass_command=bazel_pass_command,
        )},
        models={"code": "sonnet", "fix": "sonnet"},
        preset_dir=Path(preset_dir),
    )


def _make_state(plan_file="", session_dir="", packages=None):
    return PipelineState(
        steps={"code": StepState()},
        step_order=["code"],
        dependency_graph={},
        session_dir=session_dir,
        pipeline="full",
        preset="test",
        plan_file=plan_file,
        affected_packages=packages or [],
    )


# -- Prompt generation --------------------------------------------------------

class TestGeneratePrompt:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="code_agent_prompt_test_")
        self.session_dir = os.path.join(self.tmp, "session")
        os.makedirs(self.session_dir)

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_plan_content_included(self):
        plan = os.path.join(self.tmp, "plan.md")
        Path(plan).write_text("# My Plan\n\nDo the thing.")
        state = _make_state(plan_file=plan, session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        prompt = generate_prompt(state, preset)

        assert "# My Plan" in prompt
        assert "Do the thing." in prompt

    def test_no_pipeline_protocol(self):
        plan = os.path.join(self.tmp, "plan.md")
        Path(plan).write_text("Plan content")
        state = _make_state(plan_file=plan, session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        prompt = generate_prompt(state, preset)

        assert "Pipeline Protocol" not in prompt
        assert "pipeline_cli.py pass" not in prompt
        assert "pipeline_cli.py fail" not in prompt

    def test_missing_plan_raises(self):
        state = _make_state(plan_file="/nonexistent/plan.md", session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        with pytest.raises(FileNotFoundError, match="Plan file not found"):
            generate_prompt(state, preset)

    def test_codebase_brief_included(self):
        plan_dir = os.path.join(self.tmp, "artifacts")
        os.makedirs(plan_dir)
        Path(os.path.join(plan_dir, "plan.md")).write_text("Plan")
        Path(os.path.join(plan_dir, "codebase-brief.md")).write_text("Brief: uses TypeScript")

        state = _make_state(
            plan_file=os.path.join(plan_dir, "plan.md"),
            session_dir=self.session_dir,
        )
        preset = _make_preset(self.tmp)

        prompt = generate_prompt(state, preset)

        assert "## Codebase Brief" in prompt
        assert "Brief: uses TypeScript" in prompt

    def test_affected_packages_in_prompt(self):
        plan = os.path.join(self.tmp, "plan.md")
        Path(plan).write_text("Plan")
        state = _make_state(
            plan_file=plan,
            session_dir=self.session_dir,
            packages=["apps/webapp/web"],
        )
        preset = _make_preset(self.tmp)

        prompt = generate_prompt(state, preset)

        assert "apps/webapp/web" in prompt

    @patch("architect.executor.engine.agents.code_agent.is_bazel_repo", return_value=True)
    def test_build_command_uses_packages(self, _mock_bazel):
        plan = os.path.join(self.tmp, "plan.md")
        Path(plan).write_text("Plan")
        state = _make_state(
            plan_file=plan,
            session_dir=self.session_dir,
            packages=["apps/webapp/web", "libs/common"],
        )
        preset = _make_preset(
            self.tmp,
            bazel_pass_command="cd {{REPO_ROOT}} && bazel build {{BUILD_TARGETS}}",
        )

        prompt = generate_prompt(state, preset)

        assert "//apps/webapp/web/..." in prompt
        assert "//libs/common/..." in prompt

    @patch("architect.executor.engine.agents.code_agent.is_bazel_repo", return_value=True)
    def test_build_command_fallback_when_no_packages(self, _mock_bazel):
        plan = os.path.join(self.tmp, "plan.md")
        Path(plan).write_text("Plan")
        state = _make_state(plan_file=plan, session_dir=self.session_dir, packages=[])
        preset = _make_preset(
            self.tmp,
            bazel_pass_command="cd {{REPO_ROOT}} && bazel build {{BUILD_TARGETS}}",
        )

        prompt = generate_prompt(state, preset)

        assert "bazel build :tsc" in prompt

    def test_no_build_command_when_no_pass_command(self):
        plan = os.path.join(self.tmp, "plan.md")
        Path(plan).write_text("Plan")
        state = _make_state(plan_file=plan, session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        prompt = generate_prompt(state, preset)

        assert "No build verification configured" in prompt

    def test_visual_test_plan_copied_to_session(self):
        plan_dir = os.path.join(self.tmp, "artifacts")
        os.makedirs(plan_dir)
        Path(os.path.join(plan_dir, "plan.md")).write_text("Plan")
        Path(os.path.join(plan_dir, "visual-test-plan.md")).write_text("VTP content")

        state = _make_state(
            plan_file=os.path.join(plan_dir, "plan.md"),
            session_dir=self.session_dir,
        )
        preset = _make_preset(self.tmp)

        generate_prompt(state, preset)

        assert (Path(self.session_dir) / "visual-test-plan.md").read_text() == "VTP content"


# -- _has_code_changes --------------------------------------------------------

class TestHasCodeChanges:
    @patch("architect.executor.engine.agents.code_agent.subprocess.run")
    def test_tracked_changes_detected(self, mock_run):
        mock_run.side_effect = [
            MagicMock(stdout=" src/index.ts | 5 +++--\n 1 file changed\n"),
        ]
        assert _has_code_changes("/repo") is True

    @patch("architect.executor.engine.agents.code_agent.subprocess.run")
    def test_untracked_changes_detected(self, mock_run):
        mock_run.side_effect = [
            MagicMock(stdout=""),       # git diff --stat HEAD (nothing)
            MagicMock(stdout="new-file.ts\n"),  # git ls-files --others
        ]
        assert _has_code_changes("/repo") is True

    @patch("architect.executor.engine.agents.code_agent.subprocess.run")
    def test_no_changes(self, mock_run):
        mock_run.side_effect = [
            MagicMock(stdout=""),  # git diff --stat HEAD
            MagicMock(stdout=""),  # git ls-files --others
        ]
        assert _has_code_changes("/repo") is False


# -- _generate_checklist ------------------------------------------------------

class TestGenerateChecklist:
    @patch("architect.executor.engine.agents.code_agent.subprocess.run")
    def test_checklist_from_diff(self, mock_run):
        mock_run.side_effect = [
            MagicMock(stdout=" src/a.ts | 3 +++\n"),   # diff --stat
            MagicMock(stdout="src/a.ts\n"),             # diff --name-only
            MagicMock(stdout="src/b.ts\n"),             # ls-files --others
        ]
        cl = _generate_checklist("/repo")

        assert cl["step"] == "code"
        assert "src/a.ts" in cl["files_modified"]
        assert "src/b.ts" in cl["files_created"]
        assert len(cl["checklist"]) == 2
        assert cl["checklist"][0]["criteria"] == "Modified src/a.ts"
        assert cl["checklist"][1]["criteria"] == "Created src/b.ts"


# -- Full run() ---------------------------------------------------------------

class TestCodeAgentRun:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="code_agent_run_test_")
        self.session_dir = os.path.join(self.tmp, "session")
        os.makedirs(self.session_dir)
        self.activity_log = os.path.join(self.session_dir, "pipeline-activity.log")
        self.plan = os.path.join(self.tmp, "plan.md")
        Path(self.plan).write_text("# Plan\n\nAdd a button.")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    @patch("architect.executor.engine.agents.code_agent._report_pass")
    @patch("architect.executor.engine.agents.code_agent._has_code_changes", return_value=True)
    @patch("architect.executor.engine.agents.code_agent._generate_checklist", return_value={
        "step": "code", "summary": "1 file", "files_modified": ["a.ts"],
        "files_created": [], "checklist": [],
    })
    @patch.object(AgentRunner, "run", return_value=AgentResult(
        exit_code=0, stdout="done", transcript_path="/t.log", timed_out=False,
    ))
    def test_pass_when_changes_exist(self, mock_agent, mock_cl, mock_changes, mock_pass):
        state = _make_state(plan_file=self.plan, session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        outcome = run(state, preset, self.tmp, self.session_dir,
                      self.activity_log, agent_command="/bin/true")

        assert outcome.passed is True
        assert outcome.checklist is not None
        mock_pass.assert_called_once()
        # Checklist JSON written to session dir
        assert (Path(self.session_dir) / "code-checklist.json").is_file()

    @patch("architect.executor.engine.agents.code_agent._report_fail")
    @patch("architect.executor.engine.agents.code_agent._has_code_changes", return_value=False)
    @patch.object(AgentRunner, "run", return_value=AgentResult(
        exit_code=0, stdout="done", transcript_path="/t.log", timed_out=False,
    ))
    def test_fail_when_no_changes(self, mock_agent, mock_changes, mock_fail):
        state = _make_state(plan_file=self.plan, session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        outcome = run(state, preset, self.tmp, self.session_dir,
                      self.activity_log, agent_command="/bin/true")

        assert outcome.passed is False
        assert "No code changes" in outcome.reason
        mock_fail.assert_called_once_with(self.session_dir, "No code changes produced")

    @patch("architect.executor.engine.agents.code_agent._report_pass")
    @patch("architect.executor.engine.agents.code_agent._has_code_changes", return_value=True)
    @patch("architect.executor.engine.agents.code_agent._generate_checklist", return_value={
        "step": "code", "summary": "", "files_modified": [],
        "files_created": [], "checklist": [],
    })
    @patch.object(AgentRunner, "run", return_value=AgentResult(
        exit_code=1, stdout="agent crashed", transcript_path="/t.log", timed_out=False,
    ))
    def test_pass_despite_agent_crash_if_changes_exist(self, mock_agent, mock_cl,
                                                        mock_changes, mock_pass):
        """Agent exit code doesn't matter — outcome is determined by whether changes exist."""
        state = _make_state(plan_file=self.plan, session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        outcome = run(state, preset, self.tmp, self.session_dir,
                      self.activity_log, agent_command="/bin/false")

        assert outcome.passed is True

    @patch("architect.executor.engine.agents.code_agent._report_pass")
    @patch("architect.executor.engine.agents.code_agent._has_code_changes", return_value=True)
    @patch("architect.executor.engine.agents.code_agent._generate_checklist", return_value={
        "step": "code", "summary": "", "files_modified": [],
        "files_created": [], "checklist": [],
    })
    @patch.object(AgentRunner, "run", return_value=AgentResult(
        exit_code=0, stdout="timeout", transcript_path="/t.log", timed_out=True,
    ))
    def test_pass_despite_timeout_if_changes_exist(self, mock_agent, mock_cl,
                                                    mock_changes, mock_pass):
        """Timeout doesn't matter if the agent made valid changes before being killed."""
        state = _make_state(plan_file=self.plan, session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        outcome = run(state, preset, self.tmp, self.session_dir,
                      self.activity_log, agent_command="/bin/true")

        assert outcome.passed is True

    @patch("architect.executor.engine.agents.code_agent._report_pass")
    @patch("architect.executor.engine.agents.code_agent._has_code_changes", return_value=True)
    @patch("architect.executor.engine.agents.code_agent._generate_checklist")
    @patch.object(AgentRunner, "run", return_value=AgentResult(
        exit_code=0, stdout="done", transcript_path="/t.log", timed_out=False,
    ))
    def test_agent_runner_receives_correct_params(self, mock_agent, mock_cl,
                                                    mock_changes, mock_pass):
        mock_cl.return_value = {"step": "code", "summary": "", "files_modified": [],
                                "files_created": [], "checklist": []}
        state = _make_state(plan_file=self.plan, session_dir=self.session_dir,
                            packages=["apps/web"])
        preset = _make_preset(self.tmp)

        run(state, preset, self.tmp, self.session_dir,
            self.activity_log, model="opus", max_turns=30, timeout_s=1800,
            agent_command="my-stub")

        mock_agent.assert_called_once()
        call_kwargs = mock_agent.call_args
        assert call_kwargs.kwargs["model"] == "opus"
        assert call_kwargs.kwargs["max_turns"] == 30
        assert call_kwargs.kwargs["timeout_s"] == 1800


# -- Prompt snapshot ----------------------------------------------------------

class TestPromptSnapshot:
    """Verify the prompt doesn't contain protocol instructions."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="prompt_snapshot_test_")
        self.session_dir = os.path.join(self.tmp, "session")
        os.makedirs(self.session_dir)

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_no_protocol_artifacts(self):
        plan = os.path.join(self.tmp, "plan.md")
        Path(plan).write_text("Add a feature.")
        state = _make_state(plan_file=plan, session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        prompt = generate_prompt(state, preset)

        # No protocol remnants
        for forbidden in [
            "Pipeline Protocol",
            "pipeline_cli.py pass",
            "pipeline_cli.py fail",
            "On success:",
            "On failure:",
            "WARNING:",
            "discarded",
            "pass/fail",
            "checklist.json",
        ]:
            assert forbidden not in prompt, f"Found forbidden text: {forbidden}"

        # Has the plan content
        assert "Add a feature." in prompt
        # Has the task structure
        assert "## Task" in prompt
        assert "## After Implementing" in prompt


# -- Build result requirement in prompt ---------------------------------------

class TestBuildResultInPrompt:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="build_result_prompt_test_")
        self.session_dir = os.path.join(self.tmp, "session")
        os.makedirs(self.session_dir)

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_prompt_requires_build_result_json(self):
        plan = os.path.join(self.tmp, "plan.md")
        Path(plan).write_text("Add a feature.")
        state = _make_state(plan_file=plan, session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        prompt = generate_prompt(state, preset)

        assert "build-result.json" in prompt
        assert '"passed": true' in prompt
        assert "judge will reject" in prompt

    def test_prompt_includes_session_dir_for_build_result(self):
        plan = os.path.join(self.tmp, "plan.md")
        Path(plan).write_text("Plan")
        state = _make_state(plan_file=plan, session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        prompt = generate_prompt(state, preset)

        assert f"{self.session_dir}/build-result.json" in prompt
