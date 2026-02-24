
"""Tests for code_agent.py — code step wrapper with deterministic verification."""

import json
import os
import tempfile
import shutil
import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from forge.executor.engine.agents.code_agent import (
    generate_prompt,
    run,
    CodeAgentOutcome,
    _has_code_changes,
    _generate_checklist,
    _verify_build,
)
from forge.core.runner import AgentRunner, AgentResult
from forge.executor.engine.registry import StepDefinition, Preset, PipelineDefinition
from forge.executor.engine.state import PipelineState, StepState


def _make_preset(preset_dir=".", build_command="", bazel_build_command=""):
    return Preset(
        name="test",
        version=3,
        description="Test preset",
        pipelines={"full": PipelineDefinition(steps=["code"], dependencies={})},
        steps={"code": StepDefinition(name="code", step_type="ai")},
        models={"code": "sonnet", "fix": "sonnet"},
        preset_dir=Path(preset_dir),
        build_command=build_command,
        bazel_build_command=bazel_build_command,
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

    @patch("forge.executor.engine.agents.code_agent.is_bazel_repo", return_value=True)
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
            bazel_build_command="cd {{REPO_ROOT}} && bazel build {{BUILD_TARGETS}}",
        )

        prompt = generate_prompt(state, preset)

        assert "//apps/webapp/web/..." in prompt
        assert "//libs/common/..." in prompt

    @patch("forge.executor.engine.agents.code_agent.is_bazel_repo", return_value=True)
    def test_build_command_skipped_when_no_packages_and_needs_targets(self, _mock_bazel):
        plan = os.path.join(self.tmp, "plan.md")
        Path(plan).write_text("Plan")
        state = _make_state(plan_file=plan, session_dir=self.session_dir, packages=[])
        preset = _make_preset(
            self.tmp,
            bazel_build_command="cd {{REPO_ROOT}} && bazel build {{BUILD_TARGETS}}",
        )

        prompt = generate_prompt(state, preset)

        assert "No build targets" in prompt
        assert "{{BUILD_TARGETS}}" not in prompt

    def test_no_build_command_when_no_build_command_configured(self):
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
    @patch("forge.executor.engine.agents.code_agent.subprocess.run")
    def test_tracked_changes_detected(self, mock_run):
        mock_run.side_effect = [
            MagicMock(stdout=" src/index.ts | 5 +++--\n 1 file changed\n"),
        ]
        assert _has_code_changes("/repo") is True

    @patch("forge.executor.engine.agents.code_agent.subprocess.run")
    def test_untracked_changes_detected(self, mock_run):
        mock_run.side_effect = [
            MagicMock(stdout=""),       # git diff --stat HEAD (nothing)
            MagicMock(stdout="new-file.ts\n"),  # git ls-files --others
        ]
        assert _has_code_changes("/repo") is True

    @patch("forge.executor.engine.agents.code_agent.subprocess.run")
    def test_no_changes(self, mock_run):
        mock_run.side_effect = [
            MagicMock(stdout=""),  # git diff --stat HEAD
            MagicMock(stdout=""),  # git ls-files --others
        ]
        assert _has_code_changes("/repo") is False


# -- _generate_checklist ------------------------------------------------------

class TestVerifyBuild:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="verify_build_test_")
        self.session_dir = os.path.join(self.tmp, "session")
        os.makedirs(self.session_dir)

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    @patch("forge.executor.engine.agents.code_agent.is_bazel_repo", return_value=False)
    @patch("forge.executor.engine.agents.code_agent.subprocess.run")
    def test_build_succeeds(self, mock_run, _mock_bazel):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        state = _make_state(session_dir=self.session_dir)
        preset = _make_preset(self.tmp, build_command="npm run build")

        result = _verify_build(state, preset, self.tmp, self.session_dir)

        assert result["passed"] is True

    @patch("forge.executor.engine.agents.code_agent.is_bazel_repo", return_value=False)
    @patch("forge.executor.engine.agents.code_agent.subprocess.run")
    def test_build_fails_saves_errors(self, mock_run, _mock_bazel):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="error TS2304: Cannot find name 'x'\n",
        )
        state = _make_state(session_dir=self.session_dir)
        preset = _make_preset(self.tmp, build_command="npm run build")

        result = _verify_build(state, preset, self.tmp, self.session_dir)

        assert result["passed"] is False
        assert "Build failed" in result["reason"]
        # Verify errors saved to file
        build_errors_path = Path(self.session_dir) / "build-errors.txt"
        assert build_errors_path.is_file()
        assert "error TS2304" in build_errors_path.read_text()

    @patch("forge.executor.engine.agents.code_agent.is_bazel_repo", return_value=False)
    def test_no_build_command_configured(self, _mock_bazel):
        state = _make_state(session_dir=self.session_dir)
        preset = _make_preset(self.tmp, build_command="")

        result = _verify_build(state, preset, self.tmp, self.session_dir)

        assert result["passed"] is True
        assert "No build verification configured" in result["reason"]

    @patch("forge.executor.engine.agents.code_agent.is_bazel_repo", return_value=True)
    def test_no_packages_no_changes_and_template_needs_targets(self, _mock_bazel):
        """No packages in state, no code changes on disk — pass (nothing to verify)."""
        state = _make_state(session_dir=self.session_dir, packages=[])
        preset = _make_preset(
            self.tmp,
            bazel_build_command="bazel build {{BUILD_TARGETS}}",
        )

        result = _verify_build(state, preset, self.tmp, self.session_dir)

        assert result["passed"] is True
        assert "No code changes" in result["reason"]

    @patch("forge.executor.engine.agents.code_agent.is_bazel_repo", return_value=True)
    @patch("forge.executor.engine.agents.code_agent._has_code_changes", return_value=True)
    @patch("forge.executor.engine.agents.code_agent._detect_packages", return_value=[])
    def test_no_packages_with_changes_fails(self, _mock_detect, _mock_changes, _mock_bazel):
        """Code changes exist but no build targets found — must fail, not silently pass."""
        state = _make_state(session_dir=self.session_dir, packages=[])
        preset = _make_preset(
            self.tmp,
            bazel_build_command="bazel build {{BUILD_TARGETS}}",
        )

        result = _verify_build(state, preset, self.tmp, self.session_dir)

        assert result["passed"] is False
        assert "no build targets" in result["reason"].lower()

    @patch("forge.executor.engine.agents.code_agent.is_bazel_repo", return_value=True)
    @patch("forge.executor.engine.agents.code_agent._detect_packages", return_value=["platform/authoring/timeline-authoring"])
    def test_no_packages_in_state_falls_back_to_detection(self, _mock_detect, _mock_bazel):
        """Packages not in state yet — detect from working tree and build."""
        state = _make_state(session_dir=self.session_dir, packages=[])
        preset = _make_preset(
            self.tmp,
            bazel_build_command="bazel build {{BUILD_TARGETS}}",
        )

        with patch("forge.executor.engine.agents.code_agent.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = _verify_build(state, preset, self.tmp, self.session_dir)

        assert result["passed"] is True
        # Verify the build command included the detected package
        call_args = mock_run.call_args
        assert "//platform/authoring/timeline-authoring/..." in call_args[1].get("args", call_args[0][0]) or \
               "//platform/authoring/timeline-authoring/..." in str(call_args)

    @patch("forge.executor.engine.agents.code_agent.is_bazel_repo", return_value=True)
    @patch("forge.executor.engine.agents.code_agent.build_context")
    def test_build_timeout(self, mock_context, _mock_bazel):
        # Mock build_context to return valid context, then mock subprocess.run to timeout
        mock_context.return_value = {
            "REPO_ROOT": self.tmp,
            "BUILD_TARGETS": "//apps/web/...",
            "AFFECTED_PACKAGES": "apps/web",
        }

        with patch.object(subprocess, "run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("cmd", 300)
            state = _make_state(session_dir=self.session_dir, packages=["apps/web"])
            preset = _make_preset(
                self.tmp,
                bazel_build_command="bazel build {{BUILD_TARGETS}}",
            )

            result = _verify_build(state, preset, self.tmp, self.session_dir)

            assert result["passed"] is False
            assert "timed out" in result["reason"].lower()


class TestGenerateChecklist:
    @patch("forge.executor.engine.agents.code_agent.subprocess.run")
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

    @patch("forge.executor.engine.agents.code_agent._verify_build", return_value={"passed": True, "reason": "OK"})
    @patch("forge.executor.engine.agents.code_agent._report_pass")
    @patch("forge.executor.engine.agents.code_agent._has_code_changes", return_value=True)
    @patch("forge.executor.engine.agents.code_agent._generate_checklist", return_value={
        "step": "code", "summary": "1 file", "files_modified": ["a.ts"],
        "files_created": [], "checklist": [],
    })
    @patch.object(AgentRunner, "run", return_value=AgentResult(
        exit_code=0, stdout="done", transcript_path="/t.log", timed_out=False,
    ))
    def test_pass_when_changes_exist(self, mock_agent, mock_cl, mock_changes, mock_pass, mock_verify_build):
        state = _make_state(plan_file=self.plan, session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        outcome = run(state, preset, self.tmp, self.session_dir,
                      self.activity_log, agent_command="/bin/true")

        assert outcome.passed is True
        assert outcome.checklist is not None
        mock_pass.assert_called_once()
        mock_verify_build.assert_called_once()
        # Checklist JSON written to session dir
        assert (Path(self.session_dir) / "code-checklist.json").is_file()

    @patch("forge.executor.engine.agents.code_agent._report_fail")
    @patch("forge.executor.engine.agents.code_agent._has_code_changes", return_value=False)
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

    @patch("forge.executor.engine.agents.code_agent._verify_build", return_value={"passed": False, "reason": "Build failed: error TS2304"})
    @patch("forge.executor.engine.agents.code_agent._report_fail")
    @patch("forge.executor.engine.agents.code_agent._has_code_changes", return_value=True)
    @patch.object(AgentRunner, "run", return_value=AgentResult(
        exit_code=0, stdout="done", transcript_path="/t.log", timed_out=False,
    ))
    def test_fail_when_build_fails(self, mock_agent, mock_changes, mock_fail, mock_verify_build):
        state = _make_state(plan_file=self.plan, session_dir=self.session_dir)
        preset = _make_preset(self.tmp, build_command="npm run build")

        outcome = run(state, preset, self.tmp, self.session_dir,
                      self.activity_log, agent_command="/bin/true")

        assert outcome.passed is False
        assert "Build failed" in outcome.reason
        mock_fail.assert_called_once()
        call_args = mock_fail.call_args[0]
        assert "Build failed" in call_args[1]

    @patch("forge.executor.engine.agents.code_agent._verify_build", return_value={"passed": True, "reason": "OK"})
    @patch("forge.executor.engine.agents.code_agent._report_pass")
    @patch("forge.executor.engine.agents.code_agent._has_code_changes", return_value=True)
    @patch("forge.executor.engine.agents.code_agent._generate_checklist", return_value={
        "step": "code", "summary": "", "files_modified": [],
        "files_created": [], "checklist": [],
    })
    @patch.object(AgentRunner, "run", return_value=AgentResult(
        exit_code=1, stdout="agent crashed", transcript_path="/t.log", timed_out=False,
    ))
    def test_pass_despite_agent_crash_if_changes_exist(self, mock_agent, mock_cl,
                                                        mock_changes, mock_pass, mock_verify_build):
        """Agent exit code doesn't matter — outcome is determined by whether changes exist."""
        state = _make_state(plan_file=self.plan, session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        outcome = run(state, preset, self.tmp, self.session_dir,
                      self.activity_log, agent_command="/bin/false")

        assert outcome.passed is True
        mock_verify_build.assert_called_once()

    @patch("forge.executor.engine.agents.code_agent._verify_build", return_value={"passed": True, "reason": "OK"})
    @patch("forge.executor.engine.agents.code_agent._report_pass")
    @patch("forge.executor.engine.agents.code_agent._has_code_changes", return_value=True)
    @patch("forge.executor.engine.agents.code_agent._generate_checklist", return_value={
        "step": "code", "summary": "", "files_modified": [],
        "files_created": [], "checklist": [],
    })
    @patch.object(AgentRunner, "run", return_value=AgentResult(
        exit_code=0, stdout="timeout", transcript_path="/t.log", timed_out=True,
    ))
    def test_pass_despite_timeout_if_changes_exist(self, mock_agent, mock_cl,
                                                    mock_changes, mock_pass, mock_verify_build):
        """Timeout doesn't matter if the agent made valid changes before being killed."""
        state = _make_state(plan_file=self.plan, session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        outcome = run(state, preset, self.tmp, self.session_dir,
                      self.activity_log, agent_command="/bin/true")

        assert outcome.passed is True
        mock_verify_build.assert_called_once()

    @patch("forge.executor.engine.agents.code_agent._verify_build", return_value={"passed": True, "reason": "OK"})
    @patch("forge.executor.engine.agents.code_agent._report_pass")
    @patch("forge.executor.engine.agents.code_agent._has_code_changes", return_value=True)
    @patch("forge.executor.engine.agents.code_agent._generate_checklist")
    @patch.object(AgentRunner, "run", return_value=AgentResult(
        exit_code=0, stdout="done", transcript_path="/t.log", timed_out=False,
    ))
    def test_agent_runner_receives_correct_params(self, mock_agent, mock_cl,
                                                    mock_changes, mock_pass, mock_verify_build):
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
        mock_verify_build.assert_called_once()


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

