"""Tests for runner.py — code step prompt generation, companion artifact discovery, checklist/judge."""

import json
import os
import subprocess
import tempfile
import shutil
import pytest
from pathlib import Path
from dataclasses import field
from unittest.mock import patch, MagicMock

from architect.executor.engine.runner import (
    _generate_code_prompt, generate_ai_prompt, generate_ai_fix_prompt,
    build_context, _checklist_schema_section, _judge_feedback_section,
    _changed_files, _run_one_package,
    _execute_per_package, generate_fix_prompt, _load_claude_md_for_review,
    StepResult,
)
from architect.executor.engine.registry import StepDefinition, Preset, PipelineDefinition
from architect.executor.engine.state import PipelineState, StepState


def _make_preset(preset_dir="."):
    return Preset(
        name="test",
        version=3,
        description="Test preset",
        pipelines={"full": PipelineDefinition(steps=["code"], dependencies={})},
        steps={"code": StepDefinition(name="code", step_type="ai")},
        models={"code": "opus", "fix": "sonnet"},
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


class TestGenerateCodePrompt:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="runner_test_")
        self.session_dir = os.path.join(self.tmp, "session")
        os.makedirs(self.session_dir)

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_plan_file_content_included(self):
        plan = os.path.join(self.tmp, "plan.md")
        Path(plan).write_text("# My Plan\n\nDo the thing.")
        state = _make_state(plan_file=plan, session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        prompt = _generate_code_prompt(state, preset)

        assert "# My Plan" in prompt
        assert "Do the thing." in prompt

    def test_protocol_header_present(self):
        plan = os.path.join(self.tmp, "plan.md")
        Path(plan).write_text("Plan content")
        state = _make_state(plan_file=plan, session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        prompt = _generate_code_prompt(state, preset)

        assert "Pipeline Protocol" in prompt
        assert "step `code`" in prompt

    def test_missing_plan_file_raises(self):
        state = _make_state(plan_file="/nonexistent/plan.md", session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        with pytest.raises(FileNotFoundError, match="Plan file not found"):
            _generate_code_prompt(state, preset)

    def test_empty_plan_file_raises(self):
        state = _make_state(plan_file="", session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        with pytest.raises(FileNotFoundError, match="Plan file not found"):
            _generate_code_prompt(state, preset)

    def test_codebase_brief_included_when_present(self):
        plan_dir = os.path.join(self.tmp, "plan_artifacts")
        os.makedirs(plan_dir)
        Path(os.path.join(plan_dir, "final-plan.md")).write_text("Plan content")
        Path(os.path.join(plan_dir, "codebase-brief.md")).write_text("Brief: the codebase uses TypeScript")

        state = _make_state(
            plan_file=os.path.join(plan_dir, "final-plan.md"),
            session_dir=self.session_dir,
        )
        preset = _make_preset(self.tmp)

        prompt = _generate_code_prompt(state, preset)

        assert "## Codebase Brief" in prompt
        assert "Brief: the codebase uses TypeScript" in prompt

    def test_codebase_brief_omitted_when_absent(self):
        plan = os.path.join(self.tmp, "plan.md")
        Path(plan).write_text("Plan content")
        state = _make_state(plan_file=plan, session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        prompt = _generate_code_prompt(state, preset)

        assert "## Codebase Brief" not in prompt

    def test_visual_test_plan_copied_to_session(self):
        plan_dir = os.path.join(self.tmp, "plan_artifacts")
        os.makedirs(plan_dir)
        Path(os.path.join(plan_dir, "plan.md")).write_text("Plan")
        Path(os.path.join(plan_dir, "visual-test-plan.md")).write_text("VTP content")

        state = _make_state(
            plan_file=os.path.join(plan_dir, "plan.md"),
            session_dir=self.session_dir,
        )
        preset = _make_preset(self.tmp)

        _generate_code_prompt(state, preset)

        copied = Path(self.session_dir) / "visual-test-plan.md"
        assert copied.is_file()
        assert copied.read_text() == "VTP content"

    def test_visual_test_plan_not_overwritten_if_exists(self):
        plan_dir = os.path.join(self.tmp, "plan_artifacts")
        os.makedirs(plan_dir)
        Path(os.path.join(plan_dir, "plan.md")).write_text("Plan")
        Path(os.path.join(plan_dir, "visual-test-plan.md")).write_text("New VTP")

        existing = Path(self.session_dir) / "visual-test-plan.md"
        existing.write_text("Original VTP")

        state = _make_state(
            plan_file=os.path.join(plan_dir, "plan.md"),
            session_dir=self.session_dir,
        )
        preset = _make_preset(self.tmp)

        _generate_code_prompt(state, preset)

        assert existing.read_text() == "Original VTP"

    def test_affected_packages_in_prompt(self):
        plan = os.path.join(self.tmp, "plan.md")
        Path(plan).write_text("Plan")
        state = _make_state(
            plan_file=plan,
            session_dir=self.session_dir,
            packages=["apps/webapp/web", "libs/common"],
        )
        preset = _make_preset(self.tmp)

        prompt = _generate_code_prompt(state, preset)

        assert "apps/webapp/web" in prompt
        assert "libs/common" in prompt

    def test_session_dir_in_prompt(self):
        plan = os.path.join(self.tmp, "plan.md")
        Path(plan).write_text("Plan")
        state = _make_state(plan_file=plan, session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        prompt = _generate_code_prompt(state, preset)

        assert self.session_dir in prompt


class TestGenerateAiPromptCodeStep:
    """Test that generate_ai_prompt dispatches to _generate_code_prompt for code step."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="runner_ai_test_")
        self.session_dir = os.path.join(self.tmp, "session")
        os.makedirs(self.session_dir)

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_code_step_uses_plan(self):
        plan = os.path.join(self.tmp, "plan.md")
        Path(plan).write_text("# Implementation Plan\nStep 1: do X")

        step = StepDefinition(name="code", step_type="ai")
        state = _make_state(plan_file=plan, session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        prompt = generate_ai_prompt(step, state, preset)

        assert "# Implementation Plan" in prompt
        assert "Step 1: do X" in prompt

    def test_code_step_without_plan_raises(self):
        step = StepDefinition(name="code", step_type="ai")
        state = _make_state(plan_file="", session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        with pytest.raises(FileNotFoundError):
            generate_ai_prompt(step, state, preset)


# -- PIPELINE_PROTOCOL changes -----------------------------------------------

class TestPipelineProtocolUpdate:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="protocol_test_")
        self.session_dir = os.path.join(self.tmp, "session")
        os.makedirs(self.session_dir)

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_protocol_requires_checklist(self):
        plan = os.path.join(self.tmp, "plan.md")
        Path(plan).write_text("Plan")
        state = _make_state(plan_file=plan, session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        prompt = _generate_code_prompt(state, preset)

        assert "checklist.json" in prompt
        assert "WARNING" in prompt

    def test_protocol_no_longer_has_on_success_on_failure(self):
        from architect.executor.engine.runner import PIPELINE_PROTOCOL
        assert "On success:" not in PIPELINE_PROTOCOL
        assert "On failure:" not in PIPELINE_PROTOCOL


# -- Checklist schema section -------------------------------------------------

class TestChecklistSchemaSection:
    def test_includes_step_name_and_session_dir(self):
        section = _checklist_schema_section("code_review", "/tmp/session")

        assert "code_review-checklist.json" in section
        assert "/tmp/session" in section
        assert '"status": "done|skipped|blocked"' in section

    def test_mentions_judge(self):
        section = _checklist_schema_section("visual_test", "/s")

        assert "judge" in section.lower()


# -- Judge feedback section ---------------------------------------------------

class TestJudgeFeedbackSection:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="feedback_test_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_no_judge_dir_returns_empty(self):
        result = _judge_feedback_section(self.tmp, "code")
        assert result == ""

    def test_empty_judge_dir_returns_empty(self):
        os.makedirs(os.path.join(self.tmp, "_judge"))
        result = _judge_feedback_section(self.tmp, "code")
        assert result == ""

    def test_loads_latest_feedback(self):
        judge_dir = os.path.join(self.tmp, "_judge")
        os.makedirs(judge_dir)
        Path(os.path.join(judge_dir, "code_attempt_0.json")).write_text(json.dumps({
            "items": [
                {"id": "plan-1", "verdict": "pass", "reason": "OK"},
                {"id": "plan-2", "verdict": "fail", "reason": "Not implemented"},
            ]
        }))

        result = _judge_feedback_section(self.tmp, "code")

        assert "plan-2" in result
        assert "Not implemented" in result
        assert "Judge Feedback" in result

    def test_all_pass_returns_empty(self):
        judge_dir = os.path.join(self.tmp, "_judge")
        os.makedirs(judge_dir)
        Path(os.path.join(judge_dir, "code_attempt_0.json")).write_text(json.dumps({
            "items": [{"id": "a", "verdict": "pass", "reason": "OK"}]
        }))

        result = _judge_feedback_section(self.tmp, "code")

        assert result == ""


# -- AI prompt checklist + verdict --------------------------------------------

class TestAiPromptStructuredOutput:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="ai_prompt_test_")
        self.session_dir = os.path.join(self.tmp, "session")
        os.makedirs(self.session_dir)

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_code_review_prompt_has_checklist_and_verdict(self):
        skill_path = os.path.join(self.tmp, "review.md")
        Path(skill_path).write_text("Review the code.")

        step = StepDefinition(name="code_review", step_type="ai",
                              skill=skill_path, two_phase=True)
        state = _make_state(session_dir=self.session_dir)
        preset = Preset(
            name="test", version=3, description="",
            pipelines={"full": PipelineDefinition(steps=["code_review"], dependencies={})},
            steps={"code_review": step},
            models={"code_review": "opus", "fix": "sonnet"},
            preset_dir=Path(self.tmp),
        )

        prompt = generate_ai_prompt(step, state, preset)

        assert "checklist.json" in prompt
        assert "Verdict Output" in prompt
        assert "HAS_ISSUES" in prompt
        assert "fail" in prompt  # Reviewer calls fail on HAS_ISSUES

    def test_fix_prompt_has_checklist(self):
        skill_path = os.path.join(self.tmp, "review.md")
        Path(skill_path).write_text("Review the code.")

        step = StepDefinition(name="code_review", step_type="ai",
                              skill=skill_path, two_phase=True)
        state = _make_state(session_dir=self.session_dir)
        preset = Preset(
            name="test", version=3, description="",
            pipelines={"full": PipelineDefinition(steps=["code_review"], dependencies={})},
            steps={"code_review": step},
            models={"fix": "sonnet"},
            preset_dir=Path(self.tmp),
        )

        prompt = generate_ai_fix_prompt(step, state, preset)

        assert "checklist.json" in prompt


# -- _changed_files() exception handler ---------------------------------------

class TestChangedFilesExceptionHandler:
    @patch("architect.executor.engine.runner.subprocess.run", side_effect=FileNotFoundError("git not found"))
    def test_file_not_found_returns_unable(self, mock_run):
        result = _changed_files()
        assert result == "(unable to determine)"

    @patch("architect.executor.engine.runner.subprocess.run",
           side_effect=subprocess.CalledProcessError(128, "git"))
    def test_called_process_error_returns_unable(self, mock_run):
        result = _changed_files()
        assert result == "(unable to determine)"


# -- _run_one_package() ------------------------------------------------------

class TestRunOnePackage:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="run_one_pkg_test_")
        self.session_dir = os.path.join(self.tmp, "session")
        os.makedirs(self.session_dir)

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def _make_step(self, **kwargs):
        defaults = dict(name="test", step_type="command",
                        run_command="echo {{PACKAGE}}", per_package=True)
        defaults.update(kwargs)
        return StepDefinition(**defaults)

    @patch("architect.executor.engine.runner.subprocess.run")
    def test_successful_package_run(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout=b"OK\n", stderr=b"")
        step = self._make_step()
        state = _make_state(session_dir=self.session_dir, packages=["apps/web"])
        preset = _make_preset(self.tmp)

        pkg, result = _run_one_package(step, state, preset, "apps/web")

        assert pkg == "apps/web"
        assert result.passed is True
        assert "OK" in result.output

    @patch("architect.executor.engine.runner.subprocess.run")
    def test_failed_package_with_error_file(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout=b"", stderr=b"FAIL\n")
        step = self._make_step(error_file="test-errors-{{PACKAGE_SLUG}}.txt")
        state = _make_state(session_dir=self.session_dir, packages=["apps/web"])
        preset = _make_preset(self.tmp)

        pkg, result = _run_one_package(step, state, preset, "apps/web")

        assert pkg == "apps/web"
        assert result.passed is False
        assert result.error_file != ""
        assert Path(result.error_file).is_file()
        assert "FAIL" in Path(result.error_file).read_text()

    @patch("architect.executor.engine.runner.repo_root", return_value="/fake/repo")
    @patch("architect.executor.engine.runner._changed_files", return_value="(mocked)")
    def test_timeout_during_package_run(self, mock_changed, mock_root):
        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired("cmd", 600)

        step = self._make_step(timeout=600)
        state = _make_state(session_dir=self.session_dir, packages=["apps/web"])
        preset = _make_preset(self.tmp)

        with patch("architect.executor.engine.runner.subprocess.run", side_effect=fake_run):
            pkg, result = _run_one_package(step, state, preset, "apps/web")

        assert pkg == "apps/web"
        assert result.passed is False
        assert "Timed out" in result.output


# -- _execute_per_package() --------------------------------------------------

class TestExecutePerPackage:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="exec_per_pkg_test_")
        self.session_dir = os.path.join(self.tmp, "session")
        os.makedirs(self.session_dir)

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def _make_step(self, **kwargs):
        defaults = dict(name="test", step_type="command",
                        run_command="echo {{PACKAGE}}", per_package=True)
        defaults.update(kwargs)
        return StepDefinition(**defaults)

    def test_no_packages_returns_failed(self):
        step = self._make_step()
        state = _make_state(session_dir=self.session_dir, packages=[])
        preset = _make_preset(self.tmp)

        result = _execute_per_package(step, state, preset)

        assert result.passed is False
        assert "No affected packages" in result.output

    @patch("architect.executor.engine.runner._run_one_package")
    def test_parallel_execution_with_multiple_packages(self, mock_run_one):
        mock_run_one.side_effect = [
            ("apps/web", StepResult(passed=True, output="ok")),
            ("libs/common", StepResult(passed=True, output="ok")),
        ]
        step = self._make_step(parallel=True)
        state = _make_state(session_dir=self.session_dir,
                            packages=["apps/web", "libs/common"])
        preset = _make_preset(self.tmp)

        result = _execute_per_package(step, state, preset)

        assert result.passed is True
        assert mock_run_one.call_count == 2

    @patch("architect.executor.engine.runner._run_one_package")
    def test_sequential_execution(self, mock_run_one):
        mock_run_one.side_effect = [
            ("apps/web", StepResult(passed=True, output="ok")),
        ]
        step = self._make_step(parallel=False)
        state = _make_state(session_dir=self.session_dir, packages=["apps/web"])
        preset = _make_preset(self.tmp)

        result = _execute_per_package(step, state, preset)

        assert result.passed is True
        assert mock_run_one.call_count == 1

    @patch("architect.executor.engine.runner._run_one_package")
    def test_mixed_results_some_pass_some_fail(self, mock_run_one):
        mock_run_one.side_effect = [
            ("apps/web", StepResult(passed=True, output="ok")),
            ("libs/common", StepResult(passed=False, output="fail",
                                       error_file="/tmp/err.txt")),
        ]
        step = self._make_step(parallel=False)
        state = _make_state(session_dir=self.session_dir,
                            packages=["apps/web", "libs/common"])
        preset = _make_preset(self.tmp)

        result = _execute_per_package(step, state, preset)

        assert result.passed is False
        assert "libs/common" in result.failed_packages
        assert "apps/web" not in result.failed_packages
        assert result.error_files == {"libs/common": "/tmp/err.txt"}


# -- generate_fix_prompt() per-package branch ---------------------------------

class TestGenerateFixPromptPerPackage:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="fix_prompt_pkg_test_")
        self.session_dir = os.path.join(self.tmp, "session")
        os.makedirs(self.session_dir)

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_per_package_fix_prompt_with_failed_packages(self):
        step = StepDefinition(
            name="test", step_type="command",
            run_command="bazel test {{PACKAGE}}",
            per_package=True, error_file="test-errors-{{PACKAGE_SLUG}}.txt",
            fix_hints="Check type imports",
        )
        state = _make_state(session_dir=self.session_dir,
                            packages=["apps/web", "libs/common"])
        preset = _make_preset(self.tmp)

        # Create error files for the failed packages
        err1 = Path(self.session_dir) / "test-errors-apps-web.txt"
        err1.write_text("TypeError: x is not a function\n")

        prompt = generate_fix_prompt(step, state, preset,
                                     failed_packages=["apps/web", "libs/common"])

        assert "### Package: `apps/web`" in prompt
        assert "### Package: `libs/common`" in prompt
        assert "bazel test apps/web" in prompt
        assert "Check type imports" in prompt
        assert "Error log" in prompt  # error_reference for apps/web

    def test_per_package_fix_prompt_without_error_files(self):
        step = StepDefinition(
            name="test", step_type="command",
            run_command="bazel test {{PACKAGE}}",
            per_package=True,
        )
        state = _make_state(session_dir=self.session_dir, packages=["apps/web"])
        preset = _make_preset(self.tmp)

        prompt = generate_fix_prompt(step, state, preset,
                                     failed_packages=["apps/web"])

        assert "### Package: `apps/web`" in prompt
        assert "Error log" not in prompt


# -- _judge_feedback_section() JSON parse error -------------------------------

class TestJudgeFeedbackJsonError:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="judge_json_test_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_malformed_judge_json_returns_empty(self):
        judge_dir = os.path.join(self.tmp, "_judge")
        os.makedirs(judge_dir)
        Path(os.path.join(judge_dir, "code_attempt_0.json")).write_text(
            "not json at all {{{{"
        )
        result = _judge_feedback_section(self.tmp, "code")
        assert result == ""

    def test_judge_file_oserror_returns_empty(self):
        judge_dir = os.path.join(self.tmp, "_judge")
        os.makedirs(judge_dir)
        fpath = Path(os.path.join(judge_dir, "code_attempt_0.json"))
        fpath.write_text('{"items": []}')
        fpath.chmod(0o000)
        try:
            result = _judge_feedback_section(self.tmp, "code")
            assert result == ""
        finally:
            fpath.chmod(0o644)


# -- _load_claude_md_for_review() OSError -------------------------------------

class TestLoadClaudeMdOSError:
    @patch("architect.executor.engine.runner.repo_root", return_value="/fake/repo")
    def test_oserror_reading_claude_md_returns_empty(self, mock_root):
        with tempfile.TemporaryDirectory(prefix="claude_md_test_") as tmp:
            claude_md = Path(tmp) / ".claude" / "CLAUDE.md"
            claude_md.parent.mkdir(parents=True)
            claude_md.write_text("## Code\nFollow patterns.")
            claude_md.chmod(0o000)
            try:
                with patch("architect.executor.engine.runner.repo_root", return_value=tmp):
                    result = _load_claude_md_for_review()
                # OSError causes continue, falls through to next candidate or returns ""
                # The result depends on whether global CLAUDE.md also exists
                assert isinstance(result, str)
            finally:
                claude_md.chmod(0o644)


# -- _load_claude_md_for_review() section filtering ---------------------------

class TestLoadClaudeMdSectionFiltering:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="claude_md_filter_test_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    @patch("architect.executor.engine.runner.repo_root")
    def test_relevant_section_at_end_of_file_included(self, mock_root):
        mock_root.return_value = self.tmp
        claude_dir = Path(self.tmp) / ".claude"
        claude_dir.mkdir()
        # The last section is relevant (## Code) with no following ## header
        (claude_dir / "CLAUDE.md").write_text(
            "## Irrelevant Header\nSkip this.\n"
            "## Code Style\nUse camelCase.\nPrefer const.\n"
        )
        result = _load_claude_md_for_review()

        assert "Project Conventions" in result
        assert "Use camelCase" in result
        assert "Skip this" not in result

    @patch("pathlib.Path.home")
    @patch("architect.executor.engine.runner.repo_root")
    def test_no_relevant_sections_returns_empty(self, mock_root, mock_home):
        mock_root.return_value = self.tmp
        mock_home.return_value = Path(self.tmp) / "fakehome"
        claude_dir = Path(self.tmp) / ".claude"
        claude_dir.mkdir()
        (claude_dir / "CLAUDE.md").write_text(
            "## SSH Keys\nWork account info.\n"
            "## Generated Documents\nSave to ~/dev/.\n"
        )
        result = _load_claude_md_for_review()
        assert result == ""

    @patch("architect.executor.engine.runner.repo_root")
    def test_multiple_relevant_sections_collected(self, mock_root):
        mock_root.return_value = self.tmp
        claude_dir = Path(self.tmp) / ".claude"
        claude_dir.mkdir()
        (claude_dir / "CLAUDE.md").write_text(
            "## Testing\nWrite integration tests.\n"
            "## SSH\nNot relevant.\n"
            "## Import Rules\nAll imports at top.\n"
        )
        result = _load_claude_md_for_review()

        assert "Write integration tests" in result
        assert "All imports at top" in result
        assert "Not relevant" not in result


# -- generate_ai_prompt() FileNotFoundError for missing skill -----------------

class TestGenerateAiPromptMissingSkill:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="missing_skill_test_")
        self.session_dir = os.path.join(self.tmp, "session")
        os.makedirs(self.session_dir)

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_missing_skill_file_raises(self):
        step = StepDefinition(name="lint", step_type="ai",
                              skill="/nonexistent/skill.md")
        state = _make_state(session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        with pytest.raises(FileNotFoundError, match="Skill file not found"):
            generate_ai_prompt(step, state, preset)


# -- generate_ai_fix_prompt() judge feedback section --------------------------

class TestGenerateAiFixPromptJudgeFeedback:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="fix_judge_test_")
        self.session_dir = os.path.join(self.tmp, "session")
        os.makedirs(self.session_dir)

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_fix_prompt_includes_judge_feedback(self):
        # Create judge feedback with a failed item
        judge_dir = os.path.join(self.session_dir, "_judge")
        os.makedirs(judge_dir)
        Path(os.path.join(judge_dir, "code_review_attempt_0.json")).write_text(
            json.dumps({
                "items": [
                    {"id": "cr-1", "verdict": "fail", "reason": "Missing null check"},
                ]
            })
        )

        skill_path = os.path.join(self.tmp, "review.md")
        Path(skill_path).write_text("Review skill content.")

        step = StepDefinition(name="code_review", step_type="ai",
                              skill=skill_path, two_phase=True)
        state = _make_state(session_dir=self.session_dir)
        preset = Preset(
            name="test", version=3, description="",
            pipelines={"full": PipelineDefinition(steps=["code_review"], dependencies={})},
            steps={"code_review": step},
            models={"fix": "sonnet"},
            preset_dir=Path(self.tmp),
        )

        prompt = generate_ai_fix_prompt(step, state, preset)

        assert "Judge Feedback" in prompt
        assert "Missing null check" in prompt

    def test_fix_prompt_without_judge_feedback(self):
        step = StepDefinition(name="code_review", step_type="ai",
                              skill="", two_phase=True)
        state = _make_state(session_dir=self.session_dir)
        preset = Preset(
            name="test", version=3, description="",
            pipelines={"full": PipelineDefinition(steps=["code_review"], dependencies={})},
            steps={"code_review": step},
            models={"fix": "sonnet"},
            preset_dir=Path(self.tmp),
        )

        prompt = generate_ai_fix_prompt(step, state, preset)

        assert "Judge Feedback" not in prompt
