
"""Tests for engine.runner — execute, prompts, package validation, truncation."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess

from forge.executor.engine.runner import (
    execute_command, build_context, generate_fix_prompt, generate_ai_fix_prompt,
    StepResult, PIPELINE_CLI, _needs_shell, _select_command,
)
from forge.executor.engine.registry import StepDefinition, Preset
from forge.executor.engine.state import PipelineState, StepState
from tests.executor.helpers import make_state, make_preset


class TestBuildContext(unittest.TestCase):
    @patch("forge.executor.engine.runner.repo_root", return_value="/repo")
    @patch("forge.executor.engine.runner._changed_files", return_value="file.ts")
    def test_basic_context(self, _cf, _rr):
        state = make_state(session_dir="/session")
        preset = make_preset()
        ctx = build_context(state, preset)
        self.assertEqual(ctx["REPO_ROOT"], "/repo")
        self.assertEqual(ctx["SESSION_DIR"], "/session")
        self.assertIn("PIPELINE_CLI", ctx)
        self.assertEqual(ctx["CHANGED_FILES"], "file.ts")

    @patch("forge.executor.engine.runner.repo_root", return_value="/repo")
    @patch("forge.executor.engine.runner._changed_files", return_value="")
    def test_package_context(self, _cf, _rr):
        state = make_state(packages=["apps/webapp"])
        preset = make_preset()
        ctx = build_context(state, preset, package="apps/webapp")
        self.assertEqual(ctx["PACKAGE"], "apps/webapp")
        self.assertEqual(ctx["PACKAGE_SLUG"], "apps-webapp")

    @patch("forge.executor.engine.runner.repo_root", return_value="/repo")
    @patch("forge.executor.engine.runner._changed_files", return_value="")
    def test_invalid_package_raises(self, _cf, _rr):
        state = make_state()
        preset = make_preset()
        with self.assertRaises(ValueError):
            build_context(state, preset, package="pkg; rm -rf /")

    @patch("forge.executor.engine.runner.repo_root", return_value="/repo")
    @patch("forge.executor.engine.runner._changed_files", return_value="")
    def test_valid_package_names(self, _cf, _rr):
        state = make_state()
        preset = make_preset()
        for valid in ["apps/webapp", "platform/ui", "@example/test-tools", "my_pkg.v2"]:
            ctx = build_context(state, preset, package=valid)
            self.assertEqual(ctx["PACKAGE"], valid)

    @patch("forge.executor.engine.runner.repo_root", return_value="/repo")
    @patch("forge.executor.engine.runner._changed_files", return_value="")
    def test_no_packages_shows_none(self, _cf, _rr):
        state = make_state(packages=[])
        preset = make_preset()
        ctx = build_context(state, preset)
        self.assertEqual(ctx["AFFECTED_PACKAGES"], "(none)")


class TestExecuteCommand(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="run_test_")
        self.session_dir = os.path.join(self.tmp, "session")
        os.makedirs(self.session_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    @patch("forge.executor.engine.runner.repo_root", return_value="/tmp")
    @patch("forge.executor.engine.runner._changed_files", return_value="")
    def test_successful_command(self, _cf, _rr):
        step = StepDefinition(name="test_step", run_command="echo hello")
        state = make_state(session_dir=self.session_dir)
        preset = make_preset()
        result = execute_command(step, state, preset)
        self.assertTrue(result.passed)
        self.assertIn("hello", result.output)

    @patch("forge.executor.engine.runner.repo_root", return_value="/tmp")
    @patch("forge.executor.engine.runner._changed_files", return_value="")
    def test_failing_command(self, _cf, _rr):
        step = StepDefinition(name="test_step", run_command="exit 1")
        state = make_state(session_dir=self.session_dir)
        preset = make_preset()
        result = execute_command(step, state, preset)
        self.assertFalse(result.passed)

    @patch("forge.executor.engine.runner.repo_root", return_value="/tmp")
    @patch("forge.executor.engine.runner._changed_files", return_value="")
    def test_error_file_written(self, _cf, _rr):
        step = StepDefinition(name="test_step", run_command="echo ERR >&2; exit 1",
                              error_file="errors.txt")
        state = make_state(session_dir=self.session_dir)
        preset = make_preset()
        result = execute_command(step, state, preset)
        self.assertFalse(result.passed)
        self.assertTrue(os.path.isfile(result.error_file))

    @patch("forge.executor.engine.runner.repo_root", return_value="/tmp")
    @patch("forge.executor.engine.runner._changed_files", return_value="")
    def test_timeout(self, _cf, _rr):
        step = StepDefinition(name="test_step", run_command="sleep 60", timeout=1)
        state = make_state(session_dir=self.session_dir)
        preset = make_preset()
        result = execute_command(step, state, preset)
        self.assertFalse(result.passed)
        self.assertIn("timed out", result.output)

    @patch("forge.executor.engine.runner.repo_root", return_value="/tmp")
    @patch("forge.executor.engine.runner._changed_files", return_value="")
    def test_per_package_no_packages(self, _cf, _rr):
        step = StepDefinition(name="lint", run_command="echo ok", per_package=True)
        state = make_state(packages=[])
        preset = make_preset()
        result = execute_command(step, state, preset)
        self.assertFalse(result.passed)
        self.assertIn("No affected packages", result.output)


class TestGenerateFixPrompt(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fix_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    @patch("forge.executor.engine.runner.repo_root", return_value="/repo")
    @patch("forge.executor.engine.runner._changed_files", return_value="")
    def test_fix_prompt_contains_protocol(self, _cf, _rr):
        step = StepDefinition(name="build", run_command="bazel build :tsc")
        state = make_state(session_dir=self.tmp)
        preset = make_preset()
        prompt = generate_fix_prompt(step, state, preset)
        self.assertIn("Pipeline Protocol", prompt)
        self.assertIn("build", prompt)

    @patch("forge.executor.engine.runner.repo_root", return_value="/repo")
    @patch("forge.executor.engine.runner._changed_files", return_value="")
    def test_fix_prompt_references_error_file(self, _cf, _rr):
        error_path = Path(self.tmp, "errors.txt")
        error_path.write_text("error: missing import\n")
        step = StepDefinition(name="build", run_command="bazel build", error_file="errors.txt")
        state = make_state(session_dir=self.tmp)
        preset = make_preset()
        prompt = generate_fix_prompt(step, state, preset)
        self.assertIn(str(error_path), prompt)
        self.assertIn("Read this file", prompt)
        self.assertNotIn("error: missing import", prompt)

    @patch("forge.executor.engine.runner.repo_root", return_value="/repo")
    @patch("forge.executor.engine.runner._changed_files", return_value="")
    def test_fix_prompt_with_hints(self, _cf, _rr):
        step = StepDefinition(name="build", run_command="bazel build",
                              fix_hints="Check BUILD files")
        state = make_state(session_dir=self.tmp)
        preset = make_preset()
        prompt = generate_fix_prompt(step, state, preset)
        self.assertIn("Check BUILD files", prompt)


class TestGenerateAiFixPrompt(unittest.TestCase):
    @patch("forge.executor.engine.runner.repo_root", return_value="/repo")
    @patch("forge.executor.engine.runner._changed_files", return_value="")
    def test_ai_fix_prompt(self, _cf, _rr):
        step = StepDefinition(name="code_review", step_type="ai")
        state = make_state(session_dir="/session")
        preset = make_preset()
        prompt = generate_ai_fix_prompt(step, state, preset)
        self.assertIn("fix ALL issues", prompt)
        self.assertIn("/session", prompt)

    @patch("forge.executor.engine.runner.repo_root", return_value="/repo")
    @patch("forge.executor.engine.runner._changed_files", return_value="")
    def test_ai_fix_prompt_resolves_preset_dir_in_skill(self, _cf, _rr):
        """Regression: ${PRESET_DIR} in skill paths must be resolved."""
        tmp = tempfile.mkdtemp(prefix="fix_skill_test_")
        try:
            skill_file = Path(tmp) / "skills" / "review.md"
            skill_file.parent.mkdir(parents=True)
            skill_file.write_text("Review instructions here.")

            step = StepDefinition(
                name="code_review", step_type="ai",
                skill="${PRESET_DIR}/skills/review.md",
            )
            state = make_state(session_dir="/session")
            preset = make_preset(preset_dir=Path(tmp))
            prompt = generate_ai_fix_prompt(step, state, preset)
            self.assertIn("Review instructions here.", prompt)
        finally:
            shutil.rmtree(tmp)


class TestGenerateAiPrompt(unittest.TestCase):
    @patch("forge.executor.engine.runner.repo_root", return_value="/repo")
    @patch("forge.executor.engine.runner._changed_files", return_value="")
    def test_resolves_preset_dir_in_skill(self, _cf, _rr):
        """Regression: ${PRESET_DIR} in skill paths must be resolved."""
        from forge.executor.engine.runner import generate_ai_prompt

        tmp = tempfile.mkdtemp(prefix="ai_skill_test_")
        try:
            skill_file = Path(tmp) / "skills" / "code-review.md"
            skill_file.parent.mkdir(parents=True)
            skill_file.write_text("Code review skill content.")

            step = StepDefinition(
                name="code_review", step_type="ai",
                skill="${PRESET_DIR}/skills/code-review.md",
            )
            state = make_state(session_dir="/tmp/test-session")
            preset = make_preset(preset_dir=Path(tmp))
            prompt = generate_ai_prompt(step, state, preset)
            self.assertIn("Code review skill content.", prompt)
        finally:
            shutil.rmtree(tmp)

    @patch("forge.executor.engine.runner.repo_root", return_value="/repo")
    @patch("forge.executor.engine.runner._changed_files", return_value="")
    def test_unresolved_preset_dir_raises(self, _cf, _rr):
        """${PRESET_DIR} pointing to missing file should raise FileNotFoundError."""
        from forge.executor.engine.runner import generate_ai_prompt

        step = StepDefinition(
            name="code_review", step_type="ai",
            skill="${PRESET_DIR}/skills/nonexistent.md",
        )
        state = make_state(session_dir="/tmp/test-session")
        preset = make_preset(preset_dir=Path("/tmp/no-such-preset"))
        with self.assertRaises(FileNotFoundError):
            generate_ai_prompt(step, state, preset)


class TestPipelineCLIPath(unittest.TestCase):
    def test_pipeline_cli_file_exists(self):
        """Regression: PIPELINE_CLI must point to an existing file."""
        self.assertTrue(
            os.path.isfile(PIPELINE_CLI),
            f"PIPELINE_CLI does not exist: {PIPELINE_CLI}",
        )

    def test_pipeline_cli_is_python_file(self):
        self.assertTrue(PIPELINE_CLI.endswith(".py"))


class TestNeedsShell(unittest.TestCase):
    def test_pipe_needs_shell(self):
        self.assertTrue(_needs_shell("ls | grep foo"))

    def test_redirect_needs_shell(self):
        self.assertTrue(_needs_shell("echo hello > file.txt"))

    def test_and_needs_shell(self):
        self.assertTrue(_needs_shell("cd /repo && npm test"))

    def test_subshell_needs_shell(self):
        self.assertTrue(_needs_shell("$(which node) --version"))

    def test_simple_command_no_shell(self):
        self.assertFalse(_needs_shell("echo hello"))

    def test_cd_needs_shell(self):
        self.assertTrue(_needs_shell("cd /repo"))

    def test_exit_needs_shell(self):
        self.assertTrue(_needs_shell("exit 1"))

    def test_empty_string(self):
        self.assertFalse(_needs_shell(""))


class TestSelectCommand(unittest.TestCase):
    """Bug 4+6 regression: _select_command picks bazel_run_command in Bazel repos."""

    @patch("forge.executor.engine.runner.is_bazel_repo", return_value=True)
    def test_returns_bazel_command_in_bazel_repo(self, _):
        step = StepDefinition(name="build", run_command="npm run build",
                              bazel_run_command="bazel build :tsc")
        self.assertEqual(_select_command(step), "bazel build :tsc")

    @patch("forge.executor.engine.runner.is_bazel_repo", return_value=False)
    def test_returns_run_command_outside_bazel_repo(self, _):
        step = StepDefinition(name="build", run_command="npm run build",
                              bazel_run_command="bazel build :tsc")
        self.assertEqual(_select_command(step), "npm run build")

    @patch("forge.executor.engine.runner.is_bazel_repo", return_value=True)
    def test_returns_run_command_when_bazel_command_empty(self, _):
        step = StepDefinition(name="lint", run_command="npm run lint")
        self.assertEqual(_select_command(step), "npm run lint")


if __name__ == "__main__":
    unittest.main()
