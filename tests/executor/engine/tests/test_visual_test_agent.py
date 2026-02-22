
"""Tests for visual_test_agent.py."""

import json
import os
import tempfile
import shutil
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from architect.executor.engine.agents.visual_test_agent import (
    run,
    VisualTestConfig,
    VisualTestOutcome,
    _detect_feature_name,
    _find_visual_test_plan,
    _build_context_file,
    _script_generation_prompt,
    _script_fix_prompt,
    _judge_prompt,
    _triage_prompt,
    _parse_judge_verdict,
    _parse_triage,
    _write_triage,
    _generate_dashboard,
    _execute_script,
    MAX_SCRIPT_FIX_ATTEMPTS,
)
from architect.core.runner import AgentRunner, AgentResult
from architect.executor.engine.registry import StepDefinition, Preset, PipelineDefinition
from architect.executor.engine.state import PipelineState, StepState


def _make_preset(preset_dir="."):
    return Preset(
        name="test",
        version=3,
        description="Test preset",
        pipelines={"full": PipelineDefinition(steps=["code", "visual_test"], dependencies={})},
        steps={
            "code": StepDefinition(name="code", step_type="ai"),
            "visual_test": StepDefinition(name="visual_test", step_type="ai"),
        },
        models={"code": "sonnet", "visual_test": "sonnet", "fix": "sonnet"},
        preset_dir=Path(preset_dir),
    )


def _make_state(session_dir="", packages=None, plan_file=""):
    return PipelineState(
        steps={
            "code": StepState(),
            "visual_test": StepState(),
        },
        step_order=["code", "visual_test"],
        dependency_graph={},
        session_dir=session_dir,
        pipeline="full",
        preset="test",
        plan_file=plan_file,
        affected_packages=packages or [],
        dev_server_port=8080,
    )


def _make_config(tmp_dir: str) -> VisualTestConfig:
    """Build a VisualTestConfig pointing at temp directories instead of hardcoded paths."""
    skill_dir = os.path.join(tmp_dir, "skill")
    os.makedirs(os.path.join(skill_dir, "scripts"), exist_ok=True)
    os.makedirs(os.path.join(skill_dir, "references"), exist_ok=True)
    runner_dir = os.path.join(tmp_dir, "playwright-runner")
    os.makedirs(runner_dir, exist_ok=True)
    creds_path = os.path.join(tmp_dir, "credentials.env")
    Path(creds_path).write_text("# empty")
    return VisualTestConfig(
        skill_dir=skill_dir,
        quirks_path=os.path.join(skill_dir, "references", "quirks.md"),
        credentials_path=creds_path,
        playwright_runner_dir=runner_dir,
        fixture_patterns=[],
    )


# -- Feature name detection ---------------------------------------------------


class TestDetectFeatureName:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="vt_test_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_extracts_from_plan_filename(self):
        Path(os.path.join(self.tmp, "visual-test-plan-speed-change.md")).write_text("plan")
        assert _detect_feature_name(self.tmp) == "speed-change"

    def test_fallback_when_no_plan(self):
        assert _detect_feature_name(self.tmp) == "feature"

    def test_handles_plain_visual_test_plan(self):
        Path(os.path.join(self.tmp, "visual-test-plan.md")).write_text("plan")
        assert _detect_feature_name(self.tmp) == "feature"


# -- Find visual test plan ----------------------------------------------------


class TestFindVisualTestPlan:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="vt_test_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_finds_plan_file(self):
        plan = os.path.join(self.tmp, "visual-test-plan-feature.md")
        Path(plan).write_text("plan content")
        result = _find_visual_test_plan(self.tmp)
        assert result.endswith("visual-test-plan-feature.md")

    def test_returns_empty_when_missing(self):
        assert _find_visual_test_plan(self.tmp) == ""


# -- Context file generation --------------------------------------------------


class TestBuildContextFile:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="vt_ctx_test_")
        self.session_dir = os.path.join(self.tmp, "session")
        os.makedirs(self.session_dir)

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_creates_context_file(self):
        Path(os.path.join(self.session_dir, "visual-test-plan-feat.md")).write_text("# Plan\nTest this")
        state = _make_state(session_dir=self.session_dir)
        preset = _make_preset(self.tmp)
        config = _make_config(self.tmp)

        result = _build_context_file(state, preset, self.tmp, self.session_dir, "feat", config)

        assert os.path.isfile(result)
        content = Path(result).read_text()
        assert "## Visual Test Plan" in content
        assert "Test this" in content
        assert "## Dev Server" in content
        assert "8080" in content

    def test_context_includes_execution_info(self):
        state = _make_state(session_dir=self.session_dir)
        preset = _make_preset(self.tmp)
        config = _make_config(self.tmp)

        result = _build_context_file(state, preset, self.tmp, self.session_dir, "my-feat", config)

        content = Path(result).read_text()
        assert "playwright-test-my-feat.js" in content
        assert "my-feat-test-results.json" in content


# -- Prompt generation --------------------------------------------------------


class TestPromptGeneration:
    def test_script_generation_prompt(self):
        prompt = _script_generation_prompt("/tmp/ctx.md", "/tmp/script.js", "feat", "/tmp/session")
        assert "visual-test-context.md" in prompt or "ctx.md" in prompt
        assert "script.js" in prompt
        assert "Do NOT modify any source code" in prompt

    def test_script_fix_prompt_includes_reason(self):
        prompt = _script_fix_prompt("/tmp/script.js", "Wrong selector for timeline", "/tmp/session")
        assert "Wrong selector for timeline" in prompt
        assert "Fix the" in prompt

    def test_script_fix_prompt_without_reason(self):
        prompt = _script_fix_prompt("/tmp/script.js", "", "/tmp/session")
        assert "Fix the existing" in prompt

    def test_judge_prompt_includes_paths(self):
        prompt = _judge_prompt(
            "/tmp/plan.md", "/tmp/script.js", "/tmp/results.json",
            ["/tmp/s1.png", "/tmp/s2.png"], "/tmp/verdict.json",
        )
        assert "plan.md" in prompt
        assert "script.js" in prompt
        assert "results.json" in prompt
        assert "s1.png" in prompt
        assert "verdict.json" in prompt

    def test_triage_prompt_includes_paths(self):
        prompt = _triage_prompt(
            "/tmp/verdict.json", "/tmp/results.json",
            "/tmp/script.js", "/tmp/triage.json",
        )
        assert "verdict.json" in prompt
        assert "script_issue" in prompt
        assert "code_issue" in prompt


# -- Result parsing -----------------------------------------------------------


class TestParseJudgeVerdict:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="vt_parse_test_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_parses_valid_verdict(self):
        path = Path(self.tmp) / "verdict.json"
        path.write_text(json.dumps({"verdict": "PASS", "scenarios": []}))
        result = _parse_judge_verdict(path)
        assert result["verdict"] == "PASS"

    def test_returns_none_for_missing_file(self):
        assert _parse_judge_verdict(Path(self.tmp) / "nope.json") is None

    def test_returns_none_for_malformed_json(self):
        path = Path(self.tmp) / "bad.json"
        path.write_text("not json")
        assert _parse_judge_verdict(path) is None

    def test_returns_none_without_verdict_key(self):
        path = Path(self.tmp) / "no-verdict.json"
        path.write_text(json.dumps({"summary": "no verdict key"}))
        assert _parse_judge_verdict(path) is None


class TestParseTriage:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="vt_triage_test_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_parses_valid_triage(self):
        path = Path(self.tmp) / "triage.json"
        path.write_text(json.dumps({
            "diagnosis": "script_issue",
            "reason": "Wrong selector",
            "fix_hints": "Use getByTestId",
        }))
        result = _parse_triage(path)
        assert result["diagnosis"] == "script_issue"

    def test_returns_none_for_missing_file(self):
        assert _parse_triage(Path(self.tmp) / "nope.json") is None

    def test_returns_none_without_diagnosis_key(self):
        path = Path(self.tmp) / "no-diag.json"
        path.write_text(json.dumps({"reason": "something"}))
        assert _parse_triage(path) is None


class TestWriteTriage:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="vt_triage_write_test_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_writes_triage_file(self):
        attempt_dir = Path(self.tmp) / "attempt-1"
        attempt_dir.mkdir()
        _write_triage(attempt_dir, "script_issue", "Bad selector", "Use .first()")
        triage_path = attempt_dir / "triage.json"
        assert triage_path.is_file()
        data = json.loads(triage_path.read_text())
        assert data["diagnosis"] == "script_issue"
        assert data["fix_hints"] == "Use .first()"


# -- Dashboard generation ----------------------------------------------------


class TestGenerateDashboard:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="vt_dash_test_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_generates_dashboard_html(self):
        results = {
            "summary": {"total": 2, "passed": 1, "failed": 1},
            "results": [
                {
                    "number": 1, "title": "Scenario 1", "status": "PASS",
                    "assertions": [{"label": "Width", "value": "100px", "passed": True}],
                    "screenshots": {"before": "b1.png", "after": "a1.png"},
                },
                {
                    "number": 2, "title": "Scenario 2", "status": "FAIL",
                    "assertions": [{"label": "Height", "value": "0px", "passed": False}],
                    "screenshots": {},
                },
            ],
        }
        verdict = {
            "verdict": "FAIL",
            "scenarios": [
                {"number": 1, "verdict": "PASS", "notes": "Looks good"},
                {"number": 2, "verdict": "FAIL", "notes": "Height is zero"},
            ],
        }

        path = _generate_dashboard(self.tmp, "my-feature", results, verdict)

        assert os.path.isfile(path)
        html = Path(path).read_text()
        assert "Scenario 1" in html or "S1" in html
        assert "Scenario 2" in html or "S2" in html
        assert "FAIL" in html

    def test_all_pass_badge(self):
        results = {
            "summary": {"total": 1, "passed": 1, "failed": 0},
            "results": [
                {"number": 1, "title": "S1", "status": "PASS", "assertions": [], "screenshots": {}},
            ],
        }
        verdict = {"verdict": "PASS", "scenarios": [{"number": 1, "verdict": "PASS", "notes": "ok"}]}

        path = _generate_dashboard(self.tmp, "feat", results, verdict)
        html = Path(path).read_text()
        assert "ALL PASS" in html


# -- Run function (integration-style with mocks) ------------------------------


class TestRun:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="vt_run_test_")
        self.session_dir = os.path.join(self.tmp, "session")
        os.makedirs(self.session_dir)
        self.config = _make_config(self.tmp)

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    @patch("architect.executor.engine.agents.visual_test_agent._report_fail")
    @patch("architect.executor.engine.agents.visual_test_agent.AgentRunner")
    def test_fails_when_no_script_produced(self, MockRunner, mock_fail):
        """If the script generation agent doesn't produce a script, run() fails."""
        mock_instance = MockRunner.return_value
        mock_instance.run.return_value = AgentResult(
            exit_code=0, stdout="", transcript_path="", timed_out=False,
        )

        state = _make_state(session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        outcome = run(
            state=state, preset=preset, cwd=self.tmp,
            session_dir=self.session_dir,
            activity_log_path=os.path.join(self.session_dir, "activity.log"),
            config=self.config,
        )

        assert not outcome.passed
        assert "Script generation failed" in outcome.reason

    @patch("architect.executor.engine.agents.visual_test_agent._report_pass")
    @patch("architect.executor.engine.agents.visual_test_agent._execute_script")
    @patch("architect.executor.engine.agents.visual_test_agent.AgentRunner")
    def test_passes_when_judge_says_pass(self, MockRunner, mock_exec, mock_pass):
        """Full pass path: script generated -> executed -> judge says PASS."""
        mock_instance = MockRunner.return_value
        feature_name = "feat"

        Path(os.path.join(self.session_dir, f"visual-test-plan-{feature_name}.md")).write_text("plan")

        def write_script(*args, **kwargs):
            script_path = Path(self.session_dir) / f"playwright-test-{feature_name}.js"
            script_path.write_text("// test script")

            results = {
                "schemaVersion": 1, "feature": feature_name,
                "summary": {"total": 1, "passed": 1, "failed": 0},
                "results": [{"number": 1, "title": "S1", "status": "PASS", "assertions": [], "screenshots": {}}],
            }
            Path(self.session_dir, f"{feature_name}-test-results.json").write_text(json.dumps(results))
            return AgentResult(exit_code=0, stdout="", transcript_path="", timed_out=False)

        call_count = [0]

        def mock_runner_run(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return write_script()
            elif call_count[0] == 2:
                vt_dir = Path(self.session_dir) / "visual-test" / "attempt-1"
                vt_dir.mkdir(parents=True, exist_ok=True)
                verdict = {"verdict": "PASS", "scenarios": [{"number": 1, "verdict": "PASS", "notes": "ok"}]}
                (vt_dir / "judge-verdict.json").write_text(json.dumps(verdict))
                return AgentResult(exit_code=0, stdout="", transcript_path="", timed_out=False)
            return AgentResult(exit_code=0, stdout="", transcript_path="", timed_out=False)

        mock_instance.run.side_effect = mock_runner_run
        mock_exec.return_value = (True, "ok")

        state = _make_state(session_dir=self.session_dir)
        preset = _make_preset(self.tmp)

        outcome = run(
            state=state, preset=preset, cwd=self.tmp,
            session_dir=self.session_dir,
            activity_log_path=os.path.join(self.session_dir, "activity.log"),
            config=self.config,
        )

        assert outcome.passed
        assert "passed" in outcome.reason.lower()
        assert outcome.dashboard_path
        mock_pass.assert_called_once()


# -- Execute script -----------------------------------------------------------


class TestExecuteScript:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="vt_exec_test_")
        self.config = _make_config(self.tmp)
        # Create a dummy run.js so the runner_script.is_file() check passes
        runner_dir = self.config.playwright_runner_dir
        Path(runner_dir, "run.js").write_text("// runner")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    @patch("architect.executor.engine.agents.visual_test_agent.subprocess.run")
    def test_returns_success_on_zero_exit(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        ok, output = _execute_script("/tmp/test.js", "/tmp/session", "/repo", self.config)
        assert ok is True
        assert "ok" in output

    @patch("architect.executor.engine.agents.visual_test_agent.subprocess.run")
    def test_returns_failure_on_nonzero_exit(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        ok, output = _execute_script("/tmp/test.js", "/tmp/session", "/repo", self.config)
        assert ok is False

    @patch("architect.executor.engine.agents.visual_test_agent.subprocess.run")
    def test_handles_timeout(self, mock_run):
        import subprocess as sp
        mock_run.side_effect = sp.TimeoutExpired(cmd="node", timeout=300)
        ok, output = _execute_script("/tmp/test.js", "/tmp/session", "/repo", self.config)
        assert ok is False
        assert "timed out" in output.lower()
