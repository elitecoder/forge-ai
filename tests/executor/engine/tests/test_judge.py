"""Tests for engine/judge.py — criteria loading, prompt building, verdict parsing, feedback saving."""

import json
import os
import subprocess
import tempfile
import shutil
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from forge.executor.engine.judge import (
    load_criteria, build_judge_prompt, spawn_judge, save_judge_feedback,
    _parse_judge_output, _load_plan_criteria, _load_findings_criteria,
    _load_visual_test_criteria, JudgeVerdict,
)
from forge.executor.engine.registry import JudgeConfig
from forge.executor.engine.state import PipelineState, StepState


def _make_state(plan_file="", session_dir=""):
    return PipelineState(
        steps={"code": StepState()},
        step_order=["code"],
        dependency_graph={},
        session_dir=session_dir,
        pipeline="full",
        preset="test-preset",
        plan_file=plan_file,
    )


class TestLoadCriteria:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="judge_test_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_plan_criteria_from_file(self):
        plan = os.path.join(self.tmp, "plan.md")
        Path(plan).write_text("# Plan\n1. Add authentication module\n2. Create login endpoint\n3. Write unit tests\n")
        state = _make_state(plan_file=plan, session_dir=self.tmp)
        config = JudgeConfig(criteria_source="plan")

        criteria = load_criteria(self.tmp, "code", config, state)

        assert len(criteria) == 3
        assert criteria[0]["id"] == "plan-1"
        assert "authentication module" in criteria[0]["criteria"]

    def test_plan_criteria_fallback_no_file(self):
        state = _make_state(plan_file="/nonexistent", session_dir=self.tmp)
        config = JudgeConfig(criteria_source="plan")

        criteria = load_criteria(self.tmp, "code", config, state)

        assert len(criteria) == 1
        assert criteria[0]["id"] == "plan-1"

    def test_findings_criteria_from_checklist(self):
        checklist = {
            "step": "code_review",
            "checklist": [
                {"id": "finding-1", "criteria": "Replace any types", "status": "done",
                 "evidence": "line 48", "files_touched": ["foo.ts"]},
                {"id": "finding-2", "criteria": "Add bounds check", "status": "skipped",
                 "evidence": "", "files_touched": []},
            ]
        }
        Path(os.path.join(self.tmp, "code_review-checklist.json")).write_text(json.dumps(checklist))
        state = _make_state(session_dir=self.tmp)
        config = JudgeConfig(criteria_source="findings")

        criteria = load_criteria(self.tmp, "code_review", config, state)

        assert len(criteria) == 1  # skipped item excluded
        assert criteria[0]["criteria"] == "Replace any types"

    def test_findings_fallback_to_output_md(self):
        output = "# Output\n\n## code_review\n\n- Replace any constructor params\n- Add playhead bounds\n\n## test\n"
        Path(os.path.join(self.tmp, "pipeline-output.md")).write_text(output)
        state = _make_state(session_dir=self.tmp)
        config = JudgeConfig(criteria_source="findings")

        criteria = load_criteria(self.tmp, "code_review", config, state)

        assert len(criteria) == 2

    def test_visual_test_criteria(self):
        Path(os.path.join(self.tmp, "visual-test-plan.md")).write_text(
            "# VTP\n1. Verify timeline renders\n2. Check color picker\n3. Confirm export works\n"
        )
        state = _make_state(session_dir=self.tmp)
        config = JudgeConfig(criteria_source="visual_test_plan")

        criteria = load_criteria(self.tmp, "visual_test", config, state)

        assert len(criteria) == 3

    def test_visual_test_fallback_no_file(self):
        state = _make_state(session_dir=self.tmp)
        config = JudgeConfig(criteria_source="visual_test_plan")

        criteria = load_criteria(self.tmp, "visual_test", config, state)

        assert len(criteria) == 1

    def test_unknown_source_returns_default(self):
        state = _make_state(session_dir=self.tmp)
        config = JudgeConfig(criteria_source="unknown")

        criteria = load_criteria(self.tmp, "step_x", config, state)

        assert len(criteria) == 1
        assert "step_x" in criteria[0]["criteria"]


class TestLoadCriteriaChangedFiles:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="judge_cf_test_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    @patch("forge.executor.engine.judge.subprocess.run")
    def test_changed_files_criteria(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="src/app.ts\nsrc/utils.ts\n", stderr="",
        )
        state = _make_state(session_dir=self.tmp)
        config = JudgeConfig(criteria_source="changed_files")

        criteria = load_criteria(self.tmp, "code", config, state)

        assert len(criteria) == 2
        assert criteria[0]["id"] == "file-1"
        assert "src/app.ts" in criteria[0]["criteria"]
        assert criteria[1]["id"] == "file-2"

    @patch("forge.executor.engine.judge.subprocess.run")
    def test_changed_files_git_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        state = _make_state(session_dir=self.tmp)
        config = JudgeConfig(criteria_source="changed_files")

        criteria = load_criteria(self.tmp, "code", config, state)

        assert len(criteria) == 1
        assert criteria[0]["id"] == "file-1"
        assert "Review all changed files" in criteria[0]["criteria"]


class TestLoadFindingsCriteriaEdgeCases:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="judge_findings_test_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_findings_malformed_json(self):
        Path(os.path.join(self.tmp, "code_review-checklist.json")).write_text("not valid json{{{")
        state = _make_state(session_dir=self.tmp)
        config = JudgeConfig(criteria_source="findings")

        criteria = load_criteria(self.tmp, "code_review", config, state)

        assert len(criteria) == 1
        assert criteria[0]["id"] == "finding-1"
        assert "Fix all code review findings" in criteria[0]["criteria"]

    def test_findings_missing_key_in_checklist(self):
        bad_data = {"checklist": [{"no_id_key": "oops", "criteria": "test"}]}
        Path(os.path.join(self.tmp, "code_review-checklist.json")).write_text(json.dumps(bad_data))
        state = _make_state(session_dir=self.tmp)
        config = JudgeConfig(criteria_source="findings")

        criteria = load_criteria(self.tmp, "code_review", config, state)

        assert len(criteria) == 1
        assert criteria[0]["id"] == "finding-1"

    def test_findings_no_files_at_all(self):
        state = _make_state(session_dir=self.tmp)
        config = JudgeConfig(criteria_source="findings")

        criteria = load_criteria(self.tmp, "code_review", config, state)

        assert len(criteria) == 1
        assert criteria[0]["id"] == "finding-1"
        assert "Fix all code review findings" in criteria[0]["criteria"]


class TestBuildJudgePrompt:
    def test_includes_criteria_and_checklist(self):
        criteria = [{"id": "plan-1", "criteria": "Add auth"}, {"id": "plan-2", "criteria": "Write tests"}]
        checklist = {"step": "code", "checklist": [{"id": "plan-1", "status": "done"}]}
        diff = "diff --git a/foo.ts\n+const auth = true;\n"

        prompt = build_judge_prompt("code", criteria, checklist, diff)

        assert "[plan-1]" in prompt
        assert "[plan-2]" in prompt
        assert "Add auth" in prompt
        assert "code" in prompt
        assert "+const auth = true;" in prompt

    def test_truncates_large_diff(self):
        criteria = [{"id": "x", "criteria": "test"}]
        large_diff = "x" * 50000

        prompt = build_judge_prompt("code", criteria, {}, large_diff)

        assert "diff truncated" in prompt
        assert len(prompt) < 100000


class TestParseJudgeOutput:
    def test_parses_clean_json(self):
        output = '{"items": [{"id": "plan-1", "verdict": "pass", "reason": "OK"}]}'
        criteria = [{"id": "plan-1", "criteria": "test"}]

        verdict = _parse_judge_output(output, criteria)

        assert verdict.passed is True
        assert len(verdict.items) == 1

    def test_parses_json_with_fences(self):
        output = "Here's my verdict:\n```json\n{\"items\": [{\"id\": \"x\", \"verdict\": \"pass\", \"reason\": \"OK\"}]}\n```"
        criteria = [{"id": "x", "criteria": "test"}]

        verdict = _parse_judge_output(output, criteria)

        assert verdict.passed is True

    def test_fails_on_mixed_verdicts(self):
        output = json.dumps({"items": [
            {"id": "a", "verdict": "pass", "reason": "OK"},
            {"id": "b", "verdict": "fail", "reason": "Not done"},
        ]})
        criteria = [{"id": "a"}, {"id": "b"}]

        verdict = _parse_judge_output(output, criteria)

        assert verdict.passed is False

    def test_fails_on_empty_items(self):
        output = '{"items": []}'
        criteria = [{"id": "a", "criteria": "test"}]

        verdict = _parse_judge_output(output, criteria)

        assert verdict.passed is False

    def test_fails_on_unparseable(self):
        output = "This is not JSON at all"
        criteria = [{"id": "a", "criteria": "test"}]

        verdict = _parse_judge_output(output, criteria)

        assert verdict.passed is False
        assert verdict.items[0]["verdict"] == "fail"


class TestSpawnJudge:
    @patch("forge.executor.engine.judge.subprocess.run")
    def test_spawn_parses_output(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout='{"items": [{"id": "a", "verdict": "pass", "reason": "OK"}]}',
            stderr="",
        )

        criteria = [{"id": "a", "criteria": "test"}]
        verdict = spawn_judge("code", "/session", criteria, {}, "diff", "opus")

        assert verdict.passed is True
        mock_run.assert_called_once()

    @patch("forge.executor.engine.judge.subprocess.run")
    def test_timeout_returns_fail(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=[], timeout=120)

        criteria = [{"id": "a", "criteria": "test"}]
        verdict = spawn_judge("code", "/session", criteria, {}, "diff")

        assert verdict.passed is False
        assert "timed out" in verdict.summary.lower()

    @patch("forge.executor.engine.judge.subprocess.run")
    def test_missing_cli_returns_fail(self, mock_run):
        mock_run.side_effect = FileNotFoundError()

        criteria = [{"id": "a", "criteria": "test"}]
        verdict = spawn_judge("code", "/session", criteria, {}, "diff")

        assert verdict.passed is False


class TestSpawnJudgeTranscript:
    """Tests that spawn_judge writes transcript and activity log entries."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="judge_transcript_test_")
        self.activity_log = os.path.join(self.tmp, "pipeline-activity.log")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    @patch("forge.executor.engine.judge.subprocess.run")
    def test_writes_transcript_on_success(self, mock_run):
        stdout = '{"items": [{"id": "a", "verdict": "pass", "reason": "OK"}]}'
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=stdout, stderr="",
        )

        criteria = [{"id": "a", "criteria": "test"}]
        spawn_judge("code", self.tmp, criteria, {}, "diff", "opus",
                    activity_log_path=self.activity_log)

        transcript_path = os.path.join(self.tmp, "code_judge-transcript.log")
        assert Path(transcript_path).is_file()
        assert stdout in Path(transcript_path).read_text()

    @patch("forge.executor.engine.judge.subprocess.run")
    def test_writes_transcript_on_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=[], timeout=120)

        criteria = [{"id": "a", "criteria": "test"}]
        spawn_judge("code", self.tmp, criteria, {}, "diff", "opus",
                    activity_log_path=self.activity_log)

        transcript_path = os.path.join(self.tmp, "code_judge-transcript.log")
        assert Path(transcript_path).is_file()
        assert "timed out" in Path(transcript_path).read_text().lower()

    @patch("forge.executor.engine.judge.subprocess.run")
    def test_writes_activity_log(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout='{"items": [{"id": "a", "verdict": "pass", "reason": "OK"}]}',
            stderr="",
        )

        criteria = [{"id": "a", "criteria": "test"}]
        spawn_judge("code", self.tmp, criteria, {}, "diff", "opus",
                    activity_log_path=self.activity_log)

        assert Path(self.activity_log).is_file()
        log_text = Path(self.activity_log).read_text()
        assert "judge (opus)" in log_text

    @patch("forge.executor.engine.judge.subprocess.run")
    def test_no_activity_log_when_path_empty(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout='{"items": [{"id": "a", "verdict": "pass", "reason": "OK"}]}',
            stderr="",
        )

        criteria = [{"id": "a", "criteria": "test"}]
        spawn_judge("code", self.tmp, criteria, {}, "diff", "opus",
                    activity_log_path="")

        assert not Path(self.activity_log).exists()


class TestSaveJudgeFeedback:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="judge_save_test_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_saves_feedback_file(self):
        verdict = JudgeVerdict(
            passed=False,
            items=[{"id": "a", "verdict": "fail", "reason": "Not done"}],
            summary="0/1 passed",
        )

        path = save_judge_feedback(self.tmp, "code", 0, verdict)

        assert os.path.isfile(path)
        data = json.loads(Path(path).read_text())
        assert data["step"] == "code"
        assert data["attempt"] == 0
        assert data["passed"] is False
        assert len(data["items"]) == 1

    def test_creates_judge_directory(self):
        verdict = JudgeVerdict(passed=True, items=[], summary="OK")

        save_judge_feedback(self.tmp, "test", 2, verdict)

        judge_dir = os.path.join(self.tmp, "_judge")
        assert os.path.isdir(judge_dir)

    def test_multiple_attempts(self):
        verdict = JudgeVerdict(passed=False, items=[], summary="fail")

        save_judge_feedback(self.tmp, "code", 0, verdict)
        save_judge_feedback(self.tmp, "code", 1, verdict)

        judge_dir = os.path.join(self.tmp, "_judge")
        assert os.path.isfile(os.path.join(judge_dir, "code_attempt_0.json"))
        assert os.path.isfile(os.path.join(judge_dir, "code_attempt_1.json"))
