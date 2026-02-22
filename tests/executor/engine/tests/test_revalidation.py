"""Tests for auto-revalidation, artifact archival, and summary."""

import json
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from architect.executor.engine.state import PipelineState, StepState, StepStatus
from architect.executor.engine.utils import transitive_dependents


def _make_state(step_order, dependency_graph=None, statuses=None, retries=None,
                session_dir="", pipeline="full", preset="hz-web"):
    statuses = statuses or {}
    retries = retries or {}
    steps = {}
    for name in step_order:
        ss = StepState()
        if name in statuses:
            ss.status = statuses[name]
        if name in retries:
            ss.retries = retries[name]
        steps[name] = ss
    return PipelineState(
        steps=steps,
        step_order=step_order,
        dependency_graph=dependency_graph or {},
        session_dir=session_dir,
        pipeline=pipeline,
        preset=preset,
    )


# ── transitive_dependents ──────────────────────────────────────────────────

class TestTransitiveDependents:
    def test_leaf_has_no_dependents(self):
        graph = {"b": ["a"], "c": ["b"]}
        assert transitive_dependents("c", graph) == []

    def test_root_has_all_dependents(self):
        graph = {"b": ["a"], "c": ["b"]}
        assert transitive_dependents("a", graph) == ["b", "c"]

    def test_diamond_dependents(self):
        graph = {
            "left": ["root"],
            "right": ["root"],
            "sink": ["left", "right"],
        }
        result = transitive_dependents("root", graph)
        assert result == ["left", "right", "sink"]

    def test_middle_node(self):
        graph = {"b": ["a"], "c": ["b"], "d": ["c"]}
        assert transitive_dependents("b", graph) == ["c", "d"]

    def test_unknown_step_returns_empty(self):
        assert transitive_dependents("ghost", {"b": ["a"]}) == []

    def test_empty_graph(self):
        assert transitive_dependents("a", {}) == []

    def test_real_pipeline_graph(self):
        """Test with the actual hz-web full pipeline dependency graph."""
        graph = {
            "build_gen": ["code"],
            "lint": ["build_gen"],
            "test": ["build_gen"],
            "code_review": ["lint", "test"],
            "visual_test": ["lint", "test"],
            "report": ["code_review", "visual_test"],
            "create_pr": ["report"],
        }
        # lint dependents: code_review, visual_test, report, create_pr
        result = transitive_dependents("lint", graph)
        assert "code_review" in result
        assert "visual_test" in result
        assert "report" in result
        assert "create_pr" in result
        assert "test" not in result
        assert "build_gen" not in result


# ── _archive_step_artifacts ────────────────────────────────────────────────

class TestArchiveStepArtifacts:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="archive_test_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_archives_matching_files(self):
        from architect.executor.engine.registry import StepDefinition, EvidenceRule
        from architect.executor.commands import _archive_step_artifacts

        # Create fake evidence artifacts
        Path(os.path.join(self.tmp, "results.json")).write_text('{"ok": true}')
        Path(os.path.join(self.tmp, "dashboard.html")).write_text("<html/>")

        step_def = StepDefinition(
            name="visual_test",
            evidence=[
                EvidenceRule(rule="file_exists", file_glob="results.json"),
                EvidenceRule(rule="file_exists", file_glob="dashboard.html"),
            ],
        )

        _archive_step_artifacts(step_def, self.tmp, attempt=1)

        archive_dir = os.path.join(self.tmp, "_retries", "visual_test_attempt_1")
        assert os.path.isdir(archive_dir)
        assert os.path.isfile(os.path.join(archive_dir, "results.json"))
        assert os.path.isfile(os.path.join(archive_dir, "dashboard.html"))
        # Originals moved
        assert not os.path.isfile(os.path.join(self.tmp, "results.json"))
        assert not os.path.isfile(os.path.join(self.tmp, "dashboard.html"))

    def test_no_artifacts_no_archive_dir(self):
        from architect.executor.engine.registry import StepDefinition, EvidenceRule
        from architect.executor.commands import _archive_step_artifacts

        step_def = StepDefinition(
            name="lint",
            evidence=[
                EvidenceRule(rule="file_exists", file_glob="nonexistent-*.txt"),
            ],
        )

        _archive_step_artifacts(step_def, self.tmp, attempt=2)

        archive_dir = os.path.join(self.tmp, "_retries", "lint_attempt_2")
        assert not os.path.exists(archive_dir)

    def test_no_evidence_rules_noop(self):
        from architect.executor.engine.registry import StepDefinition
        from architect.executor.commands import _archive_step_artifacts

        step_def = StepDefinition(name="build")
        _archive_step_artifacts(step_def, self.tmp, attempt=1)
        # No _retries dir created
        assert not os.path.exists(os.path.join(self.tmp, "_retries"))


# ── _archive_output_section ────────────────────────────────────────────────

class TestArchiveOutputSection:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="output_test_")
        self.archive_dir = os.path.join(self.tmp, "_retries", "step_attempt_1")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_extracts_step_section(self):
        from architect.executor.commands import _archive_output_section

        output = "# Pipeline\n\n## lint\n\n- Result: passed\n- Duration: 5s\n\n## test\n\n- Result: passed\n"
        Path(os.path.join(self.tmp, "pipeline-output.md")).write_text(output)

        _archive_output_section(self.tmp, "lint", self.archive_dir)

        # Section extracted to archive
        archived = Path(os.path.join(self.archive_dir, "output-section.md")).read_text()
        assert "## lint" in archived
        assert "Result: passed" in archived

        # Section removed from original
        remaining = Path(os.path.join(self.tmp, "pipeline-output.md")).read_text()
        assert "## lint" not in remaining
        assert "## test" in remaining

    def test_no_output_file_noop(self):
        from architect.executor.commands import _archive_output_section

        # No crash when file doesn't exist
        _archive_output_section(self.tmp, "lint", self.archive_dir)


# ── cmd_pass auto-revalidation ─────────────────────────────────────────────

class TestCmdPassAutoRevalidation:
    """Test that cmd_pass triggers revalidation when retries > 0."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="revalidation_test_")
        self.checkpoint_dir = os.path.join(self.tmp, "checkpoints")
        os.makedirs(self.checkpoint_dir)

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def _make_preset_with_revalidation(self):
        from architect.executor.engine.registry import Preset, StepDefinition, PipelineDefinition
        return Preset(
            name="test", version=3, description="",
            pipelines={
                "full": PipelineDefinition(
                    steps=["code", "build_gen", "lint", "test", "code_review", "report"],
                    dependencies={
                        "build_gen": ["code"],
                        "lint": ["build_gen"],
                        "test": ["build_gen"],
                        "code_review": ["lint", "test"],
                        "report": ["code_review"],
                    },
                    revalidation_targets=["lint", "test"],
                ),
            },
            steps={
                "code": StepDefinition(name="code", step_type="inline"),
                "build_gen": StepDefinition(name="build_gen", step_type="command"),
                "lint": StepDefinition(name="lint", step_type="command"),
                "test": StepDefinition(name="test", step_type="command"),
                "code_review": StepDefinition(name="code_review", step_type="ai"),
                "report": StepDefinition(name="report", step_type="inline"),
            },
            models={"fix": "sonnet"},
        )

    def test_no_revalidation_when_retries_zero(self):
        """First pass (no retries) should not trigger revalidation."""
        state = _make_state(
            ["code", "build_gen", "lint", "test", "code_review", "report"],
            dependency_graph={
                "build_gen": ["code"], "lint": ["build_gen"],
                "test": ["build_gen"], "code_review": ["lint", "test"],
                "report": ["code_review"],
            },
            statuses={
                "code": StepStatus.COMPLETE, "build_gen": StepStatus.COMPLETE,
                "lint": StepStatus.COMPLETE, "test": StepStatus.COMPLETE,
                "code_review": StepStatus.IN_PROGRESS,
            },
            retries={"code_review": 0},
            session_dir=self.tmp, pipeline="full",
        )

        from architect.executor.commands import cmd_pass
        args = MagicMock(step="code_review")
        preset = self._make_preset_with_revalidation()

        printed = []
        with patch("architect.executor.commands._require_state", return_value=state):
            with patch("architect.executor.commands._load_preset_for_state", return_value=preset):
                with patch("architect.executor.commands._require_dependencies"):
                    with patch("architect.executor.commands._state_mgr") as mock_mgr:
                        mock_mgr.return_value.update = MagicMock(
                            side_effect=lambda fn: (fn(state), state)[1]
                        )
                        with patch("architect.executor.commands.write_checkpoint", return_value="cp"):
                            with patch("builtins.print", side_effect=lambda *a, **kw: printed.append(str(a[0]))):
                                cmd_pass(args)

        assert not any("Revalidation triggered" in p for p in printed)

    def test_revalidation_triggered_on_fix_cycle(self):
        """Pass after fix cycle (retries > 0) should reset lint + test + COMPLETE dependents."""
        state = _make_state(
            ["code", "build_gen", "lint", "test", "code_review", "report"],
            dependency_graph={
                "build_gen": ["code"], "lint": ["build_gen"],
                "test": ["build_gen"], "code_review": ["lint", "test"],
                "report": ["code_review"],
            },
            statuses={
                "code": StepStatus.COMPLETE, "build_gen": StepStatus.COMPLETE,
                "lint": StepStatus.COMPLETE, "test": StepStatus.COMPLETE,
                "code_review": StepStatus.IN_PROGRESS,
                "report": StepStatus.COMPLETE,  # Must be COMPLETE to be reset
            },
            retries={"code_review": 1},
            session_dir=self.tmp, pipeline="full",
        )

        from architect.executor.commands import cmd_pass
        args = MagicMock(step="code_review")
        preset = self._make_preset_with_revalidation()

        for step in ["lint", "test"]:
            Path(os.path.join(self.checkpoint_dir, f"{step}.passed")).write_text("fake")

        printed = []
        with patch("architect.executor.commands._require_state", return_value=state):
            with patch("architect.executor.commands._load_preset_for_state", return_value=preset):
                with patch("architect.executor.commands._require_dependencies"):
                    with patch("architect.executor.commands._state_mgr") as mock_mgr:
                        mock_mgr.return_value.update = MagicMock(
                            side_effect=lambda fn: (fn(state), state)[1]
                        )
                        with patch("architect.executor.commands.write_checkpoint", return_value="cp"):
                            with patch("builtins.print", side_effect=lambda *a, **kw: printed.append(str(a[0]))):
                                cmd_pass(args)

        revalidation_msg = [p for p in printed if "Revalidation triggered" in p]
        assert len(revalidation_msg) == 1
        msg = revalidation_msg[0]
        assert "lint" in msg
        assert "test" in msg

    def test_revalidation_skips_self(self):
        """If lint passes after retry, revalidation should not reset lint itself.
        Only revalidation_targets that are COMPLETE and != self get reset."""
        state = _make_state(
            ["code", "build_gen", "lint", "test", "code_review"],
            dependency_graph={
                "build_gen": ["code"], "lint": ["build_gen"],
                "test": ["build_gen"], "code_review": ["lint", "test"],
            },
            statuses={
                "code": StepStatus.COMPLETE, "build_gen": StepStatus.COMPLETE,
                "lint": StepStatus.IN_PROGRESS, "test": StepStatus.COMPLETE,
                "code_review": StepStatus.COMPLETE,
            },
            retries={"lint": 1},
            session_dir=self.tmp, pipeline="full",
        )

        from architect.executor.commands import cmd_pass
        args = MagicMock(step="lint")
        preset = self._make_preset_with_revalidation()

        Path(os.path.join(self.checkpoint_dir, "test.passed")).write_text("fake")

        printed = []
        with patch("architect.executor.commands._require_state", return_value=state):
            with patch("architect.executor.commands._load_preset_for_state", return_value=preset):
                with patch("architect.executor.commands._require_dependencies"):
                    with patch("architect.executor.commands._state_mgr") as mock_mgr:
                        mock_mgr.return_value.update = MagicMock(
                            side_effect=lambda fn: (fn(state), state)[1]
                        )
                        with patch("architect.executor.commands.write_checkpoint", return_value="cp"):
                            with patch("builtins.print", side_effect=lambda *a, **kw: printed.append(str(a[0]))):
                                cmd_pass(args)

        revalidation_msg = [p for p in printed if "Revalidation triggered" in p]
        assert len(revalidation_msg) == 1
        # lint itself should NOT be in the reset list
        assert "lint" not in revalidation_msg[0].split("Revalidation triggered: ")[1].split(" reset")[0]
        # test is a revalidation_target and COMPLETE — should be reset
        assert "test" in revalidation_msg[0]

    def test_revalidation_skips_failed_dependents(self):
        """Fix 1: Revalidation must not reset FAILED steps — only COMPLETE ones."""
        state = _make_state(
            ["code", "build_gen", "lint", "test", "code_review", "report"],
            dependency_graph={
                "build_gen": ["code"], "lint": ["build_gen"],
                "test": ["build_gen"], "code_review": ["lint", "test"],
                "report": ["code_review"],
            },
            statuses={
                "code": StepStatus.COMPLETE, "build_gen": StepStatus.COMPLETE,
                "lint": StepStatus.COMPLETE, "test": StepStatus.COMPLETE,
                "code_review": StepStatus.IN_PROGRESS,
                "report": StepStatus.FAILED,  # FAILED — should NOT be reset
            },
            retries={"code_review": 1},
            session_dir=self.tmp, pipeline="full",
        )

        from architect.executor.commands import cmd_pass
        args = MagicMock(step="code_review")
        preset = self._make_preset_with_revalidation()

        for step in ["lint", "test"]:
            Path(os.path.join(self.checkpoint_dir, f"{step}.passed")).write_text("fake")

        printed = []
        with patch("architect.executor.commands._require_state", return_value=state):
            with patch("architect.executor.commands._load_preset_for_state", return_value=preset):
                with patch("architect.executor.commands._require_dependencies"):
                    with patch("architect.executor.commands._state_mgr") as mock_mgr:
                        mock_mgr.return_value.update = MagicMock(
                            side_effect=lambda fn: (fn(state), state)[1]
                        )
                        with patch("architect.executor.commands.write_checkpoint", return_value="cp"):
                            with patch("builtins.print", side_effect=lambda *a, **kw: printed.append(str(a[0]))):
                                cmd_pass(args)

        revalidation_msg = [p for p in printed if "Revalidation triggered" in p]
        assert len(revalidation_msg) == 1
        # report is FAILED — must NOT be in the reset list
        assert "report" not in revalidation_msg[0]
        # lint and test ARE COMPLETE — should be reset
        assert "lint" in revalidation_msg[0]
        assert "test" in revalidation_msg[0]


# ── revalidation_targets parsing ───────────────────────────────────────────

class TestRevalidationTargetsParsing:
    def test_parsed_from_manifest(self, tmp_path):
        from architect.executor.engine.registry import load_preset

        manifest = {
            "preset": "test", "version": 3,
            "pipelines": {
                "full": {
                    "steps": ["a", "b"],
                    "dependencies": {"b": ["a"]},
                    "revalidation_targets": ["a"],
                }
            },
            "steps": {
                "a": {"type": "command", "run_command": "echo a"},
                "b": {"type": "command", "run_command": "echo b"},
            },
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        preset = load_preset(tmp_path)
        assert preset.pipelines["full"].revalidation_targets == ["a"]

    def test_defaults_to_empty(self, tmp_path):
        from architect.executor.engine.registry import load_preset

        manifest = {
            "preset": "test", "version": 3,
            "pipelines": {
                "full": {
                    "steps": ["a", "b"],
                    "dependencies": {"b": ["a"]},
                }
            },
            "steps": {
                "a": {"type": "command", "run_command": "echo a"},
                "b": {"type": "command", "run_command": "echo b"},
            },
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        preset = load_preset(tmp_path)
        assert preset.pipelines["full"].revalidation_targets == []

    def test_legacy_format_no_revalidation(self, tmp_path):
        from architect.executor.engine.registry import load_preset

        manifest = {
            "preset": "test", "version": 2,
            "pipelines": {"full": ["a", "b"]},
            "steps": {
                "a": {"type": "command", "run_command": "echo a"},
                "b": {"type": "command", "run_command": "echo b"},
            },
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        preset = load_preset(tmp_path)
        assert preset.pipelines["full"].revalidation_targets == []


# ── cmd_summary ────────────────────────────────────────────────────────────

class TestCmdSummary:
    def test_text_output(self):
        state = _make_state(
            ["code", "lint", "test", "review"],
            statuses={
                "code": StepStatus.COMPLETE,
                "lint": StepStatus.COMPLETE,
                "test": StepStatus.FAILED,
                "review": StepStatus.PENDING,
            },
            retries={"test": 2},
            pipeline="full", preset="hz-web",
        )
        state.affected_packages = ["apps/webapp"]

        from architect.executor.commands import cmd_summary
        args = MagicMock(json=False)

        printed = []
        with patch("architect.executor.commands._require_state", return_value=state):
            with patch("builtins.print", side_effect=lambda *a, **kw: printed.append(str(a[0]))):
                cmd_summary(args)

        output = "\n".join(printed)
        assert "Pipeline: full" in output
        assert "DONE: code, lint" in output
        assert "FAILED: test(retry 2)" in output
        assert "PENDING: review" in output
        assert "apps/webapp" in output

    def test_json_output(self):
        state = _make_state(
            ["a", "b"],
            statuses={"a": StepStatus.COMPLETE, "b": StepStatus.IN_PROGRESS},
            pipeline="lightweight",
        )
        state.session_dir = "/tmp/test-session"

        from architect.executor.commands import cmd_summary
        args = MagicMock(json=True)

        printed = []
        with patch("architect.executor.commands._require_state", return_value=state):
            with patch("builtins.print", side_effect=lambda *a, **kw: printed.append(str(a[0]))):
                cmd_summary(args)

        data = json.loads(printed[0])
        assert data["pipeline"] == "lightweight"
        assert data["steps"]["a"]["status"] == "complete"
        assert data["steps"]["b"]["status"] == "in_progress"
        assert data["session_dir"] == "/tmp/test-session"
