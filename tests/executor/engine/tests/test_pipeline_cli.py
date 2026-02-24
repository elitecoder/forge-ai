"""Tests for pipeline_cli.py -- _check_dependencies, worktree,
   and comprehensive coverage of all CLI commands."""

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import pytest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, PropertyMock

from forge.executor.engine.state import PipelineState, StepState, StepStatus, StateManager, _state_to_dict
from forge.executor.engine.registry import (
    Preset, StepDefinition, EvidenceRule, PipelineDefinition, load_preset,
)
from forge.executor.engine.evidence import EvidenceResult
from forge.executor.engine.runner import StepResult


def _make_state(step_order, dependency_graph=None, statuses=None):
    statuses = statuses or {}
    steps = {}
    for name in step_order:
        ss = StepState()
        if name in statuses:
            ss.status = statuses[name]
        steps[name] = ss
    return PipelineState(
        steps=steps,
        step_order=step_order,
        dependency_graph=dependency_graph or {},
    )


# Import after path setup
from forge.executor import commands as pipeline_cli
from forge.executor.commands import _check_dependencies


# -- Fixtures ----------------------------------------------------------------

def _make_manifest_dir(tmp_dir, pipelines=None, steps=None, models=None):
    """Create a minimal preset directory with manifest.json in tmp_dir."""
    preset_dir = Path(tmp_dir) / "test-preset"
    preset_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "preset": "test-preset",
        "version": 2,
        "pipelines": pipelines or {
            "full": {
                "steps": ["build", "lint"],
                "dependencies": {"lint": ["build"]},
                "revalidation_targets": ["build"],
            },
            "lightweight": ["lint"],
        },
        "models": models or {"fix": "sonnet", "code_review": "opus"},
        "steps": steps or {
            "build": {
                "type": "command",
                "run_command": "echo build ok",
                "error_file": "build-errors.txt",
                "timeout": 60,
                "evidence": [
                    {"rule": "file_exists", "file_glob": "build-output.txt"},
                ],
            },
            "lint": {
                "type": "command",
                "run_command": "echo lint ok",
                "timeout": 60,
            },
        },
    }
    (preset_dir / "manifest.json").write_text(json.dumps(manifest))
    return preset_dir


class CLITestBase:
    """Base class providing temp directories and pipeline_cli patching."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="cli_test_")
        self.session_dir = os.path.join(self.tmp, "session")
        os.makedirs(self.session_dir, exist_ok=True)
        self.state_file = Path(self.session_dir) / "agent-state.json"
        self.checkpoint_dir = os.path.join(self.session_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.presets_dir = Path(self.tmp) / "presets"
        self.presets_dir.mkdir(exist_ok=True)
        self.preset_dir = _make_manifest_dir(self.presets_dir)

        self.sessions_base = Path(self.tmp) / "sessions"
        self.sessions_base.mkdir(exist_ok=True)

        # Patches -- session resolution returns our temp session dir
        self._patches = [
            patch.object(pipeline_cli, "_active_session_dir", return_value=Path(self.session_dir)),
            patch.object(pipeline_cli, "SESSIONS_BASE", self.sessions_base),
            patch.object(pipeline_cli, "REPO_ROOT", self.tmp),
            patch.object(pipeline_cli, "PRESETS_DIR", self.presets_dir),
            patch("forge.executor.engine.pipeline_ops.presets_dir", return_value=self.presets_dir),
        ]
        for p in self._patches:
            p.start()

    def teardown_method(self):
        for p in self._patches:
            p.stop()
        shutil.rmtree(self.tmp)

    def _save_state(self, state):
        mgr = StateManager(self.state_file)
        mgr.save(state)
        return mgr

    def _make_pipeline_state(self, step_order=None, statuses=None, dep_graph=None,
                              pipeline="full", preset="test-preset",
                              session_dir=None, packages=None,
                              dev_server_port=0):
        step_order = step_order or ["build", "lint"]
        statuses = statuses or {}
        steps = {}
        for name in step_order:
            ss = StepState()
            if name in statuses:
                ss.status = statuses[name]
            steps[name] = ss
        state = PipelineState(
            phase="execution",
            pipeline=pipeline,
            preset=preset,
            preset_path=str(self.preset_dir),
            current_step=step_order[0],
            steps=steps,
            step_order=list(step_order),
            dependency_graph=dep_graph or {},
            affected_packages=packages or [],
            session_dir=session_dir or self.session_dir,
            dev_server_port=dev_server_port,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        )
        self._save_state(state)
        return state


# -- Existing tests ----------------------------------------------------------


class TestCheckDependencies:
    def test_dag_mode_no_deps(self):
        state = _make_state(
            ["a", "b", "c"],
            dependency_graph={"b": ["a"], "c": ["b"]},
        )
        assert _check_dependencies(state, "a") == []

    def test_dag_mode_deps_complete(self):
        state = _make_state(
            ["a", "b"],
            dependency_graph={"b": ["a"]},
            statuses={"a": StepStatus.COMPLETE},
        )
        assert _check_dependencies(state, "b") == []

    def test_dag_mode_deps_incomplete(self):
        state = _make_state(
            ["a", "b"],
            dependency_graph={"b": ["a"]},
            statuses={"a": StepStatus.PENDING},
        )
        result = _check_dependencies(state, "b")
        assert len(result) == 1
        assert "a" in result[0]

    def test_dag_mode_only_checks_declared_deps(self):
        """In DAG mode, b should not block c even though b comes before c in order."""
        state = _make_state(
            ["a", "b", "c"],
            dependency_graph={"c": ["a"]},  # c depends only on a, not b
            statuses={"a": StepStatus.COMPLETE, "b": StepStatus.PENDING},
        )
        assert _check_dependencies(state, "c") == []

    def test_legacy_mode_linear(self):
        state = _make_state(
            ["a", "b", "c"],
            statuses={"a": StepStatus.COMPLETE, "b": StepStatus.PENDING},
        )
        result = _check_dependencies(state, "c")
        assert len(result) == 1
        assert "b" in result[0]

    def test_legacy_mode_all_complete(self):
        state = _make_state(
            ["a", "b", "c"],
            statuses={"a": StepStatus.COMPLETE, "b": StepStatus.COMPLETE},
        )
        assert _check_dependencies(state, "c") == []


# -- Worktree enforcement (SKIPPED - subprocess tests for standalone script) --

# TestWorktreeEnforcement is removed because it invokes the old pipeline_cli.py
# as a standalone script via subprocess, which does not work in the new package
# structure.


# ====================================================================
# NEW TESTS -- coverage for uncovered lines
# ====================================================================


# -- 1. _now_iso (line 32) ---------------------------------------------------

class TestNowIso:
    def test_returns_iso_format(self):
        result = pipeline_cli._now_iso()
        assert result.endswith("Z")
        # Should be parseable
        datetime.strptime(result, "%Y-%m-%dT%H:%M:%SZ")

    def test_returns_utc(self):
        before = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
        result = pipeline_cli._now_iso()
        assert result.startswith(before[:13])  # at least same hour


# -- 2. _resolve_timeout / _resolve_max_turns (lines 49-56) ------------------

class TestResolveLimits:
    def test_resolve_timeout_known_step(self):
        assert pipeline_cli._resolve_timeout("code", "sonnet") == 3600000
        assert pipeline_cli._resolve_timeout("code", "opus") == 3600000

    def test_resolve_timeout_fix_mode(self):
        assert pipeline_cli._resolve_timeout("code", "sonnet", is_fix=True) == 3600000

    def test_resolve_timeout_unknown_step_uses_fix_defaults(self):
        result = pipeline_cli._resolve_timeout("unknown_step", "sonnet")
        assert result == 3600000  # falls back to fix defaults

    def test_resolve_timeout_unknown_model_returns_default(self):
        result = pipeline_cli._resolve_timeout("code", "unknown_model")
        assert result == 3600000  # default when model not in timeout dict

    def test_resolve_max_turns_known_step(self):
        assert pipeline_cli._resolve_max_turns("code") == 0
        assert pipeline_cli._resolve_max_turns("code_review") == 0

    def test_resolve_max_turns_fix_mode(self):
        assert pipeline_cli._resolve_max_turns("code", is_fix=True) == 0

    def test_resolve_max_turns_unknown_step(self):
        assert pipeline_cli._resolve_max_turns("unknown") == 0  # fix default


# -- 3. _last_activity_age (lines 64-70) -------------------------------------

class TestLastActivityAge:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="activity_test_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_existing_log(self):
        log_path = os.path.join(self.tmp, "pipeline-activity.log")
        Path(log_path).write_text("some activity")
        result = pipeline_cli._last_activity_age(self.tmp)
        assert result is not None
        assert result >= 0

    def test_missing_log(self):
        result = pipeline_cli._last_activity_age(self.tmp)
        assert result is None

    def test_oserror_returns_none(self):
        log_path = os.path.join(self.tmp, "pipeline-activity.log")
        Path(log_path).write_text("some activity")
        with patch("os.path.getmtime", side_effect=OSError("permission denied")):
            result = pipeline_cli._last_activity_age(self.tmp)
            assert result is None


# -- 4. _state_mgr / _require_state (lines 79, 83-87) ------------------------

class TestStateMgr(CLITestBase):
    def test_state_mgr_returns_manager(self):
        mgr = pipeline_cli._state_mgr()
        assert isinstance(mgr, StateManager)

    def test_require_state_exits_when_no_state(self):
        with pytest.raises(SystemExit) as exc_info:
            pipeline_cli._require_state()
        assert exc_info.value.code == 1

    def test_require_state_loads_when_state_exists(self):
        self._make_pipeline_state()
        state = pipeline_cli._require_state()
        assert state.pipeline == "full"


# -- 5. _load_preset_for_state (line 91) -------------------------------------

class TestLoadPresetForState(CLITestBase):
    def test_loads_preset(self):
        state = self._make_pipeline_state(preset="test-preset")
        preset = pipeline_cli._load_preset_for_state(state)
        assert preset.name == "test-preset"


# -- 6. _is_worktree (lines 95-106) ------------------------------------------

class TestIsWorktree:
    def test_worktree_detected(self):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(stdout="/repo/.git\n"),   # git-common-dir
                MagicMock(stdout="/repo/wt/.git\n"), # git-dir (different)
            ]
            is_wt, path = pipeline_cli._is_worktree()
            assert is_wt is True
            assert path != ""

    def test_not_worktree(self):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(stdout="/repo/.git\n"),  # git-common-dir
                MagicMock(stdout="/repo/.git\n"),  # git-dir (same)
            ]
            is_wt, path = pipeline_cli._is_worktree()
            assert is_wt is False
            assert path == ""

    def test_subprocess_error(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            is_wt, path = pipeline_cli._is_worktree()
            assert is_wt is False
            assert path == ""


# -- 7. _session_name (lines 110-119) ----------------------------------------

class TestSessionName:
    def test_branch_with_jira_ticket(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="feature/PROJ-123-some-feature\n")
            name = pipeline_cli._session_name()
            assert "PROJ-123" in name
            assert re.search(r"\d{4}-\d{2}-\d{2}_\d{6}Z", name)

    def test_plain_branch(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="main\n")
            name = pipeline_cli._session_name()
            assert "main" in name

    def test_branch_with_slash(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="feature/my-branch\n")
            name = pipeline_cli._session_name()
            assert "feature-my-branch" in name
            assert "/" not in name

    def test_error_fallback(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            name = pipeline_cli._session_name()
            assert "unknown" in name


# -- 8. _check_dependencies -- DAG + legacy incomplete deps (lines 148-152)

class TestRequireDependencies(CLITestBase):
    def test_require_dependencies_exits_on_incomplete(self):
        state = self._make_pipeline_state(
            dep_graph={"lint": ["build"]},
        )
        with pytest.raises(SystemExit) as exc_info:
            pipeline_cli._require_dependencies(state, "lint")
        assert exc_info.value.code == 1

    def test_require_dependencies_passes_when_complete(self):
        state = self._make_pipeline_state(
            statuses={"build": StepStatus.COMPLETE},
            dep_graph={"lint": ["build"]},
        )
        pipeline_cli._require_dependencies(state, "lint")  # should not raise


# -- 9. _archive_step_artifacts (line 184) ------------------------------------

class TestArchiveStepArtifacts:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="archive_test_")
        self.session_dir = self.tmp

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_archive_moves_matching_files(self):
        # Create an artifact matching the evidence glob
        artifact = os.path.join(self.session_dir, "build-output.txt")
        Path(artifact).write_text("build output content")

        step_def = StepDefinition(
            name="build",
            step_type="command",
            evidence=[EvidenceRule(rule="file_exists", file_glob="build-output.txt")],
        )
        pipeline_cli._archive_step_artifacts(step_def, self.session_dir, attempt=1)

        # Original should be moved
        assert not os.path.isfile(artifact)
        archived = os.path.join(
            self.session_dir, "_retries", "build_attempt_1", "build-output.txt"
        )
        assert os.path.isfile(archived)

    def test_archive_no_matching_files(self, capsys):
        step_def = StepDefinition(
            name="build",
            step_type="command",
            evidence=[EvidenceRule(rule="file_exists", file_glob="nonexistent-*.txt")],
        )
        pipeline_cli._archive_step_artifacts(step_def, self.session_dir, attempt=1)
        out = capsys.readouterr().out
        # Should not print "Archived" since nothing was moved
        assert "Archived" not in out

    def test_archive_output_section(self):
        # Create pipeline-output.md with a build section
        output_md = os.path.join(self.session_dir, "pipeline-output.md")
        content = "\n## build\n\nBuild passed.\n\n## lint\n\nLint passed.\n"
        Path(output_md).write_text(content)
        artifact = os.path.join(self.session_dir, "build-output.txt")
        Path(artifact).write_text("ok")

        step_def = StepDefinition(
            name="build",
            step_type="command",
            evidence=[EvidenceRule(rule="file_exists", file_glob="build-output.txt")],
        )
        pipeline_cli._archive_step_artifacts(step_def, self.session_dir, attempt=1)

        # The build section should have been removed from pipeline-output.md
        remaining = Path(output_md).read_text()
        assert "## build" not in remaining
        assert "## lint" in remaining


# -- 10. cmd_init (lines 217-266) --------------------------------------------

class TestCmdInit(CLITestBase):
    def _make_init_args(self, pipeline="full", preset="test-preset",
                         plan="", no_worktree=False, packages=None):
        args = argparse.Namespace(
            pipeline=pipeline,
            preset=preset,
            plan=plan,
            no_worktree=no_worktree,
            packages=packages or [],
        )
        return args

    def test_unknown_pipeline_exits(self):
        args = self._make_init_args(pipeline="nonexistent")
        with pytest.raises(SystemExit) as exc_info:
            pipeline_cli.cmd_init(args)
        assert exc_info.value.code == 1

    def test_not_in_worktree_without_flag_exits(self):
        args = self._make_init_args()
        with patch.object(pipeline_cli, "_is_worktree", return_value=(False, "")):
            with pytest.raises(SystemExit) as exc_info:
                pipeline_cli.cmd_init(args)
            assert exc_info.value.code == 1

    def test_not_in_worktree_with_flag_warns(self, capsys):
        args = self._make_init_args(no_worktree=True)
        with patch.object(pipeline_cli, "_is_worktree", return_value=(False, "")):
            with patch.object(pipeline_cli, "_session_name", return_value="test_2026-01-01"):
                pipeline_cli.cmd_init(args)
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "Pipeline initialized" in out

    def test_successful_init(self, capsys):
        args = self._make_init_args(packages=["pkg-a"])
        with patch.object(pipeline_cli, "_is_worktree", return_value=(True, "/some/path")):
            with patch.object(pipeline_cli, "_session_name", return_value="test_2026-01-01"):
                pipeline_cli.cmd_init(args)
        out = capsys.readouterr().out
        assert "Pipeline initialized: full" in out
        assert "2 steps" in out
        # State file should exist in the created session dir
        init_state_file = self.sessions_base / "test_2026-01-01" / "agent-state.json"
        assert init_state_file.is_file()
        mgr = StateManager(init_state_file)
        state = mgr.load()
        assert state.pipeline == "full"
        assert "pkg-a" in state.affected_packages


# -- 11. cmd_status (lines 270-272) ------------------------------------------

class TestCmdStatus(CLITestBase):
    def test_outputs_json(self, capsys):
        self._make_pipeline_state()
        args = argparse.Namespace(json=True)
        pipeline_cli.cmd_status(args)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "pipeline" in data
        assert data["pipeline"] == "full"

    def test_no_state_exits(self):
        args = argparse.Namespace(json=True)
        with pytest.raises(SystemExit):
            pipeline_cli.cmd_status(args)


# -- 12. cmd_next (lines 276-322) --------------------------------------------

class TestCmdNext(CLITestBase):
    def test_all_complete(self, capsys):
        self._make_pipeline_state(
            statuses={"build": StepStatus.COMPLETE, "lint": StepStatus.COMPLETE},
            dep_graph={"lint": ["build"]},
        )
        args = argparse.Namespace()
        pipeline_cli.cmd_next(args)
        out = json.loads(capsys.readouterr().out)
        assert out["step"] is None
        assert "All steps complete" in out["message"]

    def test_runnable_with_deps_met(self, capsys):
        self._make_pipeline_state(
            statuses={"build": StepStatus.COMPLETE},
            dep_graph={"lint": ["build"]},
        )
        args = argparse.Namespace()
        pipeline_cli.cmd_next(args)
        out = json.loads(capsys.readouterr().out)
        assert len(out["runnable"]) == 1
        assert out["runnable"][0]["step"] == "lint"

    def test_blocked_steps(self, capsys):
        self._make_pipeline_state(dep_graph={"lint": ["build"]})
        args = argparse.Namespace()
        pipeline_cli.cmd_next(args)
        out = json.loads(capsys.readouterr().out)
        # build is runnable (no deps), lint is blocked
        runnable_names = [r["step"] for r in out["runnable"]]
        assert "build" in runnable_names
        assert "lint" in out["blocked"]

    def test_legacy_mode_linear(self, capsys):
        """Legacy mode with no dependency graph uses linear order."""
        self._make_pipeline_state(
            statuses={"build": StepStatus.COMPLETE},
            dep_graph={},  # no dependency graph = legacy mode
        )
        args = argparse.Namespace()
        pipeline_cli.cmd_next(args)
        out = json.loads(capsys.readouterr().out)
        runnable_names = [r["step"] for r in out["runnable"]]
        assert "lint" in runnable_names


# -- 13. cmd_execute (lines 327-402) -----------------------------------------

class TestCmdExecute(CLITestBase):
    def test_unknown_step_exits(self):
        self._make_pipeline_state()
        args = argparse.Namespace(step="nonexistent")
        with pytest.raises(SystemExit):
            pipeline_cli.cmd_execute(args)

    def test_non_command_step_exits(self):
        # Create a state with an AI step
        preset_dir = _make_manifest_dir(
            self.presets_dir,
            steps={
                "code": {"type": "ai", "description": "AI step"},
                "build": {"type": "command", "run_command": "echo ok"},
            },
            pipelines={"full": {"steps": ["code", "build"], "dependencies": {"build": ["code"]}}},
        )
        self._make_pipeline_state(step_order=["code", "build"])
        args = argparse.Namespace(step="code")
        with pytest.raises(SystemExit):
            pipeline_cli.cmd_execute(args)

    def test_successful_execution(self, capsys):
        self._make_pipeline_state(
            statuses={"build": StepStatus.COMPLETE},
            dep_graph={"lint": ["build"]},
        )
        with patch.object(pipeline_cli, "execute_command") as mock_exec:
            from forge.executor.engine.runner import StepResult
            mock_exec.return_value = StepResult(passed=True, output="ok")
            args = argparse.Namespace(step="lint")
            pipeline_cli.cmd_execute(args)
        out = capsys.readouterr().out
        lines = [l for l in out.strip().split("\n") if l.startswith("{")]
        result = json.loads(lines[-1])
        assert result["result"] == "passed"

    def test_failed_execution_with_error_file(self, capsys):
        self._make_pipeline_state(dep_graph={})
        with patch.object(pipeline_cli, "execute_command") as mock_exec:
            from forge.executor.engine.runner import StepResult
            mock_exec.return_value = StepResult(
                passed=False, output="error output",
                error_file="/tmp/errors.txt",
            )
            args = argparse.Namespace(step="build")
            pipeline_cli.cmd_execute(args)
        out = capsys.readouterr().out
        lines = [l for l in out.strip().split("\n") if l.startswith("{")]
        result = json.loads(lines[-1])
        assert result["result"] == "failed"
        assert "error_file" in result

    def test_failed_execution_with_failed_packages(self, capsys):
        self._make_pipeline_state(dep_graph={})
        with patch.object(pipeline_cli, "execute_command") as mock_exec:
            from forge.executor.engine.runner import StepResult
            mock_exec.return_value = StepResult(
                passed=False, output="pkg fail",
                failed_packages=["pkg-a"],
                error_files={"pkg-a": "/tmp/pkg-a-err.txt"},
            )
            args = argparse.Namespace(step="build")
            pipeline_cli.cmd_execute(args)
        out = capsys.readouterr().out
        lines = [l for l in out.strip().split("\n") if l.startswith("{")]
        result = json.loads(lines[-1])
        assert result["result"] == "failed"
        assert "failed_packages" in result

    def test_evidence_check_failure(self, capsys):
        self._make_pipeline_state(dep_graph={})
        with patch.object(pipeline_cli, "execute_command") as mock_exec:
            from forge.executor.engine.runner import StepResult
            mock_exec.return_value = StepResult(passed=True, output="ok")
            with patch("forge.executor.commands.EvidenceChecker") as mock_checker_cls:
                mock_checker = MagicMock()
                mock_checker.check.return_value = EvidenceResult(
                    passed=False, message="Missing build-output.txt",
                )
                mock_checker_cls.return_value = mock_checker
                args = argparse.Namespace(step="build")
                pipeline_cli.cmd_execute(args)
        out = capsys.readouterr().out
        lines = [l for l in out.strip().split("\n") if l.startswith("{")]
        result = json.loads(lines[-1])
        assert result["result"] == "failed"
        assert "Missing" in result["reason"]


# -- 14. cmd_dispatch (lines 407-457) ----------------------------------------

class TestCmdDispatch(CLITestBase):
    def _make_dispatch_state(self, steps_manifest=None, pipelines_manifest=None):
        steps_m = steps_manifest or {
            "build": {"type": "command", "run_command": "echo ok", "error_file": "err.txt",
                       "fix_hints": "check deps"},
            "lint": {"type": "command", "run_command": "echo lint ok"},
            "review": {"type": "ai", "skill": "/dev/null", "subagent_type": "reviewer"},
            "report": {"type": "inline", "description": "Summary step"},
        }
        pipelines_m = pipelines_manifest or {
            "full": {
                "steps": ["build", "lint", "review", "report"],
                "dependencies": {"lint": ["build"], "review": ["lint"], "report": ["review"]},
            },
        }
        _make_manifest_dir(self.presets_dir, pipelines=pipelines_m, steps=steps_m)
        self._make_pipeline_state(
            step_order=["build", "lint", "review", "report"],
            dep_graph={"lint": ["build"], "review": ["lint"], "report": ["review"]},
        )

    def test_unknown_step_exits(self):
        self._make_pipeline_state()
        args = argparse.Namespace(step="ghost", phase="run", failed_packages="")
        with pytest.raises(SystemExit):
            pipeline_cli.cmd_dispatch(args)

    def test_inline_step_returns_info(self, capsys):
        self._make_dispatch_state()
        args = argparse.Namespace(step="report", phase="run", failed_packages="")
        pipeline_cli.cmd_dispatch(args)
        out = json.loads(capsys.readouterr().out)
        assert out["type"] == "inline"

    def test_command_step_run_phase_errors(self):
        self._make_dispatch_state()
        args = argparse.Namespace(step="build", phase="run", failed_packages="")
        with pytest.raises(SystemExit):
            pipeline_cli.cmd_dispatch(args)

    def test_command_step_fix_phase(self, capsys):
        self._make_dispatch_state()
        args = argparse.Namespace(step="build", phase="fix", failed_packages="")
        pipeline_cli.cmd_dispatch(args)
        out = json.loads(capsys.readouterr().out)
        assert out["phase"] == "fix"
        assert "prompt" in out
        assert out["model"] == "sonnet"  # fix model

    def test_ai_step_run_phase(self, capsys):
        self._make_dispatch_state()
        args = argparse.Namespace(step="review", phase="run", failed_packages="")
        with patch.object(pipeline_cli, "generate_ai_prompt", return_value="review prompt"):
            pipeline_cli.cmd_dispatch(args)
        out = json.loads(capsys.readouterr().out)
        assert out["phase"] == "run"
        assert out["subagent_type"] == "reviewer"
        assert "prompt" in out

    def test_ai_step_fix_phase(self, capsys):
        self._make_dispatch_state()
        args = argparse.Namespace(step="review", phase="fix", failed_packages="")
        with patch.object(pipeline_cli, "generate_ai_fix_prompt", return_value="fix prompt"):
            pipeline_cli.cmd_dispatch(args)
        out = json.loads(capsys.readouterr().out)
        assert out["phase"] == "fix"
        assert "prompt" in out


# -- 15. cmd_run (lines 461-480) ---------------------------------------------

class TestCmdRun(CLITestBase):
    def test_unknown_step_exits(self):
        self._make_pipeline_state()
        args = argparse.Namespace(step="ghost")
        with pytest.raises(SystemExit):
            pipeline_cli.cmd_run(args)

    def test_successful_run(self, capsys):
        self._make_pipeline_state(dep_graph={})
        args = argparse.Namespace(step="build")
        pipeline_cli.cmd_run(args)
        out = capsys.readouterr().out
        assert "in_progress" in out
        # Verify state was updated
        mgr = StateManager(self.state_file)
        state = mgr.load()
        assert state.steps["build"].status == StepStatus.IN_PROGRESS

    def test_retry_with_artifact_archival(self, capsys):
        """On retry, existing evidence artifacts should be archived."""
        self._make_pipeline_state(dep_graph={})
        # Set retries > 0 to trigger archival
        mgr = StateManager(self.state_file)
        def set_retries(s):
            s.steps["build"].retries = 1
        mgr.update(set_retries)

        # Create an artifact that the evidence glob would match
        artifact = os.path.join(self.session_dir, "build-output.txt")
        Path(artifact).write_text("old build output")

        args = argparse.Namespace(step="build")
        pipeline_cli.cmd_run(args)
        out = capsys.readouterr().out
        assert "in_progress" in out

    def test_deps_not_met_exits(self):
        self._make_pipeline_state(dep_graph={"lint": ["build"]})
        args = argparse.Namespace(step="lint")
        with pytest.raises(SystemExit):
            pipeline_cli.cmd_run(args)


# -- 16. cmd_pass (lines 487-502) --------------------------------------------

class TestCmdPass(CLITestBase):
    def test_unknown_step_exits(self):
        self._make_pipeline_state()
        args = argparse.Namespace(step="ghost")
        with pytest.raises(SystemExit):
            pipeline_cli.cmd_pass(args)

    def test_evidence_failure_exits(self):
        self._make_pipeline_state(dep_graph={})
        with patch("forge.executor.commands.EvidenceChecker") as mock_cls:
            mock_checker = MagicMock()
            mock_checker.check.return_value = EvidenceResult(
                passed=False, message="Missing file",
            )
            mock_cls.return_value = mock_checker
            args = argparse.Namespace(step="build")
            with pytest.raises(SystemExit):
                pipeline_cli.cmd_pass(args)

    def test_successful_pass(self, capsys):
        # Use lint with build already complete so deps are met
        self._make_pipeline_state(
            statuses={"build": StepStatus.COMPLETE},
            dep_graph={"lint": ["build"]},
        )
        args = argparse.Namespace(step="lint")  # lint has no evidence rules
        pipeline_cli.cmd_pass(args)
        out = capsys.readouterr().out
        assert "complete" in out
        # Verify checkpoint written
        cp = os.path.join(self.checkpoint_dir, "lint.passed")
        assert os.path.isfile(cp)

    def test_pass_with_revalidation(self, capsys):
        """Passing a step with retries > 0 triggers revalidation of targets."""
        _make_manifest_dir(
            self.presets_dir,
            pipelines={
                "full": {
                    "steps": ["build", "lint"],
                    "dependencies": {"lint": ["build"]},
                    "revalidation_targets": ["build"],
                },
            },
        )
        self._make_pipeline_state(
            statuses={"build": StepStatus.COMPLETE},
            dep_graph={"lint": ["build"]},
        )
        # Write a build checkpoint so there's something to reset
        from forge.executor.engine.checkpoint import write_checkpoint
        write_checkpoint(self.checkpoint_dir, "build", "full", None)

        # Set lint retries > 0 so revalidation triggers
        mgr = StateManager(self.state_file)
        def set_retries(s):
            s.steps["lint"].retries = 1
        mgr.update(set_retries)

        args = argparse.Namespace(step="lint")
        pipeline_cli.cmd_pass(args)
        out = capsys.readouterr().out
        assert "complete" in out
        # Revalidation should reset build to pending
        state = StateManager(self.state_file).load()
        assert state.steps["build"].status == StepStatus.PENDING


# -- 17. cmd_fail (lines 535-552) --------------------------------------------

class TestCmdFail(CLITestBase):
    def test_unknown_step_exits(self):
        self._make_pipeline_state()
        args = argparse.Namespace(step="ghost", error="")
        with pytest.raises(SystemExit):
            pipeline_cli.cmd_fail(args)

    def test_first_failure(self, capsys):
        self._make_pipeline_state(dep_graph={})
        args = argparse.Namespace(step="build", error="build broke")
        pipeline_cli.cmd_fail(args)
        out = json.loads(capsys.readouterr().out)
        assert out["status"] == "failed"
        assert out["retries"] == 1

    def test_pipeline_exhausted(self, capsys):
        """After MAX_RETRIES failures, pipeline is exhausted."""
        self._make_pipeline_state(
            dep_graph={"lint": ["build"]},
        )
        mgr = StateManager(self.state_file)
        def set_retries(s):
            s.steps["build"].retries = 2
        mgr.update(set_retries)

        args = argparse.Namespace(step="build", error="exhausted")
        pipeline_cli.cmd_fail(args)
        out = json.loads(capsys.readouterr().out)
        assert out.get("pipeline_exhausted") is True


# -- 18. cmd_reset (lines 572-591) -------------------------------------------

class TestCmdReset(CLITestBase):
    def test_unknown_step_exits(self):
        self._make_pipeline_state()
        args = argparse.Namespace(step="ghost", no_retry_inc=False)
        with pytest.raises(SystemExit):
            pipeline_cli.cmd_reset(args)

    def test_basic_reset(self, capsys):
        self._make_pipeline_state(statuses={"build": StepStatus.FAILED})
        # Create a checkpoint file
        cp = os.path.join(self.checkpoint_dir, "build.passed")
        Path(cp).write_text("step=build\n")
        args = argparse.Namespace(step="build", no_retry_inc=False)
        pipeline_cli.cmd_reset(args)
        out = capsys.readouterr().out
        assert "reset to pending" in out
        assert not os.path.isfile(cp)
        state = StateManager(self.state_file).load()
        assert state.steps["build"].status == StepStatus.PENDING


# -- 19. cmd_add_packages ----------------------------------------------------

class TestCmdAddPackages(CLITestBase):
    def test_add_packages(self, capsys):
        self._make_pipeline_state(packages=["pkg-a"])
        args = argparse.Namespace(packages=["pkg-b", "pkg-c"])
        pipeline_cli.cmd_add_packages(args)
        out = capsys.readouterr().out
        assert "pkg-a" in out
        assert "pkg-b" in out
        assert "pkg-c" in out
        state = StateManager(self.state_file).load()
        assert sorted(state.affected_packages) == ["pkg-a", "pkg-b", "pkg-c"]

    def test_add_duplicate_packages(self, capsys):
        self._make_pipeline_state(packages=["pkg-a"])
        args = argparse.Namespace(packages=["pkg-a"])
        pipeline_cli.cmd_add_packages(args)
        state = StateManager(self.state_file).load()
        assert state.affected_packages == ["pkg-a"]


# -- 21. cmd_verify (lines 658-668) ------------------------------------------

class TestCmdVerify(CLITestBase):
    def test_all_valid(self, capsys):
        self._make_pipeline_state()
        with patch("forge.executor.commands.verify_all_checkpoints") as mock_verify:
            mock_verify.return_value = (True, ["build", "lint"], [])
            args = argparse.Namespace()
            pipeline_cli.cmd_verify(args)
        out = capsys.readouterr().out
        assert "PIPELINE COMPLETE" in out

    def test_missing_checkpoints_exits(self, capsys):
        self._make_pipeline_state()
        with patch("forge.executor.commands.verify_all_checkpoints") as mock_verify:
            mock_verify.return_value = (False, ["build"], ["lint"])
            args = argparse.Namespace()
            with pytest.raises(SystemExit):
                pipeline_cli.cmd_verify(args)
        out = capsys.readouterr().out
        assert "FAILED" in out
        assert "lint" in out
        assert "Present: build" in out


# -- 22. cmd_model (lines 672-675) -------------------------------------------

class TestCmdModel(CLITestBase):
    def test_model_resolution_run(self, capsys):
        self._make_pipeline_state()
        args = argparse.Namespace(step="code_review", phase="run")
        pipeline_cli.cmd_model(args)
        out = capsys.readouterr().out.strip()
        assert out == "opus"

    def test_model_resolution_fix(self, capsys):
        self._make_pipeline_state()
        args = argparse.Namespace(step="build", phase="fix")
        pipeline_cli.cmd_model(args)
        out = capsys.readouterr().out.strip()
        assert out == "sonnet"


# -- 23. cmd_cleanup (lines 679-688) -----------------------------------------

class TestCmdCleanup(CLITestBase):
    def test_cleanup_no_removals(self, capsys):
        with patch("forge.executor.commands._core_cleanup_sessions", return_value=[]):
            args = argparse.Namespace(older_than=7)
            pipeline_cli.cmd_cleanup(args)
        out = capsys.readouterr().out
        assert "No sessions to clean up" in out

    def test_cleanup_with_removals(self, capsys):
        with patch("forge.executor.commands._core_cleanup_sessions", return_value=["/tmp/old-session"]):
            args = argparse.Namespace(older_than=7)
            pipeline_cli.cmd_cleanup(args)
        out = capsys.readouterr().out
        assert "Removed 1 session(s)" in out
        assert "/tmp/old-session" in out


# -- 24. cmd_sessions (lines 692-697) ----------------------------------------

class TestCmdSessions:
    def test_no_sessions(self, capsys):
        with patch("forge.executor.commands._core_list_sessions", return_value=[]):
            args = argparse.Namespace()
            pipeline_cli.cmd_sessions(args)
        out = capsys.readouterr().out
        assert "No pipeline sessions found" in out

    def test_with_sessions(self, capsys):
        sessions = [
            {"name": "PROJ-123_2026-01-01", "path": "/tmp/sessions/PROJ-123", "age_days": 2.0},
        ]
        with patch("forge.executor.commands._core_list_sessions", return_value=sessions):
            args = argparse.Namespace()
            pipeline_cli.cmd_sessions(args)
        out = capsys.readouterr().out
        assert "PROJ-123_2026-01-01" in out
        assert "2.0d old" in out


# -- 25. Port management (lines 706-715) -------------------------------------

class TestPortManagement:
    def test_is_port_listening_false(self):
        # Use a port that's almost certainly not listening
        assert pipeline_cli._is_port_listening(19999) is False

    def test_is_port_listening_true(self):
        # Bind a port and check
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.listen(1)
        try:
            assert pipeline_cli._is_port_listening(port) is True
        finally:
            sock.close()

    def test_find_free_port(self):
        with patch.object(pipeline_cli, "_is_port_listening", return_value=False):
            port = pipeline_cli._find_free_port()
            assert port == pipeline_cli.DEFAULT_DEV_PORT

    def test_find_free_port_no_free(self):
        with patch.object(pipeline_cli, "_is_port_listening", return_value=True):
            with pytest.raises(RuntimeError, match="No free port"):
                pipeline_cli._find_free_port()


# -- 26. cmd_dev_server (lines 720-741) --------------------------------------

class TestCmdDevServer(CLITestBase):
    def test_allocate_reuses_existing_running(self, capsys):
        self._make_pipeline_state(dev_server_port=9090)
        with patch.object(pipeline_cli, "_is_port_listening", return_value=True):
            args = argparse.Namespace(action="allocate")
            pipeline_cli.cmd_dev_server(args)
        out = json.loads(capsys.readouterr().out)
        assert out["port"] == 9090
        assert out["reused"] is True
        assert out["running"] is True

    def test_allocate_new_port(self, capsys):
        self._make_pipeline_state(dev_server_port=0)
        with patch.object(pipeline_cli, "_find_free_port", return_value=8080):
            args = argparse.Namespace(action="allocate")
            pipeline_cli.cmd_dev_server(args)
        out = json.loads(capsys.readouterr().out)
        assert out["port"] == 8080
        assert out["reused"] is False
        assert out["running"] is False
        # Verify port was saved to state
        state = StateManager(self.state_file).load()
        assert state.dev_server_port == 8080

    def test_status(self, capsys):
        self._make_pipeline_state(dev_server_port=9090)
        with patch.object(pipeline_cli, "_is_port_listening", return_value=False):
            args = argparse.Namespace(action="status")
            pipeline_cli.cmd_dev_server(args)
        out = json.loads(capsys.readouterr().out)
        assert out["port"] == 9090
        assert out["running"] is False
        assert out["allocated"] is True

    def test_status_default_port(self, capsys):
        self._make_pipeline_state(dev_server_port=0)
        with patch.object(pipeline_cli, "_is_port_listening", return_value=False):
            args = argparse.Namespace(action="status")
            pipeline_cli.cmd_dev_server(args)
        out = json.loads(capsys.readouterr().out)
        assert out["port"] == pipeline_cli.DEFAULT_DEV_PORT
        assert out["allocated"] is False


# -- 27. main (lines 751-835) ------------------------------------------------

class TestMain(CLITestBase):
    def test_no_command_exits(self):
        with patch("sys.argv", ["forge-executor"]):
            with pytest.raises(SystemExit):
                pipeline_cli.main()

    def test_status_command_dispatch(self, capsys):
        self._make_pipeline_state()
        with patch("sys.argv", ["forge-executor", "status"]):
            pipeline_cli.main()
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "pipeline" in data

    def test_next_command_dispatch(self, capsys):
        self._make_pipeline_state(dep_graph={})
        with patch("sys.argv", ["forge-executor", "next"]):
            pipeline_cli.main()
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "runnable" in data

    def test_run_command_dispatch(self, capsys):
        self._make_pipeline_state(dep_graph={})
        with patch("sys.argv", ["forge-executor", "run", "build"]):
            pipeline_cli.main()
        out = capsys.readouterr().out
        assert "in_progress" in out

    def test_pass_command_dispatch(self, capsys):
        self._make_pipeline_state(
            statuses={"build": StepStatus.COMPLETE},
            dep_graph={"lint": ["build"]},
        )
        with patch("sys.argv", ["forge-executor", "pass", "lint"]):
            pipeline_cli.main()
        out = capsys.readouterr().out
        assert "complete" in out

    def test_fail_command_dispatch(self, capsys):
        self._make_pipeline_state(dep_graph={})
        with patch("sys.argv", ["forge-executor", "fail", "build", "error msg"]):
            pipeline_cli.main()
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["status"] == "failed"

    def test_reset_command_dispatch(self, capsys):
        self._make_pipeline_state(statuses={"build": StepStatus.FAILED})
        with patch("sys.argv", ["forge-executor", "reset", "build"]):
            pipeline_cli.main()
        out = capsys.readouterr().out
        assert "reset to pending" in out

    def test_verify_command_dispatch(self, capsys):
        self._make_pipeline_state()
        with patch("forge.executor.commands.verify_all_checkpoints", return_value=(True, ["build", "lint"], [])):
            with patch("sys.argv", ["forge-executor", "verify"]):
                pipeline_cli.main()
        out = capsys.readouterr().out
        assert "PIPELINE COMPLETE" in out

    def test_model_command_dispatch(self, capsys):
        self._make_pipeline_state()
        with patch("sys.argv", ["forge-executor", "model", "code_review", "run"]):
            pipeline_cli.main()
        out = capsys.readouterr().out.strip()
        assert out == "opus"

    def test_sessions_command_dispatch(self, capsys):
        with patch("forge.executor.commands._core_list_sessions", return_value=[]):
            with patch("sys.argv", ["forge-executor", "sessions"]):
                pipeline_cli.main()
        out = capsys.readouterr().out
        assert "No pipeline sessions found" in out

    def test_add_packages_command_dispatch(self, capsys):
        self._make_pipeline_state()
        with patch("sys.argv", ["forge-executor", "add-packages", "new-pkg"]):
            pipeline_cli.main()
        out = capsys.readouterr().out
        assert "new-pkg" in out

    def test_cleanup_command_dispatch(self, capsys):
        with patch("forge.executor.commands._core_cleanup_sessions", return_value=[]):
            with patch("sys.argv", ["forge-executor", "cleanup"]):
                pipeline_cli.main()
        out = capsys.readouterr().out
        assert "No sessions to clean up" in out

    def test_init_command_dispatch(self, capsys):
        with patch.object(pipeline_cli, "_is_worktree", return_value=(True, "/some/path")):
            with patch.object(pipeline_cli, "_session_name", return_value="test_2026-01-01"):
                with patch("sys.argv", ["forge-executor", "init", "full", "--preset", "test-preset"]):
                    pipeline_cli.main()
        out = capsys.readouterr().out
        assert "Pipeline initialized" in out

    def test_dev_server_command_dispatch(self, capsys):
        self._make_pipeline_state(dev_server_port=0)
        with patch.object(pipeline_cli, "_find_free_port", return_value=8080):
            with patch("sys.argv", ["forge-executor", "dev-server", "allocate"]):
                pipeline_cli.main()
        out = json.loads(capsys.readouterr().out)
        assert out["port"] == 8080

    def test_dispatch_command_dispatch(self, capsys):
        steps_m = {
            "build": {"type": "command", "run_command": "echo ok"},
            "lint": {"type": "command", "run_command": "echo lint"},
        }
        pipelines_m = {
            "full": {"steps": ["build", "lint"], "dependencies": {"lint": ["build"]}},
        }
        _make_manifest_dir(self.presets_dir, pipelines=pipelines_m, steps=steps_m)
        self._make_pipeline_state(dep_graph={"lint": ["build"]})
        with patch("sys.argv", ["forge-executor", "dispatch", "build", "fix"]):
            pipeline_cli.main()
        out = json.loads(capsys.readouterr().out)
        assert out["phase"] == "fix"

    def test_execute_command_dispatch(self, capsys):
        # Use lint (no evidence rules) with build complete so deps are met
        self._make_pipeline_state(
            statuses={"build": StepStatus.COMPLETE},
            dep_graph={"lint": ["build"]},
        )
        with patch.object(pipeline_cli, "execute_command") as mock_exec:
            from forge.executor.engine.runner import StepResult
            mock_exec.return_value = StepResult(passed=True, output="ok")
            with patch("sys.argv", ["forge-executor", "execute", "lint"]):
                pipeline_cli.main()
        out = capsys.readouterr().out
        lines = [l for l in out.strip().split("\n") if l.startswith("{")]
        result = json.loads(lines[-1])
        assert result["result"] == "passed"


# -- Remaining coverage gaps --------------------------------------------------


class TestArchiveOutputSectionNoMatch(CLITestBase):
    """Cover line 184: _archive_output_section returns when no matching section."""

    def test_no_matching_section_returns_early(self):
        output_path = os.path.join(self.session_dir, "pipeline-output.md")
        Path(output_path).write_text("# Pipeline Output\n\n## other_step\nSome content\n")
        archive_dir = os.path.join(self.session_dir, "_retries", "build_attempt_1")
        pipeline_cli._archive_output_section(
            self.session_dir, "build", archive_dir,
        )
        assert not os.path.isdir(archive_dir)
        assert "other_step" in Path(output_path).read_text()


class TestCmdExecuteOutputOSError(CLITestBase):
    """Cover lines 365-366: OSError writing to pipeline-output.md in cmd_execute."""

    def test_oserror_writing_output_silently_ignored(self, capsys):
        # Use lint (no evidence rules) with build already complete
        self._make_pipeline_state(
            statuses={"build": StepStatus.COMPLETE},
            dep_graph={"lint": ["build"]},
        )
        original_open = open

        def failing_open(path, *args, **kwargs):
            if "pipeline-output.md" in str(path):
                raise OSError("disk full")
            return original_open(path, *args, **kwargs)

        with patch.object(pipeline_cli, "execute_command") as mock_exec:
            mock_exec.return_value = StepResult(passed=True, output="ok")
            with patch("builtins.open", side_effect=failing_open):
                args = SimpleNamespace(step="lint")
                pipeline_cli.cmd_execute(args)
        out = capsys.readouterr().out
        assert "passed" in out


class TestCmdDispatchNoStepDef(CLITestBase):
    """Cover lines 418-419: cmd_dispatch when step has no definition in preset."""

    def test_dispatch_no_step_definition_exits(self):
        self._make_pipeline_state(step_order=["build", "lint", "unknown_step"])
        args = SimpleNamespace(step="unknown_step", phase="run", failed_packages="")
        with pytest.raises(SystemExit):
            pipeline_cli.cmd_dispatch(args)


class TestCmdSummaryEnrichedJson(CLITestBase):
    """Test enriched summary JSON with per-step timing fields."""

    def test_json_includes_timing_fields(self, capsys):
        self._make_pipeline_state(
            statuses={"build": StepStatus.COMPLETE, "lint": StepStatus.IN_PROGRESS},
            dep_graph={"lint": ["build"]},
        )
        mgr = StateManager(self.state_file)
        def set_timing(s):
            s.steps["build"].started_at = "2026-02-17T10:00:00Z"
            s.steps["build"].completed_at = "2026-02-17T10:05:00Z"
            s.steps["lint"].started_at = "2026-02-17T10:05:01Z"
        mgr.update(set_timing)

        args = argparse.Namespace(json=True)
        pipeline_cli.cmd_summary(args)
        out = json.loads(capsys.readouterr().out)

        assert out["pipeline"] == "full"
        assert out["preset"] == "test-preset"
        assert "steps" in out
        assert out["steps"]["build"]["status"] == "complete"
        assert out["steps"]["build"]["duration_s"] == 300
        assert out["steps"]["build"]["started_at"] == "2026-02-17T10:00:00Z"
        assert out["steps"]["lint"]["status"] == "in_progress"
        assert "elapsed_s" in out["steps"]["lint"]
        assert out["steps"]["lint"]["elapsed_s"] > 0
        assert "driver_pid" in out
        assert "killed" in out
        assert "session_dir" in out

    def test_json_without_timing(self, capsys):
        self._make_pipeline_state(dep_graph={})
        args = argparse.Namespace(json=True)
        pipeline_cli.cmd_summary(args)
        out = json.loads(capsys.readouterr().out)

        assert out["steps"]["build"]["status"] == "pending"
        assert "started_at" not in out["steps"]["build"]
        assert "duration_s" not in out["steps"]["build"]

    def test_json_includes_last_error(self, capsys):
        self._make_pipeline_state(
            statuses={"build": StepStatus.FAILED},
            dep_graph={},
        )
        mgr = StateManager(self.state_file)
        def set_error(s):
            s.steps["build"].last_error = "compile error"
            s.steps["build"].retries = 2
        mgr.update(set_error)

        args = argparse.Namespace(json=True)
        pipeline_cli.cmd_summary(args)
        out = json.loads(capsys.readouterr().out)

        assert out["steps"]["build"]["last_error"] == "compile error"
        assert out["steps"]["build"]["retries"] == 2

    def test_text_mode_unchanged(self, capsys):
        self._make_pipeline_state(dep_graph={})
        args = argparse.Namespace(json=False)
        pipeline_cli.cmd_summary(args)
        out = capsys.readouterr().out
        assert "Pipeline: full" in out
        assert "{" not in out


class TestCmdPassEvidenceOkMessage(CLITestBase):
    """Cover line 502: evidence passes and prints OK message."""

    def test_pass_with_evidence_prints_ok(self, capsys):
        self._make_pipeline_state(statuses={"build": StepStatus.IN_PROGRESS})
        Path(os.path.join(self.session_dir, "build-output.txt")).write_text("build succeeded")

        args = SimpleNamespace(step="build")
        pipeline_cli.cmd_pass(args)
        out = capsys.readouterr().out
        assert "Evidence OK" in out
        assert "complete" in out
