# Copyright 2026. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for forge.setup — global tool installation and setup gate."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from forge.setup import check_setup, run_setup, run_preset_setup, SETUP_PATH


@pytest.fixture(autouse=True)
def _isolate_setup(tmp_path, monkeypatch):
    """Redirect SETUP_PATH to a temp directory for every test."""
    fake_path = tmp_path / "setup.json"
    monkeypatch.setattr("forge.setup.SETUP_PATH", fake_path)
    yield fake_path


class TestCheckSetup:
    def test_no_file_returns_false(self, _isolate_setup):
        assert not check_setup()

    def test_file_exists_tools_present(self, _isolate_setup):
        _isolate_setup.write_text("{}")
        with patch("forge.setup.shutil.which", return_value="/usr/bin/eslint"):
            assert check_setup()

    def test_file_exists_tool_removed(self, _isolate_setup):
        _isolate_setup.write_text("{}")
        calls = {"eslint": "/usr/bin/eslint", "prettier": None}
        with patch("forge.setup.shutil.which", side_effect=lambda n: calls.get(n)):
            assert not check_setup()


class TestRunSetup:
    @patch("forge.setup.shutil.which", return_value="/usr/bin/tool")
    def test_all_present_writes_state(self, _mock_which, _isolate_setup):
        run_setup()
        assert _isolate_setup.exists()
        data = json.loads(_isolate_setup.read_text())
        assert data["version"] == 1
        assert "completed_at" in data
        assert "eslint" in data["tools"]

    @patch("forge.setup.subprocess.run")
    @patch("forge.setup.shutil.which")
    def test_installs_missing_tools(self, mock_which, mock_run, _isolate_setup):
        installed = set()

        def which_side(name):
            if name == "npm":
                return "/usr/bin/npm"
            if name in installed:
                return f"/usr/bin/{name}"
            return None

        def run_side(cmd, **_kwargs):
            installed.update(cmd[3:])  # ["npm", "install", "-g", ...]
            return MagicMock(returncode=0)

        mock_which.side_effect = which_side
        mock_run.side_effect = run_side

        run_setup()
        mock_run.assert_called_once()
        assert "eslint" in mock_run.call_args[0][0]
        assert "prettier" in mock_run.call_args[0][0]

    @patch("forge.setup.shutil.which", return_value=None)
    def test_no_npm_raises(self, _mock_which):
        with pytest.raises(RuntimeError, match="npm is not available"):
            run_setup()

    @patch("forge.setup.subprocess.run")
    @patch("forge.setup.shutil.which")
    def test_npm_failure_raises(self, mock_which, mock_run):
        mock_which.side_effect = lambda n: "/usr/bin/npm" if n == "npm" else None
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "npm", stderr="permission denied"
        )
        with pytest.raises(RuntimeError, match="Failed to install"):
            run_setup()

    @patch("forge.setup.subprocess.run")
    @patch("forge.setup.shutil.which")
    def test_timeout_raises(self, mock_which, mock_run):
        mock_which.side_effect = lambda n: "/usr/bin/npm" if n == "npm" else None
        mock_run.side_effect = subprocess.TimeoutExpired("npm", 120)
        with pytest.raises(RuntimeError, match="Timed out"):
            run_setup()

    @patch("forge.setup.subprocess.run")
    @patch("forge.setup.shutil.which")
    def test_still_missing_after_install_raises(self, mock_which, mock_run):
        mock_which.side_effect = lambda n: "/usr/bin/npm" if n == "npm" else None
        mock_run.return_value = MagicMock(returncode=0)
        with pytest.raises(RuntimeError, match="still not found on PATH"):
            run_setup()

    @patch("forge.setup.subprocess.run")
    @patch("forge.setup.shutil.which")
    def test_force_reruns(self, mock_which, mock_run, _isolate_setup):
        def which_side(name):
            if name in ("npm", "eslint", "prettier"):
                return f"/usr/bin/{name}"
            return None

        mock_which.side_effect = which_side
        mock_run.return_value = MagicMock(returncode=0)

        run_setup(force=True)
        mock_run.assert_called_once()
        assert "eslint" in mock_run.call_args[0][0]


class TestRunPresetSetup:
    """Tests for run_preset_setup() — preset-specific setup from manifest.json."""

    def _write_manifest(self, preset_dir: Path, setup_entries: list) -> None:
        manifest = preset_dir / "manifest.json"
        manifest.write_text(json.dumps({"preset": "test", "setup": setup_entries}))

    def test_no_manifest_is_noop(self, tmp_path):
        run_preset_setup(str(tmp_path))

    def test_empty_setup_is_noop(self, tmp_path):
        self._write_manifest(tmp_path, [])
        run_preset_setup(str(tmp_path))

    def test_invalid_json_is_noop(self, tmp_path):
        (tmp_path / "manifest.json").write_text("{bad json")
        run_preset_setup(str(tmp_path))

    @patch("forge.setup.subprocess.run")
    def test_runs_command(self, mock_run, tmp_path):
        cwd = tmp_path / "skill"
        cwd.mkdir()
        self._write_manifest(tmp_path, [
            {"name": "pw", "command": "echo hello", "cwd": str(cwd)},
        ])
        mock_run.return_value = MagicMock(returncode=0)

        run_preset_setup(str(tmp_path))

        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["shell"] is True
        assert mock_run.call_args.kwargs["cwd"] == str(cwd)

    @patch("forge.setup.subprocess.run")
    def test_preset_dir_substitution(self, mock_run, tmp_path):
        skill_dir = tmp_path / "bundled-skills" / "pw"
        skill_dir.mkdir(parents=True)
        self._write_manifest(tmp_path, [
            {
                "name": "pw",
                "command": "npm install",
                "cwd": "${PRESET_DIR}/bundled-skills/pw",
                "check": "${PRESET_DIR}/bundled-skills/pw/node_modules/.lock",
            },
        ])
        mock_run.return_value = MagicMock(returncode=0)

        run_preset_setup(str(tmp_path))

        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["cwd"] == str(skill_dir.resolve())

    def test_skips_when_check_file_exists(self, tmp_path):
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        check_file = skill_dir / "done.txt"
        check_file.write_text("ok")
        self._write_manifest(tmp_path, [
            {"name": "pw", "command": "should not run", "cwd": str(skill_dir),
             "check": str(check_file)},
        ])

        with patch("forge.setup.subprocess.run") as mock_run:
            run_preset_setup(str(tmp_path))
            mock_run.assert_not_called()

    @patch("forge.setup.subprocess.run")
    def test_force_ignores_check_file(self, mock_run, tmp_path):
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        check_file = skill_dir / "done.txt"
        check_file.write_text("ok")
        self._write_manifest(tmp_path, [
            {"name": "pw", "command": "echo redo", "cwd": str(skill_dir),
             "check": str(check_file)},
        ])
        mock_run.return_value = MagicMock(returncode=0)

        run_preset_setup(str(tmp_path), force=True)

        mock_run.assert_called_once()

    def test_missing_cwd_raises(self, tmp_path):
        self._write_manifest(tmp_path, [
            {"name": "pw", "command": "echo hi", "cwd": str(tmp_path / "nonexistent")},
        ])
        with pytest.raises(RuntimeError, match="working directory does not exist"):
            run_preset_setup(str(tmp_path))

    @patch("forge.setup.subprocess.run")
    def test_command_failure_raises(self, mock_run, tmp_path):
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        self._write_manifest(tmp_path, [
            {"name": "pw", "command": "false", "cwd": str(skill_dir)},
        ])
        mock_run.side_effect = subprocess.CalledProcessError(1, "false", stderr="boom")

        with pytest.raises(RuntimeError, match="Setup 'pw' failed: boom"):
            run_preset_setup(str(tmp_path))

    @patch("forge.setup.subprocess.run")
    def test_timeout_raises(self, mock_run, tmp_path):
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        self._write_manifest(tmp_path, [
            {"name": "pw", "command": "sleep 999", "cwd": str(skill_dir)},
        ])
        mock_run.side_effect = subprocess.TimeoutExpired("sleep", 300)

        with pytest.raises(RuntimeError, match="timed out after 300s"):
            run_preset_setup(str(tmp_path))

    @patch("forge.setup.subprocess.run")
    def test_skips_entry_without_command(self, mock_run, tmp_path):
        self._write_manifest(tmp_path, [
            {"name": "empty", "command": "", "cwd": str(tmp_path)},
        ])
        run_preset_setup(str(tmp_path))
        mock_run.assert_not_called()
