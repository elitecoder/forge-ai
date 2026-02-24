"""Tests for registry.py — PipelineDefinition, dual-format loading, DAG validation, preset resolution."""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch

from forge.executor.engine.registry import (
    PipelineDefinition, _validate_dag, _parse_pipeline, load_preset,
)
from forge.executor.engine.pipeline_ops import resolve_preset_dir


class TestParsePipeline:
    def test_legacy_array_format(self):
        raw = ["build_gen", "lint", "test"]
        result = _parse_pipeline("test", raw)
        assert result.steps == ["build_gen", "lint", "test"]
        assert result.dependencies == {}
        assert result.is_legacy is True

    def test_new_object_format(self):
        raw = {
            "steps": ["build_gen", "lint", "test"],
            "dependencies": {"lint": ["build_gen"], "test": ["build_gen"]},
        }
        result = _parse_pipeline("test", raw)
        assert result.steps == ["build_gen", "lint", "test"]
        assert result.dependencies == {"lint": ["build_gen"], "test": ["build_gen"]}
        assert result.is_legacy is False

    def test_new_format_no_dependencies(self):
        raw = {"steps": ["a", "b"]}
        result = _parse_pipeline("test", raw)
        assert result.dependencies == {}
        assert result.is_legacy is False


class TestValidateDag:
    def test_valid_dag(self):
        pd = PipelineDefinition(
            steps=["a", "b", "c"],
            dependencies={"b": ["a"], "c": ["a", "b"]},
        )
        _validate_dag("test", pd)  # should not raise

    def test_root_nodes_only(self):
        pd = PipelineDefinition(steps=["a", "b", "c"], dependencies={})
        _validate_dag("test", pd)  # should not raise

    def test_dep_references_unknown_step(self):
        pd = PipelineDefinition(
            steps=["a", "b"],
            dependencies={"b": ["nonexistent"]},
        )
        with pytest.raises(ValueError, match="unknown step 'nonexistent'"):
            _validate_dag("test", pd)

    def test_dep_key_not_in_steps(self):
        pd = PipelineDefinition(
            steps=["a", "b"],
            dependencies={"c": ["a"]},
        )
        with pytest.raises(ValueError, match="dependency key 'c' not in steps"):
            _validate_dag("test", pd)

    def test_cycle_detection_simple(self):
        pd = PipelineDefinition(
            steps=["a", "b"],
            dependencies={"a": ["b"], "b": ["a"]},
        )
        with pytest.raises(ValueError, match="cycle"):
            _validate_dag("test", pd)

    def test_cycle_detection_three_node(self):
        pd = PipelineDefinition(
            steps=["a", "b", "c"],
            dependencies={"a": ["c"], "b": ["a"], "c": ["b"]},
        )
        with pytest.raises(ValueError, match="cycle"):
            _validate_dag("test", pd)

    def test_diamond_dag_no_cycle(self):
        pd = PipelineDefinition(
            steps=["root", "left", "right", "sink"],
            dependencies={
                "left": ["root"],
                "right": ["root"],
                "sink": ["left", "right"],
            },
        )
        _validate_dag("test", pd)  # should not raise


class TestLoadPreset:
    def test_loads_new_format(self, tmp_path):
        manifest = {
            "preset": "test",
            "version": 3,
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
        pd = preset.pipelines["full"]
        assert pd.steps == ["a", "b"]
        assert pd.dependencies == {"b": ["a"]}
        assert pd.is_legacy is False

    def test_loads_legacy_format(self, tmp_path):
        manifest = {
            "preset": "test",
            "version": 2,
            "pipelines": {"full": ["a", "b"]},
            "steps": {
                "a": {"type": "command", "run_command": "echo a"},
                "b": {"type": "command", "run_command": "echo b"},
            },
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        preset = load_preset(tmp_path)
        pd = preset.pipelines["full"]
        assert pd.steps == ["a", "b"]
        assert pd.dependencies == {}
        assert pd.is_legacy is True

    def test_unknown_step_reference(self, tmp_path):
        manifest = {
            "preset": "test",
            "pipelines": {
                "full": {"steps": ["a", "unknown"], "dependencies": {}},
            },
            "steps": {"a": {"type": "command", "run_command": "echo a"}},
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="unknown step 'unknown'"):
            load_preset(tmp_path)

    def test_cycle_in_manifest_rejected(self, tmp_path):
        manifest = {
            "preset": "test",
            "pipelines": {
                "full": {
                    "steps": ["a", "b"],
                    "dependencies": {"a": ["b"], "b": ["a"]},
                }
            },
            "steps": {
                "a": {"type": "command", "run_command": "echo a"},
                "b": {"type": "command", "run_command": "echo b"},
            },
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="cycle"):
            load_preset(tmp_path)

    def test_dev_server_loaded(self, tmp_path):
        manifest = {
            "preset": "test",
            "pipelines": {"full": ["a"]},
            "steps": {"a": {"type": "command", "run_command": "echo a"}},
            "dev_server": {"command": "npm run dev -- --port ${PORT}", "health_url": "http://localhost:${PORT}/"},
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        preset = load_preset(tmp_path)
        assert preset.dev_server["command"] == "npm run dev -- --port ${PORT}"
        assert preset.dev_server["health_url"] == "http://localhost:${PORT}/"

    def test_dev_server_defaults_empty(self, tmp_path):
        manifest = {
            "preset": "test",
            "pipelines": {"full": ["a"]},
            "steps": {"a": {"type": "command", "run_command": "echo a"}},
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        preset = load_preset(tmp_path)
        assert preset.dev_server == {}


def _make_test_preset(base_dir: Path, name: str = "my-preset") -> Path:
    """Create a minimal preset directory for resolution tests."""
    preset_dir = base_dir / name
    preset_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "preset": name,
        "pipelines": {"full": ["a"]},
        "steps": {"a": {"type": "command", "run_command": "echo ok"}},
    }
    (preset_dir / "manifest.json").write_text(json.dumps(manifest))
    return preset_dir


class TestResolvePresetDir:
    def test_builtin_preset(self):
        """Built-in npm-ts preset is found."""
        result = resolve_preset_dir("npm-ts")
        assert (result / "manifest.json").is_file()

    def test_builtin_python_uv(self):
        """Built-in python-uv preset is found."""
        result = resolve_preset_dir("python-uv")
        assert (result / "manifest.json").is_file()

    def test_absolute_path(self, tmp_path):
        _make_test_preset(tmp_path, "abs-preset")
        result = resolve_preset_dir(str(tmp_path / "abs-preset"))
        assert result == tmp_path / "abs-preset"

    def test_relative_path_with_slash(self, tmp_path, monkeypatch):
        _make_test_preset(tmp_path, "rel-preset")
        result = resolve_preset_dir(str(tmp_path / "rel-preset"))
        assert (result / "manifest.json").is_file()

    def test_user_presets_dir(self, tmp_path, monkeypatch):
        """~/.forge/presets/<name> is searched."""
        forge_home = tmp_path / "fakehome"
        _make_test_preset(forge_home / ".forge" / "presets", "user-preset")
        monkeypatch.setattr(Path, "home", staticmethod(lambda: forge_home))
        result = resolve_preset_dir("user-preset")
        assert result == forge_home / ".forge" / "presets" / "user-preset"

    def test_forge_presets_path_env(self, tmp_path, monkeypatch):
        """FORGE_PRESETS_PATH env var is searched."""
        ext_dir = tmp_path / "external"
        _make_test_preset(ext_dir, "env-preset")
        monkeypatch.setenv("FORGE_PRESETS_PATH", str(ext_dir))
        # Make sure user presets dir doesn't accidentally match
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "nohome"))
        result = resolve_preset_dir("env-preset")
        assert result == ext_dir / "env-preset"

    def test_not_found_raises(self, tmp_path, monkeypatch):
        """Unknown preset name raises FileNotFoundError with searched locations."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "nohome"))
        monkeypatch.delenv("FORGE_PRESETS_PATH", raising=False)
        with pytest.raises(FileNotFoundError, match="Preset 'nonexistent' not found"):
            resolve_preset_dir("nonexistent")

    def test_absolute_path_no_manifest_raises(self, tmp_path):
        """Absolute path without manifest.json raises."""
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="No manifest.json"):
            resolve_preset_dir(str(empty))

    def test_builtin_hz_web(self):
        """Built-in hz-web preset is found."""
        result = resolve_preset_dir("hz-web")
        assert (result / "manifest.json").is_file()
        assert (result / "skills" / "code-review.md").is_file()
        assert (result / "scripts" / "template.js").is_file()
        assert (result / "references" / "squirrel-quirks.md").is_file()


class TestVisualTestConfig:
    def test_visual_test_config_loaded(self, tmp_path):
        manifest = {
            "preset": "test",
            "pipelines": {"full": ["a"]},
            "steps": {"a": {"type": "command", "run_command": "echo a"}},
            "visual_test_config": {
                "template_path": "${PRESET_DIR}/scripts/template.js",
                "quirks_path": "${PRESET_DIR}/references/quirks.md",
                "fixture_patterns": ["fixtures/Main.ts"],
                "credential_env_vars": ["EMAIL", "PASSWORD"],
            },
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        preset = load_preset(tmp_path)
        assert preset.visual_test_config["template_path"] == "${PRESET_DIR}/scripts/template.js"
        assert preset.visual_test_config["fixture_patterns"] == ["fixtures/Main.ts"]

    def test_visual_test_config_defaults_empty(self, tmp_path):
        manifest = {
            "preset": "test",
            "pipelines": {"full": ["a"]},
            "steps": {"a": {"type": "command", "run_command": "echo a"}},
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        preset = load_preset(tmp_path)
        assert preset.visual_test_config == {}

    def test_hz_web_has_visual_test_config(self):
        """The hz-web preset has a non-empty visual_test_config."""
        preset_dir = resolve_preset_dir("hz-web")
        preset = load_preset(preset_dir)
        cfg = preset.visual_test_config
        assert cfg["template_path"] == "${PRESET_DIR}/scripts/template.js"
        assert "squirrel-quirks.md" in cfg["quirks_path"]
        assert len(cfg["fixture_patterns"]) >= 4
        assert "SQUIRREL_EMAIL" in cfg["credential_env_vars"]
