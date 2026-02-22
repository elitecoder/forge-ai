"""Tests for registry.py — PipelineDefinition, dual-format loading, DAG validation."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from architect.executor.engine.registry import (
    PipelineDefinition, _validate_dag, _parse_pipeline, load_preset,
)


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
