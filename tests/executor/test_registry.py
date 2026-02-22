
"""Tests for engine.registry — preset loading and model resolution."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from architect.executor.engine.registry import load_preset, get_model, Preset, StepDefinition


class TestLoadPreset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="reg_test_")
        self.preset_dir = Path(self.tmp) / "my-preset"
        self.preset_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _write_manifest(self, data: dict):
        (self.preset_dir / "manifest.json").write_text(json.dumps(data))

    def test_load_minimal_preset(self):
        self._write_manifest({
            "preset": "my-preset",
            "version": 2,
            "pipelines": {"full": ["build"]},
            "steps": {"build": {"type": "command", "run_command": "echo ok"}},
        })
        preset = load_preset(self.preset_dir)
        self.assertEqual(preset.name, "my-preset")
        self.assertEqual(preset.version, 2)
        self.assertIn("build", preset.steps)
        self.assertEqual(preset.steps["build"].step_type, "command")

    def test_missing_manifest_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_preset(self.preset_dir / "nonexistent")

    def test_unknown_step_in_pipeline_raises(self):
        self._write_manifest({
            "preset": "bad",
            "pipelines": {"full": ["ghost"]},
            "steps": {"build": {"type": "command"}},
        })
        with self.assertRaises(ValueError):
            load_preset(self.preset_dir)

    def test_evidence_parsing(self):
        self._write_manifest({
            "preset": "ev",
            "pipelines": {"full": ["check"]},
            "steps": {"check": {
                "type": "command",
                "evidence": [
                    {"rule": "file_exists", "file_glob": "*.json", "max_age_seconds": 3600},
                ],
            }},
        })
        preset = load_preset(self.preset_dir)
        rules = preset.steps["check"].evidence
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].rule, "file_exists")
        self.assertEqual(rules[0].max_age_seconds, 3600)

    def test_step_defaults(self):
        self._write_manifest({
            "preset": "dfl",
            "pipelines": {"full": ["s"]},
            "steps": {"s": {}},
        })
        preset = load_preset(self.preset_dir)
        step = preset.steps["s"]
        self.assertEqual(step.step_type, "command")
        self.assertEqual(step.timeout, 600)
        self.assertFalse(step.per_package)

    def test_ai_step_fields(self):
        self._write_manifest({
            "preset": "ai",
            "pipelines": {"full": ["review"]},
            "steps": {"review": {
                "type": "ai",
                "skill": "~/skill.md",
                "two_phase": True,
                "subagent_type": "general-purpose",
            }},
        })
        preset = load_preset(self.preset_dir)
        step = preset.steps["review"]
        self.assertEqual(step.step_type, "ai")
        self.assertTrue(step.two_phase)
        self.assertEqual(step.skill, "~/skill.md")

    def test_models_dict(self):
        self._write_manifest({
            "preset": "mdl",
            "pipelines": {"full": ["s"]},
            "steps": {"s": {}},
            "models": {"fix": "sonnet", "code_review": "opus"},
        })
        preset = load_preset(self.preset_dir)
        self.assertEqual(preset.models["fix"], "sonnet")
        self.assertEqual(preset.models["code_review"], "opus")

    def test_preset_dir_recorded(self):
        self._write_manifest({
            "preset": "dir",
            "pipelines": {"full": ["s"]},
            "steps": {"s": {}},
        })
        preset = load_preset(self.preset_dir)
        self.assertEqual(preset.preset_dir, self.preset_dir)


class TestGetModel(unittest.TestCase):
    def setUp(self):
        self.preset = Preset(
            name="test", version=2, description="",
            pipelines={"full": ["build", "code_review"]},
            steps={},
            models={"fix": "sonnet", "code_review": "opus"},
        )

    def test_fix_model_for_unknown_step(self):
        self.assertEqual(get_model(self.preset, "build", is_fix=True), "sonnet")

    def test_step_specific_model(self):
        self.assertEqual(get_model(self.preset, "code_review", is_fix=False), "opus")

    def test_fix_model_for_step_with_own_model(self):
        # is_fix=True always returns the fix model regardless of step-specific model
        self.assertEqual(get_model(self.preset, "code_review", is_fix=True), "sonnet")

    def test_default_fallback(self):
        preset = Preset(name="t", version=2, description="", pipelines={}, steps={}, models={})
        self.assertEqual(get_model(preset, "anything"), "sonnet")


if __name__ == "__main__":
    unittest.main()
