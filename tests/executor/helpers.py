
"""Shared fixtures and helpers for pipeline engine tests."""

import json
import os
import shutil
import tempfile
from pathlib import Path

from forge.executor.engine.state import PipelineState, StepState, StepStatus, StateManager, _state_to_dict
from forge.executor.engine.registry import Preset, StepDefinition, EvidenceRule


def make_temp_dir() -> str:
    """Create a temporary directory. Caller must clean up."""
    return tempfile.mkdtemp(prefix="pipeline_test_")


SAMPLE_STEPS = ["code", "build_gen", "build", "lint", "test", "code_review", "report", "create_pr"]


def make_state(
    pipeline: str = "full",
    steps: list[str] | None = None,
    session_dir: str = "/tmp/test-session",
    packages: list[str] | None = None,
) -> PipelineState:
    """Create a PipelineState with the given steps, all pending."""
    step_names = steps or SAMPLE_STEPS
    return PipelineState(
        phase="execution",
        pipeline=pipeline,
        preset="test-preset",
        current_step=step_names[0],
        steps={name: StepState() for name in step_names},
        step_order=list(step_names),
        affected_packages=packages or [],
        session_dir=session_dir,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )


def save_state(state: PipelineState, state_file: Path) -> StateManager:
    """Save a state to disk and return the manager."""
    mgr = StateManager(state_file)
    mgr.save(state)
    return mgr


def make_preset(
    steps: dict[str, StepDefinition] | None = None,
    pipelines: dict[str, list[str]] | None = None,
    preset_dir: Path | None = None,
) -> Preset:
    """Create a minimal Preset for testing."""
    return Preset(
        name="test-preset",
        version=2,
        description="Test preset",
        pipelines=pipelines or {"full": SAMPLE_STEPS},
        steps=steps or {
            "build": StepDefinition(name="build", step_type="command", run_command="echo ok"),
            "lint": StepDefinition(name="lint", step_type="command", run_command="echo ok"),
        },
        models={"fix": "sonnet", "code_review": "opus"},
        preset_dir=preset_dir or Path("."),
    )


def make_manifest_dir(tmp: str) -> Path:
    """Create a temporary preset directory with a minimal manifest.json."""
    preset_dir = Path(tmp) / "presets" / "test-preset"
    preset_dir.mkdir(parents=True)
    manifest = {
        "preset": "test-preset",
        "version": 2,
        "pipelines": {
            "full": ["build", "lint"],
            "lightweight": ["lint"],
        },
        "models": {"fix": "sonnet"},
        "steps": {
            "build": {
                "type": "command",
                "run_command": "echo build ok",
                "timeout": 60,
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
