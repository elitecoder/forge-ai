"""Preset loading and step/model resolution."""

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvidenceRule:
    rule: str
    file_glob: str = ""
    max_age_seconds: int = 0
    min_file_size: int = 0
    schema: dict[str, Any] = field(default_factory=dict)
    field_name: str = ""
    expected_value: Any = None
    command: str = ""
    file: str = ""
    cwd: str = ""


@dataclass
class JudgeConfig:
    criteria_source: str = ""
    max_retries: int = 3
    model: str = "opus"


@dataclass
class StepDefinition:
    name: str
    step_type: str = "command"
    description: str = ""
    run_command: str = ""
    bazel_run_command: str = ""
    error_file: str = ""
    fix_hints: str = ""
    timeout: int = 600
    per_package: bool = False
    parallel: bool = False
    skill: str = ""
    subagent_type: str = "general-purpose"
    two_phase: bool = False
    judge: JudgeConfig | None = None
    evidence: list[EvidenceRule] = field(default_factory=list)
    allowed_file_patterns: list[str] | None = None


@dataclass
class PipelineDefinition:
    steps: list[str]
    dependencies: dict[str, list[str]]
    revalidation_targets: list[str] = field(default_factory=list)
    is_legacy: bool = False


@dataclass
class Preset:
    name: str
    version: int
    description: str
    pipelines: dict[str, PipelineDefinition]
    steps: dict[str, StepDefinition]
    models: dict[str, str]
    preset_dir: Path = field(default_factory=lambda: Path("."))
    base_ref: str = "main"
    build_command: str = ""
    bazel_build_command: str = ""
    eslint_config: str = ""
    dev_server: dict = field(default_factory=dict)
    visual_test_config: dict = field(default_factory=dict)


def _parse_evidence(rules_data: list[dict]) -> list[EvidenceRule]:
    result = []
    for rd in rules_data:
        result.append(EvidenceRule(
            rule=rd["rule"],
            file_glob=rd.get("file_glob", ""),
            max_age_seconds=rd.get("max_age_seconds", 0),
            min_file_size=rd.get("min_file_size", 0),
            schema=rd.get("schema", {}),
            field_name=rd.get("field_name", ""),
            expected_value=rd.get("expected_value"),
            command=rd.get("command", ""),
            file=rd.get("file", ""),
            cwd=rd.get("cwd", ""),
        ))
    return result


def _validate_dag(pipeline_name: str, pipeline_def: PipelineDefinition) -> None:
    step_set = set(pipeline_def.steps)
    for step, deps in pipeline_def.dependencies.items():
        if step not in step_set:
            raise ValueError(f"Pipeline '{pipeline_name}': dependency key '{step}' not in steps list")
        for dep in deps:
            if dep not in step_set:
                raise ValueError(f"Pipeline '{pipeline_name}': step '{step}' depends on unknown step '{dep}'")

    in_degree: dict[str, int] = {s: 0 for s in pipeline_def.steps}
    for step, deps in pipeline_def.dependencies.items():
        in_degree[step] = len(deps)

    queue = deque(s for s, d in in_degree.items() if d == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for step, deps in pipeline_def.dependencies.items():
            if node in deps:
                in_degree[step] -= 1
                if in_degree[step] == 0:
                    queue.append(step)

    if visited != len(pipeline_def.steps):
        raise ValueError(f"Pipeline '{pipeline_name}': dependency graph contains a cycle")


def _parse_pipeline(pipeline_name: str, raw: list | dict) -> PipelineDefinition:
    if isinstance(raw, list):
        return PipelineDefinition(steps=raw, dependencies={}, is_legacy=True)
    return PipelineDefinition(
        steps=raw["steps"],
        dependencies=raw.get("dependencies", {}),
        revalidation_targets=raw.get("revalidation_targets", []),
        is_legacy=False,
    )


def load_preset(preset_dir: Path) -> Preset:
    manifest_path = preset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    data = json.loads(manifest_path.read_text())

    judge_model = data.get("models", {}).get("judge", "opus")

    steps: dict[str, StepDefinition] = {}
    for name, sdata in data.get("steps", {}).items():
        judge_data = sdata.get("judge")
        judge = None
        if judge_data:
            judge = JudgeConfig(
                criteria_source=judge_data.get("criteria_source", ""),
                max_retries=judge_data.get("max_retries", 3),
                model=judge_data.get("model", judge_model),
            )
        steps[name] = StepDefinition(
            name=name,
            step_type=sdata.get("type", "command"),
            description=sdata.get("description", ""),
            run_command=sdata.get("run_command", ""),
            bazel_run_command=sdata.get("bazel_run_command", ""),
            error_file=sdata.get("error_file", ""),
            fix_hints=sdata.get("fix_hints", ""),
            timeout=sdata.get("timeout", 600),
            per_package=sdata.get("per_package", False),
            parallel=sdata.get("parallel", False),
            skill=sdata.get("skill", ""),
            subagent_type=sdata.get("subagent_type", "general-purpose"),
            two_phase=sdata.get("two_phase", False),
            judge=judge,
            evidence=_parse_evidence(sdata.get("evidence", [])),
            allowed_file_patterns=sdata.get("allowed_file_patterns"),
        )

    pipelines: dict[str, PipelineDefinition] = {}
    for pipeline_name, raw in data.get("pipelines", {}).items():
        pipeline_def = _parse_pipeline(pipeline_name, raw)
        for step_name in pipeline_def.steps:
            if step_name not in steps:
                raise ValueError(f"Pipeline '{pipeline_name}' references unknown step '{step_name}'")
        if not pipeline_def.is_legacy:
            _validate_dag(pipeline_name, pipeline_def)
        pipelines[pipeline_name] = pipeline_def

    return Preset(
        name=data["preset"],
        version=data.get("version", 2),
        description=data.get("description", ""),
        pipelines=pipelines,
        steps=steps,
        models=data.get("models", {"fix": "sonnet"}),
        preset_dir=preset_dir,
        base_ref=data.get("base_ref", "main"),
        build_command=data.get("build_command", ""),
        bazel_build_command=data.get("bazel_build_command", ""),
        eslint_config=data.get("eslint_config", ""),
        dev_server=data.get("dev_server", {}),
        visual_test_config=data.get("visual_test_config", {}),
    )


def get_model(preset: Preset, step_name: str, is_fix: bool = False) -> str:
    if is_fix:
        return preset.models.get("fix", "sonnet")
    return preset.models.get(step_name, preset.models.get("fix", "sonnet"))
