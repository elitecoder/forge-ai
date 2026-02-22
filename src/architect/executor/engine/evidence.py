"""Declarative evidence validation for pipeline steps."""

import glob
import json
import os
import re
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .registry import EvidenceRule, StepDefinition


@dataclass
class EvidenceResult:
    passed: bool
    message: str
    artifact_paths: list[str] = field(default_factory=list)


def _file_age_seconds(path: str) -> float:
    return time.time() - os.path.getmtime(path)


def _check_file_exists(rule: EvidenceRule, session_dir: str) -> EvidenceResult:
    pattern = os.path.join(session_dir, rule.file_glob)
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not matches:
        return EvidenceResult(False, f"No files matching '{rule.file_glob}' in {session_dir}")

    latest = matches[0]

    if rule.max_age_seconds > 0:
        age = _file_age_seconds(latest)
        if age > rule.max_age_seconds:
            return EvidenceResult(
                False,
                f"File '{os.path.basename(latest)}' is stale ({int(age / 60)} min old, max {rule.max_age_seconds // 60} min)",
                [latest],
            )

    if rule.min_file_size > 0:
        size = os.path.getsize(latest)
        if size < rule.min_file_size:
            return EvidenceResult(
                False,
                f"File '{os.path.basename(latest)}' is {size} bytes, minimum {rule.min_file_size} bytes",
                [latest],
            )

    return EvidenceResult(True, f"Found: {os.path.basename(latest)}", [latest])


def _check_json_schema(rule: EvidenceRule, session_dir: str) -> EvidenceResult:
    import jsonschema as _jsonschema

    pattern = os.path.join(session_dir, rule.file_glob)
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not matches:
        return EvidenceResult(False, f"No files matching '{rule.file_glob}' for schema validation")

    latest = matches[0]
    paths = [latest]

    try:
        data = json.loads(Path(latest).read_text())
    except (json.JSONDecodeError, OSError) as e:
        return EvidenceResult(False, f"Cannot parse '{os.path.basename(latest)}': {e}", paths)

    schema = rule.schema

    if "type" in schema:
        try:
            _jsonschema.validate(instance=data, schema=schema)
        except _jsonschema.ValidationError as e:
            return EvidenceResult(False, f"Schema validation failed: {e.message}", paths)

    for field_name in schema.get("required_fields", []):
        if field_name not in data:
            return EvidenceResult(False, f"Missing required field '{field_name}' in {os.path.basename(latest)}", paths)

    for field_name, expected in schema.get("field_values", {}).items():
        actual = data.get(field_name)
        if actual != expected:
            return EvidenceResult(
                False,
                f"Field '{field_name}' expected {expected}, got {actual} in {os.path.basename(latest)}",
                paths,
            )

    summary = data.get("summary", {})
    for field_name, constraints in schema.get("summary_constraints", {}).items():
        val = summary.get(field_name, 0)
        if isinstance(constraints, dict):
            if "gt" in constraints and not (val > constraints["gt"]):
                return EvidenceResult(False, f"summary.{field_name} must be > {constraints['gt']}, got {val}", paths)
            if "gte" in constraints and not (val >= constraints["gte"]):
                return EvidenceResult(False, f"summary.{field_name} must be >= {constraints['gte']}, got {val}", paths)

    min_results = schema.get("min_results", 0)
    results = data.get("results", [])
    if min_results > 0 and len(results) < min_results:
        return EvidenceResult(
            False,
            f"Expected at least {min_results} results, got {len(results)}",
            paths,
        )

    summary_total = data.get("summary", {}).get("total")
    if summary_total is not None and "results" in data and summary_total != len(results):
        return EvidenceResult(
            False,
            f"summary.total ({summary_total}) != len(results) ({len(results)})",
            paths,
        )

    required_status = schema.get("results_all_status")
    if required_status:
        if not results:
            return EvidenceResult(False, f"No results to validate against status '{required_status}'", paths)
        failures = [r for r in results if r.get("status") != required_status]
        if failures:
            fail_details = "; ".join(
                f"S{r.get('number', '?')}: {r.get('title', '?')} — {r.get('status', '?')}"
                for r in failures[:5]
            )
            return EvidenceResult(
                False,
                f"{len(failures)} result(s) not {required_status}: {fail_details}",
                paths,
            )

    entry_fields = schema.get("results_entry_required_fields", [])
    if entry_fields:
        results = data.get("results", [])
        for i, entry in enumerate(results):
            for ef in entry_fields:
                if ef not in entry:
                    return EvidenceResult(False, f"results[{i}] missing field '{ef}'", paths)

    forbidden = schema.get("forbidden_title_patterns", [])
    if forbidden:
        for entry in data.get("results", []):
            title = entry.get("title", "")
            for pattern in forbidden:
                if re.search(pattern, title, re.IGNORECASE):
                    return EvidenceResult(
                        False,
                        f"Result title '{title}' matches forbidden pattern '{pattern}'",
                        paths,
                    )

    return EvidenceResult(True, f"Schema OK: {os.path.basename(latest)}", paths)


def _check_json_field_equals(rule: EvidenceRule, session_dir: str) -> EvidenceResult:
    target = rule.file or rule.file_glob
    if not target:
        return EvidenceResult(False, "json_field_equals: no file specified")

    path = os.path.join(session_dir, target) if not os.path.isabs(target) else target
    matches = glob.glob(path) if "*" in path or "?" in path else ([path] if os.path.isfile(path) else [])
    if not matches:
        return EvidenceResult(False, f"File not found: {target}")

    latest = matches[0]

    try:
        data = json.loads(Path(latest).read_text())
    except (json.JSONDecodeError, OSError) as e:
        return EvidenceResult(False, f"Cannot parse '{os.path.basename(latest)}': {e}", [latest])

    actual = data.get(rule.field_name)
    if actual != rule.expected_value:
        return EvidenceResult(
            False,
            f"Field '{rule.field_name}' expected {rule.expected_value!r}, got {actual!r}",
            [latest],
        )
    return EvidenceResult(True, f"Field '{rule.field_name}' == {rule.expected_value!r}", [latest])


def _check_command_succeeds(rule: EvidenceRule, session_dir: str) -> EvidenceResult:
    from .utils import repo_root
    cwd = repo_root() if rule.cwd == "repo_root" else session_dir
    try:
        result = subprocess.run(
            rule.command, shell=True, capture_output=True, timeout=120, cwd=cwd,
        )
        if result.returncode == 0:
            return EvidenceResult(True, f"Command succeeded: {rule.command}")
        return EvidenceResult(
            False,
            f"Command failed (exit {result.returncode}): {rule.command}\n{result.stderr.decode()[:500]}",
        )
    except subprocess.TimeoutExpired:
        return EvidenceResult(False, f"Command timed out: {rule.command}")


def _transitive_deps(step_name: str, graph: dict[str, list[str]]) -> set[str]:
    visited: set[str] = set()
    queue = deque(graph.get(step_name, []))
    while queue:
        dep = queue.popleft()
        if dep in visited:
            continue
        visited.add(dep)
        queue.extend(graph.get(dep, []))
    return visited


def _check_all_predecessors(rule: EvidenceRule, step_name: str, state_steps: dict,
                            checkpoint_dir: str,
                            dependency_graph: dict[str, list[str]] | None = None) -> EvidenceResult:
    from .checkpoint import verify_checkpoint

    if dependency_graph:
        predecessors = sorted(_transitive_deps(step_name, dependency_graph))
    else:
        step_names = list(state_steps.keys())
        if step_name not in step_names:
            return EvidenceResult(False, f"Step '{step_name}' not found in pipeline")
        idx = step_names.index(step_name)
        predecessors = step_names[:idx]

    missing = []
    for prev_name in predecessors:
        valid, msg = verify_checkpoint(checkpoint_dir, prev_name)
        if not valid:
            missing.append(prev_name)

    if missing:
        return EvidenceResult(False, f"Missing/invalid checkpoints for: {', '.join(missing)}")
    return EvidenceResult(True, f"All {len(predecessors)} predecessors checkpointed and verified")


_RULE_HANDLERS = {
    "file_exists": _check_file_exists,
    "json_schema": _check_json_schema,
    "json_field_equals": _check_json_field_equals,
    "command_succeeds": _check_command_succeeds,
}


class EvidenceChecker:
    def __init__(self, session_dir: str, checkpoint_dir: str, state_steps: Optional[dict] = None,
                 dependency_graph: dict[str, list[str]] | None = None):
        self.session_dir = session_dir
        self.checkpoint_dir = checkpoint_dir
        self.state_steps = state_steps or {}
        self.dependency_graph = dependency_graph

    def check(self, step: StepDefinition) -> EvidenceResult:
        all_paths: list[str] = []

        for rule in step.evidence:
            if rule.rule == "all_predecessors_checkpointed":
                result = _check_all_predecessors(
                    rule, step.name, self.state_steps, self.checkpoint_dir, self.dependency_graph,
                )
            else:
                handler = _RULE_HANDLERS.get(rule.rule)
                if not handler:
                    return EvidenceResult(False, f"Unknown evidence rule: {rule.rule}")
                result = handler(rule, self.session_dir)

            if not result.passed:
                return result
            all_paths.extend(result.artifact_paths)

        return EvidenceResult(True, f"All {len(step.evidence)} evidence rules passed", all_paths)
