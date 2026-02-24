"""Locked state machine delegating to core for locking."""

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from forge.core.state import LockedStateManager


class StepStatus(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class StepState:
    status: StepStatus = StepStatus.PENDING
    retries: int = 0
    last_error: str = ""
    started_at: str = ""
    completed_at: str = ""


@dataclass
class PipelineState:
    phase: str = "execution"
    pipeline: str = ""
    preset: str = ""
    preset_path: str = ""
    model_profile: str = ""
    plan_file: str = ""
    current_step: str = ""
    steps: dict[str, StepState] = field(default_factory=dict)
    step_order: list[str] = field(default_factory=list)
    dependency_graph: dict[str, list[str]] = field(default_factory=dict)
    affected_packages: list[str] = field(default_factory=list)
    session_dir: str = ""
    dev_server_port: int = 0
    dev_server_pid: int = 0
    is_worktree: bool = False
    worktree_path: str = ""
    driver_pid: int = 0
    killed: bool = False
    kill_reason: str = ""
    created_at: str = ""
    updated_at: str = ""

    def step_names_ordered(self) -> list[str]:
        if self.step_order:
            return list(self.step_order)
        return list(self.steps.keys())

    def runnable_steps(self) -> list[str]:
        if not self.dependency_graph:
            for name in self.step_names_ordered():
                ss = self.steps[name]
                if ss.status in (StepStatus.PENDING, StepStatus.FAILED, StepStatus.IN_PROGRESS):
                    return [name]
            return []

        runnable = []
        for name in self.step_names_ordered():
            ss = self.steps[name]
            if ss.status not in (StepStatus.PENDING, StepStatus.FAILED):
                continue
            deps = self.dependency_graph.get(name, [])
            if all(self.steps[dep].status == StepStatus.COMPLETE for dep in deps):
                runnable.append(name)
        return runnable


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _state_to_dict(state: PipelineState) -> dict:
    d = {
        "phase": state.phase,
        "pipeline": state.pipeline,
        "preset": state.preset,
        "preset_path": state.preset_path,
        "model_profile": state.model_profile,
        "plan_file": state.plan_file,
        "current_step": state.current_step,
        "steps": {},
        "step_order": state.step_order,
        "dependency_graph": state.dependency_graph,
        "affected_packages": state.affected_packages,
        "session_dir": state.session_dir,
        "dev_server_port": state.dev_server_port,
        "dev_server_pid": state.dev_server_pid,
        "is_worktree": state.is_worktree,
        "worktree_path": state.worktree_path,
        "driver_pid": state.driver_pid,
        "killed": state.killed,
        "kill_reason": state.kill_reason,
        "created_at": state.created_at,
        "updated_at": state.updated_at,
    }
    for name, ss in state.steps.items():
        step_dict: dict = {"status": ss.status.value, "retries": ss.retries}
        if ss.last_error:
            step_dict["last_error"] = ss.last_error
        if ss.started_at:
            step_dict["started_at"] = ss.started_at
        if ss.completed_at:
            step_dict["completed_at"] = ss.completed_at
        d["steps"][name] = step_dict
    return d


def _dict_to_state(d: dict) -> PipelineState:
    steps_raw = d.get("steps", {})
    steps: dict[str, StepState] = {}

    for name, val in steps_raw.items():
        steps[name] = StepState(
            status=StepStatus(val.get("status", "pending")),
            retries=val.get("retries", 0),
            last_error=val.get("last_error", ""),
            started_at=val.get("started_at", ""),
            completed_at=val.get("completed_at", ""),
        )

    return PipelineState(
        phase=d.get("phase", "execution"),
        pipeline=d.get("pipeline", ""),
        preset=d.get("preset", ""),
        preset_path=d.get("preset_path", ""),
        model_profile=d.get("model_profile", ""),
        plan_file=d.get("plan_file", ""),
        current_step=d.get("current_step", ""),
        steps=steps,
        step_order=d.get("step_order", []),
        dependency_graph=d.get("dependency_graph", {}),
        affected_packages=d.get("affected_packages", []),
        session_dir=d.get("session_dir", ""),
        dev_server_port=d.get("dev_server_port", 0),
        dev_server_pid=d.get("dev_server_pid", 0),
        is_worktree=d.get("is_worktree", False),
        worktree_path=d.get("worktree_path", ""),
        driver_pid=d.get("driver_pid", 0),
        killed=d.get("killed", False),
        kill_reason=d.get("kill_reason", ""),
        created_at=d.get("created_at", ""),
        updated_at=d.get("updated_at", ""),
    )


class StateManager:
    """File-backed state with fcntl locking and atomic writes."""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self._mgr = LockedStateManager(state_file, _state_to_dict, _dict_to_state)

    def exists(self) -> bool:
        return self._mgr.exists()

    def load(self) -> PipelineState:
        return self._mgr.load()

    def save(self, state: PipelineState) -> None:
        state.updated_at = _now_iso()
        self._mgr.save(state)

    def update(self, mutator: Callable[[PipelineState], None]) -> PipelineState:
        def _with_timestamp(s: PipelineState) -> None:
            mutator(s)
            s.updated_at = _now_iso()
        return self._mgr.update(_with_timestamp)
