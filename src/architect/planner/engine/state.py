# Copyright 2026. Planner engine — state machine delegating to core for locking.

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from architect.core.state import LockedStateManager


class PhaseStatus(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


PHASE_ORDER = ["recon", "architects", "critics", "refiners", "judge", "enrichment"]

TERMINAL_STATUSES = {PhaseStatus.COMPLETE, PhaseStatus.SKIPPED}


@dataclass
class PhaseState:
    status: PhaseStatus = PhaseStatus.PENDING
    retries: int = 0
    started_at: str = ""
    completed_at: str = ""
    last_error: str = ""


@dataclass
class PlannerState:
    slug: str = ""
    session_dir: str = ""
    repo_dir: str = ""
    fast_mode: bool = False
    problem_statement: str = ""
    core_tension: str = ""
    constraint_a: str = ""
    constraint_b: str = ""
    phases: dict[str, PhaseState] = field(default_factory=dict)
    killed: bool = False
    kill_reason: str = ""
    driver_pid: int = 0
    created_at: str = ""
    updated_at: str = ""

    def next_runnable(self) -> str | None:
        for phase in PHASE_ORDER:
            ps = self.phases.get(phase)
            if not ps:
                return phase
            if ps.status in (PhaseStatus.PENDING, PhaseStatus.FAILED):
                if all(
                    self.phases.get(p, PhaseState()).status in TERMINAL_STATUSES
                    for p in PHASE_ORDER[:PHASE_ORDER.index(phase)]
                ):
                    return phase
            if ps.status == PhaseStatus.IN_PROGRESS:
                return phase
        return None


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def state_to_dict(state: PlannerState) -> dict:
    d = {
        "slug": state.slug,
        "session_dir": state.session_dir,
        "repo_dir": state.repo_dir,
        "fast_mode": state.fast_mode,
        "problem_statement": state.problem_statement,
        "core_tension": state.core_tension,
        "constraint_a": state.constraint_a,
        "constraint_b": state.constraint_b,
        "phases": {},
        "killed": state.killed,
        "kill_reason": state.kill_reason,
        "driver_pid": state.driver_pid,
        "created_at": state.created_at,
        "updated_at": state.updated_at,
    }
    for name, ps in state.phases.items():
        pd: dict = {"status": ps.status.value, "retries": ps.retries}
        if ps.started_at:
            pd["started_at"] = ps.started_at
        if ps.completed_at:
            pd["completed_at"] = ps.completed_at
        if ps.last_error:
            pd["last_error"] = ps.last_error
        d["phases"][name] = pd
    return d


def _parse_status(raw: str) -> PhaseStatus:
    try:
        return PhaseStatus(raw)
    except ValueError:
        return PhaseStatus.FAILED


def dict_to_state(d: dict) -> PlannerState:
    phases = {}
    for name, pd in d.get("phases", {}).items():
        phases[name] = PhaseState(
            status=_parse_status(pd.get("status", "pending")),
            retries=pd.get("retries", 0),
            started_at=pd.get("started_at", ""),
            completed_at=pd.get("completed_at", ""),
            last_error=pd.get("last_error", ""),
        )
    return PlannerState(
        slug=d.get("slug", ""),
        session_dir=d.get("session_dir", ""),
        repo_dir=d.get("repo_dir", ""),
        fast_mode=d.get("fast_mode", False),
        problem_statement=d.get("problem_statement", ""),
        core_tension=d.get("core_tension", ""),
        constraint_a=d.get("constraint_a", ""),
        constraint_b=d.get("constraint_b", ""),
        phases=phases,
        killed=d.get("killed", False),
        kill_reason=d.get("kill_reason", ""),
        driver_pid=d.get("driver_pid", 0),
        created_at=d.get("created_at", ""),
        updated_at=d.get("updated_at", ""),
    )


class StateManager:
    """File-backed state with fcntl locking and atomic writes."""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self._mgr = LockedStateManager(state_file, state_to_dict, dict_to_state)

    def exists(self) -> bool:
        return self._mgr.exists()

    def load(self) -> PlannerState:
        return self._mgr.load()

    def save(self, state: PlannerState) -> None:
        state.updated_at = now_iso()
        self._mgr.save(state)

    def update(self, mutator: Callable[[PlannerState], None]) -> PlannerState:
        def _with_timestamp(s: PlannerState) -> None:
            mutator(s)
            s.updated_at = now_iso()
        return self._mgr.update(_with_timestamp)
