"""Checkpoint management — existence-based validation."""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .evidence import EvidenceResult


def write_checkpoint(
    checkpoint_dir: str,
    step_name: str,
    pipeline: str,
    evidence: Optional[EvidenceResult] = None,
) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        f"step={step_name}",
        f"passed_at={ts}",
        f"pipeline={pipeline}",
    ]

    if evidence and evidence.artifact_paths:
        for path in sorted(evidence.artifact_paths):
            lines.append(f"evidence:{path}")

    cp_path = os.path.join(checkpoint_dir, f"{step_name}.passed")
    Path(cp_path).write_text("\n".join(lines) + "\n")
    return cp_path


def verify_checkpoint(checkpoint_dir: str, step_name: str) -> tuple[bool, str]:
    cp_path = os.path.join(checkpoint_dir, f"{step_name}.passed")
    if not os.path.isfile(cp_path):
        return False, f"Checkpoint not found: {step_name}.passed"

    text = Path(cp_path).read_text().strip()
    lines = text.split("\n")

    for line in lines:
        if line.startswith("evidence:"):
            artifact_path = line[len("evidence:"):]
            if not os.path.isfile(artifact_path):
                return False, f"Evidence artifact missing: {artifact_path}"

    return True, f"Checkpoint '{step_name}' verified OK"


def is_manual_skip(checkpoint_dir: str, step_name: str) -> bool:
    """Return True if the checkpoint was created via manual skip."""
    cp_path = os.path.join(checkpoint_dir, f"{step_name}.passed")
    if not os.path.isfile(cp_path):
        return False
    return "manual_skip=true" in Path(cp_path).read_text()


def verify_all_checkpoints(checkpoint_dir: str, step_names: list[str],
                           exclude: Optional[set[str]] = None) -> tuple[bool, list[str], list[str]]:
    exclude = exclude or set()
    present = []
    missing = []

    for name in step_names:
        if name in exclude:
            continue
        valid, msg = verify_checkpoint(checkpoint_dir, name)
        if valid:
            present.append(name)
        else:
            missing.append(name)

    return len(missing) == 0, present, missing


def clear_checkpoints(checkpoint_dir: str) -> None:
    if os.path.isdir(checkpoint_dir):
        for f in os.listdir(checkpoint_dir):
            if f.endswith(".passed"):
                os.unlink(os.path.join(checkpoint_dir, f))
    os.makedirs(checkpoint_dir, exist_ok=True)
