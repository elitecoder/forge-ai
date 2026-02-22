"""Fail-closed pre-PR gate — blocks push/PR unless all pipeline steps verified."""

import os
from pathlib import Path

from .engine.state import StateManager
from .engine.checkpoint import verify_all_checkpoints, is_manual_skip
from .engine.utils import find_active_session

# Steps that MUST have real evidence — manual_skip is not acceptable
CRITICAL_STEPS = {"test", "visual_test"}


def main() -> int:
    session_dir = find_active_session()

    if session_dir is None:
        print("PRE-PR GATE: No pipeline detected — passing.")
        return 0

    state_file = session_dir / "agent-state.json"
    chk_dir = str(session_dir / "checkpoints")

    has_state = state_file.is_file()
    has_checkpoints = os.path.isdir(chk_dir) and any(
        f.endswith(".passed") for f in os.listdir(chk_dir)
    )

    if not has_state and has_checkpoints:
        print("=" * 42)
        print("  PRE-PR GATE FAILED")
        print("=" * 42)
        print()
        print("Checkpoint files exist but agent-state.json is missing.")
        print("This suggests a pipeline was active but the state was deleted.")
        print("Restore the state file or clean up checkpoints before pushing.")
        return 1

    if not has_state:
        print("PRE-PR GATE: No pipeline detected — passing.")
        return 0

    mgr = StateManager(state_file)
    state = mgr.load()
    step_names = state.step_names_ordered()

    all_valid, present, missing = verify_all_checkpoints(
        chk_dir, step_names, exclude={"create_pr"}
    )

    print(f"Pipeline type: {state.pipeline}")
    print(f"Checkpoints verified: {len(present)}/{len(step_names) - 1}")

    if not all_valid:
        print()
        print("=" * 42)
        print("  PRE-PR GATE FAILED")
        print("=" * 42)
        print()
        print(f"Missing/invalid checkpoints ({len(missing)}):")
        for m in missing:
            print(f"  MISSING: {m}.passed")
        if present:
            print()
            print(f"Valid checkpoints ({len(present)}):")
            for p in present:
                print(f"  OK:      {p}.passed")
        print()
        print("You MUST complete ALL pipeline steps before pushing or creating a PR.")
        return 1

    # Check critical steps for manual_skip
    skipped_critical = [s for s in present if s in CRITICAL_STEPS and is_manual_skip(chk_dir, s)]
    if skipped_critical:
        print()
        print("=" * 42)
        print("  PRE-PR GATE FAILED")
        print("=" * 42)
        print()
        for s in skipped_critical:
            print(f"  SKIPPED: {s} was manually skipped — no real evidence")
        skipped_non_critical = [s for s in present if s not in CRITICAL_STEPS and is_manual_skip(chk_dir, s)]
        for s in skipped_non_critical:
            print(f"  WARNING: {s} was manually skipped")
        print()
        print("Critical steps must run with real evidence. Re-run the pipeline.")
        return 1

    # Warn about non-critical manual skips
    for s in present:
        if s not in CRITICAL_STEPS and is_manual_skip(chk_dir, s):
            print(f"  WARNING: {s} was manually skipped")

    print()
    print("=" * 42)
    print("  PRE-PR GATE PASSED")
    print("=" * 42)
    print()
    print(f"All {len(present)} steps verified:")
    for p in present:
        print(f"  OK: {p}.passed")
    print()
    print("Safe to push and create PR.")
    return 0


def run_gate() -> bool:
    """Run the pre-PR gate. Returns True if passed."""
    return main() == 0
