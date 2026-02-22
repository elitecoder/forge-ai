# Copyright 2026. Planner engine — config-driven evidence validation for phase outputs.

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvidenceResult:
    passed: bool
    message: str


_CONFIG_FILE = Path(__file__).resolve().parent.parent / "config" / "evidence.json"
_rules_cache: dict | None = None


def _load_rules() -> dict:
    global _rules_cache
    if _rules_cache is not None:
        return _rules_cache
    if _CONFIG_FILE.is_file():
        _rules_cache = json.loads(_CONFIG_FILE.read_text())
    else:
        _rules_cache = {}
    return _rules_cache


def validate_phase(session_dir: str, phase: str) -> EvidenceResult:
    """Validate that a phase produced well-formed output files."""
    rules = _load_rules()
    phase_rules = rules.get(phase)
    if not phase_rules:
        return EvidenceResult(True, f"No evidence rules for phase '{phase}'")

    checked = 0
    for rule in phase_rules:
        rule_type = rule.get("rule", "")
        filename = rule.get("file_glob", "")
        path = os.path.join(session_dir, filename)

        if rule_type == "file_exists":
            if not os.path.isfile(path):
                return EvidenceResult(False, f"Missing output file: {filename}")
            min_lines = rule.get("min_lines", 0)
            if min_lines > 0:
                content = Path(path).read_text(encoding="utf-8", errors="replace")
                lines = content.strip().split("\n")
                if len(lines) < min_lines:
                    return EvidenceResult(
                        False,
                        f"File '{filename}' too short ({len(lines)} lines, minimum {min_lines})",
                    )
            checked += 1

        elif rule_type == "file_contains_heading":
            if not os.path.isfile(path):
                return EvidenceResult(False, f"Missing output file: {filename}")
            content = Path(path).read_text(encoding="utf-8", errors="replace")
            lines = content.strip().split("\n")
            heading_lines = [l for l in lines if l.strip().startswith("#")]
            for heading in rule.get("headings", []):
                if not any(heading in hl.lower() for hl in heading_lines):
                    return EvidenceResult(
                        False,
                        f"File '{filename}' missing required heading containing '{heading}'",
                    )
            checked += 1

    file_count = len({r.get("file_glob") for r in phase_rules if r.get("file_glob")})
    return EvidenceResult(True, f"All {file_count} files validated")
