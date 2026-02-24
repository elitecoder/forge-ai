# Copyright 2026. Tests for planner evidence validation.

import os

from forge.planner.engine.evidence import validate_phase


def _write_file(path: str, lines: int, headings: list[str] | None = None):
    content_lines = [f"line {i}" for i in range(lines)]
    if headings:
        for h in headings:
            content_lines.insert(0, f"# {h}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(content_lines))


def test_validate_recon_success(tmp_path):
    _write_file(str(tmp_path / "codebase-brief.md"), 35)
    result = validate_phase(str(tmp_path), "recon")
    assert result.passed
    assert "1 files validated" in result.message


def test_validate_recon_missing_file(tmp_path):
    result = validate_phase(str(tmp_path), "recon")
    assert not result.passed
    assert "Missing" in result.message


def test_validate_recon_too_short(tmp_path):
    _write_file(str(tmp_path / "codebase-brief.md"), 5)
    result = validate_phase(str(tmp_path), "recon")
    assert not result.passed
    assert "too short" in result.message


def test_validate_architects(tmp_path):
    _write_file(str(tmp_path / "design-a.md"), 25)
    _write_file(str(tmp_path / "design-b.md"), 25)
    result = validate_phase(str(tmp_path), "architects")
    assert result.passed


def test_validate_judge_requires_headings(tmp_path):
    _write_file(str(tmp_path / "final-plan.md"), 35)
    result = validate_phase(str(tmp_path), "judge")
    assert not result.passed
    assert "heading" in result.message


def test_validate_judge_success(tmp_path):
    _write_file(str(tmp_path / "final-plan.md"), 35, headings=["Decision", "Implementation Plan"])
    result = validate_phase(str(tmp_path), "judge")
    assert result.passed


def test_validate_unknown_phase(tmp_path):
    result = validate_phase(str(tmp_path), "unknown_phase")
    assert result.passed
    assert "No evidence rules" in result.message


def test_validate_enrichment_no_rules(tmp_path):
    result = validate_phase(str(tmp_path), "enrichment")
    assert result.passed
