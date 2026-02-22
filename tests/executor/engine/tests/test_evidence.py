"""Tests for evidence.py — _transitive_deps, DAG-aware predecessor checking, schema validation."""

import json
import os
import subprocess
import tempfile
import shutil
import pytest
from pathlib import Path
from unittest.mock import patch

from architect.executor.engine.evidence import (
    _transitive_deps, _check_all_predecessors, _check_json_schema,
    _check_file_exists, _check_json_field_equals, _check_command_succeeds,
    EvidenceChecker,
)
from architect.executor.engine.registry import EvidenceRule, StepDefinition


class TestTransitiveDeps:
    def test_no_deps(self):
        graph = {"b": ["a"]}
        assert _transitive_deps("a", graph) == set()

    def test_direct_deps(self):
        graph = {"c": ["a", "b"]}
        assert _transitive_deps("c", graph) == {"a", "b"}

    def test_transitive(self):
        graph = {"b": ["a"], "c": ["b"]}
        assert _transitive_deps("c", graph) == {"a", "b"}

    def test_diamond(self):
        graph = {
            "left": ["root"],
            "right": ["root"],
            "sink": ["left", "right"],
        }
        assert _transitive_deps("sink", graph) == {"left", "right", "root"}

    def test_deep_chain(self):
        graph = {"b": ["a"], "c": ["b"], "d": ["c"]}
        assert _transitive_deps("d", graph) == {"a", "b", "c"}

    def test_step_not_in_graph(self):
        graph = {"b": ["a"]}
        assert _transitive_deps("unknown", graph) == set()


class TestCheckAllPredecessors:
    def test_positional_fallback(self):
        """Without dependency_graph, falls back to positional check."""
        rule = EvidenceRule(rule="all_predecessors_checkpointed")
        steps = {"a": None, "b": None, "c": None}

        with patch("architect.executor.engine.checkpoint.verify_checkpoint") as mock_verify:
            mock_verify.return_value = (True, "OK")
            result = _check_all_predecessors(rule, "c", steps, "/checkpoints")
            assert result.passed
            # Should have checked a and b (predecessors of c)
            assert mock_verify.call_count == 2

    def test_dag_mode_checks_only_transitive_deps(self):
        """With dependency_graph, checks only transitive deps."""
        rule = EvidenceRule(rule="all_predecessors_checkpointed")
        steps = {"a": None, "b": None, "c": None, "d": None}
        # d depends on c which depends on a. b is unrelated.
        graph = {"c": ["a"], "d": ["c"]}

        with patch("architect.executor.engine.checkpoint.verify_checkpoint") as mock_verify:
            mock_verify.return_value = (True, "OK")
            result = _check_all_predecessors(rule, "d", steps, "/checkpoints", graph)
            assert result.passed
            # Should check a and c (transitive deps of d), NOT b
            checked = {call.args[1] for call in mock_verify.call_args_list}
            assert checked == {"a", "c"}

    def test_dag_mode_reports_missing(self):
        rule = EvidenceRule(rule="all_predecessors_checkpointed")
        steps = {"a": None, "b": None, "c": None}
        graph = {"c": ["a", "b"]}

        with patch("architect.executor.engine.checkpoint.verify_checkpoint") as mock_verify:
            mock_verify.side_effect = lambda d, name: (False, "missing") if name == "b" else (True, "OK")
            result = _check_all_predecessors(rule, "c", steps, "/checkpoints", graph)
            assert not result.passed
            assert "b" in result.message


class TestEvidenceCheckerWithGraph:
    def test_passes_graph_to_predecessors_check(self):
        graph = {"b": ["a"]}
        checker = EvidenceChecker("/session", "/checkpoints", {"a": None, "b": None}, graph)

        step = StepDefinition(
            name="b",
            evidence=[EvidenceRule(rule="all_predecessors_checkpointed")],
        )

        with patch("architect.executor.engine.evidence._check_all_predecessors") as mock_check:
            mock_check.return_value = type("R", (), {"passed": True, "message": "OK", "artifact_paths": []})()
            checker.check(step)
            _, kwargs = mock_check.call_args
            assert kwargs.get("dependency_graph") is None  # passed as positional
            # Check positional args instead
            args = mock_check.call_args.args
            assert args[4] == graph  # 5th arg is dependency_graph


# ── Fix 8: min_results + summary cross-validation ──────────────────────────

class TestJsonSchemaMinResults:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="evidence_min_test_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def _write_json(self, filename, data):
        Path(os.path.join(self.tmp, filename)).write_text(json.dumps(data))

    def test_empty_results_fails_with_min_results(self):
        self._write_json("results.json", {
            "schemaVersion": 1, "feature": "test", "timestamp": "now",
            "summary": {"total": 0}, "results": [],
        })
        rule = EvidenceRule(rule="json_schema", file_glob="results.json",
                            schema={"min_results": 2})

        result = _check_json_schema(rule, self.tmp)

        assert not result.passed
        assert "at least 2" in result.message

    def test_sufficient_results_passes(self):
        self._write_json("results.json", {
            "summary": {"total": 3},
            "results": [{"status": "PASS"}, {"status": "PASS"}, {"status": "PASS"}],
        })
        rule = EvidenceRule(rule="json_schema", file_glob="results.json",
                            schema={"min_results": 2})

        result = _check_json_schema(rule, self.tmp)

        assert result.passed

    def test_summary_total_mismatch_fails(self):
        self._write_json("results.json", {
            "summary": {"total": 5},
            "results": [{"status": "PASS"}, {"status": "PASS"}],
        })
        rule = EvidenceRule(rule="json_schema", file_glob="results.json", schema={})

        result = _check_json_schema(rule, self.tmp)

        assert not result.passed
        assert "summary.total" in result.message

    def test_summary_total_matches_passes(self):
        self._write_json("results.json", {
            "summary": {"total": 2},
            "results": [{"status": "PASS"}, {"status": "PASS"}],
        })
        rule = EvidenceRule(rule="json_schema", file_glob="results.json", schema={})

        result = _check_json_schema(rule, self.tmp)

        assert result.passed

    def test_empty_results_with_required_status_fails(self):
        """results_all_status on empty results should fail, not vacuously pass."""
        self._write_json("results.json", {
            "summary": {"total": 0}, "results": [],
        })
        rule = EvidenceRule(rule="json_schema", file_glob="results.json",
                            schema={"results_all_status": "PASS"})

        result = _check_json_schema(rule, self.tmp)

        assert not result.passed
        assert "No results" in result.message


# ── Uncovered lines: min_file_size, gte, entry fields, forbidden titles, etc. ──


class TestFileExistsMinFileSize:
    """Lines 62-64: file exists but is smaller than rule.min_file_size."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="evidence_minsize_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_file_too_small(self):
        Path(os.path.join(self.tmp, "tiny.json")).write_text("{}")
        rule = EvidenceRule(rule="file_exists", file_glob="tiny.json", min_file_size=1000)

        result = _check_file_exists(rule, self.tmp)

        assert not result.passed
        assert "minimum 1000 bytes" in result.message
        assert len(result.artifact_paths) > 0

    def test_file_large_enough(self):
        Path(os.path.join(self.tmp, "big.json")).write_text("x" * 500)
        rule = EvidenceRule(rule="file_exists", file_glob="big.json", min_file_size=100)

        result = _check_file_exists(rule, self.tmp)

        assert result.passed


class TestJsonSchemaGteConstraint:
    """Line 113: summary_constraints with gte that fails."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="evidence_gte_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def _write_json(self, filename, data):
        Path(os.path.join(self.tmp, filename)).write_text(json.dumps(data))

    def test_gte_fails(self):
        self._write_json("r.json", {"summary": {"total": 3}})
        rule = EvidenceRule(rule="json_schema", file_glob="r.json",
                            schema={"summary_constraints": {"total": {"gte": 5}}})

        result = _check_json_schema(rule, self.tmp)

        assert not result.passed
        assert "must be >= 5" in result.message
        assert "got 3" in result.message

    def test_gte_passes_equal(self):
        self._write_json("r.json", {"summary": {"total": 5}})
        rule = EvidenceRule(rule="json_schema", file_glob="r.json",
                            schema={"summary_constraints": {"total": {"gte": 5}}})

        result = _check_json_schema(rule, self.tmp)

        assert result.passed

    def test_gte_passes_above(self):
        self._write_json("r.json", {"summary": {"total": 10}})
        rule = EvidenceRule(rule="json_schema", file_glob="r.json",
                            schema={"summary_constraints": {"total": {"gte": 5}}})

        result = _check_json_schema(rule, self.tmp)

        assert result.passed


class TestJsonSchemaResultsEntryRequiredFields:
    """Lines 154-158: results_entry_required_fields check."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="evidence_entry_fields_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def _write_json(self, filename, data):
        Path(os.path.join(self.tmp, filename)).write_text(json.dumps(data))

    def test_missing_required_field_in_result_entry(self):
        self._write_json("r.json", {
            "results": [
                {"title": "Test 1", "status": "PASS"},
                {"title": "Test 2"},
            ],
        })
        rule = EvidenceRule(rule="json_schema", file_glob="r.json",
                            schema={"results_entry_required_fields": ["title", "status"]})

        result = _check_json_schema(rule, self.tmp)

        assert not result.passed
        assert "results[1] missing field 'status'" in result.message

    def test_all_entry_fields_present(self):
        self._write_json("r.json", {
            "results": [
                {"title": "Test 1", "status": "PASS"},
                {"title": "Test 2", "status": "FAIL"},
            ],
        })
        rule = EvidenceRule(rule="json_schema", file_glob="r.json",
                            schema={"results_entry_required_fields": ["title", "status"]})

        result = _check_json_schema(rule, self.tmp)

        assert result.passed


class TestJsonSchemaForbiddenTitlePatterns:
    """Lines 163-167: forbidden_title_patterns check."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="evidence_forbidden_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def _write_json(self, filename, data):
        Path(os.path.join(self.tmp, filename)).write_text(json.dumps(data))

    def test_forbidden_pattern_matches(self):
        self._write_json("r.json", {
            "results": [
                {"title": "TODO: fix this later", "status": "PASS"},
            ],
        })
        rule = EvidenceRule(rule="json_schema", file_glob="r.json",
                            schema={"forbidden_title_patterns": ["TODO"]})

        result = _check_json_schema(rule, self.tmp)

        assert not result.passed
        assert "forbidden pattern" in result.message
        assert "TODO" in result.message

    def test_no_forbidden_pattern_match(self):
        self._write_json("r.json", {
            "results": [
                {"title": "Legit test name", "status": "PASS"},
            ],
        })
        rule = EvidenceRule(rule="json_schema", file_glob="r.json",
                            schema={"forbidden_title_patterns": ["TODO", "FIXME"]})

        result = _check_json_schema(rule, self.tmp)

        assert result.passed


class TestJsonFieldEqualsFileNotFound:
    """Line 185: fallback when no file is found for json_field_equals."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="evidence_nf_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_file_not_found(self):
        rule = EvidenceRule(rule="json_field_equals", file="nonexistent.json",
                            field_name="status", expected_value="ok")

        result = _check_json_field_equals(rule, self.tmp)

        assert not result.passed
        assert "File not found: nonexistent.json" in result.message


class TestJsonFieldEqualsParseError:
    """Lines 192-193: JSON parse error in json_field_equals."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="evidence_parse_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_invalid_json(self):
        Path(os.path.join(self.tmp, "bad.json")).write_text("not valid json {{{")
        rule = EvidenceRule(rule="json_field_equals", file="bad.json",
                            field_name="status", expected_value="ok")

        result = _check_json_field_equals(rule, self.tmp)

        assert not result.passed
        assert "Cannot parse" in result.message
        assert len(result.artifact_paths) > 0


class TestCommandSucceedsTimeout:
    """Lines 219-220: subprocess.TimeoutExpired exception path."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp(prefix="evidence_timeout_")

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def test_timeout_expired(self):
        rule = EvidenceRule(rule="command_succeeds", command="sleep 999")

        with patch("architect.executor.engine.evidence.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="sleep 999", timeout=120)
            result = _check_command_succeeds(rule, self.tmp)

        assert not result.passed
        assert "timed out" in result.message
