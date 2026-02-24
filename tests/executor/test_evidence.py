
"""Tests for engine.evidence — all 5 rule handlers + EvidenceChecker."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from forge.executor.engine.evidence import (
    _check_file_exists, _check_json_schema, _check_json_field_equals,
    _check_command_succeeds, _check_all_predecessors, EvidenceChecker, EvidenceResult,
)
from forge.executor.engine.registry import EvidenceRule, StepDefinition
from forge.executor.engine.checkpoint import write_checkpoint
from forge.executor.engine.state import StepState, StepStatus


class TestCheckFileExists(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ev_file_")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_file_found(self):
        Path(self.tmp, "report.json").write_text("{}")
        rule = EvidenceRule(rule="file_exists", file_glob="report.json")
        result = _check_file_exists(rule, self.tmp)
        self.assertTrue(result.passed)
        self.assertTrue(len(result.artifact_paths) > 0)

    def test_file_not_found(self):
        rule = EvidenceRule(rule="file_exists", file_glob="missing.json")
        result = _check_file_exists(rule, self.tmp)
        self.assertFalse(result.passed)

    def test_glob_pattern(self):
        Path(self.tmp, "test-results.json").write_text("{}")
        rule = EvidenceRule(rule="file_exists", file_glob="*-results.json")
        result = _check_file_exists(rule, self.tmp)
        self.assertTrue(result.passed)

    def test_stale_file(self):
        p = Path(self.tmp, "old.json")
        p.write_text("{}")
        # Set mtime to 2 hours ago
        old_time = os.path.getmtime(str(p)) - 7200
        os.utime(str(p), (old_time, old_time))
        rule = EvidenceRule(rule="file_exists", file_glob="old.json", max_age_seconds=60)
        result = _check_file_exists(rule, self.tmp)
        self.assertFalse(result.passed)
        self.assertIn("stale", result.message)

    def test_fresh_file(self):
        Path(self.tmp, "new.json").write_text("{}")
        rule = EvidenceRule(rule="file_exists", file_glob="new.json", max_age_seconds=3600)
        result = _check_file_exists(rule, self.tmp)
        self.assertTrue(result.passed)


class TestCheckJsonSchema(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ev_schema_")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_required_fields_present(self):
        Path(self.tmp, "r.json").write_text(json.dumps({"a": 1, "b": 2}))
        rule = EvidenceRule(rule="json_schema", file_glob="r.json",
                           schema={"required_fields": ["a", "b"]})
        result = _check_json_schema(rule, self.tmp)
        self.assertTrue(result.passed)

    def test_required_field_missing(self):
        Path(self.tmp, "r.json").write_text(json.dumps({"a": 1}))
        rule = EvidenceRule(rule="json_schema", file_glob="r.json",
                           schema={"required_fields": ["a", "b"]})
        result = _check_json_schema(rule, self.tmp)
        self.assertFalse(result.passed)
        self.assertIn("Missing required field 'b'", result.message)

    def test_field_value_check(self):
        Path(self.tmp, "r.json").write_text(json.dumps({"version": 1}))
        rule = EvidenceRule(rule="json_schema", file_glob="r.json",
                           schema={"field_values": {"version": 1}})
        result = _check_json_schema(rule, self.tmp)
        self.assertTrue(result.passed)

    def test_field_value_mismatch(self):
        Path(self.tmp, "r.json").write_text(json.dumps({"version": 2}))
        rule = EvidenceRule(rule="json_schema", file_glob="r.json",
                           schema={"field_values": {"version": 1}})
        result = _check_json_schema(rule, self.tmp)
        self.assertFalse(result.passed)

    def test_summary_constraints_gt(self):
        Path(self.tmp, "r.json").write_text(json.dumps({"summary": {"total": 5}}))
        rule = EvidenceRule(rule="json_schema", file_glob="r.json",
                           schema={"summary_constraints": {"total": {"gt": 0}}})
        result = _check_json_schema(rule, self.tmp)
        self.assertTrue(result.passed)

    def test_summary_constraints_gt_fails(self):
        Path(self.tmp, "r.json").write_text(json.dumps({"summary": {"total": 0}}))
        rule = EvidenceRule(rule="json_schema", file_glob="r.json",
                           schema={"summary_constraints": {"total": {"gt": 0}}})
        result = _check_json_schema(rule, self.tmp)
        self.assertFalse(result.passed)

    def test_results_all_status(self):
        data = {"results": [{"status": "PASS"}, {"status": "PASS"}]}
        Path(self.tmp, "r.json").write_text(json.dumps(data))
        rule = EvidenceRule(rule="json_schema", file_glob="r.json",
                           schema={"results_all_status": "PASS"})
        result = _check_json_schema(rule, self.tmp)
        self.assertTrue(result.passed)

    def test_results_status_failure(self):
        data = {"results": [{"status": "PASS", "number": 1, "title": "t1"},
                            {"status": "FAIL", "number": 2, "title": "t2"}]}
        Path(self.tmp, "r.json").write_text(json.dumps(data))
        rule = EvidenceRule(rule="json_schema", file_glob="r.json",
                           schema={"results_all_status": "PASS"})
        result = _check_json_schema(rule, self.tmp)
        self.assertFalse(result.passed)
        self.assertIn("1 result(s) not PASS", result.message)

    def test_invalid_json(self):
        Path(self.tmp, "bad.json").write_text("not json")
        rule = EvidenceRule(rule="json_schema", file_glob="bad.json", schema={})
        result = _check_json_schema(rule, self.tmp)
        self.assertFalse(result.passed)
        self.assertIn("Cannot parse", result.message)

    def test_no_matching_file(self):
        rule = EvidenceRule(rule="json_schema", file_glob="nope.json", schema={})
        result = _check_json_schema(rule, self.tmp)
        self.assertFalse(result.passed)


class TestCheckJsonFieldEquals(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ev_field_")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_field_matches(self):
        Path(self.tmp, "f.json").write_text(json.dumps({"status": "ok"}))
        rule = EvidenceRule(rule="json_field_equals", file="f.json",
                           field_name="status", expected_value="ok")
        result = _check_json_field_equals(rule, self.tmp)
        self.assertTrue(result.passed)

    def test_field_mismatch(self):
        Path(self.tmp, "f.json").write_text(json.dumps({"status": "bad"}))
        rule = EvidenceRule(rule="json_field_equals", file="f.json",
                           field_name="status", expected_value="ok")
        result = _check_json_field_equals(rule, self.tmp)
        self.assertFalse(result.passed)

    def test_no_file_specified(self):
        rule = EvidenceRule(rule="json_field_equals", field_name="x", expected_value="y")
        result = _check_json_field_equals(rule, self.tmp)
        self.assertFalse(result.passed)


class TestCheckCommandSucceeds(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ev_cmd_")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_command_passes(self):
        rule = EvidenceRule(rule="command_succeeds", command="true")
        result = _check_command_succeeds(rule, self.tmp)
        self.assertTrue(result.passed)

    def test_command_fails(self):
        rule = EvidenceRule(rule="command_succeeds", command="false")
        result = _check_command_succeeds(rule, self.tmp)
        self.assertFalse(result.passed)
        self.assertIn("exit 1", result.message)

    @patch("forge.executor.engine.utils.repo_root", return_value="/tmp")
    def test_cwd_repo_root_uses_repo_root(self, mock_rr):
        rule = EvidenceRule(rule="command_succeeds", command="true", cwd="repo_root")
        result = _check_command_succeeds(rule, self.tmp)
        self.assertTrue(result.passed)

    def test_cwd_empty_uses_session_dir(self):
        rule = EvidenceRule(rule="command_succeeds", command=f"test -d {self.tmp}")
        result = _check_command_succeeds(rule, self.tmp)
        self.assertTrue(result.passed)


class TestCheckAllPredecessors(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ev_pred_")
        self.cp_dir = os.path.join(self.tmp, "checkpoints")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_all_predecessors_present_and_valid(self):
        steps = {"a": StepState(status=StepStatus.COMPLETE),
                 "b": StepState(status=StepStatus.COMPLETE),
                 "c": StepState()}
        for name in ["a", "b"]:
            write_checkpoint(self.cp_dir, name, "full")
        rule = EvidenceRule(rule="all_predecessors_checkpointed")
        result = _check_all_predecessors(rule, "c", steps, self.cp_dir)
        self.assertTrue(result.passed)

    def test_predecessor_missing_checkpoint(self):
        steps = {"a": StepState(status=StepStatus.COMPLETE),
                 "b": StepState()}
        write_checkpoint(self.cp_dir, "a", "full")
        # No checkpoint for 'a' actually — let's write one but not for 'a' predecessor
        # Actually 'b' is the target, 'a' is predecessor. We DO have a checkpoint for 'a'.
        # Let's test with missing predecessor:
        steps2 = {"x": StepState(), "y": StepState(), "z": StepState()}
        rule = EvidenceRule(rule="all_predecessors_checkpointed")
        result = _check_all_predecessors(rule, "z", steps2, self.cp_dir)
        self.assertFalse(result.passed)
        self.assertIn("Missing/invalid", result.message)

    def test_predecessor_with_deleted_checkpoint(self):
        """Deleting a predecessor's checkpoint file should cause failure."""
        steps = {"a": StepState(status=StepStatus.COMPLETE), "b": StepState()}
        path = write_checkpoint(self.cp_dir, "a", "full")
        os.unlink(path)
        rule = EvidenceRule(rule="all_predecessors_checkpointed")
        result = _check_all_predecessors(rule, "b", steps, self.cp_dir)
        self.assertFalse(result.passed)

    def test_unknown_step(self):
        rule = EvidenceRule(rule="all_predecessors_checkpointed")
        result = _check_all_predecessors(rule, "nope", {}, self.cp_dir)
        self.assertFalse(result.passed)

    def test_first_step_no_predecessors(self):
        steps = {"first": StepState()}
        rule = EvidenceRule(rule="all_predecessors_checkpointed")
        result = _check_all_predecessors(rule, "first", steps, self.cp_dir)
        self.assertTrue(result.passed)


class TestEvidenceChecker(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ev_checker_")
        self.cp_dir = os.path.join(self.tmp, "checkpoints")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_no_rules_passes(self):
        step = StepDefinition(name="s", evidence=[])
        checker = EvidenceChecker(self.tmp, self.cp_dir)
        result = checker.check(step)
        self.assertTrue(result.passed)

    def test_unknown_rule_type(self):
        step = StepDefinition(name="s", evidence=[EvidenceRule(rule="nonexistent")])
        checker = EvidenceChecker(self.tmp, self.cp_dir)
        result = checker.check(step)
        self.assertFalse(result.passed)
        self.assertIn("Unknown evidence rule", result.message)

    def test_combined_rules(self):
        Path(self.tmp, "report.json").write_text(json.dumps({"status": "ok"}))
        step = StepDefinition(name="s", evidence=[
            EvidenceRule(rule="file_exists", file_glob="report.json"),
        ])
        checker = EvidenceChecker(self.tmp, self.cp_dir)
        result = checker.check(step)
        self.assertTrue(result.passed)

    def test_first_failing_rule_stops(self):
        step = StepDefinition(name="s", evidence=[
            EvidenceRule(rule="file_exists", file_glob="missing.json"),
            EvidenceRule(rule="command_succeeds", command="true"),
        ])
        checker = EvidenceChecker(self.tmp, self.cp_dir)
        result = checker.check(step)
        self.assertFalse(result.passed)
        self.assertIn("No files matching", result.message)


if __name__ == "__main__":
    unittest.main()
