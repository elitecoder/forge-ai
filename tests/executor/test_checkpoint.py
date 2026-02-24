
"""Tests for engine.checkpoint — write, verify (existence-based), clear."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from forge.executor.engine.checkpoint import write_checkpoint, verify_checkpoint, verify_all_checkpoints, clear_checkpoints
from forge.executor.engine.evidence import EvidenceResult


class TestWriteCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cp_test_")
        self.cp_dir = os.path.join(self.tmp, "checkpoints")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_write_creates_file(self):
        path = write_checkpoint(self.cp_dir, "build", "full")
        self.assertTrue(os.path.isfile(path))
        self.assertTrue(path.endswith("build.passed"))

    def test_write_contains_metadata(self):
        path = write_checkpoint(self.cp_dir, "lint", "lightweight")
        content = Path(path).read_text()
        self.assertIn("step=lint", content)
        self.assertIn("pipeline=lightweight", content)
        self.assertIn("passed_at=", content)

    def test_write_with_evidence_paths(self):
        evidence = EvidenceResult(True, "ok", ["/tmp/foo.json"])
        path = write_checkpoint(self.cp_dir, "test", "full", evidence)
        content = Path(path).read_text()
        self.assertIn("evidence:/tmp/foo.json", content)

    def test_verify_valid_checkpoint(self):
        write_checkpoint(self.cp_dir, "build", "full")
        valid, msg = verify_checkpoint(self.cp_dir, "build")
        self.assertTrue(valid)
        self.assertIn("verified OK", msg)

    def test_verify_missing_checkpoint(self):
        valid, msg = verify_checkpoint(self.cp_dir, "ghost")
        self.assertFalse(valid)
        self.assertIn("not found", msg)

    def test_verify_evidence_artifact_exists(self):
        artifact = os.path.join(self.tmp, "result.json")
        Path(artifact).write_text('{"status": "pass"}')

        evidence = EvidenceResult(True, "ok", [artifact])
        write_checkpoint(self.cp_dir, "vt", "full", evidence)

        valid, msg = verify_checkpoint(self.cp_dir, "vt")
        self.assertTrue(valid)

    def test_verify_evidence_artifact_missing(self):
        artifact = os.path.join(self.tmp, "gone.json")
        Path(artifact).write_text("data")

        evidence = EvidenceResult(True, "ok", [artifact])
        write_checkpoint(self.cp_dir, "s", "full", evidence)
        os.unlink(artifact)

        valid, msg = verify_checkpoint(self.cp_dir, "s")
        self.assertFalse(valid)
        self.assertIn("missing", msg)


class TestVerifyAllCheckpoints(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cpall_test_")
        self.cp_dir = os.path.join(self.tmp, "checkpoints")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_all_present(self):
        for name in ["build", "lint", "test"]:
            write_checkpoint(self.cp_dir, name, "full")
        valid, present, missing = verify_all_checkpoints(self.cp_dir, ["build", "lint", "test"])
        self.assertTrue(valid)
        self.assertEqual(len(present), 3)
        self.assertEqual(missing, [])

    def test_some_missing(self):
        write_checkpoint(self.cp_dir, "build", "full")
        valid, present, missing = verify_all_checkpoints(self.cp_dir, ["build", "lint"])
        self.assertFalse(valid)
        self.assertEqual(present, ["build"])
        self.assertEqual(missing, ["lint"])

    def test_exclude_steps(self):
        write_checkpoint(self.cp_dir, "build", "full")
        valid, present, missing = verify_all_checkpoints(
            self.cp_dir, ["build", "create_pr"], exclude={"create_pr"}
        )
        self.assertTrue(valid)


class TestVerifyCheckpointNoChecksumLine(unittest.TestCase):
    """Checkpoints without checksum lines are valid in the new existence-based system."""
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cp_nochk_test_")
        self.cp_dir = os.path.join(self.tmp, "checkpoints")
        os.makedirs(self.cp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_verify_simple_checkpoint_file(self):
        cp_path = os.path.join(self.cp_dir, "build.passed")
        Path(cp_path).write_text("step=build\npassed_at=2026-01-01T00:00:00Z\npipeline=full\n")
        valid, msg = verify_checkpoint(self.cp_dir, "build")
        self.assertTrue(valid)

    def test_verify_empty_checkpoint_file(self):
        """Empty file still counts as existing checkpoint."""
        cp_path = os.path.join(self.cp_dir, "empty.passed")
        Path(cp_path).write_text("")
        valid, msg = verify_checkpoint(self.cp_dir, "empty")
        self.assertTrue(valid)


class TestClearCheckpoints(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cpclr_test_")
        self.cp_dir = os.path.join(self.tmp, "checkpoints")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_clear_removes_passed_files(self):
        write_checkpoint(self.cp_dir, "a", "full")
        write_checkpoint(self.cp_dir, "b", "full")
        clear_checkpoints(self.cp_dir)
        files = [f for f in os.listdir(self.cp_dir) if f.endswith(".passed")]
        self.assertEqual(files, [])

    def test_clear_on_nonexistent_dir(self):
        clear_checkpoints(os.path.join(self.tmp, "nope"))
        # Should create the dir
        self.assertTrue(os.path.isdir(os.path.join(self.tmp, "nope")))


class TestIsManualSkip(unittest.TestCase):
    """Bug 2 regression: is_manual_skip detects manually skipped checkpoints."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cp_skip_test_")
        self.cp_dir = os.path.join(self.tmp, "checkpoints")
        os.makedirs(self.cp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_returns_true_for_manual_skip(self):
        from forge.executor.engine.checkpoint import is_manual_skip
        cp_path = os.path.join(self.cp_dir, "test.passed")
        Path(cp_path).write_text("step=test\npassed_at=2026-01-01T00:00:00Z\npipeline=full\nmanual_skip=true\n")
        self.assertTrue(is_manual_skip(self.cp_dir, "test"))

    def test_returns_false_for_normal_checkpoint(self):
        from forge.executor.engine.checkpoint import is_manual_skip
        cp_path = os.path.join(self.cp_dir, "build.passed")
        Path(cp_path).write_text("step=build\npassed_at=2026-01-01T00:00:00Z\npipeline=full\n")
        self.assertFalse(is_manual_skip(self.cp_dir, "build"))

    def test_returns_false_for_missing_checkpoint(self):
        from forge.executor.engine.checkpoint import is_manual_skip
        self.assertFalse(is_manual_skip(self.cp_dir, "nonexistent"))


if __name__ == "__main__":
    unittest.main()
