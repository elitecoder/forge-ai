"""Tests for session cleanup — session lifecycle."""

import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from architect.core.session import list_sessions, cleanup_sessions


class TestListSessions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cleanup_test_")
        self.sessions_base = Path(self.tmp) / "pipeline-sessions"
        self.sessions_base.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_list_empty(self):
        empty = Path(self.tmp) / "empty"
        empty.mkdir()
        sessions = list_sessions(empty)
        self.assertEqual(sessions, [])

    def test_list_sessions(self):
        (self.sessions_base / "HZ-123_2026-01-01").mkdir()
        (self.sessions_base / "HZ-456_2026-01-02").mkdir()
        sessions = list_sessions(self.sessions_base)
        self.assertEqual(len(sessions), 2)
        names = [s["name"] for s in sessions]
        self.assertIn("HZ-123_2026-01-01", names)

    def test_list_ignores_files(self):
        (self.sessions_base / "session1").mkdir()
        (self.sessions_base / "not-a-dir.txt").write_text("hi")
        sessions = list_sessions(self.sessions_base)
        self.assertEqual(len(sessions), 1)


class TestCleanupSessions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cleanup_test_")
        self.sessions_base = Path(self.tmp) / "pipeline-sessions"
        self.sessions_base.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_cleanup_old_sessions(self):
        old = self.sessions_base / "old-session"
        old.mkdir()
        old_time = time.time() - (10 * 86400)
        os.utime(str(old), (old_time, old_time))

        new = self.sessions_base / "new-session"
        new.mkdir()

        removed = cleanup_sessions(self.sessions_base, older_than_days=7)
        self.assertEqual(len(removed), 1)
        self.assertFalse(old.exists())
        self.assertTrue(new.exists())

    def test_cleanup_none_old(self):
        (self.sessions_base / "recent").mkdir()
        removed = cleanup_sessions(self.sessions_base, older_than_days=7)
        self.assertEqual(removed, [])


class TestListSessionsNoBaseDir(unittest.TestCase):
    def test_returns_empty_when_sessions_base_missing(self):
        nonexistent = Path(tempfile.mkdtemp(prefix="cleanup_nobase_")) / "does-not-exist"
        result = list_sessions(nonexistent)
        self.assertEqual(result, [])
        shutil.rmtree(nonexistent.parent)


class TestCleanupSessionsNoBaseDir(unittest.TestCase):
    def test_returns_empty_when_sessions_base_missing(self):
        nonexistent = Path(tempfile.mkdtemp(prefix="cleanup_nobase_")) / "does-not-exist"
        result = cleanup_sessions(nonexistent, older_than_days=7)
        self.assertEqual(result, [])
        shutil.rmtree(nonexistent.parent)


if __name__ == "__main__":
    unittest.main()
