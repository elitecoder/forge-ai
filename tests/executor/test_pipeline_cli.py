"""Tests for executor commands — CLI integration tests via main()."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from architect.executor.commands import main


class TestPipelineCLIMain(unittest.TestCase):
    """Integration tests that invoke main() with patched sys.argv."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cli_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_no_command_exits(self):
        with patch("sys.argv", ["architect-executor"]):
            with self.assertRaises(SystemExit) as ctx:
                main()
            self.assertNotEqual(ctx.exception.code, 0)

    def test_status_without_init_fails(self):
        with patch("sys.argv", ["architect-executor", "status"]):
            with patch("architect.executor.commands.find_active_session", return_value=None):
                with self.assertRaises(SystemExit) as ctx:
                    main()
                self.assertNotEqual(ctx.exception.code, 0)

    def test_next_without_init_fails(self):
        with patch("sys.argv", ["architect-executor", "next"]):
            with patch("architect.executor.commands.find_active_session", return_value=None):
                with self.assertRaises(SystemExit) as ctx:
                    main()
                self.assertNotEqual(ctx.exception.code, 0)

    def test_help_flag(self):
        with patch("sys.argv", ["architect-executor", "--help"]):
            with self.assertRaises(SystemExit) as ctx:
                main()
            self.assertEqual(ctx.exception.code, 0)

    def test_init_subparser_help(self):
        with patch("sys.argv", ["architect-executor", "init", "--help"]):
            with self.assertRaises(SystemExit) as ctx:
                main()
            self.assertEqual(ctx.exception.code, 0)

    def test_fail_unknown_step_without_init(self):
        with patch("sys.argv", ["architect-executor", "fail", "ghost", "some error"]):
            with patch("architect.executor.commands.find_active_session", return_value=None):
                with self.assertRaises(SystemExit) as ctx:
                    main()
                self.assertNotEqual(ctx.exception.code, 0)

    def test_verify_without_init_fails(self):
        with patch("sys.argv", ["architect-executor", "verify"]):
            with patch("architect.executor.commands.find_active_session", return_value=None):
                with self.assertRaises(SystemExit) as ctx:
                    main()
                self.assertNotEqual(ctx.exception.code, 0)

    def test_sessions_runs(self):
        with patch("sys.argv", ["architect-executor", "sessions"]):
            with patch("architect.executor.commands._core_list_sessions", return_value=[]):
                main()

    def test_summary_without_init_fails(self):
        with patch("sys.argv", ["architect-executor", "summary"]):
            with patch("architect.executor.commands.find_active_session", return_value=None):
                with self.assertRaises(SystemExit) as ctx:
                    main()
                self.assertNotEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
