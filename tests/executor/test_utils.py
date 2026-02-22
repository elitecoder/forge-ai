
"""Tests for engine.utils — shared utilities."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from architect.executor.engine.utils import (
    repo_root, validate_package_name, transitive_dependents,
    session_prefix, find_active_session, SESSIONS_BASE, is_bazel_repo,
)


class TestRepoRoot(unittest.TestCase):
    @patch("architect.executor.engine.utils.subprocess.run")
    def test_returns_git_root(self, mock_run):
        mock_run.return_value.stdout = "/home/user/repo\n"
        mock_run.return_value.returncode = 0
        result = repo_root()
        self.assertEqual(result, "/home/user/repo")

    @patch("architect.executor.engine.utils.subprocess.run", side_effect=FileNotFoundError)
    def test_fallback_on_error(self, _):
        result = repo_root()
        self.assertEqual(result, ".")


class TestValidatePackageName(unittest.TestCase):
    def test_valid_names(self):
        valid = [
            "apps/webapp",
            "platform/ui",
            "@hz/test-tools",
            "my_pkg.v2",
            "simple",
            "a/b/c/d",
        ]
        for name in valid:
            validate_package_name(name)  # should not raise

    def test_invalid_names(self):
        invalid = [
            "",
            "pkg; rm -rf /",
            "foo$(whoami)",
            "pkg\nbar",
            "foo bar",
            "pkg`echo hi`",
            "a&b",
            "x|y",
        ]
        for name in invalid:
            with self.assertRaises(ValueError, msg=f"Expected ValueError for {name!r}"):
                validate_package_name(name)


class TestTransitiveDependents(unittest.TestCase):
    def test_no_dependents(self):
        graph = {"b": ["a"], "c": ["b"]}
        result = transitive_dependents("c", graph)
        self.assertEqual(result, [])

    def test_direct_dependent(self):
        graph = {"b": ["a"], "c": ["b"]}
        result = transitive_dependents("a", graph)
        self.assertEqual(result, ["b", "c"])

    def test_diamond(self):
        graph = {
            "left": ["root"],
            "right": ["root"],
            "sink": ["left", "right"],
        }
        result = transitive_dependents("root", graph)
        self.assertEqual(result, ["left", "right", "sink"])

    def test_single_step(self):
        graph = {"b": ["a"]}
        result = transitive_dependents("a", graph)
        self.assertEqual(result, ["b"])

    def test_unknown_step(self):
        graph = {"b": ["a"]}
        result = transitive_dependents("unknown", graph)
        self.assertEqual(result, [])

    def test_middle_of_chain(self):
        graph = {"b": ["a"], "c": ["b"], "d": ["c"]}
        result = transitive_dependents("b", graph)
        self.assertEqual(result, ["c", "d"])

    def test_empty_graph(self):
        result = transitive_dependents("a", {})
        self.assertEqual(result, [])


class TestSessionPrefix(unittest.TestCase):
    @patch("architect.executor.engine.utils.subprocess.run")
    def test_extracts_ticket_from_branch(self, mock_run):
        mock_run.return_value.stdout = "mukuls/DVAWV-19760-speed-change\n"
        result = session_prefix()
        self.assertEqual(result, "DVAWV-19760")

    @patch("architect.executor.engine.utils.subprocess.run")
    def test_extracts_ticket_with_prefix_only(self, mock_run):
        mock_run.return_value.stdout = "feature/ABC-123\n"
        result = session_prefix()
        self.assertEqual(result, "ABC-123")

    @patch("architect.executor.engine.utils.subprocess.run")
    def test_no_ticket_pattern_uses_branch_name(self, mock_run):
        mock_run.return_value.stdout = "feature/my-branch\n"
        result = session_prefix()
        self.assertEqual(result, "feature-my-branch")

    @patch("architect.executor.engine.utils.subprocess.run", side_effect=FileNotFoundError)
    def test_git_not_available(self, _):
        result = session_prefix()
        self.assertEqual(result, "unknown")


class TestFindActiveSession(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="session_test_")
        self.sessions_dir = Path(self.tmp) / "executor"
        self.sessions_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    @patch("architect.executor.engine.utils.SESSIONS_BASE")
    @patch("architect.executor.engine.utils.session_prefix", return_value="DVAWV-19760")
    def test_finds_matching_session(self, _prefix, mock_base):
        mock_base.__class__ = Path
        mock_base.is_dir = lambda: True

        session = self.sessions_dir / "DVAWV-19760_2026-02-21_192200Z"
        session.mkdir()
        (session / "agent-state.json").write_text("{}")

        with patch("architect.executor.engine.utils.SESSIONS_BASE", self.sessions_dir):
            result = find_active_session()
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "DVAWV-19760_2026-02-21_192200Z")

    def test_slug_named_session_not_found_by_ticket_prefix(self):
        """Regression: session named with planner slug must NOT be found
        when find_active_session uses ticket prefix from branch."""
        session = self.sessions_dir / "speed-change-shortcuts_2026-02-21_192200Z"
        session.mkdir()
        (session / "agent-state.json").write_text("{}")

        with patch("architect.executor.engine.utils.SESSIONS_BASE", self.sessions_dir), \
             patch("architect.executor.engine.utils.session_prefix", return_value="DVAWV-19760"):
            result = find_active_session()
        self.assertIsNone(result)

    def test_returns_most_recent_when_multiple_sessions(self):
        s1 = self.sessions_dir / "DVAWV-19760_2026-02-21_100000Z"
        s1.mkdir()
        (s1 / "agent-state.json").write_text("{}")

        s2 = self.sessions_dir / "DVAWV-19760_2026-02-21_200000Z"
        s2.mkdir()
        (s2 / "agent-state.json").write_text("{}")
        # Touch s2 to ensure it's newer
        os.utime(s2, None)

        with patch("architect.executor.engine.utils.SESSIONS_BASE", self.sessions_dir), \
             patch("architect.executor.engine.utils.session_prefix", return_value="DVAWV-19760"):
            result = find_active_session()
        self.assertEqual(result.name, "DVAWV-19760_2026-02-21_200000Z")

    def test_returns_none_when_no_state_file(self):
        session = self.sessions_dir / "DVAWV-19760_2026-02-21_192200Z"
        session.mkdir()
        # No agent-state.json

        with patch("architect.executor.engine.utils.SESSIONS_BASE", self.sessions_dir), \
             patch("architect.executor.engine.utils.session_prefix", return_value="DVAWV-19760"):
            result = find_active_session()
        self.assertIsNone(result)


class TestSessionNameConsistency(unittest.TestCase):
    """Regression: _session_name must produce names that find_active_session can locate."""

    @patch("architect.executor.commands.subprocess.run")
    def test_ticket_branch_with_slug_uses_ticket(self, mock_run):
        """When branch has a ticket pattern, session name must use the ticket
        even if a planner slug is provided."""
        from architect.executor.commands import _session_name
        mock_run.return_value.stdout = "mukuls/DVAWV-19760-speed-change\n"

        name = _session_name(slug="speed-change-shortcuts")
        self.assertTrue(name.startswith("DVAWV-19760_"),
                        f"Expected session name to start with 'DVAWV-19760_', got '{name}'")

    @patch("architect.executor.commands.subprocess.run")
    def test_no_ticket_branch_uses_branch_not_slug(self, mock_run):
        """When branch has no ticket pattern, branch name (not slug) is used
        so that find_active_session() can locate the session."""
        from architect.executor.commands import _session_name
        mock_run.return_value.stdout = "feature/my-branch\n"

        name = _session_name(slug="my-feature")
        self.assertTrue(name.startswith("feature-my-branch_"),
                        f"Expected session name to start with 'feature-my-branch_', got '{name}'")

    @patch("architect.executor.commands.subprocess.run")
    def test_no_ticket_no_slug_uses_branch(self, mock_run):
        """When no ticket and no slug, branch name should be used."""
        from architect.executor.commands import _session_name
        mock_run.return_value.stdout = "feature/my-branch\n"

        name = _session_name()
        self.assertTrue(name.startswith("feature-my-branch_"),
                        f"Expected session name to start with 'feature-my-branch_', got '{name}'")

    @patch("architect.executor.commands.subprocess.run")
    @patch("architect.executor.engine.utils.subprocess.run")
    def test_session_name_matches_find_active_session(self, mock_utils_run, mock_cmd_run):
        """End-to-end: session created by _session_name should be findable
        by find_active_session when on the same branch."""
        from architect.executor.commands import _session_name

        # Both functions see the same branch
        branch = "mukuls/DVAWV-19760-speed-change-v2\n"
        mock_cmd_run.return_value.stdout = branch
        mock_utils_run.return_value.stdout = branch

        tmp = tempfile.mkdtemp(prefix="e2e_session_test_")
        try:
            sessions_dir = Path(tmp) / "executor"
            sessions_dir.mkdir(parents=True)

            name = _session_name(slug="speed-change-shortcuts")
            session = sessions_dir / name
            session.mkdir()
            (session / "agent-state.json").write_text("{}")

            with patch("architect.executor.engine.utils.SESSIONS_BASE", sessions_dir):
                result = find_active_session()

            self.assertIsNotNone(result,
                                 f"find_active_session() could not find session '{name}'")
            self.assertEqual(result.name, name)
        finally:
            shutil.rmtree(tmp)


class TestIsBazelRepo(unittest.TestCase):
    """Bug 4+6 regression: is_bazel_repo detects WORKSPACE / WORKSPACE.bazel."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bazel_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_true_when_workspace_exists(self):
        Path(os.path.join(self.tmp, "WORKSPACE")).write_text("")
        self.assertTrue(is_bazel_repo(self.tmp))

    def test_true_when_workspace_bazel_exists(self):
        Path(os.path.join(self.tmp, "WORKSPACE.bazel")).write_text("")
        self.assertTrue(is_bazel_repo(self.tmp))

    def test_false_when_neither_exists(self):
        self.assertFalse(is_bazel_repo(self.tmp))


if __name__ == "__main__":
    unittest.main()
