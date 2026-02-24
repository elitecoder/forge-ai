# Copyright 2026. Tests for planner session cleanup.

import os
import time

from forge.core.session import cleanup_sessions, list_sessions


def test_list_sessions_empty(tmp_path):
    assert list_sessions(tmp_path) == []


def test_list_sessions_with_dirs(tmp_path):
    (tmp_path / "session-1").mkdir()
    (tmp_path / "session-1" / ".planner-state.json").write_text("{}")
    (tmp_path / "session-2").mkdir()
    # session-2 has no state file

    sessions = list_sessions(tmp_path, state_filename=".planner-state.json")
    assert len(sessions) == 2
    assert sessions[0]["name"] == "session-1"
    assert sessions[0]["active"] is True
    assert sessions[1]["name"] == "session-2"
    assert sessions[1]["active"] is False


def test_list_sessions_nonexistent_base(tmp_path):
    fake = tmp_path / "nonexistent"
    assert list_sessions(fake) == []


def test_cleanup_removes_old(tmp_path):
    old_dir = tmp_path / "old-session"
    old_dir.mkdir()
    old_time = time.time() - (60 * 86400)
    os.utime(str(old_dir), (old_time, old_time))

    new_dir = tmp_path / "new-session"
    new_dir.mkdir()

    removed = cleanup_sessions(tmp_path, older_than_days=30)

    assert len(removed) == 1
    assert "old-session" in removed[0]
    assert new_dir.exists()
    assert not old_dir.exists()


def test_cleanup_nonexistent_base(tmp_path):
    fake = tmp_path / "nonexistent"
    assert cleanup_sessions(fake) == []
