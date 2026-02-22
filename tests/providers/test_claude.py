# Copyright 2026. Tests for Claude provider.

import json
from unittest.mock import MagicMock, patch

from architect.providers.claude import (
    ClaudeProvider,
    _child_env,
    _extract_content_text,
)
from architect.providers.protocol import AgentResult


class TestChildEnv:
    def test_strips_claudecode(self):
        with patch.dict("os.environ", {"CLAUDECODE": "1", "HOME": "/test"}):
            env = _child_env()
        assert "CLAUDECODE" not in env
        assert env["HOME"] == "/test"

    def test_preserves_other_vars(self):
        with patch.dict("os.environ", {"PATH": "/usr/bin", "MY_VAR": "val"}, clear=True):
            env = _child_env()
        assert env["PATH"] == "/usr/bin"
        assert env["MY_VAR"] == "val"


class TestExtractContentText:
    def test_string_content(self):
        assert _extract_content_text("hello\nworld") == ["hello", "world"]

    def test_empty_string(self):
        assert _extract_content_text("") == []

    def test_list_of_strings(self):
        assert _extract_content_text(["hello", "world"]) == ["hello", "world"]

    def test_list_of_dicts(self):
        content = [{"text": "hello"}, {"text": "world"}]
        assert _extract_content_text(content) == ["hello", "world"]

    def test_none(self):
        assert _extract_content_text(None) == []

    def test_int(self):
        assert _extract_content_text(42) == []


class TestParseStreamJson:
    def setup_method(self):
        self.provider = ClaudeProvider()

    def test_assistant_text(self):
        msg = json.dumps({
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Hello world"}]},
        })
        result = self.provider._parse_stream_json(msg)
        assert result == ["Hello world"]

    def test_assistant_tool_use_bash(self):
        msg = json.dumps({
            "type": "assistant",
            "message": {"content": [{
                "type": "tool_use",
                "name": "Bash",
                "input": {"command": "ls -la"},
            }]},
        })
        result = self.provider._parse_stream_json(msg)
        assert result == ["$ ls -la"]

    def test_assistant_tool_use_read(self):
        msg = json.dumps({
            "type": "assistant",
            "message": {"content": [{
                "type": "tool_use",
                "name": "Read",
                "input": {"file_path": "/tmp/test.py"},
            }]},
        })
        result = self.provider._parse_stream_json(msg)
        assert result == ["[Read] /tmp/test.py"]

    def test_tool_result(self):
        msg = json.dumps({
            "type": "tool_result",
            "tool_result": {"content": "output text"},
        })
        result = self.provider._parse_stream_json(msg)
        assert result == ["output text"]

    def test_result_type(self):
        msg = json.dumps({"type": "result", "result": "Final answer"})
        result = self.provider._parse_stream_json(msg)
        assert result == ["Final answer"]

    def test_error_type(self):
        msg = json.dumps({"type": "error", "error": "something failed"})
        result = self.provider._parse_stream_json(msg)
        assert result == ["[error] something failed"]

    def test_system_type(self):
        msg = json.dumps({"type": "system", "message": "init done", "subtype": "init"})
        result = self.provider._parse_stream_json(msg)
        assert result == ["[system:init] init done"]

    def test_invalid_json(self):
        result = self.provider._parse_stream_json("not json")
        assert result == ["not json"]

    def test_empty_string(self):
        result = self.provider._parse_stream_json("")
        assert result == []


class TestFormatSubagentType:
    def setup_method(self):
        self.provider = ClaudeProvider()

    def test_no_namespace(self):
        assert self.provider.format_subagent_type("general-purpose") == "general-purpose"

    def test_with_namespace(self):
        assert self.provider.format_subagent_type("architect", "planner") == "planner:architect"

    def test_general_purpose_ignores_namespace(self):
        assert self.provider.format_subagent_type("general-purpose", "planner") == "general-purpose"


class TestRunAgentNotFound:
    def test_command_not_found(self, tmp_path):
        provider = ClaudeProvider(agent_command="/nonexistent/binary")
        result = provider.run_agent(
            prompt="test",
            model="sonnet",
            max_turns=5,
            cwd=str(tmp_path),
            timeout_s=5,
            step_name="test",
            session_dir=str(tmp_path),
            activity_log_path=str(tmp_path / "activity.log"),
        )
        assert result.exit_code == 127
        assert result.timed_out is False


class TestProviderModels:
    def test_provider_models_structure(self):
        from architect.providers.models import PROVIDER_MODELS
        assert "claude" in PROVIDER_MODELS
        assert "codex" in PROVIDER_MODELS
        assert PROVIDER_MODELS["claude"]["reasoning"] == "opus"
        assert PROVIDER_MODELS["claude"]["balanced"] == "sonnet"
        assert PROVIDER_MODELS["claude"]["fast"] == "haiku"
