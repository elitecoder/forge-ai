# Copyright 2026. Tests for session slug generation.

from unittest.mock import patch, MagicMock
import subprocess

from forge.core.session import generate_slug, _sanitize_slug


class TestSanitizeSlug:
    def test_simple_slug(self):
        assert _sanitize_slug("health-endpoint") == "health-endpoint"

    def test_strips_whitespace_and_newlines(self):
        assert _sanitize_slug("  health-endpoint\nignored line\n") == "health-endpoint"

    def test_strips_backticks(self):
        assert _sanitize_slug("`health-endpoint`") == "health-endpoint"

    def test_strips_quotes(self):
        assert _sanitize_slug('"health-endpoint"') == "health-endpoint"
        assert _sanitize_slug("'health-endpoint'") == "health-endpoint"

    def test_lowercases(self):
        assert _sanitize_slug("Health-Endpoint") == "health-endpoint"

    def test_replaces_spaces_and_special_chars(self):
        assert _sanitize_slug("health endpoint api!") == "health-endpoint-api"

    def test_collapses_multiple_hyphens(self):
        assert _sanitize_slug("health---endpoint") == "health-endpoint"

    def test_strips_leading_trailing_hyphens(self):
        assert _sanitize_slug("-health-endpoint-") == "health-endpoint"

    def test_truncates_to_50_chars(self):
        long_slug = "a" * 60
        assert len(_sanitize_slug(long_slug)) == 50

    def test_empty_input(self):
        assert _sanitize_slug("") == ""

    def test_only_special_chars(self):
        assert _sanitize_slug("!!!") == ""


class TestGenerateSlug:
    @patch("forge.core.session.subprocess.run")
    def test_returns_llm_slug(self, mock_run):
        mock_run.return_value = MagicMock(stdout="health-endpoint-api\n")
        result = generate_slug("Add a /health endpoint that returns JSON")
        assert result == "health-endpoint-api"
        mock_run.assert_called_once()

    @patch("forge.core.session.subprocess.run")
    def test_sanitizes_llm_output(self, mock_run):
        mock_run.return_value = MagicMock(stdout="`Health Endpoint API`\n")
        result = generate_slug("Add a /health endpoint")
        assert result == "health-endpoint-api"

    @patch("forge.core.session.subprocess.run")
    def test_fallback_on_empty_response(self, mock_run):
        mock_run.return_value = MagicMock(stdout="")
        result = generate_slug("some text", fallback="plan")
        assert result == "plan"

    @patch("forge.core.session.subprocess.run")
    def test_fallback_on_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=30)
        result = generate_slug("some text", fallback="plan")
        assert result == "plan"

    @patch("forge.core.session.subprocess.run")
    def test_fallback_on_file_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        result = generate_slug("some text", fallback="plan")
        assert result == "plan"

    @patch("forge.core.session.subprocess.run")
    def test_fallback_on_os_error(self, mock_run):
        mock_run.side_effect = OSError("broken")
        result = generate_slug("some text", fallback="plan")
        assert result == "plan"

    @patch("forge.core.session.subprocess.run")
    def test_truncates_input_text(self, mock_run):
        mock_run.return_value = MagicMock(stdout="long-text-summary")
        generate_slug("x" * 1000)
        prompt_sent = mock_run.call_args.kwargs.get("input") or mock_run.call_args[1].get("input", "")
        # Prompt should contain at most 500 chars of the input text
        assert "x" * 501 not in prompt_sent

    @patch("forge.core.session.subprocess.run")
    def test_custom_agent_command(self, mock_run):
        mock_run.return_value = MagicMock(stdout="my-slug")
        generate_slug("text", agent_command="/usr/local/bin/claude")
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/usr/local/bin/claude"

    @patch("forge.core.session.subprocess.run")
    def test_default_fallback(self, mock_run):
        mock_run.return_value = MagicMock(stdout="")
        result = generate_slug("some text")
        assert result == "session"
