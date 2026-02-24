
"""Tests for engine.templates — safe {{var}} substitution."""

import unittest

from forge.executor.engine.templates import render


class TestRender(unittest.TestCase):
    def test_simple_substitution(self):
        result = render("Hello {{NAME}}", {"NAME": "World"})
        self.assertEqual(result, "Hello World")

    def test_multiple_vars(self):
        result = render("{{A}} and {{B}}", {"A": "1", "B": "2"})
        self.assertEqual(result, "1 and 2")

    def test_missing_var_left_as_is(self):
        result = render("{{KNOWN}} {{UNKNOWN}}", {"KNOWN": "yes"})
        self.assertEqual(result, "yes {{UNKNOWN}}")

    def test_empty_template(self):
        self.assertEqual(render("", {"A": "1"}), "")

    def test_no_vars_in_template(self):
        self.assertEqual(render("plain text", {}), "plain text")

    def test_value_with_braces_not_recursive(self):
        result = render("{{A}}", {"A": "{{B}}", "B": "nope"})
        self.assertEqual(result, "{{B}}")

    def test_value_with_newlines(self):
        result = render("cmd: {{CMD}}", {"CMD": "line1\nline2"})
        self.assertEqual(result, "cmd: line1\nline2")

    def test_value_with_special_regex_chars(self):
        result = render("{{PATH}}", {"PATH": "/foo/bar/$HOME"})
        self.assertEqual(result, "/foo/bar/$HOME")

    def test_repeated_var(self):
        result = render("{{X}} {{X}}", {"X": "hi"})
        self.assertEqual(result, "hi hi")

    def test_empty_variables_dict(self):
        result = render("{{A}}", {})
        self.assertEqual(result, "{{A}}")


if __name__ == "__main__":
    unittest.main()
