"""Tests for caret-pointing error messages.

`parse(src)` and `compile_to(ast, lang, source=src)` should produce
error messages that include the offending source line — and, when the
error has a column, a `^` caret. This is the format the LLM retry
loop feeds back, and what humans see at the CLI.
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, compile_to, ParseError, CompileError


class TestParseErrorCaret(unittest.TestCase):

    def test_assignment_in_if_shows_source(self):
        try:
            parse("if x = 1\n  p \"hi\"")
            self.fail("expected ParseError")
        except ParseError as e:
            msg = str(e)
            self.assertIn("if x = 1", msg)   # source line embedded
            self.assertIn("^", msg)           # caret
            self.assertIn("did you mean '==' for comparison", msg)

    def test_caret_points_to_correct_column(self):
        # `if x = 1` — `=` is at col 6.
        try:
            parse("if x = 1")
        except ParseError as e:
            lines = str(e).splitlines()
            caret_line = [l for l in lines if "^" in l][0]
            # Caret should align under col 6 (1-indexed → 5 spaces of indent
            # after the "    " prefix).
            self.assertEqual(caret_line.lstrip(" "), "^")
            self.assertEqual(caret_line.index("^"), 4 + 5)

    def test_multiline_source_picks_right_line(self):
        src = 'p "ok"\nif y = 2\n  p y'
        try:
            parse(src)
        except ParseError as e:
            self.assertIn("if y = 2", str(e))
            self.assertNotIn("p \"ok\"", str(e))  # only the offending line


class TestCompileErrorCaret(unittest.TestCase):

    def test_unknown_arg_shows_source_when_source_passed(self):
        src = 'filter from=xs cond="x > 0"'
        try:
            compile_to(parse(src), "python", source=src)
        except CompileError as e:
            self.assertIn(src, str(e))

    def test_no_source_keeps_old_format(self):
        # Backwards compat: omitting `source` keeps the original message.
        src = 'filter from=xs cond="x > 0"'
        try:
            compile_to(parse(src), "python")
        except CompileError as e:
            # Source line should NOT be in the message when source isn't passed.
            self.assertNotIn("\n    filter from=xs", str(e))


if __name__ == "__main__":
    unittest.main(verbosity=2)
