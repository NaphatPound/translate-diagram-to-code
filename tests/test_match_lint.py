"""Tests for the if-chain → match lint suggestion."""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow.lint import lint_source


def _msgs(src):
    return [w.message for w in lint_source(src)]


class TestMatchSuggestion(unittest.TestCase):

    def test_three_branch_chain_suggested(self):
        src = (
            'status = "ok"\n'
            'if status == "a"\n'
            '  p 1\n'
            'else\n'
            '  if status == "b"\n'
            '    p 2\n'
            '  else\n'
            '    if status == "c"\n'
            '      p 3\n'
        )
        msgs = _msgs(src)
        self.assertTrue(any("match` statement" in m for m in msgs))

    def test_two_branch_chain_not_suggested(self):
        # 2-arm chain — below threshold.
        src = (
            'n = 1\n'
            'if n == 0\n'
            '  p "zero"\n'
            'else\n'
            '  if n == 1\n'
            '    p "one"\n'
        )
        msgs = _msgs(src)
        self.assertFalse(any("match` statement" in m for m in msgs))

    def test_mismatched_vars_not_suggested(self):
        # Different variable each branch — not a match candidate.
        src = (
            'if a == 1\n'
            '  p 1\n'
            'else\n'
            '  if b == 2\n'
            '    p 2\n'
            '  else\n'
            '    if c == 3\n'
            '      p 3\n'
        )
        msgs = _msgs(src)
        self.assertFalse(any("match` statement" in m for m in msgs))

    def test_non_equality_chain_not_suggested(self):
        # `<` instead of `==` — not match-shaped.
        src = (
            'if n < 10\n'
            '  p "tiny"\n'
            'else\n'
            '  if n < 100\n'
            '    p "small"\n'
            '  else\n'
            '    if n < 1000\n'
            '      p "medium"\n'
        )
        msgs = _msgs(src)
        self.assertFalse(any("match` statement" in m for m in msgs))

    def test_suggests_var_name(self):
        src = (
            'kind = "x"\n'
            'if kind == "a"\n  p 1\nelse\n'
            '  if kind == "b"\n    p 2\n  else\n'
            '    if kind == "c"\n      p 3\n'
        )
        msgs = _msgs(src)
        match_msg = next(m for m in msgs if "match` statement" in m)
        self.assertIn("kind", match_msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
