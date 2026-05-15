"""Tests for shrink + lint of explicit returns at def end."""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse
from flow.shrink import shrink, shrink_source
from flow.lint import lint_source
from flow.parser import DefStmt, ReturnStmt, ExprStmt


class TestShrinkDropReturn(unittest.TestCase):

    def test_drops_explicit_return(self):
        src = "def double x\n  return x * 2"
        ast = shrink(parse(src))
        d = ast.body[0]
        self.assertIsInstance(d, DefStmt)
        self.assertIsInstance(d.body[-1], ExprStmt)

    def test_preserves_intermediate_returns(self):
        # `return` inside an `if` is NOT the last stmt of def → kept.
        src = (
            "def fact n\n"
            "  if n <= 1\n"
            "    return 1\n"
            "  return n * fact(n - 1)\n"
        )
        ast = shrink(parse(src))
        d = ast.body[0]
        # Last stmt of def body becomes ExprStmt.
        self.assertIsInstance(d.body[-1], ExprStmt)
        # The `return 1` inside the if-body stays as ReturnStmt.
        if_stmt = d.body[0]
        self.assertIsInstance(if_stmt.then[-1], ReturnStmt)

    def test_round_trip_source_runs(self):
        import subprocess
        src = "def double x\n  return x * 2\np double(5)"
        shrunk = shrink_source(src)
        # Compile the shrunk form and run it.
        from flow import compile_to
        py = compile_to(parse(shrunk), "python")
        r = subprocess.run([sys.executable, "-c", py], capture_output=True,
                           text=True, timeout=4)
        self.assertEqual(r.stdout.strip(), "10")


class TestLintImplicitReturn(unittest.TestCase):

    def test_suggests_implicit_return(self):
        ws = lint_source("def double x\n  return x * 2")
        self.assertEqual(len(ws), 1)
        self.assertIn("implicit", ws[0].message.lower())

    def test_no_suggestion_if_no_return(self):
        ws = lint_source("def double x\n  x * 2")
        self.assertEqual(ws, [])

    def test_no_suggestion_for_bare_return(self):
        # `return` with no value isn't a candidate.
        ws = lint_source("def f\n  return")
        # Some unrelated lints may fire; just check no implicit-return suggestion.
        for w in ws:
            self.assertNotIn("implicit", w.message.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
