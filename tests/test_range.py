"""Tests for range literal `start..end` and `repeat N as i`."""
import sys
import os
import subprocess
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, compile_to, format_source
from flow.parser import Range, RepeatStmt


def _run_py(src):
    r = subprocess.run([sys.executable, "-c", src], capture_output=True,
                       text=True, timeout=4)
    if r.returncode != 0:
        raise AssertionError(f"py failed:\n{src}\n---\n{r.stderr}")
    return r.stdout


class TestRangeParser(unittest.TestCase):

    def test_simple_int_range(self):
        ast = parse("xs = 1..5")
        v = ast.body[0].value
        self.assertIsInstance(v, Range)

    def test_range_in_each(self):
        ast = parse("each i in 1..3\n  print i")
        self.assertIsInstance(ast.body[0].iterable, Range)

    def test_range_as_funccall_arg(self):
        # `sum(1..5)` should parse the `1..5` as the funccall arg.
        ast = parse("p sum(1..5)")
        # Just confirm it parsed without error.
        self.assertEqual(ast.body[0].verb, "print")


class TestRangeCompile(unittest.TestCase):

    def test_each_over_range_runs(self):
        py = compile_to(parse("each i in 1..3\n  p i"), "python")
        self.assertEqual(_run_py(py).strip().split("\n"), ["1", "2", "3"])

    def test_range_sum(self):
        py = compile_to(parse("p sum(1..10)"), "python")
        self.assertEqual(_run_py(py).strip(), "55")

    def test_inclusive_endpoint(self):
        # 1..5 should include 5 (inclusive).
        py = compile_to(parse("each i in 1..5\n  p i"), "python")
        self.assertEqual(_run_py(py).strip().split("\n"), ["1", "2", "3", "4", "5"])

    def test_round_trip_format(self):
        # `1..5` → format → `1..5`.
        out = format_source(parse("xs = 1..5"))
        self.assertIn("1..5", out)


class TestRepeatAs(unittest.TestCase):

    def test_repeat_with_var(self):
        py = compile_to(parse("repeat 3 as i\n  p i"), "python")
        self.assertEqual(_run_py(py).strip().split("\n"), ["0", "1", "2"])

    def test_repeat_var_in_scope(self):
        # `p (n + 1)` inside body should reference the loop var, not a string lit.
        py = compile_to(parse("repeat 2 as n\n  p (n + 1)"), "python")
        self.assertEqual(_run_py(py).strip().split("\n"), ["1", "2"])

    def test_repeat_no_var_still_works(self):
        py = compile_to(parse("repeat 2\n  p \"hi\""), "python")
        self.assertEqual(_run_py(py).strip().split("\n"), ["hi", "hi"])

    def test_format_preserves_repeat_as(self):
        out = format_source(parse("repeat 3 as i\n  p i"))
        self.assertIn("repeat 3 as i", out)


class TestCharSavings(unittest.TestCase):

    def test_range_saves_chars_vs_list(self):
        long_form = "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"
        range_form = "1..10"
        self.assertLess(len(range_form), len(long_form))


if __name__ == "__main__":
    unittest.main(verbosity=2)
