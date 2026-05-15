"""Tests for iteration 28: compound-assign formatter collapse + duplicate-args
+ dead-code lint."""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, format_source
from flow.shrink import shrink_source
from flow.lint import lint_source


class TestCompoundCollapse(unittest.TestCase):

    def test_plus_collapses(self):
        out = format_source(parse("x = x + 1"))
        self.assertEqual(out.strip(), "x += 1")

    def test_minus_collapses(self):
        out = format_source(parse("x = x - 5"))
        self.assertEqual(out.strip(), "x -= 5")

    def test_mul_collapses(self):
        out = format_source(parse("x = x * 2"))
        self.assertEqual(out.strip(), "x *= 2")

    def test_no_collapse_when_lhs_differs(self):
        out = format_source(parse("x = y + 1"))
        self.assertEqual(out.strip(), "x = (y + 1)")

    def test_no_collapse_when_op_not_arith(self):
        out = format_source(parse("x = x == 1"))
        self.assertIn("==", out)
        # Result should not be `x ==`= which would be malformed.
        self.assertNotIn("==", out.split("=")[-1])

    def test_shrink_loop_accumulator(self):
        # Real-world use: accumulator loop.
        src = "total = 0\neach i in 1..3\n  total = total + i\np total"
        out = shrink_source(src)
        self.assertIn("total += i", out)


class TestDuplicateArgsLint(unittest.TestCase):

    def test_two_duplicates(self):
        ws = lint_source('read file="a" file="b" -> x')
        self.assertTrue(any("duplicate arg" in w.message for w in ws))

    def test_no_dup_one_arg(self):
        ws = lint_source('read file="a.txt" -> x')
        self.assertFalse(any("duplicate" in w.message for w in ws))


class TestDeadCodeLint(unittest.TestCase):

    def test_after_return(self):
        ws = lint_source("def f x\n  return x\n  p \"never\"")
        self.assertTrue(any("unreachable" in w.message for w in ws))

    def test_after_break(self):
        ws = lint_source("each i in 1..3\n  break\n  p i")
        self.assertTrue(any("unreachable" in w.message for w in ws))

    def test_clean_code_no_warning(self):
        ws = lint_source("def f x\n  return x")
        self.assertFalse(any("unreachable" in w.message for w in ws))


if __name__ == "__main__":
    unittest.main(verbosity=2)
