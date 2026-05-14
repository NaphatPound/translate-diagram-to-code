"""Tests for flow lint."""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow.lint import lint_source


class TestLintMath(unittest.TestCase):

    def test_add_suggests_assignment(self):
        ws = lint_source("add a=1 b=2 -> s")
        self.assertEqual(len(ws), 1)
        self.assertIn("s = 1 + 2", ws[0].suggestion)

    def test_sub_suggests_assignment(self):
        ws = lint_source("sub a=10 b=3 -> d")
        self.assertEqual(len(ws), 1)
        self.assertIn("d = 10 - 3", ws[0].suggestion)

    def test_div_suggests(self):
        ws = lint_source("div a=10 b=2 -> r")
        self.assertIn("r = 10 / 2", ws[0].suggestion)

    def test_math_without_out_no_warning(self):
        # No -> name → can't suggest a clean assignment, no warning.
        ws = lint_source("add a=1 b=2")
        self.assertEqual(len(ws), 0)


class TestLintAggregators(unittest.TestCase):

    def test_count_suggests_funccall_assignment(self):
        ws = lint_source("count of=items -> n")
        self.assertEqual(len(ws), 1)
        self.assertIn("n = count(items)", ws[0].suggestion)

    def test_sum_suggests(self):
        ws = lint_source("sum of=xs -> total")
        self.assertIn("total = sum(xs)", ws[0].suggestion)


class TestLintPositional(unittest.TestCase):

    def test_print_value_suggests_positional(self):
        ws = lint_source('print value="hi"')
        self.assertEqual(len(ws), 1)
        self.assertIn('print "hi"', ws[0].suggestion)

    def test_upper_text_suggests_positional(self):
        ws = lint_source('upper text="abc" -> big')
        self.assertEqual(len(ws), 1)
        self.assertIn('upper "abc" -> big', ws[0].suggestion)

    def test_positional_form_emits_no_warning(self):
        ws = lint_source('print "hi"')
        self.assertEqual(len(ws), 0)


class TestLintInsideControl(unittest.TestCase):

    def test_finds_warnings_inside_if(self):
        ws = lint_source("if x > 0\n  add a=1 b=2 -> s")
        self.assertEqual(len(ws), 1)
        self.assertEqual(ws[0].line, 2)

    def test_finds_warnings_inside_each(self):
        ws = lint_source(
            "items = [1,2,3]\n"
            "each i in items\n"
            "  print value=i"
        )
        self.assertEqual(len(ws), 1)


class TestLintNoFalsePositives(unittest.TestCase):

    def test_compact_code_clean(self):
        src = (
            "items = [1, 2, 3]\n"
            "n = count(items)\n"
            "p n\n"
            "r \"a.csv\" | upper | p\n"
        )
        self.assertEqual(lint_source(src), [])

    def test_does_not_warn_on_filter_with_where(self):
        # filter has multiple args, positional-only suggestion shouldn't fire.
        ws = lint_source('filter from=xs where="x > 0" -> big')
        self.assertEqual(len(ws), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
