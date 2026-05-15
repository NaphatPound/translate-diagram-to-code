"""Tests for the deterministic shrink rewriter."""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, compile_to
from flow.shrink import shrink_source, shrink


def _semantically_same(a: str, b: str) -> bool:
    """Compile both to Python and check they produce identical stdout."""
    import subprocess
    pa = compile_to(parse(a), "python")
    pb = compile_to(parse(b), "python")
    out_a = subprocess.run([sys.executable, "-c", pa], capture_output=True, text=True, timeout=4).stdout
    out_b = subprocess.run([sys.executable, "-c", pb], capture_output=True, text=True, timeout=4).stdout
    return out_a == out_b


class TestMathToAssignment(unittest.TestCase):

    def test_add_used_twice_becomes_assignment(self):
        # Used twice → can't inline; stays as assignment.
        out = shrink_source("add a=3 b=4 -> s\nprint s\nprint s")
        self.assertIn("s = (3 + 4)", out)
        self.assertNotIn("add a=", out)

    def test_used_once_inlines_through(self):
        # Used once → math-rewrite + inline combine.
        out = shrink_source("add a=3 b=4 -> s\nprint value=s")
        self.assertNotIn("add a=", out)
        self.assertIn("3 + 4", out)
        self.assertIn("print", out)

    def test_chain_used_once(self):
        out = shrink_source("mul a=4 b=5 -> p\ndiv a=p b=2 -> half\nprint half")
        # half = p/2 = (4*5)/2 → all inlined into the print.
        self.assertNotIn("mul ", out)
        self.assertNotIn("div ", out)
        self.assertIn("print", out)

    def test_preserves_semantics(self):
        before = "add a=2 b=3 -> s\nprint value=s"
        after = shrink_source(before)
        self.assertTrue(_semantically_same(before, after))


class TestAggregatorToAssignment(unittest.TestCase):

    def test_count_used_twice(self):
        out = shrink_source("count of=[1, 2, 3] -> n\nprint n\nprint n")
        self.assertIn("n = count([1, 2, 3])", out)

    def test_count_used_once_inlines(self):
        out = shrink_source("count of=[1, 2, 3] -> n\nprint n")
        self.assertNotIn("count of=", out)
        self.assertIn("count([1, 2, 3])", out)
        self.assertNotIn("\nn = ", out)  # no assignment line

    def test_sum_inline_semantic_equivalence(self):
        before = "sum of=[1,2,3,4,5] -> t\nprint t"
        self.assertTrue(_semantically_same(before, shrink_source(before)))


class TestMirrorIfToTernary(unittest.TestCase):

    def test_mirror_if_collapses(self):
        # msg used twice so it stays as an assignment after mirror-if rewrite.
        before = (
            "x = 5\n"
            "if x > 0\n"
            "  msg = \"big\"\n"
            "else\n"
            "  msg = \"small\"\n"
            "print msg\n"
            "print msg"
        )
        out = shrink_source(before)
        self.assertIn("msg = ", out)
        self.assertIn("?", out)
        self.assertIn(":", out)
        # Should have eliminated the if/else.
        self.assertNotIn("\nif ", out)

    def test_mirror_if_collapses_and_inlines(self):
        # msg used once → inlined further.
        before = (
            "if x > 0\n"
            "  msg = \"big\"\n"
            "else\n"
            "  msg = \"small\"\n"
            "print msg"
        )
        out = shrink_source(before)
        self.assertIn("?", out)
        self.assertNotIn("\nif ", out)
        self.assertNotIn("msg =", out)
        self.assertIn("print", out)

    def test_unbalanced_if_not_rewritten(self):
        # then branch has 2 stmts → cannot collapse to ternary.
        before = (
            "if x > 0\n"
            "  msg = \"big\"\n"
            "  flag = true\n"
            "else\n"
            "  msg = \"small\""
        )
        out = shrink_source(before)
        # Should still contain `if`.
        self.assertIn("if ", out)

    def test_different_targets_not_rewritten(self):
        before = (
            "if x > 0\n"
            "  a = 1\n"
            "else\n"
            "  b = 2"
        )
        out = shrink_source(before)
        self.assertIn("if ", out)


class TestMirrorReturnToTernary(unittest.TestCase):
    """`if cond / return a / else / return b` → `return cond ? a : b`,
    which combined with implicit-return becomes just the bare ternary."""

    def test_mirror_return_collapses(self):
        before = (
            "def f x\n"
            "  if x > 0\n"
            "    return \"pos\"\n"
            "  else\n"
            "    return \"neg\"\n"
            "p f(1)"
        )
        out = shrink_source(before)
        self.assertIn("?", out)
        self.assertIn(":", out)
        self.assertNotIn("if x > 0", out)
        # With implicit-return, both returns drop entirely.
        self.assertNotIn("return", out)

    def test_mirror_return_preserves_semantics(self):
        before = (
            "def label x\n"
            "  if x > 0\n"
            "    return \"pos\"\n"
            "  else\n"
            "    return \"neg\"\n"
            "p label(5)\n"
            "p label(-1)"
        )
        self.assertTrue(_semantically_same(before, shrink_source(before)))

    def test_no_else_not_rewritten(self):
        # `if / return` without else (early-return pattern) must stay intact.
        before = (
            "def f x\n"
            "  if x < 0\n"
            "    return 0\n"
            "  x * 2\n"
            "p f(3)"
        )
        out = shrink_source(before)
        self.assertIn("if ", out)
        self.assertIn("return 0", out)

    def test_one_branch_unbalanced_not_rewritten(self):
        # then branch has 2 stmts → not a mirror; leave the if alone.
        before = (
            "def f x\n"
            "  if x > 0\n"
            "    log = \"pos\"\n"
            "    return 1\n"
            "  else\n"
            "    return 2\n"
            "p f(0)"
        )
        out = shrink_source(before)
        self.assertIn("if ", out)


class TestNoOpOnCompactInput(unittest.TestCase):

    def test_already_compact_unchanged_semantically(self):
        compact = 'items = [1, 2, 3]\nn = count(items)\np n'
        out = shrink_source(compact)
        self.assertTrue(_semantically_same(compact, out))


if __name__ == "__main__":
    unittest.main(verbosity=2)
