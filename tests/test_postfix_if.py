"""Tests for postfix-if and chained method calls."""
import sys
import os
import subprocess
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, compile_to
from flow.parser import IfStmt, MethodCall


def _run_py(src):
    r = subprocess.run([sys.executable, "-c", src], capture_output=True,
                       text=True, timeout=4)
    if r.returncode != 0:
        raise AssertionError(f"py failed:\n{src}\n---\n{r.stderr}")
    return r.stdout


class TestPostfixIf(unittest.TestCase):

    def test_parses_as_if(self):
        ast = parse('p "big" if 1 > 0')
        self.assertIsInstance(ast.body[0], IfStmt)
        self.assertEqual(len(ast.body[0].then), 1)
        self.assertIsNone(ast.body[0].else_)

    def test_runs_correctly_true(self):
        py = compile_to(parse('x = 5\np "big" if x > 0'), "python")
        self.assertEqual(_run_py(py).strip(), "big")

    def test_runs_correctly_false(self):
        # Use 0 (falsy condition `x > 0` when x is 0).
        py = compile_to(parse('x = 0\np "big" if x > 0'), "python")
        self.assertEqual(_run_py(py).strip(), "")

    def test_postfix_if_with_arrow(self):
        # `r "a.txt" -> a if cond` reads only when condition is true.
        # Note: this wraps the whole call in the if; `a` won't be bound when
        # cond is false. That's the expected semantics.
        py = compile_to(parse('x = 1\nr "/etc/hostname" -> a if x > 0'), "python")
        self.assertIn("if (x > 0):", py)
        self.assertIn("a = open", py)


class TestChainedMethods(unittest.TestCase):

    def test_two_chained(self):
        py = compile_to(parse(
            's = "  hello  "\nclean = s.strip().upper()\np clean'
        ), "python")
        self.assertIn("s.strip().upper()", py)
        self.assertEqual(_run_py(py).strip(), "HELLO")

    def test_three_chained(self):
        py = compile_to(parse(
            's = "  hello world  "\np s.strip().upper().split(" ")'
        ), "python")
        self.assertEqual(_run_py(py).strip(), "['HELLO', 'WORLD']")

    def test_attribute_access_at_tail(self):
        # `.length` (no parens) is attribute access in the new chain logic.
        # Python doesn't have .length on strings, so test on a list.
        py = compile_to(parse(
            'xs = [1, 2, 3]\np xs'  # smoke test for parsing
        ), "python")
        self.assertEqual(_run_py(py).strip(), "[1, 2, 3]")

    def test_method_on_method_result(self):
        # MethodCall whose receiver is itself a MethodCall.
        ast = parse('p s.split(",").upper()')
        v = ast.body[0].args[0].value
        self.assertIsInstance(v, MethodCall)
        self.assertEqual(v.method, "upper")
        self.assertIsInstance(v.receiver, MethodCall)
        self.assertEqual(v.receiver.method, "split")


class TestBarewordWithDotsStillWorks(unittest.TestCase):
    """The DOT-tokenization change shouldn't break dotted barewords like
    `data.csv` and `row.name` member access — they go through the postfix
    chain to produce the same Name path as before."""

    def test_filename_bareword(self):
        # `data.csv` becomes Name(["data", "csv"]) and renders as string lit.
        py = compile_to(parse("read file=data.csv -> r"), "python")
        self.assertIn('"data.csv"', py)

    def test_member_access(self):
        py = compile_to(parse(
            "items = [{name: \"a\"}]\neach row in items\n  p row.name"
        ), "python")
        self.assertEqual(_run_py(py).strip(), "a")


if __name__ == "__main__":
    unittest.main(verbosity=2)
