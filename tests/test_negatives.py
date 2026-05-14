"""Tests for unary-minus / negative number literals."""
import sys
import os
import subprocess
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, compile_to
from flow.parser import NumberLit, BinOp


def _run_py(src):
    r = subprocess.run([sys.executable, "-c", src], capture_output=True,
                       text=True, timeout=4)
    if r.returncode != 0:
        raise AssertionError(f"py failed:\n{src}\n---\n{r.stderr}")
    return r.stdout


class TestUnaryMinus(unittest.TestCase):

    def test_negative_literal_folded(self):
        ast = parse("x = -5")
        # Should be folded to a single NumberLit, not 0 - 5.
        self.assertIsInstance(ast.body[0].value, NumberLit)
        self.assertEqual(ast.body[0].value.value, -5)

    def test_neg_in_if_cond(self):
        py = compile_to(parse('if -5 < 0\n  p "yes"'), "python")
        self.assertEqual(_run_py(py).strip(), "yes")

    def test_neg_in_positional(self):
        py = compile_to(parse("p -10"), "python")
        self.assertEqual(_run_py(py).strip(), "-10")

    def test_neg_assignment_and_use(self):
        py = compile_to(parse("x = -5\np x"), "python")
        self.assertEqual(_run_py(py).strip(), "-5")

    def test_neg_on_variable(self):
        # Not a literal — should become BinOp(0, '-', name).
        ast = parse("x = 5\ny = -x")
        # Body[1] is an AssignStmt; .value should be a BinOp.
        self.assertIsInstance(ast.body[1].value, BinOp)
        py = compile_to(ast, "python")
        py += "\nprint(y)"
        self.assertEqual(_run_py(py).strip(), "-5")


class TestDotOnString(unittest.TestCase):
    """`"abc".upper()` works through the postfix-DOT chain (introduced
    alongside chained method calls in iteration 9)."""

    def test_method_on_string_literal(self):
        py = compile_to(parse('p "abc".upper()'), "python")
        self.assertEqual(_run_py(py).strip(), "ABC")


if __name__ == "__main__":
    unittest.main(verbosity=2)
