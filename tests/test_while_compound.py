"""Tests for `while` loop and compound assignment."""
import sys
import os
import subprocess
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, compile_to
from flow.parser import AssignStmt, BinOp, WhileStmt, Name


def _run_py(src):
    r = subprocess.run([sys.executable, "-c", src], capture_output=True,
                       text=True, timeout=4)
    if r.returncode != 0:
        raise AssertionError(f"py failed:\n{src}\n---\n{r.stderr}")
    return r.stdout


class TestCompoundAssign(unittest.TestCase):

    def test_plus_eq_desugars(self):
        ast = parse("n = 1\nn += 5")
        # Second stmt should be AssignStmt(target=n, value=BinOp(+, Name(n), 5)).
        a = ast.body[1]
        self.assertIsInstance(a, AssignStmt)
        self.assertEqual(a.target, "n")
        self.assertIsInstance(a.value, BinOp)
        self.assertEqual(a.value.op, "+")
        self.assertIsInstance(a.value.left, Name)
        self.assertEqual(a.value.left.parts, ["n"])

    def test_plus_eq_runs(self):
        py = compile_to(parse("n = 1\nn += 5\np n"), "python")
        self.assertEqual(_run_py(py).strip(), "6")

    def test_minus_times_div(self):
        src = "n = 100\nn -= 10\nn *= 2\nn /= 3\np n"
        py = compile_to(parse(src), "python")
        self.assertEqual(_run_py(py).strip(), "60.0")

    def test_compound_with_method_call_rhs(self):
        py = compile_to(parse('s = ""\ns += "hi"\np s'), "python")
        self.assertEqual(_run_py(py).strip(), "hi")


class TestWhileLoop(unittest.TestCase):

    def test_parses_as_while(self):
        ast = parse("n = 1\nwhile n < 5\n  n += 1")
        self.assertIsInstance(ast.body[1], WhileStmt)

    def test_basic_while_runs(self):
        py = compile_to(parse(
            "n = 0\ni = 1\nwhile i <= 5\n  n += i\n  i += 1\np n"
        ), "python")
        self.assertEqual(_run_py(py).strip(), "15")

    def test_while_with_break(self):
        py = compile_to(parse(
            "i = 0\nwhile true\n  i += 1\n  break if i > 3\np i"
        ), "python")
        self.assertEqual(_run_py(py).strip(), "4")

    def test_while_emits_native_per_lang(self):
        ast = parse("i = 0\nwhile i < 3\n  i += 1")
        self.assertIn("while (i < 3):", compile_to(ast, "python"))
        self.assertIn("while ((i < 3)) {", compile_to(ast, "js"))
        self.assertIn("while (i < 3) {", compile_to(ast, "rust"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
