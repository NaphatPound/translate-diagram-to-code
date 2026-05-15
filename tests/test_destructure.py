"""Tests for tuple/list destructuring assignment."""
import sys
import os
import subprocess
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, compile_to
from flow.parser import MultiAssignStmt


def _run_py(src):
    r = subprocess.run([sys.executable, "-c", src], capture_output=True,
                       text=True, timeout=4)
    if r.returncode != 0:
        raise AssertionError(f"py failed:\n{src}\n---\n{r.stderr}")
    return r.stdout


class TestDestructure(unittest.TestCase):

    def test_two_target_unpack(self):
        py = compile_to(parse(
            'pair = [10, 20]\na, b = pair\np a\np b'
        ), "python")
        out = _run_py(py).strip().split("\n")
        self.assertEqual(out, ["10", "20"])

    def test_three_target_unpack(self):
        py = compile_to(parse(
            'xs = [1, 2, 3]\na, b, c = xs\np f"{a},{b},{c}"'
        ), "python")
        self.assertEqual(_run_py(py).strip(), "1,2,3")

    def test_unpack_from_funccall(self):
        py = compile_to(parse(
            'def minmax xs\n  return [min(*xs), max(*xs)]\n'
            'lo, hi = minmax([3, 1, 4, 1, 5, 9])\np lo\np hi'
        ), "python")
        out = _run_py(py).strip().split("\n")
        self.assertEqual(out, ["1", "9"])

    def test_ast_is_multi_assign(self):
        ast = parse('a, b = [1, 2]')
        self.assertIsInstance(ast.body[0], MultiAssignStmt)
        self.assertEqual(ast.body[0].targets, ["a", "b"])

    def test_single_target_still_works(self):
        # `a = [1, 2]` is single-assign (no comma).
        ast = parse('a = [1, 2]')
        from flow.parser import AssignStmt
        self.assertIsInstance(ast.body[0], AssignStmt)

    def test_python_form(self):
        py = compile_to(parse('pair = [1, 2]\na, b = pair'), "python")
        self.assertIn("a, b = pair", py)

    def test_js_form(self):
        js = compile_to(parse('pair = [1, 2]\na, b = pair'), "js")
        self.assertIn("let [a, b] = pair;", js)

    def test_rust_form(self):
        rs = compile_to(parse('pair = [1, 2]\na, b = pair'), "rust")
        self.assertIn("let (a, b) = pair;", rs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
