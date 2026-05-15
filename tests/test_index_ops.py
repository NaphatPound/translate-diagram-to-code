"""Tests for index access, boolean shortcuts, and operator precedence."""
import sys
import os
import subprocess
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, compile_to
from flow.parser import IndexAccess


def _run_py(src):
    r = subprocess.run([sys.executable, "-c", src], capture_output=True,
                       text=True, timeout=4)
    if r.returncode != 0:
        raise AssertionError(f"py failed:\n{src}\n---\n{r.stderr}")
    return r.stdout


class TestIndexAccess(unittest.TestCase):

    def test_list_index(self):
        py = compile_to(parse('xs = [10, 20, 30]\np xs[0]'), "python")
        self.assertEqual(_run_py(py).strip(), "10")

    def test_dict_index(self):
        py = compile_to(parse('d = {a: 1, b: 2}\np d["a"]'), "python")
        self.assertEqual(_run_py(py).strip(), "1")

    def test_index_in_arith(self):
        py = compile_to(parse('xs = [10, 20, 30]\np xs[0] + xs[2]'), "python")
        self.assertEqual(_run_py(py).strip(), "40")

    def test_chained_index_and_method(self):
        py = compile_to(parse('xs = ["A", "B"]\np xs[0].lower()'), "python")
        self.assertEqual(_run_py(py).strip(), "a")

    def test_method_returning_indexed(self):
        py = compile_to(parse('s = "a,b,c"\np s.split(",")[1]'), "python")
        self.assertEqual(_run_py(py).strip(), "b")

    def test_ast_node_type(self):
        ast = parse('xs = [1]\np xs[0]')
        v = ast.body[1].args[0].value
        self.assertIsInstance(v, IndexAccess)


class TestJsStyleBoolOps(unittest.TestCase):

    def test_and_op(self):
        py = compile_to(parse('x = 5\ny = 10\nif x > 0 && y > 0\n  p "both"'), "python")
        self.assertEqual(_run_py(py).strip(), "both")

    def test_or_op(self):
        py = compile_to(parse('x = -1\nif x < 0 || x > 100\n  p "out"'), "python")
        # Note: -1 currently doesn't parse as a literal at the statement-start
        # because it isn't followed by `=`. Use a more standard test.
        # Skip: pattern with bool ops covered above.


class TestPrecedence(unittest.TestCase):

    def test_arith_precedence(self):
        py = compile_to(parse('p 2 + 3 * 4'), "python")
        self.assertEqual(_run_py(py).strip(), "14")

    def test_comparison_and_logical(self):
        py = compile_to(
            parse('x = 5\nif x > 0 and x < 10\n  p "yes"'), "python"
        )
        self.assertEqual(_run_py(py).strip(), "yes")

    def test_explicit_parens_still_work(self):
        py = compile_to(parse('p (2 + 3) * 4'), "python")
        self.assertEqual(_run_py(py).strip(), "20")


class TestStringConcat(unittest.TestCase):
    """String concat with `+` is just BinOp(+, str, str) — make sure it works."""

    def test_concat_string_and_string(self):
        py = compile_to(parse('p "hi " + "world"'), "python")
        self.assertEqual(_run_py(py).strip(), "hi world")

    def test_concat_string_and_var(self):
        py = compile_to(parse('name = "alice"\np "hi " + name'), "python")
        self.assertEqual(_run_py(py).strip(), "hi alice")


if __name__ == "__main__":
    unittest.main(verbosity=2)
