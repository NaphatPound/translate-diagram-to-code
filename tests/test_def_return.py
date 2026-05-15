"""Tests for `def`, `return`, and funccall-as-statement (ExprStmt)."""
import sys
import os
import subprocess
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, compile_to
from flow.parser import DefStmt, ReturnStmt, ExprStmt, FuncCall


def _run_py(src):
    r = subprocess.run([sys.executable, "-c", src], capture_output=True,
                       text=True, timeout=4)
    if r.returncode != 0:
        raise AssertionError(f"py failed:\n{src}\n---\n{r.stderr}")
    return r.stdout


class TestDef(unittest.TestCase):

    def test_simple_def_parses(self):
        ast = parse("def double x\n  return x * 2")
        self.assertIsInstance(ast.body[0], DefStmt)
        self.assertEqual(ast.body[0].name, "double")
        self.assertEqual(ast.body[0].params, ["x"])

    def test_def_with_multiple_params(self):
        ast = parse("def add a b\n  return a + b")
        self.assertEqual(ast.body[0].params, ["a", "b"])

    def test_def_no_params(self):
        ast = parse('def hello\n  p "hi"')
        self.assertEqual(ast.body[0].params, [])

    def test_def_runs_python(self):
        py = compile_to(parse(
            "def double x\n  return x * 2\np double(5)"
        ), "python")
        self.assertEqual(_run_py(py).strip(), "10")

    def test_recursive_factorial(self):
        py = compile_to(parse(
            "def fact n\n  if n <= 1\n    return 1\n  return n * fact(n - 1)\np fact(5)"
        ), "python")
        self.assertEqual(_run_py(py).strip(), "120")


class TestReturn(unittest.TestCase):

    def test_return_with_value(self):
        ast = parse("def x\n  return 42")
        body = ast.body[0].body
        self.assertIsInstance(body[0], ReturnStmt)
        self.assertIsNotNone(body[0].value)

    def test_bare_return(self):
        ast = parse("def x\n  return")
        self.assertIsNone(ast.body[0].body[0].value)


class TestExprStmt(unittest.TestCase):

    def test_funccall_statement(self):
        # `myfunc(1, 2)` at statement level parses as ExprStmt.
        # NO space between name and `(`.
        ast = parse("def f x\n  p x\nf(5)")
        self.assertIsInstance(ast.body[1], ExprStmt)
        self.assertIsInstance(ast.body[1].value, FuncCall)

    def test_verb_with_paren_value_unchanged(self):
        # Space before `(` means verb-with-positional, NOT funccall.
        # `p (2 + 3)` should remain a Call to `print`.
        py = compile_to(parse("p (2 + 3) * 4"), "python")
        self.assertEqual(_run_py(py).strip(), "20")

    def test_funccall_runs(self):
        py = compile_to(parse(
            'def greet name\n  p f"hi, {name}"\ngreet("alice")'
        ), "python")
        self.assertEqual(_run_py(py).strip(), "hi, alice")


class TestCrossLang(unittest.TestCase):

    SRC = "def double x\n  return x * 2\np double(5)"

    def test_python_def(self):
        py = compile_to(parse(self.SRC), "python")
        self.assertIn("def double(x):", py)

    def test_js_function(self):
        js = compile_to(parse(self.SRC), "js")
        self.assertIn("function double(x)", js)

    def test_go_func(self):
        go = compile_to(parse(self.SRC), "go")
        self.assertIn("func double(", go)

    def test_rust_fn(self):
        rs = compile_to(parse(self.SRC), "rust")
        self.assertIn("fn double(", rs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
