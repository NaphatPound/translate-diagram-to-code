"""Tests for implicit return (bare expression at end of def body)."""
import sys
import os
import subprocess
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, compile_to
from flow.parser import DefStmt, ReturnStmt, ExprStmt


def _run_py(src):
    r = subprocess.run([sys.executable, "-c", src], capture_output=True,
                       text=True, timeout=4)
    if r.returncode != 0:
        raise AssertionError(f"py failed:\n{src}\n---\n{r.stderr}")
    return r.stdout


class TestImplicitReturn(unittest.TestCase):

    def test_binop_implicit_return(self):
        py = compile_to(parse("def double x\n  x * 2\np double(5)"), "python")
        self.assertEqual(_run_py(py).strip(), "10")
        # AST keeps the ExprStmt; implicit-return is handled at compile time
        # so lint can distinguish explicit-return from already-implicit.
        ast = parse("def double x\n  x * 2")
        d = ast.body[0]
        self.assertIsInstance(d.body[-1], ExprStmt)

    def test_method_call_implicit_return(self):
        py = compile_to(parse('def shout s\n  s.upper()\np shout("hi")'), "python")
        self.assertEqual(_run_py(py).strip(), "HI")

    def test_comparison_implicit_return(self):
        py = compile_to(parse(
            "def is_positive x\n  x > 0\np is_positive(5)\np is_positive(-1)"
        ), "python")
        self.assertEqual(_run_py(py).strip().split("\n"), ["True", "False"])

    def test_compound_logical_implicit_return(self):
        py = compile_to(parse(
            "def in_range x lo hi\n  x >= lo and x <= hi\np in_range(7, 1, 10)"
        ), "python")
        self.assertEqual(_run_py(py).strip(), "True")

    def test_f_string_implicit_return(self):
        py = compile_to(parse(
            'def greet name\n  f"hi, {name}"\np greet("alice")'
        ), "python")
        self.assertEqual(_run_py(py).strip(), "hi, alice")

    def test_explicit_return_still_works(self):
        py = compile_to(parse("def double x\n  return x * 2\np double(5)"), "python")
        self.assertEqual(_run_py(py).strip(), "10")

    def test_non_terminal_expr_stmt_is_not_returned(self):
        # AST preserves source: both bare expressions stay ExprStmt. The
        # compiler is responsible for emitting `return` for the LAST one.
        ast = parse("def f x\n  x + 1\n  x * 2")
        body = ast.body[0].body
        self.assertIsInstance(body[0], ExprStmt)
        self.assertIsInstance(body[1], ExprStmt)

    def test_compiler_emits_return_for_last_bare_expr(self):
        py = compile_to(parse("def f x\n  x + 1\n  x * 2"), "python")
        # Last bare expression at the def's body is the return.
        self.assertIn("return (x * 2)", py)
        # Earlier bare expression is plain (no return).
        self.assertIn("(x + 1)", py)
        # ...but not preceded by `return` on its own line.
        lines = [l.strip() for l in py.splitlines()]
        x_plus_1_line = next(l for l in lines if "(x + 1)" in l)
        self.assertNotIn("return", x_plus_1_line)


class TestBareExprStmt(unittest.TestCase):

    def test_method_call_at_top_level(self):
        # `s.upper()` at top level — bare expression statement.
        ast = parse('s = "hi"\ns.upper()')
        self.assertIsInstance(ast.body[1], ExprStmt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
