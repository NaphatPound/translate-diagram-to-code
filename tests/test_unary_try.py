"""Tests for unary not (`!`, `not`) and try/catch."""
import sys
import os
import subprocess
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, compile_to
from flow.parser import UnaryOp, TryStmt


def _run_py(src):
    r = subprocess.run([sys.executable, "-c", src], capture_output=True,
                       text=True, timeout=4)
    if r.returncode != 0:
        raise AssertionError(f"py failed:\n{src}\n---\n{r.stderr}")
    return r.stdout


class TestUnaryNot(unittest.TestCase):

    def test_bang_parses_as_unary(self):
        ast = parse("p !true")
        v = ast.body[0].args[0].value
        self.assertIsInstance(v, UnaryOp)
        self.assertEqual(v.op, "not")

    def test_not_keyword_parses_same(self):
        ast = parse("p not true")
        v = ast.body[0].args[0].value
        self.assertIsInstance(v, UnaryOp)

    def test_negation_in_condition_runs(self):
        py = compile_to(parse('b = false\nif !b\n  p "true"'), "python")
        self.assertEqual(_run_py(py).strip(), "true")

    def test_python_uses_not(self):
        py = compile_to(parse("p !true"), "python")
        self.assertIn("not True", py)

    def test_js_uses_bang(self):
        js = compile_to(parse("p !true"), "js")
        self.assertIn("(!true)", js)


class TestTryCatch(unittest.TestCase):

    SRC = (
        'try\n'
        '  r "/no/such/path" -> data\n'
        '  p data\n'
        'catch e\n'
        '  p "got err"\n'
    )

    def test_parses_as_try_stmt(self):
        ast = parse(self.SRC)
        self.assertIsInstance(ast.body[0], TryStmt)
        self.assertEqual(ast.body[0].catch_var, "e")

    def test_python_emits_try_except(self):
        py = compile_to(parse(self.SRC), "python")
        self.assertIn("try:", py)
        self.assertIn("except Exception as e:", py)

    def test_python_runs_catch_branch(self):
        py = compile_to(parse(self.SRC), "python")
        # /no/such/path will raise FileNotFoundError → catch fires.
        self.assertIn("got err", _run_py(py))

    def test_js_emits_try_catch(self):
        js = compile_to(parse(self.SRC), "js")
        self.assertIn("try {", js)
        self.assertIn("} catch (e) {", js)

    def test_bare_catch_no_var(self):
        src = (
            'try\n'
            '  r "/no/x" -> data\n'
            'catch\n'
            '  p "err"\n'
        )
        py = compile_to(parse(src), "python")
        self.assertIn("except Exception as _e:", py)


if __name__ == "__main__":
    unittest.main(verbosity=2)
