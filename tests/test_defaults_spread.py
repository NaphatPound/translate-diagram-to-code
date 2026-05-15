"""Tests for default args and spread `*xs`."""
import sys
import os
import subprocess
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, compile_to
from flow.parser import Spread, FuncCall, ListLit


def _run_py(src):
    r = subprocess.run([sys.executable, "-c", src], capture_output=True,
                       text=True, timeout=4)
    if r.returncode != 0:
        raise AssertionError(f"py failed:\n{src}\n---\n{r.stderr}")
    return r.stdout


class TestDefaults(unittest.TestCase):

    def test_default_used_when_omitted(self):
        py = compile_to(parse(
            "def scale x factor=2\n  return x * factor\np scale(5)"
        ), "python")
        self.assertEqual(_run_py(py).strip(), "10")

    def test_default_overridden_when_passed(self):
        py = compile_to(parse(
            "def scale x factor=2\n  return x * factor\np scale(5, 3)"
        ), "python")
        self.assertEqual(_run_py(py).strip(), "15")

    def test_python_default_in_source(self):
        py = compile_to(parse(
            "def f x y=10\n  return x + y"
        ), "python")
        self.assertIn("def f(x, y=10):", py)

    def test_js_default_in_source(self):
        js = compile_to(parse(
            "def f x y=10\n  return x + y"
        ), "js")
        self.assertIn("function f(x, y = 10)", js)

    def test_string_default(self):
        py = compile_to(parse(
            'def greet name greeting="hi"\n  p f"{greeting}, {name}"\ngreet("alice")'
        ), "python")
        self.assertEqual(_run_py(py).strip(), "hi, alice")


class TestSpread(unittest.TestCase):

    def test_spread_in_list_literal(self):
        ast = parse("xs = [1, 2]\np [*xs, 3]")
        # Second body element is a Call to print with positional ListLit value.
        call = ast.body[1]
        lst = call.args[0].value
        self.assertIsInstance(lst, ListLit)
        self.assertIsInstance(lst.items[0], Spread)

    def test_spread_concat_list(self):
        py = compile_to(parse(
            "xs = [1, 2, 3]\np [*xs, 4, 5]"
        ), "python")
        self.assertEqual(_run_py(py).strip(), "[1, 2, 3, 4, 5]")

    def test_spread_in_funccall(self):
        py = compile_to(parse(
            "xs = [3, 1, 4, 1, 5]\np max(*xs)"
        ), "python")
        self.assertEqual(_run_py(py).strip(), "5")

    def test_js_uses_three_dots(self):
        js = compile_to(parse('xs = [1, 2]\np [*xs, 3]'), "js")
        self.assertIn("...xs", js)

    def test_python_uses_single_star(self):
        py = compile_to(parse('xs = [1, 2]\np [*xs, 3]'), "python")
        self.assertIn("[*xs, 3]", py)


if __name__ == "__main__":
    unittest.main(verbosity=2)
