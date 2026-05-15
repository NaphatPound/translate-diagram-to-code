"""Tests for the `in` binary operator at expression level."""
import sys
import os
import subprocess
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, compile_to


def _run(src: str) -> str:
    py = compile_to(parse(src), "python")
    r = subprocess.run([sys.executable, "-c", py],
                       capture_output=True, text=True, timeout=4)
    return r.stdout.strip() if r.returncode == 0 else f"ERR: {r.stderr.strip()}"


class TestInOperator(unittest.TestCase):

    def test_in_list_true(self):
        self.assertEqual(_run('p ("a" in ["a", "b", "c"])'), "True")

    def test_in_list_false(self):
        self.assertEqual(_run('p ("z" in ["a", "b", "c"])'), "False")

    def test_in_string(self):
        self.assertEqual(_run('p ("ell" in "hello")'), "True")

    def test_in_with_keys(self):
        out = _run('d = {"x": 1, "y": 2}\np ("x" in keys(d))')
        self.assertEqual(out, "True")

    def test_in_in_if_cond(self):
        out = _run("if 5 in [1, 2, 5]\n  p \"found\"")
        self.assertEqual(out, "found")

    def test_not_in(self):
        out = _run("p (!(\"z\" in [\"a\", \"b\"]))")
        # The negation simplifier could rewrite `not (x in ys)` but it
        # isn't in the inversion table — just ensure runtime correctness.
        self.assertEqual(out, "True")

    def test_each_loop_still_works(self):
        # KW_IN is now both a binop AND the each-keyword. Each must still parse.
        out = _run("xs = [1, 2, 3]\neach x in xs\n  p x")
        self.assertEqual(out.split(), ["1", "2", "3"])

    def test_in_combines_with_and(self):
        out = _run('p ("a" in ["a", "b"] and "z" in ["a", "b"])')
        self.assertEqual(out, "False")


class TestInOperatorJS(unittest.TestCase):
    """Smoke test: JS path renders `.includes()`."""

    def test_js_uses_includes(self):
        js = compile_to(parse('p ("a" in xs)'), "js")
        self.assertIn(".includes(", js)


if __name__ == "__main__":
    unittest.main(verbosity=2)
