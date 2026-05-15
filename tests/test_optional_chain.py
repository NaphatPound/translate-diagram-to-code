"""Tests for optional chaining `?.`"""
import sys
import os
import subprocess
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, compile_to
from flow.parser import MethodCall


def _run_py(src):
    r = subprocess.run([sys.executable, "-c", src], capture_output=True,
                       text=True, timeout=4)
    if r.returncode != 0:
        raise AssertionError(f"py failed:\n{src}\n---\n{r.stderr}")
    return r.stdout


class TestOptionalChain(unittest.TestCase):

    def test_parses_with_optional_flag(self):
        ast = parse('user = {a: 1}\np user?.a')
        # The print's arg should be a MethodCall with optional=True.
        v = ast.body[1].args[0].value
        self.assertIsInstance(v, MethodCall)
        self.assertTrue(v.optional)

    def test_python_dict_get_form(self):
        py = compile_to(parse('user = {name: "alice"}\np user?.name'), "python")
        self.assertIn('.get("name")', py)
        self.assertEqual(_run_py(py).strip(), "alice")

    def test_python_missing_key_returns_none(self):
        py = compile_to(parse('u = {a: 1}\np u?.missing'), "python")
        self.assertEqual(_run_py(py).strip(), "None")

    def test_js_uses_native_optional(self):
        js = compile_to(parse('p user?.name'), "js")
        self.assertIn("?.name", js)

    def test_chain_after_optional(self):
        # `a?.b.c` — first hop optional, then plain attribute.
        py = compile_to(parse('u = {x: "ABC"}\np u?.x'), "python")
        self.assertEqual(_run_py(py).strip(), "ABC")

    def test_plain_dot_still_works(self):
        # `user.name` (no `?`) — still extends Name path; dict access via [].
        py = compile_to(parse('u = {name: "alice"}\np u.name'), "python")
        self.assertEqual(_run_py(py).strip(), "alice")


if __name__ == "__main__":
    unittest.main(verbosity=2)
