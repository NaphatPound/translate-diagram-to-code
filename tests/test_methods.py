"""Tests for method-call syntax and JS require deduplication."""
import sys
import os
import subprocess
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, compile_to
from flow.parser import FuncCall, MethodCall


def _run_py(src):
    r = subprocess.run([sys.executable, "-c", src], capture_output=True,
                       text=True, timeout=4)
    if r.returncode != 0:
        raise AssertionError(f"py failed:\n{src}\n---\n{r.stderr}")
    return r.stdout


class TestMethodCall(unittest.TestCase):

    def test_parses_method_call(self):
        ast = parse('p name.upper()')
        v = ast.body[0].args[0].value
        self.assertIsInstance(v, MethodCall)
        self.assertEqual(v.method, "upper")

    def test_python_method_call(self):
        py = compile_to(parse('name = "hello"\np name.upper()'), "python")
        self.assertIn("name.upper()", py)
        self.assertEqual(_run_py(py).strip(), "HELLO")

    def test_method_with_args(self):
        # Method calls on string literals aren't supported (no DOT token after
        # STRING). Use a variable receiver, which IS supported because WORD
        # tokens already allow dots: `s.split` parses as a single dotted name.
        py = compile_to(parse('s = "a,b,c"\np s.split(",")'), "python")
        self.assertIn('s.split(",")', py)
        self.assertEqual(_run_py(py).strip(), "['a', 'b', 'c']")

    def test_method_returns_value_for_assignment(self):
        py = compile_to(parse('s = "hi"\nbig = s.upper()\np big'), "python")
        self.assertEqual(_run_py(py).strip(), "HI")


class TestJsRequireHoist(unittest.TestCase):

    def test_single_require_hoisted(self):
        js = compile_to(parse('r "a.txt" -> a\np a'), "js")
        # First line should be the require header.
        self.assertTrue(js.lstrip().startswith("const _fs ="))
        # Inline `require('fs')` should be gone.
        self.assertNotIn("require('fs')", js[js.find("\n\n"):])

    def test_duplicate_modules_dedupe(self):
        src = ('when start\n'
               '  read "a.txt" -> a\n'
               '  read "b.txt" -> b')
        js = compile_to(parse(src), "js")
        self.assertEqual(js.count("const _fs = require('fs')"), 1)

    def test_multiple_distinct_modules(self):
        src = ('when start\n'
               '  read "a.txt" -> a\n'
               '  ask "name? " -> n')
        js = compile_to(parse(src), "js")
        self.assertIn("const _fs = require('fs')", js)
        self.assertIn("const _readline_sync = require('readline-sync')", js)


if __name__ == "__main__":
    unittest.main(verbosity=2)
