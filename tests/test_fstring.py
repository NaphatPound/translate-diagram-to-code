"""Tests for f-strings and Python auto-import hoisting."""
import sys
import os
import subprocess
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, compile_to, format_source
from flow.parser import FString


def _run_py(src):
    r = subprocess.run([sys.executable, "-c", src], capture_output=True,
                       text=True, timeout=4)
    if r.returncode != 0:
        raise AssertionError(f"py failed:\n{src}\n---\n{r.stderr}")
    return r.stdout


class TestFStringParse(unittest.TestCase):

    def test_no_placeholders(self):
        ast = parse('p f"hello"')
        v = ast.body[0].args[0].value
        self.assertIsInstance(v, FString)
        self.assertEqual(v.parts, [("text", "hello")])

    def test_single_placeholder(self):
        ast = parse('p f"hi {name}"')
        v = ast.body[0].args[0].value
        self.assertEqual(v.parts, [("text", "hi "), ("var", "name")])

    def test_multiple_placeholders(self):
        ast = parse('p f"hi {a}, age {b}!"')
        v = ast.body[0].args[0].value
        self.assertEqual(v.parts, [("text", "hi "), ("var", "a"),
                                    ("text", ", age "), ("var", "b"),
                                    ("text", "!")])


class TestFStringCompile(unittest.TestCase):

    def test_python_fstring(self):
        py = compile_to(parse('name = "alice"\nage = 30\np f"hi {name}, age {age}"'),
                        "python")
        self.assertIn('f"hi {name}, age {age}"', py)
        self.assertEqual(_run_py(py).strip(), "hi alice, age 30")

    def test_js_template_literal(self):
        js = compile_to(parse('name = "x"\np f"hi {name}"'), "js")
        self.assertIn("`hi ${name}`", js)

    def test_go_sprintf(self):
        go = compile_to(parse('name = "x"\np f"hi {name}"'), "go")
        self.assertIn('fmt.Sprintf("hi %v", name)', go)

    def test_rust_format_macro(self):
        rs = compile_to(parse('name = "x"\np f"hi {name}"'), "rust")
        self.assertIn('format!("hi {}", name)', rs)

    def test_bash_var_interp(self):
        sh = compile_to(parse('name = "x"\np f"hi {name}"'), "bash")
        self.assertIn('"hi ${name}"', sh)

    def test_round_trip_format(self):
        out = format_source(parse('p f"hi {name}, {age}"'))
        self.assertIn('f"hi {name}, {age}"', out)


class TestPythonImportHoisting(unittest.TestCase):

    def test_imports_moved_to_top(self):
        py = compile_to(parse('when start\n  http_get "https://x.com" -> a'),
                        "python")
        lines = py.strip().splitlines()
        self.assertTrue(lines[0].startswith("import "))

    def test_duplicate_imports_dedup(self):
        src = ('when start\n'
               '  http_get "https://a.com" -> a\n'
               '  http_get "https://b.com" -> b')
        py = compile_to(parse(src), "python")
        # `import requests as _r` should appear exactly once.
        self.assertEqual(py.count("import requests as _r"), 1)

    def test_multiple_distinct_imports(self):
        src = ('when start\n'
               '  http_get "https://x.com" -> a\n'
               '  load "data.json" -> d\n'
               '  wait seconds=1')
        py = compile_to(parse(src), "python")
        self.assertIn("import requests as _r", py)
        self.assertIn("import json as _json", py)
        self.assertIn("import time as _time", py)

    def test_no_extraneous_imports_when_none_needed(self):
        py = compile_to(parse('p "hi"'), "python")
        self.assertNotIn("import ", py)


class TestCharSavings(unittest.TestCase):

    def test_fstring_shorter_than_format_verb(self):
        verbose = 'format template="hi {name}" data={name: name} -> msg\nprint msg'
        compact = 'p f"hi {name}"'
        self.assertLess(len(compact), len(verbose) / 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
