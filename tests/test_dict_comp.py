"""Tests for Python-style dict comprehensions.

`{k_expr: v_expr for var in source (if cond)}`. LLMs use these to
transform/filter dicts; the parser disambiguates from dict literals
by scanning ahead for a top-level `for` between the braces.
"""
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


class TestDictComp(unittest.TestCase):

    def test_basic(self):
        self.assertEqual(_run("p {x: x*2 for x in [1, 2, 3]}"),
                         "{1: 2, 2: 4, 3: 6}")

    def test_filtered(self):
        self.assertEqual(_run("p {x: x*x for x in [1, 2, 3, 4] if x > 2}"),
                         "{3: 9, 4: 16}")

    def test_loop_var_can_be_anything(self):
        self.assertEqual(_run('p {item: 1 for item in ["a", "b"]}'),
                         "{'a': 1, 'b': 1}")

    def test_from_range_source(self):
        self.assertEqual(_run("p {i: i+1 for i in 1..3}"),
                         "{1: 2, 2: 3, 3: 4}")

    def test_complex_key_expr(self):
        # Key can be an expression like `upper(item)`.
        out = _run('p {upper(s): len(s) for s in ["hi", "yo"]}')
        self.assertEqual(out, "{'HI': 2, 'YO': 2}")


class TestDictLiteralStillWorks(unittest.TestCase):

    def test_simple_literal(self):
        self.assertEqual(_run('p {"a": 1, "b": 2}'), "{'a': 1, 'b': 2}")

    def test_empty_dict(self):
        self.assertEqual(_run("p {}"), "{}")

    def test_ident_key(self):
        # Ident key still treated as string (legacy Flow behavior).
        self.assertEqual(_run("p {name: 1}"), "{'name': 1}")


class TestAstShape(unittest.TestCase):

    def test_dictcomp_node_used(self):
        from flow.parser import DictComp
        ast = parse("p {x: x*2 for x in [1, 2, 3]}")
        comp = ast.body[0].args[0].value
        self.assertIsInstance(comp, DictComp)
        self.assertEqual(comp.var, "x")
        self.assertIsNone(comp.cond)


class TestCrossTarget(unittest.TestCase):

    def test_python_native(self):
        py = compile_to(parse("p {x: x for x in [1]}"), "python")
        self.assertIn("for x in", py)
        self.assertIn(":", py)

    def test_js_uses_fromentries(self):
        js = compile_to(parse("p {x: x*2 for x in [1, 2]}"), "js")
        self.assertIn("Object.fromEntries", js)
        self.assertIn(".map(x =>", js)


class TestFormatter(unittest.TestCase):

    def test_round_trips(self):
        from flow.formatter import format_source
        src = "p {x: x*2 for x in xs}"
        out = format_source(parse(src))
        self.assertIn("for x in", out)
        self.assertIn(":", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
