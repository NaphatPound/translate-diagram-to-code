"""Tests for f-string format specs `f"{x:.2f}"`.

LLMs trained on Python emit format specs constantly. Flow now parses
them as a 3rd element on FString.parts and emits target-appropriate
formatting.
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


class TestPythonFormatSpec(unittest.TestCase):

    def test_fixed_point(self):
        self.assertEqual(_run('x = 3.14159\np f"{x:.2f}"'), "3.14")

    def test_zero_pad(self):
        self.assertEqual(_run('n = 7\np f"{n:03}"'), "007")

    def test_no_spec_unchanged(self):
        self.assertEqual(_run('x = 5\np f"x is {x}"'), "x is 5")

    def test_combined_text_and_spec(self):
        self.assertEqual(
            _run('p = 0.756\np f"score: {p:.1%}"'),
            "score: 75.6%",
        )

    def test_nested_ternary_not_split(self):
        # `?:` inside a placeholder must not be confused for a format spec.
        self.assertEqual(
            _run('p f"got {1 == 1 ? \\"y\\" : \\"n\\"}"'),
            "got y",
        )


class TestAstShape(unittest.TestCase):

    def test_format_spec_stored_in_parts(self):
        ast = parse('p f"{x:.2f}"')
        v = ast.body[0].args[0].value
        self.assertEqual(len(v.parts), 1)
        kind, expr, fmt = v.parts[0]
        self.assertEqual(kind, "expr")
        self.assertEqual(fmt, ".2f")

    def test_no_spec_stores_empty_string(self):
        ast = parse('p f"{x}"')
        v = ast.body[0].args[0].value
        _kind, _expr, fmt = v.parts[0]
        self.assertEqual(fmt, "")


class TestCrossTargetCompile(unittest.TestCase):

    def test_python_emits_native_spec(self):
        py = compile_to(parse('p f"{x:.2f}"'), "python")
        self.assertIn(':.2f', py)

    def test_rust_emits_native_spec(self):
        rs = compile_to(parse('p f"{x:.2f}"'), "rust")
        # Rust uses {:.2f}
        self.assertIn('{:.2f}', rs)

    def test_js_translates_fixed_to_tofixed(self):
        js = compile_to(parse('p f"{x:.2f}"'), "js")
        self.assertIn('.toFixed(2)', js)

    def test_go_translates_fixed_to_printf(self):
        go = compile_to(parse('p f"{x:.2f}"'), "go")
        self.assertIn('%.2f', go)


class TestRoundTrip(unittest.TestCase):

    def test_formatter_preserves_spec(self):
        from flow.formatter import format_source
        src = 'p f"{x:.2f}"'
        out = format_source(parse(src))
        self.assertIn(":.2f", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
