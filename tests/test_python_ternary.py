"""Tests for Python-style `X if COND else Y` ternary expressions.

LLMs trained on Python emit this constantly. The parser disambiguates
it from postfix-if (statement modifier) by looking ahead for KW_ELSE
on the same line.
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


class TestPythonTernary(unittest.TestCase):

    def test_basic(self):
        self.assertEqual(_run('p "big" if 5 > 3 else "small"'), "big")
        self.assertEqual(_run('p "big" if 1 > 3 else "small"'), "small")

    def test_in_parens(self):
        self.assertEqual(
            _run('x = 5\np ("big" if x > 3 else "small")'), "big")

    def test_in_assignment(self):
        self.assertEqual(
            _run('x = 5\nlabel = "big" if x > 3 else "small"\np label'),
            "big",
        )

    def test_in_def_body(self):
        self.assertEqual(
            _run('def grade s\n  "pass" if s >= 50 else "fail"\np grade(80)'),
            "pass",
        )

    def test_compiles_to_python_ternary(self):
        py = compile_to(parse('p "big" if 5 > 3 else "small"'), "python")
        # Python's ternary natively renders as `(then if cond else else)`.
        self.assertIn(" if ", py)
        self.assertIn(" else ", py)

    def test_compiles_to_js_ternary(self):
        js = compile_to(parse('p "big" if 5 > 3 else "small"'), "js")
        # JS uses `?:` for ternary.
        self.assertIn(" ? ", js)
        self.assertIn(" : ", js)


class TestPostfixIfStillWorks(unittest.TestCase):
    """Disambiguation: when no `else` follows, `X if COND` is postfix-if."""

    def test_postfix_if_no_else(self):
        self.assertEqual(_run('p "x" if 1 == 1'), "x")

    def test_postfix_if_false(self):
        self.assertEqual(_run('p "x" if 1 == 2'), "")  # no output

    def test_postfix_unless_unchanged(self):
        self.assertEqual(_run('p "y" unless 1 == 0'), "y")


class TestNestedAndInteraction(unittest.TestCase):

    def test_nested_python_ternary(self):
        out = _run(
            'def grade s\n'
            '  "A" if s >= 90 else "B" if s >= 80 else "C"\n'
            'p grade(95)\n'
            'p grade(85)\n'
            'p grade(75)'
        )
        self.assertEqual(out.splitlines(), ["A", "B", "C"])

    def test_qmark_ternary_still_works(self):
        # Both styles must coexist.
        self.assertEqual(_run('p (5 > 3 ? "big" : "small")'), "big")


if __name__ == "__main__":
    unittest.main(verbosity=2)
