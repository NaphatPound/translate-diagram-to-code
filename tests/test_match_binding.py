"""Tests for binding patterns in match.

A trailing `case <name>` where <name> isn't already in scope binds the
matched value to that name in the body. Replaces the verbose
`else / x = matched_value / ...` idiom.
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


class TestMatchBindingPattern(unittest.TestCase):

    def test_binding_after_literal_cases(self):
        src = (
            'def label code\n'
            '  match code\n'
            '    case "ok"\n'
            '      "yes"\n'
            '    case other\n'
            '      f"got: {other}"\n'
            'p label("ok")\n'
            'p label("err")'
        )
        out = _run(src)
        self.assertEqual(out.splitlines(), ["yes", "got: err"])

    def test_binding_as_only_case(self):
        out = _run("match 7\n  case n\n    p n")
        self.assertEqual(out, "7")

    def test_in_scope_name_falls_back_to_literal(self):
        # `match` literal-compares against `target` because `target` is in scope.
        out = _run(
            'target = "hello"\n'
            'match "hello"\n'
            '  case target\n'
            '    p "matched"\n'
            '  else\n'
            '    p "no"'
        )
        self.assertEqual(out, "matched")

    def test_classic_else_still_supported(self):
        out = _run(
            'match "hi"\n'
            '  case "hello"\n'
            '    p "yes"\n'
            '  else\n'
            '    p "fallback"'
        )
        self.assertEqual(out, "fallback")

    def test_binding_nested_in_def(self):
        # Binding pattern inside a def — captured name is visible in the
        # case body only (lexically; runtime semantics).
        out = _run(
            'def first_or_default xs\n'
            '  match count(xs)\n'
            '    case 0\n'
            '      "empty"\n'
            '    case n\n'
            '      f"len={n}"\n'
            'p first_or_default([])\n'
            'p first_or_default([1, 2, 3])'
        )
        self.assertEqual(out.splitlines(), ["empty", "len=3"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
