"""Tests for implicit return inside def's terminal match/if branches."""
import sys
import os
import subprocess
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, compile_to


def _run_py(src):
    r = subprocess.run([sys.executable, "-c", src], capture_output=True,
                       text=True, timeout=4)
    if r.returncode != 0:
        raise AssertionError(f"py failed:\n{src}\n---\n{r.stderr}")
    return r.stdout


class TestImplicitReturnInMatch(unittest.TestCase):

    def test_case_body_implicit_return(self):
        src = (
            'def grade s\n'
            '  match s\n'
            '    case "ok"\n'
            '      "good"\n'
            '    case "err"\n'
            '      "bad"\n'
            '    else\n'
            '      "unknown"\n'
            'p grade("ok")\n'
            'p grade("err")\n'
            'p grade("???")\n'
        )
        py = compile_to(parse(src), "python")
        out = _run_py(py).strip().split("\n")
        self.assertEqual(out, ["good", "bad", "unknown"])


class TestImplicitReturnInIf(unittest.TestCase):

    def test_both_branches_bare(self):
        src = (
            'def abs_val n\n'
            '  if n < 0\n'
            '    -n\n'
            '  else\n'
            '    n\n'
            'p abs_val(-7)\np abs_val(7)\n'
        )
        py = compile_to(parse(src), "python")
        out = _run_py(py).strip().split("\n")
        self.assertEqual(out, ["7", "7"])

    def test_nested_if_branches(self):
        src = (
            'def classify x\n'
            '  if x > 10\n'
            '    "big"\n'
            '  else\n'
            '    if x > 0\n'
            '      "small"\n'
            '    else\n'
            '      "zero"\n'
            'p classify(20)\np classify(3)\np classify(0)\n'
        )
        py = compile_to(parse(src), "python")
        out = _run_py(py).strip().split("\n")
        self.assertEqual(out, ["big", "small", "zero"])


class TestBareVariableAsLast(unittest.TestCase):

    def test_bare_variable_implicit_return(self):
        # `n` alone is parsed as Call(n, [], None) — the compiler should
        # reinterpret as variable reference since `n` isn't a registered verb.
        src = "def id x\n  x\np id(42)\n"
        py = compile_to(parse(src), "python")
        self.assertEqual(_run_py(py).strip(), "42")


if __name__ == "__main__":
    unittest.main(verbosity=2)
