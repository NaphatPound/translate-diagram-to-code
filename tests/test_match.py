"""Tests for the match / case statement."""
import sys
import os
import subprocess
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, compile_to
from flow.parser import MatchStmt


def _run_py(src):
    r = subprocess.run([sys.executable, "-c", src], capture_output=True,
                       text=True, timeout=4)
    if r.returncode != 0:
        raise AssertionError(f"py failed:\n{src}\n---\n{r.stderr}")
    return r.stdout


SRC = (
    'def grade s\n'
    '  match s\n'
    '    case "ok"\n'
    '      return "good"\n'
    '    case "err"\n'
    '      return "bad"\n'
    '    else\n'
    '      return "unknown"\n'
    'p grade("ok")\n'
    'p grade("err")\n'
    'p grade("???")\n'
)


class TestMatchParse(unittest.TestCase):

    def test_parses(self):
        ast = parse(SRC)
        d = ast.body[0]
        m = d.body[0]
        self.assertIsInstance(m, MatchStmt)
        self.assertEqual(len(m.cases), 2)
        self.assertIsNotNone(m.else_body)

    def test_no_cases_errors(self):
        from flow.parser import ParseError
        with self.assertRaises(ParseError):
            parse("match x\n")


class TestMatchPython(unittest.TestCase):

    def test_runs_correctly(self):
        py = compile_to(parse(SRC), "python")
        out = _run_py(py).strip().split("\n")
        self.assertEqual(out, ["good", "bad", "unknown"])

    def test_emits_if_elif_else(self):
        py = compile_to(parse(SRC), "python")
        self.assertIn("if", py)
        self.assertIn("elif", py)
        self.assertIn("else:", py)


class TestMatchOtherLangs(unittest.TestCase):

    def test_js_has_chained_else_if(self):
        js = compile_to(parse(SRC), "js")
        # Each arm should be `} else if (...) {`, NOT just `else if`.
        self.assertIn("} else if (", js)

    def test_bash_uses_double_bracket(self):
        sh = compile_to(parse(SRC), "bash")
        self.assertIn("[[", sh)
        self.assertIn("fi", sh)


class TestMatchInteger(unittest.TestCase):

    def test_int_patterns(self):
        src = (
            'def kind n\n'
            '  match n\n'
            '    case 0\n'
            '      return "zero"\n'
            '    case 1\n'
            '      return "one"\n'
            '    else\n'
            '      return "many"\n'
            'p kind(0)\n'
            'p kind(5)\n'
        )
        py = compile_to(parse(src), "python")
        out = _run_py(py).strip().split("\n")
        self.assertEqual(out, ["zero", "many"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
