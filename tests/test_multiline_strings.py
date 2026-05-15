# Tests for triple-quoted (multi-line) strings.
import sys
import os
import subprocess
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, compile_to
from flow.parser import StringLit, FString


def _run_py(src):
    r = subprocess.run([sys.executable, "-c", src], capture_output=True,
                       text=True, timeout=4)
    if r.returncode != 0:
        raise AssertionError(f"py failed:\n{src}\n---\n{r.stderr}")
    return r.stdout


class TestTripleString(unittest.TestCase):

    def test_single_line_triple(self):
        ast = parse('p """hello"""')
        v = ast.body[0].args[0].value
        self.assertIsInstance(v, StringLit)
        self.assertEqual(v.value, "hello")

    def test_multi_line_triple(self):
        src = 'msg = """hello\nworld"""\np msg'
        py = compile_to(parse(src), "python")
        self.assertEqual(_run_py(py).strip(), "hello\nworld")

    def test_triple_in_def(self):
        # Docstring-style use at top of a function body. Bare strings at
        # statement position become ExprStmt; non-terminal ones are simply
        # evaluated and discarded (like Python docstrings).
        src = (
            'def greet name\n'
            '  """Returns a greeting for the given name."""\n'
            '  f"hi, {name}"\n'
            'p greet("alice")'
        )
        py = compile_to(parse(src), "python")
        self.assertEqual(_run_py(py).strip(), "hi, alice")

    def test_triple_fstring(self):
        src = 'name = "alice"\nnote = f"""dear {name},\nbye"""\np note'
        py = compile_to(parse(src), "python")
        self.assertEqual(_run_py(py).strip(), "dear alice,\nbye")

    def test_line_numbers_preserved(self):
        # Multi-line string spans 3 lines; the `p msg` after should report
        # line 4, not line 2.
        src = 'msg = """a\nb\nc"""\np intentionally_bad +'
        try:
            parse(src)
            self.fail("expected ParseError")
        except Exception as e:
            # The error should reference line 4 (the broken statement).
            self.assertIn("line 4", str(e))


class TestTripleQuoteInsideRegularString(unittest.TestCase):

    def test_escaped_quotes_in_regular_string_unaffected(self):
        # A regular string containing escaped quotes should NOT be mistaken
        # for a triple-quote opener.
        ast = parse('p "say \\"hi\\""')
        # Just verify it parses.
        self.assertIsNotNone(ast)


if __name__ == "__main__":
    unittest.main(verbosity=2)
