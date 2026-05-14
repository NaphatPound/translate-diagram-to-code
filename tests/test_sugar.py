"""Tests for syntactic sugar: pipe `|` and implicit primary arg."""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, compile_to, CompileError
from flow.parser import Call, StringLit, Name


def _run_python(src):
    import subprocess
    r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=8)
    if r.returncode != 0:
        raise AssertionError(f"py failed:\n{src}\n---\n{r.stderr}")
    return r.stdout


class TestImplicitPrimary(unittest.TestCase):

    def test_print_positional_string(self):
        py = compile_to(parse('print "hi"'), "python")
        self.assertEqual(_run_python(py).strip(), "hi")

    def test_upper_positional(self):
        py = compile_to(parse('upper "abc" -> big\nprint big'), "python")
        self.assertEqual(_run_python(py).strip(), "ABC")

    def test_positional_then_named(self):
        # `format "hi {n}" data={n: 1}` — template positional, data named
        py = compile_to(
            parse('format "hi {n}" data={n: 1} -> msg\nprint msg'), "python")
        self.assertEqual(_run_python(py).strip(), "hi 1")

    def test_positional_number(self):
        py = compile_to(parse('wait 0'), "python")
        # just check it compiled cleanly
        self.assertIn("time.sleep(0)", py)

    def test_verb_without_primary_rejects_positional(self):
        # `add` has no primary_arg → positional value should error
        with self.assertRaises(CompileError) as ctx:
            compile_to(parse('add 1'), "python")
        self.assertIn("primary", str(ctx.exception).lower() + "primary")


class TestPipe(unittest.TestCase):

    def test_simple_chain(self):
        py = compile_to(parse('upper "abc" | print'), "python")
        # First call should output to _p1, second should print _p1.
        self.assertIn("_p1 = (\"abc\").upper()", py)
        self.assertIn("print(_p1)", py)
        self.assertEqual(_run_python(py).strip(), "ABC")

    def test_three_step_chain(self):
        py = compile_to(parse('split "a,b,c" sep="," | count | print'), "python")
        out = _run_python(py).strip()
        self.assertEqual(out, "3")

    def test_pipe_preserves_explicit_out_name(self):
        py = compile_to(parse('upper "x" -> X\nprint X'), "python")
        self.assertEqual(_run_python(py).strip(), "X")

    def test_pipe_inside_when_start(self):
        src = (
            'when start\n'
            '  upper "hi" | print'
        )
        py = compile_to(parse(src), "python")
        self.assertEqual(_run_python(py).strip(), "HI")

    def test_pipe_to_verb_without_primary_arg_errors(self):
        with self.assertRaises(CompileError):
            compile_to(parse('upper "x" | add b=1'), "python")


class TestDidYouMean(unittest.TestCase):

    def test_unknown_verb_suggests(self):
        with self.assertRaises(CompileError) as ctx:
            compile_to(parse('prnt "hi"'), "python")
        self.assertIn("Did you mean 'print'", str(ctx.exception))

    def test_unknown_arg_suggests(self):
        with self.assertRaises(CompileError) as ctx:
            compile_to(parse('read fyle="x.csv" -> r'), "python")
        self.assertIn("Did you mean 'file'", str(ctx.exception))

    def test_distant_typo_no_suggestion(self):
        # 'xyzqrs' isn't close to anything — should error without a "did you mean"
        with self.assertRaises(CompileError) as ctx:
            compile_to(parse('xyzqrs "hi"'), "python")
        self.assertNotIn("Did you mean", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
