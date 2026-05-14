"""Tests for assignment (`name = expr`) and verb aliases."""
import sys
import os
import subprocess
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, compile_to, format_source, ParseError, CompileError
from flow.parser import AssignStmt, Call


def _run_python(src):
    r = subprocess.run([sys.executable, "-c", src], capture_output=True,
                       text=True, timeout=8)
    if r.returncode != 0:
        raise AssertionError(f"py failed:\n{src}\n---\n{r.stderr}")
    return r.stdout


class TestAssignParser(unittest.TestCase):

    def test_simple(self):
        ast = parse("s = 3 + 4")
        self.assertIsInstance(ast.body[0], AssignStmt)
        self.assertEqual(ast.body[0].target, "s")

    def test_distinguishes_assignment_from_named_arg(self):
        # `print value=x` is a CALL, not an assignment. `value` is an arg name.
        ast = parse("print value=5")
        self.assertIsInstance(ast.body[0], Call)

    def test_bad_target(self):
        # Numbers can't be targets — parser sees number first then '='. Falls
        # through to pipeline parsing which errors.
        with self.assertRaises(ParseError):
            parse("123 = 4")


class TestAssignCompile(unittest.TestCase):

    def test_simple_addition_runs(self):
        py = compile_to(parse("s = 3 + 4\nprint s"), "python")
        self.assertEqual(_run_python(py).strip(), "7")

    def test_funccall_rhs(self):
        py = compile_to(
            parse("items = [1, 2, 3]\nn = count(items)\nprint n"), "python")
        self.assertEqual(_run_python(py).strip(), "3")

    def test_chain_of_assignments(self):
        py = compile_to(parse(
            "a = 2\nb = 3\nc = a + b\nprint c"
        ), "python")
        self.assertEqual(_run_python(py).strip(), "5")

    def test_js_uses_let(self):
        js = compile_to(parse("s = 5"), "js")
        self.assertIn("let s = 5;", js)

    def test_go_uses_var(self):
        go = compile_to(parse("s = 5"), "go")
        self.assertIn("var s = 5", go)

    def test_rust_uses_let(self):
        rs = compile_to(parse("s = 5"), "rust")
        self.assertIn("let s = 5;", rs)

    def test_bash_arith(self):
        sh = compile_to(parse("s = 3 + 4"), "bash")
        self.assertIn("s=$(( (3 + 4) ))", sh)


class TestVerbAliases(unittest.TestCase):

    def test_p_for_print(self):
        py = compile_to(parse('p "hi"'), "python")
        self.assertEqual(_run_python(py).strip(), "hi")

    def test_r_for_read(self):
        # Use a path that exists (/etc/hostname or similar) — skip if not present
        path = "/etc/hostname"
        if not os.path.exists(path):
            self.skipTest("no /etc/hostname")
        py = compile_to(parse(f'r "{path}" -> t\np t'), "python")
        # Just check it compiled with `open(...)`
        self.assertIn(f'open("{path}").read()', py)

    def test_f_for_filter(self):
        py = compile_to(parse('items = [1,2,3,4,5]\nf items where="x > 2" -> big\np big'), "python")
        self.assertEqual(_run_python(py).strip(), "[3, 4, 5]")

    def test_aliases_normalized_in_formatter(self):
        # `p "hi"` should format back to canonical `print value="hi"`.
        fmt = format_source(parse('p "hi"'))
        self.assertIn("print", fmt)
        self.assertNotIn("\np ", fmt)


class TestRoundTripWithSugar(unittest.TestCase):

    def test_pipeline_lowering_round_trips(self):
        ast1 = parse('read "x" | upper | print')
        # Three Calls
        self.assertEqual(len(ast1.body), 3)
        # Last call's primary arg should reference second call's output.
        fmt = format_source(ast1)
        re_parsed = parse(fmt)
        # Structurally same number of statements.
        self.assertEqual(len(ast1.body), len(re_parsed.body))


if __name__ == "__main__":
    unittest.main(verbosity=2)
