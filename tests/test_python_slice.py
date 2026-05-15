"""Tests for Python-style slicing `xs[a:b]` with exclusive end.

LLMs trained on Python emit `xs[1:-1]`, `xs[:3]`, `xs[2:]` constantly.
Flow now parses these as a Slice AST node (distinct from the inclusive
`..` Range form) and compiles to native slice syntax per target.
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


class TestPythonSlice(unittest.TestCase):

    def test_a_b_exclusive(self):
        # Python semantics: 2 elements (indices 1 and 2).
        self.assertEqual(_run("xs = [10, 20, 30, 40]\np xs[1:3]"),
                         "[20, 30]")

    def test_negative_end(self):
        self.assertEqual(_run("xs = [10, 20, 30, 40]\np xs[1:-1]"),
                         "[20, 30]")

    def test_to_index(self):
        self.assertEqual(_run("xs = [10, 20, 30, 40]\np xs[:2]"),
                         "[10, 20]")

    def test_from_index(self):
        self.assertEqual(_run("xs = [10, 20, 30, 40]\np xs[2:]"),
                         "[30, 40]")

    def test_full_copy(self):
        # `xs[:]` is the Python idiom for shallow copy.
        self.assertEqual(_run("xs = [1, 2, 3]\np xs[:]"),
                         "[1, 2, 3]")

    def test_string_slice(self):
        self.assertEqual(_run('s = "hello"\np s[1:4]'), "ell")


class TestInclusiveRangeStillWorks(unittest.TestCase):
    """`..` is inclusive; `:` is exclusive. Both must coexist."""

    def test_inclusive_range_slice(self):
        # 3 elements (indices 1, 2, 3) — inclusive end.
        self.assertEqual(_run("xs = [10, 20, 30, 40, 50]\np xs[1..3]"),
                         "[20, 30, 40]")

    def test_inclusive_and_exclusive_differ(self):
        out_inc = _run("xs = [10, 20, 30, 40]\np xs[1..2]")
        out_exc = _run("xs = [10, 20, 30, 40]\np xs[1:2]")
        # `..` includes index 2; `:` does not.
        self.assertEqual(out_inc, "[20, 30]")
        self.assertEqual(out_exc, "[20]")


class TestAstShape(unittest.TestCase):

    def test_colon_parses_as_slice(self):
        from flow.parser import Slice, IndexAccess
        ast = parse("p xs[1:3]")
        # Drill down: Call(p, [Arg(<pos>, IndexAccess(xs, Slice(1, 3)))])
        index = ast.body[0].args[0].value
        self.assertIsInstance(index, IndexAccess)
        self.assertIsInstance(index.index, Slice)

    def test_dotdot_still_parses_as_range(self):
        from flow.parser import Range, IndexAccess
        ast = parse("p xs[1..3]")
        index = ast.body[0].args[0].value
        self.assertIsInstance(index, IndexAccess)
        self.assertIsInstance(index.index, Range)


class TestCrossTarget(unittest.TestCase):

    def test_python_renders_native(self):
        # Define xs so the compiler treats it as a var, not a string lit.
        py = compile_to(parse("xs = [1,2,3,4]\np xs[1:3]"), "python")
        self.assertIn("xs[1:3]", py)

    def test_js_renders_slice_call(self):
        js = compile_to(parse("xs = [1,2,3,4]\np xs[1:3]"), "js")
        self.assertIn(".slice(1, 3)", js)

    def test_js_open_end_uses_slice_from(self):
        js = compile_to(parse("xs = [1,2,3,4]\np xs[2:]"), "js")
        self.assertIn(".slice(2)", js)

    def test_rust_renders_native_exclusive(self):
        rs = compile_to(parse("xs = [1,2,3,4]\np xs[1:3]"), "rust")
        # Rust uses `1..3` (exclusive) — note no `=`.
        self.assertIn("1..3", rs)


class TestSliceStep(unittest.TestCase):
    """`xs[::2]` (every other) and `xs[::-1]` (reverse) are universal idioms."""

    def test_every_other(self):
        self.assertEqual(_run("xs = [1, 2, 3, 4, 5]\np xs[::2]"),
                         "[1, 3, 5]")

    def test_reverse(self):
        self.assertEqual(_run("xs = [1, 2, 3]\np xs[::-1]"), "[3, 2, 1]")

    def test_string_reverse(self):
        self.assertEqual(_run('p "hello"[::-1]'), "olleh")

    def test_start_with_step(self):
        self.assertEqual(_run("xs = [1, 2, 3, 4, 5]\np xs[1::2]"),
                         "[2, 4]")

    def test_full_three_part(self):
        self.assertEqual(_run("xs = [1, 2, 3, 4, 5]\np xs[1:4:2]"),
                         "[2, 4]")


class TestFormatterRoundTrip(unittest.TestCase):

    def test_slice_formats_back(self):
        from flow.formatter import format_source
        out = format_source(parse("p xs[1:3]"))
        self.assertIn("[1:3]", out)

    def test_slice_open_end_formats(self):
        from flow.formatter import format_source
        out = format_source(parse("p xs[2:]"))
        self.assertIn("[2:]", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
