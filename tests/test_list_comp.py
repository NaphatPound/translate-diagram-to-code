"""Tests for Python-style list comprehensions.

`[expr for var in source]` and `[expr for var in source if cond]`.
LLMs trained on Python emit these constantly; previously required
rewriting as `map`/`filter` verb chains.
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


class TestListComp(unittest.TestCase):

    def test_basic_map(self):
        self.assertEqual(_run("p [x * 2 for x in [1, 2, 3]]"), "[2, 4, 6]")

    def test_filter_then_map(self):
        self.assertEqual(
            _run("p [x * 2 for x in [1, 2, 3, 4] if x > 1]"),
            "[4, 6, 8]",
        )

    def test_loop_var_can_be_anything(self):
        self.assertEqual(_run("p [i * i for i in [1, 2, 3]]"), "[1, 4, 9]")

    def test_iterates_named_list(self):
        out = _run("xs = [1, 2, 3]\np [x + 10 for x in xs]")
        self.assertEqual(out, "[11, 12, 13]")

    def test_iterates_dict_values_via_funccall(self):
        out = _run(
            'items = [{"v": 1}, {"v": 2}]\n'
            'p [item["v"] for item in items]'
        )
        self.assertEqual(out, "[1, 2]")

    def test_iterates_range_source(self):
        self.assertEqual(_run("p [i for i in 1..3]"), "[1, 2, 3]")

    def test_works_in_assignment(self):
        out = _run(
            "xs = [1, 2, 3]\n"
            "ys = [x + 1 for x in xs]\n"
            "p ys"
        )
        self.assertEqual(out, "[2, 3, 4]")

    def test_funccall_in_expr(self):
        # Built-in funccall (abs) inside the comp expression.
        self.assertEqual(_run("p [abs(x) for x in [-1, 2, -3]]"),
                         "[1, 2, 3]")


class TestRegularListStillWorks(unittest.TestCase):
    """A trailing identifier `for` only triggers comp when followed by
    `var in ...` — otherwise it's a regular list element."""

    def test_simple_list_unchanged(self):
        self.assertEqual(_run("p [1, 2, 3]"), "[1, 2, 3]")

    def test_list_with_strings(self):
        self.assertEqual(_run('p ["a", "b"]'), "['a', 'b']")


class TestAstShape(unittest.TestCase):

    def test_listcomp_node_used(self):
        from flow.parser import ListComp
        ast = parse("p [x * 2 for x in xs]")
        comp = ast.body[0].args[0].value
        self.assertIsInstance(comp, ListComp)
        self.assertEqual(comp.var, "x")
        self.assertIsNone(comp.cond)

    def test_listcomp_with_cond(self):
        from flow.parser import ListComp
        ast = parse("p [x for x in xs if x > 0]")
        comp = ast.body[0].args[0].value
        self.assertIsInstance(comp, ListComp)
        self.assertIsNotNone(comp.cond)


class TestCrossTarget(unittest.TestCase):

    def test_python_native_comp(self):
        py = compile_to(parse("xs = [1, 2]\np [x * 2 for x in xs]"), "python")
        self.assertIn("for x in xs", py)

    def test_js_uses_map(self):
        js = compile_to(parse("xs = [1, 2]\np [x * 2 for x in xs]"), "js")
        self.assertIn(".map(x =>", js)

    def test_js_filter_plus_map(self):
        js = compile_to(parse("xs = [1, 2]\np [x for x in xs if x > 0]"), "js")
        self.assertIn(".filter(", js)
        self.assertIn(".map(", js)


class TestFormatterRoundTrip(unittest.TestCase):

    def test_formats_back(self):
        from flow.formatter import format_source
        out = format_source(parse("p [x * 2 for x in xs]"))
        self.assertIn("for x in", out)

    def test_with_cond_formats(self):
        from flow.formatter import format_source
        out = format_source(parse("p [x for x in xs if x > 0]"))
        self.assertIn(" if ", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
