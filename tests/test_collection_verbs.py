"""Tests for the reverse / unique / keys / values verbs."""
import sys
import os
import subprocess
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, compile_to


def _run_python(src: str) -> str:
    py = compile_to(parse(src), "python")
    r = subprocess.run([sys.executable, "-c", py],
                       capture_output=True, text=True, timeout=5)
    return r.stdout.strip()


class TestCollectionVerbs(unittest.TestCase):

    def test_reverse(self):
        out = _run_python("xs = [1, 2, 3]\nreverse xs -> r\np r")
        self.assertEqual(out, "[3, 2, 1]")

    def test_unique_preserves_first_seen_order(self):
        out = _run_python("xs = [3, 1, 2, 1, 3, 2]\nunique xs -> u\np u")
        self.assertEqual(out, "[3, 1, 2]")

    def test_keys(self):
        out = _run_python('d = {"a": 1, "b": 2}\nkeys d -> ks\np ks')
        self.assertEqual(out, "['a', 'b']")

    def test_values(self):
        out = _run_python('d = {"a": 1, "b": 2}\nvalues d -> vs\np vs')
        self.assertEqual(out, "[1, 2]")

    def test_reverse_works_in_pipe(self):
        # Primary arg `from` makes `xs | reverse` work.
        out = _run_python("xs = [1, 2, 3]\nxs | reverse | p")
        self.assertEqual(out, "[3, 2, 1]")

    def test_keys_works_in_pipe(self):
        out = _run_python('d = {"x": 1, "y": 2}\nd | keys | p')
        self.assertEqual(out, "['x', 'y']")

    def test_unique_handles_strings(self):
        out = _run_python('xs = ["a", "b", "a", "c"]\nunique xs -> u\np u')
        self.assertEqual(out, "['a', 'b', 'c']")


class TestCollectionVerbsJS(unittest.TestCase):
    """Smoke-test that JS templates are present (don't actually run node)."""

    def test_reverse_js_template_exists(self):
        from flow.verbs import VERBS
        self.assertIn("js", VERBS["reverse"].templates)
        self.assertIn("reverse", VERBS["reverse"].templates["js"])

    def test_unique_js_uses_set(self):
        from flow.verbs import VERBS
        self.assertIn("Set", VERBS["unique"].templates["js"])

    def test_keys_values_use_object_methods(self):
        from flow.verbs import VERBS
        self.assertIn("Object.keys", VERBS["keys"].templates["js"])
        self.assertIn("Object.values", VERBS["values"].templates["js"])


class TestPipeFromLiteral(unittest.TestCase):
    """Any value expression can start a pipe, not just identifiers."""

    def test_list_literal_pipe(self):
        self.assertEqual(_run_python("[1, 2, 3] | reverse | p"), "[3, 2, 1]")

    def test_string_literal_pipe(self):
        self.assertEqual(_run_python('"hello" | upper | p'), "HELLO")

    def test_dict_literal_pipe(self):
        out = _run_python('{"a": 1, "b": 2} | keys | p')
        self.assertEqual(out, "['a', 'b']")

    def test_parenthesized_expr_pipe(self):
        # Pipe the result of an expression directly into `print`.
        out = _run_python('(1 + 2) | p')
        self.assertEqual(out, "3")

    def test_pipe_with_arrow_capture(self):
        out = _run_python(
            "[3, 1, 2] | unique -> ys\n"
            "p ys"
        )
        self.assertEqual(out, "[3, 1, 2]")

    def test_pipe_with_postfix_if(self):
        out = _run_python('[1, 2, 3] | p if true')
        self.assertEqual(out, "[1, 2, 3]")


class TestPipeFromName(unittest.TestCase):
    """`<ident> | verb` should pipe the variable into `verb`'s primary arg,
    not be parsed as a zero-arg verb call."""

    def test_name_pipe_into_verb(self):
        out = _run_python("xs = [1, 2, 3]\nxs | reverse | p")
        self.assertEqual(out, "[3, 2, 1]")

    def test_name_pipe_into_filter(self):
        out = _run_python(
            'xs = [1, 2, 3, 4]\n'
            'xs | filter where="x > 2" | p'
        )
        self.assertEqual(out, "[3, 4]")

    def test_name_pipe_with_arrow(self):
        # The pipeline can capture a final `-> name` for downstream use.
        out = _run_python(
            "xs = [1, 1, 2]\n"
            "xs | unique -> ys\n"
            "p ys"
        )
        self.assertEqual(out, "[1, 2]")

    def test_zero_arg_verb_still_works(self):
        # `now` is a verb with no args — should still parse as verb call, not
        # as pipe source. The compiled output must reference _dt.datetime.now().
        from flow import parse, compile_to
        py = compile_to(parse("now | p"), "python")
        self.assertIn("datetime.now()", py)


class TestFuncCallForms(unittest.TestCase):
    """`p reverse(xs)` / `p unique(xs)` / `p keys(d)` etc. must work directly,
    not require going through a verb-form assignment first."""

    def test_reverse_funccall(self):
        self.assertEqual(_run_python("xs = [1, 2, 3]\np reverse(xs)"), "[3, 2, 1]")

    def test_unique_funccall(self):
        self.assertEqual(_run_python("xs = [1, 1, 2]\np unique(xs)"), "[1, 2]")

    def test_keys_funccall(self):
        self.assertEqual(_run_python('d = {"a": 1, "b": 2}\np keys(d)'), "['a', 'b']")

    def test_values_funccall(self):
        self.assertEqual(_run_python('d = {"a": 1, "b": 2}\np values(d)'), "[1, 2]")

    def test_avg_funccall(self):
        self.assertEqual(_run_python("xs = [1, 2, 3, 4]\np avg(xs)"), "2.5")

    def test_sum_funccall(self):
        self.assertEqual(_run_python("xs = [1, 2, 3]\np sum(xs)"), "6")

    def test_sorted_funccall(self):
        self.assertEqual(_run_python("xs = [3, 1, 2]\np sorted(xs)"), "[1, 2, 3]")


class TestShrinkToFunccall(unittest.TestCase):
    """`reverse from=xs -> ys / p ys` should shrink + inline to `p reverse(xs)`."""

    def test_reverse_shrinks_to_funccall(self):
        from flow.shrink import shrink_source
        out = shrink_source("xs = [1, 2, 3]\nreverse from=xs -> ys\np ys")
        self.assertIn("reverse(", out)
        self.assertNotIn("reverse from=", out)

    def test_unique_shrinks_to_funccall(self):
        from flow.shrink import shrink_source
        out = shrink_source("xs = [1, 1, 2]\nunique from=xs -> ys\np ys")
        self.assertIn("unique(", out)

    def test_keys_shrinks_to_funccall(self):
        from flow.shrink import shrink_source
        out = shrink_source('d = {"a": 1}\nkeys of=d -> ks\np ks')
        self.assertIn("keys(", out)


class TestDocLists(unittest.TestCase):
    """New verbs should appear in `flow doc`."""

    def test_appears_in_compact_doc(self):
        from flow.verbs import verb_reference
        doc = verb_reference(compact=True)
        for v in ("reverse", "unique", "keys", "values"):
            self.assertIn(v, doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
