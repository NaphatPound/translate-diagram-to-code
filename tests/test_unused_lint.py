"""Tests for unused-variable / unused-param lint rules."""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow.lint import lint_source


def _msgs(src):
    return [w.message for w in lint_source(src)]


class TestUnusedVars(unittest.TestCase):

    def test_obvious_unused(self):
        msgs = _msgs("x = 5\np 1")
        self.assertTrue(any("'x' is assigned but never used" in m for m in msgs))

    def test_used_does_not_warn(self):
        msgs = _msgs("x = 5\np x")
        self.assertFalse(any("'x'" in m and "never used" in m for m in msgs))

    def test_underscore_prefix_silences(self):
        msgs = _msgs("_temp = 99")
        self.assertFalse(any("_temp" in m for m in msgs))

    def test_use_in_method_receiver_counts(self):
        msgs = _msgs('s = "hi"\np s.upper()')
        self.assertFalse(any("'s'" in m and "never used" in m for m in msgs))

    def test_use_in_fstring_counts(self):
        msgs = _msgs('name = "a"\np f"hi {name}"')
        self.assertFalse(any("'name'" in m and "never used" in m for m in msgs))

    def test_use_in_index_counts(self):
        msgs = _msgs("arr = [1, 2]\np arr[0]")
        self.assertFalse(any("'arr'" in m and "never used" in m for m in msgs))

    def test_use_in_dict_value_counts(self):
        msgs = _msgs('name = "a"\nu = {n: name}\np u')
        self.assertFalse(any("'name'" in m and "never used" in m for m in msgs))


class TestUnusedParams(unittest.TestCase):

    def test_unused_param_warns(self):
        msgs = _msgs("def f a b\n  a * 2\nf(1, 2)")
        self.assertTrue(any("'b'" in m and "never used" in m for m in msgs))

    def test_used_param_no_warn(self):
        msgs = _msgs("def f a b\n  a + b\nf(1, 2)")
        self.assertFalse(any("'a'" in m and "never used" in m for m in msgs))
        self.assertFalse(any("'b'" in m and "never used" in m for m in msgs))

    def test_underscore_param_silences(self):
        msgs = _msgs("def f a _ignored\n  a\nf(1, 2)")
        self.assertFalse(any("_ignored" in m for m in msgs))


class TestCuratedExamplesCleanLint(unittest.TestCase):

    def test_all_curated_examples_pass_lint(self):
        import pathlib
        examples_dir = pathlib.Path(__file__).parent.parent / "examples"
        for p in examples_dir.glob("*.flow"):
            src = p.read_text(encoding="utf-8")
            msgs = _msgs(src)
            # Some examples deliberately use verbose forms; just ensure no
            # spurious unused-var warnings.
            unused_msgs = [m for m in msgs if "never used" in m]
            self.assertEqual(unused_msgs, [],
                             f"{p.name}: unexpected unused warnings:\n  "
                             + "\n  ".join(unused_msgs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
