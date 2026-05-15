"""Tests for generator expressions inside funccall args + enumerate fix.

`sum(x for x in xs)`, `any(... for ... if ...)` etc. Now lower to a
ListComp argument, which builtins like sum/max/any accept as well as
they would a generator.

Also covers the fix for `each i, x in enumerate(xs)` — previously
compiled to `for i, x in enumerate(xs).items()` (runtime error).
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


class TestGeneratorExpression(unittest.TestCase):

    def test_sum_doubled(self):
        self.assertEqual(_run("p sum(x*2 for x in [1, 2, 3])"), "12")

    def test_max_filtered(self):
        self.assertEqual(_run("p max(x for x in [3, 7, 1, 5])"), "7")

    def test_any_true(self):
        self.assertEqual(_run("p any(x > 5 for x in [1, 6, 3])"), "True")

    def test_any_false(self):
        self.assertEqual(_run("p any(x > 5 for x in [1, 2, 3])"), "False")

    def test_all_true(self):
        self.assertEqual(_run("p all(x > 0 for x in [1, 2, 3])"), "True")

    def test_count_with_filter(self):
        self.assertEqual(_run("p count(x for x in [1, 2, 3, 4] if x > 2)"), "2")


class TestEnumerateIteration(unittest.TestCase):
    """`each i, x in enumerate(xs)` should iterate pairs, not call .items()."""

    def test_enumerate(self):
        out = _run(
            'xs = ["a", "b", "c"]\n'
            'each i, x in enumerate(xs)\n'
            '  p f"{i}:{x}"'
        )
        self.assertEqual(out.split(), ["0:a", "1:b", "2:c"])

    def test_zip(self):
        # zip(...) is the other pair-yielding iter we now recognize.
        out = _run(
            'each a, b in zip([1, 2], ["x", "y"])\n'
            '  p f"{a}-{b}"'
        )
        self.assertEqual(out.split(), ["1-x", "2-y"])

    def test_items_call(self):
        # `items(d)` (Flow builtin) should also yield pairs cleanly.
        out = _run(
            'd = {"a": 1, "b": 2}\n'
            'each k, v in items(d)\n'
            '  p f"{k}={v}"'
        )
        # Order is dict insertion order in Python 3.7+.
        self.assertEqual(out.split(), ["a=1", "b=2"])

    def test_dict_iteration_still_works(self):
        # Plain dict iter still uses .items() — the regression check.
        out = _run(
            'd = {"a": 1, "b": 2}\n'
            'each k, v in d\n'
            '  p f"{k}={v}"'
        )
        self.assertEqual(out.split(), ["a=1", "b=2"])


class TestEnumerateFunccall(unittest.TestCase):
    """`enumerate(xs)` as a funccall produces a list of pairs."""

    def test_enumerate_funccall(self):
        out = _run('p enumerate(["a", "b"])')
        # Python prints tuples; we just check the shape and contents.
        self.assertIn("(0, 'a')", out)
        self.assertIn("(1, 'b')", out)

    def test_items_funccall(self):
        out = _run('p items({"a": 1})')
        self.assertIn("'a'", out)
        self.assertIn("1", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
