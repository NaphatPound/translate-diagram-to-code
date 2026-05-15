"""Tests for the iteration-30 changes: enhanced `flow check` (covered by
unit-level wrappers) + shadow-aware shrink + lint --fix delegation."""
import sys
import os
import subprocess
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow.shrink import shrink_source


class TestShadowAwareInline(unittest.TestCase):
    """`def f x` shadows an outer `x` — shrink must NOT inline outer x
    into the function body."""

    def test_def_param_not_inlined(self):
        src = (
            "def double x\n"
            "  return x * 2\n"
            "\n"
            "x = 5\n"
            "p x\n"
        )
        out = shrink_source(src)
        # The def's body should still reference param `x`, NOT the literal 5.
        self.assertIn("x * 2", out)
        self.assertNotIn("5 * 2", out)

    def test_each_loop_var_not_inlined(self):
        # `each i in items` binds i; an outer `i = 99` should be safe.
        src = (
            "i = 99\n"
            "each i in [1, 2, 3]\n"
            "  p i\n"
            "p i\n"
        )
        out = shrink_source(src)
        self.assertIn("99", out)

    def test_unshadowed_name_still_inlines(self):
        # Sanity: a name with NO shadow should still be inlined.
        src = "t = 5\np t"
        out = shrink_source(src)
        self.assertIn("print 5", out)
        self.assertNotIn("t = ", out)


class TestFlowCheckCLI(unittest.TestCase):
    """Smoke-test the enhanced `flow check` output."""

    def _flow(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "flow", *args],
            capture_output=True, text=True, timeout=10,
        )

    def test_check_outputs_status_blocks(self):
        # Use a file from the curated examples.
        import pathlib
        ex = pathlib.Path(__file__).parent.parent / "examples" / "hello.flow"
        r = self._flow("check", str(ex))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("[parse]", r.stdout)
        self.assertIn("[compile]", r.stdout)
        self.assertIn("[lint]", r.stdout)
        self.assertIn("[shrink]", r.stdout)

    def test_lint_fix_applies_shrink(self):
        import pathlib, tempfile
        with tempfile.NamedTemporaryFile("w+", suffix=".flow", delete=False) as f:
            f.write("def double x\n  return x * 2\np double(5)\n")
            path = f.name
        try:
            r = self._flow("lint", "--fix", path)
            self.assertEqual(r.returncode, 0, r.stderr)
            # Shrink should drop the `return` keyword (implicit return).
            self.assertNotIn("return", r.stdout)
            self.assertIn("x * 2", r.stdout)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
