"""Tests for `-` stdin support across CLI commands."""
import sys
import os
import subprocess
import unittest


def _flow(*args, input_text=""):
    return subprocess.run(
        [sys.executable, "-m", "flow", *args],
        input=input_text,
        capture_output=True, text=True, timeout=10,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


class TestStdinSupport(unittest.TestCase):

    def test_check_from_stdin(self):
        r = _flow("check", "-", input_text='p "hi"\n')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("[parse]   ok", r.stdout)

    def test_shrink_from_stdin(self):
        r = _flow("shrink", "-", input_text='add a=1 b=2 -> s\np s\n')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("1 + 2", r.stdout)
        self.assertNotIn("add a=", r.stdout)

    def test_fmt_from_stdin(self):
        r = _flow("fmt", "-", input_text='p   "hi"\n')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("print", r.stdout)

    def test_lint_from_stdin(self):
        r = _flow("lint", "-", input_text="add a=1 b=2 -> s\n")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("s = 1 + 2", r.stdout)

    def test_lint_fix_from_stdin(self):
        r = _flow("lint", "--fix", "-", input_text="add a=1 b=2 -> s\np s\n")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("1 + 2", r.stdout)

    def test_stats_from_stdin(self):
        r = _flow("stats", "-", input_text='p "hi"\n')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("chars:", r.stdout)

    def test_parse_from_stdin(self):
        r = _flow("parse", "-", input_text='p "hi"\n')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('"kind": "program"', r.stdout)

    def test_compile_from_stdin(self):
        r = _flow("compile", "-", input_text='p "hi"\n')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('print("hi")', r.stdout)

    def test_write_with_stdin_falls_back_to_stdout(self):
        # `fmt -w -` shouldn't try to write back to "-" — should print instead.
        r = _flow("fmt", "-w", "-", input_text='p "hi"\n')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("print", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
