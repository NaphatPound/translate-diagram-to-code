"""Tests for `flow run`."""
import sys
import os
import tempfile
import subprocess
import unittest


def _flow(*args, input_text=None):
    return subprocess.run(
        [sys.executable, "-m", "flow", *args],
        capture_output=True, text=True, timeout=10,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


class TestFlowRun(unittest.TestCase):

    def test_hello_runs(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".flow", delete=False) as f:
            f.write('p "hello"')
            path = f.name
        try:
            r = _flow("run", path)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.strip(), "hello")
        finally:
            os.unlink(path)

    def test_math_runs(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".flow", delete=False) as f:
            f.write("def double x\n  x * 2\np double(7)")
            path = f.name
        try:
            r = _flow("run", path)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.strip(), "14")
        finally:
            os.unlink(path)

    def test_invalid_target_errors(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".flow", delete=False) as f:
            f.write('p "x"')
            path = f.name
        try:
            r = _flow("run", path, "--to", "rust")
            # argparse rejects rust because choices excludes it.
            self.assertNotEqual(r.returncode, 0)
        finally:
            os.unlink(path)

    def test_parse_error_returns_nonzero(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".flow", delete=False) as f:
            f.write('if x = 1')  # `=` instead of `==` in cond
            path = f.name
        try:
            r = _flow("run", path)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("ERROR", r.stderr)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
