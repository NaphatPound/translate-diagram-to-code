"""Tests for break/continue/unless control flow."""
import sys
import os
import subprocess
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, compile_to
from flow.parser import BreakStmt, ContinueStmt, IfStmt


def _run_py(src):
    r = subprocess.run([sys.executable, "-c", src], capture_output=True,
                       text=True, timeout=4)
    if r.returncode != 0:
        raise AssertionError(f"py failed:\n{src}\n---\n{r.stderr}")
    return r.stdout


class TestBreakContinue(unittest.TestCase):

    def test_break_stops_loop(self):
        py = compile_to(parse(
            "each i in 1..5\n  break if i > 2\n  p i"
        ), "python")
        self.assertEqual(_run_py(py).strip().split("\n"), ["1", "2"])

    def test_continue_skips_iteration(self):
        py = compile_to(parse(
            "each i in 1..5\n  continue if i == 3\n  p i"
        ), "python")
        self.assertEqual(_run_py(py).strip().split("\n"), ["1", "2", "4", "5"])

    def test_bare_break_alone_parses(self):
        ast = parse("each i in 1..3\n  break")
        body = ast.body[0].body
        self.assertIsInstance(body[0], BreakStmt)

    def test_continue_with_unless(self):
        # `continue unless cond` ≡ `continue if !cond`
        py = compile_to(parse(
            "each i in 1..5\n  continue unless i > 2\n  p i"
        ), "python")
        self.assertEqual(_run_py(py).strip().split("\n"), ["3", "4", "5"])


class TestUnless(unittest.TestCase):

    def test_block_unless(self):
        py = compile_to(parse(
            "x = 0\nunless x > 0\n  p \"not pos\""
        ), "python")
        self.assertEqual(_run_py(py).strip(), "not pos")

    def test_block_unless_desugars_to_if(self):
        ast = parse("unless x > 0\n  p \"a\"")
        self.assertIsInstance(ast.body[0], IfStmt)

    def test_postfix_unless(self):
        py = compile_to(parse(
            "b = false\np \"empty\" unless b"
        ), "python")
        self.assertEqual(_run_py(py).strip(), "empty")

    def test_postfix_if_still_works_alongside_unless(self):
        py = compile_to(parse(
            "x = 5\np \"big\" if x > 0\np \"none\" unless x > 0"
        ), "python")
        self.assertEqual(_run_py(py).strip(), "big")


if __name__ == "__main__":
    unittest.main(verbosity=2)
