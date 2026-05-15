"""Tests for identity defaults of missing raw_args.

`filter`/`map`/`sort` without their predicate args (where/to/by) should
behave as identity, not produce runtime errors. The previous None-filled
template silently emitted bogus code like `sorted(xs, key=lambda x: (None))`.
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
                       capture_output=True, text=True, timeout=5)
    if r.returncode != 0:
        return f"ERR: {r.stderr.strip()}"
    return r.stdout.strip()


class TestRawArgDefaults(unittest.TestCase):

    def test_sort_without_by_natural_order(self):
        self.assertEqual(_run("xs = [3, 1, 2]\nsort from=xs -> ys\np ys"),
                         "[1, 2, 3]")

    def test_sort_pipe_no_by(self):
        self.assertEqual(_run("[3, 1, 2] | sort | p"), "[1, 2, 3]")

    def test_filter_without_where_is_identity(self):
        self.assertEqual(_run("xs = [1, 2]\nfilter from=xs -> ys\np ys"),
                         "[1, 2]")

    def test_map_without_to_is_identity(self):
        self.assertEqual(_run("xs = [1, 2, 3]\nmap from=xs -> ys\np ys"),
                         "[1, 2, 3]")

    def test_sort_with_explicit_by_still_works(self):
        self.assertEqual(_run('xs = [3, 1, 2]\nsort from=xs by="-x" -> ys\np ys'),
                         "[3, 2, 1]")


if __name__ == "__main__":
    unittest.main(verbosity=2)
