"""Tests for /api/shrink and /api/lint server endpoints (via direct handler calls)."""
import sys
import os
import json
import unittest
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Smoke-test the server logic by importing the route helpers directly.
from flow.shrink import shrink_source
from flow.lint import lint_source


class TestShrinkEndpoint(unittest.TestCase):

    def test_shrink_known_pattern(self):
        out = shrink_source("add a=3 b=4 -> s\nprint value=s")
        # 7 chars or fewer expected after shrink (`print (3 + 4)`).
        self.assertIn("3 + 4", out)
        self.assertNotIn("add a=", out)


class TestLintEndpoint(unittest.TestCase):

    def test_lint_reports_warnings(self):
        ws = lint_source("add a=3 b=4 -> s\nprint value=s")
        self.assertTrue(len(ws) >= 1)

    def test_lint_quiet_on_compact(self):
        ws = lint_source('p "hi"')
        self.assertEqual(ws, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
