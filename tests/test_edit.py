"""Tests for the /api/edit logic — exercises _apply_edits + formatter."""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, format_source
from flow.server import _apply_edits


def _round_trip(src, edits):
    ast = parse(src)
    _apply_edits(ast, edits)
    return format_source(ast)


class TestApplyEdits(unittest.TestCase):

    def test_change_string_arg(self):
        out = _round_trip(
            'read file="old.csv" -> rows',
            [{"line": 1, "args": {"file": "new.csv"}}],
        )
        self.assertIn('file="new.csv"', out)
        self.assertIn("-> rows", out)

    def test_change_out_name(self):
        out = _round_trip(
            'read file="x.csv" -> rows',
            [{"line": 1, "args": {"file": "x.csv"}, "out": "data"}],
        )
        self.assertIn("-> data", out)

    def test_remove_out(self):
        out = _round_trip(
            'read file="x.csv" -> rows',
            [{"line": 1, "args": {"file": "x.csv"}, "out": ""}],
        )
        self.assertNotIn("->", out)

    def test_variable_ref_via_dollar(self):
        out = _round_trip(
            'print value=oldvar',
            [{"line": 1, "args": {"value": "$newvar"}}],
        )
        self.assertIn("value=newvar", out)
        self.assertNotIn('"newvar"', out)  # not a string literal

    def test_number_coercion(self):
        out = _round_trip(
            'wait seconds=5',
            [{"line": 1, "args": {"seconds": 10}}],
        )
        self.assertIn("seconds=10", out)

    def test_edit_inside_control_block(self):
        src = (
            "when start\n"
            "  read file=\"a.csv\" -> rows\n"
            "  print value=rows"
        )
        out = _round_trip(src, [{"line": 2, "args": {"file": "b.csv"}, "out": "rows"}])
        self.assertIn('file="b.csv"', out)

    def test_no_op_when_no_matching_line(self):
        out = _round_trip(
            'print value=hi',
            [{"line": 99, "args": {"value": "bye"}}],
        )
        self.assertIn("value=hi", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
