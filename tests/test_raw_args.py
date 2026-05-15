"""Tests for the raw_args check — filter/map/sort predicates must be quoted."""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, compile_to, CompileError


class TestRawArgsMustBeQuoted(unittest.TestCase):

    def test_filter_unquoted_where_errors_with_hint(self):
        with self.assertRaises(CompileError) as ctx:
            compile_to(parse("filter from=xs where=x > 0"), "python")
        msg = str(ctx.exception).lower()
        self.assertIn("where", msg)
        self.assertIn("quoted", msg)
        # Sample fix shown.
        self.assertIn('"x > 0"', str(ctx.exception))

    def test_map_unquoted_to_errors(self):
        with self.assertRaises(CompileError) as ctx:
            compile_to(parse("map from=xs to=x.upper()"), "python")
        self.assertIn("'to'", str(ctx.exception))
        self.assertIn("quoted", str(ctx.exception).lower())

    def test_sort_unquoted_by_errors(self):
        with self.assertRaises(CompileError) as ctx:
            compile_to(parse("sort from=xs by=-x"), "python")
        self.assertIn("'by'", str(ctx.exception))

    def test_quoted_forms_still_compile(self):
        # Sanity: the canonical compiled form still works.
        for src in [
            'filter from=xs where="x > 0"',
            'map from=xs to="x * 2"',
            'sort from=xs by="-x"',
        ]:
            try:
                compile_to(parse(src), "python")
            except CompileError as e:
                self.fail(f"{src!r} should compile but got: {e}")

    def test_hint_is_arg_specific(self):
        # The example fix should match the arg being used.
        with self.assertRaises(CompileError) as ctx:
            compile_to(parse("filter from=xs where=x"), "python")
        self.assertIn('where="x > 0"', str(ctx.exception))

        with self.assertRaises(CompileError) as ctx:
            compile_to(parse("map from=xs to=x"), "python")
        self.assertIn('to="x * 2"', str(ctx.exception))

        with self.assertRaises(CompileError) as ctx:
            compile_to(parse("sort from=xs by=x"), "python")
        self.assertIn('by="-x"', str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
