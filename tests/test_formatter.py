"""Formatter tests — round-trip property is the main thing."""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, ast_to_dict
from flow.formatter import format_source


def _structural(src):
    """Parse → AST dict (strip line numbers) for structural comparison."""
    ast = parse(src)
    d = ast_to_dict(ast)
    _strip_lines(d)
    return d


def _strip_lines(o):
    if isinstance(o, dict):
        o.pop("line", None)
        for v in o.values():
            _strip_lines(v)
    elif isinstance(o, list):
        for v in o:
            _strip_lines(v)


class TestRoundTrip(unittest.TestCase):

    def _check(self, src):
        first = _structural(src)
        formatted = format_source(parse(src))
        # The formatted source must parse and produce the same AST.
        second = _structural(formatted)
        self.assertEqual(first, second,
                         f"\noriginal:\n{src}\nformatted:\n{formatted}")

    def test_hello(self):
        self._check('print value="hi"')

    def test_with_out(self):
        self._check("today -> t\nprint value=t")

    def test_if_else(self):
        self._check("if x > 0\n  print value=a\nelse\n  print value=b")

    def test_each(self):
        self._check("each row in items\n  print value=row.name")

    def test_nested(self):
        self._check(
            "when start\n"
            "  if x > 0\n"
            "    each y in ys\n"
            "      print value=y\n"
            "  else\n"
            "    print value=\"nope\"\n"
        )

    def test_literals(self):
        self._check('foo items=[1, 2, 3] data={a: 1, b: "x"}')

    def test_funccall(self):
        self._check("if count(items) > 0\n  print value=ok")

    def test_filter_with_raw(self):
        # `where` is raw but the formatter just preserves the string literal.
        self._check('filter from=items where="x > 0" -> ys\nprint value=ys')


class TestFormat(unittest.TestCase):

    def test_uses_two_space_indent(self):
        src = "if x > 0\n  print value=a"
        out = format_source(parse(src))
        self.assertIn("  print", out)

    def test_strings_quoted(self):
        out = format_source(parse('print value="hello"'))
        self.assertIn('"hello"', out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
