"""Tests for semantic-synonym hints on unknown verb args.

LLMs frequently reach for natural English names (cond=, key=, msg=,
path=) instead of Flow's canonical names (where=, by=, value=, file=).
The compiler should suggest the canonical name in its error, so the
retry loop can correct on the next round.
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, compile_to, CompileError


def _err(src: str) -> str:
    try:
        compile_to(parse(src), "python")
    except CompileError as e:
        return str(e)
    return ""


class TestArgSynonyms(unittest.TestCase):

    def test_filter_cond_suggests_where(self):
        self.assertIn("'where'", _err('filter from=xs cond="x > 0"'))

    def test_filter_pred_suggests_where(self):
        self.assertIn("'where'", _err('filter from=xs pred="x"'))

    def test_sort_key_suggests_by(self):
        self.assertIn("'by'", _err('sort from=xs key="x.age"'))

    def test_map_transform_suggests_to(self):
        # `as` would be ideal but it's a parser keyword (used in
        # `repeat N as i`). The non-keyword synonyms still work.
        self.assertIn("'to'", _err('map from=xs transform="x * 2"'))

    def test_print_msg_suggests_value(self):
        self.assertIn("'value'", _err('print msg="hi"'))

    def test_count_list_suggests_of(self):
        self.assertIn("'of'", _err("count list=xs"))

    def test_sum_input_suggests_of(self):
        # `input` could be from-style or of-style; for sum (aggregator)
        # the priority list should pick `of`.
        self.assertIn("'of'", _err("sum input=xs"))

    def test_filter_input_suggests_from(self):
        # Same word, different verb — should pick `from` for filter.
        self.assertIn("'from'", _err('filter input=xs where="x"'))

    def test_read_path_suggests_file(self):
        self.assertIn("'file'", _err('read path="a.csv"'))

    def test_split_delim_suggests_sep(self):
        self.assertIn("'sep'", _err('split text="a,b" delim=","'))

    def test_replace_pattern_suggests_find(self):
        self.assertIn("'find'",
                      _err('replace text="a" pattern="x" to="y"'))

    def test_unknown_synonym_falls_back_to_fuzzy(self):
        # `katze` isn't in the synonym table — should fall back to
        # edit-distance fuzzy match. For filter it should not suggest
        # anything close.
        e = _err('filter from=xs katze="x"')
        # No false-positive suggestion.
        self.assertIn("doesn't accept arg 'katze'", e)


if __name__ == "__main__":
    unittest.main(verbosity=2)
