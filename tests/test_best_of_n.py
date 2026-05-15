"""Tests for `flow gen --rounds N` best-of-N sampling."""
import sys
import os
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow.gen import generate, LLMError


class TestBestOfN(unittest.TestCase):

    @patch("flow.gen._chat")
    def test_picks_shortest_valid(self, mock_chat):
        # Three rounds, the second is the shortest.
        mock_chat.side_effect = [
            'p "hello there, friends"',  # round 1
            'p "hi"',                    # round 2: shortest
            'p "medium length"',         # round 3
        ]
        out = generate("greet", retries=0, polish=False, rounds=3)
        self.assertEqual(out.strip(), 'p "hi"')

    @patch("flow.gen._chat")
    def test_single_round_unchanged(self, mock_chat):
        # rounds=1 should behave identically to no flag.
        mock_chat.return_value = 'p "hi"'
        out = generate("greet", retries=0, polish=False, rounds=1)
        self.assertEqual(out.strip(), 'p "hi"')
        self.assertEqual(mock_chat.call_count, 1)

    @patch("flow.gen._chat")
    def test_invalid_rounds_excluded(self, mock_chat):
        # Round 1 fails validation; rounds 2-3 succeed. Should pick shortest
        # of the valid ones.
        mock_chat.side_effect = [
            'invalid !!! flow',          # round 1: parse error
            'p "longer message"',        # round 2: valid
            'p "x"',                     # round 3: valid + shortest
        ]
        out = generate("test", retries=0, polish=False, rounds=3)
        self.assertEqual(out.strip(), 'p "x"')

    @patch("flow.gen._chat")
    def test_all_rounds_fail(self, mock_chat):
        # Every round invalid → raise LLMError.
        mock_chat.return_value = "totally not valid flow!!!"
        with self.assertRaises(LLMError):
            generate("test", retries=0, polish=False, rounds=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
