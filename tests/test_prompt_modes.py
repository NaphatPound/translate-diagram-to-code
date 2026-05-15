"""Tests for the `--prompt full|minimal` mode in flow gen."""
import sys
import os
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow.gen import _build_messages, generate


class TestPromptModes(unittest.TestCase):

    def test_full_is_default(self):
        msgs = _build_messages("hi")
        full_msgs = _build_messages("hi", prompt="full")
        self.assertEqual(msgs[0]["content"], full_msgs[0]["content"])

    def test_minimal_is_shorter(self):
        full = _build_messages("hi", prompt="full")[0]["content"]
        mini = _build_messages("hi", prompt="minimal")[0]["content"]
        self.assertLess(len(mini), len(full))

    def test_minimal_excludes_few_shot(self):
        mini = _build_messages("hi", prompt="minimal")[0]["content"]
        # The full prompt contains explicit few-shot example markers.
        self.assertNotIn("# Flow — Few-Shot Examples", mini)
        # But minimal still includes the core rules.
        self.assertIn("Aliases", mini)

    @patch("flow.gen._chat")
    def test_generate_minimal_runs(self, mock_chat):
        mock_chat.return_value = 'p "hi"'
        out = generate("say hi", retries=1, polish=False, prompt="minimal")
        # The model returned the alias form; with polish=False that's kept.
        self.assertEqual(out.strip(), 'p "hi"')


if __name__ == "__main__":
    unittest.main(verbosity=2)
