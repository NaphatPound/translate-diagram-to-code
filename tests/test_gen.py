"""Tests for the local-LLM loop (mocked)."""
import sys
import os
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow.gen import generate, LLMError, _extract_code, _build_messages


class TestExtract(unittest.TestCase):
    def test_extract_no_fence(self):
        self.assertEqual(_extract_code("print value=1"), "print value=1")

    def test_extract_with_flow_fence(self):
        reply = "Sure!\n```flow\nprint value=1\n```\nDone."
        self.assertEqual(_extract_code(reply), "print value=1")

    def test_extract_with_plain_fence(self):
        reply = "```\nprint value=1\n```"
        self.assertEqual(_extract_code(reply), "print value=1")


class TestLoop(unittest.TestCase):

    @patch("flow.gen._chat")
    def test_first_attempt_succeeds(self, mock_chat):
        mock_chat.return_value = 'print value="hi"'
        out = generate("say hi", retries=1, polish=False)
        self.assertIn("print", out)
        self.assertEqual(mock_chat.call_count, 1)

    @patch("flow.gen._chat")
    def test_self_correction_on_parse_error(self, mock_chat):
        # First reply is invalid; second is valid.
        mock_chat.side_effect = [
            "if x = 1\n  print value=hi",   # uses '=' instead of '=='
            'if x == 1\n  print value="hi"',
        ]
        out = generate("if x is 1 print hi", retries=2, polish=False)
        self.assertIn("print", out)
        self.assertEqual(mock_chat.call_count, 2)

    @patch("flow.gen._chat")
    def test_gives_up_after_retries(self, mock_chat):
        mock_chat.return_value = "if x = 1\n  print value=hi"  # always wrong
        with self.assertRaises(LLMError) as ctx:
            generate("bad", retries=2, polish=False)
        self.assertIn("failed after 2 retries", str(ctx.exception))
        # 1 initial attempt + 2 retries = 3 calls
        self.assertEqual(mock_chat.call_count, 3)

    @patch("flow.gen._chat")
    def test_unknown_verb_triggers_correction(self, mock_chat):
        mock_chat.side_effect = [
            "doesnotexist x=1",
            'print value="ok"',
        ]
        out = generate("print ok", retries=2, polish=False)
        self.assertEqual(out, 'print value="ok"')

    @patch("flow.gen._chat")
    def test_polish_shrinks_verbose_code(self, mock_chat):
        # Verbose first reply; deterministic shrink rewrites — no 2nd LLM call.
        # Shrink should rewrite `add` → assignment AND inline the single-use
        # result, producing `print (3 + 4)`.
        mock_chat.return_value = 'add a=3 b=4 -> s\nprint value=s'
        out = generate("3 + 4", retries=1, polish=True)
        self.assertNotIn("add a=", out)
        self.assertIn("3 + 4", out)
        self.assertEqual(mock_chat.call_count, 1)

    @patch("flow.gen._chat")
    def test_no_polish_keeps_verbose(self, mock_chat):
        mock_chat.return_value = 'add a=3 b=4 -> s\nprint value=s'
        out = generate("3 + 4", retries=1, polish=False)
        self.assertIn("add a=3", out)


class TestIncludeDocFlag(unittest.TestCase):
    """`--include-doc` (and `include_doc=True`) must inject the verb
    registry into the system prompt."""

    def test_default_prompt_no_full_verb_list(self):
        msgs = _build_messages("hi", prompt="full", include_doc=False)
        self.assertNotIn("Full verb list", msgs[0]["content"])

    def test_include_doc_adds_verb_list(self):
        msgs = _build_messages("hi", prompt="full", include_doc=True)
        sys = msgs[0]["content"]
        self.assertIn("Full verb list", sys)
        # A non-few-shot verb that the LLM might otherwise hallucinate.
        self.assertIn("http_post", sys)
        self.assertIn("classify", sys)

    def test_include_doc_works_with_minimal_prompt(self):
        msgs = _build_messages("hi", prompt="minimal", include_doc=True)
        self.assertIn("Full verb list", msgs[0]["content"])

    @patch("flow.gen._chat")
    def test_generate_passes_through_flag(self, mock_chat):
        mock_chat.return_value = 'p "x"'
        captured = {}
        def _record(messages, cfg):
            captured["sys"] = messages[0]["content"]
            return 'p "x"'
        mock_chat.side_effect = _record
        generate("x", retries=0, polish=False, include_doc=True)
        self.assertIn("Full verb list", captured["sys"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
