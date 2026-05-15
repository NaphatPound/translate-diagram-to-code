"""Tests for `flow gen --cache` (disk-backed cache)."""
import sys
import os
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow.gen import generate, _cache_path, _cache_key, cache_clear, cache_show


class TestCache(unittest.TestCase):

    def setUp(self):
        # Redirect HOME so the cache writes to a tmpdir, not the user's actual
        # ~/.flow/. Restore in tearDown.
        self._tmp = tempfile.TemporaryDirectory()
        self._old_home = os.environ.get("HOME")
        os.environ["HOME"] = self._tmp.name

    def tearDown(self):
        if self._old_home is not None:
            os.environ["HOME"] = self._old_home
        else:
            del os.environ["HOME"]
        self._tmp.cleanup()

    @patch("flow.gen._chat")
    def test_first_call_writes_cache(self, mock_chat):
        mock_chat.return_value = 'p "hi"'
        out = generate("say hi", retries=0, polish=False, cache=True)
        self.assertIn("hi", out)
        # Cache file should now exist.
        self.assertTrue(_cache_path().exists())
        data = json.loads(_cache_path().read_text())
        self.assertEqual(len(data), 1)

    @patch("flow.gen._chat")
    def test_second_call_uses_cache(self, mock_chat):
        mock_chat.return_value = 'p "hi"'
        # First call hits the LLM.
        generate("greet", retries=0, polish=False, cache=True)
        self.assertEqual(mock_chat.call_count, 1)
        # Second call same args → cache hit, no second LLM call.
        out2 = generate("greet", retries=0, polish=False, cache=True)
        self.assertEqual(mock_chat.call_count, 1)
        self.assertIn("hi", out2)

    @patch("flow.gen._chat")
    def test_different_request_skips_cache(self, mock_chat):
        mock_chat.side_effect = ['p "a"', 'p "b"']
        a = generate("first", retries=0, polish=False, cache=True)
        b = generate("second", retries=0, polish=False, cache=True)
        self.assertEqual(mock_chat.call_count, 2)
        self.assertNotEqual(a, b)

    @patch("flow.gen._chat")
    def test_different_prompt_mode_separate_cache_entry(self, mock_chat):
        mock_chat.side_effect = ['p "full"', 'p "mini"']
        generate("same", retries=0, polish=False, cache=True, prompt="full")
        generate("same", retries=0, polish=False, cache=True, prompt="minimal")
        # Two distinct cache entries.
        data = json.loads(_cache_path().read_text())
        self.assertEqual(len(data), 2)

    def test_cache_key_stable(self):
        k1 = _cache_key("hi", "full", 3, True, 1)
        k2 = _cache_key("hi", "full", 3, True, 1)
        self.assertEqual(k1, k2)
        k3 = _cache_key("hi", "minimal", 3, True, 1)
        self.assertNotEqual(k1, k3)

    @patch("flow.gen._chat")
    def test_cache_clear_removes_file(self, mock_chat):
        mock_chat.return_value = 'p "x"'
        generate("foo", retries=0, polish=False, cache=True)
        self.assertTrue(_cache_path().exists())
        n = cache_clear()
        self.assertEqual(n, 1)
        self.assertFalse(_cache_path().exists())

    def test_cache_clear_on_empty(self):
        # No file → returns 0, no error.
        self.assertEqual(cache_clear(), 0)

    @patch("flow.gen._chat")
    def test_cache_show_lists_entries(self, mock_chat):
        import io
        mock_chat.side_effect = ['p "one"', 'p "two"']
        generate("first", retries=0, polish=False, cache=True)
        generate("second", retries=0, polish=False, cache=True)
        buf = io.StringIO()
        n = cache_show(stream=buf)
        self.assertEqual(n, 2)
        out = buf.getvalue()
        self.assertIn("2 entries", out)
        self.assertIn("one", out)
        self.assertIn("two", out)

    def test_cache_show_on_empty(self):
        import io
        buf = io.StringIO()
        cache_show(stream=buf)
        self.assertIn("empty", buf.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
