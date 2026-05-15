"""Tests for `flow doc` — auto-generated verb reference."""
import sys
import os
import subprocess
import unittest


def _flow(*args):
    return subprocess.run(
        [sys.executable, "-m", "flow", *args],
        capture_output=True, text=True, timeout=10,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


class TestFlowDoc(unittest.TestCase):

    def test_default_output_contains_sections(self):
        r = _flow("doc")
        self.assertEqual(r.returncode, 0, r.stderr)
        out = r.stdout
        # Top-level structure.
        self.assertIn("Syntax", out)
        self.assertIn("Aliases", out)
        self.assertIn("p=print", out)
        # Categories.
        for cat in ("io", "data", "math", "text"):
            self.assertIn(f"## {cat}", out)
        # Some specific verbs.
        self.assertIn("filter", out)
        self.assertIn("count", out)
        self.assertIn("http_get", out)

    def test_each_verb_has_summary(self):
        r = _flow("doc")
        # A representative summary string.
        self.assertIn("Print a value to stdout", r.stdout)
        self.assertIn("Keep items where condition is true", r.stdout)

    def test_returns_arrow_shown_for_returning_verbs(self):
        r = _flow("doc")
        # `read` returns, so its signature should include `-> name`.
        # Pull the read line.
        line = next(l for l in r.stdout.splitlines() if l.strip().startswith("read "))
        self.assertIn("-> name", line)

    def test_compact_mode_is_terse(self):
        r = _flow("doc", "--compact")
        self.assertEqual(r.returncode, 0, r.stderr)
        out = r.stdout
        # No section headers in compact mode.
        self.assertNotIn("## io", out)
        self.assertNotIn("Syntax", out)
        # Still has the verb lines.
        self.assertIn("filter", out)
        # Compact should be < default.
        default_len = len(_flow("doc").stdout)
        self.assertLess(len(out), default_len)


if __name__ == "__main__":
    unittest.main(verbosity=2)
