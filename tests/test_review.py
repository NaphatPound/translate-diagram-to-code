"""Tests for the big-LLM review pipeline (mocked LLM)."""
import sys
import os
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow.review import (
    review, run_compiled, build_prompt, _parse_review,
    RunResult,
)


def _fake_call(reply: str):
    return lambda prompt: reply


class TestRunCompiled(unittest.TestCase):

    def test_runs_python(self):
        r = run_compiled("print('hi')", "python")
        self.assertEqual(r.stdout.strip(), "hi")
        self.assertEqual(r.exit_code, 0)

    def test_failure_captured(self):
        r = run_compiled("raise SystemExit(2)", "python")
        self.assertEqual(r.exit_code, 2)

    def test_timeout(self):
        # 1s timeout, sleep 5s — should time out
        r = run_compiled("import time; time.sleep(5)", "python", timeout=0.5)
        self.assertTrue(r.timed_out)


class TestParseReview(unittest.TestCase):

    def test_parse_ok(self):
        text = (
            "VERDICT: ok\n"
            "SUMMARY: prints hello.\n"
            "ISSUES:\n  - none\n"
            "SUGGEST:\n  - none\n"
        )
        r = _parse_review(text)
        self.assertEqual(r["verdict"], "ok")
        self.assertEqual(r["summary"], "prints hello.")
        self.assertEqual(r["issues"], [])
        self.assertEqual(r["suggest"], [])

    def test_parse_with_issues(self):
        text = (
            "VERDICT: issues\n"
            "SUMMARY: counts wrong.\n"
            "ISSUES:\n  - off-by-one\n  - misses empties\n"
            "SUGGEST:\n  - use count properly\n"
        )
        r = _parse_review(text)
        self.assertEqual(r["verdict"], "issues")
        self.assertEqual(r["issues"], ["off-by-one", "misses empties"])
        self.assertEqual(r["suggest"], ["use count properly"])

    def test_unknown_verdict_falls_back(self):
        r = _parse_review("VERDICT: weird\nSUMMARY: hi")
        self.assertEqual(r["verdict"], "unknown")


class TestBuildPrompt(unittest.TestCase):

    def test_includes_intent_and_run_output(self):
        rr = RunResult(stdout="hi\n", stderr="", exit_code=0, timed_out=False)
        p = build_prompt(
            flow_src='print value="hi"',
            target_src='print("hi")',
            lang="python",
            run=rr,
            intent="say hello",
        )
        self.assertIn("say hello", p)
        self.assertIn("hi\\n", p)  # repr of stdout
        self.assertIn("Flow DSL program", p)


class TestEndToEnd(unittest.TestCase):

    def test_review_simple_program(self):
        flow_src = 'print value="hello"'
        rev = review(flow_src, lang="python",
                     call=_fake_call(
                         "VERDICT: ok\n"
                         "SUMMARY: prints 'hello' to stdout.\n"
                         "ISSUES:\n  - none\n"
                         "SUGGEST:\n  - none\n"
                     ))
        self.assertEqual(rev.verdict, "ok")
        self.assertEqual(rev.run.stdout.strip(), "hello")
        self.assertEqual(rev.lang, "python")

    def test_review_no_run(self):
        rev = review('print value="hello"', lang="python", run=False,
                     call=_fake_call("VERDICT: ok\nSUMMARY: looks fine."))
        self.assertIsNone(rev.run)


if __name__ == "__main__":
    unittest.main(verbosity=2)
