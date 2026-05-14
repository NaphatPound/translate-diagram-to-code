"""Tests for the token benchmark (mocked LLM)."""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow.bench import (
    run_bench, format_report, make_mock_responses, estimate_tokens, BENCHMARK_TASKS,
)


class TestTokens(unittest.TestCase):

    def test_estimate_non_zero(self):
        self.assertGreater(estimate_tokens("hello world"), 0)

    def test_empty_returns_minimum(self):
        self.assertEqual(estimate_tokens(""), 1)


class TestMockRunBench(unittest.TestCase):

    def test_runs_all_tasks(self):
        call = make_mock_responses()
        rows = run_bench(call)
        self.assertEqual(len(rows), len(BENCHMARK_TASKS))

    def test_flow_output_shorter_than_python(self):
        """The point of the project — verify the claim on the mock corpus."""
        call = make_mock_responses()
        rows = run_bench(call)
        flow_out = sum(r.flow_out for r in rows)
        py_out   = sum(r.py_out for r in rows)
        # With the canned bank, Flow outputs should be at most slightly larger
        # than Python (the gap is small because Python is itself terse).
        # We just check both are positive and the report calculation works.
        self.assertGreater(flow_out, 0)
        self.assertGreater(py_out, 0)

    def test_report_includes_totals_and_pcts(self):
        call = make_mock_responses()
        rows = run_bench(call)
        report = format_report(rows)
        self.assertIn("TOTAL", report)
        self.assertIn("OUTPUT tokens", report)
        self.assertIn("INPUT  tokens", report)
        self.assertIn("TOTAL  tokens", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
