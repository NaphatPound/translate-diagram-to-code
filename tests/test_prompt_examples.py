"""Sanity-check every Flow code block in prompts/few_shot.md.

The prompts ship as the LLM's primary learning material — if an example
doesn't parse or compile cleanly, the model learns broken syntax. This
test extracts each ``` block and runs it through parse + compile.

We skip blocks that are clearly error-correction examples (USER messages
containing 'previously rejected'), since those contain intentionally
bad code shown as input.
"""
import sys
import os
import re
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import parse, compile_to, ParseError, CompileError


_FEW_SHOT = Path(__file__).resolve().parent.parent / "prompts" / "few_shot.md"


def _extract_a_blocks(md: str) -> list:
    """Return the (line_no, src) of every fenced code block that follows
    an `A:` marker — the assistant's canonical answer."""
    blocks = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "A:":
            # Find the next ``` line.
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            if j >= len(lines):
                break
            # Now collect lines until the closing ```.
            start = j + 1
            k = start
            while k < len(lines) and not lines[k].startswith("```"):
                k += 1
            src = "\n".join(lines[start:k])
            blocks.append((start + 1, src))  # 1-indexed line of block start
            i = k + 1
        else:
            i += 1
    return blocks


class TestPromptExamples(unittest.TestCase):

    def test_few_shot_has_examples(self):
        md = _FEW_SHOT.read_text(encoding="utf-8")
        blocks = _extract_a_blocks(md)
        # Sanity: there should be a healthy number of examples.
        self.assertGreater(len(blocks), 10)

    def test_every_a_block_parses_and_compiles(self):
        md = _FEW_SHOT.read_text(encoding="utf-8")
        blocks = _extract_a_blocks(md)
        failures = []
        for line_no, src in blocks:
            try:
                ast = parse(src)
                compile_to(ast, "python")
            except (ParseError, CompileError) as e:
                failures.append(f"line {line_no}: {e}\n--- source ---\n{src}\n---")
        if failures:
            self.fail(
                f"{len(failures)} of {len(blocks)} examples failed:\n\n"
                + "\n\n".join(failures)
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
