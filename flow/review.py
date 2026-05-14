"""
Big-LLM review pass.

Pipeline:
  1. Compile Flow to a target language (default Python).
  2. Run the compiled code in a subprocess sandbox (cwd = tempdir, timeout).
  3. Send {original Flow, compiled code, run output, optional intent} to a
     strong LLM (Anthropic or OpenAI) and ask it to judge correctness.

Auth: set ANTHROPIC_API_KEY (preferred) or OPENAI_API_KEY in the environment.

CLI:
  python -m flow review file.flow                       # judges by source alone
  python -m flow review file.flow --intent "<text>"     # check against intent
  python -m flow review file.flow --no-run              # skip execution
  python -m flow review file.flow --to js               # target JS (no auto-run)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import parse, compile_to


# ---------- sandbox run ----------


@dataclass
class RunResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


def run_compiled(source: str, lang: str, timeout: float = 8.0) -> RunResult:
    """Run compiled source in a tempdir; capture stdout/stderr."""
    if lang == "python":
        cmd = [sys.executable, "-c", source]
    elif lang == "js":
        cmd = ["node", "-e", source]
    else:
        raise ValueError(f"can't run language {lang!r}")
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=tempfile.gettempdir(),
        )
    except subprocess.TimeoutExpired as e:
        return RunResult(stdout=e.stdout or "", stderr=e.stderr or "",
                         exit_code=-1, timed_out=True)
    except FileNotFoundError:
        return RunResult(stdout="", stderr=f"runtime not found: {cmd[0]}",
                         exit_code=-1, timed_out=False)
    return RunResult(stdout=r.stdout, stderr=r.stderr,
                     exit_code=r.returncode, timed_out=False)


# ---------- prompt ----------


_REVIEW_TEMPLATE = """You are a strict code reviewer.

A user wrote this Flow DSL program:
```
{flow_src}
```

The Flow compiler translated it to {lang}:
```{lang}
{target_src}
```
{intent_section}{run_section}
Your job: review for correctness.

Answer with these sections, each prefixed exactly as shown:
VERDICT: ok | issues | broken
SUMMARY: <one-sentence judgement>
ISSUES:
  - <bullet, or "none">
SUGGEST:
  - <concrete fix to the Flow source, or "none">

Be concise. Focus on real bugs, not style. Trust the compiler — assume its
output is a faithful translation; check whether the *intent* makes sense and
whether the program behaves as the user wants."""


def build_prompt(flow_src: str, target_src: str, lang: str,
                 run: Optional[RunResult], intent: Optional[str]) -> str:
    intent_section = ""
    if intent:
        intent_section = f"\nThe user described their intent as:\n  {intent.strip()}\n"
    run_section = ""
    if run:
        ttag = " (timed out)" if run.timed_out else ""
        run_section = (
            f"\nWhen executed{ttag} (exit={run.exit_code}):\n"
            f"  stdout: {run.stdout!r}\n"
            f"  stderr: {run.stderr!r}\n"
        )
    return _REVIEW_TEMPLATE.format(
        flow_src=flow_src.strip(),
        lang=lang,
        target_src=target_src.strip(),
        intent_section=intent_section,
        run_section=run_section,
    )


# ---------- LLM calls ----------


class ReviewError(Exception):
    pass


def _call_anthropic(prompt: str, model: str, key: str, timeout: float) -> str:
    url = "https://api.anthropic.com/v1/messages"
    payload = json.dumps({
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("x-api-key", key)
    req.add_header("anthropic-version", "2023-06-01")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise ReviewError(f"Anthropic API error {e.code}: {body[:300]}")
    except urllib.error.URLError as e:
        raise ReviewError(f"Anthropic request failed: {e}") from e
    try:
        return data["content"][0]["text"]
    except (KeyError, IndexError):
        raise ReviewError(f"unexpected Anthropic response: {json.dumps(data)[:200]}")


def _call_openai(prompt: str, model: str, key: str, timeout: float) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise ReviewError(f"OpenAI API error {e.code}: {body[:300]}")
    except urllib.error.URLError as e:
        raise ReviewError(f"OpenAI request failed: {e}") from e
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise ReviewError(f"unexpected OpenAI response: {json.dumps(data)[:200]}")


def call_llm(prompt: str, timeout: float = 60.0) -> str:
    """Pick provider based on env vars; raise if no key is set."""
    anth = os.environ.get("ANTHROPIC_API_KEY")
    if anth:
        model = os.environ.get("FLOW_REVIEW_MODEL", "claude-opus-4-7")
        return _call_anthropic(prompt, model, anth, timeout)
    oai = os.environ.get("OPENAI_API_KEY")
    if oai:
        model = os.environ.get("FLOW_REVIEW_MODEL", "gpt-4o-mini")
        return _call_openai(prompt, model, oai, timeout)
    raise ReviewError("no API key found (set ANTHROPIC_API_KEY or OPENAI_API_KEY)")


# ---------- orchestrate ----------


@dataclass
class Review:
    flow_src: str
    target_src: str
    lang: str
    run: Optional[RunResult]
    verdict: str    # ok | issues | broken | unknown
    summary: str
    issues: list
    suggest: list
    raw: str


def review(flow_src: str, lang: str = "python",
           intent: Optional[str] = None, run: bool = True,
           call: callable = None) -> Review:
    """Compile + optionally run + call LLM. Returns parsed Review."""
    ast = parse(flow_src)
    target_src = compile_to(ast, lang)
    rr = run_compiled(target_src, lang) if run and lang in ("python", "js") else None
    prompt = build_prompt(flow_src, target_src, lang, rr, intent)
    raw = (call or call_llm)(prompt)
    parsed = _parse_review(raw)
    return Review(
        flow_src=flow_src, target_src=target_src, lang=lang, run=rr,
        verdict=parsed["verdict"], summary=parsed["summary"],
        issues=parsed["issues"], suggest=parsed["suggest"], raw=raw,
    )


def _parse_review(text: str) -> dict:
    """Best-effort parse of the LLM's structured reply."""
    out = {"verdict": "unknown", "summary": "", "issues": [], "suggest": []}
    section = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("VERDICT:"):
            v = line.split(":", 1)[1].strip().lower()
            out["verdict"] = v if v in ("ok", "issues", "broken") else "unknown"
            section = None
        elif line.startswith("SUMMARY:"):
            out["summary"] = line.split(":", 1)[1].strip()
            section = None
        elif line.startswith("ISSUES:"):
            section = "issues"
        elif line.startswith("SUGGEST:"):
            section = "suggest"
        elif section and line.strip().startswith("-"):
            item = line.strip().lstrip("-").strip()
            if item.lower() != "none":
                out[section].append(item)
    return out


# ---------- CLI ----------


def cli_main(args) -> None:
    flow_src = Path(args.file).read_text(encoding="utf-8")
    try:
        rev = review(flow_src, lang=args.to, intent=args.intent, run=not args.no_run)
    except ReviewError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    color = {"ok": "\033[32m", "issues": "\033[33m", "broken": "\033[31m"}.get(rev.verdict, "")
    reset = "\033[0m" if color else ""
    print(f"{color}VERDICT:{reset} {rev.verdict}")
    print(f"SUMMARY: {rev.summary}")
    if rev.issues:
        print("ISSUES:")
        for i in rev.issues:
            print(f"  - {i}")
    if rev.suggest:
        print("SUGGEST:")
        for s in rev.suggest:
            print(f"  - {s}")
    if rev.run:
        print(f"(executed: exit={rev.run.exit_code}, "
              f"{'timed out' if rev.run.timed_out else 'normal'})")
