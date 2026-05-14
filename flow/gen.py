"""
Local LLM loop: natural language → Flow code, with parser-error self-correction.

Talks to any OpenAI-compatible chat-completions endpoint:
  - Ollama       (http://localhost:11434/v1)
  - LM Studio    (http://localhost:1234/v1)
  - llama.cpp    (with --api flag)
  - vLLM, Together, etc.

Usage (Python):
    from flow.gen import generate
    code = generate("read users.json, print each name")

Usage (CLI):
    python -m flow gen "read users.json, print each name"
    python -m flow gen -f request.txt --model llama3.2 --retries 3

The loop:
  1. Send system prompt + few-shot examples + user request.
  2. Read the assistant's reply (stripped of any fences).
  3. Parse with flow.parse. If ParseError, append the error message and retry.
  4. Optionally also compile() to catch verb-level issues.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from . import parse, compile_to, ParseError, CompileError


DEFAULT_ENDPOINT = os.environ.get("FLOW_LLM_ENDPOINT", "http://localhost:11434/v1")
DEFAULT_MODEL    = os.environ.get("FLOW_LLM_MODEL",    "llama3.2")
DEFAULT_KEY      = os.environ.get("FLOW_LLM_KEY",      "")  # often empty for local

_HERE = Path(__file__).resolve().parent.parent
_PROMPTS = _HERE / "prompts"


# ---------- prompt assembly ----------


def _load_text(name: str) -> str:
    p = _PROMPTS / name
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def _build_messages(user_request: str) -> List[dict]:
    """Build the chat-completions message list."""
    system = _load_text("system.md") or _DEFAULT_SYSTEM
    few_shot = _load_text("few_shot.md")
    if few_shot:
        # Append few-shot as additional system context.
        system = system + "\n\n" + few_shot
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_request},
    ]


_DEFAULT_SYSTEM = """You are a code generator that writes Flow, a small DSL.
Format: `verb arg=value arg=value -> name`. Indent 2 spaces. Output Flow source only.
"""


# ---------- LLM call ----------


@dataclass
class LLMConfig:
    endpoint: str = DEFAULT_ENDPOINT
    model: str = DEFAULT_MODEL
    key: str = DEFAULT_KEY
    temperature: float = 0.1
    timeout: float = 60.0


def _chat(messages: List[dict], cfg: LLMConfig) -> str:
    url = cfg.endpoint.rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    if cfg.key:
        req.add_header("Authorization", f"Bearer {cfg.key}")
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise LLMError(f"could not reach {url}: {e}") from e
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise LLMError(f"unexpected response shape: {json.dumps(data)[:200]}")


class LLMError(Exception):
    pass


# ---------- extract Flow source from reply ----------

_FENCE_RE = re.compile(r"```(?:flow|)\s*\n(.*?)```", re.DOTALL)


def _extract_code(reply: str) -> str:
    """Pull Flow source out of the LLM reply, tolerating fences or commentary."""
    m = _FENCE_RE.search(reply)
    if m:
        return m.group(1).strip()
    return reply.strip()


# ---------- main loop ----------


def generate(
    request: str,
    cfg: Optional[LLMConfig] = None,
    retries: int = 3,
    verbose: bool = False,
    polish: bool = True,
) -> str:
    """Generate Flow code from a natural-language request, with self-correction.

    When `polish` is True (default), after the LLM produces valid code we run
    `flow lint` and, if there are suggestions, ask the LLM once to rewrite in
    the shorter form. We keep the polished version only if it ALSO validates.
    """
    cfg = cfg or LLMConfig()
    messages = _build_messages(request)

    last_error = None
    last_code = ""
    code = None
    for attempt in range(1, retries + 2):
        if verbose:
            print(f"--- attempt {attempt} ---", file=sys.stderr)
        reply = _chat(messages, cfg)
        code = _extract_code(reply)
        last_code = code
        if verbose:
            print(code, file=sys.stderr)

        # Validate: parse + compile.
        try:
            ast = parse(code)
            compile_to(ast, "python")   # checks verb/arg validity too
            break  # accepted — proceed to polish
        except (ParseError, CompileError) as e:
            last_error = str(e)
            if verbose:
                print(f"error: {last_error}", file=sys.stderr)
            if attempt > retries:
                raise LLMError(
                    f"failed after {retries} retries.\nLast error: {last_error}\n"
                    f"Last code:\n{last_code}"
                )
            # Feed the error back so the model can fix it.
            messages.append({"role": "assistant", "content": code})
            messages.append({
                "role": "user",
                "content": (
                    f"The above code failed validation:\n  {last_error}\n\n"
                    f"Return a corrected, complete Flow program. "
                    f"Output Flow source only, no fences, no commentary."
                ),
            })
            code = None

    assert code is not None  # we broke out of the loop above
    if not polish:
        return code

    # Polish pass: deterministic shrink (no extra LLM call). Falls back to the
    # validated-but-verbose original if anything goes wrong.
    try:
        from .shrink import shrink_source
        polished = shrink_source(code)
        # Re-validate just to be paranoid.
        compile_to(parse(polished), "python")
        if verbose and polished != code:
            print("--- polished by flow.shrink ---", file=sys.stderr)
        return polished
    except (ParseError, CompileError, Exception):
        return code


# ---------- CLI integration ----------


def cli_main(args) -> None:
    request = args.request
    if args.f:
        request = Path(args.f).read_text(encoding="utf-8")
    if not request:
        print("ERROR: no request provided (pass text or -f FILE)", file=sys.stderr)
        sys.exit(2)
    cfg = LLMConfig(
        endpoint=args.endpoint or DEFAULT_ENDPOINT,
        model=args.model or DEFAULT_MODEL,
        key=os.environ.get("FLOW_LLM_KEY", ""),
    )
    try:
        code = generate(
            request, cfg=cfg,
            retries=args.retries,
            verbose=args.verbose,
            polish=not getattr(args, "no_lint", False),
        )
    except LLMError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    print(code)
