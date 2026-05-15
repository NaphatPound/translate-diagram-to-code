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

# ---------- on-disk cache (opt-in) ----------


def _cache_path():
    """Per-user JSON cache for `flow gen --cache`."""
    return Path.home() / ".flow" / "gen_cache.json"


def _cache_key(request: str, prompt: str, retries: int, polish: bool, rounds: int) -> str:
    """Stable hash of every input that affects the output."""
    import hashlib
    parts = f"{request}\x00{prompt}\x00{retries}\x00{int(polish)}\x00{rounds}"
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()[:16]


def _cache_load() -> dict:
    p = _cache_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _cache_save(data: dict) -> None:
    p = _cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def cache_clear() -> int:
    """Delete the disk cache. Returns the number of entries removed."""
    p = _cache_path()
    if not p.exists():
        return 0
    n = len(_cache_load())
    p.unlink()
    return n


def cache_show(stream=None) -> int:
    """Pretty-print cache entries (key + first line of output). Returns the count."""
    import sys as _sys
    stream = stream or _sys.stdout
    data = _cache_load()
    if not data:
        stream.write("(cache empty)\n")
        return 0
    p = _cache_path()
    size = p.stat().st_size if p.exists() else 0
    stream.write(f"{len(data)} entr{'y' if len(data) == 1 else 'ies'} at {p} ({size} bytes)\n\n")
    for key, out in data.items():
        first = out.splitlines()[0] if out else "(empty)"
        if len(first) > 60:
            first = first[:57] + "..."
        stream.write(f"  {key}  {first}\n")
    return len(data)

_HERE = Path(__file__).resolve().parent.parent
_PROMPTS = _HERE / "prompts"


# ---------- prompt assembly ----------


def _load_text(name: str) -> str:
    p = _PROMPTS / name
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def _build_messages(user_request: str, prompt: str = "full",
                    include_doc: bool = False) -> List[dict]:
    """Build the chat-completions message list.

    `prompt`:
      "full"    — system.md + few_shot.md  (default, ~5K chars)
      "minimal" — minimal.md only          (~1.7K chars, for tiny LLMs)

    `include_doc` appends the live verb registry (compact form, ~2K chars)
    so the model sees every legal verb + signature. Recommended when the
    request may need verbs that aren't in the few-shot examples.
    """
    if prompt == "minimal":
        system = _load_text("minimal.md") or _DEFAULT_SYSTEM
    else:
        system = _load_text("system.md") or _DEFAULT_SYSTEM
        few_shot = _load_text("few_shot.md")
        if few_shot:
            system = system + "\n\n" + few_shot
    if include_doc:
        from .verbs import verb_reference
        system = system + "\n\n## Full verb list\n" + verb_reference(compact=True)
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
    prompt: str = "full",
    rounds: int = 1,
    cache: bool = False,
    include_doc: bool = False,
) -> str:
    """Generate Flow code from a natural-language request, with self-correction.

    `prompt` = "full" (default) sends system.md + few_shot.md (~5K chars).
    `prompt` = "minimal" sends only minimal.md (~1.7K chars), useful for
    tiny LLMs whose context window can't hold the long version.

    `rounds` (default 1) controls best-of-N sampling. With rounds > 1, we
    run the full generate-and-validate loop N times and return the SHORTEST
    valid candidate. Useful for non-deterministic LLMs where output quality
    varies.

    When `polish` is True (default), after each successful candidate we run
    `flow.shrink` (deterministic) to compact the output.

    `cache` (default False) reuses validated outputs stored on disk at
    `~/.flow/gen_cache.json` — identical requests skip the LLM entirely.
    """
    cache_key = None
    if cache:
        cache_key = _cache_key(request, prompt, retries, polish, rounds)
        cached = _cache_load().get(cache_key)
        if cached is not None:
            if verbose:
                print(f"--- cache hit {cache_key} ---", file=sys.stderr)
            return cached
    if rounds > 1:
        out = _generate_best_of_n(
            request, cfg=cfg, retries=retries, verbose=verbose,
            polish=polish, prompt=prompt, rounds=rounds,
            include_doc=include_doc,
        )
        if cache:
            data = _cache_load()
            data[cache_key] = out
            _cache_save(data)
        return out
    cfg = cfg or LLMConfig()
    messages = _build_messages(request, prompt=prompt, include_doc=include_doc)

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

        # Validate: parse + compile. Pass source so errors carry a caret
        # pointer the LLM can use to find the issue on retry.
        try:
            ast = parse(code)
            compile_to(ast, "python", source=code)
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
        result = code
    else:
        # Polish pass: deterministic shrink (no extra LLM call). Falls back to
        # the validated-but-verbose original if anything goes wrong.
        try:
            from .shrink import shrink_source
            polished = shrink_source(code)
            # Re-validate just to be paranoid.
            compile_to(parse(polished), "python")
            if verbose and polished != code:
                print("--- polished by flow.shrink ---", file=sys.stderr)
            result = polished
        except (ParseError, CompileError, Exception):
            result = code
    if cache and cache_key:
        data = _cache_load()
        data[cache_key] = result
        _cache_save(data)
    return result


def _generate_best_of_n(
    request: str,
    cfg: Optional[LLMConfig],
    retries: int,
    verbose: bool,
    polish: bool,
    prompt: str,
    rounds: int,
    include_doc: bool = False,
) -> str:
    """Run `generate()` N times and return the shortest valid candidate."""
    candidates: list = []
    last_error: Optional[str] = None
    for r in range(1, rounds + 1):
        if verbose:
            print(f"--- round {r}/{rounds} ---", file=sys.stderr)
        try:
            out = generate(
                request, cfg=cfg, retries=retries, verbose=verbose,
                polish=polish, prompt=prompt, rounds=1,
                include_doc=include_doc,
            )
            candidates.append(out)
        except LLMError as e:
            last_error = str(e)
            if verbose:
                print(f"round {r} failed: {e}", file=sys.stderr)
    if not candidates:
        raise LLMError(
            f"all {rounds} rounds failed.\nLast error: {last_error}"
        )
    # Pick the shortest valid candidate (lexicographic tiebreaker for determinism).
    best = min(candidates, key=lambda c: (len(c), c))
    if verbose:
        sizes = sorted(len(c) for c in candidates)
        print(f"--- picked best-of-{len(candidates)}: {len(best)} chars "
              f"(sizes: {sizes}) ---", file=sys.stderr)
    return best


# ---------- CLI integration ----------


def cli_main(args) -> None:
    # Cache management short-circuits — no LLM call needed.
    if getattr(args, "cache_show", False):
        cache_show()
        return
    if getattr(args, "cache_clear", False):
        n = cache_clear()
        print(f"cleared {n} cache entr{'y' if n == 1 else 'ies'}", file=sys.stderr)
        return

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
            prompt=getattr(args, "prompt", "full"),
            rounds=getattr(args, "rounds", 1),
            cache=getattr(args, "cache", False),
            include_doc=getattr(args, "include_doc", False),
        )
    except LLMError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    print(code)
