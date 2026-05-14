"""
Token usage benchmark.

For a set of natural-language tasks, ask an LLM to produce:
  (a) a Flow program (with our system prompt + few-shot)
  (b) a Python program (with a minimal "Python only" prompt)

…and compare token counts. Per-task results + a totals row are printed.

Caveats:
  - Output-token count is the headline number — that's what scales linearly
    with the work done. Input tokens include our system prompt (large) and
    are mostly fixed overhead.
  - Real-world chat APIs let you cache the system prompt, so the input cost
    is amortized. Take the numbers as a directional comparison, not as
    deployment-cost truth.

CLI:
  python -m flow bench           # use real LLM (FLOW_LLM_ENDPOINT)
  python -m flow bench --mock    # use canned responses, no network
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from .gen import _build_messages, _chat, LLMConfig


# A small bench corpus. Tasks should be:
#  - short (1-2 sentences) so the model can attempt them
#  - reasonably representative of Flow's intended use
BENCHMARK_TASKS: List[str] = [
    "Read users.json, keep adults (age >= 18), print each name.",
    "Fetch https://api.example.com/data, count items, print the count.",
    "Read input.txt, uppercase the contents, save as output.txt.",
    "Loop 5 times: ask the user for input, collect answers into a list, print.",
    "Translate the text in input.txt to Thai and save to output.txt.",
    "Read scores.json, sort by score descending, take top 3, print each name.",
]


_PY_SYSTEM = (
    "You are a Python code generator. Output Python source code only — "
    "no markdown fences, no commentary. Use standard library where possible."
)


# ---------- token counting ----------


def estimate_tokens(s: str) -> int:
    """Token count, preferring tiktoken if available."""
    try:
        import tiktoken                              # type: ignore
        enc = tiktoken.encoding_for_model("gpt-4o-mini")
        return len(enc.encode(s))
    except Exception:
        # Cheap fallback: ~4 chars per English token.
        return max(1, (len(s) + 3) // 4)


# ---------- run ----------


@dataclass
class BenchRow:
    task: str
    flow_in: int
    flow_out: int
    py_in: int
    py_out: int


def run_bench(call: Callable[[list], str],
              tasks: Optional[List[str]] = None) -> List[BenchRow]:
    rows: List[BenchRow] = []
    for task in (tasks or BENCHMARK_TASKS):
        flow_msgs = _build_messages(task)
        flow_in = sum(estimate_tokens(m["content"]) for m in flow_msgs)
        flow_reply = call(flow_msgs)
        flow_out = estimate_tokens(flow_reply)

        py_msgs = [
            {"role": "system", "content": _PY_SYSTEM},
            {"role": "user",   "content": task},
        ]
        py_in = sum(estimate_tokens(m["content"]) for m in py_msgs)
        py_reply = call(py_msgs)
        py_out = estimate_tokens(py_reply)

        rows.append(BenchRow(task, flow_in, flow_out, py_in, py_out))
    return rows


def format_report(rows: List[BenchRow]) -> str:
    out: List[str] = []
    header = f"{'task':<58} {'flow_in':>8} {'flow_out':>9} {'py_in':>7} {'py_out':>7}"
    out.append(header)
    out.append("-" * len(header))
    t_fi = t_fo = t_pi = t_po = 0
    for r in rows:
        task = (r.task[:55] + "...") if len(r.task) > 58 else r.task
        out.append(f"{task:<58} {r.flow_in:>8} {r.flow_out:>9} {r.py_in:>7} {r.py_out:>7}")
        t_fi += r.flow_in; t_fo += r.flow_out
        t_pi += r.py_in;   t_po += r.py_out
    out.append("-" * len(header))
    out.append(f"{'TOTAL':<58} {t_fi:>8} {t_fo:>9} {t_pi:>7} {t_po:>7}")

    def pct(new, old):
        if old <= 0:
            return "n/a"
        return f"{((new - old) / old) * 100:+.1f}%"

    out.append("")
    out.append(f"OUTPUT tokens: Flow vs Python: {pct(t_fo, t_po)}")
    out.append(f"INPUT  tokens: Flow vs Python: {pct(t_fi, t_pi)} "
               "(Flow input is fixed overhead — cacheable in real APIs)")
    out.append(f"TOTAL  tokens: Flow vs Python: {pct(t_fi + t_fo, t_pi + t_po)}")
    return "\n".join(out)


# ---------- mock for tests / dry runs ----------


def make_mock_responses():
    """Canned plausible replies — Flow shorter than Python."""
    flow_for = {
        "users.json":         'load file=users.json -> users\nfilter from=users where="x[\'age\'] >= 18" -> adults\neach u in adults\n  print value=u.name',
        "https://api":        'http_get url="https://api.example.com/data" -> d\ncount of=d -> n\nprint value=n',
        "input.txt":          'read file=input.txt -> text\nupper text=text -> big\nwrite file=output.txt text=big',
        "5 times":            'split text="" sep="," -> answers\nrepeat 5\n  ask prompt="? " -> a\n  print value=a',
        "Translate":          'read file=input.txt -> t\ntranslate text=t to=th -> th\nwrite file=output.txt text=th',
        "scores.json":        'load file=scores.json -> items\nsort from=items by="-x[\'score\']" -> ranked\ntake from=ranked n=3 -> top\neach i in top\n  print value=i.name',
    }
    py_for = {
        "users.json":  'import json\nusers=json.load(open("users.json"))\nfor u in users:\n    if u["age"]>=18:\n        print(u["name"])',
        "https://api": 'import requests\nd=requests.get("https://api.example.com/data").json()\nprint(len(d))',
        "input.txt":   'open("output.txt","w").write(open("input.txt").read().upper())',
        "5 times":     'answers=[]\nfor _ in range(5):\n    answers.append(input("? "))\nprint(answers)',
        "Translate":   'import requests\nt=open("input.txt").read()\nopen("output.txt","w").write(requests.post("https://t.example.com",json={"text":t,"to":"th"}).json()["t"])',
        "scores.json": 'import json\nitems=json.load(open("scores.json"))\ntop=sorted(items,key=lambda x:-x["score"])[:3]\nfor i in top: print(i["name"])',
    }

    def call(messages):
        # The last user message contains the task; pick a fragment to dispatch.
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        is_flow = "Flow" in sys_msg
        bank = flow_for if is_flow else py_for
        for key, val in bank.items():
            if key.lower() in user.lower():
                return val
        return "print('hello')" if not is_flow else 'print value="hello"'

    return call


# ---------- CLI ----------


def cli_main(args) -> None:
    import os, sys
    if args.mock:
        call = make_mock_responses()
    else:
        cfg = LLMConfig(
            endpoint=args.endpoint or os.environ.get("FLOW_LLM_ENDPOINT", "http://localhost:11434/v1"),
            model=args.model or os.environ.get("FLOW_LLM_MODEL", "llama3.2"),
            key=os.environ.get("FLOW_LLM_KEY", ""),
        )
        def call(msgs):
            return _chat(msgs, cfg)
    rows = run_bench(call)
    sys.stdout.write(format_report(rows) + "\n")
