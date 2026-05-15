"""Flow CLI.

Usage:
  python -m flow parse   <file.flow>            → print AST as JSON
  python -m flow compile <file.flow> [--to LANG] → emit Python (default) or JS
  python -m flow render  <file.flow> [-o path]  → emit standalone HTML w/ blocks
  python -m flow check   <file.flow>            → parse + compile, print "ok" or error
  python -m flow serve   [--port N]             → run playground at http://127.0.0.1:PORT
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import parse, compile_to, ast_to_dict, ParseError, CompileError


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def cmd_parse(args):
    src = _read(args.file)
    try:
        ast = parse(src)
    except ParseError as e:
        print(f"PARSE ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    json.dump(ast_to_dict(ast), sys.stdout, indent=2, ensure_ascii=False)
    print()


def cmd_compile(args):
    src = _read(args.file)
    try:
        ast = parse(src)
        out = compile_to(ast, args.to)
    except (ParseError, CompileError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    print(out)


def cmd_check(args):
    src = _read(args.file)
    try:
        ast = parse(src)
        compile_to(ast, "python")
    except (ParseError, CompileError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    print("ok")


def cmd_fmt(args):
    from .formatter import format_source
    src = _read(args.file)
    try:
        ast = parse(src)
    except ParseError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    formatted = format_source(ast)
    if args.write:
        Path(args.file).write_text(formatted, encoding="utf-8")
    else:
        sys.stdout.write(formatted)


def cmd_serve(args):
    from .server import serve
    serve(host=args.host, port=args.port)


def cmd_gen(args):
    from .gen import cli_main
    cli_main(args)


def cmd_review(args):
    from .review import cli_main
    cli_main(args)


def cmd_bench(args):
    from .bench import cli_main
    cli_main(args)


def cmd_lint(args):
    from .lint import cli_main
    cli_main(args)


def cmd_shrink(args):
    from .shrink import shrink_source
    src = _read(args.file)
    try:
        out = shrink_source(src)
    except ParseError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    if args.write:
        Path(args.file).write_text(out, encoding="utf-8")
    else:
        sys.stdout.write(out)


def cmd_examples(args):
    """Print curated example .flow programs. `--list` shows names + summaries,
    otherwise `--all` dumps every example, or `<name>` prints one example."""
    examples_dir = Path(__file__).resolve().parent.parent / "examples"
    files = sorted(examples_dir.glob("*.flow"))
    if not files:
        print("(no examples in examples/)", file=sys.stderr)
        sys.exit(2)

    if args.list:
        for p in files:
            summary = ""
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("#"):
                    summary = line.lstrip("# ").strip()
                    break
                if line:
                    break
            print(f"{p.stem:18}  {summary}")
        return

    if args.name:
        target = examples_dir / f"{args.name}.flow"
        if not target.exists():
            print(f"ERROR: no example named {args.name!r}", file=sys.stderr)
            sys.exit(2)
        sys.stdout.write(target.read_text(encoding="utf-8"))
        return

    # Default: dump everything (handy for LLM context priming).
    for p in files:
        sys.stdout.write(f"# === {p.stem} ===\n")
        sys.stdout.write(p.read_text(encoding="utf-8"))
        sys.stdout.write("\n")


def cmd_render(args):
    """Emit a standalone HTML file that renders this program's blocks.

    The shipped renderer/index.html already includes a built-in mini-parser,
    so we just inline the source into its textarea on load.
    """
    src = _read(args.file)
    here = Path(__file__).resolve().parent.parent
    template = (here / "renderer" / "index.html").read_text(encoding="utf-8")
    # Replace the initial sample with this program's source.
    needle = "srcEl.value = `"
    idx = template.find(needle)
    if idx < 0:
        print("ERROR: renderer template missing sample marker", file=sys.stderr)
        sys.exit(2)
    end = template.find("`;", idx + len(needle))
    if end < 0:
        print("ERROR: renderer template missing sample terminator", file=sys.stderr)
        sys.exit(2)
    new_html = template[: idx + len(needle)] + _js_escape(src) + template[end:]
    out_path = args.o or args.file.rsplit(".", 1)[0] + ".html"
    Path(out_path).write_text(new_html, encoding="utf-8")
    print(out_path)


def _js_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")


def main():
    p = argparse.ArgumentParser(prog="flow")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("parse", help="parse and print AST as JSON")
    pp.add_argument("file")
    pp.set_defaults(func=cmd_parse)

    pc = sub.add_parser("compile", help="compile to target language")
    pc.add_argument("file")
    pc.add_argument("--to", choices=["python", "js", "go", "rust", "bash"], default="python")
    pc.set_defaults(func=cmd_compile)

    pk = sub.add_parser("check", help="parse + compile silently")
    pk.add_argument("file")
    pk.set_defaults(func=cmd_check)

    pf = sub.add_parser("fmt", help="reformat Flow source canonically")
    pf.add_argument("file")
    pf.add_argument("-w", "--write", action="store_true", help="overwrite the file in place")
    pf.set_defaults(func=cmd_fmt)

    pr = sub.add_parser("render", help="generate standalone HTML block view")
    pr.add_argument("file")
    pr.add_argument("-o", help="output HTML path")
    pr.set_defaults(func=cmd_render)

    ps = sub.add_parser("serve", help="run the playground (editor + blocks + compiled output)")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8765)
    ps.set_defaults(func=cmd_serve)

    pg = sub.add_parser("gen", help="generate Flow from natural language via a local LLM")
    pg.add_argument("request", nargs="?", default="", help="natural-language request")
    pg.add_argument("-f", help="read request from this file instead")
    pg.add_argument("--endpoint", help="OpenAI-compatible endpoint (default: env FLOW_LLM_ENDPOINT or http://localhost:11434/v1)")
    pg.add_argument("--model", help="model name (default: env FLOW_LLM_MODEL or llama3.2)")
    pg.add_argument("--retries", type=int, default=3, help="self-correction retries on parse/compile error")
    pg.add_argument("--verbose", action="store_true", help="show each attempt to stderr")
    pg.add_argument("--no-lint", action="store_true",
                    help="skip the lint-driven polish pass after success")
    pg.set_defaults(func=cmd_gen)

    pv = sub.add_parser("review", help="compile + run + ask a big LLM to judge correctness")
    pv.add_argument("file")
    pv.add_argument("--intent", help="what the user wanted the program to do (free text)")
    pv.add_argument("--to", choices=["python", "js"], default="python", help="target language")
    pv.add_argument("--no-run", action="store_true", help="skip executing the compiled code")
    pv.set_defaults(func=cmd_review)

    pb = sub.add_parser("bench", help="benchmark token usage: Flow vs raw Python")
    pb.add_argument("--mock", action="store_true", help="use canned responses (no network)")
    pb.add_argument("--endpoint", help="OpenAI-compatible endpoint (default: env or http://localhost:11434/v1)")
    pb.add_argument("--model",    help="model name (default: env or llama3.2)")
    pb.set_defaults(func=cmd_bench)

    pl = sub.add_parser("lint", help="report verbose Flow patterns with shorter equivalents")
    pl.add_argument("file")
    pl.add_argument("--fail", action="store_true", help="exit 1 if any warnings")
    pl.set_defaults(func=cmd_lint)

    psh = sub.add_parser("shrink", help="rewrite Flow source to the shortest equivalent form")
    psh.add_argument("file")
    psh.add_argument("-w", "--write", action="store_true",
                     help="overwrite the file in place")
    psh.set_defaults(func=cmd_shrink)

    pe = sub.add_parser("examples", help="print curated example programs")
    pe.add_argument("name", nargs="?", help="example name (e.g. functions); omit to dump all")
    pe.add_argument("--list", action="store_true",
                    help="list example names + one-line summaries")
    pe.set_defaults(func=cmd_examples)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
