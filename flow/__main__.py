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
    """Run parse + compile + lint + shrink as a one-stop diagnostic.

    Quiet by default — exits 0 on success with a small status block. With
    --verbose, also lists each lint warning.
    """
    from .lint import lint_source
    from .shrink import shrink_source
    src = _read(args.file)
    try:
        ast = parse(src)
        compile_to(ast, "python")
    except (ParseError, CompileError) as e:
        print(f"[parse/compile] FAIL: {e}", file=sys.stderr)
        sys.exit(2)

    warnings = lint_source(src)
    shrunk = shrink_source(src)
    save_pct = (1 - len(shrunk) / len(src)) * 100 if src else 0

    print(f"[parse]   ok")
    print(f"[compile] ok (python)")
    print(f"[lint]    {len(warnings)} suggestion{'s' if len(warnings) != 1 else ''}")
    print(f"[shrink]  {len(src):>5} → {len(shrunk):<5} chars  ({save_pct:+.1f}%)")
    if args.verbose and warnings:
        print()
        for w in warnings:
            print(f"  line {w.line}: {w.message}")
            print(f"    → {w.suggestion}")


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


def cmd_watch(args):
    """Live-re-check a file: poll mtime, rerun `flow check` on each save.

    Stdlib-only (no fsevent / inotify dep). Polls every 0.4s. Clears
    screen between runs unless --no-clear is set. Ctrl+C to stop.
    """
    import time
    path = Path(args.file)
    last_mtime = None
    last_size = None
    print(f"watching {path} — Ctrl+C to stop", file=sys.stderr)
    try:
        while True:
            try:
                st = path.stat()
                key = (st.st_mtime, st.st_size)
            except FileNotFoundError:
                key = None
            if key != (last_mtime, last_size):
                last_mtime, last_size = (key or (None, None))
                if not args.no_clear:
                    sys.stdout.write("\033[2J\033[H")
                    sys.stdout.flush()
                # Build a fake-args shim for cmd_check.
                class _A: pass
                a = _A(); a.file = args.file; a.verbose = args.verbose
                try:
                    cmd_check(a)
                except SystemExit:
                    pass
                if args.run:
                    print(file=sys.stderr)
                    print("--- run ---", file=sys.stderr)
                    sys.stderr.flush()
                    # Build run-args shim.
                    ra = _A(); ra.file = args.file; ra.to = args.run_target
                    try:
                        cmd_run(ra)
                    except SystemExit:
                        pass
                sys.stdout.flush()
                print(file=sys.stderr)
                print(f"-- watching {path} for changes --", file=sys.stderr)
                sys.stderr.flush()
            time.sleep(0.4)
    except KeyboardInterrupt:
        print("\nstopped", file=sys.stderr)


def cmd_run(args):
    """Compile and execute the Flow program in one step.

    Currently runnable targets: python (always), js (needs `node`),
    bash (needs `bash`). Other targets compile but printing the result
    is delegated to the user (e.g. `flow compile foo.flow --to go`).
    """
    import subprocess
    src = _read(args.file)
    try:
        ast = parse(src)
        out = compile_to(ast, args.to)
    except (ParseError, CompileError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    runners = {
        "python": [sys.executable, "-c", out],
        "js":     ["node", "-e", out],
        "bash":   ["bash", "-c", out],
    }
    cmd = runners.get(args.to)
    if cmd is None:
        print(f"`flow run` can't directly execute {args.to!r}; "
              f"use `flow compile {args.file} --to {args.to}` and run the output yourself.",
              file=sys.stderr)
        sys.exit(2)
    try:
        r = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print(f"runtime not found: {cmd[0]}", file=sys.stderr)
        sys.exit(2)
    sys.exit(r.returncode)


def cmd_stats(args):
    """Show char count, statement breakdown, and shrink potential for a file."""
    from .shrink import shrink_source
    src = _read(args.file)
    try:
        program = parse(src)
        shrunk = shrink_source(src)
    except ParseError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    counts = _stats_count_features(program.body)

    src_chars = len(src)
    shrunk_chars = len(shrunk)
    saving = ((src_chars - shrunk_chars) / src_chars * 100) if src_chars else 0
    src_lines = len([l for l in src.splitlines() if l.strip() and not l.strip().startswith("#")])

    print(f"file:          {args.file}")
    print(f"chars:         {src_chars}")
    print(f"chars (shrunk):{shrunk_chars:>5}  ({saving:+.1f}%)")
    print(f"code lines:    {src_lines}")
    print(f"stmt total:    {sum(counts.values())}")
    if not counts:
        return
    print()
    print("statements by type:")
    for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {name:18}  {n}")


def _stats_count_features(body):
    counts: dict = {}
    def bump(name):
        counts[name] = counts.get(name, 0) + 1
    def walk(stmts):
        for s in stmts:
            kind = type(s).__name__
            bump(kind)
            for attr in ("body", "then", "else_", "try_body", "catch_body"):
                child = getattr(s, attr, None)
                if isinstance(child, list):
                    walk(child)
    walk(body)
    return counts


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

    pk = sub.add_parser("check", help="one-stop diagnostic: parse + compile + lint + shrink stats")
    pk.add_argument("file")
    pk.add_argument("-v", "--verbose", action="store_true",
                    help="also list each lint suggestion")
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
    pg.add_argument("--prompt", choices=["full", "minimal"], default="full",
                    help="prompt mode: full (~5K chars) or minimal (~1.7K) for tiny LLMs")
    pg.add_argument("--rounds", type=int, default=1,
                    help="best-of-N sampling: generate N candidates, return the shortest valid one")
    pg.add_argument("--cache", action="store_true",
                    help="reuse identical request results from ~/.flow/gen_cache.json")
    pg.add_argument("--cache-show", action="store_true",
                    help="list cached entries and exit (no LLM call)")
    pg.add_argument("--cache-clear", action="store_true",
                    help="wipe the disk cache and exit")
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
    pl.add_argument("--fix", action="store_true",
                    help="apply automated rewrites (delegates to `flow shrink`)")
    pl.add_argument("-w", "--write", action="store_true",
                    help="with --fix, overwrite the file in place")
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

    pst = sub.add_parser("stats", help="report char count, statement counts, and shrink potential")
    pst.add_argument("file")
    pst.set_defaults(func=cmd_stats)

    pr = sub.add_parser("run", help="compile + execute a Flow program (python / js / bash)")
    pr.add_argument("file")
    pr.add_argument("--to", choices=["python", "js", "bash"], default="python",
                    help="target runtime (default: python)")
    pr.set_defaults(func=cmd_run)

    pw = sub.add_parser("watch", help="live-rerun `flow check` whenever the file changes")
    pw.add_argument("file")
    pw.add_argument("-v", "--verbose", action="store_true",
                    help="show each lint suggestion on every run")
    pw.add_argument("--no-clear", action="store_true",
                    help="don't clear the screen between runs")
    pw.add_argument("--run", action="store_true",
                    help="also compile + execute on each save")
    pw.add_argument("--run-target", choices=["python", "js", "bash"], default="python",
                    help="target runtime for --run (default: python)")
    pw.set_defaults(func=cmd_watch)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
