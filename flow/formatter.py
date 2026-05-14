"""
AST → Flow source (pretty-printer).

Emits canonical formatting:
  - 2-space indent
  - One statement per line
  - `verb arg=value arg=value -> name`
  - Strings always quoted; barewords (Name, no dots) emitted as-is
  - List/dict literals use `[...]` / `{key: value, ...}`

Round-trip property: ``parse(format(parse(src))) == parse(src)`` (structural).
"""
from __future__ import annotations

import json
import re
from typing import List

from .parser import (
    Program, Call, AssignStmt, IfStmt, EachStmt, RepeatStmt, WhenStmt,
    StringLit, NumberLit, BoolLit, Name, FuncCall, BinOp, Arg,
    ListLit, DictLit, Ternary, Range,
)

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BAREWORD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")
_INDENT = "  "


def format_source(program: Program) -> str:
    lines: List[str] = []
    usage = _count_var_usages(program.body)
    _emit_block(program.body, 0, lines, usage)
    return "\n".join(lines) + ("\n" if lines else "")


# ---------- statements ----------


def _emit_block(body, depth: int, out: List[str], usage: dict) -> None:
    i = 0
    while i < len(body):
        stmt = body[i]
        # Try to coalesce a pipe chain starting at i.
        chain_end = _try_pipe_chain(body, i, usage)
        if chain_end > i:
            line = _fmt_pipe_chain(body[i:chain_end + 1])
            out.append(_INDENT * depth + line)
            i = chain_end + 1
            continue
        _emit_stmt(stmt, depth, out, usage)
        i += 1


def _emit_stmt(stmt, depth: int, out: List[str], usage: dict) -> None:
    if isinstance(stmt, Call):
        out.append(_INDENT * depth + _fmt_call(stmt))
        return
    if isinstance(stmt, AssignStmt):
        out.append(f"{_INDENT * depth}{stmt.target} = {_fmt_value(stmt.value)}")
        return
    if isinstance(stmt, IfStmt):
        out.append(_INDENT * depth + "if " + _fmt_value(stmt.cond))
        _emit_block(stmt.then, depth + 1, out, usage)
        if stmt.else_:
            out.append(_INDENT * depth + "else")
            _emit_block(stmt.else_, depth + 1, out, usage)
        return
    if isinstance(stmt, EachStmt):
        out.append(f"{_INDENT * depth}each {stmt.var} in {_fmt_value(stmt.iterable)}")
        _emit_block(stmt.body, depth + 1, out, usage)
        return
    if isinstance(stmt, RepeatStmt):
        head = f"repeat {_fmt_value(stmt.count)}"
        if stmt.var:
            head += f" as {stmt.var}"
        out.append(_INDENT * depth + head)
        _emit_block(stmt.body, depth + 1, out, usage)
        return
    if isinstance(stmt, WhenStmt):
        extras = " ".join(_fmt_value(a) for a in stmt.args)
        out.append(f"{_INDENT * depth}when {stmt.event}" + (f" {extras}" if extras else ""))
        _emit_block(stmt.body, depth + 1, out, usage)
        return
    raise ValueError(f"unknown statement: {type(stmt).__name__}")


# ---------- pipe-chain detection ----------


_PIPE_TEMP_RE = re.compile(r"^_p\d+$")


def _try_pipe_chain(body, start: int, usage: dict) -> int:
    """If body[start..end] can be merged into a single pipe expression, return end.
    Otherwise return start (meaning no chain).

    Be conservative: only merge when the intermediate name is a parser-generated
    pipe temp (`_p\\d+`). User-named variables are preserved even if they're
    used exactly once — shrink() can do the more aggressive collapsing.
    """
    end = start
    while end + 1 < len(body):
        cur = body[end]
        nxt = body[end + 1]
        if not (isinstance(cur, Call) and cur.out and isinstance(nxt, Call)):
            break
        if not _PIPE_TEMP_RE.match(cur.out):
            break
        primary = _primary_arg(nxt.verb)
        if not primary:
            break
        matching = [a for a in nxt.args if _is_var_ref(a.value, cur.out)]
        if len(matching) != 1:
            break
        ref_arg = matching[0]
        if ref_arg.name not in (primary, "<pipe>", "<pos>"):
            break
        if usage.get(cur.out, 0) != 1:
            break
        end += 1
    return end


def _fmt_pipe_chain(chain) -> str:
    """Render `[call_a, call_b, call_c]` as `a-args | b-args | c-args`.
    For all but the first call, drop the primary-arg reference (it's implicit
    via the pipe).
    """
    rendered = []
    for idx, c in enumerate(chain):
        if idx == 0:
            # Keep all args; drop the auto-pipe `->` if its name is used as
            # the pipe input by the next call.
            rendered.append(_fmt_call_drop_pipe_arrow(c, drop_arrow=True))
        else:
            # Drop the arg that consumed the upstream value.
            primary = _primary_arg(c.verb)
            upstream_name = chain[idx - 1].out
            kept = [a for a in c.args if not _is_var_ref(a.value, upstream_name)]
            stub = Call(c.verb, kept,
                        c.out if idx == len(chain) - 1 else None,
                        c.line)
            rendered.append(_fmt_call(stub))
    return " | ".join(rendered)


def _fmt_call_drop_pipe_arrow(call: Call, drop_arrow: bool) -> str:
    """Like _fmt_call but skip emitting `-> name` if drop_arrow."""
    stub = Call(call.verb, call.args, None if drop_arrow else call.out, call.line)
    return _fmt_call(stub)


def _is_var_ref(value, varname: str) -> bool:
    return isinstance(value, Name) and len(value.parts) == 1 and value.parts[0] == varname


# ---------- usage analysis ----------


def _count_var_usages(body) -> dict:
    counts: dict = {}
    _count_in_block(body, counts)
    return counts


def _count_in_block(body, counts):
    for stmt in body:
        if isinstance(stmt, Call):
            for a in stmt.args:
                _count_in_value(a.value, counts)
        elif isinstance(stmt, AssignStmt):
            _count_in_value(stmt.value, counts)
        elif isinstance(stmt, IfStmt):
            _count_in_value(stmt.cond, counts)
            _count_in_block(stmt.then, counts)
            if stmt.else_:
                _count_in_block(stmt.else_, counts)
        elif isinstance(stmt, EachStmt):
            _count_in_value(stmt.iterable, counts)
            _count_in_block(stmt.body, counts)
        elif isinstance(stmt, RepeatStmt):
            _count_in_value(stmt.count, counts)
            _count_in_block(stmt.body, counts)
        elif isinstance(stmt, WhenStmt):
            for a in stmt.args:
                _count_in_value(a, counts)
            _count_in_block(stmt.body, counts)


def _count_in_value(value, counts):
    if isinstance(value, Name):
        counts[value.parts[0]] = counts.get(value.parts[0], 0) + 1
    elif isinstance(value, FuncCall):
        for a in value.args:
            _count_in_value(a, counts)
    elif isinstance(value, BinOp):
        _count_in_value(value.left, counts)
        _count_in_value(value.right, counts)
    elif isinstance(value, Ternary):
        _count_in_value(value.cond, counts)
        _count_in_value(value.then, counts)
        _count_in_value(value.else_, counts)
    elif isinstance(value, Range):
        _count_in_value(value.start, counts)
        _count_in_value(value.end, counts)
    elif isinstance(value, ListLit):
        for x in value.items:
            _count_in_value(x, counts)
    elif isinstance(value, DictLit):
        for _, v in value.entries:
            _count_in_value(v, counts)


# ---------- calls ----------


def _fmt_call(call: Call) -> str:
    """Emit a call line in the most compact form that preserves semantics."""
    parts = [call.verb]
    primary = _primary_arg(call.verb)
    remaining = list(call.args)

    # If the first arg corresponds to the verb's primary_arg (either by being
    # the synthetic <pos>/<pipe> or by literally being named the primary), emit
    # it positionally. Skip when the value is itself a BinOp/Ternary (their
    # written form might be ambiguous without `name=`).
    if remaining and primary:
        a0 = remaining[0]
        if a0.name in (primary, "<pos>", "<pipe>"):
            val_src = _fmt_value(a0.value)
            if not _needs_named_wrap(a0.value, val_src):
                parts.append(val_src)
                remaining = remaining[1:]

    for a in remaining:
        name = _resolve_arg_name(call.verb, a.name)
        parts.append(f"{name}={_fmt_value(a.value)}")

    line = " ".join(parts)
    if call.out:
        line += f" -> {call.out}"
    return line


def _needs_named_wrap(value, rendered_src: str) -> bool:
    """Some value shapes are clearer when always rendered with `name=`."""
    # If the rendered form contains spaces it's still parseable (we accept
    # quoted strings, parens, etc.), so the check is primarily for safety.
    return False


def _primary_arg(verb: str) -> str:
    try:
        from .verbs import VERBS
    except ImportError:
        return ""
    spec = VERBS.get(verb)
    return spec.primary_arg if spec else ""


def _resolve_arg_name(verb: str, name: str) -> str:
    """Translate the synthetic <pos> / <pipe> back to the verb's primary_arg."""
    if name not in ("<pos>", "<pipe>"):
        return name
    p = _primary_arg(verb)
    return p or name


# ---------- values ----------


def _fmt_value(v) -> str:
    if isinstance(v, StringLit):
        return json.dumps(v.value, ensure_ascii=False)
    if isinstance(v, NumberLit):
        return str(int(v.value)) if v.value.is_integer() else str(v.value)
    if isinstance(v, BoolLit):
        return "true" if v.value else "false"
    if isinstance(v, Name):
        # Single-part identifier → bareword.
        # Multi-part with all parts as idents → join with dots.
        joined = ".".join(v.parts)
        if _BAREWORD_RE.match(joined):
            return joined
        return json.dumps(joined, ensure_ascii=False)
    if isinstance(v, FuncCall):
        args = ", ".join(_fmt_value(a) for a in v.args)
        return f"{v.name}({args})"
    if isinstance(v, BinOp):
        # Always parenthesize: a top-level BinOp in arg-value position would
        # otherwise be misparsed (the parser consumes only one primary value).
        l = _fmt_value(v.left)
        r = _fmt_value(v.right)
        return f"({l} {v.op} {r})"
    if isinstance(v, ListLit):
        return "[" + ", ".join(_fmt_value(x) for x in v.items) + "]"
    if isinstance(v, DictLit):
        parts = []
        for k, val in v.entries:
            key_repr = k if _IDENT_RE.match(k) else json.dumps(k, ensure_ascii=False)
            parts.append(f"{key_repr}: {_fmt_value(val)}")
        return "{" + ", ".join(parts) + "}"
    if isinstance(v, Ternary):
        return f"({_fmt_value(v.cond)} ? {_fmt_value(v.then)} : {_fmt_value(v.else_)})"
    if isinstance(v, Range):
        return f"{_fmt_value(v.start)}..{_fmt_value(v.end)}"
    raise ValueError(f"unknown value type: {type(v).__name__}")
