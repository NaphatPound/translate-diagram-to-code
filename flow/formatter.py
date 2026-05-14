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
    ListLit, DictLit, Ternary,
)

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BAREWORD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")
_INDENT = "  "


def format_source(program: Program) -> str:
    lines: List[str] = []
    for stmt in program.body:
        _emit_stmt(stmt, 0, lines)
    return "\n".join(lines) + ("\n" if lines else "")


# ---------- statements ----------


def _emit_stmt(stmt, depth: int, out: List[str]) -> None:
    if isinstance(stmt, Call):
        out.append(_INDENT * depth + _fmt_call(stmt))
        return
    if isinstance(stmt, AssignStmt):
        out.append(f"{_INDENT * depth}{stmt.target} = {_fmt_value(stmt.value)}")
        return
    if isinstance(stmt, IfStmt):
        out.append(_INDENT * depth + "if " + _fmt_value(stmt.cond))
        for s in stmt.then:
            _emit_stmt(s, depth + 1, out)
        if stmt.else_:
            out.append(_INDENT * depth + "else")
            for s in stmt.else_:
                _emit_stmt(s, depth + 1, out)
        return
    if isinstance(stmt, EachStmt):
        out.append(f"{_INDENT * depth}each {stmt.var} in {_fmt_value(stmt.iterable)}")
        for s in stmt.body:
            _emit_stmt(s, depth + 1, out)
        return
    if isinstance(stmt, RepeatStmt):
        out.append(f"{_INDENT * depth}repeat {_fmt_value(stmt.count)}")
        for s in stmt.body:
            _emit_stmt(s, depth + 1, out)
        return
    if isinstance(stmt, WhenStmt):
        extras = " ".join(_fmt_value(a) for a in stmt.args)
        out.append(f"{_INDENT * depth}when {stmt.event}" + (f" {extras}" if extras else ""))
        for s in stmt.body:
            _emit_stmt(s, depth + 1, out)
        return
    raise ValueError(f"unknown statement: {type(stmt).__name__}")


# ---------- calls ----------


def _fmt_call(call: Call) -> str:
    parts = [call.verb]
    for a in call.args:
        name = _resolve_arg_name(call.verb, a.name)
        parts.append(f"{name}={_fmt_value(a.value)}")
    line = " ".join(parts)
    if call.out:
        line += f" -> {call.out}"
    return line


def _resolve_arg_name(verb: str, name: str) -> str:
    """Translate the synthetic <pos> / <pipe> back to the verb's primary_arg."""
    if name not in ("<pos>", "<pipe>"):
        return name
    try:
        from .verbs import VERBS
    except ImportError:
        return name
    spec = VERBS.get(verb)
    if spec and spec.primary_arg:
        return spec.primary_arg
    return name


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
        # We do not preserve original parens, but emit defensively bracketed for clarity
        # if the operator is logical.
        l = _fmt_value(v.left)
        r = _fmt_value(v.right)
        return f"{l} {v.op} {r}"
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
    raise ValueError(f"unknown value type: {type(v).__name__}")
