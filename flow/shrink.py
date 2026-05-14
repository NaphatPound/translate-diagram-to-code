"""
Flow shrink — deterministic AST rewrites that produce equivalent, shorter code.

Distinct from `flow lint`: lint *describes* possible rewrites; shrink *applies*
them. Run shrink on output from a verbose-leaning LLM to get the compact form
without another LLM call.

Rules:
  1. Math verb → assignment
       add a=X b=Y -> Z      becomes      Z = X + Y
       (sub/mul/div similarly)
  2. Aggregator → assignment with funccall
       count of=X -> Y       becomes      Y = count(X)
       (sum/min/max/avg similarly)
  3. Mirror if/else assignment → ternary
       if cond                              X = cond ? a : b
         X = a
       else
         X = b

Pipe-chain detection (adjacent `verb -> tmp` then `verb name=tmp`) is handled
at format time by `flow.formatter`, not here, so shrink can stay pure-AST.

Public API:
  shrink(program)   → modifies the Program in place AND returns it.
  shrink_source(s)  → parse, shrink, format. Returns the new source string.
"""
from __future__ import annotations

from typing import List, Optional

from . import parse
from .parser import (
    Program, Call, AssignStmt, IfStmt, EachStmt, RepeatStmt, WhenStmt,
    StringLit, NumberLit, BoolLit, Name, FuncCall, BinOp, ListLit, DictLit, Ternary, Arg,
)
from .formatter import format_source


MATH_TO_OP = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
AGGS = {"count", "sum", "min", "max", "avg"}


def shrink_source(src: str) -> str:
    return format_source(shrink(parse(src)))


def shrink(program: Program) -> Program:
    program.body = _shrink_block(program.body)
    return program


def _shrink_block(body):
    new: List = []
    i = 0
    while i < len(body):
        stmt = body[i]

        # ---- recurse into children first ----
        if isinstance(stmt, IfStmt):
            stmt.then = _shrink_block(stmt.then)
            if stmt.else_:
                stmt.else_ = _shrink_block(stmt.else_)
            # Mirror-if → ternary rewrite (post-recursion).
            rewritten = _try_mirror_if_to_ternary(stmt)
            new.append(rewritten if rewritten else stmt)
            i += 1
            continue
        if isinstance(stmt, (EachStmt, RepeatStmt, WhenStmt)):
            stmt.body = _shrink_block(stmt.body)
            new.append(stmt)
            i += 1
            continue

        # ---- atomic statements ----
        if isinstance(stmt, Call):
            replacement = _try_call_to_assign(stmt)
            new.append(replacement if replacement else stmt)
            i += 1
            continue

        new.append(stmt)
        i += 1
    return new


# ---------- per-rule rewrites ----------


def _try_call_to_assign(call: Call) -> Optional[AssignStmt]:
    """Math + aggregator verbs that bind to `-> name` are nicer as assignments."""
    if not call.out:
        return None

    if call.verb in MATH_TO_OP:
        a = _arg(call, "a")
        b = _arg(call, "b")
        if a is not None and b is not None and len(call.args) == 2:
            return AssignStmt(
                target=call.out,
                value=BinOp(MATH_TO_OP[call.verb], a, b),
                line=call.line,
            )

    if call.verb in AGGS:
        of = _arg(call, "of")
        if of is not None and len(call.args) == 1:
            return AssignStmt(
                target=call.out,
                value=FuncCall(call.verb, [of]),
                line=call.line,
            )
    return None


def _try_mirror_if_to_ternary(stmt: IfStmt) -> Optional[AssignStmt]:
    """`if cond: X=a / else: X=b` → `X = cond ? a : b`."""
    if not stmt.else_:
        return None
    if len(stmt.then) != 1 or len(stmt.else_) != 1:
        return None
    a, b = stmt.then[0], stmt.else_[0]
    if not (isinstance(a, AssignStmt) and isinstance(b, AssignStmt)):
        return None
    if a.target != b.target:
        return None
    return AssignStmt(
        target=a.target,
        value=Ternary(cond=stmt.cond, then=a.value, else_=b.value),
        line=stmt.line,
    )


# ---------- helpers ----------


def _arg(call: Call, name: str):
    for a in call.args:
        if a.name == name:
            return a.value
    return None
