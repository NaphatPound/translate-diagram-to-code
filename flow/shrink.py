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
    Program, Call, AssignStmt, IfStmt, EachStmt, RepeatStmt, WhenStmt, TryStmt,
    StringLit, NumberLit, BoolLit, Name, FuncCall, BinOp, UnaryOp, ListLit, DictLit,
    Ternary, Range, FString, MethodCall, IndexAccess, Arg,
)
from .formatter import format_source


MATH_TO_OP = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
AGGS = {"count", "sum", "min", "max", "avg"}

# FuncCalls considered safe to inline (no side effects, deterministic).
PURE_BUILTINS = {"count", "sum", "min", "max", "abs", "len", "round",
                 "int", "float", "str"}


def shrink_source(src: str) -> str:
    return format_source(shrink(parse(src)))


def shrink(program: Program) -> Program:
    program.body = _shrink_block(program.body)
    program = _inline_single_use(program)
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
        if isinstance(stmt, TryStmt):
            stmt.try_body = _shrink_block(stmt.try_body)
            stmt.catch_body = _shrink_block(stmt.catch_body)
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


# ============================================================
# Inline single-use assignments
# ============================================================


def _inline_single_use(program: Program) -> Program:
    """`name = expr` whose `name` is used exactly once → drop the assignment
    and substitute `expr` at the use site.

    Only inlines when `expr` is a *safe* value: literals, names, simple
    arithmetic, ternaries, or pure-builtin function calls. Skips RHS that
    might have side effects (most FuncCalls).
    """
    counts: dict = {}
    _count_in_body(program.body, counts)

    # Collect single-use assignments with safe RHS.
    inlines: dict = {}
    for stmt in _walk_all(program.body):
        if isinstance(stmt, AssignStmt) and counts.get(stmt.target, 0) == 1:
            if _is_safe_to_inline(stmt.value):
                inlines[stmt.target] = stmt.value

    if not inlines:
        return program

    program.body = _replace_and_drop(program.body, inlines)
    return program


def _walk_all(body):
    """Yield every statement in body and its nested children."""
    for s in body:
        yield s
        if isinstance(s, IfStmt):
            yield from _walk_all(s.then)
            if s.else_:
                yield from _walk_all(s.else_)
        elif isinstance(s, (EachStmt, RepeatStmt, WhenStmt)):
            yield from _walk_all(s.body)
        elif isinstance(s, TryStmt):
            yield from _walk_all(s.try_body)
            yield from _walk_all(s.catch_body)


def _is_safe_to_inline(value) -> bool:
    if isinstance(value, (StringLit, NumberLit, BoolLit, Name, FString)):
        return True
    if isinstance(value, MethodCall):
        # Method calls may have side effects (e.g., `.read()`, `.pop()`).
        # Conservatively, don't inline.
        return False
    if isinstance(value, BinOp):
        return _is_safe_to_inline(value.left) and _is_safe_to_inline(value.right)
    if isinstance(value, Ternary):
        return (_is_safe_to_inline(value.cond)
                and _is_safe_to_inline(value.then)
                and _is_safe_to_inline(value.else_))
    if isinstance(value, Range):
        return _is_safe_to_inline(value.start) and _is_safe_to_inline(value.end)
    if isinstance(value, ListLit):
        return all(_is_safe_to_inline(x) for x in value.items)
    if isinstance(value, DictLit):
        return all(_is_safe_to_inline(v) for _, v in value.entries)
    if isinstance(value, FuncCall):
        if value.name not in PURE_BUILTINS:
            return False
        return all(_is_safe_to_inline(a) for a in value.args)
    return False


def _count_in_body(body, counts) -> None:
    for s in body:
        if isinstance(s, Call):
            for a in s.args:
                _count_in_value(a.value, counts)
        elif isinstance(s, AssignStmt):
            _count_in_value(s.value, counts)
        elif isinstance(s, IfStmt):
            _count_in_value(s.cond, counts)
            _count_in_body(s.then, counts)
            if s.else_:
                _count_in_body(s.else_, counts)
        elif isinstance(s, EachStmt):
            _count_in_value(s.iterable, counts)
            _count_in_body(s.body, counts)
        elif isinstance(s, RepeatStmt):
            _count_in_value(s.count, counts)
            _count_in_body(s.body, counts)
        elif isinstance(s, WhenStmt):
            for a in s.args:
                _count_in_value(a, counts)
            _count_in_body(s.body, counts)
        elif isinstance(s, TryStmt):
            _count_in_body(s.try_body, counts)
            _count_in_body(s.catch_body, counts)


def _count_in_value(value, counts) -> None:
    if isinstance(value, Name):
        if value.parts:
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
    elif isinstance(value, FString):
        for kind, payload in value.parts:
            if kind == "var":
                counts[payload] = counts.get(payload, 0) + 1
    elif isinstance(value, MethodCall):
        _count_in_value(value.receiver, counts)
        if value.args is not None:
            for a in value.args:
                _count_in_value(a, counts)
    elif isinstance(value, IndexAccess):
        _count_in_value(value.receiver, counts)
        _count_in_value(value.index, counts)
    elif isinstance(value, UnaryOp):
        _count_in_value(value.value, counts)


def _replace_and_drop(body, inlines):
    """Walk body: replace Name refs to keys of `inlines` with their values, and
    drop the corresponding AssignStmt definitions.
    """
    new = []
    for stmt in body:
        if isinstance(stmt, AssignStmt) and stmt.target in inlines:
            continue
        if isinstance(stmt, Call):
            stmt.args = [Arg(a.name, _replace_value(a.value, inlines)) for a in stmt.args]
        elif isinstance(stmt, AssignStmt):
            stmt.value = _replace_value(stmt.value, inlines)
        elif isinstance(stmt, IfStmt):
            stmt.cond = _replace_value(stmt.cond, inlines)
            stmt.then = _replace_and_drop(stmt.then, inlines)
            if stmt.else_:
                stmt.else_ = _replace_and_drop(stmt.else_, inlines)
        elif isinstance(stmt, EachStmt):
            stmt.iterable = _replace_value(stmt.iterable, inlines)
            stmt.body = _replace_and_drop(stmt.body, inlines)
        elif isinstance(stmt, RepeatStmt):
            stmt.count = _replace_value(stmt.count, inlines)
            stmt.body = _replace_and_drop(stmt.body, inlines)
        elif isinstance(stmt, WhenStmt):
            stmt.args = [_replace_value(a, inlines) for a in stmt.args]
            stmt.body = _replace_and_drop(stmt.body, inlines)
        elif isinstance(stmt, TryStmt):
            stmt.try_body = _replace_and_drop(stmt.try_body, inlines)
            stmt.catch_body = _replace_and_drop(stmt.catch_body, inlines)
        new.append(stmt)
    return new


def _replace_value(value, inlines, _expanding=None):
    """Substitute inlinable names with their values, recursively, with cycle
    protection so we never loop on `t = t`-style self-references."""
    _expanding = _expanding or frozenset()
    if isinstance(value, Name) and len(value.parts) == 1:
        n = value.parts[0]
        if n in inlines and n not in _expanding:
            return _replace_value(inlines[n], inlines, _expanding | {n})
        return value
    if isinstance(value, FuncCall):
        return FuncCall(value.name, [_replace_value(a, inlines, _expanding) for a in value.args])
    if isinstance(value, BinOp):
        return BinOp(value.op, _replace_value(value.left, inlines, _expanding),
                     _replace_value(value.right, inlines, _expanding))
    if isinstance(value, Ternary):
        return Ternary(_replace_value(value.cond, inlines, _expanding),
                       _replace_value(value.then, inlines, _expanding),
                       _replace_value(value.else_, inlines, _expanding))
    if isinstance(value, Range):
        return Range(_replace_value(value.start, inlines, _expanding),
                     _replace_value(value.end, inlines, _expanding))
    if isinstance(value, ListLit):
        return ListLit([_replace_value(x, inlines, _expanding) for x in value.items])
    if isinstance(value, DictLit):
        return DictLit([(k, _replace_value(v, inlines, _expanding)) for k, v in value.entries])
    if isinstance(value, FString):
        # Only replace `var` parts whose RHS is another simple Name (a rename).
        # Replacing with a non-Name value would require splicing the f-string,
        # which complicates the rendering; skip those.
        new_parts = []
        for kind, payload in value.parts:
            if kind == "var" and payload in inlines and payload not in _expanding:
                inner = inlines[payload]
                if isinstance(inner, Name) and len(inner.parts) == 1:
                    new_parts.append(("var", inner.parts[0]))
                else:
                    new_parts.append((kind, payload))
            else:
                new_parts.append((kind, payload))
        return FString(parts=new_parts)
    if isinstance(value, MethodCall):
        new_args = (None if value.args is None
                    else [_replace_value(a, inlines, _expanding) for a in value.args])
        return MethodCall(
            receiver=_replace_value(value.receiver, inlines, _expanding),
            method=value.method,
            args=new_args,
        )
    if isinstance(value, IndexAccess):
        return IndexAccess(
            receiver=_replace_value(value.receiver, inlines, _expanding),
            index=_replace_value(value.index, inlines, _expanding),
        )
    if isinstance(value, UnaryOp):
        return UnaryOp(value.op, _replace_value(value.value, inlines, _expanding))
    return value
