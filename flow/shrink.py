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
    Program, Call, AssignStmt, MultiAssignStmt, IndexAssignStmt, IfStmt, EachStmt, RepeatStmt, WhileStmt, WhenStmt, TryStmt,
    DefStmt, ReturnStmt, ExprStmt, MatchStmt,
    StringLit, NumberLit, BoolLit, Name, FuncCall, BinOp, UnaryOp, ListLit, DictLit,
    Ternary, Range, Slice, ListComp, DictComp, FString, MethodCall, IndexAccess, Spread, Arg,
)
from .formatter import format_source


MATH_TO_OP = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
# Single-arg verbs that can be rewritten as `out = verb(input)`.
# Keyed by the input arg name in the verb signature.
OF_FUNCS = {"count", "sum", "min", "max", "avg", "keys", "values"}
FROM_FUNCS = {"reverse", "unique", "first", "last", "flatten"}
AGGS = OF_FUNCS  # back-compat alias for callers

# Multi-arg verbs that can be rewritten as funccalls. Value lists the arg
# names in the order the target funccall expects them.
MULTI_FUNCS = {
    "replace":  ["text", "find", "to"],
    "split":    ["text", "sep"],
    "join":     ["from", "sep"],
    "contains": ["text", "find"],
    "zip":      ["a", "b"],
    "format":   ["template", "data"],
}

# Inverted comparison operators for negation flips (`!(a == b)` → `a != b`).
INVERTED_CMP = {
    "==": "!=",
    "!=": "==",
    "<":  ">=",
    "<=": ">",
    ">":  "<=",
    ">=": "<",
    "in":     "not in",
    "not in": "in",
}

# FuncCalls considered safe to inline (no side effects, deterministic).
PURE_BUILTINS = {"count", "sum", "min", "max", "abs", "len", "round",
                 "int", "float", "str",
                 "reverse", "unique", "keys", "values", "avg", "sorted",
                 "first", "last", "flatten",
                 "replace", "split", "join", "contains", "zip", "format",
                 "upper", "lower", "trim"}


def shrink_source(src: str) -> str:
    return format_source(shrink(parse(src)))


def shrink(program: Program) -> Program:
    program.body = _simplify_in_body(program.body)
    program.body = _shrink_block(program.body)
    _drop_unused_call_out(program)
    program = _inline_single_use(program)
    _coalesce_user_temps(program)
    return program


def _drop_unused_call_out(program: Program) -> None:
    """When a Call captures `-> name` but `name` is never referenced
    anywhere, drop the capture (the side-effect of the call remains).

    Skips names beginning with `_` (parser temps + user opt-out) and
    only acts when the verb has no `returns=False` requirement — we
    don't want to error by dropping an output the user explicitly
    needed elsewhere.
    """
    counts: dict = {}
    _count_in_body(program.body, counts)

    def walk(stmts):
        for s in stmts:
            if isinstance(s, Call) and s.out and not s.out.startswith("_"):
                if counts.get(s.out, 0) == 0:
                    s.out = None
            if isinstance(s, IfStmt):
                walk(s.then)
                if s.else_:
                    walk(s.else_)
            elif isinstance(s, (EachStmt, RepeatStmt, WhileStmt, WhenStmt)):
                walk(s.body)
            elif isinstance(s, TryStmt):
                walk(s.try_body)
                walk(s.catch_body)
            elif isinstance(s, DefStmt):
                walk(s.body)
            elif isinstance(s, MatchStmt):
                for _pat, b in s.cases:
                    walk(b)
                if s.else_body:
                    walk(s.else_body)
    walk(program.body)


# ============================================================
# Pipe-coalesce: rename single-use Call.out names to _pN so the
# formatter recognizes them as pipe chains.
# ============================================================

import re as _re
_PIPE_TEMP_RE = _re.compile(r"^_p\d+$")


def _coalesce_user_temps(program: Program) -> None:
    """Rewrite user-named temps that flow as a single arg into the next
    Call so the formatter coalesces them into pipe chains.

    Example AST in:
      Call(filter, [from=xs, where="x > 0"], out=ys)
      Call(map,    [from=ys, to="x * 2"],    out=zs)
      Call(print,  [<pos>=Name(zs)],         out=None)

    Each of ys/zs used exactly once → rename to _p1/_p2. Formatter
    then emits `xs | filter where="x > 0" | map to="x * 2" | p`.
    """
    counts: dict = {}
    _count_in_body(program.body, counts)

    # Start the fresh-name counter above any existing _pN to avoid collisions.
    existing = []
    for stmt in program.body:
        if isinstance(stmt, Call) and stmt.out and _PIPE_TEMP_RE.match(stmt.out):
            existing.append(int(stmt.out[2:]))
    next_id = max(existing, default=0) + 1
    renames: dict = {}

    for i in range(len(program.body) - 1):
        cur = program.body[i]
        nxt = program.body[i + 1]
        if not (isinstance(cur, Call) and cur.out and isinstance(nxt, Call)):
            continue
        if counts.get(cur.out, 0) != 1:
            continue
        if _PIPE_TEMP_RE.match(cur.out):
            continue
        matching = [a for a in nxt.args
                    if isinstance(a.value, Name)
                    and len(a.value.parts) == 1
                    and a.value.parts[0] == cur.out]
        if len(matching) != 1:
            continue
        renames[cur.out] = f"_p{next_id}"
        next_id += 1

    if not renames:
        return

    for stmt in program.body:
        if not isinstance(stmt, Call):
            continue
        if stmt.out and stmt.out in renames:
            stmt.out = renames[stmt.out]
        for a in stmt.args:
            if (isinstance(a.value, Name)
                    and len(a.value.parts) == 1
                    and a.value.parts[0] in renames):
                a.value = Name([renames[a.value.parts[0]]])


# ============================================================
# Value-level simplifier (negation flips, double-negation, not-bool)
# ============================================================


def _simplify_value(v):
    if isinstance(v, UnaryOp):
        inner = _simplify_value(v.value)
        if v.op == "not":
            if isinstance(inner, UnaryOp) and inner.op == "not":
                return inner.value
            if isinstance(inner, BoolLit):
                return BoolLit(not inner.value)
            if isinstance(inner, BinOp) and inner.op in INVERTED_CMP:
                return BinOp(INVERTED_CMP[inner.op], inner.left, inner.right)
        return UnaryOp(v.op, inner)
    if isinstance(v, BinOp):
        return BinOp(v.op, _simplify_value(v.left), _simplify_value(v.right))
    if isinstance(v, Ternary):
        cond = _simplify_value(v.cond)
        then = _simplify_value(v.then)
        else_ = _simplify_value(v.else_)
        # `cond ? true : false`  → `cond`
        if isinstance(then, BoolLit) and isinstance(else_, BoolLit):
            if then.value is True and else_.value is False:
                return cond
            if then.value is False and else_.value is True:
                return _simplify_value(UnaryOp("not", cond))
        return Ternary(cond, then, else_)
    if isinstance(v, FuncCall):
        return FuncCall(v.name, [_simplify_value(a) for a in v.args])
    if isinstance(v, ListLit):
        return ListLit([_simplify_value(x) for x in v.items])
    if isinstance(v, DictLit):
        return DictLit([(k, _simplify_value(vv)) for k, vv in v.entries])
    if isinstance(v, MethodCall):
        return MethodCall(
            receiver=_simplify_value(v.receiver),
            method=v.method,
            args=None if v.args is None else [_simplify_value(a) for a in v.args],
        )
    if isinstance(v, IndexAccess):
        return IndexAccess(receiver=_simplify_value(v.receiver),
                           index=_simplify_value(v.index))
    if isinstance(v, Spread):
        return Spread(_simplify_value(v.value))
    if isinstance(v, Range):
        return Range(_simplify_value(v.start), _simplify_value(v.end))
    if isinstance(v, Slice):
        return Slice(
            start=None if v.start is None else _simplify_value(v.start),
            end=None if v.end is None else _simplify_value(v.end),
            step=None if v.step is None else _simplify_value(v.step),
        )
    if isinstance(v, ListComp):
        return ListComp(
            expr=_simplify_value(v.expr),
            var=v.var,
            source=_simplify_value(v.source),
            cond=None if v.cond is None else _simplify_value(v.cond),
        )
    if isinstance(v, DictComp):
        return DictComp(
            key_expr=_simplify_value(v.key_expr),
            val_expr=_simplify_value(v.val_expr),
            var=v.var,
            source=_simplify_value(v.source),
            cond=None if v.cond is None else _simplify_value(v.cond),
        )
    if isinstance(v, FString):
        new_parts = []
        for part in v.parts:
            fmt = part[2] if len(part) > 2 else ""
            if part[0] == "expr":
                new_parts.append(("expr", _simplify_value(part[1]), fmt))
            else:
                new_parts.append((part[0], part[1], fmt))
        return FString(parts=new_parts)
    return v


def _simplify_in_body(body):
    """Apply _simplify_value to every value-typed field in body recursively."""
    for s in body:
        if isinstance(s, Call):
            s.args = [Arg(a.name, _simplify_value(a.value)) for a in s.args]
        elif isinstance(s, AssignStmt):
            s.value = _simplify_value(s.value)
        elif isinstance(s, MultiAssignStmt):
            s.value = _simplify_value(s.value)
        elif isinstance(s, IndexAssignStmt):
            s.target = _simplify_value(s.target)
            s.value = _simplify_value(s.value)
        elif isinstance(s, IfStmt):
            s.cond = _simplify_value(s.cond)
            _simplify_in_body(s.then)
            if s.else_:
                _simplify_in_body(s.else_)
        elif isinstance(s, EachStmt):
            s.iterable = _simplify_value(s.iterable)
            _simplify_in_body(s.body)
        elif isinstance(s, RepeatStmt):
            s.count = _simplify_value(s.count)
            _simplify_in_body(s.body)
        elif isinstance(s, WhileStmt):
            s.cond = _simplify_value(s.cond)
            _simplify_in_body(s.body)
        elif isinstance(s, WhenStmt):
            s.args = [_simplify_value(a) for a in s.args]
            _simplify_in_body(s.body)
        elif isinstance(s, TryStmt):
            _simplify_in_body(s.try_body)
            _simplify_in_body(s.catch_body)
        elif isinstance(s, DefStmt):
            _simplify_in_body(s.body)
        elif isinstance(s, ReturnStmt):
            if s.value is not None:
                s.value = _simplify_value(s.value)
        elif isinstance(s, ExprStmt):
            s.value = _simplify_value(s.value)
        elif isinstance(s, MatchStmt):
            s.value = _simplify_value(s.value)
            new_cases = []
            for pat, case_body in s.cases:
                _simplify_in_body(case_body)
                new_cases.append((pat, case_body))
            s.cases = new_cases
            if s.else_body is not None:
                _simplify_in_body(s.else_body)
    return body


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
        if isinstance(stmt, (EachStmt, RepeatStmt, WhileStmt, WhenStmt)):
            stmt.body = _shrink_block(stmt.body)
            new.append(stmt)
            i += 1
            continue
        if isinstance(stmt, MatchStmt):
            stmt.cases = [(p, _shrink_block(b)) for p, b in stmt.cases]
            if stmt.else_body is not None:
                stmt.else_body = _shrink_block(stmt.else_body)
            new.append(stmt)
            i += 1
            continue
        if isinstance(stmt, TryStmt):
            stmt.try_body = _shrink_block(stmt.try_body)
            stmt.catch_body = _shrink_block(stmt.catch_body)
            new.append(stmt)
            i += 1
            continue
        if isinstance(stmt, DefStmt):
            stmt.body = _shrink_block(stmt.body)
            # Implicit return: drop `return` from the def's last stmt so the
            # formatter emits the bare expression. Parser re-wraps on round-trip.
            if (stmt.body
                    and isinstance(stmt.body[-1], ReturnStmt)
                    and stmt.body[-1].value is not None):
                last = stmt.body[-1]
                stmt.body[-1] = ExprStmt(value=last.value, line=last.line)
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

    if call.verb in OF_FUNCS and len(call.args) == 1:
        norm = _normalize_args(call)
        of = norm.get("of")
        if of is not None:
            return AssignStmt(
                target=call.out,
                value=FuncCall(call.verb, [of]),
                line=call.line,
            )

    if call.verb in FROM_FUNCS and len(call.args) == 1:
        norm = _normalize_args(call)
        fr = norm.get("from")
        if fr is not None:
            return AssignStmt(
                target=call.out,
                value=FuncCall(call.verb, [fr]),
                line=call.line,
            )

    if call.verb in MULTI_FUNCS:
        norm = _normalize_args(call)
        arg_names = MULTI_FUNCS[call.verb]
        if len(norm) == len(arg_names) and all(n in norm for n in arg_names):
            ordered = [norm[n] for n in arg_names]
            return AssignStmt(
                target=call.out,
                value=FuncCall(call.verb, ordered),
                line=call.line,
            )
    return None


def _normalize_args(call) -> dict:
    """Build {name: value} for the call, mapping `<pos>` to the verb's
    primary_arg so positional and named forms compare equal."""
    from .verbs import VERBS
    spec = VERBS.get(call.verb)
    primary = spec.primary_arg if spec else None
    result = {}
    for a in call.args:
        name = a.name
        if name == "<pos>" and primary:
            name = primary
        result[name] = a.value
    return result


def _try_mirror_if_to_ternary(stmt: IfStmt):
    """Mirror-if collapse to a single ternary expression.

    - `if cond: X=a / else: X=b`            → `X = cond ? a : b`
    - `if cond: return a / else: return b`  → `return cond ? a : b`
    - `if cond: verb a / else: verb b`      → `verb (cond ? a : b)`
      (only when both calls have a single positional arg and no `-> out`)
    """
    if not stmt.else_:
        return None
    if len(stmt.then) != 1 or len(stmt.else_) != 1:
        return None
    a, b = stmt.then[0], stmt.else_[0]
    if isinstance(a, AssignStmt) and isinstance(b, AssignStmt) and a.target == b.target:
        return AssignStmt(
            target=a.target,
            value=Ternary(cond=stmt.cond, then=a.value, else_=b.value),
            line=stmt.line,
        )
    if (isinstance(a, ReturnStmt) and isinstance(b, ReturnStmt)
            and a.value is not None and b.value is not None):
        return ReturnStmt(
            value=Ternary(cond=stmt.cond, then=a.value, else_=b.value),
            line=stmt.line,
        )
    if (isinstance(a, Call) and isinstance(b, Call)
            and a.verb == b.verb
            and a.out is None and b.out is None
            and len(a.args) == 1 and len(b.args) == 1
            and a.args[0].name == "<pos>" and b.args[0].name == "<pos>"):
        return Call(
            verb=a.verb,
            args=[Arg("<pos>", Ternary(cond=stmt.cond,
                                       then=a.args[0].value,
                                       else_=b.args[0].value))],
            out=None,
            line=stmt.line,
        )
    return None


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

    Names that are shadowed by a def's param (e.g. outer `x` and `def f x`)
    are NOT inlined — the param-binding makes the inner reference semantically
    different from the outer one.
    """
    counts: dict = {}
    _count_in_body(program.body, counts)

    shadowed = _shadowed_names(program.body)

    # Collect single-use assignments with safe RHS.
    inlines: dict = {}
    for stmt in _walk_all(program.body):
        if isinstance(stmt, AssignStmt) and counts.get(stmt.target, 0) == 1:
            if stmt.target in shadowed:
                continue
            if _is_safe_to_inline(stmt.value):
                inlines[stmt.target] = stmt.value

    if not inlines:
        return program

    program.body = _replace_and_drop(program.body, inlines)
    return program


def _shadowed_names(body) -> set:
    """Return the set of names shadowed by a def's param or each-loop var
    somewhere in the program. Conservative: a name is shadowed even if
    only one def in the program has it as a param."""
    shadowed: set = set()
    def walk(stmts):
        for s in stmts:
            if isinstance(s, DefStmt):
                for n, _d in s.params:
                    shadowed.add(n)
                walk(s.body)
            elif isinstance(s, EachStmt):
                shadowed.add(s.var)
                if s.key_var:
                    shadowed.add(s.key_var)
                walk(s.body)
            elif isinstance(s, RepeatStmt):
                if s.var:
                    shadowed.add(s.var)
                walk(s.body)
            elif isinstance(s, IfStmt):
                walk(s.then)
                if s.else_:
                    walk(s.else_)
            elif isinstance(s, (WhileStmt, WhenStmt)):
                walk(s.body)
            elif isinstance(s, TryStmt):
                if s.catch_var:
                    shadowed.add(s.catch_var)
                walk(s.try_body)
                walk(s.catch_body)
    walk(body)
    return shadowed


def _walk_all(body):
    """Yield every statement in body and its nested children."""
    for s in body:
        yield s
        if isinstance(s, IfStmt):
            yield from _walk_all(s.then)
            if s.else_:
                yield from _walk_all(s.else_)
        elif isinstance(s, (EachStmt, RepeatStmt, WhileStmt, WhenStmt)):
            yield from _walk_all(s.body)
        elif isinstance(s, TryStmt):
            yield from _walk_all(s.try_body)
            yield from _walk_all(s.catch_body)
        elif isinstance(s, DefStmt):
            yield from _walk_all(s.body)


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
    if isinstance(value, Slice):
        return ((value.start is None or _is_safe_to_inline(value.start))
                and (value.end is None or _is_safe_to_inline(value.end))
                and (value.step is None or _is_safe_to_inline(value.step)))
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
        elif isinstance(s, MultiAssignStmt):
            _count_in_value(s.value, counts)
        elif isinstance(s, IndexAssignStmt):
            _count_in_value(s.target, counts)
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
        elif isinstance(s, WhileStmt):
            _count_in_value(s.cond, counts)
            _count_in_body(s.body, counts)
        elif isinstance(s, WhenStmt):
            for a in s.args:
                _count_in_value(a, counts)
            _count_in_body(s.body, counts)
        elif isinstance(s, TryStmt):
            _count_in_body(s.try_body, counts)
            _count_in_body(s.catch_body, counts)
        elif isinstance(s, DefStmt):
            _count_in_body(s.body, counts)
        elif isinstance(s, ReturnStmt):
            if s.value is not None:
                _count_in_value(s.value, counts)
        elif isinstance(s, ExprStmt):
            _count_in_value(s.value, counts)


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
    elif isinstance(value, Slice):
        if value.start is not None:
            _count_in_value(value.start, counts)
        if value.end is not None:
            _count_in_value(value.end, counts)
        if value.step is not None:
            _count_in_value(value.step, counts)
    elif isinstance(value, ListComp):
        _count_in_value(value.expr, counts)
        _count_in_value(value.source, counts)
        if value.cond is not None:
            _count_in_value(value.cond, counts)
    elif isinstance(value, DictComp):
        _count_in_value(value.key_expr, counts)
        _count_in_value(value.val_expr, counts)
        _count_in_value(value.source, counts)
        if value.cond is not None:
            _count_in_value(value.cond, counts)
    elif isinstance(value, ListLit):
        for x in value.items:
            _count_in_value(x, counts)
    elif isinstance(value, DictLit):
        for _, v in value.entries:
            _count_in_value(v, counts)
    elif isinstance(value, FString):
        for part in value.parts:
            if part[0] == "expr":
                _count_in_value(part[1], counts)
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
    elif isinstance(value, Spread):
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
        elif isinstance(stmt, MultiAssignStmt):
            stmt.value = _replace_value(stmt.value, inlines)
        elif isinstance(stmt, IndexAssignStmt):
            stmt.target = _replace_value(stmt.target, inlines)
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
        elif isinstance(stmt, WhileStmt):
            stmt.cond = _replace_value(stmt.cond, inlines)
            stmt.body = _replace_and_drop(stmt.body, inlines)
        elif isinstance(stmt, WhenStmt):
            stmt.args = [_replace_value(a, inlines) for a in stmt.args]
            stmt.body = _replace_and_drop(stmt.body, inlines)
        elif isinstance(stmt, TryStmt):
            stmt.try_body = _replace_and_drop(stmt.try_body, inlines)
            stmt.catch_body = _replace_and_drop(stmt.catch_body, inlines)
        elif isinstance(stmt, DefStmt):
            stmt.body = _replace_and_drop(stmt.body, inlines)
        elif isinstance(stmt, ReturnStmt):
            if stmt.value is not None:
                stmt.value = _replace_value(stmt.value, inlines)
        elif isinstance(stmt, ExprStmt):
            stmt.value = _replace_value(stmt.value, inlines)
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
    if isinstance(value, Slice):
        return Slice(
            start=(None if value.start is None
                   else _replace_value(value.start, inlines, _expanding)),
            end=(None if value.end is None
                 else _replace_value(value.end, inlines, _expanding)),
            step=(None if value.step is None
                  else _replace_value(value.step, inlines, _expanding)),
        )
    if isinstance(value, ListComp):
        # The bound var shadows any outer inline of the same name.
        inner = {k: v for k, v in inlines.items() if k != value.var}
        return ListComp(
            expr=_replace_value(value.expr, inner, _expanding),
            var=value.var,
            source=_replace_value(value.source, inlines, _expanding),
            cond=(None if value.cond is None
                  else _replace_value(value.cond, inner, _expanding)),
        )
    if isinstance(value, DictComp):
        inner = {k: v for k, v in inlines.items() if k != value.var}
        return DictComp(
            key_expr=_replace_value(value.key_expr, inner, _expanding),
            val_expr=_replace_value(value.val_expr, inner, _expanding),
            var=value.var,
            source=_replace_value(value.source, inlines, _expanding),
            cond=(None if value.cond is None
                  else _replace_value(value.cond, inner, _expanding)),
        )
    if isinstance(value, ListLit):
        return ListLit([_replace_value(x, inlines, _expanding) for x in value.items])
    if isinstance(value, DictLit):
        return DictLit([(k, _replace_value(v, inlines, _expanding)) for k, v in value.entries])
    if isinstance(value, FString):
        # `expr` parts are full Value AST nodes; recurse to substitute inlinables.
        new_parts = []
        for part in value.parts:
            fmt = part[2] if len(part) > 2 else ""
            if part[0] == "expr":
                new_parts.append(("expr",
                                  _replace_value(part[1], inlines, _expanding),
                                  fmt))
            else:
                new_parts.append((part[0], part[1], fmt))
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
    if isinstance(value, Spread):
        return Spread(_replace_value(value.value, inlines, _expanding))
    return value
