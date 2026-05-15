"""
Flow lint — find verbose constructs that have a shorter equivalent.

The lint is opinion-only: it does not change code. It walks the AST and
emits suggestions an LLM (or human) can apply.

Patterns detected:
  - add/sub/mul/div a=X b=Y -> Z          → Z = X (op) Y
  - count/sum/min/max/avg of=X -> Z       → Z = verb(X)
  - print value=X                         → print X    (positional primary)
  - upper/lower/trim text=X -> Y          → upper X -> Y  (positional primary)
  - filter/map/sort from=X ... -> Y       → if used immediately, suggest pipe

CLI:
  python -m flow lint file.flow
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from . import parse, ParseError
from .parser import (
    Program, Call, AssignStmt, MultiAssignStmt, IfStmt, EachStmt, RepeatStmt,
    WhenStmt, DefStmt, ReturnStmt, BreakStmt, ContinueStmt, TryStmt, WhileStmt, ExprStmt,
    StringLit, NumberLit, BoolLit, Name, FuncCall, BinOp, UnaryOp, ListLit, DictLit,
    Ternary, Range, FString, MethodCall, IndexAccess, Spread,
)
from .verbs import VERBS


MATH_TO_OP = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
AGGS = {"count", "sum", "min", "max", "avg"}


@dataclass
class LintWarning:
    line: int
    message: str
    suggestion: str

    def __str__(self) -> str:
        return f"line {self.line}: {self.message}\n  → {self.suggestion}"


# ---------- API ----------


def lint_source(src: str) -> List[LintWarning]:
    try:
        program = parse(src)
    except ParseError:
        # Lint is a no-op on un-parseable code — let the parser/compiler complain.
        return []
    return lint_program(program)


def lint_program(program: Program) -> List[LintWarning]:
    warnings: List[LintWarning] = []
    _check_dead_code(program.body, warnings)
    _check_unused_vars(program.body, warnings)
    for stmt in program.body:
        _walk(stmt, warnings)
    return warnings


def _check_unused_vars(body: list, out: List[LintWarning]) -> None:
    """Warn on AssignStmt targets that are never referenced. Names starting
    with `_` are exempt. Inside a DefStmt, the function name is exempt
    (recursion + external call). Inside a loop, the loop var is exempt."""
    # Build a global usage map from the entire program.
    usage = _count_all_names(body)

    def walk(stmts):
        for s in stmts:
            if isinstance(s, AssignStmt):
                if (not s.target.startswith("_")
                        and usage.get(s.target, 0) == 0):
                    out.append(LintWarning(
                        s.line,
                        f"variable {s.target!r} is assigned but never used",
                        f"remove the assignment, or rename to `_{s.target}` to silence",
                    ))
            # Recurse into nested blocks where assignments can also appear.
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
                _check_unused_params(s, out)
                walk(s.body)
    walk(body)


def _check_unused_params(stmt: DefStmt, out: List[LintWarning]) -> None:
    """Warn on def params that are never referenced inside the body."""
    body_usage = _count_all_names(stmt.body)
    for pname, _default in stmt.params:
        if pname.startswith("_"):
            continue
        if body_usage.get(pname, 0) == 0:
            out.append(LintWarning(
                stmt.line,
                f"param {pname!r} of def {stmt.name!r} is never used",
                f"remove from signature, or rename to `_{pname}` to silence",
            ))


def _count_all_names(body) -> dict:
    """Count Name references across body (and nested blocks). Doesn't count
    AssignStmt targets (the LHS of `name = expr`)."""
    counts: dict = {}
    def in_value(v):
        if isinstance(v, Name) and v.parts:
            counts[v.parts[0]] = counts.get(v.parts[0], 0) + 1
        elif isinstance(v, BinOp):
            in_value(v.left); in_value(v.right)
        elif isinstance(v, UnaryOp):
            in_value(v.value)
        elif isinstance(v, FuncCall):
            for a in v.args:
                in_value(a)
        elif isinstance(v, ListLit):
            for x in v.items:
                in_value(x)
        elif isinstance(v, DictLit):
            for _k, vv in v.entries:
                in_value(vv)
        elif isinstance(v, Ternary):
            in_value(v.cond); in_value(v.then); in_value(v.else_)
        elif isinstance(v, Range):
            in_value(v.start); in_value(v.end)
        elif isinstance(v, FString):
            for kind, payload in v.parts:
                if kind == "expr":
                    in_value(payload)
        elif isinstance(v, MethodCall):
            in_value(v.receiver)
            if v.args is not None:
                for a in v.args:
                    in_value(a)
        elif isinstance(v, IndexAccess):
            in_value(v.receiver); in_value(v.index)
        elif isinstance(v, Spread):
            in_value(v.value)
    def walk(stmts):
        for s in stmts:
            if isinstance(s, Call):
                for a in s.args:
                    in_value(a.value)
            elif isinstance(s, AssignStmt):
                in_value(s.value)
            elif isinstance(s, MultiAssignStmt):
                in_value(s.value)
            elif isinstance(s, IfStmt):
                in_value(s.cond)
                walk(s.then)
                if s.else_:
                    walk(s.else_)
            elif isinstance(s, EachStmt):
                in_value(s.iterable)
                walk(s.body)
            elif isinstance(s, RepeatStmt):
                in_value(s.count)
                walk(s.body)
            elif isinstance(s, WhileStmt):
                in_value(s.cond)
                walk(s.body)
            elif isinstance(s, WhenStmt):
                for a in s.args:
                    in_value(a)
                walk(s.body)
            elif isinstance(s, TryStmt):
                walk(s.try_body)
                walk(s.catch_body)
            elif isinstance(s, DefStmt):
                # The function name itself counts as used (it's a definition,
                # not a reference, but external callers may use it).
                counts[s.name] = counts.get(s.name, 0) + 1
                # Defaults can reference outer scope.
                for _n, d in s.params:
                    if d is not None:
                        in_value(d)
                walk(s.body)
            elif isinstance(s, ReturnStmt):
                if s.value is not None:
                    in_value(s.value)
            elif isinstance(s, ExprStmt):
                in_value(s.value)
    walk(body)
    return counts


def _check_dead_code(body: list, out: List[LintWarning]) -> None:
    """Warn when any statement appears after a return / break / continue
    in the same block — unreachable code."""
    terminators = (ReturnStmt, BreakStmt, ContinueStmt)
    for i, stmt in enumerate(body):
        if isinstance(stmt, terminators) and i < len(body) - 1:
            nxt = body[i + 1]
            out.append(LintWarning(
                getattr(nxt, "line", 0),
                f"statement is unreachable (preceded by {type(stmt).__name__.replace('Stmt','').lower()})",
                f"remove or move before the {type(stmt).__name__.replace('Stmt','').lower()}",
            ))
            return  # one warning per block is enough
        # Recurse into nested blocks.
        if isinstance(stmt, IfStmt):
            _check_dead_code(stmt.then, out)
            if stmt.else_:
                _check_dead_code(stmt.else_, out)
        elif isinstance(stmt, (EachStmt, RepeatStmt, WhileStmt, WhenStmt, DefStmt)):
            _check_dead_code(stmt.body, out)
        elif isinstance(stmt, TryStmt):
            _check_dead_code(stmt.try_body, out)
            _check_dead_code(stmt.catch_body, out)


# ---------- traversal ----------


def _walk(stmt, out: List[LintWarning]) -> None:
    if isinstance(stmt, Call):
        _check_call(stmt, out)
    elif isinstance(stmt, AssignStmt):
        pass  # already in the compact form
    elif isinstance(stmt, IfStmt):
        _check_if(stmt, out)
        for s in stmt.then:
            _walk(s, out)
        if stmt.else_:
            for s in stmt.else_:
                _walk(s, out)
    elif isinstance(stmt, (EachStmt, RepeatStmt, WhenStmt)):
        for s in stmt.body:
            _walk(s, out)
    elif isinstance(stmt, DefStmt):
        _check_def(stmt, out)
        for s in stmt.body:
            _walk(s, out)


def _check_def(stmt: DefStmt, out: List[LintWarning]) -> None:
    """Suggest implicit return for `def`s whose last stmt is `return VALUE`."""
    if not stmt.body:
        return
    last = stmt.body[-1]
    if isinstance(last, ReturnStmt) and last.value is not None:
        rhs = _value_to_src(last.value)
        out.append(LintWarning(
            last.line,
            f"`return` at end of def {stmt.name!r} can be implicit",
            rhs,
        ))


def _check_if(stmt: IfStmt, out: List[LintWarning]) -> None:
    """`if !cond` with a MULTI-statement body → suggest `unless cond` block form.
    Single-stmt bodies are indistinguishable from already-postfix-unless source,
    so we don't suggest those (would produce false positives)."""
    cond = stmt.cond
    if (isinstance(cond, UnaryOp) and cond.op == "not"
            and stmt.else_ is None and len(stmt.then) >= 2):
        inner = _value_to_src(cond.value)
        out.append(LintWarning(
            stmt.line,
            f"`if !{inner}` block can use `unless {inner}`",
            f"unless {inner}",
        ))


# ---------- per-call checks ----------


def _check_call(call: Call, out: List[LintWarning]) -> None:
    # Duplicate arg names. The compiler errors on this; lint surfaces it
    # earlier with a clearer hint.
    seen: dict = {}
    for a in call.args:
        if a.name in ("<pos>", "<pipe>"):
            continue
        seen[a.name] = seen.get(a.name, 0) + 1
    for name, n in seen.items():
        if n > 1:
            out.append(LintWarning(
                call.line,
                f"verb {call.verb!r} has duplicate arg {name!r} ({n}×)",
                f"keep only one `{name}=...` and drop the rest",
            ))

    suggested = False

    # 1. Math verbs with -> name → assignment.
    if call.verb in MATH_TO_OP and call.out:
        a = _arg_src(call, "a")
        b = _arg_src(call, "b")
        if a and b:
            op = MATH_TO_OP[call.verb]
            out.append(LintWarning(
                call.line,
                f"`{call.verb}` with `-> {call.out}` is verbose for arithmetic",
                f"{call.out} = {a} {op} {b}",
            ))
            suggested = True

    # 2. Aggregator verbs with -> name → assignment with funccall.
    if not suggested and call.verb in AGGS and call.out:
        of = _arg_src(call, "of")
        if of:
            out.append(LintWarning(
                call.line,
                f"`{call.verb} of=... -> {call.out}` can be an assignment",
                f"{call.out} = {call.verb}({of})",
            ))
            suggested = True

    # 3. Named primary arg when positional would do (only if we didn't already
    #    suggest a stronger rewrite).
    if not suggested:
        spec = VERBS.get(call.verb)
        if spec and spec.primary_arg:
            primary = spec.primary_arg
            named_primary = next(
                (a for a in call.args if a.name == primary), None
            )
            if named_primary and len(call.args) == 1:
                val_src = _value_to_src(named_primary.value)
                arrow = f" -> {call.out}" if call.out else ""
                out.append(LintWarning(
                    call.line,
                    f"`{call.verb} {primary}=...` can use the positional form",
                    f"{call.verb} {val_src}{arrow}",
                ))


# ---------- helpers ----------


def _arg_src(call: Call, name: str) -> Optional[str]:
    for a in call.args:
        if a.name == name:
            return _value_to_src(a.value)
    return None


def _value_to_src(v) -> str:
    """Render a value back to Flow source — best effort, lint-display only.

    Delegates to the formatter to avoid re-implementing every value shape.
    """
    try:
        from .formatter import _fmt_value
        return _fmt_value(v)
    except Exception:
        return "<?>"


# ---------- CLI ----------


def cli_main(args) -> None:
    import sys
    from pathlib import Path

    src = Path(args.file).read_text(encoding="utf-8")
    try:
        program = parse(src)
    except ParseError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    # --fix: apply automated rewrites via shrink. Idiomatic for codemod tools.
    if getattr(args, "fix", False):
        from .shrink import shrink_source
        out = shrink_source(src)
        if getattr(args, "write", False):
            Path(args.file).write_text(out, encoding="utf-8")
            print(f"wrote {args.file}", file=sys.stderr)
        else:
            sys.stdout.write(out)
        return

    warnings = lint_program(program)
    if not warnings:
        print("no lint warnings")
        return
    for w in warnings:
        print(w)
    if args.fail:
        sys.exit(1)
