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
    Program, Call, AssignStmt, IfStmt, EachStmt, RepeatStmt, WhenStmt,
    DefStmt, ReturnStmt,
    StringLit, NumberLit, BoolLit, Name, FuncCall, BinOp, UnaryOp, ListLit, DictLit,
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
    for stmt in program.body:
        _walk(stmt, warnings)
    return warnings


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
    """`if !cond` (or `if not cond`) without an else → `unless cond`."""
    cond = stmt.cond
    if (isinstance(cond, UnaryOp) and cond.op == "not"
            and stmt.else_ is None and len(stmt.then) == 1):
        # Single-stmt then-block w/ negated cond → suggest postfix unless.
        inner = _value_to_src(cond.value)
        # Find what the single stmt looks like by re-formatting it. For
        # simplicity we just describe the rewrite — the user can apply.
        out.append(LintWarning(
            stmt.line,
            f"`if !{inner}` with a single-stmt body can use postfix `unless`",
            f"<body> unless {inner}",
        ))


# ---------- per-call checks ----------


def _check_call(call: Call, out: List[LintWarning]) -> None:
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

    warnings = lint_program(program)
    if not warnings:
        print("no lint warnings")
        return
    for w in warnings:
        print(w)
    if args.fail:
        sys.exit(1)
