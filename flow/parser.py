"""
Flow parser — tokenizer + line-based recursive-descent parser.

Language summary:
    verb arg=value arg=value -> name        # action block
    if <expr>                                # control block (indent 2 spaces)
      <stmt>...
    else
      <stmt>...
    each <name> in <value>
      <stmt>...
    repeat <value>
      <stmt>...
    when <event...>
      <stmt>...
    # comment
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union, Tuple, Any


# ============================================================
# AST
# ============================================================

Value = Union["StringLit", "NumberLit", "BoolLit", "Name", "FuncCall", "BinOp", "UnaryOp", "ListLit", "DictLit", "Ternary", "Range", "FString", "MethodCall", "IndexAccess", "Spread"]
Stmt = Union["Call", "AssignStmt", "MultiAssignStmt", "IfStmt", "EachStmt", "RepeatStmt", "WhileStmt", "WhenStmt", "TryStmt", "BreakStmt", "ContinueStmt", "DefStmt", "ReturnStmt", "ExprStmt", "MatchStmt"]


@dataclass
class StringLit:
    value: str
    kind: str = "string"


@dataclass
class NumberLit:
    value: float
    kind: str = "number"


@dataclass
class BoolLit:
    value: bool
    kind: str = "bool"


@dataclass
class Name:
    """A dotted path. parts[0] may be a variable; rest is member access."""
    parts: List[str]
    kind: str = "name"


@dataclass
class FuncCall:
    """Reporter / inline call: count(x), max(a, b)."""
    name: str
    args: List[Value]
    kind: str = "funccall"


@dataclass
class BinOp:
    op: str
    left: Value
    right: Value
    kind: str = "binop"


@dataclass
class UnaryOp:
    """Prefix unary: `not x`, `!x`. `op` is the canonical name (e.g. 'not')."""
    op: str
    value: Value
    kind: str = "unary"


@dataclass
class Spread:
    """`*xs` inside a list literal or funccall args: expands the iterable
    into the surrounding collection / argument list."""
    value: Value
    kind: str = "spread"


@dataclass
class ListLit:
    items: List[Value]
    kind: str = "list"


@dataclass
class DictLit:
    entries: List[Tuple[str, Value]]   # (key, value) pairs
    kind: str = "dict"


@dataclass
class Ternary:
    cond: Value
    then: Value
    else_: Value
    kind: str = "ternary"


@dataclass
class Range:
    """Inclusive integer range literal: `start..end` ≡ list [start..end]."""
    start: Value
    end: Value


@dataclass
class ListComp:
    """Python-style list comprehension: `[expr for var in source]`
    or `[expr for var in source if cond]`.

    Lowered to a native comprehension/map per target language.
    """
    expr: Value
    var: str
    source: Value
    cond: Optional[Value] = None


@dataclass
class DictComp:
    """Python-style dict comprehension: `{k_expr: v_expr for var in source (if cond)}`.

    Lowered to a native dict comprehension in Python; `Object.fromEntries`
    + filter/map in JS; HashMap collect chains in Rust.
    """
    key_expr: Value
    val_expr: Value
    var: str
    source: Value
    cond: Optional[Value] = None


@dataclass
class Slice:
    """Python-style exclusive slice inside `[]`. Any bound may be None.

    `xs[a:b]`     → Slice(a, b)
    `xs[a:]`      → Slice(a, None)
    `xs[:b]`      → Slice(None, b)
    `xs[:]`       → Slice(None, None)
    `xs[::2]`     → Slice(None, None, 2)        every other
    `xs[::-1]`    → Slice(None, None, -1)        reverse
    `xs[1:5:2]`   → Slice(1, 5, 2)

    Distinct from Range so the compiler can emit the natural `xs[a:b]`
    in Python (no +1) and not confuse this with the inclusive `..` form.
    """
    start: Optional[Value]
    end: Optional[Value]
    step: Optional[Value] = None


@dataclass
class FString:
    """Interpolated string: `f"hi {name}, age {age}"`.

    Parts are tuples ("text" | "var", payload). "text" payload is a literal
    fragment; "var" payload is a Flow identifier referencing a variable in
    scope.
    """
    parts: List[Tuple[str, str]]
    kind: str = "fstring"


@dataclass
class MethodCall:
    """`receiver.method(args)` — produced by the postfix DOT chain in the
    parser. `args` is None for plain attribute access (`receiver.member`).
    `optional` is True when `?.` was used: emit a null-safe access."""
    receiver: Value
    method: str
    args: Optional[List[Value]]
    optional: bool = False
    kind: str = "methodcall"


@dataclass
class IndexAccess:
    """`receiver[index]` — postfix bracket access. Works on lists, dicts,
    strings, anywhere the target language supports `[]`."""
    receiver: Value
    index: Value
    kind: str = "index"


@dataclass
class Arg:
    name: str
    value: Value


@dataclass
class Call:
    verb: str
    args: List[Arg]
    out: Optional[str]
    line: int
    kind: str = "call"


@dataclass
class AssignStmt:
    target: str
    value: Value
    line: int
    kind: str = "assign"


@dataclass
class IndexAssignStmt:
    """`d["k"] = v` / `xs[0] = v` — assign to an indexed location."""
    target: "IndexAccess"
    value: Value
    line: int
    kind: str = "index_assign"


@dataclass
class MultiAssignStmt:
    """`a, b = expr` — unpacks the RHS into multiple targets in order."""
    targets: List[str]
    value: Value
    line: int
    kind: str = "multi_assign"


@dataclass
class IfStmt:
    cond: Value
    then: List[Stmt]
    else_: Optional[List[Stmt]]
    line: int
    kind: str = "if"


@dataclass
class EachStmt:
    var: str
    iterable: Value
    body: List[Stmt]
    line: int
    key_var: Optional[str] = None   # `each k, v in dict` → key_var=k, var=v
    kind: str = "each"


@dataclass
class RepeatStmt:
    count: Value
    body: List[Stmt]
    line: int
    var: Optional[str] = None   # `repeat N as i` binds i to 0..N-1 in body
    kind: str = "repeat"


@dataclass
class WhileStmt:
    cond: Value
    body: List[Stmt]
    line: int
    kind: str = "while"


@dataclass
class WhenStmt:
    event: str
    args: List[Value]
    body: List[Stmt]
    line: int
    kind: str = "when"


@dataclass
class TryStmt:
    """`try ... catch [name] ...`. catch_var is None when user wrote bare `catch`."""
    try_body: List[Stmt]
    catch_var: Optional[str]
    catch_body: List[Stmt]
    line: int
    kind: str = "try"


@dataclass
class BreakStmt:
    line: int
    kind: str = "break"


@dataclass
class ContinueStmt:
    line: int
    kind: str = "continue"


@dataclass
class DefStmt:
    """User-defined function: `def name p1 p2 p3=default ...` + indented body.
    Each entry in params is (name, default) where default is None for
    required-positional params."""
    name: str
    params: List[Tuple[str, Optional[Value]]]
    body: List[Stmt]
    line: int
    kind: str = "def"


@dataclass
class ReturnStmt:
    value: Optional[Value]
    line: int
    kind: str = "return"


@dataclass
class ExprStmt:
    """A statement that's just an expression evaluated for side effect.
    Produced when the line starts with `WORD (` — i.e., a funccall used
    where we'd otherwise expect a verb statement."""
    value: Value
    line: int
    kind: str = "expr"


@dataclass
class MatchStmt:
    """`match value` followed by one or more `case PATTERN` arms and an
    optional `else` arm. Patterns are literal expressions; matching is
    equality (==)."""
    value: Value
    cases: List[Tuple[Value, List[Stmt]]]
    else_body: Optional[List[Stmt]]
    line: int
    kind: str = "match"


@dataclass
class Program:
    body: List[Stmt]
    kind: str = "program"


# ============================================================
# Errors
# ============================================================


class ParseError(Exception):
    def __init__(self, msg: str, line: int = 0, col: int = 0):
        self.msg = msg
        self.line = line
        self.col = col
        loc = f"line {line}" + (f":{col}" if col else "")
        super().__init__(f"{loc}: {msg}")

    def with_source(self, source_lines: list) -> "ParseError":
        """Attach a source-line snippet with a `^` caret. Returns self."""
        if 0 < self.line <= len(source_lines):
            self.args = (_format_error_with_caret(
                self.msg, source_lines, self.line, self.col,
            ),)
        return self


def _format_error_with_caret(msg: str, source_lines: list, line: int, col: int) -> str:
    """Render `line N:C: msg\\n    <src>\\n        ^` for human / LLM eyes."""
    loc = f"line {line}" + (f":{col}" if col else "")
    if 0 < line <= len(source_lines):
        src = source_lines[line - 1]
        # Strip trailing whitespace but keep leading indent for accurate caret.
        src = src.rstrip()
        out = f"{loc}: {msg}\n    {src}"
        if col > 0:
            out += "\n    " + " " * (col - 1) + "^"
        return out
    return f"{loc}: {msg}"


# ============================================================
# Tokenizer
# ============================================================

KEYWORDS = {"if", "else", "unless", "each", "in", "repeat", "while", "when", "as",
            "try", "catch", "break", "continue", "def", "return",
            "match", "case",
            "and", "or", "not", "true", "false"}

# Single-letter aliases for the most common verbs. The parser substitutes
# the canonical name before constructing the Call node.
VERB_ALIASES = {
    "p": "print",
    "r": "read",
    "w": "write",
    "f": "filter",
    "m": "map",
    "c": "count",
    "u": "upper",
    "l": "lower",
    "s": "sort",
    "t": "trim",
}

# Order matters: longer / more-specific patterns first.
TOKEN_SPEC = [
    ("FSTRING", r'f"(?:\\.|[^"\\\n])*"'),
    ("STRING", r'"(?:\\.|[^"\\\n])*"'),
    ("ARROW", r"->"),
    ("CMP", r">=|<=|==|!="),
    ("COMPOUND", r"[+\-*/]="),
    ("DOTDOT", r"\.\."),
    ("OROP", r"\|\|"),
    ("ANDOP", r"&&"),
    ("NULLCOAL", r"\?\?"),
    ("OPTDOT", r"\?\."),
    ("PIPE", r"\|"),
    ("QMARK", r"\?"),
    ("BANG", r"!"),
    ("OP", r"[=<>+\-*/%]"),
    ("NUMBER", r"\d+(?:\.\d+)?"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("LBRACK", r"\["),
    ("RBRACK", r"\]"),
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("COLON", r":"),
    ("COMMA", r","),
    ("SEMI", r";"),
    ("DOT", r"\."),
    # Bareword: starts with letter/underscore. May include `-` for file/url
    # fragments. Dots are tokenised separately (DOT) so `data.csv`, `a.b.c`,
    # and `s.upper()` all flow through the postfix-DOT parser rule.
    ("WORD", r"[A-Za-z_][A-Za-z0-9_\-]*"),
    ("WS", r"[ \t]+"),
]

_TOKEN_RE = re.compile("|".join(f"(?P<{n}>{p})" for n, p in TOKEN_SPEC))


@dataclass
class Token:
    kind: str
    value: str
    line: int
    col: int


def _tokenize_line(text: str, line_no: int) -> List[Token]:
    pos = 0
    tokens: List[Token] = []
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m:
            raise ParseError(f"unexpected character {text[pos]!r}", line_no, pos + 1)
        kind = m.lastgroup or ""
        value = m.group()
        if kind == "WS":
            pos = m.end()
            continue
        if kind == "WORD" and value in KEYWORDS:
            kind = f"KW_{value.upper()}"
        if kind == "STRING":
            value = _unescape(value[1:-1])
        if kind == "FSTRING":
            # Strip leading `f` and surrounding quotes; keep raw content
            # so the parser can split `{name}` placeholders.
            value = _unescape(value[2:-1])
        tokens.append(Token(kind, value, line_no, pos + 1))
        pos = m.end()
    return tokens


def _unescape(s: str) -> str:
    return s.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\\\", "\\")


# ============================================================
# Line preprocessing
# ============================================================


@dataclass
class _Line:
    indent: int
    tokens: List[Token]
    line_no: int
    raw: str


def _split_lines(src: str) -> List[_Line]:
    """Return non-empty, non-comment lines with indent + tokens.

    Lines containing top-level `;` are split into separate _Line entries at
    the same indent so `a = 1; b = 2; p a + b` parses as three statements.
    """
    out: List[_Line] = []
    for i, raw in enumerate(src.splitlines(), start=1):
        code = _strip_comment(raw)
        if not code.strip():
            continue
        if "\t" in code[: len(code) - len(code.lstrip())]:
            raise ParseError("tabs not allowed for indentation; use 2 spaces", i, 1)
        indent_spaces = len(code) - len(code.lstrip(" "))
        if indent_spaces % 2 != 0:
            raise ParseError(f"indent must be a multiple of 2 (got {indent_spaces})", i, 1)
        indent = indent_spaces // 2
        tokens = _tokenize_line(code[indent_spaces:], i)
        if not tokens:
            continue
        # Split at top-level SEMI tokens.
        for seg in _split_at_top_level_semi(tokens):
            if seg:
                out.append(_Line(indent, seg, i, raw))
    return out


def _split_at_top_level_semi(tokens):
    """Yield token segments separated by `;` outside any (), [], {} group."""
    segment: list = []
    depth = 0
    for t in tokens:
        if t.kind in ("LPAREN", "LBRACK", "LBRACE"):
            depth += 1
        elif t.kind in ("RPAREN", "RBRACK", "RBRACE"):
            depth = max(0, depth - 1)
        if t.kind == "SEMI" and depth == 0:
            yield segment
            segment = []
            continue
        segment.append(t)
    yield segment


def _strip_comment(line: str) -> str:
    """Remove # comments but ignore # inside double-quoted strings."""
    out = []
    in_str = False
    i = 0
    while i < len(line):
        c = line[i]
        if c == '"' and (i == 0 or line[i - 1] != "\\"):
            in_str = not in_str
            out.append(c)
        elif c == "#" and not in_str:
            break
        else:
            out.append(c)
        i += 1
    return "".join(out)


# ============================================================
# Parser (recursive descent over lines)
# ============================================================


class _Parser:
    def __init__(self, lines: List[_Line]):
        self.lines = lines
        self.idx = 0
        self.pipe_counter = 0

    def parse_program(self) -> Program:
        body, _ = self._parse_block(0)
        if self.idx != len(self.lines):
            line = self.lines[self.idx]
            raise ParseError("unexpected indentation", line.line_no, 1)
        return Program(body=body)

    # ---------- block ----------

    def _parse_block(self, base_indent: int) -> Tuple[List[Stmt], int]:
        body: List[Stmt] = []
        while self.idx < len(self.lines):
            line = self.lines[self.idx]
            if line.indent < base_indent:
                break
            if line.indent > base_indent:
                raise ParseError(
                    f"unexpected indent (expected {base_indent * 2} spaces)",
                    line.line_no,
                    1,
                )
            stmt = self._parse_stmt(base_indent)
            if isinstance(stmt, list):   # pipeline → list of calls
                body.extend(stmt)
            else:
                body.append(stmt)
        return body, self.idx

    def _parse_stmt(self, base_indent: int) -> Stmt:
        line = self.lines[self.idx]
        first = line.tokens[0]

        if first.kind == "KW_IF":
            return self._parse_if(base_indent)
        if first.kind == "KW_EACH":
            return self._parse_each(base_indent)
        if first.kind == "KW_REPEAT":
            return self._parse_repeat(base_indent)
        if first.kind == "KW_WHILE":
            return self._parse_while(base_indent)
        if first.kind == "KW_WHEN":
            return self._parse_when(base_indent)
        if first.kind == "KW_TRY":
            return self._parse_try(base_indent)
        if first.kind == "KW_MATCH":
            return self._parse_match(base_indent)
        if first.kind == "KW_UNLESS":
            return self._parse_unless(base_indent)
        if first.kind == "KW_BREAK":
            return self._parse_break_continue(line, BreakStmt, "break")
        if first.kind == "KW_CONTINUE":
            return self._parse_break_continue(line, ContinueStmt, "continue")
        if first.kind == "KW_DEF":
            return self._parse_def(base_indent)
        if first.kind == "KW_RETURN":
            return self._parse_return(line)
        if first.kind == "KW_ELSE":
            raise ParseError("'else' without matching 'if'", line.line_no, 1)
        if first.kind == "WORD":
            toks = line.tokens
            # Multi-target assignment: `a, b = expr`. Check via lookahead.
            if (len(toks) >= 4
                    and toks[1].kind == "COMMA"
                    and _looks_like_multi_assign(toks)):
                return self._parse_multi_assign(line)
            # Assignment: `name = expr`. Check that toks[1] is `=` (not `==`).
            if (len(toks) >= 3
                    and toks[1].kind == "OP" and toks[1].value == "="):
                return self._parse_assignment(line)
            # Index assignment: `d["k"] = v` / `xs[0] = v` — `name [ ... ] =`.
            if _is_index_assign(toks):
                return self._parse_index_assignment(line)
            # Compound index assignment: `xs[0] += 5`, `d["k"] *= 2`.
            if _is_compound_index_assign(toks):
                return self._parse_compound_index_assignment(line)
            # Compound assignment: `name += expr` → `name = name + expr`.
            if (len(toks) >= 3 and toks[1].kind == "COMPOUND"):
                return self._parse_compound_assign(line)
            # Funccall as statement: `name(args)` (NO space before `(`) — for
            # side-effect calls of user-defined functions. With a space,
            # `p (x)` reads as verb-with-positional-value instead.
            if (len(toks) >= 2 and toks[1].kind == "LPAREN"
                    and toks[1].col == toks[0].col + len(toks[0].value)):
                return self._parse_expr_stmt(line)
            # Bare expression statement: `x + 1`, `x.method()`, etc. Trigger
            # when toks[1] is an operator-like token so it can't be a verb
            # call (a verb call would have `arg=value` or `->`).
            if len(toks) >= 2 and _is_expr_continuation(toks[1]):
                return self._parse_expr_stmt(line)
            return self._parse_pipeline(line)
        # Non-WORD starts that are still valid as bare expression statements.
        if first.kind in ("NUMBER", "STRING", "FSTRING", "LBRACK", "LBRACE",
                          "LPAREN", "BANG", "KW_TRUE", "KW_FALSE"):
            return self._parse_expr_stmt(line)
        if first.kind == "OP" and first.value == "-":
            # Unary-minus prefix: `-x` as bare expression.
            return self._parse_expr_stmt(line)
        raise ParseError(
            f"expected verb or keyword, got {first.value!r}", first.line, first.col
        )

    def _parse_multi_assign(self, line: _Line) -> "MultiAssignStmt":
        """`a, b = expr` — multiple targets separated by `,` then a single `=`."""
        toks = line.tokens
        targets: List[str] = [toks[0].value]
        i = 1
        while i < len(toks) and toks[i].kind == "COMMA":
            i += 1
            if i >= len(toks) or toks[i].kind != "WORD" or not _is_ident(toks[i].value):
                raise ParseError(
                    f"expected identifier after ',' in multi-assign",
                    toks[i].line if i < len(toks) else line.line_no,
                    toks[i].col if i < len(toks) else 1,
                )
            targets.append(toks[i].value)
            i += 1
        if i >= len(toks) or toks[i].kind != "OP" or toks[i].value != "=":
            raise ParseError(
                f"expected '=' after multi-assign targets",
                toks[i].line if i < len(toks) else line.line_no,
                toks[i].col if i < len(toks) else 1,
            )
        i += 1
        value, i = self._parse_expr(toks, i, line.line_no)
        # Tuple RHS: `a, b = 1, 2` — collect comma-separated exprs into a list
        # that Python (and other targets) can unpack.
        if i < len(toks) and toks[i].kind == "COMMA":
            items = [value]
            while i < len(toks) and toks[i].kind == "COMMA":
                i += 1
                more, i = self._parse_expr(toks, i, line.line_no)
                items.append(more)
            value = ListLit(items)
        if i != len(toks):
            extra = toks[i]
            raise ParseError(
                f"unexpected token after multi-assign expression: {extra.value!r}",
                extra.line, extra.col,
            )
        self.idx += 1
        return MultiAssignStmt(targets=targets, value=value, line=line.line_no)

    def _parse_index_assignment(self, line: _Line):
        """`d["k"] = v` / `xs[0] = v`. Caller verified the shape."""
        toks = line.tokens
        # Parse the LHS as a single value — it'll be an IndexAccess after
        # `_parse_value` finishes (member-access loop handles `[`).
        target, i = self._parse_value(toks, 0, line.line_no)
        if not isinstance(target, IndexAccess):
            raise ParseError(
                "left side of index assignment must be of the form `name[...]`",
                toks[0].line, toks[0].col,
            )
        if i >= len(toks) or toks[i].kind != "OP" or toks[i].value != "=":
            raise ParseError(
                "expected '=' after index target",
                toks[i].line if i < len(toks) else line.line_no,
                toks[i].col if i < len(toks) else 1,
            )
        i += 1
        value, i = self._parse_expr(toks, i, line.line_no)
        if i != len(toks):
            raise ParseError(
                f"unexpected token after index-assign value: {toks[i].value!r}",
                toks[i].line, toks[i].col,
            )
        self.idx += 1
        return IndexAssignStmt(target=target, value=value, line=line.line_no)

    def _parse_compound_index_assignment(self, line: _Line):
        """`xs[0] += 5` desugars to `xs[0] = xs[0] + 5`."""
        toks = line.tokens
        target, i = self._parse_value(toks, 0, line.line_no)
        if not isinstance(target, IndexAccess):
            raise ParseError(
                "compound index assign: left side must be `name[...]`",
                toks[0].line, toks[0].col,
            )
        if i >= len(toks) or toks[i].kind != "COMPOUND":
            raise ParseError(
                "expected compound assignment operator (+=, -=, *=, /=)",
                toks[i].line if i < len(toks) else line.line_no,
                toks[i].col if i < len(toks) else 1,
            )
        op_char = toks[i].value[0]   # `+=` → '+', etc.
        i += 1
        rhs, i = self._parse_expr(toks, i, line.line_no)
        if i != len(toks):
            raise ParseError(
                f"unexpected token after compound-index-assign expression: {toks[i].value!r}",
                toks[i].line, toks[i].col,
            )
        self.idx += 1
        return IndexAssignStmt(
            target=target,
            value=BinOp(op_char, target, rhs),
            line=line.line_no,
        )

    def _parse_compound_assign(self, line: _Line) -> "AssignStmt":
        """`name += expr` desugars to `name = name + expr`. Same for `-= *= /=`."""
        toks = line.tokens
        target = toks[0].value
        if not _is_ident(target):
            raise ParseError(
                f"compound assignment target must be a plain identifier (got {target!r})",
                toks[0].line, toks[0].col,
            )
        op_char = toks[1].value[0]  # `+=` → '+', `-=` → '-', etc.
        rhs, i = self._parse_expr(toks, 2, line.line_no)
        if i != len(toks):
            extra = toks[i]
            raise ParseError(
                f"unexpected token after compound-assign expression: {extra.value!r}",
                extra.line, extra.col,
            )
        self.idx += 1
        return AssignStmt(
            target=target,
            value=BinOp(op_char, Name([target]), rhs),
            line=line.line_no,
        )

    def _parse_assignment(self, line: _Line) -> "AssignStmt":
        toks = line.tokens
        target = toks[0].value
        if not _is_ident(target):
            raise ParseError(
                f"assignment target must be a plain identifier (got {target!r})",
                toks[0].line, toks[0].col,
            )
        # toks[1] is '=' (already verified by caller)
        expr, i = self._parse_expr(toks, 2, line.line_no)
        if i < len(toks):
            extra = toks[i]
            raise ParseError(
                f"unexpected token after assignment expression: {extra.value!r}",
                extra.line, extra.col,
            )
        self.idx += 1
        return AssignStmt(target=target, value=expr, line=line.line_no)

    # ---------- pipeline / call ----------

    def _parse_pipeline(self, line: _Line):
        """Parse a (possibly piped) call line. Returns Call, List[Call], or
        IfStmt when a trailing `if <cond>` wraps the line.

        Special case: a bare identifier followed by `|` is treated as the
        pipe SOURCE (a variable being piped into the next verb) rather than
        a zero-arg verb call. So `xs | reverse | p` works without writing
        `reverse from=xs | p`. Skipped if the identifier names a known verb
        (e.g. `now | format ...`), which keeps zero-arg verb pipes working.
        """
        from .verbs import VERBS
        toks = line.tokens
        calls: List[Call] = []
        i = 0
        pipe_in: Optional[str] = None
        postfix_if_cond: Optional[Value] = None
        # Pipe-from-name: `<ident> | ...`.
        if (len(toks) >= 2
                and toks[0].kind == "WORD"
                and _is_ident(toks[0].value)
                and toks[1].kind == "PIPE"
                and toks[0].value not in VERBS
                and toks[0].value not in VERB_ALIASES):
            pipe_in = toks[0].value
            i = 2
        while i < len(toks):
            call, i = self._parse_call_segment(toks, i, line.line_no, pipe_in)
            calls.append(call)
            if i >= len(toks):
                break
            # Postfix-if / postfix-unless: ` ... if <cond>` or ` ... unless <cond>`
            # at the tail wraps everything in IfStmt.
            if toks[i].kind in ("KW_IF", "KW_UNLESS"):
                is_unless = (toks[i].kind == "KW_UNLESS")
                i += 1
                postfix_if_cond, i = self._parse_expr(toks, i, line.line_no)
                if is_unless:
                    postfix_if_cond = UnaryOp("not", postfix_if_cond)
                if i < len(toks):
                    raise ParseError(
                        f"unexpected token after postfix-if: {toks[i].value!r}",
                        toks[i].line, toks[i].col,
                    )
                break
            if toks[i].kind != "PIPE":
                raise ParseError(
                    f"unexpected token after call: {toks[i].value!r}",
                    toks[i].line, toks[i].col,
                )
            # Pipe: the previous call must have an output name (auto-assign if not).
            i += 1
            if call.out is None:
                call.out = self._fresh_pipe_name()
            pipe_in = call.out
        self.idx += 1
        if postfix_if_cond is not None:
            return IfStmt(cond=postfix_if_cond, then=list(calls),
                          else_=None, line=line.line_no)
        return calls if len(calls) > 1 else calls[0]

    def _fresh_pipe_name(self) -> str:
        self.pipe_counter += 1
        return f"_p{self.pipe_counter}"

    def _parse_call_segment(self, toks: List[Token], i: int, line_no: int,
                            pipe_in: Optional[str]) -> Tuple[Call, int]:
        """Parse one verb-and-args segment. Stops at PIPE or end of tokens."""
        if toks[i].kind != "WORD":
            raise ParseError(f"verb must be a name, got {toks[i].value!r}",
                             toks[i].line, toks[i].col)
        verb = toks[i].value
        if not _is_ident(verb):
            raise ParseError(f"verb name must be a plain identifier (got {verb!r})",
                             toks[i].line, toks[i].col)
        # Expand single-letter aliases to canonical verb names.
        verb = VERB_ALIASES.get(verb, verb)
        i += 1
        args: List[Arg] = []
        out: Optional[str] = None

        # Inject pipe input as <pipe> arg — compiler resolves to primary_arg.
        if pipe_in is not None:
            args.append(Arg("<pipe>", Name([pipe_in])))

        # Optional positional value first: `print "hi"` ≡ `print value="hi"`.
        # Only consume if the next token is a value-starter that is NOT an
        # `ident =` pair (which would be a named arg). Use the full expression
        # parser so arithmetic and method-chains work without explicit parens.
        if i < len(toks) and self._looks_like_positional(toks, i):
            value, i = self._parse_expr(toks, i, line_no)
            args.append(Arg("<pos>", value))

        # Named args + optional ->name; stop at PIPE or postfix-if/unless.
        while i < len(toks) and toks[i].kind not in ("PIPE", "KW_IF", "KW_UNLESS"):
            t = toks[i]
            if t.kind == "ARROW":
                if i + 1 >= len(toks):
                    raise ParseError("expected output name after '->'", t.line, t.col)
                name_tok = toks[i + 1]
                if name_tok.kind != "WORD" or not _is_ident(name_tok.value):
                    raise ParseError(
                        f"expected output variable after '->', got {name_tok.value!r}",
                        name_tok.line, name_tok.col,
                    )
                out = name_tok.value
                i += 2
                if i < len(toks) and toks[i].kind not in ("PIPE", "KW_IF", "KW_UNLESS"):
                    extra = toks[i]
                    raise ParseError(
                        f"unexpected token after output name: {extra.value!r}",
                        extra.line, extra.col,
                    )
                break
            # Must be: NAME '=' VALUE
            if t.kind != "WORD" or not _is_ident(t.value):
                raise ParseError(
                    f"expected arg name (got {t.value!r}); did you forget '=' ?",
                    t.line, t.col,
                )
            arg_name = t.value
            if i + 1 >= len(toks) or toks[i + 1].kind != "OP" or toks[i + 1].value != "=":
                raise ParseError(
                    f"arg {arg_name!r} must be followed by '='",
                    t.line, t.col,
                )
            i += 2
            value, i = self._parse_expr(toks, i, line_no)
            args.append(Arg(arg_name, value))

        return Call(verb=verb, args=args, out=out, line=line_no), i

    def _looks_like_positional(self, toks: List[Token], i: int) -> bool:
        """True if toks[i] starts a value AND is not an `ident =` pair."""
        t = toks[i]
        # `ident =` (named arg) — not positional
        if t.kind == "WORD" and _is_ident(t.value):
            if i + 1 < len(toks) and toks[i + 1].kind == "OP" and toks[i + 1].value == "=":
                return False
        if t.kind in ("STRING", "FSTRING", "NUMBER", "LBRACK", "LBRACE", "LPAREN",
                      "KW_TRUE", "KW_FALSE", "WORD"):
            return True
        # Unary minus at start of a value: -5, -name, -(expr)
        if t.kind == "OP" and t.value == "-":
            return True
        # Unary not at start of a value: !x or `not x`
        if t.kind in ("BANG", "KW_NOT"):
            return True
        return False

    # ---------- value / expression ----------

    def _parse_value(self, toks: List[Token], i: int, line_no: int) -> Tuple[Value, int]:
        """Parse a single value starting at toks[i]. Returns (value, new_i).
        Handles trailing `..end` (range), and postfix `.name` / `.name(args)`
        chains so `data.csv`, `s.upper()`, and `s.split(",")[.length]` parse
        correctly."""
        v, i = self._parse_primary(toks, i, line_no)
        # Range literal: <primary>..<primary>
        if i < len(toks) and toks[i].kind == "DOTDOT":
            i += 1
            end, i = self._parse_primary(toks, i, line_no)
            v = Range(start=v, end=end)
            return v, i
        # Postfix chain: `.name` / `?.name` / `.name(args)` / `[expr]`.
        while i < len(toks) and toks[i].kind in ("DOT", "OPTDOT", "LBRACK"):
            if toks[i].kind == "LBRACK":
                i += 1
                # Python-style slice: optional start, `:`, optional end,
                # optional `:`, optional step.
                #   xs[a:b], xs[a:], xs[:b], xs[:], xs[a:b:c], xs[::-1]
                start_expr = None
                if i < len(toks) and toks[i].kind not in ("COLON", "RBRACK"):
                    start_expr, i = self._parse_expr(toks, i, line_no)
                if i < len(toks) and toks[i].kind == "COLON":
                    i += 1
                    end_expr = None
                    if i < len(toks) and toks[i].kind not in ("COLON", "RBRACK"):
                        end_expr, i = self._parse_expr(toks, i, line_no)
                    step_expr = None
                    if i < len(toks) and toks[i].kind == "COLON":
                        i += 1
                        if i < len(toks) and toks[i].kind != "RBRACK":
                            step_expr, i = self._parse_expr(toks, i, line_no)
                    idx_expr: Value = Slice(start=start_expr, end=end_expr,
                                            step=step_expr)
                else:
                    if start_expr is None:
                        raise ParseError(
                            "empty index access — use xs[i] or xs[a:b]",
                            toks[i].line if i < len(toks) else line_no,
                            toks[i].col if i < len(toks) else 1,
                        )
                    idx_expr = start_expr
                if i >= len(toks) or toks[i].kind != "RBRACK":
                    raise ParseError(
                        "expected ']' to close index access",
                        toks[i].line if i < len(toks) else line_no,
                        toks[i].col if i < len(toks) else 1,
                    )
                i += 1
                v = IndexAccess(receiver=v, index=idx_expr)
                continue
            # DOT or OPTDOT
            is_optional = toks[i].kind == "OPTDOT"
            i += 1
            if i >= len(toks) or toks[i].kind != "WORD" or not _is_ident(toks[i].value):
                raise ParseError(
                    "expected attribute or method name after '.'",
                    toks[i].line if i < len(toks) else line_no,
                    toks[i].col if i < len(toks) else 1,
                )
            member = toks[i].value
            i += 1
            if i < len(toks) and toks[i].kind == "LPAREN":
                # method call — parse args
                i += 1
                m_args: List[Value] = []
                if i < len(toks) and toks[i].kind == "RPAREN":
                    i += 1
                else:
                    while True:
                        arg_v, i = self._parse_value_maybe_spread(toks, i, line_no)
                        m_args.append(arg_v)
                        if i >= len(toks):
                            raise ParseError("expected ')' to close method call",
                                             line_no, 1)
                        if toks[i].kind == "COMMA":
                            i += 1
                            continue
                        if toks[i].kind == "RPAREN":
                            i += 1
                            break
                        raise ParseError(
                            f"expected ',' or ')' in method call, got {toks[i].value!r}",
                            toks[i].line, toks[i].col,
                        )
                v = MethodCall(receiver=v, method=member, args=m_args,
                               optional=is_optional)
            else:
                # Attribute access. Plain `.name` on a Name extends the path
                # (so dotted barewords still work). With `?.name` we always
                # use MethodCall to carry the optional flag.
                if isinstance(v, Name) and not is_optional:
                    v = Name(parts=v.parts + [member])
                else:
                    v = MethodCall(receiver=v, method=member, args=None,
                                   optional=is_optional)
        return v, i

    def _parse_primary(self, toks: List[Token], i: int, line_no: int) -> Tuple[Value, int]:
        if i >= len(toks):
            raise ParseError("expected a value", line_no, 1)
        t = toks[i]
        # Unary minus: `-N`, `-x`, `-(expr)`. Implemented as 0 - inner, except
        # we fold for literal numbers so `-5` stays a NumberLit.
        if t.kind == "OP" and t.value == "-":
            inner, j = self._parse_primary(toks, i + 1, line_no)
            if isinstance(inner, NumberLit):
                return NumberLit(-inner.value), j
            return BinOp("-", NumberLit(0.0), inner), j
        # Unary not: `!x`, `not x`.
        if t.kind == "BANG" or t.kind == "KW_NOT":
            inner, j = self._parse_primary(toks, i + 1, line_no)
            return UnaryOp("not", inner), j
        if t.kind == "STRING":
            return StringLit(t.value), i + 1
        if t.kind == "FSTRING":
            return _parse_fstring(t.value, t.line, t.col), i + 1
        if t.kind == "NUMBER":
            return NumberLit(float(t.value) if "." in t.value else float(int(t.value))), i + 1
        if t.kind == "KW_TRUE":
            return BoolLit(True), i + 1
        if t.kind == "KW_FALSE":
            return BoolLit(False), i + 1
        if t.kind == "LBRACK":
            return self._parse_list(toks, i, line_no)
        if t.kind == "LBRACE":
            return self._parse_dict(toks, i, line_no)
        if t.kind == "LPAREN":
            i += 1
            expr, i = self._parse_expr(toks, i, line_no)
            if i >= len(toks) or toks[i].kind != "RPAREN":
                raise ParseError(
                    "expected ')' to close grouped expression",
                    toks[i].line if i < len(toks) else line_no,
                    toks[i].col if i < len(toks) else 1,
                )
            return expr, i + 1
        if t.kind == "WORD":
            if i + 1 < len(toks) and toks[i + 1].kind == "LPAREN":
                return self._parse_funccall(toks, i, line_no)
            parts = _split_word_path(t.value)
            return Name(parts), i + 1
        raise ParseError(f"expected a value, got {t.value!r}", t.line, t.col)

    def _parse_list(self, toks: List[Token], i: int, line_no: int):
        i += 1  # consume '['
        items: List[Value] = []
        if i < len(toks) and toks[i].kind == "RBRACK":
            return ListLit(items), i + 1
        first_val, i = self._parse_value_maybe_spread(toks, i, line_no)
        # Comprehension detection: `[EXPR for VAR in SOURCE (if COND)?]`.
        # `for` is a soft keyword here — only triggers inside brackets.
        if (i < len(toks) and toks[i].kind == "WORD"
                and toks[i].value == "for"):
            i += 1
            if i >= len(toks) or toks[i].kind != "WORD" or not _is_ident(toks[i].value):
                raise ParseError(
                    "list comprehension: expected variable name after 'for'",
                    toks[i].line if i < len(toks) else line_no,
                    toks[i].col if i < len(toks) else 1,
                )
            var = toks[i].value
            i += 1
            if i >= len(toks) or toks[i].kind != "KW_IN":
                raise ParseError(
                    "list comprehension: expected 'in' after the variable",
                    toks[i].line if i < len(toks) else line_no,
                    toks[i].col if i < len(toks) else 1,
                )
            i += 1
            source, i = self._parse_expr(toks, i, line_no)
            cond: Optional[Value] = None
            if i < len(toks) and toks[i].kind == "KW_IF":
                i += 1
                cond, i = self._parse_expr(toks, i, line_no)
            if i >= len(toks) or toks[i].kind != "RBRACK":
                raise ParseError(
                    "expected ']' to close list comprehension",
                    toks[i].line if i < len(toks) else line_no,
                    toks[i].col if i < len(toks) else 1,
                )
            return ListComp(expr=first_val, var=var, source=source, cond=cond), i + 1
        items.append(first_val)
        # Otherwise continue as a regular list literal.
        while True:
            if i >= len(toks):
                raise ParseError("expected ']' to close list", line_no, 1)
            if toks[i].kind == "COMMA":
                i += 1
                val, i = self._parse_value_maybe_spread(toks, i, line_no)
                items.append(val)
                continue
            if toks[i].kind == "RBRACK":
                return ListLit(items), i + 1
            raise ParseError(f"expected ',' or ']' in list, got {toks[i].value!r}",
                             toks[i].line, toks[i].col)

    def _parse_value_maybe_spread(self, toks: List[Token], i: int, line_no: int) -> Tuple[Value, int]:
        """Parse a value, with optional leading `*` for spread. Used inside
        list literals and funccall args."""
        if i < len(toks) and toks[i].kind == "OP" and toks[i].value == "*":
            i += 1
            inner, i = self._parse_expr(toks, i, line_no)
            return Spread(value=inner), i
        return self._parse_expr(toks, i, line_no)

    def _parse_dict(self, toks: List[Token], i: int, line_no: int):
        i += 1  # consume '{'
        # Lookahead for `for` at depth 0 → dict comprehension.
        if _has_for_in_braces(toks, i):
            return self._parse_dict_comp(toks, i, line_no)
        entries: List[Tuple[str, Value]] = []
        if i < len(toks) and toks[i].kind == "RBRACE":
            return DictLit(entries), i + 1
        while True:
            # key: must be a string literal or a bare identifier (no dots)
            kt = toks[i]
            if kt.kind == "STRING":
                key = kt.value
                i += 1
            elif kt.kind == "WORD" and _is_ident(kt.value):
                key = kt.value
                i += 1
            else:
                raise ParseError(
                    f"dict key must be a string or identifier, got {kt.value!r}",
                    kt.line, kt.col,
                )
            if i >= len(toks) or toks[i].kind != "COLON":
                raise ParseError(f"expected ':' after dict key {key!r}",
                                 toks[i].line if i < len(toks) else kt.line,
                                 toks[i].col  if i < len(toks) else kt.col)
            i += 1
            val, i = self._parse_expr(toks, i, line_no)
            entries.append((key, val))
            if i >= len(toks):
                raise ParseError("expected '}' to close dict", line_no, 1)
            if toks[i].kind == "COMMA":
                i += 1
                continue
            if toks[i].kind == "RBRACE":
                return DictLit(entries), i + 1
            raise ParseError(f"expected ',' or '}}' in dict, got {toks[i].value!r}",
                             toks[i].line, toks[i].col)

    def _parse_dict_comp(self, toks: List[Token], i: int, line_no: int):
        """Parse `key_expr : val_expr for var in source (if cond) }`.
        Caller has already consumed `{`."""
        key_expr, i = self._parse_expr(toks, i, line_no)
        if i >= len(toks) or toks[i].kind != "COLON":
            raise ParseError(
                "dict comprehension: expected ':' between key and value",
                toks[i].line if i < len(toks) else line_no,
                toks[i].col if i < len(toks) else 1,
            )
        i += 1
        val_expr, i = self._parse_expr(toks, i, line_no)
        if i >= len(toks) or toks[i].kind != "WORD" or toks[i].value != "for":
            raise ParseError(
                "dict comprehension: expected 'for' after the value expression",
                toks[i].line if i < len(toks) else line_no,
                toks[i].col if i < len(toks) else 1,
            )
        i += 1
        if i >= len(toks) or toks[i].kind != "WORD" or not _is_ident(toks[i].value):
            raise ParseError(
                "dict comprehension: expected variable name after 'for'",
                toks[i].line if i < len(toks) else line_no,
                toks[i].col if i < len(toks) else 1,
            )
        var = toks[i].value
        i += 1
        if i >= len(toks) or toks[i].kind != "KW_IN":
            raise ParseError(
                "dict comprehension: expected 'in' after the variable",
                toks[i].line if i < len(toks) else line_no,
                toks[i].col if i < len(toks) else 1,
            )
        i += 1
        source, i = self._parse_expr(toks, i, line_no)
        cond: Optional[Value] = None
        if i < len(toks) and toks[i].kind == "KW_IF":
            i += 1
            cond, i = self._parse_expr(toks, i, line_no)
        if i >= len(toks) or toks[i].kind != "RBRACE":
            raise ParseError(
                "expected '}' to close dict comprehension",
                toks[i].line if i < len(toks) else line_no,
                toks[i].col if i < len(toks) else 1,
            )
        return DictComp(key_expr=key_expr, val_expr=val_expr,
                        var=var, source=source, cond=cond), i + 1

    def _parse_funccall(self, toks: List[Token], i: int, line_no: int) -> Tuple[FuncCall, int]:
        name = toks[i].value
        if not _is_ident(name):
            raise ParseError(f"function name must be an identifier (got {name!r})",
                             toks[i].line, toks[i].col)
        i += 2  # skip name + '('
        args: List[Value] = []
        if i < len(toks) and toks[i].kind == "RPAREN":
            return FuncCall(name, args), i + 1
        first_val, i = self._parse_value_maybe_spread(toks, i, line_no)
        # Generator expression: `f(expr for var in src (if cond))`. We lower
        # to a ListComp argument — Python builtins like sum/max/any accept
        # a list just as happily as a generator.
        if (i < len(toks) and toks[i].kind == "WORD"
                and toks[i].value == "for"):
            i += 1
            if i >= len(toks) or toks[i].kind != "WORD" or not _is_ident(toks[i].value):
                raise ParseError(
                    "generator: expected variable name after 'for'",
                    toks[i].line if i < len(toks) else line_no,
                    toks[i].col if i < len(toks) else 1,
                )
            var = toks[i].value
            i += 1
            if i >= len(toks) or toks[i].kind != "KW_IN":
                raise ParseError(
                    "generator: expected 'in' after the variable",
                    toks[i].line if i < len(toks) else line_no,
                    toks[i].col if i < len(toks) else 1,
                )
            i += 1
            source, i = self._parse_expr(toks, i, line_no)
            cond: Optional[Value] = None
            if i < len(toks) and toks[i].kind == "KW_IF":
                i += 1
                cond, i = self._parse_expr(toks, i, line_no)
            if i >= len(toks) or toks[i].kind != "RPAREN":
                raise ParseError(
                    "expected ')' to close generator expression",
                    toks[i].line if i < len(toks) else line_no,
                    toks[i].col if i < len(toks) else 1,
                )
            return FuncCall(name, [ListComp(expr=first_val, var=var,
                                            source=source, cond=cond)]), i + 1
        args.append(first_val)
        while True:
            if i >= len(toks):
                raise ParseError("expected ')' to close function call", line_no, 1)
            if toks[i].kind == "COMMA":
                i += 1
                val, i = self._parse_value_maybe_spread(toks, i, line_no)
                args.append(val)
                continue
            if toks[i].kind == "RPAREN":
                return FuncCall(name, args), i + 1
            raise ParseError(f"expected ',' or ')' in function call, got {toks[i].value!r}",
                             toks[i].line, toks[i].col)

    def _parse_expr(self, toks: List[Token], i: int, line_no: int,
                    min_prec: int = 0) -> Tuple[Value, int]:
        """Precedence-climbing expression parser.

        Precedence (low → high):
          or/||          : 1
          and/&&         : 2
          == != < > <= >=: 3
          + -            : 4
          * /            : 5
        Ternary `?:` sits at precedence 0 (looser than everything else).
        """
        left, i = self._parse_value(toks, i, line_no)
        while i < len(toks):
            op_tok = toks[i]
            # Two-token operators: `not in`.
            if (op_tok.kind == "KW_NOT"
                    and i + 1 < len(toks)
                    and toks[i + 1].kind == "KW_IN"):
                prec = _OP_PRECEDENCE["not in"]
                if prec < min_prec:
                    break
                i += 2
                right, i = self._parse_expr(toks, i, line_no, prec + 1)
                left = BinOp("not in", left, right)
                continue
            op = _op_from_token(op_tok)
            if op is None:
                break
            if op == "=":
                raise ParseError("did you mean '==' for comparison?",
                                 op_tok.line, op_tok.col)
            prec = _OP_PRECEDENCE.get(op)
            if prec is None or prec < min_prec:
                break
            i += 1
            right, i = self._parse_expr(toks, i, line_no, prec + 1)
            left = BinOp(op, left, right)
        # Ternary is the loosest binder; only consume one level at min_prec 0.
        if min_prec <= 0 and i < len(toks) and toks[i].kind == "QMARK":
            i += 1
            then_val, i = self._parse_expr(toks, i, line_no)
            if i >= len(toks) or toks[i].kind != "COLON":
                raise ParseError(
                    "ternary '?' must be followed by '... : ...'",
                    toks[i].line if i < len(toks) else line_no,
                    toks[i].col if i < len(toks) else 1,
                )
            i += 1
            else_val, i = self._parse_expr(toks, i, line_no)
            left = Ternary(cond=left, then=then_val, else_=else_val)
        # Python-style ternary `X if COND else Y` at expression level.
        # Disambiguated from postfix-if (statement modifier) by looking ahead
        # for KW_ELSE on the same line — present means it's the expr ternary;
        # absent means postfix-if and we don't consume.
        elif (min_prec <= 0 and i < len(toks) and toks[i].kind == "KW_IF"
                and any(t.kind == "KW_ELSE" for t in toks[i + 1:])):
            i += 1  # consume `if`
            cond, i = self._parse_expr(toks, i, line_no, min_prec=1)
            if i >= len(toks) or toks[i].kind != "KW_ELSE":
                raise ParseError(
                    "expected 'else' after `X if COND` ternary condition",
                    toks[i].line if i < len(toks) else line_no,
                    toks[i].col if i < len(toks) else 1,
                )
            i += 1  # consume `else`
            else_val, i = self._parse_expr(toks, i, line_no)
            left = Ternary(cond=cond, then=left, else_=else_val)
        return left, i

    # ---------- control blocks ----------

    def _parse_if(self, base_indent: int) -> IfStmt:
        line = self.lines[self.idx]
        toks = line.tokens
        cond, i = self._parse_expr(toks, 1, line.line_no)
        if i != len(toks):
            raise ParseError(f"unexpected token in 'if' condition: {toks[i].value!r}",
                             toks[i].line, toks[i].col)
        self.idx += 1
        then_body, _ = self._parse_block(base_indent + 1)
        else_body: Optional[List[Stmt]] = None
        if self.idx < len(self.lines):
            nxt = self.lines[self.idx]
            if nxt.indent == base_indent and nxt.tokens and nxt.tokens[0].kind == "KW_ELSE":
                if len(nxt.tokens) != 1:
                    raise ParseError("'else' must be on its own line",
                                     nxt.tokens[1].line, nxt.tokens[1].col)
                self.idx += 1
                else_body, _ = self._parse_block(base_indent + 1)
        return IfStmt(cond, then_body, else_body, line.line_no)

    def _parse_each(self, base_indent: int) -> EachStmt:
        line = self.lines[self.idx]
        toks = line.tokens
        if len(toks) < 4:
            raise ParseError("'each' needs: each <name>[, <name>] in <value>",
                             line.line_no, 1)
        if toks[1].kind != "WORD" or not _is_ident(toks[1].value):
            raise ParseError(f"expected variable name after 'each', got {toks[1].value!r}",
                             toks[1].line, toks[1].col)
        var = toks[1].value
        key_var: Optional[str] = None
        i = 2
        # Optional second var: `each k, v in dict`
        if toks[i].kind == "COMMA":
            if i + 1 >= len(toks) or toks[i + 1].kind != "WORD" or not _is_ident(toks[i + 1].value):
                raise ParseError("expected second variable name after ','",
                                 toks[i].line, toks[i].col)
            key_var = var
            var = toks[i + 1].value
            i += 2
        if toks[i].kind != "KW_IN":
            raise ParseError(f"expected 'in' after each-variable, got {toks[i].value!r}",
                             toks[i].line, toks[i].col)
        iterable, i = self._parse_expr(toks, i + 1, line.line_no)
        if i != len(toks):
            raise ParseError(f"unexpected token after iterable: {toks[i].value!r}",
                             toks[i].line, toks[i].col)
        self.idx += 1
        body, _ = self._parse_block(base_indent + 1)
        return EachStmt(var=var, iterable=iterable, body=body, line=line.line_no,
                        key_var=key_var)

    def _parse_repeat(self, base_indent: int) -> RepeatStmt:
        line = self.lines[self.idx]
        toks = line.tokens
        if len(toks) < 2:
            raise ParseError("'repeat' needs a count: repeat <number-or-name>", line.line_no, 1)
        count, i = self._parse_expr(toks, 1, line.line_no)
        var: Optional[str] = None
        if i < len(toks) and toks[i].kind == "KW_AS":
            if i + 1 >= len(toks):
                raise ParseError("expected variable name after 'as'", toks[i].line, toks[i].col)
            name_tok = toks[i + 1]
            if name_tok.kind != "WORD" or not _is_ident(name_tok.value):
                raise ParseError(f"expected variable name after 'as', got {name_tok.value!r}",
                                 name_tok.line, name_tok.col)
            var = name_tok.value
            i += 2
        if i != len(toks):
            raise ParseError(f"unexpected token after repeat: {toks[i].value!r}",
                             toks[i].line, toks[i].col)
        self.idx += 1
        body, _ = self._parse_block(base_indent + 1)
        return RepeatStmt(count, body, line.line_no, var=var)

    def _parse_def(self, base_indent: int) -> DefStmt:
        """`def name p1 p2 ...` followed by an indented body."""
        line = self.lines[self.idx]
        toks = line.tokens
        if len(toks) < 2:
            raise ParseError("'def' needs a name: def <name> [params...]",
                             line.line_no, 1)
        name_tok = toks[1]
        if name_tok.kind != "WORD" or not _is_ident(name_tok.value):
            raise ParseError(
                f"function name must be a plain identifier (got {name_tok.value!r})",
                name_tok.line, name_tok.col,
            )
        name = name_tok.value
        params: List[Tuple[str, Optional[Value]]] = []
        i = 2
        while i < len(toks):
            t = toks[i]
            if t.kind != "WORD" or not _is_ident(t.value):
                raise ParseError(
                    f"def param must be a plain identifier (got {t.value!r})",
                    t.line, t.col,
                )
            pname = t.value
            i += 1
            default: Optional[Value] = None
            # Optional default: `name=expr`. Stop at next param boundary
            # (a fresh WORD token) or end of line.
            if i < len(toks) and toks[i].kind == "OP" and toks[i].value == "=":
                i += 1
                default, i = self._parse_expr(toks, i, line.line_no)
            params.append((pname, default))
        self.idx += 1
        body, _ = self._parse_block(base_indent + 1)
        # Implicit return is handled at compile time, NOT at parse time, so
        # the AST preserves whether the user wrote `return` explicitly. That
        # lets lint distinguish "you already have implicit return" from "you
        # could drop your explicit return".
        return DefStmt(name=name, params=params, body=body, line=line.line_no)

    def _parse_return(self, line: _Line) -> ReturnStmt:
        toks = line.tokens
        if len(toks) == 1:
            self.idx += 1
            return ReturnStmt(value=None, line=line.line_no)
        value, i = self._parse_expr(toks, 1, line.line_no)
        if i != len(toks):
            raise ParseError(
                f"unexpected token after return value: {toks[i].value!r}",
                toks[i].line, toks[i].col,
            )
        self.idx += 1
        return ReturnStmt(value=value, line=line.line_no)

    def _parse_expr_stmt(self, line: _Line):
        toks = line.tokens
        value, i = self._parse_expr(toks, 0, line.line_no)
        # Pipe-from-value: `<expr> | verb | ...`. Lower to a synthetic
        # `_pN = <expr>` assignment followed by the rest of the pipeline.
        if i < len(toks) and toks[i].kind == "PIPE":
            name = self._fresh_pipe_name()
            stmts: List = [AssignStmt(target=name, value=value, line=line.line_no)]
            pipe_in = name
            i += 1
            postfix_if_cond: Optional[Value] = None
            while i < len(toks):
                call, i = self._parse_call_segment(toks, i, line.line_no, pipe_in)
                stmts.append(call)
                if i >= len(toks):
                    break
                if toks[i].kind in ("KW_IF", "KW_UNLESS"):
                    is_unless = (toks[i].kind == "KW_UNLESS")
                    i += 1
                    postfix_if_cond, i = self._parse_expr(toks, i, line.line_no)
                    if is_unless:
                        postfix_if_cond = UnaryOp("not", postfix_if_cond)
                    if i < len(toks):
                        raise ParseError(
                            f"unexpected token after postfix-if: {toks[i].value!r}",
                            toks[i].line, toks[i].col,
                        )
                    break
                if toks[i].kind != "PIPE":
                    raise ParseError(
                        f"unexpected token after call: {toks[i].value!r}",
                        toks[i].line, toks[i].col,
                    )
                i += 1
                if call.out is None:
                    call.out = self._fresh_pipe_name()
                pipe_in = call.out
            self.idx += 1
            if postfix_if_cond is not None:
                return IfStmt(cond=postfix_if_cond, then=stmts,
                              else_=None, line=line.line_no)
            return stmts
        if i != len(toks):
            raise ParseError(
                f"unexpected token after expression: {toks[i].value!r}",
                toks[i].line, toks[i].col,
            )
        self.idx += 1
        return ExprStmt(value=value, line=line.line_no)

    def _parse_break_continue(self, line: _Line, ctor, name: str):
        """Parse `break` / `continue`, optionally followed by `if cond` or
        `unless cond` (postfix conditional)."""
        toks = line.tokens
        self.idx += 1
        stmt = ctor(line=line.line_no)
        if len(toks) == 1:
            return stmt
        kw = toks[1].kind
        if kw not in ("KW_IF", "KW_UNLESS"):
            raise ParseError(
                f"'{name}' must be alone or followed by 'if/unless <cond>'",
                toks[1].line, toks[1].col,
            )
        cond, i = self._parse_expr(toks, 2, line.line_no)
        if i != len(toks):
            raise ParseError(
                f"unexpected token after postfix conditional: {toks[i].value!r}",
                toks[i].line, toks[i].col,
            )
        if kw == "KW_UNLESS":
            cond = UnaryOp("not", cond)
        return IfStmt(cond=cond, then=[stmt], else_=None, line=line.line_no)

    def _parse_match(self, base_indent: int) -> MatchStmt:
        line = self.lines[self.idx]
        toks = line.tokens
        if len(toks) < 2:
            raise ParseError("'match' needs a value: match <expr>",
                             line.line_no, 1)
        value, i = self._parse_expr(toks, 1, line.line_no)
        if i != len(toks):
            raise ParseError(f"unexpected token after match value: {toks[i].value!r}",
                             toks[i].line, toks[i].col)
        self.idx += 1
        cases: List[Tuple[Value, List["Stmt"]]] = []
        else_body: Optional[List["Stmt"]] = None
        # Collect `case PATTERN` and optional `else` arms at one indent deeper.
        case_indent = base_indent + 1
        while self.idx < len(self.lines):
            l = self.lines[self.idx]
            if l.indent != case_indent:
                break
            if l.tokens[0].kind == "KW_CASE":
                if len(l.tokens) < 2:
                    raise ParseError("'case' needs a pattern: case <expr>",
                                     l.line_no, 1)
                pat, j = self._parse_expr(l.tokens, 1, l.line_no)
                if j != len(l.tokens):
                    raise ParseError(
                        f"unexpected token after case pattern: {l.tokens[j].value!r}",
                        l.tokens[j].line, l.tokens[j].col,
                    )
                self.idx += 1
                body, _ = self._parse_block(case_indent + 1)
                cases.append((pat, body))
            elif l.tokens[0].kind == "KW_ELSE":
                if len(l.tokens) != 1:
                    raise ParseError(
                        "'else' inside match must be alone on its line",
                        l.tokens[1].line, l.tokens[1].col,
                    )
                self.idx += 1
                else_body, _ = self._parse_block(case_indent + 1)
                break
            else:
                break
        if not cases and else_body is None:
            raise ParseError(
                "'match' must have at least one `case` or `else` arm",
                line.line_no, 1,
            )
        return MatchStmt(value=value, cases=cases, else_body=else_body,
                         line=line.line_no)

    def _parse_unless(self, base_indent: int) -> IfStmt:
        """`unless cond` ≡ `if !cond`. Desugars to an IfStmt at parse time."""
        line = self.lines[self.idx]
        toks = line.tokens
        cond, i = self._parse_expr(toks, 1, line.line_no)
        if i != len(toks):
            raise ParseError(f"unexpected token in 'unless': {toks[i].value!r}",
                             toks[i].line, toks[i].col)
        self.idx += 1
        then_body, _ = self._parse_block(base_indent + 1)
        # No else for unless; keep semantics simple.
        return IfStmt(cond=UnaryOp("not", cond), then=then_body,
                      else_=None, line=line.line_no)

    def _parse_try(self, base_indent: int) -> TryStmt:
        line = self.lines[self.idx]
        toks = line.tokens
        if len(toks) != 1:
            raise ParseError("'try' must be alone on its line",
                             toks[1].line, toks[1].col)
        self.idx += 1
        try_body, _ = self._parse_block(base_indent + 1)
        # Expect a 'catch' at the same indent.
        if self.idx >= len(self.lines):
            raise ParseError("expected 'catch' after 'try' block", line.line_no, 1)
        nxt = self.lines[self.idx]
        if nxt.indent != base_indent or not nxt.tokens or nxt.tokens[0].kind != "KW_CATCH":
            raise ParseError("expected 'catch' at same indent as 'try'",
                             nxt.line_no, 1)
        ctoks = nxt.tokens
        catch_var: Optional[str] = None
        if len(ctoks) == 1:
            pass  # bare `catch`
        elif len(ctoks) == 2 and ctoks[1].kind == "WORD" and _is_ident(ctoks[1].value):
            catch_var = ctoks[1].value
        else:
            raise ParseError("`catch` accepts at most one variable name",
                             ctoks[1].line, ctoks[1].col)
        self.idx += 1
        catch_body, _ = self._parse_block(base_indent + 1)
        return TryStmt(try_body=try_body, catch_var=catch_var,
                       catch_body=catch_body, line=line.line_no)

    def _parse_while(self, base_indent: int) -> WhileStmt:
        line = self.lines[self.idx]
        toks = line.tokens
        if len(toks) < 2:
            raise ParseError("'while' needs a condition: while <expr>",
                             line.line_no, 1)
        cond, i = self._parse_expr(toks, 1, line.line_no)
        if i != len(toks):
            raise ParseError(f"unexpected token after while condition: {toks[i].value!r}",
                             toks[i].line, toks[i].col)
        self.idx += 1
        body, _ = self._parse_block(base_indent + 1)
        return WhileStmt(cond=cond, body=body, line=line.line_no)

    def _parse_when(self, base_indent: int) -> WhenStmt:
        line = self.lines[self.idx]
        toks = line.tokens
        if len(toks) < 2:
            raise ParseError("'when' needs an event: when <event>", line.line_no, 1)
        event_tok = toks[1]
        if event_tok.kind != "WORD" or not _is_ident(event_tok.value):
            raise ParseError(f"expected event name after 'when', got {event_tok.value!r}",
                             event_tok.line, event_tok.col)
        event = event_tok.value
        args: List[Value] = []
        i = 2
        while i < len(toks):
            val, i = self._parse_expr(toks, i, line.line_no)
            args.append(val)
        self.idx += 1
        body, _ = self._parse_block(base_indent + 1)
        return WhenStmt(event, args, body, line.line_no)


# ============================================================
# Helpers
# ============================================================


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_expr_continuation(tok) -> bool:
    """True if `tok` can only appear as part of an expression after a value,
    such that `WORD tok ...` couldn't possibly be a verb call.

    Used to detect bare expression statements like `x.method()`, `x == 1`,
    `x + 1`, `x ?? y`. This enables implicit return: `def f x\\n  x * 2`.

    Carefully excluded:
      - LBRACK: `p [1, 2]` is a verb call with a positional list value.
      - OP `-`: `verb -10` is a verb call with a negative positional value
        (unary minus prefix). Subtraction with `-` must use explicit `return`.
    """
    if tok.kind in ("DOT", "OPTDOT", "DOTDOT",
                    "CMP", "NULLCOAL", "ANDOP", "OROP", "QMARK",
                    "KW_AND", "KW_OR"):
        return True
    if tok.kind == "OP" and tok.value in ("+", "*", "/", "%", "<", ">"):
        return True
    return False


def _is_index_assign(toks) -> bool:
    """`WORD [ ... ] =` pattern. The `=` must be plain, not `==`."""
    op_pos = _index_assign_op_pos(toks)
    return (op_pos >= 0
            and toks[op_pos].kind == "OP"
            and toks[op_pos].value == "=")


def _is_compound_index_assign(toks) -> bool:
    """`WORD [ ... ] (+=|-=|*=|/=)`."""
    op_pos = _index_assign_op_pos(toks)
    return op_pos >= 0 and toks[op_pos].kind == "COMPOUND"


def _index_assign_op_pos(toks) -> int:
    """Return the position of the assignment-op token after `WORD [ ... ]`,
    or -1 if the line doesn't match that shape."""
    if len(toks) < 5 or toks[0].kind != "WORD" or toks[1].kind != "LBRACK":
        return -1
    depth = 1
    i = 2
    while i < len(toks) and depth > 0:
        if toks[i].kind == "LBRACK":
            depth += 1
        elif toks[i].kind == "RBRACK":
            depth -= 1
        i += 1
    if i >= len(toks):
        return -1
    return i


def _looks_like_multi_assign(toks) -> bool:
    """Return True iff `toks` is `WORD , WORD (, WORD)* OP=`.
    Used as a lookahead in _parse_stmt to disambiguate from pipelines.
    """
    if not toks or toks[0].kind != "WORD" or not _is_ident(toks[0].value):
        return False
    i = 1
    while i < len(toks):
        if toks[i].kind == "OP" and toks[i].value == "=":
            return i >= 3  # need at least one comma + name before the '='
        if toks[i].kind != "COMMA":
            return False
        i += 1
        if i >= len(toks) or toks[i].kind != "WORD" or not _is_ident(toks[i].value):
            return False
        i += 1
    return False

_OP_PRECEDENCE = {
    "??": 1,  # null-coalescing — lowest, paired with or
    "or":  1, "and": 2,
    "==": 3, "!=": 3, "<":  3, ">":  3, "<=": 3, ">=": 3,
    "in": 3, "not in": 3,
    "+":  4, "-":  4,
    "*":  5, "/":  5, "%":  5,
}


def _op_from_token(tok: "Token") -> Optional[str]:
    """Return the canonical operator name for an operator-like token, or None."""
    if tok.kind == "NULLCOAL":
        return "??"
    if tok.kind == "ANDOP":
        return "and"
    if tok.kind == "OROP":
        return "or"
    if tok.kind == "KW_AND":
        return "and"
    if tok.kind == "KW_OR":
        return "or"
    if tok.kind == "KW_IN":
        return "in"
    if tok.kind in ("CMP", "OP"):
        return tok.value
    return None


def _has_for_in_braces(toks: List["Token"], i: int) -> bool:
    """Scan from `i` (just past `{`) until the matching `}` at depth 0.
    Return True if a soft-keyword `for` appears at that depth — i.e. the
    braces enclose a comprehension, not a dict literal."""
    depth = 0
    while i < len(toks):
        t = toks[i]
        if t.kind == "RBRACE":
            if depth == 0:
                return False
            depth -= 1
        elif t.kind in ("LBRACE", "LBRACK", "LPAREN"):
            depth += 1
        elif t.kind in ("RBRACK", "RPAREN"):
            if depth > 0:
                depth -= 1
        elif t.kind == "WORD" and t.value == "for" and depth == 0:
            return True
        i += 1
    return False


def _is_ident(s: str) -> bool:
    return bool(_IDENT_RE.match(s))


def _split_word_path(w: str) -> List[str]:
    """Split a WORD token into dotted parts. Filenames like 'data.csv' → ['data','csv']."""
    return w.split(".")


# Accept arbitrary placeholder content (anything but `{` or `}`). Languages
# like Python and JS evaluate the expression inside the placeholder natively,
# so users can write `f"{x + 1}"` or `f"{count(items)}"` and have it work.
_FSTRING_PLACEHOLDER = re.compile(r"\{([^{}]+)\}")


def _parse_fstring(content: str, line: int, col: int) -> "FString":
    """Split an f-string body into text segments and parsed-expression segments.

    Returns FString.parts as 3-tuples:
      ("text", str, "")              literal text fragment
      ("expr", Value, fmt_spec_str)  parsed Flow expression + optional format spec

    The format spec is the part after a top-level `:` in the placeholder
    (e.g. `.2f` in `{x:.2f}`). Nested `:` inside parens/brackets/braces
    or after a `?` (ternary) are not treated as format-spec separators.
    """
    parts: List[Tuple[str, object, str]] = []
    pos = 0
    while pos < len(content):
        m = _FSTRING_PLACEHOLDER.search(content, pos)
        if not m:
            parts.append(("text", content[pos:], ""))
            break
        if m.start() > pos:
            parts.append(("text", content[pos:m.start()], ""))
        placeholder = m.group(1)
        expr_str, fmt_spec = _split_fstring_format_spec(placeholder)
        # Tokenize + parse the expr part as a Flow expression.
        try:
            sub_tokens = _tokenize_line(expr_str, line)
        except ParseError as e:
            raise ParseError(
                f"invalid f-string placeholder {{{placeholder}}}: {e.msg}",
                line, col + m.start(),
            )
        if not sub_tokens:
            raise ParseError(f"empty f-string placeholder", line, col + m.start())
        sub_parser = _Parser([])
        expr, end = sub_parser._parse_expr(sub_tokens, 0, line)
        if end != len(sub_tokens):
            raise ParseError(
                f"unexpected token {sub_tokens[end].value!r} in f-string placeholder",
                sub_tokens[end].line, sub_tokens[end].col,
            )
        parts.append(("expr", expr, fmt_spec))
        pos = m.end()
    parts = [p for p in parts if not (p[0] == "text" and p[1] == "")]
    return FString(parts=parts)


def _split_fstring_format_spec(content: str) -> Tuple[str, str]:
    """Split `expr:fmt` at the first top-level `:` not inside [],(),{} or
    consumed by a ternary `?:`. Returns (expr_str, fmt_spec) — fmt_spec is
    an empty string when no spec is present."""
    depth = 0
    q_pending = 0   # `?` seen at depth 0 awaiting its matching `:`
    in_str = False
    str_quote = ""
    i = 0
    while i < len(content):
        ch = content[i]
        if in_str:
            if ch == "\\" and i + 1 < len(content):
                i += 2
                continue
            if ch == str_quote:
                in_str = False
        else:
            if ch in '"\'':
                in_str = True
                str_quote = ch
            elif ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == "?" and depth == 0:
                q_pending += 1
            elif ch == ":" and depth == 0:
                if q_pending > 0:
                    q_pending -= 1  # this `:` belongs to a ternary
                else:
                    return content[:i], content[i + 1:]
        i += 1
    return content, ""


# ============================================================
# Public API
# ============================================================


def parse(src: str) -> Program:
    """Parse Flow source into an AST. Raises ParseError on failure."""
    src_for_caret = src.splitlines()
    src_pp = _preprocess_triple_strings(src)
    lines = _split_lines(src_pp)
    try:
        return _Parser(lines).parse_program()
    except ParseError as e:
        raise e.with_source(src_for_caret)


def _preprocess_triple_strings(src: str) -> str:
    # Convert triple-quoted strings (possibly multi-line, possibly f-prefixed)
    # into single-line equivalents by escaping internal newlines as `\n`.
    # Newlines that were inside the string are added back as blank lines AFTER
    # the string so subsequent line numbers stay accurate.
    out: List[str] = []
    i = 0
    in_string = False  # are we inside a regular `"..."` (not triple)
    while i < len(src):
        c = src[i]
        if in_string:
            out.append(c)
            if c == "\\" and i + 1 < len(src):
                out.append(src[i + 1])
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
            continue
        # Detect `"""` or `f"""` triple-quote opener.
        is_f = src[i:i + 4] == 'f"""'
        if is_f or src[i:i + 3] == '"""':
            start = i + (4 if is_f else 3)
            close = src.find('"""', start)
            if close < 0:
                # Unterminated — leave as-is; tokenizer will report.
                out.append(c)
                i += 1
                continue
            content = src[start:close]
            n_nl = content.count("\n")
            esc = (content.replace("\\", "\\\\")
                          .replace('"', '\\"')
                          .replace("\n", "\\n"))
            out.append(("f" if is_f else "") + '"' + esc + '"')
            out.append("\n" * n_nl)   # preserve line numbering downstream
            i = close + 3
            continue
        if c == '"':
            in_string = True
        out.append(c)
        i += 1
    return "".join(out)


def ast_to_dict(node: Any) -> Any:
    """Convert AST nodes (and lists / primitives) to plain dict for JSON export."""
    if isinstance(node, list):
        return [ast_to_dict(x) for x in node]
    if hasattr(node, "__dataclass_fields__"):
        return {k: ast_to_dict(v) for k, v in asdict(node).items()}
    return node
