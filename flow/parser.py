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
Stmt = Union["Call", "AssignStmt", "MultiAssignStmt", "IfStmt", "EachStmt", "RepeatStmt", "WhileStmt", "WhenStmt", "TryStmt", "BreakStmt", "ContinueStmt", "DefStmt", "ReturnStmt", "ExprStmt"]


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
    kind: str = "range"


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


# ============================================================
# Tokenizer
# ============================================================

KEYWORDS = {"if", "else", "unless", "each", "in", "repeat", "while", "when", "as",
            "try", "catch", "break", "continue", "def", "return",
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
            # Compound assignment: `name += expr` → `name = name + expr`.
            if (len(toks) >= 3 and toks[1].kind == "COMPOUND"):
                return self._parse_compound_assign(line)
            # Funccall as statement: `name(args)` (NO space before `(`) — for
            # side-effect calls of user-defined functions. With a space,
            # `p (x)` reads as verb-with-positional-value instead.
            if (len(toks) >= 2 and toks[1].kind == "LPAREN"
                    and toks[1].col == toks[0].col + len(toks[0].value)):
                return self._parse_expr_stmt(line)
            return self._parse_pipeline(line)
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
        if i != len(toks):
            extra = toks[i]
            raise ParseError(
                f"unexpected token after multi-assign expression: {extra.value!r}",
                extra.line, extra.col,
            )
        self.idx += 1
        return MultiAssignStmt(targets=targets, value=value, line=line.line_no)

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
        IfStmt when a trailing `if <cond>` wraps the line."""
        toks = line.tokens
        calls: List[Call] = []
        i = 0
        pipe_in: Optional[str] = None
        postfix_if_cond: Optional[Value] = None
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
                idx_expr, i = self._parse_expr(toks, i, line_no)
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

    def _parse_list(self, toks: List[Token], i: int, line_no: int) -> Tuple[ListLit, int]:
        i += 1  # consume '['
        items: List[Value] = []
        if i < len(toks) and toks[i].kind == "RBRACK":
            return ListLit(items), i + 1
        while True:
            val, i = self._parse_value_maybe_spread(toks, i, line_no)
            items.append(val)
            if i >= len(toks):
                raise ParseError("expected ']' to close list", line_no, 1)
            if toks[i].kind == "COMMA":
                i += 1
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

    def _parse_dict(self, toks: List[Token], i: int, line_no: int) -> Tuple[DictLit, int]:
        i += 1  # consume '{'
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

    def _parse_funccall(self, toks: List[Token], i: int, line_no: int) -> Tuple[FuncCall, int]:
        name = toks[i].value
        if not _is_ident(name):
            raise ParseError(f"function name must be an identifier (got {name!r})",
                             toks[i].line, toks[i].col)
        i += 2  # skip name + '('
        args: List[Value] = []
        if i < len(toks) and toks[i].kind == "RPAREN":
            return FuncCall(name, args), i + 1
        while True:
            val, i = self._parse_value_maybe_spread(toks, i, line_no)
            args.append(val)
            if i >= len(toks):
                raise ParseError("expected ')' to close function call", line_no, 1)
            if toks[i].kind == "COMMA":
                i += 1
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

    def _parse_expr_stmt(self, line: _Line) -> ExprStmt:
        toks = line.tokens
        value, i = self._parse_expr(toks, 0, line.line_no)
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
    if tok.kind in ("CMP", "OP"):
        return tok.value
    return None


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
    """Split `hi {name}, age {age}` into [("text", "hi "), ("var", "name"),
    ("text", ", age "), ("var", "age")]. Escape `\\{` to keep a literal `{`."""
    parts: List[Tuple[str, str]] = []
    pos = 0
    while pos < len(content):
        m = _FSTRING_PLACEHOLDER.search(content, pos)
        if not m:
            parts.append(("text", content[pos:]))
            break
        if m.start() > pos:
            parts.append(("text", content[pos:m.start()]))
        parts.append(("var", m.group(1)))
        pos = m.end()
    # Filter out empty text parts for cleaner downstream handling.
    parts = [p for p in parts if not (p[0] == "text" and p[1] == "")]
    return FString(parts=parts)


# ============================================================
# Public API
# ============================================================


def parse(src: str) -> Program:
    """Parse Flow source into an AST. Raises ParseError on failure."""
    lines = _split_lines(src)
    return _Parser(lines).parse_program()


def ast_to_dict(node: Any) -> Any:
    """Convert AST nodes (and lists / primitives) to plain dict for JSON export."""
    if isinstance(node, list):
        return [ast_to_dict(x) for x in node]
    if hasattr(node, "__dataclass_fields__"):
        return {k: ast_to_dict(v) for k, v in asdict(node).items()}
    return node
