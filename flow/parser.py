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

Value = Union["StringLit", "NumberLit", "BoolLit", "Name", "FuncCall", "BinOp", "ListLit", "DictLit"]
Stmt = Union["Call", "IfStmt", "EachStmt", "RepeatStmt", "WhenStmt"]


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
class ListLit:
    items: List[Value]
    kind: str = "list"


@dataclass
class DictLit:
    entries: List[Tuple[str, Value]]   # (key, value) pairs
    kind: str = "dict"


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
    kind: str = "each"


@dataclass
class RepeatStmt:
    count: Value
    body: List[Stmt]
    line: int
    kind: str = "repeat"


@dataclass
class WhenStmt:
    event: str
    args: List[Value]
    body: List[Stmt]
    line: int
    kind: str = "when"


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

KEYWORDS = {"if", "else", "each", "in", "repeat", "when", "and", "or", "not", "true", "false"}

# Order matters: longer / more-specific patterns first.
TOKEN_SPEC = [
    ("STRING", r'"(?:\\.|[^"\\\n])*"'),
    ("ARROW", r"->"),
    ("CMP", r">=|<=|==|!="),
    ("OP", r"[=<>+\-*/]"),
    ("NUMBER", r"\d+(?:\.\d+)?"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("LBRACK", r"\["),
    ("RBRACK", r"\]"),
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("COLON", r":"),
    ("COMMA", r","),
    # Bareword: starts with letter/underscore. May include . and -
    # (e.g. data.csv, my-file.txt). URLs and paths with / or : must be quoted.
    ("WORD", r"[A-Za-z_][A-Za-z0-9_.\-]*"),
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
    """Return non-empty, non-comment lines with indent + tokens."""
    out: List[_Line] = []
    for i, raw in enumerate(src.splitlines(), start=1):
        # strip trailing comment
        code = _strip_comment(raw)
        if not code.strip():
            continue
        # measure indent (count leading spaces; tabs forbidden for simplicity)
        if "\t" in code[: len(code) - len(code.lstrip())]:
            raise ParseError("tabs not allowed for indentation; use 2 spaces", i, 1)
        indent_spaces = len(code) - len(code.lstrip(" "))
        if indent_spaces % 2 != 0:
            raise ParseError(f"indent must be a multiple of 2 (got {indent_spaces})", i, 1)
        indent = indent_spaces // 2
        tokens = _tokenize_line(code[indent_spaces:], i)
        if not tokens:
            continue
        out.append(_Line(indent, tokens, i, raw))
    return out


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
        if first.kind == "KW_WHEN":
            return self._parse_when(base_indent)
        if first.kind == "KW_ELSE":
            raise ParseError("'else' without matching 'if'", line.line_no, 1)
        if first.kind in ("WORD",):
            return self._parse_call(line)
        raise ParseError(
            f"expected verb or keyword, got {first.value!r}", first.line, first.col
        )

    # ---------- call (action block) ----------

    def _parse_call(self, line: _Line) -> Call:
        toks = line.tokens
        if toks[0].kind != "WORD":
            raise ParseError(f"verb must be a name, got {toks[0].value!r}", toks[0].line, toks[0].col)
        verb = toks[0].value
        if not _is_ident(verb):
            raise ParseError(f"verb name must be a plain identifier (got {verb!r})", toks[0].line, toks[0].col)
        i = 1
        args: List[Arg] = []
        out: Optional[str] = None
        # parse args and optional -> name
        while i < len(toks):
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
                if i + 2 < len(toks):
                    extra = toks[i + 2]
                    raise ParseError(
                        f"unexpected token after output name: {extra.value!r}",
                        extra.line, extra.col,
                    )
                break
            # parse arg: NAME '=' VALUE
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
            i += 2  # skip name and '='
            value, i = self._parse_value(toks, i, line.line_no)
            args.append(Arg(arg_name, value))
        self.idx += 1
        return Call(verb=verb, args=args, out=out, line=line.line_no)

    # ---------- value / expression ----------

    def _parse_value(self, toks: List[Token], i: int, line_no: int) -> Tuple[Value, int]:
        """Parse a single value starting at toks[i]. Returns (value, new_i)."""
        if i >= len(toks):
            raise ParseError("expected a value", line_no, 1)
        t = toks[i]
        if t.kind == "STRING":
            return StringLit(t.value), i + 1
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
        if t.kind == "WORD":
            # Could be: ident, ident.ident.ident, ident(args), or filename-like
            # If followed by '(' → FuncCall
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
            val, i = self._parse_value(toks, i, line_no)
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
            val, i = self._parse_value(toks, i, line_no)
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
            val, i = self._parse_value(toks, i, line_no)
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

    def _parse_expr(self, toks: List[Token], i: int, line_no: int) -> Tuple[Value, int]:
        """Parse a (possibly binary) expression. Used for 'if' conditions."""
        left, i = self._parse_value(toks, i, line_no)
        while i < len(toks) and toks[i].kind in ("CMP", "OP", "KW_AND", "KW_OR"):
            op_tok = toks[i]
            if op_tok.kind == "OP" and op_tok.value == "=":
                raise ParseError("did you mean '==' for comparison?", op_tok.line, op_tok.col)
            op = op_tok.value if op_tok.kind in ("CMP", "OP") else op_tok.value.lower()
            i += 1
            right, i = self._parse_value(toks, i, line_no)
            left = BinOp(op, left, right)
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
            raise ParseError("'each' needs the form: each <name> in <value>", line.line_no, 1)
        if toks[1].kind != "WORD" or not _is_ident(toks[1].value):
            raise ParseError(f"expected variable name after 'each', got {toks[1].value!r}",
                             toks[1].line, toks[1].col)
        if toks[2].kind != "KW_IN":
            raise ParseError(f"expected 'in' after each-variable, got {toks[2].value!r}",
                             toks[2].line, toks[2].col)
        var = toks[1].value
        iterable, i = self._parse_value(toks, 3, line.line_no)
        if i != len(toks):
            raise ParseError(f"unexpected token after iterable: {toks[i].value!r}",
                             toks[i].line, toks[i].col)
        self.idx += 1
        body, _ = self._parse_block(base_indent + 1)
        return EachStmt(var, iterable, body, line.line_no)

    def _parse_repeat(self, base_indent: int) -> RepeatStmt:
        line = self.lines[self.idx]
        toks = line.tokens
        if len(toks) < 2:
            raise ParseError("'repeat' needs a count: repeat <number-or-name>", line.line_no, 1)
        count, i = self._parse_value(toks, 1, line.line_no)
        if i != len(toks):
            raise ParseError(f"unexpected token after repeat count: {toks[i].value!r}",
                             toks[i].line, toks[i].col)
        self.idx += 1
        body, _ = self._parse_block(base_indent + 1)
        return RepeatStmt(count, body, line.line_no)

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
            val, i = self._parse_value(toks, i, line.line_no)
            args.append(val)
        self.idx += 1
        body, _ = self._parse_block(base_indent + 1)
        return WhenStmt(event, args, body, line.line_no)


# ============================================================
# Helpers
# ============================================================


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_ident(s: str) -> bool:
    return bool(_IDENT_RE.match(s))


def _split_word_path(w: str) -> List[str]:
    """Split a WORD token into dotted parts. Filenames like 'data.csv' → ['data','csv']."""
    return w.split(".")


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
