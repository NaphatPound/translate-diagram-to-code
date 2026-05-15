"""
Flow compiler — AST → target language source.

Currently supports: python, js.

Compilation strategy:
  1. Validate verbs against registry.
  2. Track variable scope (which `-> name` bindings are live).
  3. Render Value nodes into target-language expressions
     (using scope to distinguish variable refs from string-literal barewords).
  4. Apply each verb's template, substituting rendered args.
  5. Emit control flow (if/each/repeat/when) natively per target.
"""

from __future__ import annotations

import json
import re
from typing import List, Set

from .parser import (
    Program, Call, AssignStmt, MultiAssignStmt, IfStmt, EachStmt, RepeatStmt, WhileStmt, WhenStmt, TryStmt,
    BreakStmt, ContinueStmt, DefStmt, ReturnStmt, ExprStmt, MatchStmt,
    StringLit, NumberLit, BoolLit, Name, FuncCall, BinOp, UnaryOp, Arg,
    ListLit, DictLit, Ternary, Range, FString, MethodCall, IndexAccess, Spread,
)

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
from .verbs import VERBS, VerbSpec


class CompileError(Exception):
    def __init__(self, msg: str, line: int = 0):
        self.msg = msg
        self.line = line
        loc = f"line {line}: " if line else ""
        super().__init__(f"{loc}{msg}")


SUPPORTED_LANGS = ("python", "js", "go", "rust", "bash")


# ============================================================
# Entry point
# ============================================================


def compile_to(program: Program, lang: str) -> str:
    if lang not in SUPPORTED_LANGS:
        raise CompileError(f"unsupported target language: {lang!r} "
                           f"(supported: {', '.join(SUPPORTED_LANGS)})")
    c = _Compiler(lang)
    c.emit_program(program)
    src = c.source()
    if lang == "python":
        src = _hoist_python_imports(src)
    elif lang == "js":
        src = _hoist_js_requires(src)
    return src


_PY_INLINE_IMPORT_RE = re.compile(
    r"^(import\s+[A-Za-z_][\w.]*(?:\s+as\s+[A-Za-z_]\w*)?)\s*;\s*(.*)$"
)


_JS_REQUIRE_RE = re.compile(r"require\((['\"])([A-Za-z_][\w-]*)\1\)")


def _hoist_js_requires(src: str) -> str:
    """Replace every `require('X')` with a hoisted `_X` const so the module
    is required only once at the top of the file."""
    found: list = []
    seen: set = set()

    def _safe(name: str) -> str:
        return "_" + name.replace("-", "_")

    def _sub(match):
        mod = match.group(2)
        if mod not in seen:
            seen.add(mod)
            found.append(mod)
        return _safe(mod)

    new_src = _JS_REQUIRE_RE.sub(_sub, src)
    if not found:
        return src
    header = "\n".join(
        f"const {_safe(m)} = require('{m}');" for m in found
    ) + "\n\n"
    return header + new_src


def _hoist_python_imports(src: str) -> str:
    """Pull `import X as _Y;` prefixes off each line, dedupe, and put a single
    block of imports at the top of the file."""
    imports: list = []
    seen: set = set()
    out_lines: list = []
    for line in src.split("\n"):
        stripped = line.lstrip(" ")
        indent = line[:len(line) - len(stripped)]
        while True:
            m = _PY_INLINE_IMPORT_RE.match(stripped)
            if not m:
                break
            imp = m.group(1)
            if imp not in seen:
                seen.add(imp)
                imports.append(imp)
            stripped = m.group(2)
        out_lines.append(indent + stripped if stripped else "")
    header = ("\n".join(imports) + "\n\n") if imports else ""
    return header + "\n".join(out_lines)


# ============================================================
# Compiler
# ============================================================


class _Compiler:
    def __init__(self, lang: str):
        self.lang = lang
        self.lines: List[str] = []
        self.scope: Set[str] = set()
        self.indent = 0
        if lang == "python":
            self.indent_str = "    "
        elif lang == "bash":
            self.indent_str = "  "
        else:  # js, go, rust
            self.indent_str = "  "

    def source(self) -> str:
        return "\n".join(self.lines) + ("\n" if self.lines else "")

    # ---------- writers ----------

    def _emit(self, line: str) -> None:
        self.lines.append(self.indent_str * self.indent + line)

    def _block_open(self) -> None:
        self.indent += 1

    def _block_close(self, py_pass_if_empty: bool = True) -> None:
        # If we just opened a Python block and emitted nothing, drop a 'pass'.
        if self.lang == "python" and py_pass_if_empty:
            i_prefix = self.indent_str * self.indent
            if not self.lines or not self.lines[-1].startswith(i_prefix):
                self._emit("pass")
        self.indent -= 1
        if self.lang in ("js", "go", "rust"):
            self._emit("}")
        # Bash block-close handled in emit_if / emit_each / emit_repeat directly.

    # ---------- program ----------

    def emit_program(self, program: Program) -> None:
        for stmt in program.body:
            self.emit_stmt(stmt)

    # ---------- statements ----------

    def emit_stmt(self, stmt) -> None:
        if isinstance(stmt, Call):
            self.emit_call(stmt)
        elif isinstance(stmt, AssignStmt):
            self.emit_assign(stmt)
        elif isinstance(stmt, MultiAssignStmt):
            self.emit_multi_assign(stmt)
        elif isinstance(stmt, IfStmt):
            self.emit_if(stmt)
        elif isinstance(stmt, EachStmt):
            self.emit_each(stmt)
        elif isinstance(stmt, RepeatStmt):
            self.emit_repeat(stmt)
        elif isinstance(stmt, WhileStmt):
            self.emit_while(stmt)
        elif isinstance(stmt, WhenStmt):
            self.emit_when(stmt)
        elif isinstance(stmt, TryStmt):
            self.emit_try(stmt)
        elif isinstance(stmt, BreakStmt):
            self._emit("break" if self.lang != "bash" else "break")
        elif isinstance(stmt, ContinueStmt):
            self._emit("continue" if self.lang != "bash" else "continue")
        elif isinstance(stmt, DefStmt):
            self.emit_def(stmt)
        elif isinstance(stmt, ReturnStmt):
            self.emit_return(stmt)
        elif isinstance(stmt, ExprStmt):
            self.emit_expr_stmt(stmt)
        elif isinstance(stmt, MatchStmt):
            self.emit_match(stmt)
        else:
            raise CompileError(f"unknown statement type: {type(stmt).__name__}")

    def emit_match(self, stmt: MatchStmt) -> None:
        """Lower `match value` to a chained if/elif/else over `== pattern`."""
        val_src = self._render_value(stmt.value)
        tmp = self._fresh_temp("_m")
        # Emit the cache assignment.
        if self.lang == "python":
            self._emit(f"{tmp} = {val_src}")
        elif self.lang == "js":
            self._emit(f"const {tmp} = {val_src};")
        elif self.lang == "go":
            self._emit(f"var {tmp} = {val_src}")
        elif self.lang == "rust":
            self._emit(f"let {tmp} = {val_src};")
        elif self.lang == "bash":
            self._emit(f"{tmp}={val_src}")
        self.scope.add(tmp)
        # Lang-specific helpers for `if cond` / `else if cond` / `else` / close.
        is_c_like = self.lang in ("js", "go", "rust")
        for idx, (pat, body) in enumerate(stmt.cases):
            pat_src = self._render_value(pat)
            if self.lang == "python":
                kw = "if" if idx == 0 else "elif"
                self._emit(f"{kw} {tmp} == {pat_src}:")
                self._block_open()
                for s in body:
                    self.emit_stmt(s)
                self._block_close()
            elif self.lang == "bash":
                kw = "if" if idx == 0 else "elif"
                self._emit(f"{kw} [[ ${tmp.lstrip('$')} == {pat_src} ]]; then")
                self._block_open()
                for s in body:
                    self.emit_stmt(s)
                self.indent -= 1
            elif is_c_like:
                head = (f"if ({tmp} == {pat_src}) {{" if idx == 0
                        else f"}} else if ({tmp} == {pat_src}) {{")
                self._emit(head)
                self._block_open()
                for s in body:
                    self.emit_stmt(s)
                self.indent -= 1
        # Trailing else / close.
        if stmt.else_body:
            if self.lang == "python":
                self._emit("else:")
                self._block_open()
                for s in stmt.else_body:
                    self.emit_stmt(s)
                self._block_close()
            elif self.lang == "bash":
                self._emit("else")
                self._block_open()
                for s in stmt.else_body:
                    self.emit_stmt(s)
                self.indent -= 1
            elif is_c_like:
                self._emit("} else {")
                self._block_open()
                for s in stmt.else_body:
                    self.emit_stmt(s)
                self._block_close()
        else:
            if is_c_like:
                self._emit("}")
        if self.lang == "bash":
            self._emit("fi")
        self.scope.discard(tmp)

    def _fresh_temp(self, prefix: str = "_t") -> str:
        n = getattr(self, "_temp_counter", 0) + 1
        self._temp_counter = n
        return f"{prefix}{n}"

    def emit_def(self, stmt: DefStmt) -> None:
        # Render `name` or `name=default` per language.
        def _param_src(name: str, default: Optional[object], lang: str) -> str:
            if default is None:
                return name
            ds = self._render_value(default)
            if lang in ("python", "js"):
                return f"{name}={ds}" if lang == "python" else f"{name} = {ds}"
            # Go/Rust: no defaults; ignore (caller must pass).
            return name
        if self.lang == "python":
            ps = ", ".join(_param_src(n, d, "python") for n, d in stmt.params)
            self._emit(f"def {stmt.name}({ps}):")
        elif self.lang == "js":
            ps = ", ".join(_param_src(n, d, "js") for n, d in stmt.params)
            self._emit(f"function {stmt.name}({ps}) {{")
        elif self.lang == "go":
            params_typed = ", ".join(f"{n} any" for n, _ in stmt.params)
            self._emit(f"func {stmt.name}({params_typed}) any {{")
        elif self.lang == "rust":
            params_typed = ", ".join(f"{n}: impl std::fmt::Debug" for n, _ in stmt.params)
            self._emit(f"fn {stmt.name}({params_typed}) {{")
        elif self.lang == "bash":
            self._emit(f"{stmt.name}() {{")
            # Bash positional args are $1, $2, ...; bind each to the param name,
            # honoring defaults via ${1:-default} syntax.
            for j, (pn, pd) in enumerate(stmt.params, start=1):
                self._block_open()
                if pd is None:
                    self._emit(f"local {pn}=\"${j}\"")
                else:
                    ds = self._render_value(pd)
                    self._emit(f"local {pn}=\"${{{j}:-{ds}}}\"")
                self.indent -= 1
        # Bring params into scope while we emit the body.
        prev_scope = self.scope.copy()
        for pn, _ in stmt.params:
            self.scope.add(pn)
        # Also register the function name so calls to it parse as variable refs
        # rather than string-literal barewords.
        self.scope.add(stmt.name)
        # Implicit return at compile time: if the last stmt of the body is a
        # bare expression (ExprStmt), wrap it as a return. Recursively descend
        # into IfStmt branches and MatchStmt cases so each terminal branch's
        # last bare expression becomes a return too.
        body = list(stmt.body)
        if body:
            body[-1] = _wrap_terminal_returns(body[-1])
        self._block_open()
        for s in body:
            self.emit_stmt(s)
        # Restore scope (but keep the function name).
        self.scope = prev_scope
        self.scope.add(stmt.name)
        if self.lang == "bash":
            self.indent -= 1
            self._emit("}")
        else:
            self._block_close()

    def emit_return(self, stmt: ReturnStmt) -> None:
        if stmt.value is None:
            self._emit("return" if self.lang != "bash" else "return")
            return
        src = self._render_value(stmt.value)
        if self.lang == "bash":
            self._emit(f"echo {src}; return")
        else:
            self._emit(f"return {src}" + (";" if self.lang in ("js", "go", "rust") else ""))

    def emit_expr_stmt(self, stmt: ExprStmt) -> None:
        src = self._render_value(stmt.value)
        if self.lang in ("js", "rust"):
            self._emit(f"{src};")
        elif self.lang == "go":
            self._emit(f"_ = {src}")
        else:
            self._emit(src)

    def emit_try(self, stmt: TryStmt) -> None:
        var = stmt.catch_var or "_e"
        if self.lang == "python":
            self._emit("try:")
            self._block_open()
            for s in stmt.try_body:
                self.emit_stmt(s)
            self._block_close()
            self._emit(f"except Exception as {var}:")
            prev = var in self.scope
            self.scope.add(var)
            self._block_open()
            for s in stmt.catch_body:
                self.emit_stmt(s)
            if not prev:
                self.scope.discard(var)
            self._block_close()
            return
        if self.lang == "js":
            self._emit("try {")
            self._block_open()
            for s in stmt.try_body:
                self.emit_stmt(s)
            self.indent -= 1
            self._emit(f"}} catch ({var}) {{")
            self._block_open()
            prev = var in self.scope
            self.scope.add(var)
            for s in stmt.catch_body:
                self.emit_stmt(s)
            if not prev:
                self.scope.discard(var)
            self._block_close()
            return
        if self.lang == "rust":
            # Rust uses Result; emit a best-effort match against a closure.
            self._emit(f"match (|| -> Result<_, Box<dyn std::error::Error>> {{")
            self._block_open()
            for s in stmt.try_body:
                self.emit_stmt(s)
            self._emit("Ok(())")
            self.indent -= 1
            self._emit(f"}})() {{")
            self._block_open()
            self._emit(f"Ok(_) => {{}},")
            self._emit(f"Err({var}) => {{")
            self._block_open()
            prev = var in self.scope
            self.scope.add(var)
            for s in stmt.catch_body:
                self.emit_stmt(s)
            if not prev:
                self.scope.discard(var)
            self.indent -= 1
            self._emit("}}")
            self.indent -= 1
            self._emit("};")
            return
        if self.lang == "go":
            # Go: defer+recover() approximates a catch block.
            self._emit("func() {")
            self._block_open()
            self._emit("defer func() {")
            self._block_open()
            self._emit(f"if {var} := recover(); {var} != nil {{")
            self._block_open()
            prev = var in self.scope
            self.scope.add(var)
            for s in stmt.catch_body:
                self.emit_stmt(s)
            if not prev:
                self.scope.discard(var)
            self._block_close()  # if
            self._block_close()  # func()
            self._emit("}()")
            for s in stmt.try_body:
                self.emit_stmt(s)
            self._block_close()  # outer func
            self._emit("}()")
            return
        if self.lang == "bash":
            # Bash: run the try-body in a subshell; on failure run catch.
            self._emit("if ( ")
            self._block_open()
            for s in stmt.try_body:
                self.emit_stmt(s)
            self.indent -= 1
            self._emit(") ; then :; else")
            self._block_open()
            for s in stmt.catch_body:
                self.emit_stmt(s)
            self.indent -= 1
            self._emit("fi")
            return
        raise CompileError(f"try/catch not supported for lang {self.lang!r}")

    def emit_multi_assign(self, stmt: MultiAssignStmt) -> None:
        val_src = self._render_value(stmt.value)
        # All targets enter scope.
        for t in stmt.targets:
            self.scope.add(t)
        targets_str = ", ".join(stmt.targets)
        if self.lang == "python":
            self._emit(f"{targets_str} = {val_src}")
        elif self.lang == "js":
            self._emit(f"let [{targets_str}] = {val_src};")
        elif self.lang == "rust":
            self._emit(f"let ({targets_str}) = {val_src};")
        elif self.lang == "go":
            # Go can do this only via type assertion; emit a best-effort comment + index.
            self._emit(f"// multi-assign: {targets_str} = {val_src}")
            for i, t in enumerate(stmt.targets):
                self._emit(f"var {t} = ({val_src})[{i}]")
        elif self.lang == "bash":
            # Bash array indexing.
            for i, t in enumerate(stmt.targets):
                self._emit(f"{t}=\"${{{val_src.lstrip('$')}[{i}]}}\"")

    def emit_assign(self, stmt: AssignStmt) -> None:
        val_src = self._render_value(stmt.value)
        # Bring target into scope BEFORE rendering, so a self-reference (rare,
        # but `n = n + 1` style) resolves to the variable.
        self.scope.add(stmt.target)
        if self.lang == "python":
            self._emit(f"{stmt.target} = {val_src}")
        elif self.lang == "js":
            # Use `let` so reassignment works; user can write multiple `n = ...`
            # lines targeting the same variable.
            self._emit(f"let {stmt.target} = {val_src};")
        elif self.lang == "go":
            self._emit(f"var {stmt.target} = {val_src}")
        elif self.lang == "rust":
            self._emit(f"let {stmt.target} = {val_src};")
        elif self.lang == "bash":
            # Bash: numeric expressions go through (( )); strings use plain =.
            if isinstance(stmt.value, (NumberLit, BinOp)):
                self._emit(f"{stmt.target}=$(( {val_src} ))")
            else:
                self._emit(f"{stmt.target}={val_src}")

    def emit_call(self, call: Call) -> None:
        spec = VERBS.get(call.verb)
        if spec is None:
            suggestion = _did_you_mean(call.verb, VERBS.keys())
            extra = f". Did you mean {suggestion!r}?" if suggestion else ""
            raise CompileError(
                f"unknown verb {call.verb!r} (not in registry){extra}", call.line,
            )

        # Resolve implicit args (<pos> from positional, <pipe> from pipeline)
        # to the verb's declared primary_arg.
        resolved_args: List[Arg] = []
        for a in call.args:
            if a.name in ("<pos>", "<pipe>"):
                if not spec.primary_arg:
                    label = "positional value" if a.name == "<pos>" else "pipe input"
                    raise CompileError(
                        f"verb {call.verb!r} does not accept a {label} "
                        f"(no primary arg declared); use named args.",
                        call.line,
                    )
                resolved_args.append(Arg(spec.primary_arg, a.value))
            else:
                resolved_args.append(a)
        call.args = resolved_args

        # Validate args
        seen: Set[str] = set()
        for a in call.args:
            if a.name in seen:
                raise CompileError(
                    f"duplicate arg {a.name!r} on verb {call.verb!r}",
                    call.line,
                )
            seen.add(a.name)
            if a.name not in spec.args:
                allowed = ", ".join(sorted(spec.args.keys())) or "(none)"
                suggestion = _did_you_mean(a.name, spec.args.keys())
                hint = f". Did you mean {suggestion!r}?" if suggestion else ""
                raise CompileError(
                    f"verb {call.verb!r} doesn't accept arg {a.name!r}{hint}. "
                    f"Allowed: {allowed}",
                    call.line,
                )

        if call.out and not spec.returns:
            raise CompileError(
                f"verb {call.verb!r} does not return a value but '-> {call.out}' was given",
                call.line,
            )

        template = spec.templates.get(self.lang)
        if template is None:
            raise CompileError(
                f"verb {call.verb!r} has no template for {self.lang!r}",
                call.line,
            )

        # Render args to target-language expressions.
        # If an arg is listed in raw_args and its value is a StringLit, embed raw.
        ctx: dict = {}
        for name, arg in _pairs_by_name(call.args):
            if name in spec.raw_args and isinstance(arg.value, StringLit):
                ctx[name] = arg.value.value
            else:
                ctx[name] = self._render_value(arg.value)
        # Fill missing args (declared but not provided) with safe defaults
        for arg_name in spec.args:
            if arg_name not in ctx:
                ctx[arg_name] = "None" if self.lang == "python" else "null"
        ctx["out"] = call.out or "_"

        try:
            rendered = template.format(**ctx)
        except KeyError as e:
            raise CompileError(
                f"template for {call.verb!r} references missing arg {e}",
                call.line,
            )

        self._emit(rendered)

        if call.out:
            self.scope.add(call.out)

    def emit_if(self, stmt: IfStmt) -> None:
        cond_src = self._render_value(stmt.cond)
        if self.lang == "python":
            self._emit(f"if {cond_src}:")
            self._block_open()
            for s in stmt.then:
                self.emit_stmt(s)
            self._block_close()
            if stmt.else_:
                self._emit("else:")
                self._block_open()
                for s in stmt.else_:
                    self.emit_stmt(s)
                self._block_close()
            return
        if self.lang == "bash":
            self._emit(f"if (( {cond_src} )); then")
            self._block_open()
            for s in stmt.then:
                self.emit_stmt(s)
            self.indent -= 1
            if stmt.else_:
                self._emit("else")
                self._block_open()
                for s in stmt.else_:
                    self.emit_stmt(s)
                self.indent -= 1
            self._emit("fi")
            return
        # js, go, rust — use idiomatic `} else {`
        open_line = (f"if {cond_src} {{" if self.lang in ("go", "rust")
                     else f"if ({cond_src}) {{")
        self._emit(open_line)
        self._block_open()
        for s in stmt.then:
            self.emit_stmt(s)
        self.indent -= 1
        if stmt.else_:
            self._emit("} else {")
            self._block_open()
            for s in stmt.else_:
                self.emit_stmt(s)
            self._block_close()
        else:
            self._emit("}")

    def emit_each(self, stmt: EachStmt) -> None:
        iter_src = self._render_value(stmt.iterable)
        kv = stmt.key_var is not None
        if self.lang == "python":
            if kv:
                self._emit(f"for {stmt.key_var}, {stmt.var} in ({iter_src}).items():")
            else:
                self._emit(f"for {stmt.var} in {iter_src}:")
        elif self.lang == "js":
            if kv:
                self._emit(f"for (const [{stmt.key_var}, {stmt.var}] of Object.entries({iter_src})) {{")
            else:
                self._emit(f"for (const {stmt.var} of {iter_src}) {{")
        elif self.lang == "go":
            if kv:
                self._emit(f"for {stmt.key_var}, {stmt.var} := range {iter_src} {{")
            else:
                self._emit(f"for _, {stmt.var} := range {iter_src} {{")
        elif self.lang == "rust":
            if kv:
                self._emit(f"for ({stmt.key_var}, {stmt.var}) in ({iter_src}).iter() {{")
            else:
                self._emit(f"for {stmt.var} in {iter_src} {{")
        elif self.lang == "bash":
            # Bash assoc-array iteration is messy; emit a best-effort `for k in "${!d[@]}"`.
            if kv:
                self._emit(f"for {stmt.key_var} in \"${{!{iter_src.lstrip('$')}[@]}}\"; do {stmt.var}=\"${{{iter_src.lstrip('$')}[${stmt.key_var}]}}\"")
            else:
                self._emit(f"for {stmt.var} in \"${{{iter_src.lstrip('$')}[@]}}\"; do")
        previously = stmt.var in self.scope
        prev_k = stmt.key_var in self.scope if kv else False
        self.scope.add(stmt.var)
        if kv:
            self.scope.add(stmt.key_var)
        self._block_open()
        for s in stmt.body:
            self.emit_stmt(s)
        if not previously:
            self.scope.discard(stmt.var)
        if kv and not prev_k:
            self.scope.discard(stmt.key_var)
        if self.lang == "bash":
            self.indent -= 1
            self._emit("done")
        else:
            self._block_close()

    def emit_repeat(self, stmt: RepeatStmt) -> None:
        count_src = self._render_value(stmt.count)
        var_name = stmt.var or "_i"
        if self.lang == "python":
            if stmt.var:
                self._emit(f"for {var_name} in range(int({count_src})):")
            else:
                self._emit(f"for _ in range(int({count_src})):")
        elif self.lang == "js":
            self._emit(f"for (let {var_name} = 0; {var_name} < ({count_src}); {var_name}++) {{")
        elif self.lang == "go":
            self._emit(f"for {var_name} := 0; {var_name} < {count_src}; {var_name}++ {{")
        elif self.lang == "rust":
            self._emit(f"for {var_name} in 0..({count_src}) {{")
        elif self.lang == "bash":
            self._emit(f"for {var_name} in $(seq 0 $(({count_src} - 1))); do")
        # Bring the loop var into scope while we emit the body.
        prev_in_scope = var_name in self.scope
        if stmt.var:
            self.scope.add(stmt.var)
        self._block_open()
        for s in stmt.body:
            self.emit_stmt(s)
        if stmt.var and not prev_in_scope:
            self.scope.discard(stmt.var)
        if self.lang == "bash":
            self.indent -= 1
            self._emit("done")
        else:
            self._block_close()

    def emit_while(self, stmt: WhileStmt) -> None:
        cond = self._render_value(stmt.cond)
        if self.lang == "python":
            self._emit(f"while {cond}:")
        elif self.lang == "js":
            self._emit(f"while ({cond}) {{")
        elif self.lang == "go":
            self._emit(f"for {cond} {{")
        elif self.lang == "rust":
            self._emit(f"while {cond} {{")
        elif self.lang == "bash":
            self._emit(f"while (( {cond} )); do")
        self._block_open()
        for s in stmt.body:
            self.emit_stmt(s)
        if self.lang == "bash":
            self.indent -= 1
            self._emit("done")
        else:
            self._block_close()

    def emit_when(self, stmt: WhenStmt) -> None:
        # MVP: only `when start` is meaningful — treat as program entry.
        if stmt.event == "start":
            if self.lang == "python":
                self._emit("def main():")
                self._block_open()
                for s in stmt.body:
                    self.emit_stmt(s)
                self._block_close()
                self._emit('if __name__ == "__main__":')
                self._block_open()
                self._emit("main()")
                self._block_close(py_pass_if_empty=False)
            elif self.lang == "js":
                self._emit("async function main() {")
                self._block_open()
                for s in stmt.body:
                    self.emit_stmt(s)
                self._block_close()
                self._emit("main();")
            elif self.lang == "go":
                self._emit("func main() {")
                self._block_open()
                for s in stmt.body:
                    self.emit_stmt(s)
                self._block_close()
            elif self.lang == "rust":
                self._emit("fn main() {")
                self._block_open()
                for s in stmt.body:
                    self.emit_stmt(s)
                self._block_close()
            elif self.lang == "bash":
                self._emit("main() {")
                self._block_open()
                for s in stmt.body:
                    self.emit_stmt(s)
                self.indent -= 1
                self._emit("}")
                self._emit("main")
            return
        # Other events: emit a comment shell so a host can register.
        comment = "# " if self.lang in ("python", "bash") else "// "
        args = " ".join(self._render_value(a) for a in stmt.args)
        self._emit(f"{comment}event handler: when {stmt.event} {args}")
        if self.lang == "python":
            self._emit(f"def on_{stmt.event}():")
            self._block_open()
            for s in stmt.body:
                self.emit_stmt(s)
            self._block_close()
        elif self.lang == "js":
            self._emit(f"async function on_{stmt.event}() {{")
            self._block_open()
            for s in stmt.body:
                self.emit_stmt(s)
            self._block_close()
        elif self.lang in ("go", "rust"):
            kw = "func" if self.lang == "go" else "fn"
            self._emit(f"{kw} on_{stmt.event}() {{")
            self._block_open()
            for s in stmt.body:
                self.emit_stmt(s)
            self._block_close()
        elif self.lang == "bash":
            self._emit(f"on_{stmt.event}() {{")
            self._block_open()
            for s in stmt.body:
                self.emit_stmt(s)
            self.indent -= 1
            self._emit("}")

    # ---------- values ----------

    def _render_value(self, v) -> str:
        if isinstance(v, StringLit):
            return self._str_literal(v.value)
        if isinstance(v, NumberLit):
            return str(int(v.value)) if v.value.is_integer() else str(v.value)
        if isinstance(v, BoolLit):
            if self.lang == "python":
                return "True" if v.value else "False"
            if self.lang == "bash":
                return "1" if v.value else "0"
            return "true" if v.value else "false"
        if isinstance(v, ListLit):
            if self.lang == "bash":
                # Space-separated array literal
                bparts = " ".join(self._render_value(x) for x in v.items)
                return f"({bparts})"
            parts = ", ".join(self._render_value(x) for x in v.items)
            if self.lang == "go":   return f"[]any{{{parts}}}"
            if self.lang == "rust": return f"vec![{parts}]"
            return f"[{parts}]"
        if isinstance(v, DictLit):
            if self.lang in ("go",):
                # map[string]any{"k": v, ...}
                parts = ", ".join(
                    f"{self._str_literal(k)}: {self._render_value(val)}"
                    for k, val in v.entries
                )
                return f"map[string]any{{{parts}}}"
            if self.lang in ("rust", "bash"):
                raise CompileError(f"dict literals are not supported in {self.lang}")
            if self.lang == "python":
                parts = ", ".join(f"{self._str_literal(k)}: {self._render_value(val)}" for k, val in v.entries)
            else:  # js
                parts = ", ".join(
                    f"{k if _IDENT_RE.match(k) else self._str_literal(k)}: {self._render_value(val)}"
                    for k, val in v.entries
                )
            return "{" + parts + "}"
        if isinstance(v, Name):
            return self._render_name(v)
        if isinstance(v, FuncCall):
            return self._render_funccall(v)
        if isinstance(v, BinOp):
            l = self._render_value(v.left)
            r = self._render_value(v.right)
            op = v.op
            if op == "??":
                # Null-coalescing — render per target.
                if self.lang == "python":
                    return f"({l} if {l} is not None else {r})"
                if self.lang in ("js", "rust"):
                    return f"({l} ?? {r})"
                if self.lang == "go":
                    # No native; emit a function-call wrapper to keep semantics close.
                    return f"func() any {{ if v := {l}; v != nil {{ return v }}; return {r} }}()"
                if self.lang == "bash":
                    # ${var:-default} substitutes when unset/empty.
                    return f"${{{l.lstrip('$')}:-{r}}}"
                raise CompileError(f"`??` not supported for {self.lang!r}")
            if op == "and":
                op = "and" if self.lang == "python" else "&&"
            elif op == "or":
                op = "or"  if self.lang == "python" else "||"
            return f"({l} {op} {r})"
        if isinstance(v, UnaryOp):
            inner = self._render_value(v.value)
            if v.op == "not":
                if self.lang == "python":
                    return f"(not {inner})"
                if self.lang == "bash":
                    return f"(! {inner})"
                return f"(!{inner})"
            raise CompileError(f"unknown unary op {v.op!r}")
        if isinstance(v, Spread):
            inner = self._render_value(v.value)
            if self.lang in ("python", "js"):
                return f"*{inner}" if self.lang == "python" else f"...{inner}"
            if self.lang == "go":
                return f"{inner}..."
            if self.lang == "rust":
                # No general spread; emit `inner.iter().copied()` as a hint.
                return f"{inner}"
            if self.lang == "bash":
                return f"\"${{{inner.lstrip('$')}[@]}}\""
            return inner
        if isinstance(v, Ternary):
            cond = self._render_value(v.cond)
            then = self._render_value(v.then)
            else_ = self._render_value(v.else_)
            if self.lang == "python":
                return f"({then} if {cond} else {else_})"
            if self.lang in ("js", "rust"):
                return f"({cond} ? {then} : {else_})"
            if self.lang == "go":
                # Go has no ternary — emit an IIFE.
                return f"func() any {{ if {cond} {{ return {then} }}; return {else_} }}()"
            if self.lang == "bash":
                # Bash: `[[ cond ]] && echo a || echo b` in a subshell.
                return f"$(if (( {cond} )); then echo {then}; else echo {else_}; fi)"
            raise CompileError(f"ternary not supported for lang {self.lang!r}")
        if isinstance(v, FString):
            return self._render_fstring(v)
        if isinstance(v, MethodCall):
            rec = self._render_value(v.receiver)
            args = (", ".join(self._render_value(a) for a in v.args)
                    if v.args is not None else None)
            if v.optional:
                # Null-safe access.
                if self.lang == "js":
                    if args is None:
                        return f"{rec}?.{v.method}"
                    return f"{rec}?.{v.method}({args})"
                if self.lang == "python":
                    if args is None:
                        # `?.member` → dict.get-style null-safe access. Works
                        # for dicts (most Flow data); on objects the user can
                        # use plain `.member` instead.
                        key = self._str_literal(v.method)
                        return f"({rec}.get({key}) if {rec} is not None else None)"
                    return f"({rec}.{v.method}({args}) if {rec} is not None else None)"
                # Go/Rust/Bash: best-effort fallback emits non-optional access
                # plus a comment marker. Document the limitation.
                if args is None:
                    return f"{rec}.{v.method} /* ?. */"
                return f"{rec}.{v.method}({args}) /* ?. */"
            if args is None:
                return f"{rec}.{v.method}"
            return f"{rec}.{v.method}({args})"
        if isinstance(v, IndexAccess):
            rec = self._render_value(v.receiver)
            # Slice: `s[a..b]` (inclusive). Detect Range index, emit native slice.
            if isinstance(v.index, Range):
                s_src = self._render_value(v.index.start)
                e_src = self._render_value(v.index.end)
                if self.lang == "python":
                    return f"{rec}[{s_src}:({e_src}) + 1]"
                if self.lang == "js":
                    return f"({rec}).slice({s_src}, ({e_src}) + 1)"
                if self.lang == "go":
                    return f"{rec}[{s_src}:({e_src}) + 1]"
                if self.lang == "rust":
                    return f"&{rec}[{s_src}..=({e_src})]"
                if self.lang == "bash":
                    # Bash slice: ${var:offset:length}
                    return f"${{{rec.lstrip('$')}:{s_src}:(({e_src}) - {s_src} + 1)}}"
            idx = self._render_value(v.index)
            if self.lang == "bash":
                return f"${{{rec.lstrip('$')}[{idx}]}}"
            return f"{rec}[{idx}]"
        if isinstance(v, Range):
            s = self._render_value(v.start)
            e = self._render_value(v.end)
            # Inclusive range: emit a concrete list per target lang.
            if self.lang == "python":
                return f"list(range({s}, {e} + 1))"
            if self.lang == "js":
                return f"Array.from({{length: ({e}) - ({s}) + 1}}, (_, i) => i + ({s}))"
            if self.lang == "go":
                return f"func() []int {{ var _r []int; for _i := {s}; _i <= {e}; _i++ {{ _r = append(_r, _i) }}; return _r }}()"
            if self.lang == "rust":
                return f"({s}..=({e})).collect::<Vec<_>>()"
            if self.lang == "bash":
                return f"$(seq {s} {e})"
            raise CompileError(f"range not supported for lang {self.lang!r}")
        raise CompileError(f"cannot render value of type {type(v).__name__}")

    def _render_name(self, n: Name) -> str:
        """
        A Name like ['rows'] or ['row', 'name'] or ['data', 'csv'].

        Rule:
          - If parts[0] is in scope → variable reference, dots = member access.
          - Otherwise → treat the whole thing as a string literal.
        """
        first = n.parts[0]
        if first in self.scope:
            if self.lang == "bash":
                # Bash: $var, member access not supported — render as $var only.
                return f"${first}"
            if self.lang == "go":
                # Go has no dict bracket access for `any`; emit a type-asserted
                # path that is best-effort. Pure variable refs work fine.
                if len(n.parts) == 1:
                    return first
                inner = first + "".join(f".(map[string]any)[{self._str_literal(p)}]" for p in n.parts[1:])
                return inner
            if self.lang == "rust":
                # Rust: same caveat — member access on opaque values is non-trivial.
                if len(n.parts) == 1:
                    return first
                return first + "".join(f'[{self._str_literal(p)}]' for p in n.parts[1:])
            # python, js
            if len(n.parts) == 1:
                return first
            return first + "".join(f"[{self._str_literal(p)}]" for p in n.parts[1:])
        # Not in scope — treat as string literal.
        return self._str_literal(".".join(n.parts))

    def _render_funccall(self, f: FuncCall) -> str:
        # Inline builtins; otherwise treat as host function (user-provided).
        BUILTINS = {
            "count": {"python": "len", "js": "(_x => _x.length)", "go": "len",     "rust": "<no-op>",   "bash": ""},
            "len":   {"python": "len", "js": "(_x => _x.length)", "go": "len",     "rust": "<no-op>",   "bash": ""},
            "min":   {"python": "min", "js": "Math.min",          "go": "<no-op>", "rust": "<no-op>",   "bash": ""},
            "max":   {"python": "max", "js": "Math.max",          "go": "<no-op>", "rust": "<no-op>",   "bash": ""},
            "abs":   {"python": "abs", "js": "Math.abs",          "go": "math.Abs", "rust": "<no-op>",  "bash": ""},
            "str":   {"python": "str", "js": "String",            "go": "fmt.Sprintf", "rust": "format!", "bash": ""},
            "int":   {"python": "int", "js": "parseInt",          "go": "int",     "rust": "i64::from", "bash": ""},
            "float": {"python": "float","js": "parseFloat",       "go": "float64", "rust": "f64::from", "bash": ""},
            "sum":   {"python": "sum", "js": "<no-op>",           "go": "<no-op>", "rust": "<no-op>",   "bash": ""},
            "round": {"python": "round","js": "Math.round",       "go": "<no-op>", "rust": "<no-op>",   "bash": ""},
            "sorted":{"python": "sorted","js": "<no-op>",         "go": "<no-op>", "rust": "<no-op>",   "bash": ""},
        }
        # Single-arg verb funccalls that need a template (not just a name swap).
        # Keyed by (lang, verb). The arg-rendered string is substituted in.
        VERB_TEMPLATES = {
            ("python", "reverse"): "list(reversed({0}))",
            ("python", "unique"):  "list(dict.fromkeys({0}))",
            ("python", "keys"):    "list(({0}).keys())",
            ("python", "values"):  "list(({0}).values())",
            ("python", "avg"):     "(sum({0}) / len({0}))",
            ("js",     "reverse"): "[...({0})].reverse()",
            ("js",     "unique"):  "[...new Set({0})]",
            ("js",     "keys"):    "Object.keys({0})",
            ("js",     "values"):  "Object.values({0})",
            ("js",     "sum"):     "({0}).reduce((a,b)=>a+b,0)",
            ("js",     "sorted"):  "[...({0})].sort()",
        }
        args = ", ".join(self._render_value(a) for a in f.args)
        tmpl = VERB_TEMPLATES.get((self.lang, f.name))
        if tmpl is not None and len(f.args) == 1:
            return tmpl.format(args)
        if f.name in BUILTINS and BUILTINS[f.name].get(self.lang, "<no-op>") != "<no-op>":
            return f"{BUILTINS[f.name][self.lang]}({args})"
        return f"{f.name}({args})"

    def _str_literal(self, s: str) -> str:
        return json.dumps(s, ensure_ascii=False)

    def _render_fstring(self, fs: FString) -> str:
        """Render `f"..."` per target language. Placeholder content is a
        parsed Flow expression: render it with `_render_value` so language-
        specific transformations (e.g. count → len, dotted name → dict
        access) apply consistently with the rest of the program."""
        if self.lang == "python":
            buf = []
            for kind, payload in fs.parts:
                if kind == "text":
                    buf.append(_escape_for_fstring(payload, "python"))
                else:
                    buf.append("{" + self._render_value(payload) + "}")
            return 'f"' + "".join(buf) + '"'
        if self.lang == "js":
            buf = []
            for kind, payload in fs.parts:
                if kind == "text":
                    buf.append(_escape_for_fstring(payload, "js"))
                else:
                    buf.append("${" + self._render_value(payload) + "}")
            return "`" + "".join(buf) + "`"
        if self.lang == "go":
            fmt = ""
            args = []
            for kind, payload in fs.parts:
                if kind == "text":
                    fmt += payload.replace("%", "%%")
                else:
                    fmt += "%v"
                    args.append(self._render_value(payload))
            quoted = json.dumps(fmt, ensure_ascii=False)
            if args:
                return f"fmt.Sprintf({quoted}, {', '.join(args)})"
            return quoted
        if self.lang == "rust":
            fmt = ""
            args = []
            for kind, payload in fs.parts:
                if kind == "text":
                    fmt += payload.replace("{", "{{").replace("}", "}}")
                else:
                    fmt += "{}"
                    args.append(self._render_value(payload))
            quoted = json.dumps(fmt, ensure_ascii=False)
            if args:
                return f"format!({quoted}, {', '.join(args)})"
            return quoted
        if self.lang == "bash":
            buf = []
            for kind, payload in fs.parts:
                if kind == "text":
                    buf.append(payload)
                else:
                    rendered = self._render_value(payload)
                    # If the rendered form already starts with `$`, embed
                    # bare. Otherwise wrap as `$(...)` so Bash expands the
                    # arithmetic/command expression.
                    if rendered.startswith("$"):
                        buf.append(rendered)
                    else:
                        buf.append("$(" + rendered + ")")
            return '"' + "".join(buf).replace('"', '\\"') + '"'
        raise CompileError(f"f-string not supported for lang {self.lang!r}")


def _pairs_by_name(args: List[Arg]):
    for a in args:
        yield a.name, a


def _wrap_terminal_returns(stmt):
    """If `stmt` is a bare ExprStmt, wrap as ReturnStmt. If it's an IfStmt
    or MatchStmt, recurse into each branch's LAST statement and apply the
    same wrap so implicit return works inside conditional / match arms.

    Also: a bare `Call(verb, [], None)` whose verb is NOT a registered Flow
    verb is reinterpreted as a variable reference and implicit-returned.
    This makes `def abs n: ... else: n` work — without it, `n` parses as a
    zero-arg verb call which the compiler rejects.
    """
    if isinstance(stmt, ExprStmt):
        return ReturnStmt(value=stmt.value, line=stmt.line)
    if isinstance(stmt, Call) and not stmt.args and stmt.out is None:
        if stmt.verb not in VERBS:
            return ReturnStmt(value=Name([stmt.verb]), line=stmt.line)
    if isinstance(stmt, IfStmt):
        if stmt.then:
            stmt.then[-1] = _wrap_terminal_returns(stmt.then[-1])
        if stmt.else_:
            stmt.else_[-1] = _wrap_terminal_returns(stmt.else_[-1])
        return stmt
    if isinstance(stmt, MatchStmt):
        new_cases = []
        for pat, body in stmt.cases:
            if body:
                body[-1] = _wrap_terminal_returns(body[-1])
            new_cases.append((pat, body))
        stmt.cases = new_cases
        if stmt.else_body:
            stmt.else_body[-1] = _wrap_terminal_returns(stmt.else_body[-1])
        return stmt
    return stmt


def _escape_for_fstring(text: str, lang: str) -> str:
    """Escape literal-text segments of an f-string for the target's own
    string/template syntax."""
    if lang == "python":
        # Python f-strings can't have raw newlines; convert to `\n`. Also
        # double `{` and `}`, escape `"` and `\`.
        return (text.replace("\\", "\\\\")
                    .replace('"', '\\"')
                    .replace("\n", "\\n")
                    .replace("\t", "\\t")
                    .replace("{", "{{")
                    .replace("}", "}}"))
    if lang == "js":
        # JS template literals allow embedded newlines, but escape ` and ${.
        return text.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    return text


def _did_you_mean(needle: str, candidates) -> str:
    """Return the closest candidate by edit distance, or '' if none is close."""
    needle = needle.lower()
    best, best_d = "", 99
    for c in candidates:
        d = _edit_distance(needle, c.lower())
        if d < best_d:
            best, best_d = c, d
    # Threshold: at most one third the length of the needle, min 1, max 4.
    threshold = max(1, min(4, len(needle) // 3 + 1))
    return best if best_d <= threshold else ""


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (small inputs, simple DP)."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]
