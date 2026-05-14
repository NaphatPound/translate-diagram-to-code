"""Tiny HTTP server for the Flow playground.

Endpoints
---------
GET  /                  → renderer/index.html (playground UI)
POST /api/parse         body: {"source": "..."}             → {"ok": true, "ast": {...}} | {"ok": false, "error": "..."}
POST /api/compile       body: {"source": "...", "lang": "python"|"js"}
                          → {"ok": true, "output": "..."}    | {"ok": false, "error": "..."}
GET  /api/health        → {"ok": true}

The server is intentionally minimal (stdlib only). It binds to 127.0.0.1 by
default so it isn't exposed to other machines.
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from . import parse, compile_to, ast_to_dict, ParseError, CompileError
from .verbs import VERBS
from .formatter import format_source
from .parser import (
    Call, Arg, StringLit, NumberLit, BoolLit, Name,
)


_RENDERER = Path(__file__).resolve().parent.parent / "renderer" / "index.html"
_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _list_examples():
    out = []
    if not _EXAMPLES_DIR.exists():
        return out
    for p in sorted(_EXAMPLES_DIR.glob("*.flow")):
        summary = ""
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("#"):
                    summary = line.lstrip("# ").strip()
                    break
                if line:
                    break
        except OSError:
            pass
        out.append({"name": p.stem, "summary": summary})
    return out


def _read_example(name: str) -> str:
    # Block path traversal: only allow alphanumeric, dash, underscore.
    if not name or not all(c.isalnum() or c in "-_" for c in name):
        raise ValueError(f"invalid example name {name!r}")
    p = _EXAMPLES_DIR / f"{name}.flow"
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(name)
    return p.read_text(encoding="utf-8")


class FlowHandler(BaseHTTPRequestHandler):
    server_version = "FlowPlayground/0.1"

    # --- helpers ---

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON: {e}")

    def log_message(self, fmt, *args):
        # Quieter logs — single line, no timestamp prefix.
        sys.stderr.write("[flow] " + (fmt % args) + "\n")

    # --- routing ---

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            try:
                html = _RENDERER.read_bytes()
            except FileNotFoundError:
                self._send_text(500, "text/plain", b"renderer/index.html not found")
                return
            self._send_text(200, "text/html; charset=utf-8", html)
            return
        if path == "/api/health":
            self._send_json(200, {"ok": True})
            return
        if path == "/api/verbs":
            verbs = [
                {
                    "name": v.name,
                    "category": v.category,
                    "summary": v.summary,
                    "args": v.args,
                    "returns": v.returns,
                    "raw_args": v.raw_args,
                }
                for v in VERBS.values()
            ]
            self._send_json(200, {"verbs": verbs})
            return
        if path == "/api/examples":
            self._send_json(200, {"examples": _list_examples()})
            return
        if path == "/api/example":
            from urllib.parse import parse_qs
            qs = parse_qs(urlparse(self.path).query)
            name = (qs.get("name") or [""])[0]
            try:
                content = _read_example(name)
            except (ValueError, FileNotFoundError) as e:
                self._send_json(404, {"ok": False, "error": str(e)})
                return
            self._send_text(200, "text/plain; charset=utf-8", content.encode("utf-8"))
            return
        self._send_text(404, "text/plain", b"not found")

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._read_json()
        except ValueError as e:
            self._send_json(400, {"ok": False, "error": str(e)})
            return

        if path == "/api/parse":
            source = body.get("source") or ""
            try:
                ast = parse(source)
            except ParseError as e:
                self._send_json(200, {"ok": False, "error": str(e)})
                return
            self._send_json(200, {"ok": True, "ast": ast_to_dict(ast)})
            return

        if path == "/api/compile":
            source = body.get("source") or ""
            lang = body.get("lang") or "python"
            try:
                ast = parse(source)
                output = compile_to(ast, lang)
            except (ParseError, CompileError) as e:
                self._send_json(200, {"ok": False, "error": str(e)})
                return
            self._send_json(200, {"ok": True, "output": output})
            return

        if path == "/api/format":
            source = body.get("source") or ""
            try:
                ast = parse(source)
            except ParseError as e:
                self._send_json(200, {"ok": False, "error": str(e)})
                return
            self._send_json(200, {"ok": True, "source": format_source(ast)})
            return

        if path == "/api/edit":
            source = body.get("source") or ""
            edits = body.get("edits") or []
            try:
                ast = parse(source)
                _apply_edits(ast, edits)
            except (ParseError, ValueError) as e:
                self._send_json(200, {"ok": False, "error": str(e)})
                return
            self._send_json(200, {"ok": True, "source": format_source(ast)})
            return

        self._send_text(404, "text/plain", b"not found")


def _apply_edits(ast, edits):
    """Apply a list of edits in place.

    Each edit shape: {"line": int, "args": {name: value}, "out": str | null}

    Values in `args` are raw JSON values; we coerce by type:
      - str → StringLit  (UNLESS the verb arg is raw or the user passes a Name-shape via "$name")
      - int/float → NumberLit
      - bool → BoolLit
    Names (variable refs) cannot be expressed in plain JSON unambiguously, so
    we use the convention: any string starting with "$" is treated as a Name
    on the part after the "$".
    """
    targets = {e["line"]: e for e in edits if "line" in e}
    if not targets:
        return

    def walk(stmts):
        for s in stmts:
            if isinstance(s, Call) and s.line in targets:
                _patch_call(s, targets[s.line])
            for child in _children(s):
                walk(child)

    walk(ast.body)


def _children(stmt):
    from .parser import IfStmt, EachStmt, RepeatStmt, WhenStmt
    if isinstance(stmt, IfStmt):
        yield stmt.then
        if stmt.else_:
            yield stmt.else_
    elif isinstance(stmt, (EachStmt, RepeatStmt, WhenStmt)):
        yield stmt.body


def _patch_call(call, edit):
    new_args_raw = edit.get("args") or {}
    spec = VERBS.get(call.verb)
    raw_args = set(spec.raw_args) if spec else set()
    new_args = []
    for k, v in new_args_raw.items():
        new_args.append(Arg(k, _coerce_value(v, k in raw_args)))
    call.args = new_args
    if "out" in edit:
        out = edit["out"]
        call.out = out if (isinstance(out, str) and out) else None


def _coerce_value(v, is_raw):
    if isinstance(v, bool):
        return BoolLit(v)
    if isinstance(v, (int, float)):
        return NumberLit(float(v))
    if isinstance(v, str):
        if v.startswith("$"):
            return Name(v[1:].split("."))
        return StringLit(v)
    raise ValueError(f"can't coerce value: {v!r}")


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), FlowHandler)
    url = f"http://{host}:{port}"
    print(f"Flow playground: {url}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", file=sys.stderr)
        server.shutdown()
