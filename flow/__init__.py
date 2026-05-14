"""Flow — a tiny verb-first DSL designed for small local LLMs."""

from .parser import parse, ParseError, ast_to_dict

try:
    from .compiler import compile_to, CompileError
    from .verbs import VERBS, register_verb
    from .formatter import format_source
except ImportError:  # compiler/verbs/formatter not yet present during incremental build
    pass

__version__ = "0.1.0"
__all__ = [
    "parse",
    "ParseError",
    "ast_to_dict",
    "compile_to",
    "CompileError",
    "VERBS",
    "register_verb",
]
