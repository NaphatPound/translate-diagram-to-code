"""
Verb registry for Flow.

Each verb declares:
  - category    (UI color / grouping)
  - returns     (True if it can appear on the RHS of `-> name`)
  - args        (allowed arg names + brief type hint)
  - templates   (per target language)

Templates use Python `.format(**ctx)` substitution. The context contains:
  - {out}    output variable name (empty string if no `-> name`)
  - {<arg>}  each declared arg, already rendered to a target-language expression

The compiler is responsible for rendering values into the target language
before substitution. See compiler.py.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class VerbSpec:
    name: str
    category: str
    summary: str
    args: Dict[str, str]            # arg_name -> "type/description"
    returns: bool = False           # supports `-> name`
    templates: Dict[str, str] = field(default_factory=dict)  # lang -> template
    raw_args: List[str] = field(default_factory=list)
    # ^ args whose string value is embedded raw (no quoting). Use for
    # predicates / expressions like 'where', 'by'. Inside these expressions,
    # the loop item is bound to `x`.
    primary_arg: str = ""
    # ^ The argument that receives the positional value (e.g. `print "hi"`
    # → primary_arg="value"). Also used as the pipe target: `... | upper`
    # feeds the upstream value into this arg. Leave empty to require all
    # args be named.


VERBS: Dict[str, VerbSpec] = {}


def register_verb(spec: VerbSpec) -> None:
    VERBS[spec.name] = spec


def _v(name, category, summary, args, returns=False, raw_args=None,
       primary=None, **templates):
    register_verb(VerbSpec(
        name, category, summary, args, returns, templates,
        raw_args or [], primary or "",
    ))


# ============================================================
# io  (gray)
# ============================================================

_v("read", "io", "Read a text file into a string",
   {"file": "path to file"},
   returns=True,
   python="{out} = open({file}).read()",
   js="const {out} = require('fs').readFileSync({file}, 'utf8');")

_v("write", "io", "Write a string to a file",
   {"file": "path", "text": "string content"},
   python="open({file}, 'w').write({text})",
   js="require('fs').writeFileSync({file}, {text});")

_v("print", "io", "Print a value to stdout",
   {"value": "any"},
   python="print({value})",
   js="console.log({value});")

_v("ask", "io", "Read one line from stdin",
   {"prompt": "string shown to user"},
   returns=True,
   python="{out} = input({prompt})",
   js="const {out} = require('readline-sync').question({prompt});")

_v("load", "io", "Load JSON/YAML from a file",
   {"file": "path"},
   returns=True,
   python="import json as _json; {out} = _json.load(open({file}))",
   js="const {out} = JSON.parse(require('fs').readFileSync({file}, 'utf8'));")

_v("save", "io", "Save a value as JSON",
   {"value": "any", "file": "path"},
   python="import json as _json; _json.dump({value}, open({file}, 'w'))",
   js="require('fs').writeFileSync({file}, JSON.stringify({value}));")


# ============================================================
# data  (blue)
# ============================================================

_v("filter", "data", "Keep items where condition is true",
   {"from": "list", "where": "predicate using x (e.g. x.age > 18)"},
   returns=True, raw_args=["where"],
   python="{out} = [x for x in {from} if {where}]",
   js="const {out} = ({from}).filter(x => ({where}));")

_v("map", "data", "Transform each item",
   {"from": "list", "to": "expression using x (e.g. x.name)"},
   returns=True, raw_args=["to"],
   python="{out} = [({to}) for x in {from}]",
   js="const {out} = ({from}).map(x => ({to}));")

_v("sort", "data", "Sort list (optionally by key)",
   {"from": "list", "by": "key expression using x (e.g. x.age)"},
   returns=True, raw_args=["by"],
   python="{out} = sorted({from}, key=lambda x: ({by}))",
   js="const {out} = [...({from})].sort((a, b) => {{const f = x => ({by}); return f(a) < f(b) ? -1 : f(a) > f(b) ? 1 : 0;}});")

_v("take", "data", "Take first N items",
   {"from": "list", "n": "number"},
   returns=True,
   python="{out} = ({from})[:int({n})]",
   js="const {out} = ({from}).slice(0, {n});")

_v("skip", "data", "Skip first N items",
   {"from": "list", "n": "number"},
   returns=True,
   python="{out} = ({from})[int({n}):]",
   js="const {out} = ({from}).slice({n});")

_v("count", "data", "Number of items in a list",
   {"of": "list"},
   returns=True,
   python="{out} = len({of})",
   js="const {out} = ({of}).length;")

_v("join", "data", "Join list of strings with a separator",
   {"from": "list", "sep": "string separator"},
   returns=True,
   python="{out} = ({sep}).join(str(x) for x in {from})",
   js="const {out} = ({from}).join({sep});")

_v("split", "data", "Split string by separator",
   {"text": "string", "sep": "separator"},
   returns=True,
   python="{out} = ({text}).split({sep})",
   js="const {out} = ({text}).split({sep});")


# ============================================================
# math  (green)
# ============================================================

_v("add", "math", "a + b",
   {"a": "number", "b": "number"},
   returns=True,
   python="{out} = ({a}) + ({b})",
   js="const {out} = ({a}) + ({b});")

_v("sub", "math", "a - b",
   {"a": "number", "b": "number"},
   returns=True,
   python="{out} = ({a}) - ({b})",
   js="const {out} = ({a}) - ({b});")

_v("mul", "math", "a * b",
   {"a": "number", "b": "number"},
   returns=True,
   python="{out} = ({a}) * ({b})",
   js="const {out} = ({a}) * ({b});")

_v("div", "math", "a / b",
   {"a": "number", "b": "number"},
   returns=True,
   python="{out} = ({a}) / ({b})",
   js="const {out} = ({a}) / ({b});")

_v("sum", "math", "Sum of a list",
   {"of": "list of numbers"},
   returns=True,
   python="{out} = sum({of})",
   js="const {out} = ({of}).reduce((a,b) => a+b, 0);")

_v("avg", "math", "Mean of a list",
   {"of": "list of numbers"},
   returns=True,
   python="{out} = sum({of}) / len({of})",
   js="const {out} = ({of}).reduce((a,b) => a+b, 0) / ({of}).length;")

_v("min", "math", "Smallest item",
   {"of": "list"},
   returns=True,
   python="{out} = min({of})",
   js="const {out} = Math.min(...({of}));")

_v("max", "math", "Largest item",
   {"of": "list"},
   returns=True,
   python="{out} = max({of})",
   js="const {out} = Math.max(...({of}));")

_v("round", "math", "Round to nearest integer",
   {"value": "number"},
   returns=True,
   python="{out} = round({value})",
   js="const {out} = Math.round({value});")


# ============================================================
# text  (purple)
# ============================================================

_v("format", "text", "Format a string with named placeholders ({key})",
   {"template": "string with {placeholders}", "data": "dict/object"},
   returns=True,
   python="{out} = ({template}).format(**({data}))",
   js="const {out} = ({template}).replace(/\\{{(\\w+)\\}}/g, (_,k) => ({data})[k]);")

_v("upper", "text", "Uppercase",
   {"text": "string"}, returns=True,
   python="{out} = ({text}).upper()",
   js="const {out} = ({text}).toUpperCase();")

_v("lower", "text", "Lowercase",
   {"text": "string"}, returns=True,
   python="{out} = ({text}).lower()",
   js="const {out} = ({text}).toLowerCase();")

_v("trim", "text", "Strip whitespace",
   {"text": "string"}, returns=True,
   python="{out} = ({text}).strip()",
   js="const {out} = ({text}).trim();")

_v("replace", "text", "Replace all occurrences",
   {"text": "string", "find": "string", "to": "replacement string"},
   returns=True,
   python="{out} = ({text}).replace({find}, {to})",
   js="const {out} = ({text}).split({find}).join({to});")

_v("contains", "text", "Does a contain b?",
   {"text": "string", "find": "substring"},
   returns=True,
   python="{out} = ({find}) in ({text})",
   js="const {out} = ({text}).includes({find});")


# ============================================================
# time  (orange)
# ============================================================

_v("now", "time", "Current datetime",
   {}, returns=True,
   python="import datetime as _dt; {out} = _dt.datetime.now()",
   js="const {out} = new Date();")

_v("today", "time", "Today's date string (YYYY-MM-DD)",
   {}, returns=True,
   python="import datetime as _dt; {out} = _dt.date.today().isoformat()",
   js="const {out} = new Date().toISOString().slice(0,10);")

_v("wait", "time", "Sleep for N seconds",
   {"seconds": "number"},
   python="import time as _time; _time.sleep({seconds})",
   js="await new Promise(r => setTimeout(r, ({seconds})*1000));")


# ============================================================
# net  (red)
# ============================================================

_v("http_get", "net", "HTTP GET request, return parsed JSON",
   {"url": "URL"}, returns=True,
   python="import requests as _r; {out} = _r.get({url}).json()",
   js="const {out} = await (await fetch({url})).json();")

_v("http_post", "net", "HTTP POST JSON request",
   {"url": "URL", "body": "object/dict"}, returns=True,
   python="import requests as _r; {out} = _r.post({url}, json={body}).json()",
   js="const {out} = await (await fetch({url}, {{method:'POST', body: JSON.stringify({body}), headers:{{'Content-Type':'application/json'}}}})).json();")

_v("download", "net", "Download URL to file",
   {"url": "URL", "to": "file path"},
   python="import requests as _r; open({to}, 'wb').write(_r.get({url}).content)",
   js="const _r = await fetch({url}); require('fs').writeFileSync({to}, Buffer.from(await _r.arrayBuffer()));")


# ============================================================
# ai  (pink)  — placeholders; user wires their own model
# ============================================================

_v("ask_ai", "ai", "Send a prompt to an LLM and get text back",
   {"prompt": "string", "input": "any (will be JSON-stringified)"},
   returns=True,
   python="{out} = __flow_ai__({prompt}, {input})",
   js="const {out} = await __flow_ai__({prompt}, {input});")

_v("classify", "ai", "Classify input into one of labels",
   {"input": "any", "labels": "list of label strings"},
   returns=True,
   python="{out} = __flow_classify__({input}, {labels})",
   js="const {out} = await __flow_classify__({input}, {labels});")

_v("summarize", "ai", "Summarize input text",
   {"input": "string"}, returns=True,
   python="{out} = __flow_ai__('Summarize concisely:', {input})",
   js="const {out} = await __flow_ai__('Summarize concisely:', {input});")

_v("translate", "ai", "Translate text into target language",
   {"text": "string", "to": "language code"}, returns=True,
   python="{out} = __flow_ai__(f'Translate to {{{to}}}:', {text})",
   js="const {out} = await __flow_ai__(`Translate to ${{({to})}}:`, {text});")


# ============================================================
# Extra-language templates (Go / Rust / Bash)
# ============================================================
# Added separately to keep the core verb definitions readable.
# Languages here are best-effort; not every verb maps cleanly.

def _add(verb_name: str, **templates) -> None:
    if verb_name in VERBS:
        VERBS[verb_name].templates.update(templates)


# ---------- Go ----------
_add("read",   go='var {out} = func() string {{ b, _ := os.ReadFile({file}); return string(b) }}()')
_add("write",  go='os.WriteFile({file}, []byte({text}), 0644)')
_add("print",  go='fmt.Println({value})')
_add("ask",    go='var {out} = func() string {{ var s string; fmt.Scanln(&s); return s }}()')

_add("add",    go='var {out} = ({a}) + ({b})')
_add("sub",    go='var {out} = ({a}) - ({b})')
_add("mul",    go='var {out} = ({a}) * ({b})')
_add("div",    go='var {out} = ({a}) / ({b})')
_add("round",  go='var {out} = math.Round({value})')

_add("upper",  go='var {out} = strings.ToUpper({text})')
_add("lower",  go='var {out} = strings.ToLower({text})')
_add("trim",   go='var {out} = strings.TrimSpace({text})')
_add("replace",go='var {out} = strings.ReplaceAll({text}, {find}, {to})')
_add("contains",go='var {out} = strings.Contains({text}, {find})')
_add("split",  go='var {out} = strings.Split({text}, {sep})')
_add("join",   go='var {out} = strings.Join({from}, {sep})')

_add("now",    go='var {out} = time.Now()')
_add("today",  go='var {out} = time.Now().Format("2006-01-02")')
_add("wait",   go='time.Sleep(time.Duration({seconds}) * time.Second)')

_add("http_get", go='var {out} = func() string {{ r, _ := http.Get({url}); defer r.Body.Close(); b, _ := io.ReadAll(r.Body); return string(b) }}()')


# ---------- Rust ----------
_add("read",   rust='let {out} = std::fs::read_to_string({file}).unwrap();')
_add("write",  rust='std::fs::write({file}, {text}).unwrap();')
_add("print",  rust='println!("{{:?}}", {value});')

_add("add",    rust='let {out} = ({a}) + ({b});')
_add("sub",    rust='let {out} = ({a}) - ({b});')
_add("mul",    rust='let {out} = ({a}) * ({b});')
_add("div",    rust='let {out} = ({a}) / ({b});')

_add("upper",  rust='let {out} = ({text}).to_uppercase();')
_add("lower",  rust='let {out} = ({text}).to_lowercase();')
_add("trim",   rust='let {out} = ({text}).trim().to_string();')
_add("replace",rust='let {out} = ({text}).replace({find}, {to});')
_add("contains",rust='let {out} = ({text}).contains({find});')
_add("split",  rust='let {out}: Vec<&str> = ({text}).split({sep}).collect();')

_add("now",    rust='let {out} = std::time::SystemTime::now();')
_add("wait",   rust='std::thread::sleep(std::time::Duration::from_secs({seconds} as u64));')


# ---------- Bash ----------
# Bash templates: side-effect oriented. `out` is bash variable name. Returns
# are captured via command substitution. Skip verbs that don't fit shell idiom.
_add("read",   bash='{out}=$(<{file})')
_add("write",  bash='printf "%s" {text} > {file}')
_add("print",  bash='echo {value}')
_add("ask",    bash='read -p {prompt} {out}')

_add("add",    bash='{out}=$(( {a} + {b} ))')
_add("sub",    bash='{out}=$(( {a} - {b} ))')
_add("mul",    bash='{out}=$(( {a} * {b} ))')
_add("div",    bash='{out}=$(( {a} / {b} ))')

_add("upper",  bash='{out}=$(echo {text} | tr "[:lower:]" "[:upper:]")')
_add("lower",  bash='{out}=$(echo {text} | tr "[:upper:]" "[:lower:]")')
_add("trim",   bash='{out}=$(echo {text} | xargs)')

_add("now",    bash='{out}=$(date "+%Y-%m-%d %H:%M:%S")')
_add("today",  bash='{out}=$(date "+%Y-%m-%d")')
_add("wait",   bash='sleep {seconds}')

_add("http_get", bash='{out}=$(curl -fsSL {url})')
_add("download", bash='curl -fsSL {url} -o {to}')


# ============================================================
# Primary-arg marks (positional + pipe target)
# ============================================================
# These let the parser accept compact forms:
#   print "hi"             ≡ print value="hi"
#   read "a.csv" | upper   ≡ read file="a.csv" -> _p1 ; upper text=_p1
_PRIMARY = {
    "read": "file", "write": "text", "print": "value", "ask": "prompt",
    "load": "file", "save": "value",
    "filter": "from", "map": "from", "sort": "from",
    "take": "from", "skip": "from",
    "count": "of", "join": "from", "split": "text",
    "sum": "of", "avg": "of", "min": "of", "max": "of",
    "round": "value",
    "format": "template",
    "upper": "text", "lower": "text", "trim": "text",
    "replace": "text", "contains": "text",
    "wait": "seconds",
    "http_get": "url", "http_post": "url", "download": "url",
    "ask_ai": "prompt", "classify": "input",
    "summarize": "input", "translate": "text",
}
for _name, _p in _PRIMARY.items():
    if _name in VERBS:
        VERBS[_name].primary_arg = _p


# ============================================================
# Reference rendering (used by `flow doc` and `flow gen --include-doc`)
# ============================================================


def _signature(spec: "VerbSpec") -> str:
    parts = []
    for a, desc in spec.args.items():
        hint = desc.split(" ")[0] if desc else a
        parts.append(f"{a}=<{hint}>")
    sig = " ".join(parts)
    if spec.returns:
        sig = (sig + " -> name").lstrip()
    return sig


def verb_reference(compact: bool = False) -> str:
    """Render the verb registry as a human/LLM-readable reference.

    `compact=True` produces one line per verb, no headings — suitable for
    inclusion in an LLM system prompt without burning a lot of tokens.
    """
    by_cat: Dict[str, List[VerbSpec]] = {}
    cat_order: List[str] = []
    for spec in VERBS.values():
        if spec.category not in by_cat:
            cat_order.append(spec.category)
            by_cat[spec.category] = []
        by_cat[spec.category].append(spec)

    if compact:
        lines = []
        for cat in cat_order:
            for spec in by_cat[cat]:
                lines.append(f"{spec.name:10} {_signature(spec)}")
        return "\n".join(lines)

    out: List[str] = []
    for cat in cat_order:
        out.append(f"## {cat}")
        for spec in by_cat[cat]:
            out.append(f"  {spec.name:10} {_signature(spec):42}  {spec.summary}")
        out.append("")
    return "\n".join(out)
