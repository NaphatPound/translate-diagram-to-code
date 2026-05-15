# Flow — Minimal Prompt

Output Flow source only. No prose, fences, or commentary.

## Format
- Indent 2 spaces (never tabs).
- Strings: `"..."`, f-strings: `f"hi {expr}"`.
- Lists: `[a, b, c]`. Dicts: `{k: v}`. Ranges: `1..5` (inclusive).
- Comprehensions: `[expr for x in xs]`, `[expr for x in xs if cond]`,
  `{k: v for x in xs}`, `{k: v for x in xs if cond}`.
- Numbers: `3`, `3.14`, `-5`. Booleans: `true` / `false`.

## Statements
- Verb call: `verb arg=value -> name` (first arg can be positional).
- Assignment: `name = expr`, `a, b = expr`, `name += expr`.
- Pipe: `a | b | c` (output flows into next call's primary arg).
  Any value can start a pipe: `xs | reverse | p`, `[1,2,3] | sum | p`,
  `"hello" | upper | p`.
- Builtin funccalls (1-arg, return a value, no `-> name` needed):
  `count(xs) sum(xs) min(xs) max(xs) avg(xs) sorted(xs) reverse(xs)
   unique(xs) keys(d) values(d) first(xs) last(xs) flatten(xss)
   len(s) abs(n) round(n) str(x) int(x) float(x)`.
- Control: `if cond / else`, `unless`, `while`, `each x in xs`,
  `each k, v in dict`, `repeat N as i`, `break`, `continue`,
  `try / catch e`, `return value`,
  `match v / case PAT / ... / case x` (last `case <fresh-name>` binds v to x).
- Postfix: `X if cond`, `X unless cond`.
- Function: `def name p1 p2=default` + body (last bare expr is the return).

## Expressions
- Operators: `+ - * / %`, `== != < > <= >=`, `and/&&`, `or/||`, `not/!`,
  `x in xs` / `x not in xs` (membership),
  `cond ? a : b` or `a if cond else b` (ternary, either form),
  `a ?? b`, `a?.b`.
- Member: `obj.attr`, `obj["k"]`, `arr[i]`, `arr[a..b]` (inclusive),
  `arr[a:b]` / `arr[:b]` / `arr[a:]` / `arr[:]` (Python-style exclusive),
  `s.method(args)`.
- Spread: `*xs` inside lists / funccall args.

## Aliases (use to save tokens)
  p=print  r=read  w=write  f=filter  m=map
  c=count  u=upper  l=lower  s=sort  t=trim

## Verbs by category
  io      read write print ask load save
  data    filter map sort take skip count join split reverse unique keys values
          first last flatten zip
  text    format upper lower trim replace contains
  math    add sub mul div sum avg min max round
  time    now today wait
  net     http_get http_post download
  ai      ask_ai classify summarize translate

`filter`/`map`/`sort` use `where=`/`to=`/`by=` with a quoted predicate where
the item is bound to `x`, e.g. `filter from=xs where="x > 0"`.

## On error
If validation fails (`line N: ...`), return the corrected full program.
