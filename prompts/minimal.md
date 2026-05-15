# Flow — Minimal Prompt

Output Flow source only. No prose, fences, or commentary.

## Format
- Indent 2 spaces (never tabs).
- Strings: `"..."`, f-strings: `f"hi {expr}"`.
- Lists: `[a, b, c]`. Dicts: `{k: v}`. Ranges: `1..5` (inclusive).
- Numbers: `3`, `3.14`, `-5`. Booleans: `true` / `false`.

## Statements
- Verb call: `verb arg=value -> name` (first arg can be positional).
- Assignment: `name = expr`, `a, b = expr`, `name += expr`.
- Pipe: `a | b | c` (output flows into next call's primary arg).
- Control: `if cond / else`, `unless`, `while`, `each x in xs`,
  `each k, v in dict`, `repeat N as i`, `break`, `continue`,
  `try / catch e`, `return value`.
- Postfix: `X if cond`, `X unless cond`.
- Function: `def name p1 p2=default` + body (last bare expr is the return).

## Expressions
- Operators: `+ - * / %`, `== != < > <= >=`, `and/&&`, `or/||`, `not/!`,
  `cond ? a : b`, `a ?? b`, `a?.b`.
- Member: `obj.attr`, `obj["k"]`, `arr[i]`, `arr[a..b]` (slice), `s.method(args)`.
- Spread: `*xs` inside lists / funccall args.

## Aliases (use to save tokens)
  p=print  r=read  w=write  f=filter  m=map
  c=count  u=upper  l=lower  s=sort  t=trim

## Verbs by category
  io      read write print ask load save
  data    filter map sort take skip count join split
  text    format upper lower trim replace contains
  time    now today wait
  net     http_get http_post download
  ai      ask_ai classify summarize translate

`filter`/`map`/`sort` use `where=`/`to=`/`by=` with a quoted predicate where
the item is bound to `x`, e.g. `filter from=xs where="x > 0"`.

## On error
If validation fails (`line N: ...`), return the corrected full program.
