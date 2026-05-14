# Flow — Few-Shot Examples

Use these as the message history for the local LLM. Each block is one
`user` → `assistant` exchange. Stop tokens for the assistant are not
required; the model should learn to emit only Flow code.

---

### Example 1 — Hello world

USER:
print "hello"

ASSISTANT:
```
print value="hello"
```

---

### Example 2 — Read a file and count lines

USER:
read notes.txt and count lines

ASSISTANT:
```
read file=notes.txt -> text
split text=text sep="\n" -> lines
count of=lines -> n
print value=n
```

---

### Example 3 — Filter and summarize

USER:
fetch users from the API, keep adults, summarize them, save report.txt

ASSISTANT:
```
when start
  http_get url=https://api.example.com/users -> users
  filter from=users where="x['age'] >= 18" -> adults
  summarize input=adults -> summary
  write file=report.txt text=summary
  print value="report saved"
```

---

### Example 4 — Loop with formatted output

USER:
for every user, print "hi NAME, age N"

ASSISTANT:
```
each user in users
  format template="hi {name}, age {age}" data=user -> msg
  print value=msg
```

---

### Example 5 — Conditional branch

USER:
if balance is more than 1000, print "approved", else "rejected"

ASSISTANT:
```
if balance > 1000
  print value="approved"
else
  print value="rejected"
```

---

### Example 6 — Repeat with side effect

USER:
ask user three times and collect their answers

ASSISTANT:
```
split text="" sep="," -> answers
repeat 3
  ask prompt="answer? " -> a
  join from=answers sep="," -> joined
  format template="{prev},{new}" data={"prev": joined, "new": a} -> answers
```

(NOTE: this example shows the limit of one-liners — when state accumulation
gets complex, prefer using `each` + a helper variable.)

---

### Example 7 — Error correction

USER (the parser previously rejected the assistant's code):
```
read file=data.csv -> rows
filter rows where age>18 -> adults
```
parser error: `line 2:13: arg 'rows' must be followed by '='`

ASSISTANT (fixed):
```
read file=data.csv -> rows
filter from=rows where="x['age'] > 18" -> adults
```

(The fix: use named arg `from=`, and quote the predicate.)

---

### Example 8 — Translation pipeline

USER:
read paragraph from input.txt, translate to Thai, save as output.txt

ASSISTANT:
```
read file=input.txt -> text
translate text=text to=th -> translated
write file=output.txt text=translated
print value="done"
```

---

### Example 9 — Top-3 by score

USER:
load scores.json, sort by score descending, take top 3, print

ASSISTANT:
```
load file=scores.json -> items
sort from=items by="-x['score']" -> ranked
take from=ranked n=3 -> top
each item in top
  print value=item
```

---

### Example 10 — Compact form (pipe + positional primary)

USER:
read input.txt, uppercase, save to output.txt

ASSISTANT (compact — preferred for linear pipelines):
```
read "input.txt" | upper | write file="output.txt"
```

ASSISTANT (equivalent longhand):
```
read file="input.txt" -> text
upper text=text -> big
write file="output.txt" text=big
```

The compact form uses two pieces of sugar:
  - Positional primary arg: `read "x"` ≡ `read file="x"`, `upper "y"` ≡ `upper text="y"`.
  - Pipe `|`: the upstream value flows into the next verb's primary arg, so
    you don't have to name intermediates.

---

### Example 11 — Mixing pipe with named args

USER:
fetch users, keep adults, summarize

ASSISTANT:
```
http_get "https://api.example.com/users" | filter where="x['age'] >= 18" | summarize -> note
print note
```

---

### Anti-pattern reminders (do NOT do these)

- ❌ `let x = read("data.csv")`  (no `let`, no parens; use `read file=data.csv -> x`)
- ❌ `if x = 1`                  (use `==` for comparison)
- ❌ `print(x)`                  (no parens around args; use `print value=x`)
- ❌ `each row in rows:`         (no trailing colon)
- ❌ 4-space indent              (always 2 spaces)
