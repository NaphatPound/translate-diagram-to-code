# Flow — Few-Shot Examples

Use these as message history for the local LLM. Each block is a user request
paired with the canonical compact Flow answer. The assistant should output
only Flow source — no fences, no commentary.

The examples lean on the language's compact features (single-letter aliases,
implicit return, postfix-if, pipe, f-strings) so the model learns the short
form by imitation.

---

USER: print hello

A:
```
p "hello"
```

---

USER: read notes.txt and count lines

A:
```
r "notes.txt" -> text
split text=text sep="\n" -> lines
p count(lines)
```

---

USER: fetch users from the API, keep adults, summarize, save to report.txt

A:
```
when start
  http_get "https://api.example.com/users" -> users
  filter from=users where="x['age'] >= 18" -> adults
  summarize input=adults -> note
  w file="report.txt" text=note
  p "saved"
```

---

USER: for every user, print "hi NAME, age N"

A:
```
each user in users
  p f"hi {user.name}, age {user.age}"
```

---

USER: print "big" if balance > 1000, else "small"

A:
```
p (balance > 1000 ? "big" : "small")
```

---

USER: ask three times and join the answers

A:
```
parts = []
repeat 3 as i
  ask "answer? " -> a
  parts = [*parts, a]
p join(parts, ",")
```

---

USER (parser previously rejected):
    read file=data.csv -> rows
    filter rows where age>18 -> adults
error: `line 2:13: arg 'rows' must be followed by '='`

A (fixed — use named arg + quote the predicate, then chain):
```
r "data.csv" | filter where="x['age'] > 18" -> adults
```

---

USER: read input.txt, translate to Thai, save as output.txt

A:
```
r "input.txt" | translate to=th | w file="output.txt"
```

---

USER: load scores.json, sort by score descending, take top 3, print

A:
```
load "scores.json" -> items
sort from=items by="-x['score']" | take n=3 -> top
each item in top
  p item
```

---

USER: define double(x) and call it

A:
```
def double x
  x * 2

p double(5)
```

---

USER: factorial of 10

A:
```
def fact n
  if n <= 1
    return 1
  n * fact(n - 1)

p fact(10)
```

---

USER: safely look up a config key, fall back to a default

A:
```
host = config?.db?.host ?? "localhost"
p host
```

---

## Anti-patterns to avoid

- ❌ `let x = ...`            ✅ `x = ...`
- ❌ `if x = 1`               ✅ `if x == 1`
- ❌ `print(x)` / `print value=x`     ✅ `p x`
- ❌ `each row in rows:`              ✅ `each row in rows` (no colon)
- ❌ 4-space indent                   ✅ 2-space indent
- ❌ `return x` at def end            ✅ bare `x` (implicit return)
- ❌ `add a=1 b=2 -> s`               ✅ `s = 1 + 2` (use assignment for math)
- ❌ `count of=items -> n`            ✅ `n = count(items)` (assignment + funccall)
