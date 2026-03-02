# DSAI 545 – Basic Text Processing Cheat Sheet & Practice Questions

---

## 1. Regular Expressions (RegEx)

A formal language for specifying text string patterns. Used everywhere in NLP pipelines — preprocessing, tokenization, text cleaning, pattern extraction.

---

## 2. Core RegEx Syntax

### Character Classes (Disjunctions)

| Pattern | Matches | Example |
|---------|---------|---------|
| `[wW]oodchuck` | Woodchuck or woodchuck | Case-insensitive match |
| `[A-Z]` | Any uppercase letter | D, M, Z |
| `[a-z]` | Any lowercase letter | a, m, z |
| `[0-9]` | Any single digit | 0, 5, 9 |
| `[abc]` | a, b, or c | Same as `a\|b\|c` |

### Negation

| Pattern | Matches | Note |
|---------|---------|------|
| `[^A-Z]` | NOT uppercase | Carat `^` negates only when **first** inside `[]` |
| `[^Ss]` | Neither S nor s | |
| `[^.]` | Not a period | Special chars lose meaning inside `[]` |
| `[e^]` | e OR ^ literal | `^` not first → literal character |

### Convenient Aliases

| Alias | Expansion | Matches |
|-------|-----------|---------|
| `\d` | `[0-9]` | Any digit |
| `\D` | `[^0-9]` | Any non-digit |
| `\w` | `[a-zA-Z0-9_]` | Alphanumeric or underscore |
| `\W` | `[^\w]` | Non-alphanumeric |
| `\s` | `[ \r\t\n\f]` | Whitespace |
| `\S` | `[^\s]` | Non-whitespace |

### Quantifiers (Wildcards, Optionality, Repetition)

| Pattern | Meaning | Example: matches |
|---------|---------|-----------------|
| `.` | Any single character | `beg.n` → begin, begun, beg3n |
| `?` | 0 or 1 of previous | `woodchucks?` → woodchuck, woodchucks |
| `*` | 0 or more of previous | `to*` → t, to, too, tooo |
| `+` | 1 or more of previous | `to+` → to, too, tooo (NOT t) |

### Disjunction (OR)

| Pattern | Matches |
|---------|---------|
| `groundhog\|woodchuck` | Either full string |
| `[gG]roundhog\|[Ww]oodchuck` | Case-insensitive both |

### Anchors

| Pattern | Meaning | Example |
|---------|---------|---------|
| `^` | Start of line | `^[A-Z]` → line starts with uppercase |
| `$` | End of line | `\.$` → line ends with period |

---

## 3. Substitution & Capture Groups

### Substitution
```
s/regexp/replacement/
s/colour/color/          → replaces "colour" with "color"
```

### Capture Groups — `()` and `\1`, `\2`, ...
Parentheses **capture** matched text into numbered registers.

```
s/([0-9]+)/<\1>/         → "the 35 boxes" becomes "the <35> boxes"
```

### Backreferences
```
/the (.*)er they (.*), the \1er we \2/

✓ matches: "the faster they ran, the faster we ran"
✗ no match: "the faster they ran, the faster we ate"
```

### Non-Capturing Groups — `(?:...)`
Groups without capturing (doesn't fill a register):
```
/(?:some|a few) (people|cats) like some \1/

✓ "some cats like some cats"
✗ "some cats like some some"   ← \1 = "cats", not "some"
```

### Lookahead Assertions
| Syntax | Meaning |
|--------|---------|
| `(?=pattern)` | Positive lookahead — matches if pattern ahead, zero-width |
| `(?!pattern)` | Negative lookahead — matches if pattern NOT ahead |

```
/^(?!Volcano)[A-Za-z]+/   → word at line start that doesn't begin with "Volcano"
```

---

## 4. The Iterative RegEx Process

Finding "the" in text — a classic example of precision/recall tradeoff:

| Attempt | Problem |
|---------|---------|
| `the` | Misses "The" (capitalized) |
| `[tT]he` | False positives: "other", "Theology" |
| `\W[tT]he\W` | Misses "THE" (all caps) |

Each fix addresses one error type but may introduce another → iterative refinement.

---

## 5. Precision & Recall

| Metric | Formula | Minimizes |
|--------|---------|-----------|
| **Precision** | TP / (TP + FP) | False positives |
| **Recall** | TP / (TP + FN) | False negatives |

- **False Positive:** Matched something we shouldn't have (e.g., "other" when searching for "the")
- **False Negative:** Missed something we should have matched (e.g., "The" with pattern `the`)
- These two goals are often **antagonistic** — improving one may hurt the other.

---

## 6. Words and Corpora

### Key Terminology

| Term | Definition | Example |
|------|-----------|---------|
| **Lemma** | Base form (same stem + POS + rough sense) | cat, cats → lemma "cat" |
| **Wordform** | Full inflected surface form | "cat" and "cats" are different wordforms |
| **Token** | An instance of a word in running text | "the cat sat on the mat" → 6 tokens |
| **Type** | A unique element of the vocabulary | "the cat sat on the mat" → 5 types |

### Example
> "they lay back on the San Francisco grass and looked at the stars and their"

- **Tokens:** 15 (or 14 depending on "San Francisco" treatment)
- **Types:** 13 (or 12) — "the" and "and" each appear twice

### Heaps' Law (Herdan's Law)

```
|V| = k · N^β     where 0.67 < β < 0.75
```

- **N** = number of tokens in corpus
- **|V|** = vocabulary size (number of types)
- Vocabulary grows with **greater than square root** of token count
- Implication: you'll always encounter new words as corpus grows (no saturation)

### Reference Corpus Sizes

| Corpus | Tokens (N) | Types (\|V\|) |
|--------|-----------|--------------|
| Switchboard (phone) | 2.4M | 20K |
| Shakespeare | 884K | 31K |
| COCA | 440M | 2M |
| Google N-grams | 1T | 13M+ |

> Shakespeare has more types than Switchboard despite fewer tokens — reflects richer vocabulary/creative language.

---

## 7. ELIZA — Simple RegEx Application

An early NLP chatbot (1966, Weizenbaum) simulating a therapist using pattern matching + substitution:

```
s/.* I'M (depressed|sad) .*/I AM SORRY TO HEAR YOU ARE \1/
s/.* all .*/IN WHAT WAY?/
s/.* always .*/CAN YOU THINK OF A SPECIFIC EXAMPLE?/
```

Key insight: Simple regex rules can create surprisingly convincing conversational behavior — but it's pattern matching, not understanding.

---

## 8. Corpora Context

Every text has context that matters for NLP:
- **Writer(s)** — who produced it
- **Time** — when it was written
- **Variety** — dialect, register, domain
- **Language** — which language
- **Function** — purpose (news, conversation, academic, etc.)

This affects tokenization decisions, vocabulary, and model behavior.

---

## Quick Reference Card

```
┌──────────────────────────────────────────────┐
│  REGEX CHEAT CARD                            │
│                                              │
│  [abc]  = a or b or c                        │
│  [^abc] = NOT a, b, or c                     │
│  a|b    = a OR b (string-level)              │
│  .      = any character                      │
│  ?      = 0 or 1      *  = 0 or more         │
│  +      = 1 or more                          │
│  ^      = start of line   $ = end of line    │
│  \d \w \s = digit, word, space               │
│  \D \W \S = negations of above               │
│  (...)  = capture group   \1 = backreference │
│  (?:...) = non-capturing group               │
│  (?=...) = positive lookahead                │
│  (?!...) = negative lookahead                │
│                                              │
│  WORDS: Token=instance, Type=unique          │
│  Lemma=base form, Wordform=inflected         │
│  Heaps: |V| = kN^β  (β ≈ 0.67–0.75)        │
│  Precision = TP/(TP+FP)                      │
│  Recall    = TP/(TP+FN)                      │
└──────────────────────────────────────────────┘
```

---

# Practice Midterm Questions

---

## Q1: RegEx Pattern Matching (Easy)

**Which of the following strings does the pattern `[A-Z][a-z]*` match?**

(a) "Hello"  (b) "hello"  (c) "HELLO"  (d) "Hello World"  (e) "A"

### Solution

The pattern means: one uppercase letter, followed by zero or more lowercase letters.

- **(a) "Hello"** → H + ello → ✓ matches "Hello"
- **(b) "hello"** → starts with lowercase → ✗ no match
- **(c) "HELLO"** → H matches `[A-Z]`, then E is not `[a-z]` → matches only "H" (partial match)
- **(d) "Hello World"** → matches "Hello" (then space breaks it)
- **(e) "A"** → A + zero lowercase → ✓ matches "A"

**Answer: (a) and (e) fully match. (c) and (d) partially match.**

---

## Q2: Write the RegEx (Easy-Medium)

**Write a regular expression that matches Turkish phone numbers in the format `0XXX XXX XX XX` where X is a digit.**

### Solution

```
0\d{3}\s\d{3}\s\d{2}\s\d{2}
```

Or more explicitly:
```
0[0-9]{3} [0-9]{3} [0-9]{2} [0-9]{2}
```

If you want to also match the `+90` prefix format:
```
(\+90|0)\d{3}\s\d{3}\s\d{2}\s\d{2}
```

---

## Q3: False Positives & Negatives (Medium)

**You want to match email addresses using the regex `\w+@\w+\.\w+`. Identify one false positive and one false negative this pattern would produce.**

### Solution

**False Negative (missed valid email):**
- `yigit.kaya@boun.edu.tr` → the dots in username and multiple dots in domain are not matched by `\w+` (which doesn't include `.`)
- `user+tag@gmail.com` → the `+` is not in `\w`

**False Positive (incorrectly matched):**
- `not_an_email@single` embedded in text "contact not_an_email@single word" — could match if the TLD part matches a following word boundary incorrectly. Though this specific pattern is actually fairly restrictive.
- More realistically, `___@___._` would match even though it's not a real email.

**Key takeaway:** Simple regex for email always has precision/recall tradeoffs. The official email regex (RFC 5322) is notoriously complex.

---

## Q4: Capture Groups & Substitution (Medium)

**Given the substitution rule:**
```
s/(\w+) (\w+)/\2 \1/
```
**What is the output for each input?**

(a) `"Hello World"`
(b) `"Natural Language Processing"`

### Solution

The rule captures two consecutive words and swaps them.

**(a)** `"Hello World"` → captures Hello=\1, World=\2 → **"World Hello"**

**(b)** `"Natural Language Processing"` → first match: Natural=\1, Language=\2 → **"Language Natural Processing"**

(Only the first match is replaced unless using global flag `g`)

With `s/(\w+) (\w+)/\2 \1/g`:
(b) → "Language Natural" then "Processing" has no pair → **"Language Natural Processing"** (same result since "Processing" is alone)

---

## Q5: Tokens, Types, Lemmas (Medium)

**Given the sentence:**
> "The cats chased the cats and the dogs chased the cats"

**(a)** How many tokens?
**(b)** How many types?
**(c)** How many lemmas?

### Solution

**(a) Tokens:** Count every word instance:
The, cats, chased, the, cats, and, the, dogs, chased, the, cats → **11 tokens**

**(b) Types:** Unique wordforms:
{The, the, cats, chased, and, dogs} → **6 types** (or 5 if case-insensitive: {the, cats, chased, and, dogs})

**(c) Lemmas:** Base forms:
{the, cat, chase, and, dog} → **5 lemmas**

> **Exam trap:** "The" vs "the" — are they the same type? Depends on whether tokenization is case-sensitive. State your assumption.

---

## Q6: Heaps' Law Computation (Medium)

**A corpus has N = 1,000,000 tokens. Using Heaps' Law with k = 30 and β = 0.7, estimate the vocabulary size.**

### Solution

```
|V| = k · N^β
|V| = 30 · (1,000,000)^0.7
```

Compute (1,000,000)^0.7:
- log10(1,000,000) = 6
- 6 × 0.7 = 4.2
- 10^4.2 ≈ 15,849

```
|V| = 30 × 15,849 ≈ 475,470
```

**Answer: ~475K types**

This means even with 1M tokens, you'd expect nearly half a million unique words — showing why vocabulary management (subword tokenization, BPE) is critical in NLP.

---

## Q7: RegEx Debugging (Medium-Hard)

**You are given the regex `^[A-Za-z]+\s\d+$` and the following strings. Which ones match?**

(a) `"Chapter 42"`
(b) `"chapter42"`
(c) `"HELLO 007"`
(d) `"A 1"`
(e) `"Test 12 34"`

### Solution

Pattern breakdown: `^` start → `[A-Za-z]+` one or more letters → `\s` exactly one whitespace → `\d+` one or more digits → `$` end.

- **(a) "Chapter 42"** → letters + space + digits → ✓
- **(b) "chapter42"** → no space between letters and digits → ✗
- **(c) "HELLO 007"** → letters + space + digits → ✓
- **(d) "A 1"** → one letter + space + one digit → ✓ (+ means 1 or more)
- **(e) "Test 12 34"** → has two spaces/digit groups → ✗ (pattern expects exactly one space then digits to end)

**Answer: (a), (c), (d)**

---

## Q8: Lookahead Application (Hard)

**Write a regex that matches words containing "ing" but NOT starting with "s".**

### Solution

```
\b(?!s)[a-zA-Z]*ing\b
```

Breakdown:
- `\b` — word boundary
- `(?!s)` — negative lookahead: next char is NOT 's'
- `[a-zA-Z]*` — zero or more letters
- `ing` — literal "ing"
- `\b` — word boundary

✓ matches: running, playing, king, bring
✗ rejects: singing, string, swing

---

## Q9: Conceptual Questions (Exam-Style)

### 9a. Why are precision and recall antagonistic in regex design?

Making a pattern broader (more general) catches more true positives → **recall increases**, but also matches more false positives → **precision decreases**. Making it stricter improves precision but misses edge cases → recall drops. For example, `[tT]he` has better recall than `the` but worse precision (matches "there", "other").

### 9b. Why does Shakespeare have more types than Switchboard despite having fewer tokens?

Shakespeare's creative literary language introduces many unique words (archaic forms, coined words, poetic vocabulary), while phone conversations reuse a smaller set of common everyday words repeatedly. This illustrates that vocabulary size depends not just on corpus size but on **language variety and register**.

### 9c. What is the difference between `(...)` and `(?:...)`?

Both group the enclosed pattern, but `(...)` also **captures** the matched text into a numbered register (\1, \2, etc.) for backreference or substitution. `(?:...)` groups without capturing — useful when you need grouping for alternation or quantifiers but don't want to waste a register.

### 9d. ELIZA uses regex substitution to simulate conversation. Why is this not "real" NLP understanding?

ELIZA applies fixed pattern-matching rules with no semantic comprehension, memory of context, or world knowledge. It maps surface patterns to canned responses. It can't handle novel sentence structures, resolve ambiguity, or maintain coherent multi-turn reasoning. It demonstrates the **ELIZA effect** — humans' tendency to attribute understanding to systems that mimic conversational patterns.

---

## Q10: Comprehensive Application (Hard)

**Design a regex-based pipeline to extract dates in the format "DD/MM/YYYY" from a document, where DD is 01-31, MM is 01-12, and YYYY is a four-digit year. Then use a substitution to reformat them to "YYYY-MM-DD" (ISO format).**

### Solution

**Step 1 — Match pattern:**
```
(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/(\d{4})
```

Breakdown:
- Day: `0[1-9]` (01-09) or `[12][0-9]` (10-29) or `3[01]` (30-31)
- Month: `0[1-9]` (01-09) or `1[0-2]` (10-12)
- Year: `\d{4}` (any four digits)

**Step 2 — Substitution:**
```
s/(0[1-9]|[12][0-9]|3[01])\/(0[1-9]|1[0-2])\/(\d{4})/\3-\2-\1/
```

Example: `28/02/2026` → `2026-02-28`

> **Limitation:** This regex doesn't validate actual calendar dates (e.g., 31/02/2026 would match). Full date validation requires logic beyond regex — nice thing to mention in an exam for bonus points.
