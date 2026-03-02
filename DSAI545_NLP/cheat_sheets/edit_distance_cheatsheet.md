# DSAI 545 – Edit Distance Cheat Sheet & Practice Questions

---

## 1. Core Concept

**Edit Distance** (aka Minimum Edit Distance) = the minimum number of editing operations needed to transform one string into another.

**Three operations:**
- **Insertion** – add a character
- **Deletion** – remove a character
- **Substitution** – replace a character with another

---

## 2. Cost Schemes

| Scheme | Insert | Delete | Substitution (mismatch) | Match |
|--------|--------|--------|------------------------|-------|
| Simple | 1 | 1 | 1 | 0 |
| Levenshtein | 1 | 1 | **2** | 0 |
| Weighted | del[x(i)] | ins[y(j)] | sub[x(i), y(j)] | 0 |

> **Key distinction for exams:** When they say "Levenshtein distance", substitution costs **2** (equivalent to a delete + insert). When they say "edit distance with unit costs", substitution costs **1**.

---

## 3. Dynamic Programming Formulation

### Notation
- **X** = source string of length **n**
- **Y** = target string of length **m**
- **D(i, j)** = edit distance between X[1..i] and Y[1..j]
- **Answer** = D(n, m)

### Initialization (Base Cases)

```
D(i, 0) = i    (delete all i characters from X)
D(0, j) = j    (insert all j characters of Y)
```

### Recurrence (Levenshtein)

```
            ┌ D(i-1, j)   + 1                          ← deletion
D(i,j) = min│ D(i, j-1)   + 1                          ← insertion
            └ D(i-1, j-1) + cost(X[i], Y[j])           ← substitution/match

where cost(a, b) = 0 if a == b, else 2 (Levenshtein) or 1 (simple)
```

### Termination
```
D(n, m) = minimum edit distance
```

### Complexity
| | Time | Space |
|--|------|-------|
| DP Table | O(n·m) | O(n·m) |
| Backtrace | O(n+m) | – |

---

## 4. How to Fill the Table (Step-by-Step)

**Example: INTENTION → EXECUTION (Levenshtein, sub=2)**

**Step 1:** Write target (Y = EXECUTION) along columns, source (X = INTENTION) along rows.

**Step 2:** Fill base cases — first row = 0,1,2,...,m and first column = 0,1,2,...,n.

**Step 3:** For each cell (i,j), compute three candidates and take the minimum:
- **From above** D(i-1, j) + 1 → deletion
- **From left** D(i, j-1) + 1 → insertion  
- **From diagonal** D(i-1, j-1) + 0 (if match) or +2 (if mismatch) → sub/match

**Step 4:** D(n, m) in the top-right corner is the answer.

### Completed Table (Levenshtein, sub=2)

```
        #   E   X   E   C   U   T   I   O   N
  #     0   1   2   3   4   5   6   7   8   9
  I     1   2   3   4   5   6   7   6   7   8
  N     2   3   4   5   6   7   8   7   8   7
  T     3   4   5   6   7   8   7   8   9   8
  E     4   3   4   5   6   7   8   9  10   9
  N     5   4   5   6   7   8   9  10  11  10
  T     6   5   6   7   8   9   8   9  10  11
  I     7   6   7   8   9  10   9   8   9  10
  O     8   7   8   9  10  11  10   9   8   9
  N     9   8   9  10  11  12  11  10   9   8
```

**Answer: D(9,9) = 8**

---

## 5. Backtrace (Alignment)

To recover the **alignment** (not just the distance), store a pointer at each cell recording which neighbor you came from:

| Pointer direction | Operation | Meaning |
|-------------------|-----------|---------|
| ↖ Diagonal | Match/Substitution | Align X[i] with Y[j] |
| ↑ From below | Deletion | X[i] aligned to gap |
| ← From left | Insertion | Y[j] aligned to gap |

Trace back from D(n,m) to D(0,0) following the pointers. The resulting alignment for INTENTION → EXECUTION:

```
I N T E * N T I O N
| | | | | | | | | |
* E X E C U T I O N
```

Operations: d (delete I), s (I→E), s (N→X), (E=E match), s (N→C), s (T→U), (T=T match), (I=I match), (O=O match), (N=N match) — but the specific path depends on tie-breaking.

---

## 6. Weighted Edit Distance

When different operations have different costs (e.g., from a **confusion matrix** of common typos):

```
D(i-1, j)   + del[X(i)]
D(i,j) = min  D(i, j-1)   + ins[Y(j)]
D(i-1, j-1) + sub[X(i), Y(j)]
```

**Use case:** Spell correction — 'e' and 'r' are adjacent on keyboard, so sub('e','r') should cost less than sub('e','z').

---

## 7. Applications Quick Reference

| Domain | How edit distance is used |
|--------|--------------------------|
| **Spell correction** | Find dictionary word closest to misspelled word |
| **Computational biology** | Sequence alignment of DNA/protein |
| **Machine translation** | Evaluate translation quality (word-level edit distance) |
| **Speech recognition** | Word Error Rate (WER) between hypothesis and reference |
| **Named entity resolution** | Match "IBM Inc." with "IBM" |
| **Handwriting recognition** | Compare OCR output with reference |

---

## 8. Key Formulas to Memorize

```
1. D(i,0) = i,  D(0,j) = j

2. D(i,j) = min { D(i-1,j)+1,  D(i,j-1)+1,  D(i-1,j-1)+cost }

3. Time & Space: O(n·m)

4. Backtrace: O(n+m)
```

---

# Practice Midterm Questions

---

## Q1: Basic Computation (Easy)

**Compute the minimum edit distance between "CAT" and "CAR" using Levenshtein distance (sub=2).**

### Solution

```
      #   C   A   R
  #   0   1   2   3
  C   1   0   1   2
  A   2   1   0   1
  T   3   2   1   2
```

**Step-by-step for the interesting cell D(3,3) — comparing T vs R:**
- From above: D(2,3) + 1 = 1 + 1 = 2
- From left: D(3,2) + 1 = 1 + 1 = 2
- Diagonal: D(2,2) + 2 = 0 + 2 = 2 (T ≠ R, so cost = 2)
- min(2, 2, 2) = **2**

**Answer: 2** (one substitution T→R with cost 2, or equivalently delete T + insert R each with cost 1).

---

## Q2: Full Table Construction (Medium)

**Compute the Levenshtein distance (sub=2) between "SUNNY" and "SNOWY". Show the full DP table.**

### Solution

Source X = SUNNY (length 5), Target Y = SNOWY (length 5)

```
        #   S   N   O   W   Y
  #     0   1   2   3   4   5
  S     1   0   1   2   3   4
  U     2   1   2   3   4   5
  N     3   2   1   2   3   4
  N     4   3   2   3   4   5
  Y     5   4   3   4   5   4
```

**Walkthrough of key cells:**

**D(1,1): S vs S** → match → D(0,0) + 0 = 0 ✓

**D(2,1): U vs S** → min(D(1,1)+1, D(2,0)+1, D(1,0)+2) = min(1, 3, 3) = **1**

**D(3,2): N vs N** → match → D(2,1) + 0 = 1 ✓

**D(5,5): Y vs Y** → match → D(4,4) + 0 = 4 ✓

**Answer: 4**

**Alignment (one possible):**
```
S U N N _ Y
S _ N O W Y
```
Operations: match S, delete U, match N, sub N→O (cost 2), insert W, match Y → total = 1+2+1 = 4 ✓

---

## Q3: Simple vs Levenshtein (Medium)

**What is the edit distance between "ABCD" and "ACBD" under (a) simple edit distance (sub=1) and (b) Levenshtein (sub=2)?**

### Solution

**(a) Simple (sub=1):**

```
        #   A   C   B   D
  #     0   1   2   3   4
  A     1   0   1   2   3
  B     2   1   1   1   2
  C     3   2   1   2   2
  D     4   3   2   2   2
```

**Answer: 2** (swap B and C = 2 substitutions at cost 1 each)

**(b) Levenshtein (sub=2):**

```
        #   A   C   B   D
  #     0   1   2   3   4
  A     1   0   1   2   3
  B     2   1   2   1   2
  C     3   2   1   2   2
  D     4   3   2   2   2
```

**Answer: 2** (delete B, insert B after C — two operations at cost 1 each, which is cheaper than 1 substitution at cost 2)

> **Exam tip:** Under Levenshtein, it's sometimes cheaper to do delete+insert instead of substitution since del(1)+ins(1) = 2 = sub cost. They tie here, but if sub > 2, delete+insert would be strictly preferred.

---

## Q4: Backtrace & Alignment (Medium-Hard)

**Given the strings X = "PARK" and Y = "SPAKE", compute the Levenshtein distance and provide the alignment using backtrace.**

### Solution

```
        #   S   P   A   K   E
  #     0   1   2   3   4   5
  P     1   2   1   2   3   4
  A     2   3   2   1   2   3
  R     3   4   3   2   3   4
  K     4   5   4   3   2   3
```

**Answer: 3**

**Backtrace from D(4,5):**
- D(4,5)=3 ← D(3,4)=3+0? No, K≠E. Check: D(3,5)+1=5, D(4,4)+1=3, D(3,4)+2=5 → came from left D(4,4)+1
- D(4,4)=2 ← K=K match → diagonal D(3,3)=2
- D(3,3)=2 ← R≠A, so check: D(2,3)+1=2, D(3,2)+1=3, D(2,2)+2=4 → from above D(2,3)+1
- D(2,3)=1 ← A=A match → diagonal D(1,2)=1
- D(1,2)=1 ← P=P match → diagonal D(0,1)=1
- D(0,1)=1 ← from D(0,0)+1 (insert S)

**Alignment:**
```
_ P A R K _
S P A _ K E
```
Operations: insert S (cost 1), match P, match A, delete R (cost 1), match K, insert E (cost 1) → **total = 3** ✓

---

## Q5: Conceptual Questions (Exam-Style)

### 5a. Why is dynamic programming used instead of brute-force search?

The space of all possible edit sequences is exponentially large. DP avoids redundant computation by storing solutions to subproblems — many distinct paths lead to the same intermediate state, and we only need to keep the shortest path to each state. This reduces time from exponential to **O(n·m)**.

### 5b. When would you use weighted edit distance over standard Levenshtein?

When errors are not equally likely. In spell correction, adjacent keyboard keys (e.g., 'r' and 't') are more commonly confused than distant keys (e.g., 'r' and 'z'), so substituting 'r'→'t' should cost less. In biology, certain mutations are more frequent than others. A **confusion matrix** captures these empirical costs.

### 5c. What does each direction in the backtrace represent?

- **Diagonal (↖):** The characters are aligned (match or substitution)
- **Up (↑):** Source character is deleted (aligned with gap in target)
- **Left (←):** Target character is inserted (aligned with gap in source)

### 5d. Can two strings have multiple optimal alignments? Why?

Yes. When multiple directions yield the same minimum cost at a cell, there are multiple equally optimal paths through the table. The backtrace can follow any of them, producing different valid alignments with the same total cost.

---

## Q6: Tricky Edge Case (Hard)

**Compute edit distance (sub=1, simple) between "AAA" and "A".**

### Solution

```
      #   A
  #   0   1
  A   1   0
  A   2   1
  A   3   2
```

**Answer: 2** (delete two A's). This tests understanding that D(i,0) = i initialization represents deleting all characters.

---

## Quick Reference Card

```
┌─────────────────────────────────────────────┐
│  EDIT DISTANCE CHEAT CARD                   │
│                                             │
│  Base:  D(i,0)=i,  D(0,j)=j                │
│                                             │
│  Recurrence:                                │
│    D(i,j) = min(                            │
│      D(i-1,j) + 1,        ← delete         │
│      D(i,j-1) + 1,        ← insert         │
│      D(i-1,j-1) + cost    ← sub/match      │
│    )                                        │
│                                             │
│  cost = 0 if match                          │
│       = 1 (simple) or 2 (Levenshtein)       │
│                                             │
│  Time: O(nm)  Space: O(nm)  Trace: O(n+m)  │
│                                             │
│  Pointer: ↖=sub/match, ↑=del, ←=ins        │
└─────────────────────────────────────────────┘
```
