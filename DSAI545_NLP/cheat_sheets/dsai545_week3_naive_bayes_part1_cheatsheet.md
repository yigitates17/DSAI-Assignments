# DSAI 545 — Week 3: Text Classification & Naive Bayes (Part 1)

---

## 1. Core Concepts

### Text Classification

The task of assigning a document `d` to one class `c` from a fixed set `C = {c₁, c₂, ..., cⱼ}`. Applications: sentiment analysis, spam detection, topic categorization, language ID, authorship attribution.

Two approaches:
- **Hand-coded rules** — high accuracy if expert-tuned, but expensive to maintain.
- **Supervised ML** — learn a classifier `γ: d → c` from labeled training data `{(d₁,c₁), ..., (dₘ,cₘ)}`.

### Bag of Words (BoW)

Represent a document as an **unordered multiset** of its words, discarding position/grammar. Only word frequencies matter. This is the representation Naive Bayes operates on.

### Naive Bayes — The Two "Naive" Assumptions

1. **Bag of Words assumption** — word position doesn't matter.
2. **Conditional independence** — feature probabilities `P(xᵢ|c)` are independent given class `c`.

These assumptions are almost always **violated** in real text (words are correlated), yet NB still works surprisingly well in practice — especially with limited training data.

### Generative vs. Discriminative (context)

Naive Bayes is a **generative classifier**: it models `P(d|c)P(c)` and picks the class that most likely "generated" the document. Contrast with **discriminative classifiers** (logistic regression, SVM) that model `P(c|d)` directly.

### NB is a Linear Classifier

In log-space, the decision rule becomes `argmax[log P(c) + Σ log P(xᵢ|c)]` — a weighted sum of inputs. This makes NB a **linear classifier** that defines a hyperplane decision boundary.

---

## 2. All Formulas

### Bayes' Rule

```
P(c|d) = P(d|c) · P(c) / P(d)
```

### MAP Decision Rule

```
c_MAP = argmax  P(c|d)
         c∈C

      = argmax  P(d|c) · P(c)       ← P(d) dropped (constant across classes)
         c∈C
```

**MAP** = Maximum A Posteriori = the most likely class.

### Naive Bayes Classifier (applying conditional independence)

```
c_NB = argmax  P(c) · ∏ P(xᵢ|c)
        c∈C        i∈positions
```

Where `positions` = all word positions in the document (each word token contributes).

### Log-Space Version (used in practice)

```
c_NB = argmax [ log P(c) + Σ log P(xᵢ|c) ]
        c∈C              i∈positions
```

Why: multiplying many small probabilities → floating-point underflow. Log converts products to sums. Log is monotonic so ranking is preserved.

### Prior Estimation (MLE)

```
P̂(cⱼ) = N_cⱼ / N_total
```

- `N_cⱼ` = number of documents with class `cⱼ`
- `N_total` = total number of training documents

### Likelihood Estimation (MLE)

```
P̂(wᵢ|cⱼ) = count(wᵢ, cⱼ) / Σ_{w∈V} count(w, cⱼ)
```

- Numerator: how many times `wᵢ` appears in all documents of class `cⱼ` (the "mega-document")
- Denominator: total word count across all documents of class `cⱼ`

### Laplace (Add-1) Smoothing

```
P̂(wᵢ|c) = (count(wᵢ, c) + 1) / (Σ_{w∈V} count(w, c) + |V|)
```

More generally with smoothing parameter `α`:

```
P̂(wᵢ|c) = (count(wᵢ, c) + α) / (Σ_{w∈V} count(w, c) + α·|V|)
```

- `|V|` = vocabulary size (number of unique word types in **all** training data)
- Ensures no probability is ever zero

---

## 3. Worked Example (from Lecture Slide 28-29)

### Training Data

| # | Cat | Document |
|---|-----|----------|
| 1 | −   | just plain boring |
| 2 | −   | entirely predictable and lacks energy |
| 3 | −   | no surprises and very few laughs |
| 4 | +   | very powerful |
| 5 | +   | the most fun film of the summer |

**Test:** `predictable with no fun` → classify as + or −?

### Step 1: Compute Priors

```
P(−) = 3/5 = 0.6
P(+) = 2/5 = 0.4
```

### Step 2: Build Vocabulary

Collect all unique words from training:
`{just, plain, boring, entirely, predictable, and, lacks, energy, no, surprises, very, few, laughs, powerful, the, most, fun, film, of, summer}`

**|V| = 20**

### Step 3: Handle Unknown Words

"with" is NOT in vocabulary → **drop it** from the test document.

Test becomes: `predictable no fun`

### Step 4: Count Words per Class (Mega-documents)

**Negative mega-doc** (docs 1+2+3):
`just plain boring entirely predictable and lacks energy no surprises and very few laughs`
→ Total word count in (−) = **14**

**Positive mega-doc** (docs 4+5):
`very powerful the most fun film of the summer`
→ Total word count in (+) = **9**

### Step 5: Compute Likelihoods with Add-1 Smoothing

| Word | count(w, −) | P(w\|−) = (count+1)/(14+20) | count(w, +) | P(w\|+) = (count+1)/(9+20) |
|------|------------|---------------------------|------------|--------------------------|
| predictable | 1 | 2/34 | 0 | 1/29 |
| no | 1 | 2/34 | 0 | 1/29 |
| fun | 0 | 1/34 | 1 | 2/29 |

### Step 6: Score Each Class

```
P(−) · P(S|−) = 3/5 × (2/34) × (2/34) × (1/34)
              = 0.6 × 2 × 2 × 1 / 34³
              = 0.6 × 4 / 39304
              = 2.4 / 39304
              ≈ 6.1 × 10⁻⁵

P(+) · P(S|+) = 2/5 × (1/29) × (1/29) × (2/29)
              = 0.4 × 1 × 1 × 2 / 29³
              = 0.4 × 2 / 24389
              = 0.8 / 24389
              ≈ 3.3 × 10⁻⁵
```

### Step 7: Decision

```
6.1 × 10⁻⁵  >  3.3 × 10⁻⁵   →   c_NB = Negative (−)
```

The model classifies "predictable no fun" as **negative**. Makes sense — "predictable" and "no" both appeared in negative training docs.

---

## 4. Binary Multinomial Naive Bayes

### Why?

For sentiment analysis, whether a word **occurs** matters more than **how many times** it occurs. Seeing "fantastic" once is informative; seeing it 5 times doesn't add much.

### How it differs

**Per-document binarization:** Before building the mega-document, clip each word's count to 1 **within each document**. Then concatenate.

This means the mega-document binary count for a word can still be > 1 (if the word appears in multiple docs of that class), but never > (number of docs in that class).

### Example (Slide 33-34)

| Word | NB Count (+) | NB Count (−) | Binary Count (+) | Binary Count (−) |
|------|:---:|:---:|:---:|:---:|
| great | 3 | 1 | 2 | 1 |
| scenes | 1 | 2 | 1 | 2 |
| the | 0 | 2 | 0 | 1 |

Notice `great` has NB count 3 in (+) because one doc repeated it, but binary count is 2 because it appeared in 2 positive docs.

**Key trap:** Binary counts can exceed 1! Binarization is **within-doc**, not across the whole class.

---

## 5. Dealing with Negation

### Simple Baseline (Pang et al., 2002)

Prepend `NOT_` to every word between a negation word and the next punctuation:

```
didn't like this movie , but I
  →  didn't NOT_like NOT_this NOT_movie , but I
```

This effectively doubles the vocabulary — `like` and `NOT_like` become separate features.

### Advanced (Barnes et al., 2021)

Multi-task learning with BiLSTMs + CRFs: jointly learn sentiment classification and negation detection.

---

## 6. Key Distinctions & Exam Traps

| Trap | Clarification |
|------|--------------|
| **Why drop P(d)?** | P(d) is constant across all classes during argmax. It doesn't affect the ranking. |
| **Vocabulary V** | Built from **all training docs** (both classes), not just one class. |
| **Unknown words at test time** | Simply **remove** them. Don't assign any probability. |
| **Add-1 denominator** | It's `Σ count(w,c) + |V|`, NOT `Σ count(w,c) + 1`. You add `|V|` to the denominator because you added 1 to each of the `|V|` words. |
| **Binary NB ≠ Bernoulli NB** | Binary Multinomial NB clips counts to 1 per doc then uses the multinomial formula. Bernoulli NB models word presence/absence with a different formula. |
| **Binary counts can be > 1** | Binarization is per-document. A word appearing in 3 negative docs gets binary count 3 for class (−). |
| **Stop words** | Most NB systems do **not** remove them — it usually doesn't help. |
| **Log-space** | Always work in log-space in practice. Log is monotonic → same argmax result. |
| **NB parameter count** | Without independence: O(\|X\|ⁿ · \|C\|) — intractable. With independence: O(\|X\| · \|C\|) — just count frequencies. |
| **Smoothing α** | Lecture uses α=1 (Laplace). In practice α can be tuned (often α < 1 works better). |

---

## 7. Quick Reference Card

```
╔══════════════════════════════════════════════════════════════════╗
║                   NAIVE BAYES CHEAT SHEET                       ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  c_NB = argmax [ log P(c) + Σᵢ log P(xᵢ|c) ]                  ║
║          c∈C                                                     ║
║                                                                  ║
║  Prior:      P̂(c) = N_c / N_total                               ║
║                                                                  ║
║  Likelihood: P̂(w|c) = count(w,c) + α                           ║
║                        ─────────────────────                     ║
║                        Σ_V count(w,c) + α|V|                    ║
║                                                                  ║
║  Laplace:    α = 1                                               ║
║  Vocab |V|:  unique words across ALL training docs               ║
║  Unknown w:  drop from test doc (ignore entirely)                ║
║  Binary NB:  clip word count to 1 PER DOC before mega-doc        ║
║                                                                  ║
║  KEY IDENTITIES:                                                 ║
║  • log(ab) = log(a) + log(b)   → products become sums           ║
║  • argmax is preserved under log (monotonic)                     ║
║  • NB = linear classifier in log-space                           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 8. Practice Midterm Questions

---

### Q1 — Prior Computation (Easy)

Given 800 training documents: 500 are spam, 300 are ham. What are P(spam) and P(ham)?

<details>
<summary>Solution</summary>

```
P(spam) = 500/800 = 0.625
P(ham)  = 300/800 = 0.375
```
</details>

---

### Q2 — Likelihood with Smoothing (Easy)

The word "free" appears 80 times in spam docs and 5 times in ham docs. Total words in spam mega-doc: 10,000. Total words in ham mega-doc: 8,000. Vocabulary size |V| = 50,000. Using add-1 smoothing, compute P("free"|spam) and P("free"|ham).

<details>
<summary>Solution</summary>

```
P("free"|spam) = (80 + 1) / (10000 + 50000) = 81/60000 = 0.00135
P("free"|ham)  = (5 + 1)  / (8000 + 50000)  = 6/58000  = 0.000103
```

Note how smoothing has a big impact when |V| is large — it pushes all probabilities toward uniform.
</details>

---

### Q3 — Full NB Classification (Medium)

Training data:

| Doc | Class | Text |
|-----|-------|------|
| 1 | sports | goal kick player |
| 2 | sports | goal score |
| 3 | tech | code bug fix |
| 4 | tech | code deploy server fix |

Classify the test document: **"goal code"** using Multinomial NB with add-1 smoothing.

<details>
<summary>Solution</summary>

**Vocabulary:** {goal, kick, player, score, code, bug, fix, deploy, server} → |V| = 9

**Priors:**
```
P(sports) = 2/4 = 0.5
P(tech)   = 2/4 = 0.5
```

**Mega-documents:**
- Sports: goal kick player goal score → total = 5
- Tech: code bug fix code deploy server fix → total = 7

**Likelihoods (add-1):**

| Word | count(w, sports) | P(w\|sports) = (c+1)/(5+9) | count(w, tech) | P(w\|tech) = (c+1)/(7+9) |
|------|:---:|:---:|:---:|:---:|
| goal | 2 | 3/14 | 0 | 1/16 |
| code | 0 | 1/14 | 2 | 3/16 |

**Scoring:**
```
P(sports) × P(goal|sports) × P(code|sports) = 0.5 × (3/14) × (1/14)
    = 0.5 × 3/196 = 1.5/196 ≈ 0.00765

P(tech) × P(goal|tech) × P(code|tech) = 0.5 × (1/16) × (3/16)
    = 0.5 × 3/256 = 1.5/256 ≈ 0.00586
```

**Decision: c_NB = sports** (0.00765 > 0.00586)

"goal" is a strong sports indicator that outweighs "code" being a tech indicator here.
</details>

---

### Q4 — Log-Space Computation (Medium)

Redo Q3 in log-space. Verify you get the same answer.

<details>
<summary>Solution</summary>

```
log P(sports) + log P(goal|sports) + log P(code|sports)
= log(0.5) + log(3/14) + log(1/14)
= -0.693 + (-1.540) + (-2.639)
= -4.872

log P(tech) + log P(goal|tech) + log P(code|tech)
= log(0.5) + log(1/16) + log(3/16)
= -0.693 + (-2.773) + (-1.674)
= -5.140
```

**-4.872 > -5.140 → sports wins.** Same result. (Less negative = higher log-probability.)
</details>

---

### Q5 — Binary NB vs Regular NB (Medium)

Consider one positive document: **"great great great movie"**

(a) What is the word count for "great" in regular Multinomial NB?
(b) What is the word count for "great" in Binary Multinomial NB?
(c) If there are 2 positive documents and the other one also contains "great" once, what is the Binary count for "great" in the positive mega-document?

<details>
<summary>Solution</summary>

(a) **3** — we count every occurrence.

(b) **1** — we clip to 1 per document.

(c) **2** — Doc 1 contributes 1 (clipped from 3), Doc 2 contributes 1. Binary mega-doc count = 1 + 1 = **2**. Binarization is within-doc, not across the whole class.
</details>

---

### Q6 — Zero Probability Trap (Medium)

A NB classifier has P("amazing"|positive) = 0 (the word "amazing" never appeared in positive training docs) without smoothing. A test document is: "This movie is amazing and wonderful." Even if every other word strongly indicates positive, what will the classifier predict? Why?

<details>
<summary>Solution</summary>

The classifier will **never predict positive** for this document regardless of other words.

Because: `P(+) × ... × P("amazing"|+) × ... = P(+) × ... × 0 × ... = 0`

Any product containing a zero factor is zero. This is exactly why **Laplace smoothing** is essential — it ensures no word-class probability is ever zero.
</details>

---

### Q7 — Vocabulary and Unknown Words (Medium)

Training: Class A has docs {"cat dog"}, Class B has docs {"fish bird dog"}. Test doc: **"cat fish hamster"**.

(a) What is |V|?
(b) Which test words are unknown?
(c) What does the test document become after handling unknowns?

<details>
<summary>Solution</summary>

(a) V = {cat, dog, fish, bird} → **|V| = 4**

(b) **"hamster"** is not in V → it's unknown.

(c) Remove "hamster" → test becomes **"cat fish"**
</details>

---

### Q8 — Conceptual: Why "Naive"? (Short Answer)

Explain why Naive Bayes is called "naive" and give a concrete example of when this assumption is violated in text.

<details>
<summary>Solution</summary>

It's "naive" because it assumes all features (words) are **conditionally independent** given the class. Formally: P(x₁, x₂, ..., xₙ | c) = ∏ P(xᵢ | c).

**Concrete violation:** In a movie review, if "Hong" appears, "Kong" is very likely to follow. P("Kong" | positive) is not independent of P("Hong" | positive). Similarly, "New" and "York" are heavily correlated. The independence assumption treats them as unrelated.

Despite this, NB works well because: (1) we only need the **ranking** of classes to be correct, not the exact probabilities; (2) the errors from independence often cancel out across many features.
</details>

---

### Q9 — Tricky: Smoothing Denominator (Hard)

Training data for class C₁ has a mega-document with these word counts: {the: 50, cat: 10, dog: 15, fish: 5}. Total vocabulary across all classes: V = {the, cat, dog, fish, bird, car, house}. Compute P("bird" | C₁) with add-1 smoothing.

<details>
<summary>Solution</summary>

```
count("bird", C₁) = 0
Σ count(w, C₁) = 50 + 10 + 15 + 5 = 80
|V| = 7

P("bird" | C₁) = (0 + 1) / (80 + 7) = 1/87 ≈ 0.0115
```

**Common mistake:** Using |V| = 4 (only words seen in C₁). Wrong! |V| is the vocabulary from **all training data across all classes**, which is 7 here. The denominator must account for smoothing counts added to all |V| words.
</details>

---

### Q10 — Full Pipeline with Binary NB (Hard)

Using the exact training data from slide 28:

| # | Cat | Document |
|---|-----|----------|
| 1 | −   | just plain boring |
| 2 | −   | entirely predictable and lacks energy |
| 3 | −   | no surprises and very few laughs |
| 4 | +   | very powerful |
| 5 | +   | the most fun film of the summer |

Classify **"very fun and boring"** using **Binary Multinomial NB** with add-1 smoothing.

<details>
<summary>Solution</summary>

**Step 1: Vocabulary** — same |V| = 20 (all unique words across training). All test words are in V, so no drops needed.

**Step 2: Priors**
```
P(−) = 3/5,  P(+) = 2/5
```

**Step 3: Binarize training documents** (clip to 1 per word per doc, then build mega-docs)

Docs are already all unique words within each doc (no repeats in this dataset), so binary counts = regular counts here.

Negative binary mega-doc: just plain boring entirely predictable and lacks energy no surprises and very few laughs
→ "and" appears in doc 2 and doc 3 → binary count = 2 (but regular count is also 2 here)
→ Total tokens in (−) binary mega-doc = 14

Positive binary mega-doc: very powerful the most fun film of the summer
→ "the" appears twice in doc 5 → binary count = 1 (clipped!)
→ Total tokens in (+) binary mega-doc = 9 − 1 = **8**

**Step 4: Compute likelihoods**

| Word | bin_count(w, −) | P(w\|−) = (c+1)/(14+20) | bin_count(w, +) | P(w\|+) = (c+1)/(8+20) |
|------|:---:|:---:|:---:|:---:|
| very | 1 | 2/34 | 1 | 2/28 |
| fun | 0 | 1/34 | 1 | 2/28 |
| and | 2 | 3/34 | 0 | 1/28 |
| boring | 1 | 2/34 | 0 | 1/28 |

**Step 5: Binarize test doc** — all words are unique already → stays "very fun and boring"

**Step 6: Score**
```
P(−) × P(very|−) × P(fun|−) × P(and|−) × P(boring|−)
= (3/5) × (2/34) × (1/34) × (3/34) × (2/34)
= 0.6 × 12 / 34⁴
= 7.2 / 1,336,336
≈ 5.39 × 10⁻⁶

P(+) × P(very|+) × P(fun|+) × P(and|+) × P(boring|+)
= (2/5) × (2/28) × (2/28) × (1/28) × (1/28)
= 0.4 × 4 / 28⁴
= 1.6 / 614,656
≈ 2.60 × 10⁻⁶
```

**Decision: c_NB = Negative (−)**

5.39 × 10⁻⁶ > 2.60 × 10⁻⁶ → Negative wins. The words "boring" and "and" (which appears twice in negative docs) tip the scale.

**Key difference from regular NB:** In binary NB the positive mega-doc has 8 tokens instead of 9 (because "the" was clipped in doc 5). This changes all positive likelihoods slightly.
</details>

---

*DSAI 545 — Boğaziçi University — Week 3, Part 1*
