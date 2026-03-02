# DSAI 545 – N-gram Language Modeling Cheat Sheet & Practice Questions

---

## 1. Core Goal

A **language model (LM)** assigns a probability to a sentence or sequence of words. Two formulations:

- **Joint probability:** P(W) = P(w₁, w₂, ..., wₙ)
- **Next word prediction:** P(wₙ | w₁, w₂, ..., wₙ₋₁)

**Applications:** machine translation, spell correction, speech recognition, summarization, question answering.

---

## 2. Chain Rule of Probability

To compute joint probability, decompose it:

```
P(w₁, w₂, ..., wₙ) = P(w₁) × P(w₂|w₁) × P(w₃|w₁,w₂) × ... × P(wₙ|w₁,...,wₙ₋₁)

Compact:  P(w₁...wₙ) = ∏ᵢ P(wᵢ | w₁...wᵢ₋₁)
```

**Problem:** We can't estimate P(wₙ | w₁...wₙ₋₁) directly — too many possible histories, we'll never see most of them in any corpus.

---

## 3. Markov Assumption

**Key idea:** Instead of conditioning on the entire history, approximate by only looking at the last **k** words:

```
P(wᵢ | w₁...wᵢ₋₁) ≈ P(wᵢ | wᵢ₋ₖ...wᵢ₋₁)
```

This gives us **N-gram models** where N = k + 1:

| Model | Conditions on | Formula |
|-------|--------------|---------|
| **Unigram** (N=1) | Nothing | P(wᵢ) |
| **Bigram** (N=2) | Previous 1 word | P(wᵢ \| wᵢ₋₁) |
| **Trigram** (N=3) | Previous 2 words | P(wᵢ \| wᵢ₋₂, wᵢ₋₁) |

Higher N → more context → better predictions, but much more data needed.

---

## 4. Estimating N-gram Probabilities (MLE)

**Maximum Likelihood Estimation** — just count and divide:

### Bigram:
```
P(wᵢ | wᵢ₋₁) = C(wᵢ₋₁, wᵢ) / C(wᵢ₋₁)
```

### Trigram:
```
P(wᵢ | wᵢ₋₂, wᵢ₋₁) = C(wᵢ₋₂, wᵢ₋₁, wᵢ) / C(wᵢ₋₂, wᵢ₋₁)
```

### Worked Example

Corpus:
```
<s> I am Sam </s>
<s> Sam I am </s>
<s> I do not like green eggs and ham </s>
```

**Bigram calculations:**

- P(I | \<s\>) = C(\<s\>, I) / C(\<s\>) = 2/3 = 0.67
- P(Sam | \<s\>) = C(\<s\>, Sam) / C(\<s\>) = 1/3 = 0.33
- P(am | I) = C(I, am) / C(I) = 2/3 = 0.67
- P(do | I) = C(I, do) / C(I) = 1/3 = 0.33
- P(\</s\> | Sam) = C(Sam, \</s\>) / C(Sam) = 1/2 = 0.5
- P(Sam | am) = C(am, Sam) / C(am) = 1/2 = 0.5

---

## 5. Sentence Probability Computation

**Always include \<s\> and \</s\> tokens!**

Using bigrams:
```
P(<s> I am Sam </s>) = P(I|<s>) × P(am|I) × P(Sam|am) × P(</s>|Sam)
                     = 0.67 × 0.67 × 0.5 × 0.5
                     = 0.112
```

---

## 6. Practical Issue: Log Space

Multiplying many small probabilities → numerical underflow. Solution: work in log space.

```
log(p₁ × p₂ × p₃ × p₄) = log(p₁) + log(p₂) + log(p₃) + log(p₄)
```

**Benefits:** avoids underflow, addition is faster than multiplication.

---

## 7. N-gram Limitations

- **Long-distance dependencies:** "The computer which I had just put into the machine room on the fifth floor **crashed**." — N-grams can't capture this.
- **Sparsity:** Many valid N-grams will have zero counts in training data.
- **Unigrams** produce word-salad (no word order).
- **Bigrams** capture local patterns but miss broader context.
- Higher N-grams need exponentially more data.

---

## 8. Evaluation

### Extrinsic (in-vivo)
- Put the LM into a real task (MT, speech recognition)
- Measure task performance (translation accuracy, WER, etc.)
- Expensive and time-consuming

### Intrinsic (in-vitro)
- Measure the LM directly using **perplexity**
- Cheaper, single metric, but doesn't always correlate with task performance

---

## 9. Train / Dev / Test Split

| Set | Purpose |
|-----|---------|
| **Training** | Learn model parameters (count N-grams) |
| **Dev (devset)** | Tune hyperparameters, compare models during development |
| **Test** | Final evaluation, run only once |

**Train/Test Contamination:** If test sentences leak into training, the model gets artificially high scores. Never allow this.

**Dev set rationale:** If you test on the test set many times, you implicitly tune to it. Dev set protects the test set's integrity.

---

## 10. Perplexity (PP)

### Formula

```
PP(W) = P(w₁w₂...wₙ)^(-1/N) = ᴺ√(1 / P(w₁w₂...wₙ))
```

### With Bigrams (Chain Rule applied)

```
PP(W) = ᴺ√(∏ᵢ 1/P(wᵢ|wᵢ₋₁))
```

### Intuition

- Perplexity = how "surprised" the model is by the test data
- **Lower perplexity = better model** (less surprised)
- Perplexity range: [1, ∞)
- Probability range: [0, 1]
- **Minimizing perplexity = maximizing probability**

### Reference Values (WSJ, 38M training words)

| Model | Perplexity |
|-------|-----------|
| Unigram | 962 |
| Bigram | 170 |
| Trigram | 109 |

More context → lower perplexity → better model.

### Why not just use raw probability?

Raw probability depends on text length — longer text = smaller probability regardless of model quality. Perplexity normalizes by N (number of words), making it comparable across different-length test sets.

---

## 11. Key Formulas to Memorize

```
Chain Rule:     P(w₁...wₙ) = ∏ P(wᵢ | w₁...wᵢ₋₁)

Markov:         P(wᵢ | w₁...wᵢ₋₁) ≈ P(wᵢ | wᵢ₋ₖ...wᵢ₋₁)

Bigram MLE:     P(wᵢ | wᵢ₋₁) = C(wᵢ₋₁, wᵢ) / C(wᵢ₋₁)

Perplexity:     PP(W) = P(w₁...wₙ)^(-1/N)

Log space:      log(p₁ × p₂ × ... × pₙ) = Σ log(pᵢ)
```

---

# Practice Midterm Questions

---

## Q1: Bigram Probability Computation (Easy)

**Given this corpus:**
```
<s> the cat sat </s>
<s> the cat ate </s>
<s> the dog sat </s>
```

**Compute:** P(cat | the), P(sat | cat), P(ate | cat)

### Solution

**P(cat | the)** = C(the, cat) / C(the) = 2 / 3 = **0.67**

"the" appears 3 times. "the cat" appears 2 times.

**P(sat | cat)** = C(cat, sat) / C(cat) = 1 / 2 = **0.5**

"cat" appears 2 times. "cat sat" appears 1 time.

**P(ate | cat)** = C(cat, ate) / C(cat) = 1 / 2 = **0.5**

Note: P(sat|cat) + P(ate|cat) = 1.0 — the probabilities of all words following "cat" must sum to 1.

---

## Q2: Sentence Probability (Medium)

**Using the corpus from Q1, compute the bigram probability of:**
```
<s> the cat sat </s>
```

### Solution

```
P(<s> the cat sat </s>) = P(the|<s>) × P(cat|the) × P(sat|cat) × P(</s>|sat)
```

Step by step:
- P(the | \<s\>) = C(\<s\>, the) / C(\<s\>) = 3/3 = 1.0
- P(cat | the) = 2/3 = 0.67
- P(sat | cat) = 1/2 = 0.5
- P(\</s\> | sat) = C(sat, \</s\>) / C(sat) = 2/2 = 1.0

```
P = 1.0 × 0.67 × 0.5 × 1.0 = 0.333
```

**Answer: 0.333**

---

## Q3: Log Probabilities (Medium)

**Convert the sentence probability from Q2 to log space (base 2).**

### Solution

```
log₂(P) = log₂(1.0) + log₂(0.67) + log₂(0.5) + log₂(1.0)
        = 0 + (-0.585) + (-1.0) + 0
        = -1.585
```

**Answer: -1.585**

To recover: 2^(-1.585) = 0.333 ✓

> **Exam tip:** Log probabilities are always negative (since probabilities are between 0 and 1). Less negative = higher probability = better.

---

## Q4: Perplexity Computation (Medium-Hard)

**Compute the perplexity of the sentence "\<s\> the cat sat \</s\>" using the bigram model from Q1.**

### Solution

N = 4 (counting: the, cat, sat, \</s\> — we count \</s\> but not \<s\>)

```
PP = P(sentence)^(-1/N)
   = (0.333)^(-1/4)
   = (1/0.333)^(1/4)
   = (3.0)^(0.25)
   = 1.316
```

**Answer: PP ≈ 1.316**

This is very low perplexity (close to 1), which makes sense — this exact sentence appears in the training corpus, so the model is not surprised at all.

---

## Q5: Comparing Models via Perplexity (Medium)

**Model A has perplexity 150 on a test set. Model B has perplexity 95. Which is better and why?**

### Solution

**Model B is better.** Lower perplexity means the model assigns higher probability to the test set — it's less "surprised" by the data. Perplexity of 95 means the model is, on average, as uncertain as choosing uniformly among 95 words at each step, versus 150 for Model A.

---

## Q6: Zero Probability Problem (Medium)

**Given the corpus from Q1, what is P(\<s\> the cat ran \</s\>)?**

### Solution

```
P = P(the|<s>) × P(cat|the) × P(ran|cat) × P(</s>|ran)
```

P(ran | cat) = C(cat, ran) / C(cat) = 0/2 = **0**

The entire sentence probability = **0**.

**Why this is a problem:** "the cat ran" is a perfectly valid English sentence, but because "cat ran" never appeared in our tiny corpus, the model assigns it zero probability. This also makes perplexity undefined (division by zero).

> This is the **sparsity problem** — it motivates smoothing techniques (Laplace, Add-k, backoff, interpolation) which are likely covered in a future lecture.

---

## Q7: Unigram vs Bigram Understanding (Easy)

**Why does a unigram model produce nonsensical sentences while a bigram model produces slightly more coherent ones?**

### Solution

A unigram model treats each word as independent — P(W) = ∏ P(wᵢ). It picks words based only on their individual frequency, ignoring all word order. So you get "fifth, an, of, futures, the" — high frequency words in random order.

A bigram model conditions each word on the previous one — P(wᵢ | wᵢ₋₁). This captures local word-to-word transitions ("new car", "parking lot", "would be"), producing phrases that sound locally reasonable even if the overall sentence doesn't make global sense.

---

## Q8: Train/Dev/Test Conceptual (Exam-Style)

### 8a. Why do we need a dev set in addition to train and test?

If we repeatedly evaluate on the test set and adjust our model based on results, we implicitly overfit to the test set. The dev set acts as a proxy for testing — we tune on dev, then evaluate on test only once at the end, preserving its integrity as an unbiased estimate of generalization.

### 8b. What is train/test contamination and why is it dangerous?

It means test data leaked into training. The model memorizes test sentences and assigns them artificially high probabilities, making evaluation metrics (like perplexity) look much better than real-world performance. The model appears to generalize well but actually doesn't.

### 8c. Why is perplexity preferred over raw probability for evaluation?

Raw probability shrinks with sentence length — a 100-word sentence will always have lower probability than a 10-word sentence, regardless of model quality. Perplexity normalizes by the number of words (Nth root of inverse probability), making it comparable across different-length test sets. It gives a per-word measure of model confidence.

---

## Q9: MLE from Count Table (Hard)

**Given this raw bigram count table and unigram counts:**

| | i | want | to | eat |
|------|---|------|-----|-----|
| i | 5 | 827 | 0 | 9 |
| want | 2 | 0 | 608 | 1 |
| to | 2 | 0 | 4 | 686 |
| eat | 0 | 0 | 2 | 0 |

Unigram counts: i=2533, want=927, to=2417, eat=746

**Compute P(to | want) and P(eat | to). Then compute P(\<s\> i want to eat) assuming P(i | \<s\>) = 0.25.**

### Solution

**P(to | want)** = C(want, to) / C(want) = 608 / 927 = **0.656**

**P(eat | to)** = C(to, eat) / C(to) = 686 / 2417 = **0.284**

**Sentence probability:**
```
P(<s> i want to eat) = P(i|<s>) × P(want|i) × P(to|want) × P(eat|to)
                     = 0.25 × (827/2533) × (608/927) × (686/2417)
                     = 0.25 × 0.326 × 0.656 × 0.284
                     = 0.01518
```

**Answer: ≈ 0.015**

---

## Quick Reference Card

```
┌──────────────────────────────────────────────────┐
│  N-GRAM LM CHEAT CARD                           │
│                                                  │
│  Chain Rule:                                     │
│    P(w₁...wₙ) = ∏ P(wᵢ | w₁...wᵢ₋₁)           │
│                                                  │
│  Markov Assumption (bigram):                     │
│    P(wᵢ | w₁...wᵢ₋₁) ≈ P(wᵢ | wᵢ₋₁)           │
│                                                  │
│  Bigram MLE:                                     │
│    P(wᵢ|wᵢ₋₁) = C(wᵢ₋₁,wᵢ) / C(wᵢ₋₁)         │
│                                                  │
│  Perplexity:                                     │
│    PP(W) = P(w₁...wₙ)^(-1/N)                    │
│    Lower = better,  range [1, ∞)                 │
│    Min PP = Max probability                      │
│                                                  │
│  Log space: log(∏pᵢ) = Σlog(pᵢ)                 │
│    Avoids underflow, always negative             │
│                                                  │
│  Train/Dev/Test: never contaminate test          │
│  Tune on dev, evaluate on test ONCE              │
└──────────────────────────────────────────────────┘
```
