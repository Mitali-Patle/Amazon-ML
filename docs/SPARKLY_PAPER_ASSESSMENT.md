# Sparkly (PVLDB 2023) vs. our blocking implementation

**Paper:** Paulsen, Govind, Doan — *Sparkly: A Simple yet Surprisingly Strong TF/IDF Blocker for Entity Matching*, PVLDB 16(6): 1507–1519, 2023. [doi:10.14778/3583140.3583163](https://doi.org/10.14778/3583140.3583163) · [open PDF](https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf)

**Verdict:** the paper independently validates the three load-bearing decisions in our retrieval design. It identifies two concrete refinements we have not implemented. Both are recommended as **post-baseline experiments, not now** — see §4 for why.

---

## 1. What the paper claims

Sparkly does top-k TF/IDF blocking using Lucene (BM25) on a Spark cluster, and **outperforms 8 state-of-the-art blockers including deep-learning ones**, at smaller output size and much faster. Their stated takeaway: "tf/idf blocking needs more attention, and Sparkly forms a strong baseline that future blocking work should compare against."

---

## 2. Where our implementation already agrees

These were reached independently from our own measurements, before reading the paper.

| Sparkly finding | Our implementation | Our evidence |
|---|---|---|
| **"Use top-k instead of thresholding — this is the most important decision that we made."** Thresholds are impossible to tune on noisy data: high α kills recall, low α blows up output size unpredictably. | Top-k retrieval | We rejected threshold-union after measuring **78 billion candidate pairs** at df<50k (H6) |
| **One-sided top-k.** Two-sided "can significantly increase runtime yet improve recall only minimally." | One-sided (S1 → pool) | Adopted by construction; the task asks for candidates *per Source-1 entity* |
| **TF/IDF beats DL blockers** at smaller output size | Whole retrieval stack is TF/IDF | Recall@50 = 94.4% at 82,667–123,738× reduction |
| Evaluate blockers on **recall, output size, runtime** | All three instrumented per country and per script | `phase8b_country_blocking.json` |
| Real data is "so noisy that similarity scores of many matching tuple pairs can be quite low" | Directly observed | 14.4% of true pairs have **zero** name-token overlap; rescued by the address channel |

Three of the paper's headline conclusions match decisions we made on our own data. That is meaningful independent corroboration of the architecture.

---

## 3. Two genuine gaps

### 3.1 BM25 instead of plain IDF-sum — **recommended**

Sparkly scores with Okapi BM25 (Lucene's default):

```
s(D,Q) = Σ_{t∈Q}  tf(t,D)·(k1+1) / ( tf(t,D) + k1·(1 - b + b·|D|/avgdl) )  ·  idf(t)
         idf(t) = log( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 ),  k1 ∈ [1.2, 2.0],  b = 0.75
```

Ours is `Σ idf(t)` over shared tokens — **binary term frequency and no document-length normalization**. Two differences that matter:

- **Length normalization (`b`).** Without it, long records score higher purely for having more tokens to match on. Our address fields range from empty to 249 characters, so this bias is real and unmeasured.
- **TF saturation (`k1`).** Low impact here — business names rarely repeat a token.

The IDF formula also differs slightly; ours is clamped at 0 (see `candidates.py`), theirs is smoothed and always positive.

**Expected effect:** improved *ranking* within the candidate pool. Cheap to implement — it only changes the weights in `to_matrix`, not the architecture.

### 3.2 Index the smaller table, probe from the larger — **interesting, unproven for us**

Sparkly indexes the *smaller* table and probes from the larger, reporting "probing from the larger table rather than the smaller one tends to produce higher recall, given the same k value."

We do the opposite: index the pool (10.3M) and probe with Source 1 (2.2M).

Reversing would mean indexing S1 and asking each S2/S3 record "which S1 entity do I belong to?", then inverting to per-entity candidate lists. **This fits our verified many-to-one structure unusually well** — every pool record belongs to at most one S1 entity, so that question has exactly one right answer. It is arguably the more natural formulation of our problem.

Caveat: it changes candidate-set semantics. An S1 entity would accumulate a variable number of candidates rather than exactly k, so `candidate_pairs.tsv` sizing and the decision layer's assumptions would both need rechecking.

### 3.3 Noted but not applicable

- **3-gram tokenization.** Sparkly Manual concatenates attributes and tokenizes into 3-grams. We tested char-n-grams as an *additional scoring channel* and recall collapsed 90.1% → 27.6% (D7) — but that was a fusion failure, not a verdict on n-grams as a primary tokenization. Untested in Sparkly's form.
- **Sparkly Auto** selects (attribute, tokenizer) configs by maximizing "discriminativeness" (normalized AUC of the top-k score curve; steeper slope = more discriminative). Elegant, but we have only two text attributes and already know both are needed — name alone collides for 31.06% of S1 rows.
- **Spark/Lucene distribution.** Irrelevant: we are on one machine and already run in ~9 min per 110k queries after the 2.26× optimization.

---

## 4. Why neither change is being made now

The oracle analysis (`data/reports/oracle_ceiling.json`) settles the priority question:

| scenario | macro F_0.5 |
|---|---|
| **oracle — perfect matcher on current candidates** | **0.9833** |
| oracle without singleton detection | 0.9276 |
| predict nothing | 0.0557 |
| predict all 100 candidates | 0.0409 |

Current candidates already support **0.9833**, and **87.3% of validation entities are perfectly solvable today**. Lifting recall from 95.4% to 97% would move that ceiling by well under one point.

Meanwhile the spread between a good matcher and a careless one is nearly the entire score — taking all 100 candidates scores 0.0409, *worse than predicting nothing at all*.

**Candidate generation is no longer the bottleneck.** Sparkly's refinements are good ideas aimed at a component that is already ahead of where the score is being lost. They belong in the queue after a working matcher exists, where their effect can be measured against F_0.5 rather than against recall.

---

## 5. If revisited later, in priority order

1. **BM25 weights** in `to_matrix` — the pool-side values change; the `side="pool"`/`side="query"` split already isolates exactly this. Measure against the fixed reference numbers (India R@50 = 90.97%, US = 96.67%) using `exp_pool_size.py` as the harness.
2. **Length normalization alone** (`b=0.75`, no TF saturation) — isolates the change most likely to matter, since our records vary widely in length.
3. **Reverse indexing direction** — largest potential gain, largest blast radius. Would need `candidate_pairs.tsv` sizing and the decision layer both re-verified.
