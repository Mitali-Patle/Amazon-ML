# COUNTRY BLOCKING STRATEGY

**Date:** 25 Sep 2026 · **Branch:** `kundhave-dev`
**Central question:** *can country be used as a computational partition without creating an unacceptable candidate-recall failure when the test set contains unseen, missing, variant, or noisy country values?*
**Answer:** yes — but only because the architecture never lets an unresolved country mean "no retrieval". The evidence and the design that earns that answer are below.

Evidence: `data/reports/exp_country_audit.json` (reproduce with `python -m src.prep.exp_country_audit`).

---

## 1. FINDINGS FROM THE ACTUAL DATA

### 1.1 Country values — exhaustive, byte level, all 6 files

| File | Distinct | Values |
|---|---|---|
| train_source1 | 2 | `'US'` 1,323,633 · `'India'` 883,188 |
| train_source2 | 2 | `'US'` 3,016,817 · `'India'` 2,017,799 |
| train_source3 | 2 | `'US'` 3,170,056 · `'India'` 2,115,547 |
| test_source1 | 3 | `'India'` 809,986 · `'US'` 663,106 · `'France'` 259,452 |
| test_source2 | 3 | `'India'` 2,312,565 · `'US'` 1,871,330 · `'France'` 703,378 |
| test_source3 | 3 | `'India'` 2,405,000 · `'US'` 1,945,701 · `'France'` 731,615 |

**Across all 24.2M records there are exactly three distinct values: `US`, `India`, `France`.**

| Contamination check | Result |
|---|---|
| Empty / whitespace-only country | **0** |
| Leading or trailing whitespace | **none** |
| Casing variants | **none** |
| NFKC-unstable values | **none** |
| Values collapsing under strip+casefold | **none** |
| Spelling variants (`USA`, `United States`, `IND`, …) | **none present** |
| Abbreviation variants | **none present** |

The field is effectively an enum. There is no `USA` vs `US` vs `United States` problem **in this data**.

### 1.2 Known vs unseen

- In test but not train: **`France`** (259,452 S1 entities = 14.97% of the test set).
- In train but not test: **none**.

### 1.3 Partition coverage — the failure mode that actually matters

An S1 country with an empty S2/S3 pool would silently yield zero candidates for every entity carrying it. Checked explicitly:

| Split | Country | S1 | S2 | S3 | pool/S1 |
|---|---|---|---|---|---|
| train | India | 883,188 | 2,017,799 | 2,115,547 | 4.68 |
| train | US | 1,323,633 | 3,016,817 | 3,170,056 | 4.67 |
| test | France | 259,452 | **703,378** | **731,615** | 5.53 |
| test | India | 809,986 | 2,312,565 | 2,405,000 | 5.82 |
| test | US | 663,106 | 1,871,330 | 1,945,701 | 5.76 |

**No country has an empty or one-sided pool.** France is fully populated in both S2 and S3 with a healthy 5.53 ratio — comparable to US (5.76) and India (5.82). France is therefore an *ordinary retrieval partition at inference*; it lacks training **labels**, not retrieval **data**.

### 1.4 Hard-block cost on the evidence we have

Exhaustive over all ground-truth pairs:

```
cross-country true pairs: 0 / 7,638,365  =  0.000000%
rule-of-three 95% upper bound given zero observations: 0.000039%
```

Even in the worst case consistent with the data, a hard country block costs **under 4 in 100,000** true pairs.

---

## 2. HARD-BLOCK SAFETY ANALYSIS

The four questions kept separate, as required:

**A. Training evidence — strong.** 7,638,365/7,638,365 pairs agree on country; 0 entities span countries. Upper bound on loss 0.000039%.

**B. Test-time structural risk — low but unverifiable.** Test has no labels, so test-side losslessness cannot be proven by any method. The mitigation is architectural, not statistical: country is used as a *filter with a fallback*, so even if the assumption broke, affected records would still retrieve rather than return nothing. The residual exposure is that a cross-country true pair would be missed — not that a record is dropped.

**C. Unseen-country behaviour — a solved problem, not a risk.** The decisive point is that the retrieval index is **label-free**. It is built from the S2/S3 pool being searched, and the test pool contains 1.43M French records. So France gets a real index built from real test data. "No training labels" does **not** imply "no retrieval index"; conflating the two would have been the expensive mistake.

**D. Missing / variant country — absent from this data, still handled.** Zero empty and zero variant values exist. A defensive layer is implemented anyway because the failure is silent and the cost is negligible.

**Verdict:** country is safe as a computational partition, conditional on the architecture in §3 — specifically on unresolved countries falling back rather than failing.

---

## 3. WHAT WAS WRONG WITH THE EXISTING IMPLEMENTATION

Inspected before changing anything (`src/prep/run_validation.py`).

| # | Problem | Why it is wrong | Severity |
|---|---|---|---|
| 1 | **No fallback of any kind** | A query whose country has no pool partition would retrieve nothing and be silently dropped. Nothing logged, no error. | **High** — the exact failure the brief targets |
| 2 | **S2 and S3 concatenated into one pool** | Violates country × source partitioning; a global top-K can be monopolised by one source even though 80.48% of entities match in both | Medium — measured in §5 |
| 3 | **No country normalisation** | `country` used as a raw string, so `"US"` and `" us "` would become separate partitions | Low here (data is clean), high if the assumption ever breaks |
| 4 | `build_vocab` on an empty partition | Would produce an empty vocabulary and zero candidates without raising | Medium |

What was already **right**: countries are enumerated dynamically via `s1["country"].unique()` — there is no hard-coded country list, so France would have received a partition automatically.

---

## 4. APPROVED DESIGN

```
S1 record
    |
normalize_country()  -> (country_key, status)
    |
resolve against partitions PRESENT IN THE POOL   <-- not against training countries
    |
    +---------------------------+
    |                           |
partition exists          missing / unknown-in-pool
    |                           |
country x source index    country-AGNOSTIC index for that source
(S2 and S3 separately)    (same machinery, no country restriction)
    |                           |
    +---------------------------+
                |
        top-K per source -> merge, dedup, score-sort
```

### 4.1 Country normalisation — deliberately minimal
Because the data demonstrates no variants, the layer is thin and auditable rather than an invented dictionary: NFKC → strip → collapse whitespace → casefold → a small alias table covering only the three observed countries. An **unrecognised value is never discarded** — it keeps its own canonical spelling and becomes its own partition key; the partitioner then decides whether such a partition exists.

Fields are additive, raw preserved: `country` (raw) · `country_key` (partition key) · `country_status` ∈ `{ok, aliased, missing, unknown}`.

### 4.2 Routing rules
| Condition | Path | Pool searched |
|---|---|---|
| `country_key` has a partition in the pool | partition | that country × source |
| `country_status = missing` | fallback | entire source pool |
| `country_key` not in pool | fallback | entire source pool |

**Known vs unseen is decided against the pool, never against the training countries.** France resolves as a normal partition at inference.

### 4.3 Fallback bounding
The fallback uses the identical sparse-retrieval machinery with the country restriction lifted, and returns the same top-K. It is **never** a Cartesian product: cost is one extra index build per source plus K candidates per affected query. It is built **lazily** — only if at least one query needs it — so on clean data it costs nothing. Activation counts are logged.

### 4.4 Partitions are discovered, never hard-coded
`build_partitions()` enumerates `pool["country_key"].unique()`. Adding a country to the data adds a partition with no code change.

---

## 5. VALIDATION PLAN

Recall@{10,25,50,100,200} on the 110,341-entity internal validation set, broken down by country, **source (S2/S3)**, script (latin / deva / indic-other / mixed), and retrieval path. Plus an A/B on the one open design question:

- `combined_50` — pooled S2+S3, global top-50 (previous behaviour)
- `per_source_25` — S2 top-25 + S3 top-25 (same 50 budget)
- `per_source_50` — S2 top-50 + S3 top-50 (100 budget)

Reference figures to reproduce: India R@50 ≈ 90.8%, US R@50 ≈ 96.8%, latin ≈ 93.8%, deva ≈ 76.5%, indic-other ≈ 77.4%.

Plus five explicit routing tests (A known · B unseen/France · C missing · D variants · E unknown-in-pool), each asserting the record is not dropped, the right path is chosen, candidates are produced, activation is logged, and no Cartesian explosion occurs.

---

## 6. COMPUTATIONAL COMPLEXITY

Country × source partitioning bounds the pair space before any scoring. For the full test set:

| | pairs |
|---|---|
| Unpartitioned S1 × (S2+S3) | 1.73e6 × 9.97e6 = **1.73e13** |
| Country-partitioned | **6.72e12** (2.57× reduction) |
| × source split, then top-50 per source | ~1.7e8 |

No code path materialises S1 × S2 or S1 × S3. Scoring is a sparse `Q @ P.T` in query batches of 512; the only dense object is the per-batch score block.

---

## 7. FINAL APPROVED STRATEGY

1. Normalise country into `country_key` + `country_status`; keep raw.
2. Discover partitions from the pool; build one index per (country × source).
3. Route each query to its partition; route missing/unknown to a lazily built country-agnostic fallback for the same source.
4. Retrieve top-K per source, merge, dedup keeping the best score.
5. Score with the approved formula, IDF on the pool side only.
6. Measure recall by country, source, script and path; never declare done without it.
7. Never discard a record for a country reason; log every fallback activation.

Forbidden, unchanged from the playbook: char-n-gram retrieval scoring, edit distance as primary retrieval, phonetic blocking, PIN/ZIP as primary blocking key, destructive suffix removal, global ASCII folding, state/street expansion in retrieval.
