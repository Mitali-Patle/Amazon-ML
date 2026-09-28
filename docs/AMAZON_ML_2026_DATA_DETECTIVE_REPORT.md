# AMAZON ML 2026 — DATA DETECTIVE REPORT

**Status:** PLANNING MODE. No preprocessing implemented, no dataset modified, no model selected.
**Date:** 25 Sep 2026
**Scope:** Business Entity Resolution Challenge — dataset forensics, ER signal analysis, hypothesis testing, leakage audit, validation design, preprocessing strategy, candidate-generation strategy, scalability plan, implementation playbook.

Evidence classes are marked throughout:
**[D]** = measured on our actual dataset · **[R]** = external research/methodology · **[I]** = my inference from D+R.

---

## SECTION 1 — EXECUTIVE FINDINGS

1. **Country is a verified-lossless hard partition.** All 7,638,365 ground-truth pairs checked exhaustively: **100.000000% country agreement, 0 violations, 0 entities spanning countries.** Reduces the pair space 1.92× on train, 2.57× on test. **[D]**
2. **The relation is strictly many-to-one.** Unique matched IDs = total matched IDs = 7,638,365. No S2/S3 record is ever claimed by two S1 entities. A global mutual-exclusivity constraint is available and is directly worth precision under F_0.5. **[D]**
3. **Neither name nor address identifies an entity alone; the conjunction does.** 31.06% of S1 rows share an exact normalized *name* with another S1 row; 3.47% share an exact normalized *address* (up to 14 distinct businesses at one address). Yet **zero** S1 rows duplicate on (name + address + country). **[D]**
4. **Candidate generation is a ranking problem, not a coverage problem.** 99.99% of true pairs share ≥1 word token (name ∪ address). The recall ceiling is essentially free; the difficulty is getting true matches into the top-K. **[D]**
5. **The loss is concentrated in cross-script records.** Recall@50 by script: US Latin 96.81%, India Latin 93.90%, **Devanagari 71.82%, other Indic 72.75%**. This ~22-point gap is the single largest addressable recall loss. **[D]**
6. **A plausible-sounding normalization was tested and REJECTED.** Expanding state/street abbreviations and folding diacritics in the retrieval token stream *lowered* India recall@50 by 1.04pp and lowered the ceiling 99.50%→98.85%. See §5-H4. **[D]**
7. **Singletons are undetectable from the S1 record.** Singleton rate is 5.58% (US) vs 5.59% (India); name length, address length and token counts are identical to two decimal places between singleton and matched entities. Singleton detection must be a *confidence-threshold* decision, never a classifier on S1 features. **[D][I]**
8. **Test composition differs sharply from train.** Train S1 is 60% US / 40% India. Test S1 is **38.3% US / 46.8% India / 15.0% France**. The hardest partition is now the largest, and ~15% of the macro-average comes from a country with zero training data. **[D]**
9. **The training objective is misaligned with the metric.** Entities with ≤2 matches are 28.0% of the macro-average but only 11.4% of true pairs; singletons are 5.58% of the score and 0% of pairs. Pair-level optimization systematically under-weights what is scored. **[D][I]**
10. **The environment is the immediate blocker.** numpy, pandas, polars, pyarrow, scipy, sklearn, rapidfuzz are **all missing**. Pure-Python retrieval extrapolates to ~46 days for a full pass. **[D]**

---

## SECTION 2 — COMPETITION UNDERSTANDING

**Task.** Given business records from 3 independent sources with no shared identifiers, find for each Source-1 entity all matching records in Source 2 and/or Source 3.

**Unit of prediction.** One row per Source-1 entity: `source1_entity_id` → comma-separated `matched_entity_ids`. Never a pair; always an entity's full match *set*.

**Sources.** Source 1 is the deduplicated reference set (verified: 0 exact duplicates on name+address+country). Sources 2 and 3 are independent noisy observations. 80.48% of entities match in **both** S2 and S3; 6.48% only S2; 7.45% only S3; 5.58% neither. Per-source cardinality is bounded: max 5 matches in S2, max 6 in S3, max 11 total. **[D]**

**Ground truth.** 2,206,821 rows, exactly one per S1 entity, ID sets identical to `train_source1.tsv`. All 7,638,365 matched IDs resolve. Empty match list = singleton.

**Metric.** `F_0.5 = (1.25 × P × R) / (0.25 × P + R)`, computed **per Source-1 entity then macro-averaged over all entities**. Singletons included: correct empty prediction = 1.0, any prediction = 0.0. Precision weighted 2× recall.

- **False positives** are ~2× as costly as false negatives, and on a singleton a single FP takes that entity from 1.0 to 0.0.
- **False negatives** reduce recall only; an entity with 4 true matches where we find 3 correctly still scores 0.938.
- **[I]** Consequence: the optimal operating point is conservative. For a low-confidence candidate, predicting nothing usually beats guessing.

**Deliverables.** `matching_results.tsv` (scored) and `candidate_pairs.tsv` (the final pre-model candidate set, audited by organizers for recall ceiling and reduction ratio). Matches must be a subset of candidates. Plus runnable code, pinned requirements, and the filled methodology template.

**Constraints / disqualification.** Final model MIT/Apache-2.0, ≤8B parameters. **External lookup of business identities = immediate disqualification** — no registries, geocoding APIs, commercial ER services, or internet augmentation. Research on general methodology is permitted.

---

## SECTION 3 — DATASET FORENSICS

### Files and structure **[D]**

| File | Rows | Bytes |
|---|---|---|
| train_source1.tsv | 2,206,821 | 210 MB |
| train_source2.tsv | 5,034,616 | 489 MB |
| train_source3.tsv | 5,285,603 | 504 MB |
| train_ground_truth.tsv | 2,206,821 | 127 MB |
| test_source1.tsv | 1,732,544 | 175 MB |
| test_source2.tsv | 4,887,273 | 509 MB |
| test_source3.tsv | 5,082,316 | 506 MB |

Schema: `entity_id, business_name, business_address, country`; GT: `source1_entity_id, matched_entity_ids`. Tab-separated; `sep="\t"` mandatory (addresses and ID lists contain commas).

**Integrity — all verified clean:**
- Zero malformed rows (field count = 4 on every row of every source file).
- ID-prefix count = line count − 1 in all 7 files ⇒ **no embedded newlines**; `wc -l` is a true row count.
- GT ↔ Source-1 ID sets identical (0 only-in-either).
- All 7,638,365 matched IDs exist in train_source2/3; 0 dangling; 0 S1 self-matches.

### Completeness and encoding **[D]**

| | S1 | S2 | S3 |
|---|---|---|---|
| empty business_name | 0 | 0 | 0 |
| empty business_address | **0** | 168,967 (3.36%) | 175,916 (3.33%) |
| non-ASCII name | **0 (0.00%)** | 764,608 (15.19%) | 606,737 (11.48%) |
| non-ASCII address | 554 (0.03%) | 478,453 (9.50%) | 476,588 (9.02%) |
| mean name / address length | 24.0 / 52.1 | 25.1 / 46.2 | 25.2 / 46.7 |

**Source 1 is entirely ASCII and always has an address.** Sources 2/3 carry all the script noise and all the missingness. This asymmetry is structural, not incidental.

### Distributions **[D]**

- **Match counts:** 0→5.58%, 1→5.40%, 2→17.00%, 3→24.05%, 4→21.94%, 5→14.59%, 6→7.47%, 7→2.90%, 8+→1.06%. Mean 3.46.
- **Countries (train):** byte-exact `'US'` and `'India'` only, no variants/nulls. S1: 1,323,633 US / 883,188 India.
- **Countries (test):** `'India'` 809,986 / `'US'` 663,106 / `'France'` 259,452 (S1).
- **Vocabulary:** 1,457,821 distinct name tokens over S2+S3; **80.3% are hapax** (appear once). Top tokens are legal suffixes: `limited` 11.65%, `private` 11.32%, `llc` 10.66%, `ltd` 8.01%, `inc` 7.95%.
- **Address tokens:** `no` 17.71%, `road` 12.48%, `street` 6.73%, `new` 6.56%, `st` 6.44%.
- **Unmatched pool:** 2,681,854 S2/S3 records (26.0%) match nothing in train.

### Anomalies **[D]**
- 561 S1 / 2,946 S2 / 17,006 S3 names of ≤3 characters.
- Noise-token injection observed in names: `...Private Limited #65459`, `... | www.sarasrajf.com`, `Mr Red It Ltd`.
- Two LibreOffice lock files were present on `train_source1.tsv` and `train_ground_truth.tsv`. **These files have 2.2M rows; LibreOffice truncates at ~1.05M on save.** Operational hazard — raw files must never be opened in a spreadsheet.

---

## SECTION 4 — ENTITY-RESOLUTION SIGNAL ANALYSIS

Measured over 414,666 sampled true pairs unless noted. **[D]**

| Signal | True-pair rate |
|---|---|
| country agrees | **100.00%** (exhaustive: 7,638,365/7,638,365) |
| address shares ≥1 token | **95.58%** |
| name shares ≥1 token | 85.57% |
| name OR address shares ≥1 token | **99.99%** |
| address shares ≥1 number | 79.95% |
| name exact, suffix-stripped | 41.44% |
| name exact, normalized | 21.93% |
| address exact, normalized | 8.25% |
| name exact, raw | 4.67% |
| both sides have PIN/ZIP | **5.07%** — but agrees 93.30% when present |
| mean name Jaccard / address Jaccard | 0.616 / 0.596 |

**Address is the stronger recall channel than name** (95.58% vs 85.57%) — the opposite of the intuitive ordering, and the reason address must be a first-class blocking field.

### The noise model, from inspecting matched pairs across similarity bands **[D]**

| Pattern | Observed example | Implication |
|---|---|---|
| Word-order transposition | `Atlantic Research Ventures-Revere` → `Atlantic Ventures-Revere Research`; `Akriti Mindset Private Limited` → `Akriti Limited Private Mindset` | **Token-set similarity is mandatory.** Sequence edit distance as a primary name signal would fail outright. |
| State abbrev ↔ full name | MO↔Missouri, MH↔Maharashtra, GJ↔Gujarat, TG↔Telangana | Tempting to normalize — **tested and rejected**, see §5-H4. |
| Diacritic injection | `Bránds Ínc`↔`Brands Inc`, `Vídyalaya`↔`Vidyalaya` | Real noise, but folding it did not help retrieval (§5-H4). |
| Multiple Indic scripts | Devanagari `अल इन्वेस्टमेंट एलएलपी`, Bengali `পশ্চিমবঙ্গ`, Kannada `ಕರ್ನಾಟಕ`, Gujarati `સિટી ટ્રેડિંગ` | Not a Devanagari-only problem. Transliteration must cover multiple scripts. |
| Typos incl. digits | `RVERE`/`REVERE`, `Sanhg`/`Sangh`, `968`/`368 Midway Road` | Char n-grams as a secondary channel; house numbers are corruptible. |
| Noise-token injection | `#65459`, `| www.sarasrajf.com`, `Mr ` prefix | Asymmetric containment scoring > symmetric Jaccard. |
| Ordinals / city aliases | `11th Ave`↔`Eleventh Ave`; `Madras`↔`Chennai` | Ordinals safe in principle; city aliases risky — leave to the matcher. |
| Component dropping | Address truncated to fewer components | Never require field completeness. |

### Cross-script behaviour **[D]**
S1 names are 100% Latin; 15.19% of S2 and 11.48% of S3 names are non-ASCII. Script-pair distribution over true pairs: Latin→Latin 92.77%, Latin→Devanagari 3.83%, Latin→other-Indic 3.09%, Latin→mixed 0.31%.

Crucially, **cross-script pairs still reach 99.99% word overlap — via the address**, which usually stays Latin even when the name does not (e.g. `Al Investment LLP` ↔ `अल इन्वेस्टमेंट एलएलपी` with address Jaccard 0.82). So cross-script records are *retrievable* but *rank poorly*: recall@50 of 71.82% (Devanagari) and 72.75% (other Indic) vs 93.90% (India Latin).

**[I]** This localises the single biggest recall opportunity precisely: it is a **name-channel scoring problem for ~7% of pairs**, not a coverage problem, and not fixable by raising K (K=200 only reaches 92.12% overall).

---

## SECTION 5 — HYPOTHESES TESTED

### H1 — "Country is a safe hard blocking key." → **SUPPORTED (exhaustively)**
*Why it matters:* determines whether the pair space can be partitioned for free.
*Method:* loaded country for all 12,527,040 train records; checked every one of the 7,638,365 GT pairs.
*Evidence:* 7,638,365 agree / **0 disagree** / 0 unresolvable / 0 entities spanning countries. Cross-tab: US→US 4,578,522; India→India 3,059,843. Country values byte-exact with no variants.
*Implication:* partition all retrieval by country. Reduction 1.92× (train) / 2.57× (test).
*Caveat:* verified on **train only** — test has no labels, so test-side losslessness is assumed. France is generated by an unobserved process. Audit at inference: if France entities show anomalously low candidate counts, revisit.

### H2 — "Address overlap is a stronger signal than name overlap." → **SUPPORTED**
*Evidence:* 95.58% of true pairs share an address token vs 85.57% a name token; address is also the channel that rescues cross-script pairs.
*Implication:* address is a primary blocking field, not a verification-only field.

### H3 — "Legal-suffix removal improves recall but creates dangerous collisions." → **SUPPORTED, and the collision side dominates**
*Method:* measured exact-match uplift on true pairs, then measured S1-vs-S1 collisions under the same transform.
*Evidence:* exact-name match rises 21.93% → **41.44%** on true pairs. But **41.93% of all S1 rows** (925,303 rows in 204,422 groups) collide with another S1 row on (suffix-stripped name, country): `primary care` ×786, `urgent care` ×784, `behavioral health` ×758, `dental` ×727.
*Implication:* **never use suffix-stripped name as a standalone blocking key or identity key.** Extract the suffix to its own field; suffix *agreement* is a useful feature, suffix *deletion* is a merge hazard.

### H4 — "Expanding state/street abbreviations and folding diacritics improves candidate recall." → **REJECTED**
*Why it matters:* this was my own prior recommendation; it looked obviously correct from the noise examples.
*Method:* built two complete inverted indexes per country over the full same-country pool (India 4,133,346; US 6,186,873) — baseline normalizer vs improved normalizer (country-conditioned US/India state maps, street-abbreviation map, ordinal normalization, Latin-only diacritic folding). Same 1,200 queries, same IDF scorer, same K grid.

| Partition | R@10 | R@25 | R@50 | R@100 | R@200 | ceiling |
|---|---|---|---|---|---|---|
| India BASE | 85.80 | 88.87 | **90.15** | 91.89 | 93.04 | 99.50 |
| India IMPR | 84.92 | 87.75 | **89.11** | 90.65 | 92.12 | **98.85** |
| US BASE | 93.61 | 95.54 | **96.81** | 97.60 | 98.23 | 100.00 |
| US IMPR | 94.12 | 96.04 | **96.82** | 97.41 | 98.08 | 100.00 |

*Evidence:* India **−1.04pp at K=50, negative at every K** (consistent direction across 5 thresholds; SE≈0.45pp). US **+0.01pp at K=50** — noise. Worse, the India *ceiling* fell 99.50% → 98.85%, meaning the transform actively destroyed reachable matches.
*Mechanism* **[I]**: expanding `MH`→`maharashtra` converts a rarer token into a corpus-frequent one and emits multi-token expansions (`tamil nadu`), diluting the IDF ranking by boosting millions of same-state candidates equally. It buys token agreement on a field that carries almost no identifying information.
*Implication:* **do not apply state/street expansion to the retrieval token stream.** It may still be valid as a *pair-verification feature* during matching — a different role, and untested. This is the clearest case in the whole investigation of a transformation that looks like cleaning and is actually signal destruction.

### H5 — "Candidate recall saturates after K." → **SUPPORTED, saturates early and low**
*Evidence (India, baseline):* R@10 85.80 → R@50 90.15 → R@200 93.04, against a 99.50% ceiling. Each doubling past K=50 buys ~1pp while doubling downstream cost.
*Implication:* K≈50 is the efficient operating point; the residual gap is not a K problem.

### H6 — "Token-union blocking is feasible at this scale." → **REJECTED**
*Evidence:* blocking on any shared token with df<50,000 gives 99.62% recall but **78 billion candidate pairs**. At df<1,000 it falls to 0.6B pairs but 10.93% of entities get zero keys.
*Implication:* threshold-union blocking is unusable. Must be top-K retrieval. **[R]** matches the literature: kNN-join blocking dominates ε-join/threshold blocking, and token blocking is known to produce heavy redundancy.

### H7 — "Naive punctuation stripping is safe." → **REJECTED (destructive)**
*Method:* compared `re.sub(r"[^\w\s]"," ",s)` against Unicode-category-aware stripping.
*Evidence:* `राम मार्केटिंग प्राइवेट लिमिटेड` → `['र','म','म','र','क','ट','ग','प','र','इव','ट','ल','म','ट','ड']`. Python's `\w` excludes Indic combining marks (category `Mn`), shattering every Devanagari word into bare consonants. Under the naive tokenizer the corpus "top tokens" include `ल`, `ट`, `र`, `ड` as if they were words.
*Impact:* silently corrupts ~13% of S2/S3 names — the exact subpopulation already performing worst.
*Implication:* **mandatory:** strip only Unicode categories `P*`/`S*`/`Cc`/`Cf`. Never `[^\w\s]`. Never apply `unidecode`/ASCII-fold to non-Latin tokens.

### H8 — "Singletons can be identified from the Source-1 record." → **REJECTED**
*Evidence:*

| group | n | name_len | addr_len | name_tok | addr_tok |
|---|---|---|---|---|---|
| SINGLETON / India | 49,351 | 26.3 | 77.7 | 3.70 | 12.39 |
| matched / India | 833,837 | 26.4 | 77.7 | 3.70 | 12.39 |
| SINGLETON / US | 73,896 | 22.4 | 34.9 | 3.49 | 5.90 |
| matched / US | 1,249,737 | 22.5 | 35.0 | 3.49 | 5.91 |

Indistinguishable. Singleton rate 5.58% (US) vs 5.59% (India) — no country signal either.
*Implication:* singleton-ness is a property of *absence in S2/S3*, not of the S1 record. Detect it by confident non-match at threshold, never by a classifier on S1 features. Worth 5.58% of the macro score.

### H9 — "A random entity-level 95/5 split leaks." → **REJECTED (it is safe), for a structural reason**
*Evidence:* the many-to-one property (unique matched IDs = total = 7,638,365) means the bipartite match graph is a **forest of stars centred on S1 entities**. No connected component can span two S1 entities, so no transitive leakage is possible. Additionally 0 exact S1 duplicates on (name, address, country).
*Implication:* grouping by match-component is equivalent to grouping by S1 entity — no extra machinery needed. Stratify by country (composition differs by source and shifts at test).

### H10 — "Validation can be inflated by pool construction." → **SUPPORTED (this is the real leakage risk)**
*Evidence:* train ratio S1:(S2+S3) = 1:4.68; test = **1:5.75**, consistent per partition (US 1:5.76, India 1:5.82, France 1:5.53). A naive 5% validation slice searched against only the 5% of S2/S3 would face ~20× fewer distractors than reality.
*Implication:* see §8. This, not the split itself, is where validation optimism would come from.

---

## SECTION 6 — DATA QUALITY ASSESSMENT (ranked by impact)

| # | Issue | Evidence | Impact | Action |
|---|---|---|---|---|
| 1 | Cross-script names rank poorly | R@50 71.8% (Deva) / 72.8% (other Indic) vs 93.9% Latin | **Largest addressable recall loss** | Transliteration as an additional name channel (§11) |
| 2 | Naive punctuation stripping destroys Indic text | `[^\w\s]` shatters Devanagari into consonants; ~13% of S2/S3 names | Silent, catastrophic, affects the weakest subgroup | Category-aware stripping only — non-negotiable |
| 3 | Suffix-stripped names collide massively | 41.93% of S1 rows collide; `primary care` ×786 | False merges; 2× penalty under F_0.5 | Extract suffix, never delete; never block on name alone |
| 4 | Legal suffixes dominate the vocabulary | `limited` 11.65%, `private` 11.32%, `llc` 10.66% | Retrieval cost + ranking dilution | IDF handles weighting; drop from the index for cost |
| 5 | 3.3–3.4% of S2/S3 have empty addresses | 4.41% of *true-matched* records have none | Dropping them loses real matches | Never drop; name-only path must exist |
| 6 | Same-address distinct businesses | 3.47% of S1 rows; up to 14 per address | Hard negatives; address alone cannot identify | Require name+address conjunction for confident match |
| 7 | Train/test country shift | 60/40 US/India → 38/47/15 US/India/France | Hardest partition is now largest; 15% unlabelled | Country-agnostic defaults; weight India in tuning |
| 8 | Test is distractor-richer | 1:4.68 → 1:5.75 | Threshold optimistic on precision | Construct validation pool at test ratio |
| 9 | Short/degenerate names | 17,006 S3 names ≤3 chars | Unreliable blocking keys | Flag, force address-anchored retrieval |
| 10 | Spreadsheet truncation hazard | Lock files seen on 2.2M-row TSVs; LibreOffice caps ~1.05M | Silent destruction of source data | Never open raw files in a spreadsheet |

---

## SECTION 7 — LEAKAGE AUDIT

| Path | Mechanism | Present? | Prevention |
|---|---|---|---|
| Transitive components spanning the split | A~B, B~C ⇒ component straddles split | **NO** — many-to-one makes the graph a star forest (H9) | Entity-level split suffices |
| Duplicate/near-duplicate S1 across split | Twin entity memorised from the other side | **NO** exact dupes (0 on name+addr+country) | — |
| S2/S3 record shared by two S1 entities | Record in both train and val pools with a label | **NO** — mutual exclusivity verified | — |
| **Validation pool under-population** | Val entities face far fewer distractors than test | **YES — primary risk** | Build val pool at 1:5.75 with padded distractors (§8) |
| Threshold tuned on an easy pool | Decision threshold optimistic on precision | **YES if §8 ignored** | Tune only on the test-ratio pool; add conservative offset |
| IDF/df fitted across the split | Corpus statistics see validation text | **Label-free** — see ruling below | Fit per searched pool, consistently at train and test |
| Label-derived statistics | Match-rate priors, singleton rate, per-token match likelihood, learned blocking weights, calibration curves | Would leak | **Train-95% only**, always |
| Target-derived features | "This record is already claimed" — encodes the answer | Would leak badly | Forbid any feature built from GT match sets at inference time |
| Official test set influencing development | Any decision informed by test | Must not occur | Test used only for final inference (§ scope note) |

**Ruling on corpus statistics (IDF/df).** These are **label-free** and computable at inference from the test corpus itself. Fitting them on the pool being searched is not leakage — it *mirrors* inference, and restricting them to the 95% would actually mis-simulate test conditions. Anything label-derived is train-only without exception. **[I]**

**Scope note on test reads.** During this investigation the test files were read for: row counts, column headers, `country` value counts, and the first 3 data rows of each file (initial integrity verification). No name/address distributions, token statistics, or derived features were computed from test, and no threshold, feature, or preprocessing decision was informed by test content. The country composition was used only to size partitions and to flag the France risk — which the problem statement discloses independently.

---

## SECTION 8 — VALIDATION STRATEGY

**Split unit:** the Source-1 entity. **Grouping:** by match-component — which, by H9, is exactly the S1 entity, so no extra machinery is required. **Stratification:** by country. **Ratio:** **95 / 5** as directed; the structure gives no evidence-based reason to deviate. 5% ≈ 110,341 validation entities — ample (±0.3pp on a 90% recall estimate).

**What goes where**

- Internal training (95%): ~2,096,480 S1 entities + the S2/S3 records they own + a proportional share of unmatched distractors.
- Internal validation (5%): ~110,341 S1 entities + the S2/S3 records they own + **padded distractors to reach 1:5.75**.

**Fitting rules**

| Quantity | May be fit on | Reason |
|---|---|---|
| token df / IDF | the pool being searched (train pool for train, val pool for val, test pool at inference) | label-free, available at inference, mirrors test |
| tokenizer, normalizer, script detection | deterministic — no fitting | per-record, order-independent |
| match-rate priors, singleton rate | **95% only** | label-derived |
| decision threshold, calibration | **95% only**, evaluated on the 5% once | the whole point of holdout |
| blocking weights (if learned) | **95% only** | supervised |

**Inference simulation — the part that matters most.** Validation must run the *full* pipeline: query the 5% entities against a realistically-sized same-country pool, generate candidates top-K, score, threshold, and emit the same submission format. Then compute **macro F_0.5 per entity including singletons**, not pairwise F_0.5. Given that entities with ≤2 matches are 28.0% of the macro-average but only 11.4% of pairs, and singletons are 5.58% of the score and 0% of pairs, a pairwise validation number would be measuring the wrong thing. **[D][I]**

**Known, unfixable blind spot:** no French data exists in training, so ~15% of the test macro-average cannot be validated by any construction. Report validation scores with this stated explicitly, and prefer country-agnostic mechanisms wherever a choice exists.

---

## SECTION 9 — PREPROCESSING STRATEGY

Ordering rationale in §15. Each step: objective · transformation · rationale · risk · leakage · effect.

**P1 — Canonical ingestion.** TSV → columnar (parquet), chunked, dtypes fixed, `country` categorical. *Rationale:* 2.4 GB across 7 files; 15 GB RAM. *Risk:* none. *Leakage:* none. *Effect:* enables every later stage; ~5–10× faster repeat reads.

**P2 — Structural assertions.** Re-assert row counts, ID-prefix integrity, GT↔S1 identity, many-to-one, country partition. *Rationale:* H1 and H9 are load-bearing; a silent violation invalidates the split and the blocking. *Effect:* fail fast.

**P3 — Unicode NFKC + whitespace collapse.** *Risk:* none measured. *Effect:* prerequisite for stable tokenization.

**P4 — Category-aware punctuation stripping.** Replace only `P*`/`S*`/`Cc`/`Cf` with space. **Never `[^\w\s]`.** *Rationale:* H7 — the naive form destroys ~13% of S2/S3 names. *Effect:* the single highest-value correctness fix in the pipeline.

**P5 — Casefold into `norm`; retain `raw` verbatim.** *Rationale:* raw exact-name match is 4.67% of true pairs and a strong precision signal that normalization erases. *Risk:* storage only. *Effect:* enables exact-match features alongside fuzzy ones.

**P6 — Script detection per field.** Label `latin | deva | indic-other | mixed`. *Rationale:* gates every script-sensitive operation; identifies the 71.8%/72.8% recall subgroup. *Effect:* prevents applying Latin transforms to Indic text.

**P7 — Legal-suffix extraction (non-destructive).** Extract to `legal_suffix[]`; keep the name intact; emit `name_core` as an additional field. *Rationale:* H3 — uplift is real (21.9%→41.4%) but 41.93% S1 self-collision makes deletion unsafe. *Risk:* using `name_core` as a standalone block key would be a major false-merge source — forbid it. *Effect:* suffix agreement becomes a feature; identity information preserved.

**P8 — Token and char-n-gram derivation.** Order-insensitive token sets (H: word-order transposition is pervasive) plus char-3-grams for typos. *Effect:* feeds retrieval channels.

**P9 — Address component parsing (additive).** Extract `addr_pin`, `addr_numbers[]`, trailing state token — as *separate fields*, leaving `addr_norm` untouched. *Rationale:* PIN covers only 5.07% of pairs so it is useless for blocking, but agrees 93.30% when present ⇒ strong verification feature. *Risk:* parsing must degrade gracefully on France's unseen grammar.

**P10 — Transliteration of non-Latin names into an ADDITIONAL field.** Never replacing the original. *Rationale:* the 22-point cross-script recall gap (§4). *Risk:* must be verified to help before adoption — H4 is the cautionary precedent. **Gate: adopt only if it raises Devanagari/other-Indic R@50 in an A/B on the same harness.**

**DO NOT DO — with the evidence that rules each out:**

| Rejected | Evidence |
|---|---|
| State/street expansion in the retrieval stream | H4: India −1.04pp @K=50, ceiling 99.50→98.85 |
| `[^\w\s]` punctuation stripping | H7: destroys Devanagari |
| Global `unidecode` / ASCII folding | Same mechanism in reverse |
| Destructive suffix removal | H3: 41.93% S1 collision |
| Dropping empty-address rows | 4.41% of true-matched records |
| Deduplicating Source 1 | 0 exact duplicates exist |
| One-hot encoding `country` | France unseen; spec forbids it |
| Threshold-union blocking | H6: 78B pairs |
| PIN/ZIP as a primary blocking key | 5.07% coverage |
| Sequence edit distance as the primary name signal | Word-order transposition is pervasive |
| Soundex/phonetic encoding for Indic names | **[R]** documented as ill-suited for Indic morphology/cross-script variation |
| A singleton classifier on S1 features | H8: singletons are statistically identical |
| Discarding raw fields after normalization | Raw exact match = 4.67% of pairs |

---

## SECTION 10 — REPRESENTATION STRATEGY

Per record, keyed by `entity_id`. **Principle: raw + normalized + derived; never overwrite.**

```
raw        business_name, business_address, country        (verbatim)
norm       name_norm, addr_norm                            (NFKC → category-safe → casefold)
script     name_script, addr_script                        (latin|deva|indic-other|mixed)
derived    name_tokens[], addr_tokens[]                    (order-insensitive sets)
           name_char3grams[]                               (typo channel)
           legal_suffix[], name_core                       (extracted; name_core = feature only)
           addr_pin, addr_numbers[], addr_state_tok        (parsed, additive)
           name_translit                                   (conditional on P10 gate)
```

Justification per layer is in §9. Note what is deliberately **absent**: no state-expanded address stream (H4), no suffix-deleted canonical name as an identity key (H3), no phonetic encoding (**[R]**).

---

## SECTION 11 — CANDIDATE GENERATION STRATEGY

**Formulation: per-country top-K retrieval, multi-channel union, IDF-ranked.** Not threshold-union (H6), not single-channel.

**Measured baseline** (full same-country pools, untuned IDF scorer, name weight 1.0 / address weight 0.7):

| Partition | R@10 | R@25 | R@50 | R@100 | R@200 | ceiling |
|---|---|---|---|---|---|---|
| **US** | 93.61 | 95.54 | **96.81** | 97.60 | 98.23 | 100.00 |
| **India** | 85.80 | 88.87 | **90.15** | 91.89 | 93.04 | 99.50 |
| India — Latin | | | 93.90 | | | |
| India — Devanagari | | | **71.82** | | | |
| India — other Indic | | | **72.75** | | | |

**Channels to prepare** (preprocessing must preserve the inputs for each):
- **A — IDF token retrieval** over name ∪ address. Drop corpus stopwords (`limited`/`private`/`llc`, each >10% df) for cost; IDF already discounts them.
- **B — Address-anchored retrieval.** 95.58% single-channel coverage; the channel that reaches cross-script records.
- **C — Char-3-gram retrieval** for typos (`RVERE`/`REVERE`).
- **D — Transliterated-name retrieval**, gated on P10.

Union, rank by combined score, keep top-K. **[R]** Weighting candidates by how many channels/blocks they co-occur in is *meta-blocking*, and is the established fix for token-blocking redundancy.

**K recommendation:** start at **K=50** (India R@50 90.15%, US 96.81%; ~46 negatives per positive; 87M test pairs). **Do not lock K** until channel D is tested — H5 shows K is the wrong lever for the residual gap, so the decision should follow the transliteration A/B, not precede it.

**Reduction ratio:** full test cross product 1.73e13 → country-partitioned 6.72e12 → top-50 8.7e7. Overall ~2×10⁵ reduction.

**Instrumentation is mandatory from day one:** recall@K and reduction ratio, reported **per country and per script**, on validation. The organizers audit `candidate_pairs.tsv` on exactly these. A true match not in the candidate set is unrecoverable by any downstream model.

---

## SECTION 12 — SCALABILITY STRATEGY

**Measured environment [D]:** Python 3.14; **numpy, pandas, polars, pyarrow, scipy, sklearn, unidecode, rapidfuzz, Levenshtein ALL MISSING**; 20 cores; **15 GB RAM (~8 GB free)**; RTX 4050 **6 GB VRAM**; 398 GB free disk.

**Measured costs [D]:** full-corpus tokenization + df ≈ 2m15s/pass; inverted index over 4.1M records ≈ 32M postings; 1,200 queries ≈ 2min ⇒ pure Python extrapolates to **~46 days** for 2.2M queries. **Sparse linear algebra is not optional.**

- **CHUNKING** — *ingestion/transformation*. Stream TSV→parquet in ~500k-row chunks; peak RAM <1 GB. Justified by 210–509 MB files against 8 GB free.
- **SHARDING** — *workload partitioning*. Natural axis **(country × source)**: 4 shards in train, 6 at test (France). Country is verified-lossless (H1), so sharding along it costs **zero recall**. 20 cores ⇒ shards in parallel; each index fits comfortably.
- **BATCHING** — *model/compute grouping*. Only for the matcher's pass over ~87M test pairs. **[I]** Note for the modelling agents: 6 GB VRAM makes a cross-encoder over 87M pairs implausible; this constrains architecture, though the choice is not mine.
- **Storage:** parquet + categorical dtypes; token IDs as `int32` arrays, never Python strings; postings as `array('i')`/CSR. Cache every stage keyed by `entity_id` so no stage is recomputed.
- **Cloud:** not required for preprocessing — it fits locally. Revisit only if a GPU-bound matcher needs more than 6 GB VRAM. **[I]**

---

## SECTION 13 — ALTERNATIVE STRATEGIES CONSIDERED

| Decision | Alternatives | Chosen | Why |
|---|---|---|---|
| Normalization posture | (a) aggressive canonicalization (b) conservative (c) multi-representation | **(c)** | H3 and H4 both show aggressive transforms destroy signal; (b) forgoes free wins like H7's fix |
| Blocking | (a) deterministic keys (b) threshold-union (c) multi-channel top-K | **(c)** | H6 kills (b) at 78B pairs; (a) cannot reach 99.99% coverage; **[R]** kNN-join > ε-join |
| Split | (a) random row (b) entity-level (c) component-level (d) time/source-based | **(b) ≡ (c)** | H9 — star-forest structure makes them identical; no ordering signal exists for (d) |
| Val pool | (a) proportional 5% slice (b) full corpus (c) test-ratio padded | **(c)** | (a) ~20× too easy; (b) over-harsh vs test's 1:5.75 |
| Corpus stats | (a) 95%-only (b) per-searched-pool (c) full dataset | **(b)** | Label-free and inference-mirroring; (a) mis-simulates test |
| Cross-script | (a) ignore (b) char n-grams (c) transliteration channel (d) multilingual embeddings | **(c), gated** | (a) forfeits ~7% of pairs at 72% recall; (b) cannot bridge scripts at all; (d) is a modelling decision, not mine |
| K | fixed now vs after channel D | **defer** | H5: K is the wrong lever for the residual gap |

---

## SECTION 14 — RESEARCH EVIDENCE

Consulted for **general methodology only**. No external business lookup, registry query, geocoding, or record enrichment was performed or is proposed — that is a disqualification-level rule. *(Note: a search surfaced another team's public competition repository; it was not opened or used.)*

- **Blocking and Filtering Techniques for Entity Resolution: A Survey** (Papadakis et al., ACM CSUR 53(2), 2020) — token blocking produces high redundancy; block-collection quality is measured by recall ceiling and reduction ratio. Supports §11's instrumentation requirement. [link](https://helios2.mi.parisdescartes.fr/~themisp/publications/csur20-blockingfiltering.pdf)
- **Benchmarking Filtering Techniques for Entity Resolution** (Papadakis et al., 2022) — kNN-join blocking outperforms ε-join/threshold blocking across datasets. Directly supports choosing top-K over threshold-union (H6). [link](https://arxiv.org/pdf/2202.12521)
- **Generalized Supervised Meta-blocking** (2022) — weighting candidate pairs by co-occurrence across blocks prunes redundancy without losing recall. Supports the multi-channel union scoring in §11. [link](https://arxiv.org/pdf/2204.08801)
- **Towards Universal Dense Blocking for Entity Resolution** (2024) — design-space trade-offs between semantic precision, compute, and memory for embedding-based candidate generation. Context for channel D and the 6 GB VRAM constraint. [link](https://arxiv.org/pdf/2404.14831)
- **How to Evaluate Entity Resolution Systems** (2024) — entity-centric vs pairwise evaluation diverge; threshold performance should be assessed across a range. Supports §8's insistence on macro-per-entity validation and conservative threshold calibration. [link](https://arxiv.org/pdf/2404.05622)
- **Cross-script / Indic transliteration literature** — phonetic algorithms such as Soundex are documented as ill-suited to Indic morphological and cross-script variation; multilingual transliteration generalizes better than bilingual. Supports rejecting phonetic encoding (§9) and gating channel D. AI4Bharat's IndicXlit covers 21 Indic languages — **license and parameter count must be checked against the MIT/Apache-2.0 + ≤8B rule before any use.** [link](https://arxiv.org/html/2605.23597v1) · [link](https://github.com/AI4Bharat/indicnlp_catalog)

---

## SECTION 15 — FINAL PREPROCESSING PLAYBOOK

**Derived ordering.** The canonical "ingest → audit → split → fit → clean → represent" sequence is *nearly* right but wrong in one specific way: **representation construction must come before the split, not after.** Every transform in P3–P9 is deterministic and per-record — none fits a parameter — so running it pre-split is leakage-free, avoids duplicated work, and guarantees train and validation are transformed by identical code. Only *fitted* quantities must wait for the split. The playbook below reflects that.

| Phase | Objective | Input | Output | Depends on | Leakage | Compute | Expected effect |
|---|---|---|---|---|---|---|---|
| **0. Environment** | Install + pin numpy, pandas, pyarrow, scipy, sklearn, rapidfuzz; fix seeds; version configs | — | `requirements.txt`, env | — | none | minutes | **Hard blocker** — nothing else runs |
| **1. Ingestion** | TSV → parquet, chunked | 7 raw TSVs | `data/interim/*.parquet` | 0 | none | ~10 min, <1 GB RAM | 5–10× faster reruns |
| **2. Structural audit** | Assert row counts, ID integrity, GT↔S1, many-to-one, country partition | Phase 1 | assertion report | 1 | none | ~1 min | Fails fast if H1/H9 break |
| **3. Representation build** | P3–P9: NFKC, category-safe strip, casefold, script tag, suffix extraction, tokens, n-grams, address parse. **Raw preserved.** | Phase 1 | enriched record tables | 2 | **none — nothing is fitted** | ~15 min, shardable | The H7 fix alone repairs ~13% of S2/S3 names |
| **4. Leakage-safe split** | 95/5 on S1 entities, stratified by country, grouped by match-component (≡ entity, per H9) | Phase 3 | `split_assignment.parquet` | 3 | **the gate for everything after** | seconds | Trustworthy holdout |
| **5. Validation pool construction** | Pad val pool to test ratio **1:5.75**; mirror inference conditions | Phases 3, 4 | val pool manifest | 4 | prevents the §7 primary risk | minutes | Stops the largest source of validation optimism |
| **6. Fitted statistics** | df/IDF **per searched pool**; label-derived priors **95%-only** | Phases 3, 4, 5 | per-shard stat tables | 4, 5 | label-free vs label-derived split enforced here | ~5 min/shard | Correct, inference-mirroring weighting |
| **7. Candidate-generation prep** | Sparse indexes per (country × source) shard; channels A–C (D gated) | Phases 3, 6 | CSR indexes | 6 | none | parallel over 20 cores | Enables top-K retrieval |
| **8. Channel-D A/B (gate)** | Test transliteration on the §11 harness, per script | Phase 7 | decision + numbers | 7 | val-only measurement | ~30 min | **Decides P10 and final K** |
| **9. Candidate generation + instrumentation** | Top-K retrieval; report R@K and reduction ratio **per country and per script** | Phases 7, 8 | `candidate_pairs` (val) | 8 | none | GPU/CPU sparse matmul | The recall ceiling everything downstream inherits |
| **10. Pair-feature preparation** | Features only over surviving candidates | Phase 9 | pair feature tables | 9 | features must be computable at test; no GT-derived fields | bounded by K | Model-ready |
| **11. Model-ready artifacts + reproducibility** | Emit train/val matrices, `candidate_pairs.tsv` schema, manifest, seeds, config hashes, one-command regeneration | Phase 10 | `data/processed/` | 10 | — | — | Handoff to modelling; required for the submission zip |

**Gates that must not be skipped:** Phase 2 (H1/H9 must hold), Phase 4 (nothing fitted before it), Phase 5 (or validation is meaningless), Phase 8 (K is not fixed until this returns).

---

## OPEN QUESTIONS

1. **Does transliteration actually help?** Untested — no library installed. Gate at Phase 8. This is the highest-value remaining unknown (~22pp on ~7% of pairs).
2. **France's address grammar is unobservable.** ~15% of the macro-average cannot be validated. Country-agnostic fallbacks everywhere; audit candidate counts for France at inference.
3. **Is test-side country losslessness real?** Verified 7,638,365/7,638,365 on train; unverifiable on test.
4. **Optimal K** — deferred to Phase 8 by design.
5. **Is mutual exclusivity exploitable as a global assignment?** Verified in train; an assignment/deconfliction post-process could convert it into precision, but that is a modelling decision for `ml-architect`.

---

*End of report. Implementation requires explicit authorization.*
