# PREPROCESSING PLAYBOOK — Amazon ML Challenge 2026 (Business Entity Resolution)

**Status:** APPROVED FOR IMPLEMENTATION pending human authorization.
**Evidence base:** `docs/AMAZON_ML_2026_DATA_DETECTIVE_REPORT.md`. This document is the implementation contract; the report is the justification.
**Date:** 25 Sep 2026

Every parameter below is either **LOCKED** (measured, do not change without re-running the stated experiment) or **TUNE** (start here, optimise during implementation).

---

## PART 1 — DECISION REGISTER

### Locked decisions

| # | Decision | Value | Evidence |
|---|---|---|---|
| D1 | Partition all retrieval by `country` | hard partition | 7,638,365/7,638,365 pairs agree; 0 violations |
| D2 | Punctuation stripping | Unicode categories `P*`,`S*`,`Cc`,`Cf` only | `[^\w\s]` shatters Devanagari into consonants (~13% of S2/S3 names) |
| D3 | Field retention | raw + norm + derived, never overwrite | raw-exact name match = 4.67% of true pairs |
| D4 | Legal suffixes | extract to own field; **never delete** | deletion → 41.93% of S1 rows collide (`primary care` ×786) |
| D5 | State/street abbreviation expansion in retrieval stream | **FORBIDDEN** | India R@50 −1.04pp, ceiling 99.50→98.85 |
| D6 | Transliteration | Indic→Latin tokens **folded into the name-token space** | deva R@50 73.0→78.2, indic-other 78.7→81.4, Latin unchanged |
| D7 | Char-n-grams summed as an extra scoring channel | **FORBIDDEN** | R@50 collapsed 90.11→72.92→46.39→27.63 as weight rose |
| D8 | Blocking formulation | per-country top-K retrieval | threshold-union = 78B pairs |
| D9 | Address weight in retrieval score | `w_addr = 0.7` | sweep: 0.3→83.92, 0.5→89.54, **0.7→90.80**, 1.0→89.45 |
| D10 | K | **50** | India 90.80%, US ~96.8%; +1pp per doubling beyond |
| D11 | Split | 95/5 on S1 entities, stratified by country | match graph is a star forest ⇒ entity-level ≡ component-level |
| D12 | Validation pool ratio | **1 : 5.75** (S1 : S2+S3) | train is 1:4.68, test is 1:5.75 |
| D13 | IDF/df fitting | per searched pool | label-free, available at inference, mirrors test |
| D14 | Label-derived statistics | **95% internal-train only** | thresholds, priors, calibration, learned weights |
| D15 | Validation metric | macro F_0.5 **per entity**, singletons included | ≤2-match entities = 28.0% of score but 11.4% of pairs |
| D16 | Singleton handling | confidence threshold, **never a classifier on S1 features** | singleton vs matched statistically identical |
| D17 | Shard axis | (country × source) | country partition is lossless ⇒ zero recall cost |
| D18 | Representation build position | **before** the split | all transforms are per-record and fit nothing |

### Forbidden list — do not implement any of these

`[^\w\s]` punctuation stripping · global `unidecode`/ASCII-folding · state/street expansion in retrieval tokens · destructive suffix removal · char-n-gram score summation · dropping empty-address rows (4.41% of true-matched records) · deduplicating Source 1 (0 duplicates exist) · one-hot `country` (France unseen) · PIN/ZIP as a primary blocking key (5.07% coverage) · sequence edit distance as the primary name signal (word-order transposition is pervasive) · Soundex/phonetic encoding for Indic names · a singleton classifier on S1 features · discarding raw fields after normalization · any external business lookup, registry, geocoder, or ER API (**disqualification**).

### Open for tuning during implementation

| Item | Start at | How to improve |
|---|---|---|
| `w_addr` script-conditional | 0.7 global | Dual-pass: 0.7 general + 1.0 restricted to non-Latin records, union. At `w_addr=1.0` deva hits 80.0% but Latin drops to 91.3% — a real Pareto trade a dual-pass should capture |
| K | 50 | Re-measure R@K after D6 lands; raise only if the matcher is cheap |
| Stopword df cutoff | 60,000 | Cost/recall trade only |
| Transliteration table | rule-based (provided below) | AI4Bharat IndicXlit if licence ≤ Apache-2.0 **and** ≤8B params — verify before use |

---

## PART 2 — PHASE-BY-PHASE PLAYBOOK

### PHASE 0 — Environment (BLOCKER)
**Why first:** `numpy, pandas, polars, pyarrow, scipy, sklearn, rapidfuzz` are **all missing**. Pure Python extrapolates to ~46 days for one full retrieval pass. Nothing below runs without this.
**Do:** install + pin the above; fix global seed; create versioned config; `requirements.txt`.
**Accept when:** `python -c "import numpy,pandas,pyarrow,scipy,sklearn,rapidfuzz"` succeeds and versions are pinned.

### PHASE 1 — Ingestion
**In:** 7 raw TSVs (2.4 GB). **Out:** `data/interim/{train,test}_{source1,source2,source3}.parquet`, `train_ground_truth.parquet`.
**Spec:** `sep="\t"`, `dtype=str`, `keep_default_na=False`, `quoting=QUOTE_NONE`. Chunk at 500k rows. `country` → categorical.
**Critical:** `na_filter` must be off — empty addresses are meaningful (4.41% of true-matched records), not nulls.
**Accept when:** row counts match exactly — S1 2,206,821 · S2 5,034,616 · S3 5,285,603 · GT 2,206,821 · test 1,732,544 / 4,887,273 / 5,082,316.

### PHASE 2 — Structural audit (fail-fast gate)
**Assert:**
1. ID prefixes `S1-`/`S2-`/`S3-` consistent with file.
2. GT ↔ Source-1 ID sets identical.
3. All matched IDs resolve in S2/S3; no S1 self-matches.
4. **Many-to-one:** `len(all_matched_ids) == len(set(all_matched_ids)) == 7,638,365`.
5. **Country:** every GT pair agrees.
**Accept when:** all 5 pass. If 4 or 5 fail, **stop** — D1, D11 and D17 all depend on them.

### PHASE 3 — Representation build (per-record, fits nothing)
**In:** Phase 1. **Out:** enriched tables keyed by `entity_id`.
**Order within the phase matters:**
1. NFKC normalize; collapse whitespace.
2. Strip only Unicode `P*`/`S*`/`Cc`/`Cf` → space. **(D2)**
3. Casefold → `name_norm`, `addr_norm`. **Keep `business_name`, `business_address` verbatim. (D3)**
4. Script-tag each field: `latin | deva | indic-other | mixed`.
5. Extract `legal_suffix[]`; emit `name_core`; **leave the name intact. (D4)**
6. Tokenize to order-insensitive sets: `name_tokens[]`, `addr_tokens[]`.
7. **Transliterate** non-Latin names/addresses; **union the resulting tokens into `name_tokens[]` / `addr_tokens[]`.** Do not create a separate channel. **(D6, D7)**
8. Parse additively: `addr_pin`, `addr_numbers[]`, `addr_state_tok`. Leave `addr_norm` untouched.
9. Emit `name_char3grams[]` — for **pair features only**, never for retrieval scoring. **(D7)**

**Transliteration spec (rule-based, validated):** realign Brahmic blocks (Bengali 0x0980, Gurmukhi 0x0A00, Gujarati 0x0A80, Oriya 0x0B00, Tamil 0x0B80, Telugu 0x0C00, Kannada 0x0C80, Malayalam 0x0D00) to Devanagari by codepoint offset; consonant → Latin + inherent `a`; matra overrides the inherent vowel; virama suppresses it; skip nukta; anusvara→`n`, visarga→`h`; **delete word-final schwa**. Verified output: `राम मार्केटिंग प्राइवेट लिमिटेड` → `raam maarketing praaivet limited`; `ಕರ್ನಾಟಕ` → `karnaatak`; `સિટી ટ્રેડિંગ` → `sitee treding`.

**Runs on train and test alike** — it fits nothing, so this is not leakage.
**Accept when:** no row loses its raw fields; Devanagari round-trips to multi-character tokens (spot-check the 7 strings above).

### PHASE 4 — Leakage-safe split (the gate)
**In:** Phase 3. **Out:** `split_assignment.parquet` (`entity_id` → `train|val`).
**Spec:** 95/5 over **Source-1 entities**, stratified by `country`, seeded. Because the match graph is a star forest, grouping by match-component is identical to grouping by entity — no component machinery needed.
**Sizes:** ~2,096,480 train / ~110,341 val.
**Rule from here on:** nothing fitted may see validation labels. **(D14)**
**Accept when:** stratification within 0.1% per country; no S2/S3 record owned by entities on both sides (guaranteed by many-to-one, but assert it).

### PHASE 5 — Validation pool construction
**Why this phase exists:** it is the single largest source of validation optimism. A naive 5% slice searched against only 5% of S2/S3 faces ~20× fewer distractors than reality.
**Spec:** val pool = the S2/S3 records owned by the 110,341 val entities **+ padded unmatched distractors until the ratio reaches 1 : 5.75**, per country. **(D12)**
**Accept when:** `|val_pool| / |val_entities| ≈ 5.75` per country.

### PHASE 6 — Fitted statistics
**Label-free (fit per searched pool — train pool, val pool, and at inference the test pool):** token `df`, `idf = log(N/(1+df))`, stopword list at `df ≥ 60,000`.
**Label-derived (95% internal-train ONLY):** match-rate priors, singleton rate, decision threshold, calibration, any learned blocking weights.
**Accept when:** a code-level assertion prevents label-derived statistics from touching val rows.

### PHASE 7 — Candidate index construction
**Spec:** one CSR/inverted index per **(country × source)** shard — 4 in train, 6 at test (France). Tokens with `df ≥ 60,000` excluded (they carry ~0 IDF: `limited` 11.65%, `private` 11.32%, `llc` 10.66%). Postings as `int32`.
**Parallelism:** shards across 20 cores. Each shard's index fits in RAM (largest observed: 45M postings).
**Accept when:** each shard index builds and the postings count is logged.

### PHASE 8 — Candidate generation + instrumentation
**Spec:** for each S1 entity, score same-country S2/S3 records as
`score = Σ_{name tokens} idf + 0.7 × Σ_{addr tokens} idf`
Take **top-50**. Emit `candidate_pairs` in the official schema.
**Instrument — mandatory, this is what the organizers audit:** recall@K and reduction ratio, reported **per country AND per script**.
**Expected on validation (India):** R@10 86.15 · R@25 89.38 · **R@50 90.80** · R@100 92.01 · R@200 93.78; by script latin 93.1% / deva 78.2% / indic-other 81.4%. **US should be ≈96.8% at K=50.**
**Accept when:** measured R@50 is within ~1pp of the above. A large shortfall means Phase 3 step 7 (transliteration) or step 2 (punctuation) is wrong — check those first.

### PHASE 9 — Pair-feature preparation
**In:** the ~50 candidates per entity from Phase 8 — only now is the pair set small enough.
**Feature families:** token Jaccard / containment (asymmetric — noise-token injection is common) on name and address · raw-exact and norm-exact name match · `legal_suffix` agreement · `addr_pin` agreement (**5.07% coverage but 93.30% precision when present — a verification feature, never a blocking key**) · shared address numbers · char-3gram similarity · script-pair indicator · retrieval score and rank.
**Forbidden:** any feature derived from the ground-truth match sets (e.g. "this record is already claimed") — it encodes the answer and is uncomputable at inference.
**Accept when:** every feature is computable from test inputs alone.

### PHASE 10 — Model-ready artifacts + reproducibility
**Out:** `data/processed/` train/val pair matrices, `candidate_pairs.tsv` writer, manifest (seeds, config hash, library versions), single-command regeneration.
**Accept when:** one command regenerates every artifact from the raw TSVs.

---

## PART 3 — HANDOFF NOTES TO MODELLING

Not my decisions, but they follow from the data and should not be rediscovered late:

1. **Exploit mutual exclusivity.** No S2/S3 record is ever claimed by two S1 entities (verified on all 7,638,365 pairs). A global deconfliction/assignment post-process converts this into precision — the expensive direction under F_0.5.
2. **Train on the metric you're scored on.** Entities with ≤2 matches are 28.0% of the macro-average but only 11.4% of pairs; singletons are 5.58% of the score and 0% of pairs. A pair-level objective silently ignores 5.58% of the score.
3. **Threshold conservatively.** Test is distractor-richer (1:5.75 vs 1:4.68) and precision is weighted 2×. A validation-tuned threshold is optimistic; bias it toward precision.
4. **France is ~15% of the macro-average with zero training data** and cannot be validated by any construction. Prefer country-agnostic mechanisms; audit France candidate counts at inference for anomalies.
5. **6 GB VRAM** makes a cross-encoder over ~87M test pairs implausible. Size the matcher accordingly.

## Residual risks

| Risk | Mitigation |
|---|---|
| Country losslessness verified on train only | Audit France candidate counts at inference |
| France address grammar unobservable | Country-agnostic fallbacks throughout |
| Rule-based transliteration is approximate | Measured +5.2pp deva; revisit with IndicXlit only if licence/size compliant |
| Recall@50 still ~78–81% for Indic scripts | Largest remaining headroom; try the dual-pass `w_addr` |

---

*Implementation requires explicit human authorization. Nothing in this document has been executed.*
