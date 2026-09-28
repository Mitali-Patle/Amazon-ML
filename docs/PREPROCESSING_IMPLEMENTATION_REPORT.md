# PREPROCESSING IMPLEMENTATION REPORT

**Date:** 25 Sep 2026 · **Branch:** `kundhave-dev` · **Config hash:** `7353aff5b0eb` · **Seed:** `20260925`
**Scope:** ingestion, structural audit, representations, 95/5 split, corpus statistics, candidate generation, recall measurement.
**Stopped before:** model training, model selection, final submission — as instructed.

Reproduce everything with:

```bash
.venv/bin/python -m src.prep.run_pipeline
```

---

## 1. PLAYBOOK VERIFICATION

### Verified and implemented as specified

| Decision | Verification |
|---|---|
| D1 country hard partition | Re-verified from parquet: **7,638,365 / 7,638,365** pairs agree, 0 disagree, 0 entities spanning countries |
| D2 category-aware punctuation | Unit-tested: `राम मार्केटिंग प्राइवेट लिमिटेड` keeps 4 tokens; `[^\w\s]` control shatters it into 15 fragments |
| D3 raw preserved | All 6 representation files carry `business_name` / `business_address` verbatim |
| D4 suffix extracted not deleted | Implemented; `name_core` is emitted as a feature and never used as a key |
| D6 transliteration as folded tokens | 752,869 train records transliterated (S2 474,345 + S3 278,524 — exactly the probe figure) |
| D7 no char-n-gram scoring channel | `char_ngrams` exists for pair features only; retrieval indexes name/addr tokens only |
| D8 top-K retrieval | Implemented as sparse `Q @ P.T` with per-row top-K |
| D10 K=50 | Measured across K ∈ {10,25,50,100,200}; 50 confirmed as the knee |
| D11 95/5 entity split | 2,096,480 / 110,341 = exactly 5.000%, stratified per country to 5.0% |
| D14 label-derived stats train-only | No label-derived statistic is computed in this phase at all |
| D17 (country × source) sharding | Retrieval runs per country; vocabulary and index built per partition |
| D18 representations before split | Phase 3 fits nothing; verified by construction |
| INV-3 many-to-one | Re-verified: unique matched IDs = total = 7,638,365 |
| INV-5 S1 deduplicated | 0 exact duplicates on (name, address, country) |

### Changed — playbook was wrong

**D12 → D12b: validation pool must be the FULL corpus, not ratio-matched.**

*Playbook said:* pad the validation pool to the test ratio of 1 : 5.75 (≈634k records).

*Data showed:* a true match's rank is set by how many distractors outscore it, which scales with the **absolute** pool size, not the ratio. Measured on identical queries, identical weights, identical code path:

| partition | ratio pool (634k) | full pool (10.32M) | inflation |
|---|---|---|---|
| India R@50 | 95.63% | 90.92% | **+4.71pp** |
| US R@50 | 98.92% | 96.75% | **+2.17pp** |

*Why they differ:* the playbook reasoned about pool *composition* and silently assumed that fixing the ratio fixes difficulty. It does not — ratio controls the proportion of distractors, absolute size controls how many there are.

*Implemented instead:* validation entities are searched against the **full train S2/S3 corpus (10,320,219 records)**, within 3.5% of the test pool size (9,969,589), so difficulty matches test. Including records owned by training entities is not leakage — at test time an entity is likewise searched against records owned by other entities, and no labels are consulted. The ratio-matched pool is still built and saved (`data/validation/val_pool.parquet`) but is **not** used for the headline numbers.

*Evidence:* `data/reports/exp_pool_size_comparison.json`, reproducible via `python -m src.prep.exp_pool_size`.

### Rejected — confirmed the playbook's rejections, did not reintroduce

- **State/street abbreviation expansion** (D5) — not implemented. The playbook measured −1.04pp and a ceiling drop 99.50→98.85. Not re-litigated.
- **Char-n-gram scoring channel** (D7) — not implemented in retrieval. Measured collapse 90.1→27.6.
- **`[^\w\s]` stripping, global ASCII folding, destructive suffix removal, dropping empty-address rows, S1 deduplication, one-hot country, threshold-union blocking, PIN as blocking key** — none implemented; each has a recorded counter-measurement.

### Bugs found during implementation

**B1 — squared IDF in the sparse scorer (silent, cost ~2pp recall).**
The target score is linear: `Σ idf(name) + 0.7 · Σ idf(addr)`. My first implementation applied IDF weights to *both* the query and pool matrices, so `Q @ P.T` computed `Σ idf²` and the address weight became `0.7² = 0.49`. It produced no error — only a quietly wrong ranking. Caught because US recall@50 came out **94.65%** against the independently-measured probe value of 96.81%.
*Fix:* weights are applied on exactly one side (`side="pool"` carries IDF, `side="query"` carries 1.0 / `addr_weight`). After the fix US measured **96.75%** and India **90.92%**, reproducing the probes (96.81% / 90.80%).
*Regression guard:* `tests/test_scoring.py` asserts the linear formula numerically and asserts that the both-sides variant produces a different number.

**B2 — cross-script legal suffixes were invisible.**
`legal_suffix` was extracted from pre-transliteration tokens, so a Devanagari `लिमिटेड` yielded `[]` while its Latin counterpart yielded `['ltd']` — making suffix agreement systematically false-negative for exactly the cross-script subgroup that already performs worst.
*Fix:* suffix detection resolves Indic tokens through transliteration and maps every variant to a canonical class. Variant lists are **empirical**, taken from the top transliterated tokens actually observed in Indic-script names (`limited` 481,001 · `praaivet` 274,666 · `praivet` 86,304 · `limitet` 44,199 · `praaibhet` 42,900 · `piraivet` 38,080 · `elaelapee` 26,995), not guessed. `राम मार्केटिंग प्राइवेट लिमिटेड` and `Akriti Mindset Private Limited` now both yield `['pvt','ltd']`.
*Note:* `li` and `pra` (from the Indian abbreviation `प्रा. लि.`, ~63k occurrences each) are recorded as weak suffix hints but never stripped from `name_core` — too short, too collision-prone.

**B3 — negative IDF footgun.** `log(N/(1+df))` goes negative once a token appears in over half the corpus, which would make sharing a token *penalise* a candidate. The `DF_STOPWORD` cut keeps df/N ≤ ~1.5% in production so it never binds, but the value is now clamped at 0.

---

## 2. ENVIRONMENT

Phase 0 was a hard blocker — the system Python is externally managed with no `pip`, and none of the numeric stack was present. Created `.venv` (Python 3.14.6).

| Package | Version |
|---|---|
| numpy | 2.5.3 |
| pandas | 3.0.6 |
| polars | 1.44.2 |
| pyarrow | 25.0.1 |
| scipy | 1.18.1 |
| RapidFuzz | 3.14.6 |

Pinned in `requirements.txt`. Hardware: 20 cores, 15 GB RAM, RTX 4050 (6 GB, unused in this phase), 398 GB free disk.

---

## 3. DATA INGESTION

Raw TSVs are **immutable** — nothing in the pipeline writes inside the dataset directory. All output goes to `data/`.

| File | Rows | Expected | Parquet |
|---|---|---|---|
| train_source1 | 2,206,821 | ✓ | 78.8 MB |
| train_source2 | 5,034,616 | ✓ | 191.6 MB |
| train_source3 | 5,285,603 | ✓ | 197.5 MB |
| train_ground_truth | 2,206,821 | ✓ | 54.7 MB |
| test_source1 | 1,732,544 | ✓ | 63.7 MB |
| test_source2 | 4,887,273 | ✓ | 193.1 MB |
| test_source3 | 5,082,316 | ✓ | 195.0 MB |

2.4 GB → 975 MB, **5.2 s** total. Checks passed: row counts, ID prefix consistency, ID uniqueness, zero null cells.

**Critical parsing detail:** `empty_string_is_null=False`. Empty addresses are *data*, not missing values — 168,967 in S2 and 175,916 in S3, and 4.41% of true-matched records have one. Letting the reader null them and dropping the rows would discard real matches. Verified the counts survive ingestion exactly.

### Phase 2 — structural audit (all invariants re-verified from parquet)

```
[OK] INV-1  GT<->S1 id sets identical         only_in_s1=0 only_in_gt=0
[OK] INV-2  all matched ids resolve            dangling=0 s1_self_matches=0
[OK] INV-3  many-to-one                        total=7,638,365 unique=7,638,365
[OK] INV-4  country agreement                  7,638,365 pairs, disagree=0
[OK] INV-5  S1 exact duplicates                0
```

Descriptive: 123,247 singletons (5.58%) · mean 3.461 matches/entity · max 11 · S2 3,693,619 / S3 3,944,746 pairs · cross-tab `US→US 4,578,522`, `India→India 3,059,843` · 2,681,854 S2/S3 records match nothing.

---

## 4. REPRESENTATIONS

Per record, 14 columns. Raw always retained.

| Layer | Fields |
|---|---|
| raw | `business_name`, `business_address`, `country` |
| norm | `name_norm`, `addr_norm` (NFKC → category-safe punctuation → casefold) |
| script | `name_script`, `addr_script` ∈ {latin, deva, indic-other, mixed} |
| derived | `name_tokens[]`, `addr_tokens[]` (transliterated forms unioned in), `legal_suffix[]`, `name_core`, `addr_pin`, `addr_numbers[]` |

Built in **3 m 51 s** across 18 workers; 24.2 M records → 3.1 GB parquet. Row counts preserved exactly; zero records lost all name tokens.

Transliteration counts: train S2 474,345 · S3 278,524 (= 752,869, matching the probe exactly) · test S2 546,606 · S3 320,639. `train_source1` and `test_source1` required **zero** — Source 1 is 100% ASCII, as the report found.

Verified sample:
```
RAW    राम मार्केटिंग प्राइवेट लिमिटेड
tokens ['राम','मार्केटिंग','प्राइवेट','लिमिटेड','raam','maarketing','praaivet','limited']
suffix ['pvt','ltd']       core 'राम मार्केटिंग'
```
Both scripts retained; the Latin `limited` is what lets this record meet an English Source-1 name.

**Not implemented, deliberately:** state/street-expanded token stream (D5), char-n-gram retrieval keys (D7).

---

## 5. VALIDATION SPLIT

95/5 over Source-1 entities, stratified by country, seeded `20260925`, saved to `data/validation/split_assignment.parquet`. Never regenerated per run.

| | train | val | val % |
|---|---|---|---|
| India | 839,029 | 44,159 | 5.0% |
| US | 1,257,451 | 66,182 | 5.0% |
| **total** | **2,096,480** | **110,341** | **5.000%** |

Entity-level grouping is sufficient because the match graph is a **forest of stars** (INV-3: many-to-one). This is asserted, not assumed: **0 S2/S3 records are claimed across both folds**.

---

## 6. LEAKAGE CONTROLS

| Control | Status |
|---|---|
| Official test set untouched | Test files read only for ingestion + deterministic representations. No labels, no thresholds, no tuning. Raw files never modified. |
| Nothing fitted before the split | Phase 3 is purely per-record; the only fitted quantity (df/IDF) is computed in phase 6, after phase 4. |
| Label-derived statistics | **None computed in this phase.** Thresholds, priors and calibration belong to modelling and are train-only. |
| df/IDF scope | Computed over the pool being searched — label-free and available at inference, so it mirrors test rather than mis-simulating it. |
| Records straddling folds | Asserted = 0. |
| Validation difficulty | Corrected per D12b; the ratio pool would have inflated recall by up to +4.71pp. |

---

## 7. CANDIDATE GENERATION

Per-country top-K retrieval. `score = Σ idf(shared name tokens) + 0.7 · Σ idf(shared address tokens)`, tokens with df ≥ 60,000 excluded from the index, sparse `Q @ P.T` in query batches of 512.

Measured on **all 110,341 validation entities** against the full train corpus — 382,095 true pairs.

### Pair recall@K

| partition | @10 | @25 | **@50** | @100 | @200 |
|---|---|---|---|---|---|
| India | 86.22 | 89.34 | **90.89** | 92.15 | 93.52 |
| US | 93.47 | 95.72 | **96.67** | 97.29 | 97.93 |
| **aggregate** | **90.57** | **93.17** | **94.36** | **95.23** | **96.17** |

### Entity recall@K (the shape the macro-averaged metric actually rewards)

| partition | @10 | @25 | **@50** | @100 | @200 |
|---|---|---|---|---|---|
| India | 86.26 | 89.25 | **90.78** | 92.01 | 93.40 |
| US | 93.63 | 95.77 | **96.69** | 97.31 | 97.94 |
| **aggregate** | **90.68** | **93.16** | **94.32** | **95.19** | **96.12** |

Pair and entity recall track within 0.1pp, so K=50 is not favouring high-match entities.

### By script @50 — where the remaining loss sits

| script | true pairs | recall@50 |
|---|---|---|
| latin (US) | 229,298 | 96.7% |
| latin (India) | 125,226 | 93.8% |
| **deva** | 14,487 | **76.5%** |
| **indic-other** | 11,018 | **77.4%** |
| mixed | 2,066 | 88.1% |

### Reduction ratio

| partition | full pair space | top-50 | reduction |
|---|---|---|---|
| India | 44,159 × 4,133,346 | 2,207,950 | **82,667×** |
| US | 66,182 × 6,186,873 | 3,309,100 | **123,738×** |

**K=50 confirmed.** Beyond it each doubling buys ~0.9pp (94.36 → 95.23 → 96.17) while doubling downstream matcher cost. The residual gap is a cross-script *ranking* problem, not a K problem.

---

## 8. COMPUTATIONAL PERFORMANCE

| Phase | Wall time | Peak notes |
|---|---|---|
| 0 environment | ~3 min | one-off |
| 1 ingestion | **5.2 s** | chunked, <1 GB |
| 2 audit | **8.2 s** | ~3 GB (ID sets in memory) |
| 3 representations | **3 m 51 s** | 18 worker processes |
| 4/5 split + pool | **~40 s** | |
| 8 candidate generation | **9 m 25 s** | India 187 s + US 249 s retrieval |
| **total** | **~14 min** | |

Disk: `parquet` 975 MB · `representations` 3.1 GB · `candidates` 70 MB · `validation` 16 MB · `reports` 40 KB. Raw stays 2.4 GB, untouched.

Chunking (ingestion, 500k rows) · batching (retrieval, 512 queries) · sharding (per country × source) are used distinctly as specified. GPU unused — sparse CPU retrieval is fast enough.

---

## 9. VALIDATION RESULTS SUMMARY

Headline: **entity recall@50 = 94.32%**, pair recall@50 = 94.36%, reduction 82,667–123,738×, mean 50.0 candidates/entity.

Output integrity (`data/candidates/val_candidate_pairs.tsv`): 110,341 rows = exactly one per validation entity, ID set matches the split exactly, mean/min/max candidates all 50, **0** rows containing an `S1-` id, **0** rows with duplicate ids in a list. Format matches the official `candidate_pairs.tsv` schema.

Cross-check against the independent planning-phase probes, which used a completely separate pure-Python implementation:

| | probe | pipeline | Δ |
|---|---|---|---|
| India R@50 | 90.80% | 90.92% | +0.12 |
| US R@50 | 96.81% | 96.75% | −0.06 |

Two independent implementations agreeing within 0.12pp is the strongest evidence available that the retrieval is correct.

---

## 10. DEVIATIONS FROM PLAYBOOK

| # | Deviation | Reason |
|---|---|---|
| 1 | **Validation pool = full corpus, not ratio-matched (D12→D12b)** | Measured +4.71pp / +2.17pp inflation. Playbook doc updated. |
| 2 | `legal_suffix` resolves Indic forms via transliteration; variants are empirical | Original spec produced empty suffixes for all cross-script records (B2) |
| 3 | IDF clamped at ≥ 0 | Removes a negative-IDF footgun (B3); no effect in production ranges |
| 4 | `li` / `pra` treated as weak hints, never stripped | Too short and collision-prone to remove safely |
| 5 | Indexes not persisted to `data/indexes/` | Rebuild is ~35 s/partition vs ~1 GB on disk; will revisit if modelling needs repeated retrieval |
| 6 | Train-split candidates not yet generated | Only validation is needed to measure the ceiling; train candidates are a modelling-phase input. Command is ready. |

No other deviation. Nothing was changed silently.

---

## 11. PRODUCED ARTIFACTS

**Code** — `src/prep/`: `config.py` (all locked parameters + config hash), `normalize.py`, `ingest.py`, `audit.py`, `represent.py`, `split.py`, `candidates.py`, `run_validation.py`, `exp_pool_size.py`, `run_pipeline.py`.
**Tests** — `tests/test_normalize.py` (27 assertions), `tests/test_scoring.py` (14 assertions). Both pass.
**Data** — `data/parquet/` (7 files), `data/representations/` (6 files), `data/validation/split_assignment.parquet` + `val_pool.parquet`, `data/candidates/val_candidate_pairs.tsv`.
**Reports** — `data/reports/`: `phase1_ingest.json`, `phase2_audit.json`, `phase3_represent.json`, `phase4_split.json`, `phase8_candidate_recall.json`, `exp_pool_size_comparison.json`, `run_manifest.json`.
**Config** — `requirements.txt`; `.gitignore` excludes `data/` but keeps `data/reports/*.json` tracked.

---

## 12. NEXT STEPS

**Ready for modelling:**
1. Representations for all 6 source files, train and test, with raw preserved.
2. A frozen, leakage-checked 95/5 split.
3. Candidate generation achieving **94.32% entity recall@50** — the hard ceiling every downstream model inherits.
4. `val_candidate_pairs.tsv` in official format, integrity-verified.
5. One-command reproduction with a config hash and version manifest.

**Not done, by instruction:** no model trained, selected or tuned; no predictions; no submission.

**Open items for whoever picks this up:**
1. **Cross-script recall is the biggest remaining lever** — 76.5% (deva) / 77.4% (indic-other) against 96.7% for US Latin. The playbook's untested dual-pass idea (`addr_weight` 1.0 for non-Latin records, unioned with the 0.7 pass) is the obvious next experiment.
2. **Train-split candidates must be generated** before matcher training — same command, ~3 h for 2.1 M entities at the measured rate.
3. **France remains unvalidatable.** It is ~15% of the test macro-average with zero labelled data. Country-agnostic paths are preserved throughout, but no measurement is possible.
4. **Recall@50 is a ceiling, not a score.** 94.32% recall caps what any matcher can achieve; precision is entirely the matcher's problem, and F_0.5 weights it 2×.
5. **Pair features (Phase 9) are not built.** `char_ngrams`, `addr_pin`, `legal_suffix` and the retrieval score/rank are all available as inputs.

---

*Preprocessing implementation complete. Stopping before model development, as instructed.*
