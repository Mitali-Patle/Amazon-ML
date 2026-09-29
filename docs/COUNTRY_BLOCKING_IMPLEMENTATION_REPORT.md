# COUNTRY BLOCKING — IMPLEMENTATION REPORT

**Date:** 25 Sep 2026 · **Branch:** `kundhave-dev` · **Config hash:** `7353aff5b0eb`
**Strategy doc:** `docs/COUNTRY_BLOCKING_STRATEGY.md` (written and evidenced before any code changed)

---

## 1. WHAT WAS IMPLEMENTED

Country × source partitioned retrieval with a bounded country-agnostic fallback, replacing the previous pooled-source retrieval.

| Component | File | Role |
|---|---|---|
| `country_normalizer` | `src/prep/country.py` | `normalize_country()` → `(country_key, status)`; alias table; `resolve_partition()`; `needs_fallback()` |
| `country_partitioner` | `src/prep/retrieval.py` | `build_partitions()` — discovers countries from the pool, one index per (country × source) |
| `retrieval_index` | `src/prep/retrieval.py` | `Partition` dataclass: ids, script, vocab, idf, CSR matrix, stats |
| `fallback_retriever` | `src/prep/retrieval.py` | `build_agnostic()` — lazily built, same machinery, country restriction lifted |
| `candidate_generator` | `src/prep/retrieval.py` | `retrieve_source()` routes each query; `merge_sources()` dedups keeping best score |
| `candidate_recall_evaluator` | `src/prep/run_blocking.py` | `score_recall()` — recall@K by country, source, script, path |
| audit | `src/prep/exp_country_audit.py` | the byte-level country investigation |
| tests | `tests/test_country_blocking.py` | TEST A–E, 34 assertions |

`src/prep/run_validation.py` is retained for the pooled-source baseline; `run_blocking.py` is the new path.

---

## 2. COUNTRY NORMALIZATION BEHAVIOUR

**Evidence first: the data did not require it.** All 24.2M records carry exactly three values — `US`, `India`, `France` — with zero empty values, zero whitespace variants, zero casing variants, zero NFKC-unstable values, and no two values collapsing under strip+casefold. No `USA`/`United States`/`IND` variants exist.

A deliberately thin layer was implemented anyway because the failure is silent and the cost is ~60 lines: NFKC → strip → collapse whitespace → casefold → small alias table covering only the three observed countries. **No invented world-geography dictionary.**

Fields are additive; raw is preserved:

| field | meaning |
|---|---|
| `country` | raw, untouched |
| `country_key` | partition key |
| `country_status` | `ok` \| `aliased` \| `missing` \| `unknown` |

An unrecognised but well-formed value is **never discarded** — it keeps a canonical spelling and becomes its own partition key; the partitioner then decides whether such a partition exists in the pool.

Verified routing (`tests/test_country_blocking.py`):

```
'US' -> ('US', ok)                'IND'  -> ('India', ok/aliased)
'  usa ' -> ('US', aliased)       'fr'   -> ('France', aliased)
'U.S.A.' -> ('US', aliased)       ''     -> ('', missing)
'united states of america' -> ('US', aliased)     None -> ('', missing)
```

---

## 3. KNOWN-COUNTRY BEHAVIOUR

Query routes to its (country × source) index. Partitions are **discovered** from `pool["country_key"].unique()` — there is no hard-coded country list anywhere in the codebase. Indexes built on the validation run:

| Partition | Records | Indexed keys |
|---|---|---|
| India × S2 | 2,017,799 | 666,590 |
| US × S2 | 3,016,817 | 817,635 |
| India × S3 | 2,115,547 | 670,280 |
| US × S3 | 3,170,056 | 848,134 |

---

## 4. UNSEEN-COUNTRY BEHAVIOUR (France)

The central correction. **"No training labels" does not mean "no retrieval index."** The index is label-free and built from the pool being searched; the test pool contains 703,378 French S2 records and 731,615 French S3 records.

Verified in TEST B with a synthetic France pool: France is routed to `FrancexS2` — a **partition, not the fallback** — the true match is retrieved at rank 1, and `France` appears in `available_partitions`, proving it was discovered from the data rather than hard-coded. At inference the same code will create `France × S2` and `France × S3` automatically.

The only thing withheld for an unseen country is **label-derived** statistics, which are not computed in this phase at all.

---

## 5. FALLBACK BEHAVIOUR

| Trigger | Condition |
|---|---|
| `country_status = missing` | empty / whitespace-only / null |
| `country_status = unknown` | well-formed but no such partition in the pool |

**Pool searched:** the entire source pool, country restriction lifted, same sparse machinery.
**K:** identical to the normal path.
**Bounding:** returns K candidates — never a Cartesian product. Verified in TEST C/E.
**Cost:** one extra index build per source **only if** at least one query needs it. `build_agnostic()` is lazy, so on clean data it costs **nothing**.
**Activation:** counted in `diagnostics["routed_to_fallback"]` and logged at INFO.

**Measured on the real validation run: `{'ok': 110341}` — 0 fallback activations, 0 extra cost.** The fallback exists for the case the data does not currently exhibit.

With `allow_fallback=False` the affected rows get an explicit empty list and a `"none"` path rather than an exception — never a silent drop.

---

## 6. RETRIEVAL ALGORITHM

For each S1 record: normalise country → resolve against pool partitions → retrieve from (country × source) index for S2 and S3 **separately** → merge, dedup keeping the best score, sort.

```
score = SUM idf(shared name tokens) + 0.7 * SUM idf(shared address tokens)
```

IDF applied on the **pool side only** (`side="pool"`); the query carries `1.0` / `addr_weight`. Applying it on both sides silently squares it — pinned by `tests/test_scoring.py`.

Nothing on the forbidden list was introduced: no char-n-gram retrieval scoring, no edit distance as primary retrieval, no phonetic blocking, no PIN/ZIP blocking key, no destructive suffix removal, no global ASCII folding, no state/street expansion.

---

## 7. RECALL@K RESULTS

110,341 validation entities, 382,095 true pairs, searched against the full train corpus.

### Mode comparison — the open design question, settled

| mode | @10 | @25 | @50 | @100 | mean cands |
|---|---|---|---|---|---|
| `combined_50` (previous) | 90.55 | 93.17 | 94.36 | — | 50 |
| **`per_source_25`** | 90.54 | 93.17 | **94.46** | — | **50** |
| `per_source_50` (adopted) | 90.54 | 93.17 | 94.39 | **95.38** | 100 |

**At equal budget (50 candidates), per-source beats pooled: 94.46% vs 94.36%.** The gain concentrates exactly where it matters:

| script | combined_50 | per_source_25 | Δ |
|---|---|---|---|
| **Devanagari** | 76.45% | **77.90%** | **+1.45pp** |
| **other Indic** | 77.40% | **78.41%** | **+1.01pp** |
| latin | 95.65% | 95.67% | +0.02 |

Mechanism: 80.48% of entities match in both S2 and S3, so a single global cut can be monopolised by one source. Cross-script records — whose name channel is weak and which depend on the address channel — are the ones starved. The +1.45pp on Devanagari (14,487 pairs, SE ≈ 0.35pp) is ~4 SE and clearly real.

### Adopted configuration: `per_source_50`

| metric | @10 | @25 | @50 | @100 |
|---|---|---|---|---|
| pair recall | 90.54 | 93.17 | 94.39 | **95.38** |
| entity recall | 90.65 | 93.15 | 94.36 | **95.33** |

**By country @50:** India 90.97% · US 96.67%
**By source @50:** S2 94.75% · S3 94.05%
**By script @50:** latin 95.64% · mixed 87.80% · deva 77.70% · indic-other 77.48%
**By path:** 100% partition, 0% fallback

### Against the playbook's expected values

| Reference | Expected | Measured | Δ |
|---|---|---|---|
| India R@50 | ≈90.80% | 90.97% | +0.17 |
| India R@10 | ≈86.15% | ~86.3% | +0.15 |
| US R@50 | ≈96.8% | 96.67% | −0.13 |
| latin | ≈93.1% (India-only) | 93.8% India / 96.7% US | consistent |
| Devanagari | ≈78.2% | 77.70% | −0.50 |
| other Indic | ≈81.4% | 77.48% | **−3.92** |

All within sampling noise except **other Indic, −3.92pp**. Explanation: the playbook figure came from a 1,500-query probe with only ~364 indic-other pairs (SE ≈ 2.1pp); this run measures 11,018 pairs (SE ≈ 0.4pp). The two intervals overlap, and the larger sample is the reliable one. Not a regression — a better estimate.

---

## 8. REDUCTION RATIO, RUNTIME, MEMORY

| | value |
|---|---|
| Unpartitioned S1 × (S2+S3), validation | 110,341 × 10,320,219 = 1.14e12 |
| Country × source partitioned, then top-50 each | 1.10e7 |
| **Reduction** | **~1.0e5×** |

Per-country full space → top-K: India 82,667× · US 123,738× (single-source measurement).

| | |
|---|---|
| Runtime, `per_source_50`, 110,341 queries | **1,265 s** (~21 min) |
| Peak RAM | ~8 GB of 15 GB |
| Largest index | US × S3, 3,170,056 records, 848,134 keys |
| Candidate file | 137 MB, 110,341 rows |

No code path materialises S1 × S2 or S1 × S3. Scoring is sparse `Q @ P.T` in query batches of 512; the only dense object is the per-batch score block.

---

## 9. TESTS PERFORMED

`tests/test_country_blocking.py` — **34 assertions, all passing**:

| Test | Result |
|---|---|
| **A** known country | routes to `USxS2`, candidates only from that partition, true match ranked first |
| **B** unseen country (France) | routes to `FrancexS2` **partition**, true match at rank 1, partition discovered from pool |
| **C** missing country | routes to fallback, still retrieves, true match reachable, activation logged |
| **D** variants (`"  us  "`, `"USA"`, `"U.S.A."`, `"united states of america"`) | all route to `USxS2` |
| **E** unknown-in-pool (`"Germany"`) | routes to fallback, candidates produced |
| no silent drops | every query gets a path and ≥1 candidate |
| bounded fallback | ≤ K candidates, no Cartesian explosion |
| fallback disabled | explicit empty + `"none"` path, no exception |
| merge | dedups, both sources represented, score-sorted |

Full suite: `test_normalize` PASS (27) · `test_scoring` PASS (14) · `test_country_blocking` PASS (34).

Output integrity on the real 110,341-row file: 110,341 unique source1 ids · **0** rows containing an `S1-` id · **0** rows with duplicate ids · **0** zero-candidate rows · S2 mean 50.0 / S3 mean 50.0 (per-source balance confirmed).

---

## 10. DEVIATIONS FROM THE PLAYBOOK

| # | Deviation | Justification |
|---|---|---|
| 1 | **S2/S3 retrieved separately, not pooled** | Measured: +0.10pp overall and +1.45pp Devanagari at equal budget |
| 2 | Adopted `per_source_50` (100 candidates) as default | Brief specifies K=50 per source; yields 95.38% vs 94.39%. `per_source_25` remains available if the matcher budget is capped at 50 — still beats the old pooled config |
| 3 | Country normalisation layer added though data needs none | Silent, catastrophic failure mode vs ~60 lines; alias table restricted to observed countries only |
| 4 | Fallback built lazily | Zero cost when unused — measured 0 activations on real data |

No other deviation. Nothing changed silently.

---

## 11. REMAINING RISKS

1. **Test-side country losslessness is unverifiable.** 0/7,638,365 cross-country pairs on train, rule-of-three 95% upper bound 0.000039%. Test has no labels. Mitigation is architectural: an unresolved country falls back rather than failing. Residual exposure is a *missed cross-country pair*, not a dropped record.
2. **France cannot be recall-validated** — ~15% of the test macro-average with no labels. The architecture is verified to route it correctly (TEST B), but its actual recall is unmeasurable. **Recommended inference check:** compare mean candidate count and score distribution for France against US/India; a marked shortfall is the tell.
3. **Cross-script recall remains the biggest lever** — 77.7% (deva) / 77.5% (indic-other) vs 96.7% (US latin). Per-source retrieval recovered ~1.5pp; the untested dual-pass (`addr_weight` 1.0 for non-Latin, unioned with the 0.7 pass) is the next experiment.
4. **Runtime.** 21 min for 110k queries ⇒ ~5.5 h for the 1.73M test entities. `to_matrix` is a pure-Python row loop and is the bottleneck; vectorising it is the obvious optimisation before the test run.
5. **Indexes are not persisted**, so each run rebuilds them (~35 s per partition). Acceptable now; worth caching if retrieval is re-run repeatedly.

---

## 12. FILES CHANGED

**New:** `src/prep/country.py` · `src/prep/retrieval.py` · `src/prep/run_blocking.py` · `src/prep/exp_country_audit.py` · `tests/test_country_blocking.py` · `docs/COUNTRY_BLOCKING_STRATEGY.md` · this report.
**Modified:** `src/prep/candidates.py` (unchanged scoring; reused by `retrieval.py`).
**Unchanged:** raw TSVs (verified immutable), `normalize.py`, `ingest.py`, `audit.py`, `represent.py`, `split.py`.
**Artifacts:** `data/candidates/val_candidate_pairs_per_source_50.tsv` (137 MB) · `data/reports/exp_country_audit.json` · `data/reports/phase8b_country_blocking.json`.
