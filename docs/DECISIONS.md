# DECISIONS.md — Architecture Decision Log (ADR-style)

Append-only. Never delete another agent's entries. Each ADR: context → options → decision → consequences → open questions.

---

## ADR-001: Train/Val Split Strategy for Entity Resolution (source1 anchors across shared source2/source3 candidate pools)

**Author:** competition-strategist (drafted Day 1, pre-PROBLEM_BRIEF.md, based on raw schema inspection). **Status:** PROPOSED — pending data-detective's graph/near-dup audit and validation-guardian's implementation + ratification. ml-architect should review/ratify given this is a core architecture decision.

**Date:** 2026-09-25 (H+0, problem-analyst has not yet produced docs/PROBLEM_BRIEF.md; metric is unconfirmed — see open questions).

### Context

2026 is entity resolution / record linkage across three business-listing sources, not a 2023-25-style regression/extraction task. Schema (train): `source1` (2,206,822 rows, anchor entities), `source2` (5,034,617 rows) and `source3` (5,285,604 rows) are large shared candidate pools, and `train_ground_truth.tsv` has exactly one row per `source1_entity_id` with a comma-separated list of matched `S2-*`/`S3-*` ids (123,247 rows / ~5.6% have an empty list = no match anywhere). No duplicate `source1_entity_id` in ground truth (checked). `data/test/*` has the same three source files with NO ground truth and must never be split, sampled, or used for threshold-tuning — it is write-only (predictions out), per the shared playbook's anti-overfitting-to-LB principle.

We need a train/val split of `data/train` for local CV that (a) has no leakage, (b) is realistic enough to stand in for the private LB, and (c) is cheap enough to iterate on inside the ~72h hackathon window (kickoff 25 Sep 00:00 IST → 27 Sep 23:59 IST).

### Decision 1 — Splitting unit: match-graph connected components, not raw `source1_entity_id`

**Options considered:**
- (a) Split individual `source1_entity_id` rows randomly (simplest).
- (b) Split at the connected-component level of the bipartite/tripartite match graph (nodes = all S1/S2/S3 ids that appear in ground truth; edges = ground-truth match pairs), merged with near-duplicate text clusters.

**Decision: (b).** Ground truth is keyed one-row-per-source1-entity, which makes (a) tempting, but two risks make raw row-level splitting insufficient:
1. **Cross-anchor sharing of the same S2/S3 record.** It is not yet confirmed whether an `S2-*`/`S3-*` id can appear in more than one source1 row's matched-ids list. If it can, that S2/S3 record's identity spans two anchors — splitting those anchors into different folds means the SAME target record is a labeled positive on both sides of the split, which breaks the clean "train pool vs. val pool" separation needed for Decision 2 below. Connected components make this a non-issue by construction: a component that contains matches shared across anchors is never split.
2. **Near-duplicate source1 rows.** If the same real business appears as near-identical source1 rows (e.g., a scrape duplicate, or a chain location where name/address differ only slightly), a model can memorize name/address patterns from a train-fold twin and get inflated val performance on its near-identical val-fold twin. Connected components do NOT catch this for the ~5.6% no-match rows (they have no graph edges to group by), so this must be caught separately by data-detective via near-dup text clustering (normalized name + address similarity, e.g. MinHash/LSH), then merged into the same grouping via union-find.

**Consequences:** most components will likely be small (singletons or size 2-4, given matches look closer to 1:1/1:few rather than dense many:many) — a large number of small groups, which is fine for stratified group sampling at this row count (2.2M). This must be validated once problem-analyst/data-detective confirm the match cardinality (see open questions).

### Decision 2 — Val-specific candidate pool vs. full shared pool

**Options considered:**
- (a) Full pool for both: run val-fold retrieval/scoring against the entire train source2+source3 files (matching real test-time scale/difficulty).
- (b) Carved pool: hold out just the true matches for val-fold anchors + sampled distractors, and only score against that curated subset.

**Decision: hybrid, two-tier.**
- **Tier 1 (fast dev loop, most iteration):** curated val candidate pool = true matches for val-fold anchors (safe once Decision 1's grouping is respected) + hard-negative distractors mined via blocking key (same normalized name token / same country / same city where address permits) + random negatives from the pool of S2/S3 records that are unmatched by ANY source1 entity (these carry no train/val identity at all — a record either belongs to a train-only, val-only, or no-match component; the "matched-by-nobody" reservoir is safe to freely share between train and val since it can never leak a label). Sized to mirror realistic per-query candidate density.
- **Tier 2 (trustworthy, run before every submission and at major checkpoints):** re-score val-fold anchors against the FULL source2+source3 train pool — this is the one CV number that stands in for the private LB. Because Tier 2 is expensive at 2.2M-anchor scale, use a stratified sub-sample of val (e.g. 5-10k anchors) for frequent spot-checks, and the full val set only at major checkpoints (roughly every ~12h) and pre-submission.

**Rationale:** Tier 1 alone risks being optimistic if distractor sampling is too soft; Tier 2 alone is too slow to run after every small change. Never report/trust Tier 1 numbers as "the" CV — only Tier 2 numbers go in `docs/EXPERIMENTS.md` as the CV column.

**Critical leakage rule that is independent of row-level splitting:** any learned component used for candidate generation or scoring (fine-tuned embedding/bi-encoder for blocking, learned thresholds, learned blocking rules) must be fit using ONLY train-fold matched/unmatched pairs. Fitting such a component on the full ground truth before splitting leaks val-fold match signal into the retrieval stage itself, even if row-level IDs are cleanly partitioned. validation-guardian must audit this explicitly.

### Decision 3 — Stratification and ratio

**Decision:** stratify groups (from Decision 1) by (has-at-least-one-match flag, country-bucket) so the val split preserves the ~5.6% no-match rate and the country distribution. Recommend a single fixed stratified holdout (not k-fold) at roughly 90/10 or 95/5 (group-level, fixed seed, e.g. 42), because:
- 2.2M anchors is large enough that even a 5% val (~110k anchors, ~6.2k of them no-match) gives statistically stable estimates — k-fold would multiply the expensive Tier-2 full-pool scoring cost k-fold for little variance benefit.
- Time is the scarcest resource in a 72h window; a single reversible, cheap holdout is preferred over k-fold per the shared playbook.

### Decision 4 — Usage discipline going forward

- This val split stands in for the private LB. Every threshold decision (match/no-match cutoff, top-k matches per anchor given multi-match is possible) is tuned ONLY on val (Tier-2 numbers), never on public LB probing.
- `data/test/*` is write-only: never split, sampled, or used to pick thresholds/features/models. The only touch is a single final inference pass once choices are frozen on val, followed by submission-qa's format gate.
- validation-guardian owns: the split implementation (`src/folds.py` or equivalent), the connected-component + near-dup union-find grouping logic (consuming data-detective's near-dup clusters), the official metric implementation once confirmed, and CV-vs-LB gap tracking.

### Open questions (blocking finalization)

1. **[problem-analyst / human]** Exact scoring metric and submission format are NOT yet confirmed from the official rules — do not assume. This also determines whether multi-match (comma-separated matched_entity_ids) is scored as a set (F1/IoU per anchor) or something else, which affects top-k/threshold tuning logic.
2. **[data-detective]** Can an `S2-*`/`S3-*` id appear in more than one source1 row's matched list? (needed to know how non-trivial the match-graph components actually are.)
3. **[data-detective]** Near-duplicate audit on source1 `business_name` + `business_address` (and ideally source2/source3 too): what fraction of source1 rows are near-duplicates of another source1 row, and do near-duplicate pairs share overlapping matched ids? This directly determines whether naive splitting would leak, and is required before the split is finalized. Flagging explicitly per this task's instructions — do not assume it's absent.
4. **[data-detective]** Country and match-status distribution in source1, to confirm what to stratify on and whether country is skewed enough to matter.
5. **[validation-guardian]** Confirm whether matched_entity_ids ever mixes S2 and S3 ids in the same row (implied by "S2-*/S3-* comma-separated list") — if so, per-source recall/precision may need separate tracking in addition to the combined metric.

### ADR-001 UPDATE (validation-guardian, 2026-09-25) — supersedes Decision 1 and part of Decision 3
- **Connected-component question resolved.** EDA: GT match graph = disjoint stars (each S2/S3 id has exactly one source1 anchor). Split unit is therefore the source1 anchor id; union-find on labels is unnecessary. Do NOT group by normalized name (chains form a giant component).
- **Split chosen:** single holdout, 5% of anchors, stratified by has_match x country, seed 42 (`src/folds.py` -> `data/interim/split.parquet`). Train 2,096,479 / val 110,342 anchors (val singletons 6,163). Checks pass: stratum proportions identical, no S2/S3 id on both sides.
- **Metric (UNCONFIRMED):** per-anchor F_0.5 on id sets, singleton correct = 1.0, macro mean over anchors (`src/metric.py`, `averaging` switchable to micro). Must be reconciled with PROBLEM_BRIEF.md once official formula is known.
- Learned blockers/thresholds must still be fit on train-side anchors only (Decision 2 rule unchanged).
- **Metric CONFIRMED (validation-guardian, 2026-09-25):** official PDF (docs/PROBLEM_BRIEF.md sec. 4) = per-source1-entity F_0.5, macro over ALL entities, singleton correct-empty 1.0 / any prediction 0.0. `src/metric.py` matches (macro default; micro is diagnostic only); 7 hand-computed cases + PDF 0.714 example in tests/test_metric.py. Official validator rejects duplicate ids in a list; `strict=True` counts them as FPs. Supersedes the UNCONFIRMED bullet above.


---

## ADR-002: Candidate generation (blocking) architecture   (2026-09-25 IST, ml-architect)
Status: STOP RULE MET on the TRUE full train pool (ml-engineer, 2026-09-25; see 'Full-pool dense blocking results' at the end of this file): dense e5-small top-50 alone gives pair R 95.2 / non-Latin 87.9 / cluster-complete 85.2; dense+expansion+lexical at 135 mean cands gives 97.7 / 92.9 / 92.6. Original status: PROPOSED. Evidence: docs/EDA_FINDINGS.md #4/#5 plus first measured baselines from `src/blocking/harness.py` (3,000 seeded matched anchors, seed 1, full S2+S3 train pool, no folds.py yet).

Context: blocking recall is the ceiling. Country filter is free (100% agree; US/India only). Only 42.5% of true pairs have equal normalized names; 15% share no token; non-Latin S2/S3 names are 9.4%/5.3% (India only). Pool is SHARED: every S2/S3 id belongs to at most one anchor, so pool-side work (normalization, indexes, embeddings) is done once per pool, never per anchor. Machine: 24 cores, 15 GB RAM, RTX 4060 8 GB, no faiss/torch installed yet (verify).

Measured baselines (pair recall / any-hit-per-anchor / cluster-complete; candidates per anchor mean / p95):
- prefix4+country: 79.2 / 95.8 / 52.6; 7,588 / 24,327 (query-weighted; the EDA median 166 was per unique key, hides that queries land in big blocks)
- token (df cap 5000): 57.9 / 65.1 / 42.3; 1,073 / 4,376
- addr-number + name[:2]: 62.3 / 87.0 / 30.0; 364 / 2,195
- union of the three: 86.5 / 96.6 / 70.5; 8,777 / 24,662
- Non-Latin target names: ~3% pair recall in every lexical blocker (7.3% of pairs; India pair recall 74% vs US 94%). Lexical blocking is dead for them; needs dense/transliteration.

Options: (A) lexical only (prefix/token/addr), cheap, cap ~87%, huge candidate sets. (B) A + char n-gram TF-IDF sparse kNN (fixes typos, glued/shuffled tokens; not scripts). (C) B + multilingual dense ANN (fixes scripts, semantic name variants). (D) fine-tuned bi-encoder on train pairs (best recall per candidate, but 6+ h and more risk; only if C stalls).

Decision: C, staged, all per-country (US, India separate indexes), each blocker returning top-k rather than whole blocks:
1. Country filter (hard).
2. Char 3-4gram TF-IDF (sublinear, on normalized name, plus a second view on name+address-number tokens) with sparse cosine top-k (k=50) per anchor. Prune n-grams with df above 0.5%. Implement with chunked sparse matmul or sparse_dot_topn.
3. Dense ANN: `intfloat/multilingual-e5-small` (MIT, 118M) first, base (MIT, 278M) only if small misses; input "name | first address tokens". Alternatives to verify by model-scout: BAAI/bge-m3 (MIT, 568M), Qwen3-Embedding-0.6B (Apache-2.0). All are far below 8B; the 2025 rule was MIT/Apache-2.0 and <=8B: VERIFY 2026 rules with problem-analyst. HNSW/IVF-PQ on GPU or CPU (faiss), top-k=50.
4. Address key: (country, number token, name[:2]) block, kept as a cheap high-precision fallback (recovers 83% of pairs by number). Keep only as rank-limited candidates (cap block size).
5. Cluster expansion (important, since 80% of anchors have 3+ targets that are noisy copies of each other): from any seed candidate found for an anchor, add its pool-side nearest neighbors (pool-to-pool kNN within country, same TF-IDF/dense index). Any-hit recall is already 96.6% for lexical union while cluster-complete is 70.5%, so seeding plus expansion is cheaper than raising per-record recall. Pool-side kNN graph is built once and reused for all anchors (shared pool).
6. Union, dedupe, cap at K candidates per anchor ranked by max blocker score (target K<=150).

Wall-clock/memory estimates for ~12M records (train S1 2.2M + pool 10.3M; test similar), single machine (rough, verify by 100k benchmark):
- normalize + cache pool (measured: <1 min pandas/arrow per 10M for the harness prep; full 12M ~2 min), 3 GB.
- TF-IDF fit+transform: 15-25 min; matrix ~ 10M x ~45 nnz x 8 B = ~4 GB, so do per country (US 6.2M, India 4.1M), 24 cores. Sparse top-k for 2.2M anchors x 6M pool: the long pole, est. 2-4 h with df pruning and chunking; run on a 100k-anchor sample for dev, full only at final.
- Embeddings: e5-small fp16 on the 4060, short strings ~3-5k/s, so 12M in ~1 h (base ~2-3 h). Store fp16 384-d: 9 GB total, so keep on disk per country/source as npy memmap. HNSW in RAM for one country at a time (US 6.2M x 384 x 2 B = 4.8 GB plus graph ~2 GB): tight in 15 GB; use IVF-PQ or int8 if OOM. If RAM is the wall, AWS box (compute-ops).
- Candidate set for pairwise scoring: 2.2M anchors x <=150 = <=330M pairs; feature computation must be vectorized/C-level (rapidfuzz cdist or GPU), budget 2-4 h. For training use a 200k-anchor subsample only.

Decision rule / kill criteria (measured on >=5k val anchors, folds.py split, Tier-2 full pool):
- STOP adding blockers and move on when union pair recall >= 95% at <=150 candidates/anchor (mean) AND non-Latin pair recall >= 85% AND cluster-complete >= 85% after expansion.
- Accept fallback (move on anyway) if the last added blocker gains <1 pt pair recall, or by H+18 whichever first; then record the ceiling in ADR and compensate in the matcher. F_0.5 is precision-weighted, so recall between 92 and 95% costs less than unstable precision.
- If TF-IDF+embedding union stalls below 92%: escalate to (D) fine-tuned bi-encoder (train-fold pairs only, per ADR-001 leakage rule).
- Drop the embedding blocker if e5-small adds <2 pts over TF-IDF+addr union (then non-Latin needs a transliteration-only path, e.g. indic transliteration to Latin, MIT/Apache libs only).

Consequences: commits ~4-6 h of GPU/CPU indexing before the matcher is trainable at scale; we develop the matcher on a 100-200k-anchor subset with candidates from the same blockers, so it does not wait. Risks: RAM (15 GB), sparse top-k cost, k too large hurts matcher throughput, non-Latin India names are the likely recall floor.

---
## Model-scout: licence/size verification (2026-09-25, model-scout)
Source: HF API (`huggingface.co/api/models/<id>`, cardData license + safetensors total). No benchmark run (torch not installed; deferred).
| rank | model | licence (card) | params | notes |
|---|---|---|---|---|
| 1 | intfloat/multilingual-e5-small | mit | 117.7M | ADR-002 pick confirmed; needs "query:"/"passage:" prefixes |
| 2 | intfloat/multilingual-e5-base | mit | 278.0M | fallback if small misses |
| 3 | BAAI/bge-m3 | mit | ~568M (no safetensors count in API; from memory) | strongest multilingual, 1024-d, heavy for 12M rows |
| 4 | paraphrase-multilingual-MiniLM-L12-v2 | apache-2.0 | 117.7M | fast, weaker on names |
| 5 | Qwen3-Embedding-0.6B | apache-2.0 | 595.8M | 1024-d (MRL), slow |
Others OK: LaBSE apache-2.0 470.9M; gte-multilingual-base apache-2.0 305M (remote code); snowflake-arctic-embed-m-v2.0 apache-2.0 305M (remote code); nomic-embed-text-v2-moe apache-2.0 475M (remote code); e5-large mit 559.9M.
Pairwise/fine-tune: xlm-roberta-base mit 278.9M; mdeberta-v3-base mit (~280M, from memory); Multilingual-MiniLM-L12-H384 mit; mmBERT-base mit (~307M, from memory; ModernBERT-style, 1800+ langs, best modern pick for French/Hindi); ModernBERT-base apache-2.0 149.7M but English-only (skip); cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 apache-2.0 117.6M; bge-reranker-v2-m3 apache-2.0 567.8M (too slow for 150 cands x 2M anchors); IndicBERTv2 mit, muril apache-2.0.
AVOID: jina-embeddings-v3 (cc-by-nc-4.0), embeddinggemma-300m (Gemma terms), any Llama-licence model.
Licence question: "final model" plausibly covers the matcher, but a blocker embedder is part of the shipped pipeline, so treat as counting (all shortlisted are MIT/Apache so it is moot). Ask organizers: "Does the MIT/Apache-2.0 and <=8B requirement apply to every pretrained model in our pipeline (e.g. an embedding model used only for candidate generation), or only to the final matching model? All ours are MIT/Apache-2.0 and <1B."

---
## ml-engineer: dense embedding blocker benchmark (2026-09-25)
Env: `.venv` (git-ignored, Python 3.12, uv), torch 2.14.0+cu126 (CUDA visible, RTX 4060 8GB), sentence-transformers 6.1.0, faiss-cpu 1.15.1; pins in `requirements.txt`. Code: `EmbANN` in `src/blocking/blockers.py`, `norm.embed_text`, bench `python -m src.blocking.bench_emb`. Input "name | address" normalized (norm.py, original script, no transliteration), pool "passage: ", anchors "query: ", fp16, max_len 64, bs 512, length-sorted, exact GPU top-50 per country (countries derived from data).
Setup: 2,000 random matched VAL anchors (US 1217 / India 783; 7,332 true pairs), same-country sub-pool from S2+S3 = all true targets + random distractor sample. RECALL IS OPTIMISTIC (fewer distractors than the 10.3M full pool): frac 4% -> pool 419k; frac 25% -> pool 2.59M (25.1% of pool). Lexical baselines below are on the FULL pool, top-50 vs unbounded blocks (mean 8,777 cands), not like-for-like.
| run | pair R@50 | any-hit | cluster-complete | US pair | India pair | India non-Latin pair | Latin pair |
|---|---|---|---|---|---|---|---|
| e5-small, 4% pool | 98.0 | 99.7 | 93.3 | 99.2 | 96.3 | 96.6 | 98.1 |
| e5-base, 4% pool | 98.3 | 99.9 | 94.3 | 99.2 | 97.1 | 97.6 | 98.4 |
| e5-small, 25% pool | 96.7 | 99.6 | 89.5 | 98.3 | 94.2 | 91.7 | 97.1 |
| ADR-002 lexical union (full pool, ~8.8k cands) | 86.5 | 96.6 | 70.5 | ~94 | ~74 | ~3 | - |
Recall falls ~1.3 pt from 4% to 25% pool for small, so full-pool R@50 is likely ~95-96% (extrapolation, not measured); non-Latin decays fastest. Non-Latin is solved by dense (lexical ~3%).
Throughput (encode, RTX 4060 fp16): small 8.7-9.1k rec/s, peak VRAM 0.55 GB; base 3.4k rec/s, 1.2 GB. Full 10.3M pool: small ~20 min, base ~50 min (12M: 23 / 58 min). Query encode ~3.5-9k/s. Search of 2k anchors vs 2.6M pool: 1.4 s exact; full-pool exact per country feasible on GPU by streaming chunks. Storage fp16: 384-d 7.9 GB, 768-d 15.9 GB for 10.3M (disk memmap; RAM 15 GB is the constraint, bench peak RSS 6.8-10.8 GB mostly the raw pool frames).
Recommendation: e5-small as the default dense blocker (meets the ADR-002 keep criterion by a wide margin: +10 pts pair recall, +19 cluster over lexical); e5-base gains only +0.3-0.9 pt at 2.6x cost, so run it only if final full-pool recall stalls below the 95% kill-criterion. Ran no TF-IDF blocker; union with lexical/addr keys and pool-side cluster expansion still untested.


---
## ml-engineer: FULL-pool dense blocking results (2026-09-25)
Code: `src/blocking/full_pool.py` (new; EmbANN/bench_emb untouched). Stages: `encode` (resumable per 100k chunk), `search`, `lex`, `report`; log `logs/full_pool_report.log`. Artifacts in git-ignored `data/interim/full_pool/` (7.5 GB: per-country fp16 memmaps + pos ids + val_dense/val_lex npz + val_grid.csv).
Setup: multilingual-e5-small fp16 (max_len 64; only 0.01% of texts reach 64 tokens, 0.1% of non-Latin), pool = all 10,320,219 S2+S3 train records (US 6.19M, India 4.13M, countries derived from data), exact inner-product top-100 streamed from disk on GPU (fp32 scores, 200k-row blocks) against the FULL same-country pool. Anchors: 5,000 seeded VAL anchors (is_val), 4,721 matched + 279 singletons (5.6%, proportional), 17,184 true pairs. Encode 20.1 min (8.5k rec/s), search 34 s.
| config | mean cands | pair R | any-hit | cluster-complete | non-Latin R | India R | US R |
|---|---|---|---|---|---|---|---|
| dense@20 | 20 | 93.95 | 99.41 | 82.0 | 83.0 | 89.8 | 96.8 |
| dense@50 (TRUE) | 50 | 95.16 | 99.43 | 85.2 | 87.9 | 91.7 | 97.6 |
| dense@100 | 100 | 95.98 | 99.47 | 87.4 | 90.9 | 92.9 | 98.1 |
| dense50 + 20/seed-3 pool-expansion | 59 | 95.90 | 99.45 | 87.2 | 91.5 | 93.1 | 97.8 |
| dense50 + top-100 lex(all four) by dense score | 141 | 97.18 | 99.68 | 91.2 | 89.3 | 95.3 | 98.5 |
| D50+E20+L40 (vote-ranked) | 97 | 97.38 | 99.62 | 91.7 | 91.9 | 95.9 | 98.4 |
| D70+E20+L60 (vote-ranked), best <=150 | 135 | 97.70 | 99.66 | 92.6 | 93.0 | 96.3 | 98.7 |
Breakdown dense@50: S2 95.9 / S3 94.5 pair R; US Latin 97.6; India Latin 92.5; India non-Latin 87.9 (any-hit 93.3). Best config (135 cands): S2 97.8 / S3 97.6; India Latin 97.0, non-Latin 93.0; India cluster-complete 88.7, US 95.3.
Config notation: D=dense top-N; E=pool-side expansion (top-3 dense seeds, each seed's 10 nearest pool records, best 20 by anchor score, not already in D); L=lexical candidates not already in D/E, top-M by (number of lexical blockers proposing it, then dense score). Lexical blockers: AddrNum(cap 3000, name[:2]) 358 pairs/anchor, AddrNum(name-free) 498, Prefix4(cap 2000) 192, Token(df cap 300) 52 (all fit on the full pool). Lexical candidates ranked by dense score only add little (+0.7 to +2.0 pt); vote ranking is better. Every added blocker gives >1 pt at <=+85 cands, so keep the union.
Findings: (1) the optimistic 25%-sub-pool number (96.7) was ~1.5 pt high; true dense@50 is 95.2 as extrapolated. (2) Expansion is cheap (+9 cands) and mainly helps non-Latin and cluster-complete. (3) Lexical prefix/token add the most beyond dense (typo/word-order names dense misses), addr keys mostly help India. (4) Caveat: the grid over (D,E,L) was chosen on the same 5k anchors, so expect ~0.1-0.3 pt optimism; re-check on a second disjoint val sample before freezing. Cluster-complete is measured on the candidate set, before any matcher.
Singletons (279): candidate sets are full-size (50 dense; 141 in the best union), i.e. blocking never abstains, so the matcher MUST abstain on them. Dense top-1 cosine separates only weakly (median 0.935 singleton vs 0.952 matched; a threshold keeping 90% of matched anchors abstains on just 43% of singletons), so abstention needs the pair-level matcher score, not top-1 cosine.
Stop rule (pair R >=95% at <=150 mean cands, non-Latin >=85%, cluster-complete >=85%): MET, by dense@50 alone (95.2/87.9/85.2, marginally) and comfortably by the union at 135 cands (97.7/93.0/92.6). ADR-002 stage-2 TF-IDF blocker is not needed; e5-base is not needed. Recommendation: freeze D70+E20+L60[votes] (or D50+E20+L40 at 97 cands to leave matcher headroom), and only spend time on the matcher.
Resources: RAM peak 8.0 GB (encode), 11 GB (report); GPU <=3.5 GB; disk 7.5 GB. Total wall-clock ~35 min (encode 20 + search 0.6 + lex 1 + report 3 + dev).


---
## ADR-003: Pairwise matcher + decision layer   (2026-09-25 IST, ml-architect)
Status: PROPOSED. Evidence: ADR-002 + full-pool blocking (D70+E20+L60[votes], 135 cands/anchor, pair R 97.7, any-hit 99.7, cluster-complete 92.6, India non-Latin 93.0; singletons get full 141-cand sets); EDA findings 2,3,5,6; metric = per-anchor F0.5 macro incl. singletons.

### Pipeline
```
anchor --e5-small--> D70 + pool-expansion E20 + lexical votes L60 (<=150)   [frozen, ADR-002; = candidate_pairs.tsv]
  -> exact-dup collapse of pool records (name,addr) [score one rep, expand copies later]
  -> S1 pair GBDT (LightGBM, ~50 label-free similarity feats)  -> p1
  -> S2 anchor-context GBDT (p1, rank, gap, sum/max p, #cands>.5, source, competition feats) -> p2 (calibrated)
  -> [phase 3, band only] cross-encoder score as extra S1/S2 feature
  -> decision: disjoint-stars conflict resolution -> per-anchor expected-F0.5 top-k OR empty -> copy exact dups -> matching_results.tsv
```

### Q1 scorer stack. Options
A. LightGBM on hand features: gain = the whole first version; ~1 min to compute features per 1M pairs on 22 procs (rapidfuzz, cdist-style, chunked), train 5M pairs ~5-8 min. Risk low. Reversible.
B. Fine-tuned cross-encoder (Multilingual-MiniLM-L12-H384 MIT 118M, else xlm-roberta-base MIT / mmBERT-base MIT). Handles non-Latin, typos, shuffled addresses better; but 234M test pairs at ~6-8k pairs/s (MiniLM) = 8+ h, base ~3k/s = 20+ h: impossible on all pairs, so ONLY usable on a band (stage-1 p in ~[0.05,0.95], est. 6 pairs/anchor = ~10M pairs, ~30 min MiniLM). Train 600k hard pairs ~10-30 min/epoch.
Decision: A first (submission 0/1), B later as a stacked FEATURE on the uncertain band, never as the only scorer.
Features (all language/country-agnostic, no country one-hot): name: normalized exact, token Jaccard, char-3gram Jaccard, rapidfuzz ratio/partial/token_set/token_sort, Jaro-Winkler, Levenshtein norm, TF-IDF (char 3-gram and token, fitted on S1+pool names, label-free) cosine, suffix-stripped variants, first-token/initial equal, length ratio, name-frequency (chain penalty: log count of that normalized name in S1 and in pool), is-empty/URL flags; address: order-free token bag Jaccard, numeric-token set overlap (leading number equal, any number equal, zip/PIN-like 5-6 digit equal), state/city token overlap, empty-address flags per side, char-3gram cos, casing/script flags for both sides (latin/nonlatin, ALLCAPS); dense: e5 cosine, rank of pair among the anchor's dense list, blocker vote count, which blockers proposed it, expansion flag; source (S2/S3); candidate-set stats (n cands, top-1/top-2 dense gap, p-rank relative to anchor max). Non-Latin: no transliteration lib; rely on dense cos + flags (feature values fall back to NaN for string sims when scripts differ, so trees do not misread 0 as "dissimilar"; LightGBM handles NaN).
Kill criteria: S1 GBDT val F0.5 (with the dumbest thresholded decision) < 0.75 after 3 h of feature work -> debug features/candidates before anything else. Cross-encoder: continue only if band-feature raises val F0.5 by >= 0.7 pt on 20k val anchors after 1 epoch; if S1+S2 GBDT already >= 0.93 skip it. xlm-r-base/mmBERT only if MiniLM gains >= 1 pt AND band pairs <= 15M.

### Q2 training set
- Anchors: train-split only (is_val false). T1 = 100k anchors (S1 scorer, ~13.5M pairs, ~520k positives, 3.8% pos; first fit on 40k to draw a learning curve, keep growing only if val moves >= 0.3 pt), T2 = 60k disjoint anchors (S2 context model, uses S1 predictions from T1 model, so honest without OOF), T3 = 300k further train anchors only if the competition-density simulation (Q3) is run. Val 110k untouched: 20k val anchors for rapid iteration, full only at checkpoints, val split in halves for tune/confirm.
- Negatives: ALL blocker candidates (no subsampling for T2/val; T1 may downsample easy negatives, dense rank>60 and no name/addr overlap, at rate 0.3 with weights 1/0.3). Same candidate function as test (one code path `candidates(anchor_ids, world)`), so train/inference distribution matches, incl. singletons (their candidates are all negatives, 5.6% of anchors, must be kept).
- Leakage: (1) no feature may use GT (no target encodings of names, no "is matched" flag on pool records); (2) frequency/TF-IDF/idf fit only on unlabeled text (S1 + pool, allowed for test too, transductive but label-free); (3) pool-side expansion and dense scores are label-free; (4) S2 model trained on disjoint anchors from S1 model; (5) thresholds tuned on val only; (6) a train anchor's pool includes its own true records, exactly like test, no artificial removal. Do not touch data/test/.

### Q3 decision layer (F0.5 = 1.25 TP / (k + 0.25 T); k=#predicted, T=|truth|)
Options: (a) global per-pair threshold; (b) per-anchor top-k; (c) plug-in expected-F0.5 optimum per anchor plus abstain.
Math: adding an item with calibrated prob p helps iff p > F/1.25 ~ 0.8*F, i.e. threshold ~0.7 at F~0.9 (not 0.5). Plug-in: for each anchor sort by p2, for k=0..K compute E_k = 1.25*sum_{i<=k}p_i/(k+0.25*T_hat), T_hat = sum_all p_i / recall_ceiling(0.977); empty option scores prod(1-p_i) (P(singleton)); pick argmax. Start with (a)+abstain (fastest, robust): predict set {p2 >= tau}; if max p2 < tau_a predict empty; tune (tau, tau_a) on val by grid on the real macro metric incl. singleton term; then test (c) and keep it only if it beats (a) on the confirm half by >= 0.2 pt.
Disjoint stars: each pool record has exactly one true anchor. After scoring, for each pool record claimed (p2>=tau) by several anchors keep the argmax anchor, drop the others; anchors emptied become empty (also lifts singleton term). This is precision-safe by construction and label-free; the tuning risk is only tau. Caveat: val holds 5% of anchors so conflicts are under-represented there; measure gain on a density simulation (val 110k + T3 300k held-out train anchors = ~19% density; ~1 h) or just report fraction of claims resolved. Add soft variant p2*=lambda for losers, lambda tuned. Per-source: S2 and S3 use one tau but source is a feature; report P/R per source and adopt separate tau only if val gain >= 0.2 pt. Exact dups: score one representative per (name,addr), copy all copies of every predicted id (100% of dup rows are matched in EDA).
Thresholds by country/script: single global tau. Per-country tuning adds parameters for < 0.2 pt and leaves France without a value. Use tau_FR = tau + 0.08 (France = any country not seen in train, determined from data: a country whose name is not in the train country set) as a conservative default; size the offset by a leave-country-out proxy (train on US anchors, tune on India and vice versa; measure the optimal-tau shift and F drop; set offset = max observed shift, capped at +0.10).
Back-of-envelope val F0.5: matched anchors (94.4%): P~0.94, R~0.88 pooled but per-anchor variance and 2.3% recall loss give ~0.88-0.91; singletons: abstain rate 60-80% -> 0.034-0.045 total. GBDT ~0.85 (range 0.80-0.89); + context, assignment, cross-encoder ~0.88 (range 0.85-0.91). Ceiling from blocking alone ~0.97. Estimates, not measurements.

### Q4/Q5 plan and costs: see docs/STATUS.md phase table. Test inference (1.73M anchors, 10.0M pool; estimates, verify with a 50k-anchor benchmark): encode pool 22 min + anchors 4 min; same-country exact GPU search with qchunk >= 8k (the 5k-anchor bench used qchunk 1500, do NOT reuse it: 1.73M/1500 passes over 8 GB) ~1-2 h; lexical blockers at 1.73M anchors are the RAM/time risk (1,070 raw pairs/anchor = 1.85B raw pairs; process in 50k-anchor chunks, vote-count and keep top-60 immediately) ~1-2 h; features 234M pairs: two-stage (cheap features on all 135, expensive rapidfuzz/TF-IDF only for top-40 by p_cheap) ~1.5-3 h; GBDT predict 2-4 min per 10M pairs -> ~1 h with context; cross-encoder band ~0.5 h. Total ~6-9 h wall clock for a full pass; stream in 50k-anchor chunks, write per-chunk parquet, resumable; peak RAM < 10 GB (memmap the 8 GB train / 7.7 GB test embeddings; gather pool rows sorted by index); disk: ~8 GB train emb + ~8 GB test emb + ~4 GB candidates/scores; compute-ops to check free disk before the test encode.
Compliance: e5-small MIT, MiniLM/xlm-r MIT, LightGBM MIT, rapidfuzz MIT; no external data/lookups; hand-written suffix/state maps only.

### Decision
Two-stage LightGBM (pair features, then anchor-context) on the frozen ADR-002 candidate set, F0.5-tuned threshold + abstain + disjoint-stars conflict resolution + dup expansion as submission 0/1; cross-encoder on the uncertain band as a gated upgrade; one global threshold with a conservative France offset.
Consequences: commits ~1 day of feature code; gives up end-to-end learned retrieval and any bi-encoder fine-tune; if the blocker recall at test scale drops (larger test pool ratio) the ceiling drops, so recompute recall only on val, never on test.
Biggest risks: (1) France: unvalidatable, calibration/format shift (accents, 5-digit postal codes, different chain names); (2) test-scale pipeline failure (RAM in lexical blockers, run time) discovered late, so rehearse on a 50k-anchor chunk end-to-end before the full run; (3) chain-name false merges at full density; (4) threshold tuned at 5% anchor density but applied at 100%.
Open questions for the human: submission count/day and tie-break rule; is MIT/Apache scope limited to the final model or every pretrained component (all ours qualify, ask anyway); free disk space; whether GPU windows are reserved for us; France share of test if the organizers say (do not infer it from test data).


---
## ml-engineer: blocking holdout re-check on a 2nd disjoint val sample (2026-09-25)
Code `src/blocking/recheck.py` (gen, eval); log `logs/recheck_eval.log`; misses `logs/blocking_misses.tsv` (30 stratified misses, sample2, D70+E20+L60). Sample 2: 10,001 is_val anchors, excluding the 5,000 sample-1 ids (set difference on source1 id), proportional by country x has_match (US 5665 matched/337 singleton, India 3777/222; seed 23). Configs frozen, no re-tuning. Cluster bootstrap over anchors (400 reps) for 95% CIs.
| config | sample | cands mean/max | pair R [95% CI] | cluster-complete | any-hit |
|---|---|---|---|---|---|
| D70+E20+L60 | 1 (tuned) | 135/150 | 97.70 [97.46,97.97] | 92.61 | 99.66 |
| D70+E20+L60 | 2 (holdout) | 135/150 | 97.55 [97.37,97.73] | 92.09 [91.57,92.67] | 99.75 |
| D50+E20+L40 | 1 | 97.5/110 | 97.38 | 91.72 | 99.62 |
| D50+E20+L40 | 2 | 97.5/110 | 97.34 [97.17,97.52] | 91.48 | 99.72 |
Real optimism: ~0.15 pt pair R (D70) and ~0.04 pt (D50), within noise; the gap between the two configs is 0.2 pt pair / 0.6 pt cluster. Sample 2 by group (D70): US 98.65, India 95.87 (Latin 96.71, non-Latin 92.12; non-Latin any-hit 94.6), S2 97.74 / S3 97.36; matches per anchor 1 / 2-3 / 4+: 97.7 / 97.6 / 97.5 pair R but cluster-complete 97.7 / 94.4 / 89.4 (large clusters rarely complete); anchor name exact-equals target 99.0 vs different name 96.5 (misses concentrate on noisy targets: dropped suffix, honorific prefixes "Smt/Sri/Dr", typos, digit-for-letter, non-Latin transliterations with a translated word). Singleton candidate sets: mean 141.7 (D70) / 104.0 (D50), min 71 / 52: blocking never abstains.
Recommendation: FREEZE D70+E20+L60[votes] (135 cands; <=150 cap; recall stable on holdout). Fallback if candidate volume or matcher throughput bites: D50+E20+L40 (97.5 cands, -0.2 pt pair R, -0.6 pt cluster). Blocking is done; the remaining ~2.5% pair loss is concentrated in India non-Latin and noisy-name pairs, best addressed by the matcher's own retrieval-free rescoring rather than more blockers. France is untested (no train), so the test-time candidate cap and dense-only fallback for unknown scripts stay as is.

## ADR-004: Preprocessing plan = PREPROCESSING_PLAYBOOK.md reconciled with the frozen blocking (lead, 2026-09-25, ~55h left, ACCEPTED by user go-ahead)
Context: the playbook (repo root) specifies Phases 0-10. Its cited evidence file docs/AMAZON_ML_2026_DATA_DETECTIVE_REPORT.md is NOT in the repo, so its numbers are unverified here; we adopt its rules where they are cheap and consistent with our measurements, and keep our measured results where they conflict.
Decisions:
1. ADOPT Phase 3 representation build (non-destructive): NFKC, strip only Unicode P*/S*/Cc/Cf, casefold into *_norm, raw fields kept verbatim, script tag (latin|deva|indic-other|mixed), legal suffix EXTRACTED to its own field (never deleted; mined from train per country + French forms SARL/SAS/SA/EURL/SCI from general knowledge), order-free name/addr token sets, rule-based Indic->Latin transliteration whose tokens are UNIONED into the name/addr token space, addr_pin/addr_numbers/addr_state_tok as extra fields, name_char3grams for PAIR FEATURES ONLY. Runs on train and test alike (per-record, fits nothing).
2. KEEP the frozen blocking (dense e5-small D70 + expansion E20 + lexical L60, <=150 cands; val pair recall 97.55%/cluster 92.09% on a holdout, India 95.9, US 98.7). The playbook's token-IDF top-50 retrieval reports India 90.8 / US 96.8 (unverified), which is below ours, so Phases 7-8 are NOT adopted as a replacement. The new tokens are tested as an upgrade of the lexical L channel and non-Latin pair features; replace only if measured better on val.
3. KEEP src/folds.py split (5%, seed 42, stratified has_match x country); re-splitting would invalidate every val number. Playbook D11/D14/D18 are consistent with it.
4. Phase 5 (pad val pool to 1:5.75 S1:S2+S3) is NOT reproducible: it would need distractors beyond the train pool (would mean using test records). Our val already searches the full train pool (1:4.68), so there is no thin-slice optimism. The 4.68 vs 5.75 gap becomes (a) a precision-biased threshold margin and (b) a density-sensitivity check in the matcher phase.
5. Forbidden list adopted verbatim (no [^\w\s] stripping, no global unidecode, no state/street expansion in retrieval tokens, no destructive suffix removal, no char-ngram score summation in retrieval, no Soundex/phonetic for Indic names, do not drop empty-address rows, no singleton classifier on S1 features, no country one-hot, PIN only as a verification feature, no external lookups). Our earlier idea of phonetic keys for Indic variants is DROPPED.
6. Phase 9 feature list is merged into ADR-003 features: asymmetric token containment, raw-exact and norm-exact name match, legal_suffix agreement, addr_pin agreement (verification only), shared address numbers, script-pair indicator, retrieval score/rank.
7. Environment: playbook Phase 0 is mostly done in .venv (numpy/pandas/pyarrow/scipy/sklearn/torch/faiss present); missing polars, rapidfuzz, lightgbm are installed and pinned in requirements.txt.
8. Test data handling: ingestion to parquet, row-count assertions and per-record deterministic transforms ONLY. No profiling, statistics, sampling or tuning on data/test/.
9. Frozen artifacts (data/interim/pool_norm.parquet, data/interim/full_pool/, split.parquet, src/blocking/norm.py) are NOT modified; the new package is src/prep/ with outputs in data/processed/prep/.
10. src/blocking/full_pool.py fixes B1/B2/B4/B5 from the code-review (encode into train dir overwrite, absent-country in dense scoring, slug collisions, tiny countries) are prerequisites for test-time candidate generation and run in parallel with preprocessing.
Schedule (55h left at 2026-09-25): preprocessing + full_pool fixes <=3h; matcher features + LightGBM + val F0.5 by ~T-45h; SUBMISSION 0 full test pass (6-9h) must start no later than ~T-38h; code freeze T-8h; final submission + zip T-2h.

## ml-engineer: ADR-003 Phase 0 results - unified `candidates()` + 50k rehearsal + test launch (2026-09-25 ~19:00 IST)
Code: `src/blocking/candidates.py` (one function `candidates(anchor_df, pool_df, emb_dir, outdir, Q)`; identical for train/val/test anchors; plus `encode_queries`, `write_candidate_pairs_tsv`, CLI stages encode-test | prep-test | rehearse | cands-val | cands-test). Tests `tests/test_candidates.py` (cap<=150, no dup (a,p), no cross-country, absent-country anchors empty, crash-resume identical, tsv format, GPU==CPU within tie noise). full_pool.py NOT changed. Frozen config unchanged (D70+E20+L60[votes], cap 150). Additive design notes: (1) country pool embeddings made GPU-resident (<=5GB) -> pair scoring/expansion on GPU; (2) expansion neighbours computed once per UNIQUE seed row; (3) lexical votes computed per 10k-anchor block, only the vote tiers that can reach the top-60 are cosine-scored (exact, assumes cosine in [0,1) so votes dominate; rehearsal metrics identical to the full-scoring run); (4) deterministic tie-break (smaller pool pos) so resume is bit-identical; (5) Test pool prepped per record with `norm.prep` into `data/interim/test_pool_norm.parquet`; test embeddings in `data/interim/test_pool/` (countries.json: France, India, US; train dir untouched).
Rehearsal (50k proportional val anchors, seed 5, full train pool, train embeddings): pair R 97.53 (frozen 97.5-97.7), cluster-complete 92.08 (92.1), any-hit 99.75, cand mean 134.7 / max 150 (mean 141.3 for singletons). Reproduces within 0.03 pt.
Cost (measured, GPU shared with nothing else, nice'd): 5-7 ms/anchor (dense 60%, lexical 25%, select 10%), peak RSS 7.4 GB (train US pool 6.2M is the worst case; test countries are 4.7M/3.8M/1.4M). Extrapolation to 1.73M test anchors: 2.4-3.4 h. Cheap 2x on the dense part is available (threshold-prefiltered top-k, bench 3.6s vs 7.4s per 10k x 4M) if time is needed.
Gate PASSED -> chained job launched: val candidates (all 110k val anchors -> `data/processed/cands/val/`, log `logs/val_cands.log`) then FULL test candidates (`data/processed/cands/test/`, `candidate_pairs.tsv` written at the end, log `logs/test_cands.log`). Test-time rules honoured: per-record transforms, embedding, search only; France anchors are routed to the France pool (in the test pool) and any anchor whose country had no pool records would get empty candidates with a logged count.
