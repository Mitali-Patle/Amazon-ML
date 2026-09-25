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


---

## ADR-002: Candidate generation (blocking) architecture   (2026-09-25 IST, ml-architect)
Status: PROPOSED. Evidence: docs/EDA_FINDINGS.md #4/#5 plus first measured baselines from `src/blocking/harness.py` (3,000 seeded matched anchors, seed 1, full S2+S3 train pool, no folds.py yet).

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
