# STATUS.md — Current State

Updated by whichever agent has fresh information. Append/update your section; do not delete others' entries.

---

## 2026-09-25 — H+0 (problem drop day), pre-brief

**Time remaining:** ~71.5h of the 72h assessment window (25 Sep 00:00 IST → 27 Sep 23:59 IST). No baseline submitted yet.

**Best CV / Best LB:** none yet — no metric implementation, no model, no submission.

**Running jobs:** none.

**Problem type (provisional, from raw schema inspection, NOT yet in a ratified PROBLEM_BRIEF.md):** entity resolution / record linkage across 3 business-listing sources. source1 = 2.2M anchor entities (train) needing to be matched against a shared pool of source2 (5.03M) + source3 (5.29M) candidate records. ~5.6% of source1 anchors have no match anywhere. Multi-match is possible (comma-separated matched_entity_ids). Test sources are same schema, no ground truth, write-only.

**Blockers:**
1. Official scoring metric and submission format are NOT confirmed — problem-analyst has not run yet. Do not build the metric/threshold logic until this lands.
2. Train/val split is not finalized — see `docs/DECISIONS.md` ADR-001. Pending: data-detective's match-graph cardinality check and near-duplicate audit on source1 (and ideally source2/3).

**Next 3 actions:**
1. `problem-analyst` — pull the official problem statement/rules and write `docs/PROBLEM_BRIEF.md` (metric, submission format, any sanity checker, compliance rules).
2. `data-detective` — (a) check whether any S2/S3 id is matched by more than one source1 row, (b) near-dup cluster audit on source1 business_name+address, (c) country/match-status distribution. All three block finalizing ADR-001's split.
3. `validation-guardian` — once (1) and (2) land, implement the split (`src/folds.py`): connected-component + near-dup union-find grouping, stratified ~90/10 or 95/5 holdout, fixed seed. Implement the metric once confirmed.

**In parallel (should already be running per playbook):** `compute-ops` hardware/environment setup; consider whether images are involved this year (schema shown has no image_link column — if confirmed text-only, skip the "start downloads immediately" rule from the shared playbook, but verify with problem-analyst first).

## ADR-001 pointer

See `docs/DECISIONS.md` ADR-001 for the full train/val split strategy and reasoning (competition-strategist, PROPOSED, pending ratification).

## 2026-09-25 - data-detective EDA pointer
See `docs/EDA_FINDINGS.md`. Top findings: (1) match graph = disjoint stars, every S2/S3 id matched by exactly one anchor, so split by anchor id is leak-free; (2) 1-to-many: mean 3.46 matches/anchor, 80.5% mix S2+S3, 5.58% empty, 26% of S2/S3 are pure distractors; (3) chain names common (48.9% of S1 share a name) with disjoint matches, address must discriminate; (4) blocking ceiling: 15% of true pairs share no name token, 42.5% exact name, first-4-char recall 79%, need char-ngram + multilingual embedding + address-number union (country hard-filter is free, 100% agree); (5) noise is in S2/S3 (9.4%/5.3% non-Latin names, ~3.5% empty addresses, ALL-CAPS, typos).
