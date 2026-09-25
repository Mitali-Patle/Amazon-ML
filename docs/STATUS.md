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

## 2026-09-25 - problem-analyst pointer
See `docs/PROBLEM_BRIEF.md` (canonical PDF-derived brief: metric, output rules, rejection conditions, open questions).

## 2026-09-25 - ml-architect: ADR-003 (matcher + decision layer, PROPOSED). Blocking frozen at D70+E20+L60[votes].
T0 = start of matcher work; wall clocks are estimates.
| Ph | Task | Owner | Wall | Gate |
|---|---|---|---|---|
| 0 | Generalize full_pool.py into `candidates(anchor_ids)` used for train/val/test; benchmark 50k-anchor end-to-end (search qchunk>=8k, lexical chunked); start TEST encode (pool ~22 min, anchors ~4 min) in background when GPU is free | ml-engineer, compute-ops | 2h | recall reproduces on val 5k |
| 1 | Candidates for T1 100k/T2 60k train anchors + val 20k; cheap+expensive features; S1 LightGBM; val F0.5 with global tau + abstain | ml-engineer, validation-guardian, code-reviewer | 4h | val F0.5 >= 0.75 |
| 2 | Tune tau/tau_a on val, dup expansion, conflict resolution; S2 context model | ml-engineer, ensemble-optimizer | 3h | +0.2pt to keep each piece |
| 3 | SUBMISSION 0: full test pass (D50 + cheap feats OK if time-critical), matching_results.tsv + candidate_pairs.tsv, validate_submission.py, submission-qa PASS | compute-ops, submission-qa | 6-9h | PASS |
| 4 | Cross-encoder MiniLM on band (train 30 min, val check) ; leave-country-out France offset; density simulation | finetuning-specialist, validation-guardian | 6h | +0.7pt else drop |
| 5 | Submission 1/2 (full stack), freeze T-8h, doc | submission-qa, doc-writer | 8h | |
Kill/parking: see ADR-003. Next 3 actions: (1) ml-engineer generalize candidate builder + 50k rehearsal; (2) compute-ops check disk/GPU windows, schedule test encode; (3) validation-guardian define tuning halves and leave-country-out protocol.

## 2026-09-25 - lead: ~55h left; ADR-004 accepted (preprocessing playbook reconciled). Branch: feat/blocking (user commits themselves).
Running: (1) ml-engineer: playbook Phases 0-3 in src/prep/ (train+test representation build, suffix extraction, transliteration, tests); (2) ml-engineer: full_pool.py fixes B1/B2/B4/B5. Next: code-reviewer on both, then ADR-003 Phase 0/1 (candidate builder, features, LightGBM).

## 2026-09-25 19:05 IST - ml-engineer: ADR-003 Phase 0 DONE (candidate builder + rehearsal), test candidate job launched
Done: test encode (pool 9.94M + 1.73M anchors, `data/interim/test_pool/`, countries France/India/US), `src/blocking/candidates.py`, 58 tests pass. Rehearsal 50k val: pair R 97.53 / cluster 92.08 / mean 134.7 / max 150 (frozen numbers reproduced), peak RSS 7.4GB.
Running (chained, one shell): (1) val candidates 110k -> `data/processed/cands/val/` (~15 min, `logs/val_cands.log`); (2) FULL test candidates -> `data/processed/cands/test/` (est. 2.4-3.4h, `logs/test_cands.log`, resumable: rerun `python -m src.blocking.candidates cands-test`). GPU and ~7GB RAM are busy until it finishes: run matcher feature work on CPU / small RAM, no big GPU jobs meanwhile.
Next: matcher Phase 1 features on val/train candidates (`candidates()` works for any anchor set; for train anchors encode queries with `encode_queries` into a NEW dir).
