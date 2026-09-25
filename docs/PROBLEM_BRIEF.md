# PROBLEM_BRIEF.md - Amazon ML Challenge 2026: Business Entity Resolution

Author: problem-analyst, 2026-09-25. Source of truth: `resources/6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf` (8 pages; pages 1-7 read in full, page 8 blank).
Legend: **[PDF]** = stated in the statement. **[INF]** = my inference, not stated. **[EDA]** = measured on train by data-detective. **[VAL]** = read from `utils/validate_submission.py`.

Files: `utils/validate_submission.py` is present in the repo (read; stdlib validator). `Documentation_template.md` is NOT present (only the PDF is in `resources/`); the PDF says it is "provided", so it must be fetched from the Unstop portal (Q9). The PDF also links an explanation video (not viewable here).

## 1. Task in one sentence
[PDF] Given business records from 3 independent sources (noisy names/addresses, no shared identifiers), for EVERY Source 1 (S1) test entity output all matching Source 2 (S2) and/or Source 3 (S3) `entity_id`s (zero, one, or many). S1 is the deduplicated reference.

## 2. Data schema and files
[PDF] All files are tab-separated `.tsv`; submissions must be tab-separated too (addresses and id lists contain commas). Read with `pd.read_csv(path, sep="\t")`; without `sep` pandas silently yields one column.
[EDA] Safer read: `quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False`.

Each `*_source{1,2,3}.tsv` [PDF]: `entity_id`, `business_name`, `business_address`, `country`.
- `entity_id`: unique; prefix `S1-`/`S2-`/`S3-` gives the source (no separate source column).
- `business_name`: abbreviations, legal suffixes, typos, transliterations.
- `business_address`: partial addresses, format variations, missing components, landmark references.
- `country`: train = `US`, `India`; test additionally `France`. [PDF] "Treat `country` as an open set of string labels: do not hard-code, filter, or one-hot your pipeline to only {US, India}, and remember that every test entity - France included - must appear in your submission."

| File (PDF paths; repo uses `data/train`, `data/test`) | Content |
|---|---|
| train/train_source1.tsv, train_source2.tsv, train_source3.tsv | records per source |
| train/train_ground_truth.tsv | `source1_entity_id`, `matched_entity_ids` (comma-separated S2-/S3- ids, empty if no match) |
| test/test_source1.tsv | S1 test records; "Generate matches for every entity in this file" |
| test/test_source2.tsv, test_source3.tsv | S2 / S3 test records |

No test ground truth; validate on a train holdout. [EDA] Train rows: S1 2,206,822; S2 5,034,617; S3 5,285,604; 123,247 GT rows empty (5.58%). Test row counts and France share are not stated in the PDF (the validator docstring mentions a "full ~1.7M-entity test set" [VAL], unverified).

## 3. Noise patterns listed [PDF]
- Names: abbreviations (Corp/Corporation, Pvt/Private, Ltd/Limited), legal-suffix inconsistencies, DBA/trade names, punctuation (& vs "and"), word-order transpositions, typos.
- Addresses: abbreviations (Rd/Road, St/Street), transliteration variants, missing components (no PIN, no state), landmark references (Near SBI ATM), municipal numbering formats, component reordering.
Empirical extras (Devanagari names, ALL-CAPS, URL-like names, empty addresses): EDA_FINDINGS.md finding 5.

## 4. Target and metric
Target [PDF]: per S1 entity a SET of S2/S3 ids (variable cardinality, may be empty). [EDA] mean 3.46, max 11.

Metric [PDF]: F_beta with beta = 0.5, `F_0.5 = 1.25*P*R / (0.25*P + R)`, computed per S1 entity, then macro-averaged over ALL S1 entities in the evaluation set. Singletons included: empty prediction = 1.0, any predicted match = 0.0. Higher is better; precision weighted 2x over recall (false merges cost more than misses).
PDF worked example: predict [S2-00047, S2-00193, S3-00812], truth [S2-00047, S3-00812] -> P = 2/3, R = 1.0, F_0.5 = 0.714.

Not stated [INF, safest reading]: non-empty truth with empty prediction scores 0.0 (P undefined, R = 0); P/R are set-based over the union of S2 and S3 ids (no per-source split); the public/private split is applied over entities.

```python
def f05_entity(pred: set, truth: set) -> float:
    if not truth:
        return 1.0 if not pred else 0.0          # [PDF] singleton rule
    if not pred:
        return 0.0                               # [INF]
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)

def score(pred: dict, truth: dict) -> float:     # macro over ALL S1 ids in truth
    return sum(f05_entity(pred.get(k, set()), t) for k, t in truth.items()) / len(truth)
```
Hand-computed tests (put in `src/metric.py` tests):
| pred | truth | P | R | F_0.5 |
|---|---|---|---|---|
| {a,b,c} | {a,b} | 2/3 | 1 | 0.714 (PDF example) |
| {a} | {a,b} | 1 | 0.5 | 0.625/0.75 = 0.833 |
| {a,b,c,d} | {a,b} | 0.5 | 1 | 0.625/1.125 = 0.556 |
| {} | {} | - | - | 1.0 |
| {a} | {} | - | - | 0.0 |
| {} | {a} | - | - | 0.0 [INF] |
| {a,b} | {c} | 0 | 0 | 0.0 |
Consequences: with P=1,R=.5 the score is 0.833 but P=.5,R=1 gives 0.556, so favour precision. A wrong guess on a singleton (5.6% of train anchors) costs a full 1.0, so build a calibrated abstain option. All-empty submission scores about 0.056 [EDA rate]. ADR-001 said the metric was "UNCONFIRMED": it is now confirmed as macro with singleton 1.0/0.0; validation-guardian should update `src/metric.py` and add the table as tests.

## 5. Output files (both TSV, in `output/`) [PDF]
### matching_results.tsv (the only scored file; upload this to the Portal)
Header exactly `source1_entity_id<TAB>matched_entity_ids`. One row per S1 test entity. Ids comma-separated, no quoting, no spaces. Example: `S1-00001<TAB>S2-00047,S2-00193,S3-00812`; singleton `S1-00003<TAB>` (empty).
### candidate_pairs.tsv (not scored; audit of blocking recall ceiling and reduction ratio)
Header `source1_entity_id<TAB>candidate_entity_ids`. Same rules: one row per S1 entity, empty when blocking found nothing, S2-/S3- ids only, no duplicates within a list. Must be the exact set fed into the matching model at inference (the LAST filtering stage right before the model scores it, not raw early blocking output). Every id in matching_results.tsv should appear here (final matches are a SUBSET of candidates; the validator only warns). No size limit is stated [INF: keep <=150 per entity per ADR-002].

## 6. Rejection conditions (file not evaluated) [PDF Constraints 1-4 + VAL]
1. Fails validation: not tab-separated (validator specifically flags comma-separated files), wrong header (exactly `['source1_entity_id','matched_entity_ids']`, compared after strip/lowercase) [VAL].
2. `matched_entity_ids` contains S1 ids (self-matches), or ids not in the test set [PDF says rejected; VAL says the scorer only lowers the score for nonexistent S2/S3 ids, and self-matches/non-S2/S3 prefixes are errors: treat all as forbidden].
3. Any test S1 entity missing -> rejection.
4. Duplicate ids inside any id list, or duplicate `source1_entity_id` rows -> rejection.
Additional [VAL] errors: a row for an S1 id not in test_source1; a row with no tab (malformed); empty file; missing file.
A correctly formatted upload shows status `SCORED` with the F_0.5.

## 7. Validator usage
`utils/validate_submission.py` (stdlib; exit 0 = PASS, 1 = issues; warnings never fail). In this repo:
```
python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir data/test
```
The PDF example uses `dataset/test` (default) relative to `student_resource/`. Only `test_source1.tsv` is read by default (id list); `--check-ids` also loads S2/S3 ids (a few GB RAM; drop `--candidate` if OOM). It does NOT check: score quality, same-country, candidate-set size, or ID existence unless `--check-ids`; subset-of-candidates is a warning only.

## 7b. Final submission package [PDF]
```
<team_name>_submission.zip
  output/matching_results.tsv            (same file as uploaded to leaderboard)
  output/candidate_pairs.tsv
  code/business_entity_resolution/src/             (all source)
  code/business_entity_resolution/README.md        (reproduce: data -> blocking -> matching -> output)
  code/business_entity_resolution/requirements.txt (pinned deps)
  Documentation_template.md                        (filled-in; .md or .pdf; no need to rename)
```
Anyone must regenerate both output files from the provided train/test only. The doc must cover methodology, candidate generation/blocking, model architecture and feature engineering, other relevant info. [PDF] "There is no page limit - prioritise clarity and technical depth", which supersedes the generic 1-2 page note in the shared context. All teams submit; top teams' packages are reviewed in detail before rankings are confirmed.

## 8. Leaderboard mechanics [PDF]
Public LB = subset of test, live. Private LB = remaining part, revealed after the challenge; final ranking = private LB. You always submit predictions for the FULL test set; the split is applied at scoring. Submissions per day/total, best-vs-last selection, tie-break: not in PDF (Q3). Shared context says max score with earlier-time tie-break [INF from 2025, verify].

## 9. Rules and constraints [PDF]
- Constraint 5: "Final model should be a MIT/Apache 2.0 License model and up to 8 Billion parameters."
- STRICTLY PROHIBITED: external databases/APIs/services to look up business identities or resolve entities: commercial ER APIs, government business registries, geocoding APIs to normalize addresses, any external data augmentation from internet sources. Evidence -> immediate disqualification; approaches and code are reviewed. "Using only the provided training data."
- Not stated: compute limits, team size, policy on other pretrained-weight data licences.

## 10. Organizer tips [PDF]
Blocking sets the recall ceiling; Jaccard/Levenshtein/TF-IDF cosine for name+address; country-specific address patterns; F_0.5 favours precision; do not neglect singletons; validate format before submitting.

## 11. Known facts from EDA (train only; see `docs/EDA_FINDINGS.md`)
- Match graph = disjoint stars: each S2/S3 id matched by exactly one anchor, so an anchor-id split is leak-free (ADR-001 update).
- 5.58% singletons; mean 3.46 matches (max 11); 80.5% of anchors match both S2 and S3; 26% of S2/S3 rows are pure distractors.
- Every true target shares the anchor's country (US/India in train): a country filter is free, but derive it from the data (open set; France unseen). This is an EDA fact, not a PDF rule.
- Blocking ceiling: only 42.5% of true pairs have identical normalized names; 15% share no name token; non-Latin names in 9.4% of S2 / 5.3% of S3 (India only).
- Chain names are common (48.9% of S1 share a name) with disjoint match lists; address must discriminate.

## 12. Open questions / ambiguities (ordered by impact)
1. **France in test (biggest risk).** [PDF] France appears only in test and must be in the submission. Unknown: its share of test, address formats, diacritics. Cannot be validated. Judgement: language/script-agnostic features (char n-grams, accent-stripped Unicode normalisation, multilingual embeddings), no country hard-coding, and a conservative match threshold for countries unseen in train (precision-heavy metric). Do not tune on test. Ask forum only for the France share if cheap.
2. **Licence scope.** [PDF] says "Final model" MIT/Apache-2.0 and <=8B. Unclear whether this includes the blocking embedder, tokenizers, or fine-tuned derivatives. Judgement (safest): apply to EVERY pretrained model in the pipeline, including embedders. Candidates: multilingual-e5-small/base (MIT), bge-m3 (MIT), Qwen3-Embedding-0.6B (Apache-2.0). Avoid CC-BY-NC, Llama/Gemma-licensed weights. Record model/licence/size in the doc. A lexical-only pipeline is the zero-risk fallback.
3. **Submissions per day/total and tie-break:** not in PDF. Ask Unstop FAQ; until known treat uploads as scarce and do not probe the public LB.
4. **Deadline/timing:** not in the PDF. Shared context says 25 Sep 00:00 IST to 27 Sep 23:59 IST (24 Sep 18:30 UTC to 27 Sep 18:29 UTC); verify on the portal.
5. **candidate_pairs.tsv size limit:** none stated; "reduction ratio" is audited, so huge sets may hurt doc review (not score). Judgement: <=150 ids per entity (ADR-002), and confirm portal/zip size limits.
6. **Subset rule:** the PDF says "should" (validator warns). Judgement: enforce by construction (matching derived from candidates).
7. **Prohibited lookups vs local resources:** does a bundled gazetteer, offline transliteration lib, or pretrained multilingual model count as "external data augmentation"? Judgement: MIT/Apache pip libraries and model weights used offline are code/models, acceptable; shipping an external data table (city/PIN/registry gazetteers, downloaded name lists) or any network call is unsafe. Small hand-written maps (state abbreviations, legal-suffix lists) are fine and should be documented. Never call an API at run time. Flag borderline items in the doc; ask the forum if using any library that ships data (e.g. libpostal models).
8. **Public/private split composition:** unknown whether France concentrates in one part; do not overfit public LB.
9. **Documentation_template.md:** not in the repo; obtain from the portal and start filling early (doc-writer).
10. **Empty-prediction/non-empty-truth scoring:** inferred as 0.0; confirm nothing is excluded or NaN-handled.
11. **Example ids** (`S1-00001`) are illustrative; use exact strings from the files, keep test file row order, LF endings, UTF-8 no BOM, no quotes, no index column, no trailing spaces.

## 13. Archetype match
Closest past edition: 2024 (strict output format plus validator, F-score with an abstain trade-off) combined with 2025's licence/size and no-external-lookup rules. Otherwise new: a retrieval + cluster-assignment problem (blocking recall ceiling, then a pairwise/list matcher with a precision-tuned threshold and a singleton abstain). No images, no regression.

## 14. Red flags
- Test differs from train: YES (France).
- ID leakage: none expected across splits; label graph in train is stars [EDA].
- Abstaining is rewarded only for singletons; tune the threshold for F_0.5 on val.
- Format traps: tab not comma; exact header names; no spaces/quotes in lists; empty list = the tab followed by nothing; one row per S1 test entity.

Handoffs: metric -> validation-guardian (section 4), schema -> data-detective (sections 2, 12.1), format -> submission-qa (sections 5-7b), archetype -> competition-strategist (section 13).
