# EDA_FINDINGS.md (data-detective, Day 1, train only; data/test never touched)

Scripts: `src/eda/00_convert.py` (TSV to parquet, `data/interim/`, git-ignored), `01_graph.py`, `02_dist.py`, `03_neardup_block.py`, `04_pool.py`. Raw stdout of 03 is `data/interim/03.out`. Read with quoting off (QUOTE_NONE), dtype=str, keep_default_na=False; row counts equal `wc -l - 1`.

## Ranked findings

### 1. The match graph is a disjoint set of stars: split by anchor is leak-free (ADR-001 Q2/Q5 answered)
- 7,638,365 match pairs; every S2/S3 id is matched by exactly ONE anchor (0 targets shared, 0 duplicate pairs). 2,083,574 components = 2,083,574 anchors with >=1 match; 0 components contain >1 anchor. Component sizes 2..12 (anchor + 1..11 targets), mode 4-5 nodes.
- Zero ids in ground truth missing from source files; zero S1 ids missing from GT (1:1, same order as source1 file); no duplicate ids in any file; all prefixes valid.
- Implication: connected-component grouping adds nothing beyond `source1_entity_id`; random/stratified split by anchor is already group-safe on the label graph. Removes ADR-001 Decision 1 risk #1.
- Action (validation-guardian): split by anchor id, stratify on (has_match, country). Keep union-find only for near-dup risk (finding 3).

### 2. Match lists are 1-to-MANY and mix S2 and S3; the metric must be set-based
- Match-list length: 0: 123,247 (5.58%); 1: 119,157; 2: 375,212; 3: 530,841; 4: 484,115; 5: 321,957; 6: 164,868; 7: 63,968; 8: 18,680; 9+: 4,776. Mean 3.46 overall, 3.66 among matched. Max 11.
- S2 count per anchor 0..5, S3 count 0..6. Anchors having both an S2 and an S3 match: 1,776,047 (80.5%); S2 only 143,029; S3 only 164,498; none 123,247. P(has S2) = 87.0%, P(has S3) = 87.9%.
- Each anchor's targets are several noisy variants of the same entity (dupes inside S2 and S3), so the task is "retrieve a cluster", not top-1. Expect recall to depend on cluster completion; top-k must be adaptive (typical answer 3-5 ids).
- Pool distractors: 26.6% of S2 (1,341k rows) and 25.4% of S3 (1,341k) are matched by NO train anchor (also `is_matched` is 73.4%/74.6%). Unmatched are pure distractors: exact (name,addr) overlap with any S1 anchor = 0 of 1.34M (matched: 1 and 17).
- Action: model as anchor-to-cluster assignment; threshold + per-anchor cardinality prediction; Tier-1 val pool can sample from the 1.34M+1.34M unmatched reservoir safely (ADR-001 Decision 2 confirmed). Track precision/recall by source S2/S3 too.

### 3. Near-duplicates in S1: chain names are pervasive, true (name,addr) dupes are not
- Exact (name,addr) dupes in S1: 0. Normalized name+normalized address dupes: 4 rows. Exact address dupes: 116,304 rows (5.3%, same address, different name). Normalized-name (lowercased, accents/punct/legal suffix stripped) shared by another S1 row: 48.9% of rows; 31,782 names repeated >=5 times (658k rows); top: "meridian" 552, "cedar" 336. Same name + same city: 166,821 rows; same name + same leading street number: 123,566 rows.
- Chain rows (e.g. "1 800 Flowers" x dozens across cities) are different locations with DISJOINT match lists (guaranteed by finding 1). So this is a precision hazard, not a leakage bug: same name is not sufficient evidence, address must discriminate.
- Leakage risk to split: low. A model could memorize chain-name priors, which also holds at test time (fair). Not worth union-find merging on names (would create a giant component of common names).
- Action: do not cluster by name for splitting; optionally group by (normalized name, city) only for a leakage-stress split later. Ensure the matcher weights address and street number, and add a "name frequency" feature (common-name penalty).

### 4. Blocking is the ceiling problem: ~15% of true pairs share no name token; ~57% are not exact-name
Measured on 20,000 random matched anchors (~73.8k true pairs) against the full S2+S3 pool (`03_neardup_block.py`). Assumption: sample is representative (random, seed 0).
- Country of every true target equals the anchor's country (100%): hard-block on country for free (halves the pool).
- Normalized-name exact equality on true pairs: 42.5% (S2 43.4%, S3 41.7%). First-N-chars equality: 3 chars 79.7%, 4 chars 78.8%, 6 chars 75.1%, 8 chars 69.1%.
- Token overlap: 15.0% of true pairs have ZERO shared name tokens; median Jaccard 0.75, 25th pct 0.4. Recall by "shares any token" (df cap per 1M rows): df<=50: 37.5%; <=500: 56.2%; <=5000: 84.2%; no cap: 85.0%. So token blocking tops out at ~85% and rare-token-only blocking at 37-56%.
- 4-char-prefix+country block: median 166 candidates, mean 1,001, p95 5,028 at 78.8% pair recall; big blocks for common prefixes.
- Zero-overlap causes (see `03.out`): (a) target name in another script (Devanagari and others) while anchor is Latin; (b) target name empty or a URL-like string ("rkkproducts com"); (c) OCR/keyboard typos ("lnedirn" for Lantern, "wi1lis center", "Zeque"->"zree", "Artoobor" inserted); (d) token order shuffles, glued tokens.
- Address helps: leading/any numeric token overlap among true pairs (where anchor has a number) = 82.9% overall (India 90.5%, US 78.2%). Address in targets is missing for 4.5% of matched S2/S3 rows (vs 0.28% of unmatched).
- Implication: a single lexical blocker is insufficient. Need union of: (i) char n-gram TF-IDF/BM25 on normalized name (handles typos, glued tokens), (ii) multilingual/transliteration-aware dense embeddings (handles non-Latin), (iii) address-number + city keys for India/US. Target candidate recall >= 95% at k~50-100 per anchor per source, measured on val.
- Action (multimodal-feature-engineer/ml-architect): build blocker recall harness first; log recall@k per country and per target script.

### 5. Scripts and noise are asymmetric: S1 is clean Latin; S2/S3 carry the noise
- Countries: only US and India everywhere. S1: US 1,323,633 (60.0%), India 883,188 (40.0%). S2: US 3,016,817 / India 2,017,799 (60/40). S3: US 3,170,056 / India 2,115,547 (60/40).
- No-match rate identical across countries: US 5.58%, India 5.59%; mean matches/anchor 3.459 vs 3.465. Country is not predictive of no-match or cardinality, so stratifying on country is harmless but not essential.
- Non-Latin names: S1 0.0%; S2 9.4% (Devanagari 5.4%, remainder Tamil, Gujarati, Kannada, etc.); S3 5.3% (Devanagari 3.0%). Addresses also partly in regional script (state names, e.g. Karnataka rendered natively), ~5.2-5.5% Devanagari addresses in S2/S3, 0 in S1. Both appear only in India records.
- Empty addresses: S1 0; S2 168,967 (3.4%); S3 175,916 (3.5%). Names <3 chars: S2 704, S3 9,275. S1 has no empty/short names.
- Casing: S2 18.9% ALL-CAPS names (S3 3.0%, S1 0%), S3 6.2% all-lowercase; ~4% of S2/S3 names are URL-like (".com").
- Injected noise types (visible in samples): typos, accent injection ("Ínc", "Súnmark"), digit-for-letter ("Ve1ma"), reordered legal suffix ("Inc Hughes Optimal Tree"), doubled or bracketed suffix ("Private (Limited)", "((Ltd))"), suffix truncation ("Private"), "X trading as Y", "Sri" prefix, punctuation/space corruption ("Quartz,-LLC").
- Legal suffix on S1 name: Inc/LLC/Ltd/Corp 35.2% of names, Private Limited/Pvt Ltd 25.1%; drop or normalize before comparing (suffixes are frequently altered in targets).
- Address formats: S1 US = "street, city, ST"; S2 US often ALL-CAPS with shuffled component order ("KS, 124 ERIE ST, WICHITA", "CHICAGO, 10047 FOREST AVE, IL"), S3 US uses full state names ("Ohio", "Texas") and shuffled order ("New York, PO Box..., Yorktown"); S3 ends with a 2-letter state only 25% vs S1 52% / S2 50%. India: long free-text with landmarks, "NO ##8TH FLOOR" style corruption, district names in the middle. Zip appears in only ~6.5% of US-ish rows; PIN in <1%.
- Action: per-country normalizers: US = tokenize addresses into an order-free bag of (number, street, city, state with abbreviation<->full-name map), India = number/tokens + city/state with regional-script maps; strip legal suffixes; casefold; transliterate or embed non-Latin.

### 6. Exact duplicates inside S2/S3 exist and are always matched
- S2: 25,891 rows repeat another row's (name,addr); S3: 18,881. 100% of these duplicated rows are matched to some anchor (none in the unmatched reservoir): duplicates are within-cluster copies. Not a leakage issue; predicting all copies together is correct, and a dedup-then-expand trick is valid: predict for one representative and add exact copies.
- Action: after matching, expand to include exact (name,addr) copies of any predicted id (cheap recall gain, precision-safe given 100% matched).

### 7. Weak/unused signals for no-match (singleton) prediction
- No-match rate is flat by country (5.58/5.59%), address-has-digit (5.6/5.6%), and name-length quintile (~5.6% each). No cheap metadata predicts singletons; must come from "no candidate scores above threshold". Distractors carry near-zero empty-address (0.28%) so empty-address candidates are more likely true matches (4.5% of matched rows) than distractors: a mild prior.
- Action: singleton decision = calibrated max-score threshold tuned for F0.5 on val (Tier 2).

## ADR-001 open-question answers
- Q2 (shared S2/S3 across anchors): NO. 0 shared; components == stars.
- Q3 (near-dup S1 leak): no exact/near (name+addr) dupes (4 rows). Chain-name repeats are common (48.9% share a name) but have disjoint matches. Do not merge on name.
- Q4 (country/match status): US 60/India 40 in every source; no-match rate 5.6% in both; stratify on has_match (and optionally country) is enough.
- Q5 (S2/S3 mixing): yes, 80.5% of anchors have both.

## Recommended next steps
1. validation-guardian: finalize split by anchor id (10% or 5% val, seed 42), stratified by has_match x country. No union-find needed.
2. ml-architect/multimodal-feature-engineer: blocker with country hard-filter and union of char-ngram TF-IDF + multilingual dense embeddings + address-number key; report recall@k. Target >=95% pair recall.
3. Assumptions: 20k-anchor sample for finding 4 (random seed 0); shares-token stat uses a 1M-row df sample and first 60k pairs; regex-based script detection (Devanagari U+0900-097F; "non-Latin" = code points above U+024F).
