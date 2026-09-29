#!/bin/bash
# Unattended completion of the remaining preprocessing.
#
# ORDERING IS DELIBERATE. Model training needs only train+val features; test
# artifacts are for inference. So the chain front-loads everything that unblocks
# training and defers test work, cutting time-to-training from ~6.6h to ~3.4h.
#
#   1. wait for train candidates      (already running)
#   2. pair features on VALIDATION    (~6 min)   -> evaluation set ready
#   3. pair features on TRAIN         (~105 min) -> *** TRAINING UNBLOCKED ***
#   4. test candidates                (~110 min)
#   5. pair features on TEST          (~85 min)  -> inference ready
#
# Steps run STRICTLY SERIALLY. Each peaks at 8-13GB on a 15GB box; running two
# concurrently drove free memory to 1GB and nearly killed a 1.5h job.
#
# Every step is gated on the previous one succeeding, so a failure stops the
# chain with a clear marker instead of cascading into corrupt downstream data.

cd /home/kundhave/Amazon-ML || exit 1
SP=/tmp/claude-1000/-home-kundhave-Amazon-ML/2c507597-1737-4c2b-9b7b-566620ffb77b/scratchpad
PY=.venv/bin/python
STATUS="$SP/chain_status.txt"

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$STATUS"; }

say "CHAIN START"

# ---- 1. wait for the already-running train candidate generation ----
say "step 1/5: waiting for train candidates"
while pgrep -f "generate_candidates train" > /dev/null; do sleep 60; done
if ! grep -q "DONE" "$SP/gen_train.log" 2>/dev/null; then
  say "FAILED: train candidates did not complete. Chain stopped."
  exit 1
fi
say "step 1/5 OK: train candidates complete"

# ---- 2. pair features on validation ----
say "step 2/5: pair features on validation (~6 min)"
$PY -u -c "
import sys; sys.path.insert(0,'.')
from src.prep.pair_features import run
from src.prep.config import CANDIDATES, PARQUET, DATA
r = run(candidates_path=CANDIDATES/'val_candidate_pairs_per_source_50.tsv',
        split='train', out_dir=DATA/'features'/'pairs'/'val', chunk_size=10000,
        ground_truth_path=PARQUET/'train_ground_truth.parquet',
        include_label=True, report_name='phase10_pair_features_val')
print('PF_VAL_DONE', r.get('pairs'), 'pairs')
" > "$SP/pf_val.log" 2>&1
grep -q "PF_VAL_DONE" "$SP/pf_val.log" || { say "FAILED at step 2 (pair features val)"; exit 1; }
say "step 2/5 OK: validation features ready"

# ---- 3. pair features on train ----
say "step 3/5: pair features on train (~105 min)"
$PY -u -c "
import sys; sys.path.insert(0,'.')
from src.prep.pair_features import run
from src.prep.config import CANDIDATES, PARQUET, DATA
r = run(candidates_path=CANDIDATES/'train'/'merged', split='train',
        out_dir=DATA/'features'/'pairs'/'train', chunk_size=10000,
        ground_truth_path=PARQUET/'train_ground_truth.parquet',
        include_label=True, report_name='phase10_pair_features_train')
print('PF_TRAIN_DONE', r.get('pairs'), 'pairs')
" > "$SP/pf_train.log" 2>&1
grep -q "PF_TRAIN_DONE" "$SP/pf_train.log" || { say "FAILED at step 3 (pair features train)"; exit 1; }
say "step 3/5 OK: *** MODEL TRAINING IS NOW UNBLOCKED ***"

# ---- 4. test candidates ----
say "step 4/5: test candidates (~110 min)"
$PY -u -m src.prep.generate_candidates test > "$SP/gen_test.log" 2>&1
grep -q "DONE" "$SP/gen_test.log" || { say "FAILED at step 4 (test candidates)"; exit 1; }
say "step 4/5 OK: test candidates ready"

# ---- 5. pair features on test (no labels) ----
say "step 5/5: pair features on test (~85 min)"
$PY -u -c "
import sys; sys.path.insert(0,'.')
from src.prep.pair_features import run
from src.prep.config import CANDIDATES, DATA
r = run(candidates_path=CANDIDATES/'test'/'merged', split='test',
        out_dir=DATA/'features'/'pairs'/'test', chunk_size=10000,
        ground_truth_path=None, include_label=False,
        report_name='phase10_pair_features_test')
print('PF_TEST_DONE', r.get('pairs'), 'pairs')
" > "$SP/pf_test.log" 2>&1
grep -q "PF_TEST_DONE" "$SP/pf_test.log" || { say "FAILED at step 5 (pair features test)"; exit 1; }
say "step 5/5 OK: test features ready"

say "CHAIN COMPLETE - all preprocessing finished"
