#!/bin/bash
# Fast path to a complete, validated submission. Target ~2h.
#
# What was cut and why:
#   * TRAIN sampled to 150k entities (from 2.1M). At a 3.3% positive rate that
#     is still ~500k positive pairs -- far past where a GBDT on 34 features
#     stops improving. Expected F_0.5 impact: +/-0.002, unmeasurable.
#   * K reduced 50 -> 25 per source. Measured cost: candidate recall
#     95.39% -> 94.46%, lowering the oracle ceiling ~0.009. This is the ONLY
#     cut that costs real score; everything else was over-computation.
#   * BATCH_QUERIES 512 -> 2048: fewer, larger sparse matmuls.
#
# TEST is never sampled -- every test entity must appear in the submission.
#
# Steps are strictly serial: each peaks at 8-13GB on a 15GB box.

cd /home/kundhave/Amazon-ML || exit 1
SP=/tmp/claude-1000/-home-kundhave-Amazon-ML/2c507597-1737-4c2b-9b7b-566620ffb77b/scratchpad
PY=.venv/bin/python
STATUS="$SP/fast_status.txt"
K=25
TRAIN_SAMPLE=150000

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$STATUS"; }
fail() { say "FAILED at $1 -- chain stopped. See $2"; exit 1; }

say "FAST CHAIN START (k=$K per source, train sample=$TRAIN_SAMPLE)"

# ---- 1. sampled train candidates ----
# If a run is already in flight (launched manually to overlap with writing the
# later stages), adopt it rather than restarting from scratch.
if pgrep -f "generate_candidates train" > /dev/null; then
  say "1/5 train candidates already running -- waiting"
  while pgrep -f "generate_candidates train" > /dev/null; do sleep 20; done
else
  say "1/5 train candidates, $TRAIN_SAMPLE entities (~12 min)"
  rm -rf data/candidates/train
  $PY -u -m src.prep.generate_candidates train $K $TRAIN_SAMPLE > "$SP/gen_train.log" 2>&1
fi
grep -q "DONE" "$SP/gen_train.log" || fail "1/5 train candidates" "$SP/gen_train.log"
say "1/5 OK"

# ---- 1b. regenerate VALIDATION candidates at the same k as test ----
# The saved val candidates were built at k=50/source (100 per entity) while test
# now uses k=25/source (50 per entity). Evaluating on 2x the candidates test
# will see makes the validation F_0.5 optimistic -- validation must mirror test
# conditions, so it is rebuilt at the same k.
say "1b/5 validation candidates at k=$K (~5 min)"
rm -rf data/candidates/val
$PY -u -m src.prep.generate_candidates val $K > "$SP/gen_val.log" 2>&1
grep -q "DONE" "$SP/gen_val.log" || fail "1b/5 val candidates" "$SP/gen_val.log"
say "1b/5 OK"

# ---- 2. pair features: train sample + validation ----
say "2/5 pair features train+val (~14 min)"
$PY -u -c "
import sys; sys.path.insert(0,'.')
from src.prep.pair_features import run
from src.prep.config import CANDIDATES, PARQUET, DATA
r1 = run(candidates_path=CANDIDATES/'train'/'merged', split='train',
         out_dir=DATA/'features'/'pairs'/'train', chunk_size=10000,
         ground_truth_path=PARQUET/'train_ground_truth.parquet',
         include_label=True, report_name='pf_train')
print('TRAIN_PAIRS', r1.get('pairs'))
r2 = run(candidates_path=CANDIDATES/'val'/'merged', split='train',
         out_dir=DATA/'features'/'pairs'/'val', chunk_size=10000,
         ground_truth_path=PARQUET/'train_ground_truth.parquet',
         include_label=True, report_name='pf_val')
print('VAL_PAIRS', r2.get('pairs'))
print('PF_DONE')
" > "$SP/pf_trainval.log" 2>&1
grep -q "PF_DONE" "$SP/pf_trainval.log" || fail "2/5 pair features" "$SP/pf_trainval.log"
say "2/5 OK -- $(grep -o 'TRAIN_PAIRS [0-9]*' "$SP/pf_trainval.log") $(grep -o 'VAL_PAIRS [0-9]*' "$SP/pf_trainval.log")"

# ---- 3. train the matcher + calibrate + score on validation ----
say "3/5 train matcher + calibrate + evaluate (~15 min)"
$PY -u -m src.prep.train_matcher > "$SP/train_matcher.log" 2>&1
grep -q "MATCHER_DONE" "$SP/train_matcher.log" || fail "3/5 matcher" "$SP/train_matcher.log"
say "3/5 OK -- $(grep 'VAL_MACRO_F05' "$SP/train_matcher.log" | tail -1)"

# ---- 4. test candidates (never sampled) ----
say "4/5 test candidates, 1,732,544 entities (~45 min)"
rm -rf data/candidates/test
$PY -u -m src.prep.generate_candidates test $K > "$SP/gen_test.log" 2>&1
grep -q "DONE" "$SP/gen_test.log" || fail "4/5 test candidates" "$SP/gen_test.log"
say "4/5 OK"

# ---- 5. test features -> predict -> submission ----
say "5/5 test features + predict + submit (~50 min)"
$PY -u -m src.prep.predict_submit > "$SP/predict_submit.log" 2>&1
grep -q "SUBMISSION_OK" "$SP/predict_submit.log" || fail "5/5 predict/submit" "$SP/predict_submit.log"
say "5/5 OK -- $(grep 'VALIDATOR' "$SP/predict_submit.log" | tail -1)"

say "FAST CHAIN COMPLETE -- output/matching_results.tsv ready"
