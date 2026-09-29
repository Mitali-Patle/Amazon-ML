#!/bin/bash
# Unattended overnight run: candidates -> GBDT -> cross-encoder -> submission.
#
# Designed to be left alone. Every stage is RESUMABLE (pass A skips score shards
# that already exist), every stage is gated on the previous one succeeding, and
# a failure stops the chain with a clear marker rather than cascading into a
# corrupt submission.
#
# SAFETY: the existing output/matching_results.tsv is backed up before pass C
# overwrites it, so a mid-write failure cannot leave us with no submission at
# all. The previously validated file is always recoverable.

cd /home/kundhave/Amazon-ML || exit 1
SP=/tmp/claude-1000/-home-kundhave-Amazon-ML/2c507597-1737-4c2b-9b7b-566620ffb77b/scratchpad
PY=.venv/bin/python
ST="$SP/overnight_status.txt"

say() { echo "[$(date '+%a %H:%M:%S')] $*" | tee -a "$ST"; }
fail() { say "!! FAILED at $1 -- chain stopped. log: $2"; say "   output/ still holds the last validated submission."; exit 1; }

say "OVERNIGHT CHAIN START"

# ---- 1. wait for K=150 candidate generation (already running) ----
if pgrep -f "generate_candidates test" > /dev/null; then
  say "1/4 waiting for test candidate generation"
  while pgrep -f "generate_candidates test" > /dev/null; do sleep 60; done
fi
if ! grep -q "DONE in" "$SP/gen_test150.log" 2>/dev/null; then
  fail "1/4 candidates" "$SP/gen_test150.log"
fi
NSH=$(ls data/candidates/test/merged/*.parquet 2>/dev/null | wc -l)
[ "$NSH" -eq 0 ] && fail "1/4 candidates (no merged shards)" "$SP/gen_test150.log"
say "1/4 OK -- $NSH merged shards"

# ---- 1b. stronger cross-encoder: 6 epochs instead of 3 ----
# It had not plateaued: val AUC 0.9903 -> 0.9939 -> 0.9947 across epochs 1-3.
# Chosen over raising K to 200, which measured only +0.0013 of ceiling
# (0.9884 -> 0.9897) and risks LOWER matcher efficiency from extra distractors.
# Gated: the previous model is restored if the longer run does not validate better.
say "1b/4 retraining cross-encoder, 6 epochs (~50 min, GPU)"
cp -r data/models/crossencoder data/models/crossencoder_3ep 2>/dev/null
PREV=$($PY -c "import json;print(json.load(open('data/reports/cascade_eval.json'))['best_macro_f05'])" 2>/dev/null || echo 0)
$PY -u -m src.prep.train_crossencoder --epochs 6 --bs 64 --lr 3e-5 > "$SP/ce_6ep.log" 2>&1
if grep -q "CE_DONE" "$SP/ce_6ep.log"; then
  $PY -u -m src.prep.eval_cascade > "$SP/cascade_eval_6ep.log" 2>&1
  NEW=$($PY -c "import json;print(json.load(open('data/reports/cascade_eval.json'))['best_macro_f05'])" 2>/dev/null || echo 0)
  BETTER=$($PY -c "print(1 if $NEW > $PREV else 0)")
  if [ "$BETTER" = "1" ]; then
    say "1b/4 OK -- 6-epoch model BETTER: $PREV -> $NEW (keeping it)"
  else
    say "1b/4 6-epoch model NOT better ($PREV -> $NEW) -- restoring 3-epoch model"
    rm -rf data/models/crossencoder && mv data/models/crossencoder_3ep data/models/crossencoder
  fi
else
  say "1b/4 retrain failed -- keeping the 3-epoch model and continuing"
  rm -rf data/models/crossencoder && cp -r data/models/crossencoder_3ep data/models/crossencoder
fi

# ---- 1c. joint calibration of the blended cascade score ----
# Stage 1 and stage 2 are each calibrated alone but never together, and the seam
# is measurable: predicted 0.1987 -> actual 0.1485, predicted 0.3856 -> 0.3290.
# That over-confidence sits exactly where the rule decides whether to take one
# more candidate, so it converts into false positives at 2x weight.
# Fitted on TRAIN only; eval_cascade then measures it on validation and the
# result is kept only if it actually helps.
say "1c/4 fitting joint calibrator on train (~15 min)"
$PY -u -m src.prep.joint_calibrate > "$SP/joint_cal.log" 2>&1
if grep -q "JOINT_CALIBRATION_DONE" "$SP/joint_cal.log"; then
  $PY -u -m src.prep.eval_cascade > "$SP/eval_jointcal.log" 2>&1
  BESTCFG=$($PY -c "import json;print(json.load(open('data/reports/cascade_eval.json'))['best_w'])" 2>/dev/null || echo "")
  BESTVAL=$($PY -c "import json;print(json.load(open('data/reports/cascade_eval.json'))['best_macro_f05'])" 2>/dev/null || echo 0)
  case "$BESTCFG" in
    *jointcal*) say "1c/4 OK -- joint calibration HELPS: best=$BESTCFG at $BESTVAL (keeping)" ;;
    *) say "1c/4 joint calibration did NOT help (best=$BESTCFG at $BESTVAL) -- disabling it"
       mv data/models/joint_calibrator.pkl data/models/joint_calibrator_unused.pkl 2>/dev/null ;;
  esac
else
  say "1c/4 joint calibration failed -- continuing without it"
  rm -f data/models/joint_calibrator.pkl
fi

# ---- 2. pass A: features + GBDT over ~520M pairs (resumable) ----
say "2/4 pass A: features + GBDT (~5h, resumable per shard)"
for attempt in 1 2; do
  $PY -u -m src.prep.predict_cascade --only a >> "$SP/cas150_a.log" 2>&1
  DONE=$(ls data/scores/test/*.parquet 2>/dev/null | wc -l)
  [ "$DONE" -ge "$NSH" ] && break
  say "   pass A incomplete ($DONE/$NSH) -- retrying once (it resumes from existing shards)"
done
DONE=$(ls data/scores/test/*.parquet 2>/dev/null | wc -l)
[ "$DONE" -ge "$NSH" ] || fail "2/4 pass A ($DONE/$NSH shards)" "$SP/cas150_a.log"
say "2/4 OK -- $DONE score shards"

# ---- 3. pass B: cross-encoder on the ambiguous band ----
say "3/4 pass B: cross-encoder on band 0.02-0.98 (~75 min, GPU)"
$PY -u -m src.prep.predict_cascade --only b > "$SP/cas150_b.log" 2>&1
[ -f data/scores/test_band_ce.parquet ] || fail "3/4 pass B" "$SP/cas150_b.log"
say "3/4 OK -- $(grep -o 'scored [0-9,]* band pairs' "$SP/cas150_b.log" | tail -1)"

# ---- 4. pass C: decide, deconflict, validate, submit ----
say "4/4 pass C: decision + deconfliction + official validator"
mkdir -p output/backup
[ -f output/matching_results.tsv ] && cp output/matching_results.tsv output/backup/matching_results_prev.tsv
$PY -u -m src.prep.predict_cascade --only c > "$SP/cas150_c.log" 2>&1
if grep -q "CASCADE_SUBMISSION_OK" "$SP/cas150_c.log"; then
  say "4/4 OK -- validator PASSED"
  say "SUBMISSION READY: output/matching_results.tsv"
  grep -E "mean preds|deconflict removed|VALIDATOR" "$SP/cas150_c.log" | tail -3 | tee -a "$ST"
else
  cp output/backup/matching_results_prev.tsv output/matching_results.tsv 2>/dev/null
  fail "4/4 pass C (previous submission restored)" "$SP/cas150_c.log"
fi

say "OVERNIGHT CHAIN COMPLETE"
