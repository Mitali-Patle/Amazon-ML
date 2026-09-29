#!/bin/bash
# Recovery chain: wait for pass B, then pass C, then the ensemble.
#
# The original overnight_chain died during pass B (it loaded all 21.4M band
# pairs at once -- France's wider band made that 4x larger than planned). Pass B
# has been restarted with chunking and is resumable; this picks up the rest.
#
# Every stage is gated and the previous submission is preserved, so a failure
# here leaves the last good file in place rather than nothing.

cd /home/kundhave/Amazon-ML || exit 1
SP=/tmp/claude-1000/-home-kundhave-Amazon-ML/2c507597-1737-4c2b-9b7b-566620ffb77b/scratchpad
PY=.venv/bin/python
THEIRS=data/external/v3_pair_probabilities.tsv
ST="$SP/finish_status.txt"

say() { echo "[$(date '+%a %H:%M:%S')] $*" | tee -a "$ST"; }

say "FINISH CHAIN ARMED"

# ---- wait for pass B ----
say "waiting for pass B (cross-encoder on 21.4M band pairs, ~3.6h)"
while pgrep -f "predict_cascade --only b" > /dev/null; do sleep 120; done
if [ ! -f data/scores/test_band_ce.parquet ]; then
  say "!! pass B did not produce test_band_ce.parquet -- retrying once (resumes from written parts)"
  $PY -u -m src.prep.predict_cascade --only b >> "$SP/cas150_b.log" 2>&1
fi
[ -f data/scores/test_band_ce.parquet ] || { say "!! pass B FAILED twice -- stopping. Previous submission untouched."; exit 1; }
say "pass B OK -- $($PY -c "import polars as pl;print(f'{pl.read_parquet(\"data/scores/test_band_ce.parquet\").height:,} band pairs scored')")"

# ---- pass C: our solo submission ----
mkdir -p output/backup
[ -f output/matching_results.tsv ] && cp output/matching_results.tsv output/backup/matching_results_prev.tsv
say "pass C: decision + deconfliction + validator (~45 min)"
$PY -u -m src.prep.predict_cascade --only c > "$SP/cas150_c.log" 2>&1
if grep -q "CASCADE_SUBMISSION_OK" "$SP/cas150_c.log"; then
  say "pass C OK -- solo submission validated"
  cp output/matching_results.tsv output/backup/matching_results_solo.tsv
  grep -E "mean preds|deconflict removed" "$SP/cas150_c.log" | tail -2 | tee -a "$ST"
else
  say "!! pass C FAILED -- restoring previous submission"
  cp output/backup/matching_results_prev.tsv output/matching_results.tsv 2>/dev/null
  tail -5 "$SP/cas150_c.log" | tee -a "$ST"
  exit 1
fi

# ---- ensemble: the step that actually beats 0.974 ----
if [ ! -s "$THEIRS" ]; then
  say "!! teammate file missing -- solo submission stands"
  exit 0
fi
say "ensemble: comparing blend policies (no test labels, so judged on distribution)"
for POL in union_pess union_mean intersect; do
  $PY -u -m src.prep.ensemble --theirs "$THEIRS" --policy "$POL" --dry-run \
      >> "$SP/ensemble_compare.log" 2>&1
  say "  $POL: $(grep -E 'mean predictions' "$SP/ensemble_compare.log" | tail -1 | tr -s ' ')"
done
say "writing ensemble submission (policy=union_pess)"
$PY -u -m src.prep.ensemble --theirs "$THEIRS" --policy union_pess > "$SP/ensemble_final.log" 2>&1
if grep -q "ENSEMBLE_SUBMISSION_OK" "$SP/ensemble_final.log"; then
  say "ENSEMBLE SUBMISSION READY -- output/matching_results.tsv (validator PASSED)"
  grep -E "mean predictions|empty \(singleton\)|pairs:" "$SP/ensemble_final.log" | tee -a "$ST"
else
  say "!! ensemble FAILED -- restoring solo submission (still a valid entry)"
  cp output/backup/matching_results_solo.tsv output/matching_results.tsv 2>/dev/null
  tail -5 "$SP/ensemble_final.log" | tee -a "$ST"
fi
say "FINISH CHAIN DONE"
