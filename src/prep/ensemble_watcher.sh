#!/bin/bash
# Auto-runs the ensemble the moment BOTH preconditions are met:
#   1. our own test scores exist (pass A + pass B finished)
#   2. the teammate's v3 per-pair probabilities have been copied in
#
# Deliberately INDEPENDENT of overnight_chain.sh. Editing a running bash script
# is unreliable (bash reads it incrementally), and this step is too important to
# attach to a process already in flight. It polls instead, so it works whatever
# order things finish in -- including if the file is copied in tomorrow morning.
#
# WHY THIS MATTERS: solo we project ~0.973, against the teammate's measured v3
# bar of 0.974. The ensemble is not an optimisation, it is the only path past
# parity. Two models at the same score with uncorrelated errors is exactly the
# setup where probability averaging pays.

cd /home/kundhave/Amazon-ML || exit 1
SP=/tmp/claude-1000/-home-kundhave-Amazon-ML/2c507597-1737-4c2b-9b7b-566620ffb77b/scratchpad
PY=.venv/bin/python
THEIRS=data/external/v3_pair_probabilities.tsv
ST="$SP/ensemble_status.txt"

say() { echo "[$(date '+%a %H:%M:%S')] $*" | tee -a "$ST"; }

say "ENSEMBLE WATCHER ARMED"
say "  waiting for: $THEIRS  AND  our test scores + cross-encoder band"

while true; do
  HAVE_THEIRS=0; [ -s "$THEIRS" ] && HAVE_THEIRS=1
  NS=$(ls data/scores/test/*.parquet 2>/dev/null | wc -l)
  HAVE_CE=0; [ -f data/scores/test_band_ce.parquet ] && HAVE_CE=1
  if [ "$HAVE_THEIRS" = "1" ] && [ "$NS" -ge 18 ] && [ "$HAVE_CE" = "1" ]; then
    break
  fi
  sleep 120
done

say "both ready -- theirs=$(du -h "$THEIRS" | cut -f1), our score shards=$NS"

# Back up the solo submission first: it is a valid fallback and must survive.
mkdir -p output/backup
[ -f output/matching_results.tsv ] && cp output/matching_results.tsv output/backup/matching_results_solo.tsv
say "solo submission backed up to output/backup/matching_results_solo.tsv"

# Compare blending policies. No test labels exist, so the choice is made on
# DISTRIBUTIONAL sanity (train truth: mean 3.46 matches/entity, 5.58% singletons)
# rather than on a score -- there is no score to tune against.
for POL in union_pess union_mean intersect; do
  say "ensemble policy=$POL (dry run, no write)"
  $PY -u -m src.prep.ensemble --theirs "$THEIRS" --policy "$POL" --dry-run \
      >> "$SP/ensemble_compare.log" 2>&1
  grep -E "mean predictions|empty \(singleton\)|pairs:" "$SP/ensemble_compare.log" | tail -3 | tee -a "$ST"
done

# Write the default (union_pess) so a validated ensemble submission exists even
# if nobody is awake to choose. The alternatives stay one command away.
say "writing submission with policy=union_pess"
$PY -u -m src.prep.ensemble --theirs "$THEIRS" --policy union_pess \
    > "$SP/ensemble_final.log" 2>&1
if grep -q "ENSEMBLE_SUBMISSION_OK" "$SP/ensemble_final.log"; then
  say "ENSEMBLE SUBMISSION READY -- output/matching_results.tsv (validator PASSED)"
  grep -E "mean predictions|empty \(singleton\)" "$SP/ensemble_final.log" | tee -a "$ST"
else
  say "!! ensemble write FAILED -- restoring the solo submission"
  cp output/backup/matching_results_solo.tsv output/matching_results.tsv 2>/dev/null
  tail -5 "$SP/ensemble_final.log" | tee -a "$ST"
fi
say "WATCHER DONE"
