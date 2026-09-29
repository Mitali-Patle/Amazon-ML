"""Assemble and verify the competition submission.

Produces the two official files:

    output/matching_results.tsv   -- the ONLY file scored on the leaderboard
    output/candidate_pairs.tsv    -- the blocking set fed to the matcher,
                                     audited by the organisers for recall
                                     ceiling and reduction ratio

Then runs the organisers' own ``utils/validate_submission.py`` against them, so
a format rejection is caught locally instead of costing a submission.

HARD RULES the scorer enforces (each asserted here before the validator runs):
  * every Source-1 test entity appears exactly once -- a missing entity causes
    rejection, not a zero;
  * ``matched_entity_ids`` is empty for singletons, never the string "nan" or
    a placeholder;
  * no duplicate ids within a list, no duplicate source1_entity_id rows;
  * S2-/S3- ids only, never an S1- id (self-match) ;
  * matches must be a SUBSET of candidates -- a matched id that never appeared
    as a candidate signals a pipeline bug and the validator warns about it.

Tab-separated with no quoting: addresses and id lists contain commas, which is
why the whole challenge is TSV rather than CSV.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl

from .config import PARQUET, REPORTS, ROOT

OUTPUT = ROOT / "output"
VALIDATOR = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "utils" / "validate_submission.py"
STUDENT_DIR = VALIDATOR.parents[1]

MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]


def required_entities(split: str = "test") -> list[str]:
    """Every Source-1 entity that MUST appear, in file order."""
    return pl.read_parquet(PARQUET / f"{split}_source1.parquet",
                           columns=["entity_id"])["entity_id"].to_list()


def write_tsv(path: Path, header: Sequence[str], entities: Sequence[str],
              values: Mapping[str, Sequence[str]]) -> dict:
    """Write one official file. Every entity gets a row, even if its list is empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_empty = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        # QUOTE_NONE with no escapechar: ids and the comma-joined list must be
        # written verbatim. Entity ids never contain a tab, so nothing can need
        # escaping -- and any quoting would corrupt the official format.
        w = csv.writer(fh, delimiter="\t", lineterminator="\n",
                       quoting=csv.QUOTE_NONE, quotechar=None, escapechar=None)
        w.writerow(header)
        for eid in entities:
            ids = values.get(eid) or []
            if not ids:
                n_empty += 1
            w.writerow([eid, ",".join(ids)])
    return {"path": str(path), "rows": len(entities), "empty_lists": n_empty,
            "mb": round(path.stat().st_size / 2**20, 2)}


def preflight(entities: Sequence[str], matches: Mapping[str, Sequence[str]],
              candidates: Mapping[str, Sequence[str]] | None = None) -> dict:
    """Check every scorer rule BEFORE writing, so failures are cheap."""
    problems: list[str] = []
    ent_set = set(entities)

    if len(ent_set) != len(entities):
        problems.append("duplicate entity ids in the required-entity list")

    extra = set(matches) - ent_set
    if extra:
        problems.append(f"{len(extra)} predicted entities are not in the test set "
                        f"(e.g. {sorted(extra)[:3]})")

    bad_self = bad_dup = bad_prefix = 0
    total_ids = 0
    for eid in entities:
        ids = list(matches.get(eid) or [])
        total_ids += len(ids)
        if len(ids) != len(set(ids)):
            bad_dup += 1
        for i in ids:
            if i.startswith("S1-"):
                bad_self += 1
            elif not (i.startswith("S2-") or i.startswith("S3-")):
                bad_prefix += 1
    if bad_dup:
        problems.append(f"{bad_dup} rows contain duplicate ids within one list")
    if bad_self:
        problems.append(f"{bad_self} matched ids are S1- self-matches")
    if bad_prefix:
        problems.append(f"{bad_prefix} matched ids have an unexpected prefix")

    # global mutual exclusivity -- verified true in training over all 7,638,365
    # pairs, so a violation here means the deconfliction step was skipped
    seen: dict[str, str] = {}
    contested = 0
    for eid in entities:
        for i in matches.get(eid) or []:
            if i in seen:
                contested += 1
            else:
                seen[i] = eid
    if contested:
        problems.append(f"{contested} S2/S3 ids are claimed by more than one entity "
                        "(violates the verified many-to-one structure)")

    not_subset = 0
    if candidates is not None:
        for eid in entities:
            c = set(candidates.get(eid) or [])
            if not set(matches.get(eid) or []).issubset(c):
                not_subset += 1
        if not_subset:
            problems.append(f"{not_subset} rows predict a match that was never a candidate")

    return {
        "entities": len(entities),
        "predicted_ids": total_ids,
        "mean_predictions": round(total_ids / len(entities), 3) if entities else 0.0,
        "empty_predictions": sum(1 for e in entities if not matches.get(e)),
        "contested_ids": contested,
        "rows_not_subset_of_candidates": not_subset,
        "problems": problems,
        "ok": not problems,
    }


def run_official_validator(matching: Path, candidate: Path | None,
                           test_dir: Path, check_ids: bool = False) -> dict:
    """Invoke the organisers' validator. Its verdict is what counts, not ours."""
    if not VALIDATOR.exists():
        return {"ran": False, "reason": f"validator not found at {VALIDATOR}"}
    cmd = [sys.executable, str(VALIDATOR),
           "--matching", str(matching), "--test-dir", str(test_dir)]
    if candidate is not None:
        cmd += ["--candidate", str(candidate)]
    if check_ids:
        cmd += ["--check-ids"]
    proc = subprocess.run(cmd, cwd=STUDENT_DIR, capture_output=True, text=True)
    return {"ran": True, "exit_code": proc.returncode, "passed": proc.returncode == 0,
            "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-2000:],
            "command": " ".join(cmd)}


def build(matches: Mapping[str, Sequence[str]],
          candidates: Mapping[str, Sequence[str]] | None = None,
          split: str = "test", validate: bool = True,
          check_ids: bool = False, tag: str = "submission") -> dict:
    """Full path: preflight -> write both files -> official validator."""
    entities = required_entities(split)
    report: dict = {"split": split, "tag": tag}

    report["preflight"] = preflight(entities, matches, candidates)
    if not report["preflight"]["ok"]:
        report["ok"] = False
        (REPORTS / f"{tag}.json").write_text(json.dumps(report, indent=2, default=str))
        return report

    report["matching_file"] = write_tsv(
        OUTPUT / "matching_results.tsv", MATCHING_HEADER, entities, matches)
    if candidates is not None:
        report["candidate_file"] = write_tsv(
            OUTPUT / "candidate_pairs.tsv", CANDIDATE_HEADER, entities, candidates)

    if validate:
        test_dir = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "test"
        report["validator"] = run_official_validator(
            OUTPUT / "matching_results.tsv",
            OUTPUT / "candidate_pairs.tsv" if candidates is not None else None,
            test_dir, check_ids=check_ids)
        report["ok"] = report["validator"].get("passed", False)
    else:
        report["ok"] = True

    (REPORTS / f"{tag}.json").write_text(json.dumps(report, indent=2, default=str))
    return report
