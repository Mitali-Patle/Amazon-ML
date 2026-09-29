"""End-to-end preprocessing pipeline.

    .venv/bin/python -m src.prep.run_pipeline [--from PHASE] [--only PHASE]

Regenerates every artifact from the immutable raw TSVs. Phases are ordered by
dependency; each writes a JSON verification report to data/reports/ and aborts
the run on a failed gate.

Phase 4 is the leakage gate: nothing fitted on data may run before it. Phase 3
deliberately runs BEFORE the split because it fits nothing -- it is pure
per-record transformation, so running it pre-split is leakage-free and
guarantees train and validation are transformed by identical code.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

from .config import CONFIG_VERSION, REPORTS, SEED, config_hash

PHASES = ["env", "ingest", "audit", "represent", "split", "validate"]


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def manifest() -> dict:
    import numpy, pandas, polars, pyarrow, scipy  # noqa: PLC0415
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "config_version": CONFIG_VERSION,
        "config_hash": config_hash(),
        "seed": SEED,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "numpy": numpy.__version__, "pandas": pandas.__version__,
            "polars": polars.__version__, "pyarrow": pyarrow.__version__,
            "scipy": scipy.__version__,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=PHASES, default="ingest")
    ap.add_argument("--only", choices=PHASES)
    args = ap.parse_args()

    todo = [args.only] if args.only else PHASES[PHASES.index(args.start):]
    mf = manifest()
    (REPORTS / "run_manifest.json").write_text(json.dumps(mf, indent=2))
    print(f"config_hash={mf['config_hash']} git={mf['git_sha']} seed={mf['seed']}")
    print(f"phases: {' -> '.join(todo)}\n")

    timings = {}
    for phase in todo:
        if phase == "env":
            continue
        t0 = time.time()
        print(f"{'=' * 62}\nPHASE: {phase}\n{'=' * 62}")
        if phase == "ingest":
            from .ingest import ingest_all
            r = ingest_all()
            ok = r["ok"]
        elif phase == "audit":
            from .audit import audit
            r = audit()
            ok = r["ok"]
            if r.get("critical_failed"):
                print(f"CRITICAL INVARIANT FAILED: {r['critical_failed']}")
                return 2
        elif phase == "represent":
            from .represent import main as rep_main
            r = rep_main()
            ok = r["ok"]
        elif phase == "split":
            from .split import build_split
            r = build_split()
            ok = r["ok"]
        elif phase == "validate":
            from .run_validation import run
            r = run()
            ok = True
            a = r["aggregate"]
            print(f"\n  pair recall@50   = {a['pair_recall']['@50']}%")
            print(f"  entity recall@50 = {a['entity_recall']['@50']}%")
        else:
            raise SystemExit(f"unknown phase {phase}")

        timings[phase] = round(time.time() - t0, 1)
        if not ok:
            print(f"\nPHASE {phase} FAILED -- see data/reports/")
            return 1
        print(f"\n[{phase} ok in {timings[phase]}s]\n")

    mf["phase_seconds"] = timings
    (REPORTS / "run_manifest.json").write_text(json.dumps(mf, indent=2))
    print("PIPELINE COMPLETE")
    print("  " + "  ".join(f"{k}={v}s" for k, v in timings.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
