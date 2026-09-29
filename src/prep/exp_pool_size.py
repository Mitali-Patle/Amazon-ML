"""EXPERIMENT -- does the validation pool SIZE change measured recall?

The playbook (D12) specified a validation pool matched to the test S1:(S2+S3)
RATIO of 1:5.75. That controls composition but not difficulty: a true match's
rank is driven by how many distractors outscore it, which scales with the
ABSOLUTE pool size. The ratio-matched pool holds 634k records; test searches
9.97M. If the ratio pool measures materially higher recall, D12 is wrong and
validation must instead search the full corpus (10.32M, ~= the test pool size).

Same queries, same weights, same code path -- only the pool differs.
"""
from __future__ import annotations

import json

from .candidates import write_report
from .config import K_GRID, REPORTS
from .run_validation import run

N_QUERIES = 8000

if __name__ == "__main__":
    out = {}
    for mode in ("ratio", "full"):
        print(f"\n===== pool_mode = {mode} =====")
        out[mode] = run(pool_mode=mode, query_limit=N_QUERIES,
                        save_candidates=False, tag=f"exp_pool_{mode}")

    print("\n\n===== COMPARISON (same queries) =====")
    for country in sorted(out["full"]["by_country"]):
        r = out["ratio"]["by_country"][country]
        f = out["full"]["by_country"][country]
        print(f"\n  {country}: pool {r['pool_records']:,} (ratio) vs {f['pool_records']:,} (full)")
        print(f"    {'K':>5} {'ratio-pool':>12} {'full-pool':>12} {'inflation':>11}")
        for k in K_GRID:
            a = r["pair_recall"][f"@{k}"]
            b = f["pair_recall"][f"@{k}"]
            print(f"    {k:>5} {a:>11.2f}% {b:>11.2f}% {a - b:>+10.2f}pp")
        print(f"    mean candidates/query: {r['mean_candidates']:,.0f} -> {f['mean_candidates']:,.0f}")

    write_report("exp_pool_size_comparison", {
        "n_queries": N_QUERIES,
        "comparison": {
            c: {
                "ratio_pool_records": out["ratio"]["by_country"][c]["pool_records"],
                "full_pool_records": out["full"]["by_country"][c]["pool_records"],
                "ratio_pair_recall": out["ratio"]["by_country"][c]["pair_recall"],
                "full_pair_recall": out["full"]["by_country"][c]["pair_recall"],
                "inflation_pp": {
                    f"@{k}": round(out["ratio"]["by_country"][c]["pair_recall"][f"@{k}"]
                                   - out["full"]["by_country"][c]["pair_recall"][f"@{k}"], 2)
                    for k in K_GRID
                },
            } for c in out["full"]["by_country"]
        },
    })
