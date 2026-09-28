"""Regression tests for the Phase 9/10 pair-feature builder.

Follows the plain check()/FAILS convention used by tests/test_normalize.py
and tests/test_scoring.py (no pytest dependency).
Run: .venv/bin/python -m tests.test_pair_features
"""
import shutil
import sys
import tempfile
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prep import pair_features as PF  # noqa: E402

FAILS = []


def check(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILS.append(msg)


def close(a, b, tol=1e-6):
    return abs(a - b) < tol


# ==========================================================================
# PART 1 -- add_pair_features() on a small hand-built "already joined" fixture
# ==========================================================================
print("== building fixture: one row per scenario, expected values worked out by hand ==")

ROWS = {
    "identity": dict(  # trivial control: everything matches exactly
        q_country="US", c_country="US",
        q_business_name="Acme Labs", c_business_name="Acme Labs",
        q_name_norm="acme labs", c_name_norm="acme labs",
        q_addr_norm="1 main st springfield", c_addr_norm="1 main st springfield",
        q_name_script="latin", c_name_script="latin",
        q_addr_script="latin", c_addr_script="latin",
        q_name_tokens=["acme", "labs"], c_name_tokens=["acme", "labs"],
        q_addr_tokens=["1", "main", "st", "springfield"], c_addr_tokens=["1", "main", "st", "springfield"],
        q_legal_suffix=[], c_legal_suffix=[],
        q_addr_pin="12345", c_addr_pin="12345",
        q_addr_numbers=["1"], c_addr_numbers=["1"],
    ),
    "transposed": dict(  # H: word-order transposition must NOT hurt token jaccard
        q_country="US", c_country="US",
        q_business_name="Atlantic Research Ventures-Revere", c_business_name="Atlantic Ventures-Revere Research",
        q_name_norm="atlantic research ventures revere", c_name_norm="atlantic ventures revere research",
        q_addr_norm="", c_addr_norm="",
        q_name_script="latin", c_name_script="latin",
        q_addr_script="latin", c_addr_script="latin",
        q_name_tokens=["atlantic", "research", "ventures", "revere"],
        c_name_tokens=["atlantic", "ventures", "revere", "research"],
        q_addr_tokens=[], c_addr_tokens=[],
        q_legal_suffix=[], c_legal_suffix=[],
        q_addr_pin="", c_addr_pin="",
        q_addr_numbers=[], c_addr_numbers=[],
    ),
    "cross_script": dict(  # name script differs but address (Latin) rescues it
        q_country="India", c_country="India",
        q_business_name="Al Investment LLP", c_business_name="अल इन्वेस्टमेंट एलएलपी",
        q_name_norm="al investment llp", c_name_norm="अल इन्वेस्टमेंट एलएलपी",
        q_addr_norm="123 mg road bangalore", c_addr_norm="123 mg road bangalore",
        q_name_script="latin", c_name_script="deva",
        q_addr_script="latin", c_addr_script="latin",
        q_name_tokens=["al", "investment", "llp"],
        c_name_tokens=["अल", "इन्वेस्टमेंट", "एलएलपी", "al", "investment", "llp"],  # translit unioned in, per D6
        q_addr_tokens=["123", "mg", "road", "bangalore"], c_addr_tokens=["123", "mg", "road", "bangalore"],
        q_legal_suffix=["llp"], c_legal_suffix=["llp"],
        q_addr_pin="", c_addr_pin="",
        q_addr_numbers=["123"], c_addr_numbers=["123"],
    ),
    "empty_addr": dict(  # candidate side has an address, query does not
        q_country="US", c_country="US",
        q_business_name="Prime Money", c_business_name="Prime Money LLC",
        q_name_norm="prime money", c_name_norm="prime money llc",
        q_addr_norm="", c_addr_norm="17560 ellis road tahlequah ok",
        q_name_script="latin", c_name_script="latin",
        q_addr_script="latin", c_addr_script="latin",
        q_name_tokens=["prime", "money"], c_name_tokens=["prime", "money", "llc"],
        q_addr_tokens=[], c_addr_tokens=["17560", "ellis", "road", "tahlequah", "ok"],
        q_legal_suffix=[], c_legal_suffix=["llc"],
        q_addr_pin="", c_addr_pin="17560",
        q_addr_numbers=[], c_addr_numbers=["17560"],
    ),
    "noise_injected": dict(  # candidate has query's tokens PLUS injected junk (#65459, suffix)
        q_country="US", c_country="US",
        q_business_name="Acme Laboratories", c_business_name="Acme Laboratories Private Limited #65459",
        q_name_norm="acme laboratories", c_name_norm="acme laboratories private limited 65459",
        q_addr_norm="9 oak ave", c_addr_norm="9 oak ave",
        q_name_script="latin", c_name_script="latin",
        q_addr_script="latin", c_addr_script="latin",
        q_name_tokens=["acme", "laboratories"],
        c_name_tokens=["acme", "laboratories", "private", "limited", "65459"],
        q_addr_tokens=["9", "oak", "ave"], c_addr_tokens=["9", "oak", "ave"],
        q_legal_suffix=[], c_legal_suffix=["pvt", "ltd"],
        q_addr_pin="", c_addr_pin="",
        q_addr_numbers=["9"], c_addr_numbers=["9"],
    ),
    "suffix_disjoint": dict(  # both have a suffix, but a different one -- not equal, not intersecting
        q_country="US", c_country="US",
        q_business_name="Bright Co Ltd", c_business_name="Bright Co Pvt",
        q_name_norm="bright co ltd", c_name_norm="bright co pvt",
        q_addr_norm="5 pine rd", c_addr_norm="9 pine rd",
        q_name_script="latin", c_name_script="latin",
        q_addr_script="latin", c_addr_script="latin",
        q_name_tokens=["bright", "co", "ltd"], c_name_tokens=["bright", "co", "pvt"],
        q_addr_tokens=["5", "pine", "rd"], c_addr_tokens=["9", "pine", "rd"],
        q_legal_suffix=["ltd"], c_legal_suffix=["pvt"],
        q_addr_pin="500001", c_addr_pin="500002",
        q_addr_numbers=["5"], c_addr_numbers=["9"],
    ),
    "country_mismatch": dict(  # defensive: should never happen post-H1, but must compute, not crash
        q_country="US", c_country="India",
        q_business_name="Zed Traders", c_business_name="Zed Traders",
        q_name_norm="zed traders", c_name_norm="zed traders",
        q_addr_norm="1 zed way", c_addr_norm="1 zed way",
        q_name_script="latin", c_name_script="latin",
        q_addr_script="latin", c_addr_script="latin",
        q_name_tokens=["zed", "traders"], c_name_tokens=["zed", "traders"],
        q_addr_tokens=["1", "zed", "way"], c_addr_tokens=["1", "zed", "way"],
        q_legal_suffix=[], c_legal_suffix=[],
        q_addr_pin="", c_addr_pin="",
        q_addr_numbers=["1"], c_addr_numbers=["1"],
    ),
}

names = list(ROWS.keys())
fixture = pl.DataFrame(
    {"source1_entity_id": [f"S1-{i}" for i in range(len(names))],
     "candidate_entity_id": [("S2-" if i % 2 == 0 else "S3-") + str(100 + i) for i in range(len(names))]}
).with_columns(
    **{k: pl.Series([ROWS[n][k] for n in names]) for k in ROWS[names[0]]}
)

feats = PF.add_pair_features(fixture)
row = {n: feats.row(i, named=True) for i, n in enumerate(names)}

print("\n== schema ==")
check(set(PF.ID_COLS) <= set(feats.columns), "id columns present")
check(set(PF.FEATURE_COLS) <= set(feats.columns), "all declared feature columns present")
check("retrieval_score" not in feats.columns and "retrieval_rank" not in feats.columns,
      "no retrieval columns when absent from input (else-omitted rule)")
check("label" not in feats.columns, "no label column ever produced by add_pair_features")

print("\n== identity row: exact match on everything ==")
r = row["identity"]
check(r["name_raw_exact"] == 1, "raw exact")
check(r["name_norm_exact"] == 1, "norm exact")
check(close(r["name_tok_jaccard"], 1.0), "name jaccard 1.0")
check(close(r["name_token_set_ratio"], 1.0), "name token_set_ratio 1.0")
check(r["addr_norm_exact"] == 1, "addr norm exact")
check(r["addr_pin_both_present"] == 1 and r["addr_pin_equal"] == 1, "pin both present and equal")
check(r["country_equal"] == 1, "country equal")
check(r["candidate_source"] == "S2", "candidate_source parsed as S2")

print("\n== word-order transposition: token jaccard must be high despite raw mismatch ==")
r = row["transposed"]
check(r["name_raw_exact"] == 0, "raw NOT exact (order differs)")
check(close(r["name_tok_jaccard"], 1.0), f"token jaccard == 1.0 (same token SET), got {r['name_tok_jaccard']}")
check(close(r["name_token_set_ratio"], 1.0), "rapidfuzz token_set_ratio == 1.0 (order-insensitive)")
check(r["q_addr_empty"] == 1 and r["c_addr_empty"] == 1 and r["either_addr_empty"] == 1,
      "both addresses empty -> all three empty-flags fire")
check(close(r["addr_tok_jaccard"], 0.0), "empty-vs-empty address jaccard defined as 0.0, not NaN/1.0")

print("\n== cross-script pair ==")
r = row["cross_script"]
check(r["name_cross_script"] == 1, "name_cross_script fires (latin vs deva)")
check(r["addr_cross_script"] == 0, "addr_cross_script does NOT fire (both latin)")
check(r["name_script_pair"] == "latin__deva", f"name_script_pair == 'latin__deva', got {r['name_script_pair']!r}")
check(r["name_tok_jaccard"] > 0.3, f"unioned-translit tokens still overlap substantially, got {r['name_tok_jaccard']}")
check(r["addr_norm_exact"] == 1, "address (Latin on both sides) still exact -- the channel that rescues cross-script pairs")
check(r["suffix_sets_equal"] == 1 and r["suffix_sets_intersect"] == 1, "shared llp suffix agrees across scripts")

print("\n== empty-address pair ==")
r = row["empty_addr"]
check(r["q_addr_empty"] == 1 and r["c_addr_empty"] == 0, "only query side empty")
check(r["either_addr_empty"] == 1, "either_addr_empty fires")
check(close(r["addr_tok_jaccard"], 0.0), "no address tokens to overlap -> 0.0, not NaN")
check(r["addr_pin_both_present"] == 0, "pin not present on both (query has none)")
check(r["addr_pin_equal"] == 0, "pin equal is 0 when not both present (distinct from both_present)")

print("\n== noise-token injection: asymmetric containment must show it, jaccard alone would not ==")
r = row["noise_injected"]
check(close(r["name_tok_contain_q_in_c"], 1.0),
      f"ALL query tokens found inside the noisy candidate -> containment(q in c) == 1.0, got {r['name_tok_contain_q_in_c']}")
check(r["name_tok_contain_c_in_q"] < 0.5,
      f"candidate has lots of tokens absent from the query -> containment(c in q) low, got {r['name_tok_contain_c_in_q']}")
check(r["name_tok_contain_q_in_c"] > r["name_tok_jaccard"],
      "containment(q in c) is a stronger/less-punished signal than plain jaccard under noise injection")
check(r["suffix_count_q"] == 0 and r["suffix_count_c"] == 2, "suffix extracted from candidate's injected junk, none on query")

print("\n== suffix disjoint (both present, different classes) ==")
r = row["suffix_disjoint"]
check(r["suffix_sets_equal"] == 0, "not equal")
check(r["suffix_sets_intersect"] == 0, "not intersecting")
check(close(r["suffix_jaccard"], 0.0), "jaccard 0.0")
check(r["suffix_count_q"] == 1 and r["suffix_count_c"] == 1, "one suffix each")
check(r["addr_pin_both_present"] == 1 and r["addr_pin_equal"] == 0, "pins present but differ")
check(r["addr_num_shared_count"] == 0, "no shared address numbers (5 vs 9)")

print("\n== defensive: country mismatch still computes cleanly (no crash, no NaN) ==")
r = row["country_mismatch"]
check(r["country_equal"] == 0, "country_equal correctly 0")
check(close(r["name_tok_jaccard"], 1.0), "name still compares fine independent of country")

print("\n== no NaN/inf anywhere in the whole fixture ==")
try:
    PF.assert_features_sane(feats)
    check(True, "assert_features_sane passes on the full fixture")
except AssertionError as e:
    check(False, f"assert_features_sane raised: {e}")

# ==========================================================================
# PART 2 -- retrieval score/rank pass-through: present -> kept, absent -> omitted
# ==========================================================================
print("\n== retrieval score/rank: present in input -> present in output ==")
with_retrieval = fixture.with_columns(
    pl.Series("retrieval_score", [float(i) for i in range(len(names))]),
    pl.Series("retrieval_rank", list(range(len(names)))),
)
feats_r = PF.add_pair_features(with_retrieval)
check("retrieval_score" in feats_r.columns and "retrieval_rank" in feats_r.columns,
      "retrieval columns carried through when present on the input")
check(feats_r.drop(["retrieval_score", "retrieval_rank"]).equals(feats),
      "presence of retrieval columns does not change any other feature value")

# ==========================================================================
# PART 3 -- end-to-end run(): chunking, sharding, and label-noninterference
# ==========================================================================
print("\n== end-to-end run(): tiny synthetic corpus, chunk_size smaller than the data ==")

tmp = Path(tempfile.mkdtemp(prefix="pair_features_test_"))
try:
    repr_dir = tmp / "representations"
    repr_dir.mkdir()

    def make_repr(entity_ids, names_, addrs, country="US"):
        rows = []
        for eid, nm, ad in zip(entity_ids, names_, addrs):
            from src.prep.normalize import (addr_numbers, addr_pin, extract_suffixes,
                                             name_core, normalize, script_of, tokens_with_translit)
            nn = normalize(nm)
            an = normalize(ad)
            ntoks = tokens_with_translit(nn)
            atoks = tokens_with_translit(an)
            rows.append(dict(
                entity_id=eid, country=country, business_name=nm, business_address=ad,
                name_norm=nn, addr_norm=an,
                name_script=script_of(nm), addr_script=script_of(ad),
                name_tokens=ntoks, addr_tokens=atoks,
                legal_suffix=extract_suffixes(ntoks), name_core=name_core(ntoks),
                addr_pin=addr_pin(ad), addr_numbers=addr_numbers(ad),
            ))
        return pl.DataFrame(rows)

    s1_ids = [f"S1-{i}" for i in range(6)]
    s1_names = ["Acme Labs", "Bright Co Ltd", "Zed Traders", "Prime Money",
                "Atlantic Research Ventures", "Delta Inc"]
    s1_addrs = ["1 Main St", "5 Pine Rd", "1 Zed Way", "17560 Ellis Road",
                "10 Harbor Blvd", "914 Pierpont Ave"]
    make_repr(s1_ids, s1_names, s1_addrs).write_parquet(repr_dir / "train_source1.parquet")

    s2_ids = [f"S2-{i}" for i in range(6)]
    s2_names = ["Acme Labs", "Bright Co Pvt", "Zed Traders", "Prime Money LLC",
                "Atlantic Ventures Research", "Nothing Matching Co"]
    s2_addrs = ["1 Main St", "9 Pine Rd", "1 Zed Way", "17560 Ellis Road",
                "10 Harbor Blvd", "0 Nowhere"]
    make_repr(s2_ids, s2_names, s2_addrs).write_parquet(repr_dir / "train_source2.parquet")

    s3_ids = [f"S3-{i}" for i in range(2)]
    make_repr(s3_ids, ["Unrelated Corp", "Another Unrelated LLC"], ["1 Away St", "2 Away St"]).write_parquet(
        repr_dir / "train_source3.parquet")

    cand_tsv = tmp / "candidates.tsv"
    all_cands = s2_ids + s3_ids
    with open(cand_tsv, "w") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1_ids:
            f.write(f"{sid}\t{','.join(all_cands)}\n")

    gt = pl.DataFrame({
        "source1_entity_id": s1_ids,
        "matched_entity_ids": ["S2-0", "S2-1", "S2-2", "S2-3", "S2-4", ""],
    })
    gt_path = tmp / "gt.parquet"
    gt.write_parquet(gt_path)

    out_with = tmp / "out_with_label"
    out_without = tmp / "out_without_label"

    report_with = PF.run(
        candidates_path=cand_tsv, split="train", out_dir=out_with,
        chunk_size=2,  # force multiple shards: 6 s1 rows / chunk 2 = 3 shards
        ground_truth_path=gt_path, include_label=True, repr_dir=repr_dir,
    )
    report_without = PF.run(
        candidates_path=cand_tsv, split="train", out_dir=out_without,
        chunk_size=2, ground_truth_path=gt_path, include_label=False, repr_dir=repr_dir,
    )

    check(report_with["n_shards"] == 3, f"3 shards written (6 s1 rows / chunk_size 2), got {report_with['n_shards']}")
    check(report_with["n_pairs"] == 6 * 8, f"6 queries x 8 candidates = 48 pairs, got {report_with['n_pairs']}")
    check(report_with["label_positive_pairs"] == 5, f"5 true matches (S1-0..4 each match S2-0..4), got {report_with['label_positive_pairs']}")
    check(report_without["label_positive_pairs"] is None, "no label computed when include_label=False")

    with_df = pl.concat([pl.read_parquet(p) for p in sorted(out_with.glob("*.parquet"))]).sort(
        ["source1_entity_id", "candidate_entity_id"])
    without_df = pl.concat([pl.read_parquet(p) for p in sorted(out_without.glob("*.parquet"))]).sort(
        ["source1_entity_id", "candidate_entity_id"])

    check("label" in with_df.columns, "label column present when requested")
    check("label" not in without_df.columns, "label column absent when not requested")
    check(with_df.drop("label").equals(without_df),
          "identical feature columns whether or not the label was requested")

    check(int(with_df.filter(pl.col("source1_entity_id") == "S1-0").filter(pl.col("candidate_entity_id") == "S2-0")["label"][0]) == 1,
          "true match S1-0/S2-0 labelled 1")
    check(int(with_df.filter(pl.col("source1_entity_id") == "S1-0").filter(pl.col("candidate_entity_id") == "S3-0")["label"][0]) == 0,
          "non-match S1-0/S3-0 labelled 0")
    check(int(with_df.filter(pl.col("source1_entity_id") == "S1-5").filter(pl.col("candidate_entity_id") == "S2-0")["label"][0]) == 0,
          "singleton S1-5 has label 0 against everything")

    try:
        PF.assert_features_sane(with_df.drop("label"))
        check(True, "no NaN/inf anywhere in the full end-to-end output")
    except AssertionError as e:
        check(False, f"end-to-end output failed sanity check: {e}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
if FAILS:
    print(f"{len(FAILS)} FAILURE(S)")
    for f in FAILS:
        print("   -", f)
    sys.exit(1)
print("ALL PAIR-FEATURE TESTS PASSED")
