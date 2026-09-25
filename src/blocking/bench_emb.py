"""Embedding-blocker benchmark on VALIDATION anchors (split.parquet is_val).
python -m src.blocking.bench_emb --n 2000 --frac 0.04 --models intfloat/multilingual-e5-small,intfloat/multilingual-e5-base
Sub-pool = ALL true targets of the sampled anchors + random `frac` of the rest of the pool (per country, derived from data).
=> recall is OPTIMISTIC (fewer distractors than the full pool). Cache: data/interim/emb_bench/ (git-ignored)."""
import argparse, os, time, resource, numpy as np, pandas as pd
from . import norm, blockers as B, harness as H

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=2000); ap.add_argument('--frac', type=float, default=0.04)
    ap.add_argument('--models', default='intfloat/multilingual-e5-small,intfloat/multilingual-e5-base'); ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--k', type=int, default=50); a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    sp = pd.read_parquet(norm.I + 'split.parquet'); val = set(sp[sp.is_val].source1_entity_id)
    gt = pd.read_parquet(norm.I + 'gt.parquet'); gt = gt[(gt.matched_entity_ids != '') & gt.source1_entity_id.isin(val)].sample(a.n, random_state=a.seed).reset_index(drop=True)
    s1 = pd.read_parquet(norm.I + 'source1.parquet'); s1 = s1.set_index('entity_id').loc[gt.source1_entity_id].reset_index()
    anchors = norm.prep(s1); anchors['text'] = norm.embed_text(s1.business_name, s1.business_address)
    raw = pd.concat([pd.read_parquet(norm.I + f'source{i}.parquet') for i in (2, 3)], ignore_index=True)
    pool = norm.load_pool(); assert (pool.id.values == raw.entity_id.values).all()
    tids = set(gt.matched_entity_ids.str.split(',').explode())
    is_t = pool.id.isin(tids).values
    keep = is_t | (rng.random(len(pool)) < a.frac)
    # sample fraction restricted to non-target distractors
    print(f'anchors {len(anchors)} by country {anchors.country.value_counts().to_dict()}; true targets {is_t.sum()}; sub-pool {keep.sum()} of {len(pool)} ({keep.mean():.3%}); distractor frac {a.frac}')
    sub = pool[keep].reset_index(drop=True); sub['text'] = norm.embed_text(raw.business_name[keep], raw.business_address[keep])
    print('sub-pool by country', sub.country.value_counts().to_dict(), 'by script', sub.groupby(['country', 'script']).size().to_dict())
    idx = pd.Series(np.arange(len(anchors)), index=anchors.id.values); tp = H.truth_pairs(gt, idx, sub)
    assert tp.p.notna().all() and tp.a.notna().all()
    print('sample text:', anchors.text[0], '||', sub.text[0])
    for m in a.models.split(','):
        b = B.EmbANN(m, topk=a.k).fit(sub); H.evaluate(b, anchors, sub, tp, label=f'(sub-pool {len(sub)}, optimistic)')
        print(m, b.stats, f'RAM maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f} GB')
        del b; import torch, gc; gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__': main()
