"""Blocking recall harness.  python -m src.blocking.harness [--n 5000] [--blockers prefix4,token,addrnum,union]
Anchors: seeded (seed=1, differs from EDA seed 0) random sample of MATCHED train anchors, since
src/folds.py did not exist at write time. Swap `sample_anchors` to folds.val split when it lands.
Reports: pair recall, cluster-complete recall (all targets found), any-hit recall, candidate-set size
(mean/median/p95 per anchor) -- overall, by country, by target script (latin/nonlatin), by source (S2/S3)."""
import argparse, time, numpy as np, pandas as pd, sys
from . import norm, blockers as B

def sample_anchors(n, seed=1):
    gt = pd.read_parquet(norm.I + 'gt.parquet'); gt = gt[gt.matched_entity_ids != ''].sample(n, random_state=seed)
    return gt.reset_index(drop=True)

def truth_pairs(gt, anchors_all_idx, pool):
    e = gt.assign(t=gt.matched_entity_ids.str.split(',')).explode('t')
    e['a'] = e.source1_entity_id.map(anchors_all_idx).values
    e['p'] = e.t.map(pd.Series(np.arange(len(pool)), index=pool.id.values)).values
    return e[['a', 'p']].astype(np.int64)

def evaluate(blocker, anchors, pool, tp, label=''):
    t = time.time(); a, p = blocker.query(anchors); dt = time.time() - t
    cand = pd.DataFrame({'a': a, 'p': p}); cand['hit'] = 1
    tp = tp.merge(cand, on=['a', 'p'], how='left'); tp['hit'] = tp.hit.fillna(0)
    tp['country'] = anchors.country.values[tp.a.values]; tp['script'] = pool.script.values[tp.p.values]
    tp['src'] = pool.id.values[tp.p.values].astype('U2')
    def rep(g): return pd.Series({'pairs': len(g), 'pair_recall': g.hit.mean(),
                                  'cluster_full': g.groupby('a').hit.min().mean(), 'any_hit': g.groupby('a').hit.max().mean()})
    out = {'overall': rep(tp)}
    for c, g in tp.groupby('country'): out['country=' + c] = rep(g)
    for c, g in tp.groupby('script'): out['script=' + c] = rep(g)
    for c, g in tp.groupby('src'): out['src=' + c] = rep(g)
    for c, g in tp[(tp.country == 'India')].groupby('script'): out['India,script=' + c] = rep(g)
    sz = np.bincount(a, minlength=len(anchors))
    res = pd.DataFrame(out).T
    print(f'\n=== {blocker.name} {label} | query {dt:.1f}s | cand/anchor mean {sz.mean():.0f} median {np.median(sz):.0f} p95 {np.quantile(sz,.95):.0f} p99 {np.quantile(sz,.99):.0f} max {sz.max()}')
    print(res.round(4).to_string()); return res, sz

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--blockers', default='prefix4,token,addrnum,union'); a = ap.parse_args()
    pool = norm.load_pool(); s1 = norm.load_anchors()
    idx = pd.Series(np.arange(len(s1)), index=s1.id.values)
    gt = sample_anchors(a.n); anchors = s1.iloc[idx[gt.source1_entity_id].values].reset_index(drop=True)
    an_idx = pd.Series(np.arange(len(anchors)), index=anchors.id.values)
    tp = truth_pairs(gt, an_idx, pool)
    reg = {'prefix4': lambda: B.Prefix(4), 'prefix3': lambda: B.Prefix(3), 'token': lambda: B.Token(5000),
           'addrnum': lambda: B.AddrNum(3000, 2),
           'union': lambda: B.Union(B.Prefix(4), B.Token(5000), B.AddrNum(3000, 2))}
    for nm in a.blockers.split(','):
        t = time.time(); b = reg[nm]().fit(pool); print(f'[{nm}] fit {time.time()-t:.0f}s'); evaluate(b, anchors, pool, tp)

if __name__ == '__main__': main()
