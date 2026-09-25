"""Holdout re-check of FROZEN blocking configs on a 2nd disjoint val-anchor sample (train data only, no tuning).
  python -m src.blocking.recheck gen [--n 10000] [--seed 23]   # sample, dense search+expansion, lexical -> data/interim/full_pool/recheck/
  python -m src.blocking.recheck eval                          # metrics for sample1 and sample2, CIs, misses dump
Sample 2 = is_val anchors NOT in sample 1 (val_anchors.parquet ids), stratified proportionally by has_match x country."""
import argparse, os, time
import numpy as np, pandas as pd, pyarrow.parquet as pq
from . import norm
from . import full_pool as F

D2 = F.D + 'recheck/'; K = F.K
CFGS = {'D70+E20+L60': (70, 20, 60), 'D50+E20+L40': (50, 20, 40)}

def sample2(n, seed):
    sp = pd.read_parquet(norm.I + 'split.parquet'); v = sp[sp.is_val]
    used = set(pd.read_parquet(F.D + 'val_anchors.parquet').id)
    v = v[~v.source1_entity_id.isin(used)]          # exclusion by id
    assert len(used) == 5000 and not (set(v.source1_entity_id) & used)
    parts = []; tot = len(v)
    for (c, hm), g in v.groupby(['country', 'has_match']):
        k = int(round(n * len(g) / tot)); parts.append(g.sample(k, random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

def gen(n, seed, seeds_exp=3, nb=11):
    import torch
    from sentence_transformers import SentenceTransformer
    from . import blockers as B
    t0 = time.time(); os.makedirs(D2, exist_ok=True); pick = sample2(n, seed)
    print('sample2', len(pick), pick.groupby(['country', 'has_match']).size().to_dict(), flush=True)
    s1 = pq.read_table(norm.I + 'source1.parquet').to_pandas().set_index('entity_id').loc[pick.source1_entity_id].reset_index()
    assert (s1.entity_id.values == pick.source1_entity_id.values).all() and (s1.country.values == pick.country.values).all()
    an = norm.prep(s1); an['has_match'] = pick.has_match.values; txt = norm.embed_text(s1.business_name, s1.business_address)
    m = SentenceTransformer(F.MODEL, device='cuda', model_kwargs={'torch_dtype': 'float16'}); m.max_seq_length = 64
    Q = m.encode(['query: ' + t for t in txt], batch_size=512, normalize_embeddings=True, convert_to_numpy=True).astype(np.float16)
    assert np.isfinite(Q).all(); del m; torch.cuda.empty_cache()
    an.to_parquet(D2 + 'val_anchors.parquet'); np.save(D2 + 'val_Q.npy', Q)
    cpos = np.full((len(an), K), -1, np.int32); csc = np.full((len(an), K), -np.inf, np.float32)
    epos = np.full((len(an), seeds_exp * (nb - 1)), -1, np.int32)
    for c in sorted(an.country.unique()):
        pos, mm = F.open_country(c); ai = np.flatnonzero(an.country.values == c)
        r, v = F.topk_stream(torch.from_numpy(Q[ai]).cuda(), mm); cpos[ai] = pos[r]; csc[ai] = v
        seed_rows = r[:, :seeds_exp].ravel(); srt = np.argsort(seed_rows); inv = np.empty_like(srt); inv[srt] = np.arange(len(srt))
        S = torch.from_numpy(np.ascontiguousarray(mm[np.sort(seed_rows)])).cuda()
        nr, _ = F.topk_stream(S[torch.from_numpy(inv).cuda()], mm, k=nb)
        nr = nr.reshape(len(ai), seeds_exp, nb); keep = nr != r[:, :seeds_exp, None]
        out = np.full((len(ai), seeds_exp, nb - 1), -1, np.int64)
        for a_ in range(len(ai)):
            for s_ in range(seeds_exp):
                x = nr[a_, s_][keep[a_, s_]][:nb - 1]; out[a_, s_, :len(x)] = x
        e = out.reshape(len(ai), -1); epos[ai] = np.where(e >= 0, pos[np.clip(e, 0, None)], -1)
        print(f'[{c}] {len(ai)} anchors {time.time()-t0:.0f}s', flush=True)
    assert (cpos >= 0).all() and np.isfinite(csc).all()
    np.savez(D2 + 'val_dense.npz', cpos=cpos, csc=csc, epos=epos)
    pool = norm.load_pool(); out = {}
    for nm, b in [('addrnum2', B.AddrNum(3000, 2)), ('addrnum0', B.AddrNum(3000, 0)), ('prefix4c', B.Prefix(4, cap=2000)), ('tokc300', B.Token(300))]:
        b.fit(pool); a, p = b.query(an); out[nm + '_a'] = a.astype(np.int32); out[nm + '_p'] = p.astype(np.int32); del b
        print(f'[lex {nm}] {len(a)/len(an):.0f}/anchor {time.time()-t0:.0f}s', flush=True)
    np.savez(D2 + 'val_lex.npz', **out); print('GEN DONE', time.time() - t0)

def load_sample(d):
    """returns anchors, Q, dense npz, lex npz"""
    return pd.read_parquet(d + 'val_anchors.parquet'), np.load(d + 'val_Q.npy'), np.load(d + 'val_dense.npz'), np.load(d + 'val_lex.npz')

def candidates(d, pool, cfgs):
    an, Q, dn, lx = load_sample(d); nA = len(an); C = an.country.values; cache = {}
    cp, cs = dn['cpos'], dn['csc']
    D_ = pd.DataFrame({'a': np.repeat(np.arange(nA), K), 'p': cp.ravel().astype(np.int64), 'rank': np.tile(np.arange(K), nA)})
    ep = dn['epos']; E_ = pd.DataFrame({'a': np.repeat(np.arange(nA), ep.shape[1]), 'p': ep.ravel().astype(np.int64)}); E_ = E_[E_.p >= 0].drop_duplicates(['a', 'p'])
    E_['score'] = F._dense_scores(E_.a.values, E_.p.values, Q, None, C, cache)
    ls = []
    for nm in ['addrnum2', 'addrnum0', 'prefix4c', 'tokc300']:
        x = pd.DataFrame({'a': lx[nm + '_a'].astype(np.int64), 'p': lx[nm + '_p'].astype(np.int64)}).drop_duplicates(); x['v'] = 1; ls.append(x)
    L = pd.concat(ls).groupby(['a', 'p'], as_index=False).v.sum().rename(columns={'v': 'votes'})
    L['score'] = F._dense_scores(L.a.values, L.p.values, Q, None, C, cache); L['rk'] = L.votes + L.score
    res = {}
    for tag, (Kd, Me, Ml) in cfgs.items():
        dd = D_[D_['rank'] < Kd][['a', 'p']]; ex = F._keys(dd.a.values, dd.p.values); parts = [dd]
        x = F._topm_excl(E_, Me, ex, 'score'); parts.append(x[['a', 'p']]); ex = np.concatenate([ex, F._keys(x.a.values, x.p.values)])
        y = F._topm_excl(L, Ml, ex, 'rk'); parts.append(y[['a', 'p']])
        c = pd.concat(parts); assert not c.duplicated(['a', 'p']).any(); res[tag] = c
    return an, cs, res

def truth(an, pool):
    gt = pd.read_parquet(norm.I + 'gt.parquet').set_index('source1_entity_id').loc[an.id.values].matched_entity_ids
    assert (an.has_match.values == (gt.values != '')).all()
    e = pd.DataFrame({'a': np.arange(len(an)), 't': gt.str.split(',').values}).explode('t'); e = e[e.t.notna() & (e.t != '')]
    e['p'] = pd.Index(pool.id.values).get_indexer(e.t.values); assert (e.p >= 0).all()
    tp = e[['a', 'p']].astype(np.int64).reset_index(drop=True)
    tp['country'] = an.country.values[tp.a]; tp['script'] = pool.script.values[tp.p]; tp['src'] = pool.id.values[tp.p].astype('U2')
    tp['nm'] = tp.groupby('a').a.transform('size'); tp['nm_bin'] = pd.cut(tp.nm, [0, 1, 3, 10**9], labels=['1', '2-3', '4+']).astype(str)
    return tp

def boot(t, nA_boot=400, seed=0):
    """cluster bootstrap over anchors. t has cols a, hit. returns dict metric -> (est, lo, hi)."""
    g = t.groupby('a').hit.agg(['sum', 'size', 'max', 'min']); ids = g.index.values
    S, N, mx, mn = g['sum'].values, g['size'].values, g['max'].values, g['min'].values
    rng = np.random.default_rng(seed); n = len(g); out = {'pair_R': [], 'any_hit': [], 'cluster_full': []}
    for _ in range(nA_boot):
        i = rng.integers(0, n, n); out['pair_R'].append(S[i].sum() / N[i].sum()); out['any_hit'].append(mx[i].mean()); out['cluster_full'].append(mn[i].mean())
    est = {'pair_R': S.sum() / N.sum(), 'any_hit': mx.mean(), 'cluster_full': mn.mean()}
    return {k: (est[k], *np.percentile(out[k], [2.5, 97.5])) for k in out}

def fmt(b, k): return f'{100*b[k][0]:.2f} [{100*b[k][1]:.2f},{100*b[k][2]:.2f}]'

def evaluate():
    pool = norm.load_pool(); pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_colwidth', 45)
    summ = []; miss_dump = None
    for sname, d in [('sample1(5k,tuned)', F.D), ('sample2(holdout)', D2)]:
        an, cs, res = candidates(d, pool, CFGS); tp = truth(an, pool); tk = F._keys(tp.a.values, tp.p.values); nA = len(an)
        # name-noise proxy: any true target with exact normalized-name equality to anchor name
        tp['exact_nm'] = np.where(pool.nn.values[tp.p.values] == an.nn.values[tp.a.values], 'exact_name', 'diff_name')
        print(f'\n######## {sname}: anchors {nA} (matched {an.has_match.sum()}, singleton {(~an.has_match).sum()}), true pairs {len(tp)}, {an.country.value_counts().to_dict()}')
        for tag, cand in res.items():
            hit = np.isin(tk, F._keys(cand.a.values, cand.p.values)); t = tp.assign(hit=hit.astype(float)); sz = np.bincount(cand.a.values, minlength=nA)
            single = ~an.has_match.values; b = boot(t)
            print(f'\n== {sname} {tag}: cands mean {sz.mean():.1f} max {sz.max()} | matched {sz[~single].mean():.1f} | singleton mean {sz[single].mean():.1f} min {sz[single].min()} max {sz[single].max()}')
            print(f'   pair_R {fmt(b,"pair_R")}  any_hit {fmt(b,"any_hit")}  cluster {fmt(b,"cluster_full")}')
            summ.append(dict(sample=sname, cfg=tag, cand_mean=sz.mean(), cand_max=sz.max(), single_mean=sz[single].mean(), **{k: b[k][0] * 100 for k in b}, pair_lo=b['pair_R'][1] * 100, pair_hi=b['pair_R'][2] * 100))
            groups = [('country', lambda x: x.country), ('script', lambda x: x.script), ('src', lambda x: x.src), ('nmatch', lambda x: x.nm_bin), ('name', lambda x: x.exact_nm),
                      ('India,script', lambda x: np.where(x.country == 'India', 'India-' + x.script, 'other')), ('US,script', lambda x: np.where(x.country == 'US', 'US-' + x.script, 'other'))]
            rows = {}
            for gn, f in groups:
                key = np.asarray(f(t))
                for v in np.unique(key):
                    if v == 'other': continue
                    m = key == v; bb = boot(t[m], 200); rows[f'{gn}={v}'] = {'pairs': int(m.sum()), 'anchors': t[m].a.nunique(), 'pair_R': fmt(bb, 'pair_R'), 'cluster': fmt(bb, 'cluster_full'), 'any_hit': fmt(bb, 'any_hit')}
            print(pd.DataFrame(rows).T.to_string())
            if sname.startswith('sample2') and tag == 'D70+E20+L60': miss_dump = (t, an, cand)
    print('\n=== summary'); S = pd.DataFrame(summ); print(S.round(2).to_string(index=False)); S.to_csv(D2 + 'recheck_summary.csv', index=False)
    # ---- misses dump (frozen best config, sample 2): stratified pick of 30 misses proportional to (country, script, src) cells
    t, an, cand = miss_dump; ms = t[t.hit == 0].copy(); print('\nmisses in sample2 D70+E20+L60:', len(ms))
    ms['cell'] = ms.country + '|' + ms.script + '|' + ms.src
    w = ms.cell.value_counts(); alloc = (w / w.sum() * 30).round().astype(int).clip(lower=1)
    while alloc.sum() > 30: alloc[alloc.idxmax()] -= 1
    sel = pd.concat([ms[ms.cell == c].sample(min(k, (ms.cell == c).sum()), random_state=1) for c, k in alloc.items()])
    s1 = pq.read_table(norm.I + 'source1.parquet').to_pandas().set_index('entity_id')
    raw = F.load_raw_cols().to_pandas().set_index('entity_id')
    aid = an.id.values[sel.a.values]; tid = pool.id.values[sel.p.values]
    out = pd.DataFrame({'anchor_id': aid, 'anchor_country': sel.country.values, 'anchor_name': s1.loc[aid].business_name.values, 'anchor_addr': s1.loc[aid].business_address.values,
                        'missed_id': tid, 'missed_src': sel.src.values, 'missed_script': sel.script.values, 'missed_name': raw.loc[tid].business_name.values, 'missed_addr': raw.loc[tid].business_address.values,
                        'n_matches': sel.nm.values, 'exact_name': sel.exact_nm.values})
    os.makedirs('logs', exist_ok=True); out.to_csv('logs/blocking_misses.tsv', sep='\t', index=False); print(out[['anchor_name', 'missed_name', 'missed_src', 'missed_script', 'n_matches']].head(30).to_string())

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('--n', type=int, default=10000); ap.add_argument('--seed', type=int, default=23); a = ap.parse_args()
    gen(a.n, a.seed) if a.stage == 'gen' else evaluate()
