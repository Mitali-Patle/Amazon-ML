"""Full-pool dense blocking (train only). Adds new code; does not modify EmbANN / bench_emb.
Stages:
  python -m src.blocking.full_pool encode [--chunk 100000] [--limit N]   # per-country fp16 memmap, resumable
  python -m src.blocking.full_pool search [--n 5000]                    # exact top-K val anchors vs FULL same-country pool
Artifacts (git-ignored) in data/interim/full_pool/:
  emb_<country>.f16   (n_c, 384) float16 memmap, rows ordered by ascending pool position
  pos_<country>.npy   int32 pool positions (row i of the memmap -> row of pool_norm.parquet = concat(S2, S3))
  progress_<country>.json  rows done (resume point)
Pool order == data/interim/pool_norm.parquet (asserted). Countries are derived from the data."""
import argparse, json, os, re, time, resource
import numpy as np, pandas as pd, pyarrow.parquet as pq
from . import norm

D = norm.I + 'full_pool/'
DIM = 384
MODEL = 'intfloat/multilingual-e5-small'

def cslug(c): return re.sub(r'[^A-Za-z0-9]+', '_', c)
def ckey(c):
    """THE canonical country key (used by encode, open_country, search, _dense_scores): whitespace-collapsed + casefolded.
    Punctuation is significant: 'US' and 'U.S.' are DIFFERENT countries (no alias rule); 'India'/'india '/'INDIA' are one."""
    return re.sub(r'\s+', ' ', str(c)).strip().casefold()
MANIFEST = 'countries.json'   # ckey -> file slug, written by encode() in each outdir (train dir predates it: legacy fallback)

def _assign_slugs(labels):
    """labels: raw country labels -> {ckey: slug}. Slug = cslug(most frequent raw label); a hash suffix disambiguates
    keys whose slugs collide (e.g. 'New-York' vs 'New York'). Slugs are asserted unique."""
    import hashlib
    by = {}
    for l in pd.Series(list(labels)).value_counts().index: by.setdefault(ckey(l), l)   # most frequent raw label per key
    out, used = {}, {}
    for k in sorted(by):
        sl = cslug(by[k]) or 'x'
        if sl.casefold() in used: sl = f'{sl}_{hashlib.md5(k.encode()).hexdigest()[:8]}'
        assert sl.casefold() not in used, f'slug collision {sl}'
        used[sl.casefold()] = k; out[k] = sl
    assert len(set(v.casefold() for v in out.values())) == len(out)
    return out

def _slug_for(c, D):
    """file slug for country label c in artifact dir D, or None if absent. Manifest first; legacy (no manifest, train dir)
    falls back to exact then case-insensitive slug match on pos_*.npy files."""
    k = ckey(c); mf = D + MANIFEST
    if os.path.exists(mf): return json.load(open(mf)).get(k)
    if os.path.exists(D + f'pos_{cslug(str(c))}.npy'): return cslug(str(c))
    want = cslug(k).casefold()
    for f in (os.listdir(D) if os.path.isdir(D) else []):
        if f.startswith('pos_') and f.endswith('.npy') and f[4:-4].casefold() == want: return f[4:-4]
    return None

def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

def load_raw_cols():
    """returns (ids, country, name, address) as pyarrow chunked arrays concatenated S2 then S3."""
    ts = [pq.read_table(norm.I + f'source{i}.parquet') for i in (2, 3)]
    import pyarrow as pa
    t = pa.concat_tables(ts)
    return t

def _ids_hash(ids):
    import hashlib
    return hashlib.md5('\x00'.join(map(str, ids)).encode()).hexdigest()

def encode(chunk=100_000, limit=None, bs=512, max_len=64, pool=None, raw=None, encoder=None, outdir=None):
    """pool: DataFrame with id,country (default: train pool_norm); raw: pyarrow Table with entity_id,business_name,
    business_address in the same row order (default: source2+source3). Pass the test S2/S3 pool+raw for inference:
    then `outdir` is REQUIRED and must not be the train dir. Progress files carry n + hash of the pool ids and refuse to
    resume on mismatch. encoder: optional callable(list[str]) -> (n,DIM) array, replaces e5 (tests / CPU)."""
    if pool is not None or raw is not None:
        assert pool is not None and raw is not None, 'pass pool and raw together'
        assert outdir, 'encode(pool=...) requires an explicit outdir (never the train embeddings dir)'
    D = outdir or globals()['D']
    assert pool is None or os.path.abspath(D) != os.path.abspath(globals()['D']), 'refusing to write a custom pool into the train dir'
    D = D if D.endswith('/') else D + '/'
    os.makedirs(D, exist_ok=True)
    if pool is None: pool = norm.load_pool()
    if raw is None: raw = load_raw_cols()
    if encoder is None:
        import torch
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(MODEL, device='cuda', model_kwargs={'torch_dtype': 'float16'}); m.max_seq_length = max_len
        encoder = lambda t: m.encode(t, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    assert raw.num_rows == len(pool) and (raw.column('entity_id').to_numpy(zero_copy_only=False) == pool.id.values).all()
    pk = pool.country.map(ckey).values                       # one canonical key everywhere
    slugs = _assign_slugs(pool.country.values)
    mf = D + MANIFEST
    if os.path.exists(mf):
        old = json.load(open(mf)); assert all(slugs[k] == v for k, v in old.items() if k in slugs), 'country->slug manifest conflict in outdir'
        slugs = {**old, **slugs}; assert len({v.casefold() for v in slugs.values()}) == len(slugs), 'slug collision with existing manifest'
    json.dump(slugs, open(mf + '.tmp', 'w')); os.replace(mf + '.tmp', mf)
    t0 = time.time(); done_total = 0; total = len(pool) if limit is None else min(limit, len(pool))
    for k in sorted(slugs):
        pos = np.flatnonzero(pk == k).astype(np.int32)
        if len(pos) == 0: continue
        if limit: pos = pos[:limit]
        sl = slugs[k]; h = _ids_hash(pool.id.values[pos])
        pf = D + f'progress_{sl}.json'; fn = D + f'emb_{sl}.f16'; pn = D + f'pos_{sl}.npy'
        done = 0
        if os.path.exists(pf):
            pr = json.load(open(pf)); done = pr['done']
            if 'ids_hash' in pr:
                assert pr['ids_hash'] == h and pr['n'] == len(pos), f'[{k}] progress file belongs to a different pool: refusing to resume (delete {pf} and its emb/pos to redo)'
            else:   # legacy progress (no hash): only trusted for the default train pool in the train dir
                assert os.path.abspath(D) == os.path.abspath(globals()['D']), f'[{k}] legacy progress file in a non-train dir'
                assert os.path.exists(pn) and (np.load(pn) == pos).all(), f'[{k}] legacy progress but pos mismatch'
        if os.path.exists(pn) and done: assert (np.load(pn) == pos).all(), f'[{k}] existing pos file differs from this pool'
        np.save(pn, pos)
        if os.path.exists(fn): assert os.path.getsize(fn) == len(pos) * DIM * 2, f'[{k}] existing emb file has wrong size'
        mm = np.memmap(fn, dtype=np.float16, mode='r+' if os.path.exists(fn) else 'w+', shape=(len(pos), DIM))
        print(f'[{k}] rows {len(pos)} resume at {done}', flush=True)
        for s in range(done, len(pos), chunk):
            p = pos[s:s + chunk]
            sub = raw.take(p)
            nm = pd.Series(sub.column('business_name').to_numpy(zero_copy_only=False))
            ad = pd.Series(sub.column('business_address').to_numpy(zero_copy_only=False))
            txt = np.asarray(['passage: ' + t for t in norm.embed_text(nm, ad)], dtype=object)
            order = np.argsort([len(t) for t in txt], kind='stable')
            e = encoder(list(txt[order]))
            out = np.empty((len(p), DIM), dtype=np.float16); out[order] = e.astype(np.float16)
            assert np.isfinite(out).all()
            mm[s:s + len(p)] = out; mm.flush()
            json.dump({'done': s + len(p), 'n': len(pos), 'ids_hash': h}, open(pf + '.tmp', 'w')); os.replace(pf + '.tmp', pf)
            done_total += len(p); el = time.time() - t0
            print(f'[{k}] {s + len(p)}/{len(pos)}  elapsed {el/60:.1f}m  rate {done_total/el:.0f}/s  ETA {(total-done_total)/(done_total/el)/60:.1f}m  rss {rss():.1f}GB', flush=True)
        del mm
    print('ENCODE DONE', time.time() - t0, flush=True)


# ----------------------------------------------------------------------------- search
K = 100          # stored dense depth; recall@50 is the first 50 columns
def open_country(c, outdir=None, missing_ok=False):
    """returns (pos, memmap); (None, None) if country never encoded and missing_ok (else FileNotFoundError).
    Routing is by ckey(c) via the outdir manifest (legacy train dir: slug match)."""
    D = outdir or globals()['D']
    sl = _slug_for(c, D)
    if sl is None or not (os.path.exists(D + f'pos_{sl}.npy') and os.path.exists(D + f'emb_{sl}.f16')):
        if missing_ok: return None, None
        raise FileNotFoundError(f'no encoded country {c!r} in {D}')
    pos = np.load(D + f'pos_{sl}.npy')
    mm = np.memmap(D + f'emb_{sl}.f16', dtype=np.float16, mode='r', shape=(len(pos), DIM))
    assert json.load(open(D + f'progress_{sl}.json'))['done'] == len(pos), f'{c} encode incomplete'
    return pos, mm

def _dev():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def topk_stream(Q, mm, k=K, pchunk=200_000, qchunk=1500):
    rs, vs = [], []
    for s in range(0, Q.shape[0], qchunk):
        r, v = _topk_stream(Q[s:s + qchunk], mm, k, pchunk); rs.append(r); vs.append(v)
    return np.concatenate(rs), np.concatenate(vs)

def _topk_stream(Q, mm, k=K, pchunk=200_000):
    """exact inner-product top-k of Q (nq,384 fp16 torch cuda) over memmap rows, streamed in pchunk blocks.
    Scores cast to fp32 before topk (fp16 ties blur the boundary). returns (rows int64, scores float32) numpy; fewer than k columns if the pool has fewer than k rows (callers pad)."""
    import torch
    if Q.device.type == 'cpu': Q = Q.float()
    bv = bi = None
    for s in range(0, mm.shape[0], pchunk):
        blk = torch.from_numpy(np.ascontiguousarray(mm[s:s + pchunk])).to(Q.device)
        if Q.device.type == 'cpu': blk = blk.float()
        sc = (Q @ blk.T).float()
        v, i = sc.topk(min(k, blk.shape[0]), dim=1); i = i + s
        if bv is None: bv, bi = v, i
        else:
            v2, j = torch.cat([bv, v], 1).topk(min(k, bv.shape[1] + v.shape[1]), dim=1); bi = torch.cat([bi, i], 1).gather(1, j); bv = v2
        del sc, blk
    return bi.cpu().numpy(), bv.cpu().numpy()

def sample_val_anchors(n, seed=11):
    sp = pd.read_parquet(norm.I + 'split.parquet'); v = sp[sp.is_val]
    ns = int(round(n * (~v.has_match).mean()))
    pick = pd.concat([v[v.has_match].sample(n - ns, random_state=seed), v[~v.has_match].sample(ns, random_state=seed)])
    return pick.sample(frac=1, random_state=seed).reset_index(drop=True)

def dense_search(Q, an_country, seeds_exp=3, nb=11, outdir=None, k=K):
    """THE country-routing point for candidate search. Q (n,DIM) fp16 query embeddings, an_country raw labels.
    Anchors are grouped by ckey(country) and searched only inside that country's memmap (open_country, same key as encode).
    Returns cpos (n,k) int32 pool positions, csc (n,k) float32, epos (n, seeds_exp*(nb-1)) int32, padded with -1 / -inf.
    Absent country -> all -1 (no crash, never a candidate from another country). Pools smaller than k are padded."""
    import torch
    dev = _dev(); n = len(an_country)
    cpos = np.full((n, k), -1, np.int32); csc = np.full((n, k), -np.inf, np.float32)
    epos = np.full((n, seeds_exp * (nb - 1)), -1, np.int32)   # pool-side expansion: neighbours of top seeds
    ak = pd.Series(np.asarray(an_country, dtype=object)).map(ckey).values
    for c in sorted(set(ak)):
        pos, mm = open_country(c, outdir=outdir, missing_ok=True); ai = np.flatnonzero(ak == c)
        if pos is None:   # country absent from pool (e.g. France at test time): empty candidate set, no crash
            print(f'WARNING [{c}] no pool records: {len(ai)} anchors get empty candidates', flush=True); continue
        kk = min(k, len(pos)); se = min(seeds_exp, kk); nk = min(nb, len(pos))   # clamp to pool size, pad below
        r, v = topk_stream(torch.from_numpy(Q[ai]).to(dev), mm, k=kk)
        assert r.shape == (len(ai), kk)
        cpos[ai, :kk] = pos[r]; csc[ai, :kk] = v
        # expansion: seeds = top-s dense candidates; neighbours = their top-nb in pool (minus self)
        seed_rows = r[:, :se].ravel()
        S = torch.from_numpy(np.ascontiguousarray(mm[np.sort(seed_rows)])).to(dev)
        srt = np.argsort(seed_rows); inv = np.empty_like(srt); inv[srt] = np.arange(len(srt))
        nr, nv = topk_stream(S[torch.from_numpy(inv).to(dev)], mm, k=nk)
        nr = nr.reshape(len(ai), se, nk); sr = r[:, :se, None]
        # drop the seed itself (may not be at rank 0 if exact dup embeddings): mask, keep first nb-1 others
        out = np.full((len(ai), seeds_exp, nb - 1), -1, np.int64)
        keep = nr != sr
        for a_ in range(len(ai)):
            for s_ in range(se):
                x = nr[a_, s_][keep[a_, s_]][:nb - 1]; out[a_, s_, :len(x)] = x
        e = out.reshape(len(ai), -1); epos[ai] = np.where(e >= 0, pos[np.clip(e, 0, None)], -1)
        print(f'[{c}] searched {len(ai)} anchors vs {len(pos)} pool', flush=True)
    return cpos, csc, epos

def dense_table(cpos, csc, k=K):
    """long (a, p, score, rank) table of the REAL dense candidates only: drops -1 padding / non-finite scores."""
    nA = len(cpos)
    d = pd.DataFrame({'a': np.repeat(np.arange(nA), cpos.shape[1]), 'p': cpos.ravel().astype(np.int64), 'score': csc.ravel(), 'rank': np.tile(np.arange(cpos.shape[1]), nA)})
    d = d[(d.p >= 0) & np.isfinite(d.score)]
    assert not d.duplicated(['a', 'p']).any(), 'repeated (a,p) dense candidates'
    return d

def search(n=5000, seed=11, seeds_exp=3, nb=11):
    import torch
    from sentence_transformers import SentenceTransformer
    t0 = time.time(); os.makedirs(D, exist_ok=True)
    pick = sample_val_anchors(n, seed)
    s1 = pq.read_table(norm.I + 'source1.parquet').to_pandas().set_index('entity_id').loc[pick.source1_entity_id].reset_index()
    assert (s1.entity_id.values == pick.source1_entity_id.values).all() and (s1.country.values == pick.country.values).all()
    an = norm.prep(s1); an['has_match'] = pick.has_match.values; an['text'] = norm.embed_text(s1.business_name, s1.business_address)
    m = SentenceTransformer(MODEL, device='cuda', model_kwargs={'torch_dtype': 'float16'}); m.max_seq_length = 64
    Q = m.encode(['query: ' + t for t in an.text], batch_size=512, normalize_embeddings=True, convert_to_numpy=True).astype(np.float16)
    assert np.isfinite(Q).all(); del m; torch.cuda.empty_cache()
    an.drop(columns='text').to_parquet(D + 'val_anchors.parquet'); np.save(D + 'val_Q.npy', Q)
    cpos, csc, epos = dense_search(Q, an.country.values, seeds_exp=seeds_exp, nb=nb)
    routed = cpos[:, 0] >= 0   # anchors whose country is absent from the pool keep empty (-1) candidates
    assert ((cpos >= 0) == np.isfinite(csc)).all()   # padding (-1) <=> -inf score
    print(f'anchors with empty candidate set (country not in pool): {(~routed).sum()}/{len(an)}', flush=True)
    np.savez(D + 'val_dense.npz', cpos=cpos, csc=csc, epos=epos)
    print('SEARCH DONE', time.time() - t0, 'rss', rss(), flush=True)

# ----------------------------------------------------------------------------- lexical candidates
def lex(cap_addr=3000):
    from . import blockers as B
    t0 = time.time(); pool = norm.load_pool(); an = pd.read_parquet(D + 'val_anchors.parquet')
    s1 = pq.read_table(norm.I + 'source1.parquet').to_pandas().set_index('entity_id').loc[an.id].reset_index()
    ap = norm.prep(s1); assert (ap.id.values == an.id.values).all()
    out = {}
    for nm, b in [('addrnum2', B.AddrNum(cap_addr, 2)), ('addrnum0', B.AddrNum(cap_addr, 0)), ('prefix4c', B.Prefix(4, cap=2000)), ('tokc300', B.Token(300))]:
        b.fit(pool); a, p = b.query(ap); out[nm + '_a'] = a.astype(np.int32); out[nm + '_p'] = p.astype(np.int32)
        print(f'[lex {nm}] pairs {len(a)} per-anchor {len(a)/len(ap):.0f}  {time.time()-t0:.0f}s rss {rss():.1f}', flush=True); del b
    np.savez(D + 'val_lex.npz', **out)
    print('LEX DONE', flush=True)

# ----------------------------------------------------------------------------- report
def _keys(a, p): return a.astype(np.int64) * (1 << 32) + p.astype(np.int64)

def _dense_scores(a, p, Q, pool_country, an_country, cache, outdir=None):
    """exact q.e for arbitrary (anchor idx, pool pos) pairs, gathered from the country memmaps (chunked).
    Pairs whose anchor country is absent from the pool get -1.0 (cosine floor)."""
    out = np.full(len(a), -1.0, np.float32)
    ak = pd.Series(np.asarray(an_country, dtype=object)).map(ckey).values
    for c in np.unique(ak):
        ck = (outdir, c)
        if ck not in cache: cache[ck] = open_country(c, outdir=outdir, missing_ok=True)
        pos, mm = cache[ck]
        if pos is None: continue
        sel = np.flatnonzero(ak[a] == c)
        rows = np.searchsorted(pos, p[sel]); assert (rows < len(pos)).all() and (pos[rows] == p[sel]).all()
        for s in range(0, len(sel), 400_000):
            ss = sel[s:s + 400_000]
            out[ss] = np.einsum('ij,ij->i', Q[a[ss]].astype(np.float32), mm[rows[s:s + 400_000]].astype(np.float32))
    return out

def _topm_excl(df, M, excl_keys, by):
    """per anchor, keep the top-M rows by `by` (desc) among rows whose key is not in excl_keys."""
    if M <= 0 or len(df) == 0: return df.iloc[:0]
    d = df[~np.isin(_keys(df.a.values, df.p.values), excl_keys)]
    d = d.sort_values(['a', by], ascending=[True, False])
    return d[d.groupby('a').cumcount().values < M]

def report(topn=25):
    from . import blockers as B
    t0 = time.time(); pool = norm.load_pool(); an = pd.read_parquet(D + 'val_anchors.parquet'); Q = np.load(D + 'val_Q.npy')
    dn = np.load(D + 'val_dense.npz'); lx = np.load(D + 'val_lex.npz'); nA = len(an)
    gt = pd.read_parquet(norm.I + 'gt.parquet').set_index('source1_entity_id').loc[an.id.values].matched_entity_ids
    e = pd.DataFrame({'a': np.arange(nA), 't': gt.str.split(',').values}).explode('t'); e = e[e.t.notna() & (e.t != '')]
    e['p'] = pd.Index(pool.id.values).get_indexer(e.t.values); assert (e.p >= 0).all()
    assert (an.has_match.values == (gt.values != '')).all()
    tp = e[['a', 'p']].astype(np.int64).reset_index(drop=True); tk = _keys(tp.a.values, tp.p.values)
    tp['country'] = an.country.values[tp.a.values]; tp['script'] = pool.script.values[tp.p.values]; tp['src'] = pool.id.values[tp.p.values].astype('U2')
    print(f'anchors {nA}: matched {an.has_match.sum()} singletons {(~an.has_match).sum()}; true pairs {len(tp)}; countries {an.country.value_counts().to_dict()}')
    cache = {}; C = an.country.values
    # candidate tables
    cp, cs = dn['cpos'], dn['csc']
    D_ = dense_table(cp, cs)   # real candidates only (no -1 padding)
    ep = dn['epos']; E_ = pd.DataFrame({'a': np.repeat(np.arange(nA), ep.shape[1]), 'p': ep.ravel().astype(np.int64)}); E_ = E_[E_.p >= 0].drop_duplicates(['a', 'p'])
    E_['score'] = _dense_scores(E_.a.values, E_.p.values, Q, None, C, cache); E_['votes'] = 0
    lexd = {}
    for nm in ['addrnum2', 'addrnum0', 'prefix4c', 'tokc300']:
        d = pd.DataFrame({'a': lx[nm + '_a'].astype(np.int64), 'p': lx[nm + '_p'].astype(np.int64)}).drop_duplicates(); d['v'] = 1; lexd[nm] = d
    def lexunion(names):
        d = pd.concat([lexd[n] for n in names]).groupby(['a', 'p'], as_index=False).v.sum().rename(columns={'v': 'votes'})
        d['score'] = _dense_scores(d.a.values, d.p.values, Q, None, C, cache); return d
    L = {nm: lexunion([nm]) for nm in lexd}; L['all'] = lexunion(list(lexd)); L['addr2+0'] = lexunion(['addrnum2', 'addrnum0'])
    print(f'tables built {time.time()-t0:.0f}s; lex pairs/anchor: ' + ', '.join(f'{k} {len(v)/nA:.0f}' for k, v in L.items()), f'expansion {len(E_)/nA:.1f}', flush=True)
    def build(Kd, Me, Ml, lname='all', by='score'):
        d = D_[D_['rank'] < Kd][['a', 'p']]; ex = _keys(d.a.values, d.p.values)
        parts = [d]
        x = _topm_excl(E_, Me, ex, 'score'); parts.append(x[['a', 'p']]); ex = np.concatenate([ex, _keys(x.a.values, x.p.values)]) if len(x) else ex
        if Ml:
            l = L[lname].assign(rk=lambda z: z.votes + z.score) if by == 'votes' else L[lname]
            y = _topm_excl(l, Ml, ex, 'rk' if by == 'votes' else 'score'); parts.append(y[['a', 'p']])
        return pd.concat(parts)
    def hits(cand):
        ck = np.unique(_keys(cand.a.values, cand.p.values)); return np.isin(tk, ck)
    def summary(cand, tag):
        h = hits(cand); t = tp.assign(hit=h.astype(float)); sz = np.bincount(cand.a.values, minlength=nA)
        return dict(tag=tag, cand_mean=sz.mean(), cand_matched=sz[an.has_match.values].mean(), cand_single=sz[~an.has_match.values].mean(),
                    pair_R=h.mean(), any_hit=t.groupby('a').hit.max().mean(), cluster_full=t.groupby('a').hit.min().mean(),
                    nonlatin_R=t[t.script == 'nonlatin'].hit.mean(), India_R=t[t.country == 'India'].hit.mean(), US_R=t[t.country == 'US'].hit.mean()), t, sz
    rows = []
    for Kd in (20, 30, 50, 75, 100): rows.append(summary(build(Kd, 0, 0), f'dense@{Kd}')[0])
    for nm in L:
        for Ml in (25, 50, 100): rows.append(summary(build(50, 0, Ml, nm), f'dense50+lex[{nm}]top{Ml}')[0])
    for Me in (10, 20, 30): rows.append(summary(build(50, Me, 0), f'dense50+expand{Me}')[0])
    grid = []
    for Kd in (30, 40, 50, 70):
        for Me in (0, 10, 20):
            for Ml in (0, 10, 20, 40, 60):
                for by in ('score', 'votes'):
                    if Ml == 0 and by == 'votes': continue
                    r = summary(build(Kd, Me, Ml, 'all', by), f'D{Kd}+E{Me}+L{Ml}[{by}]')[0]; grid.append(r)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    R = pd.DataFrame(rows); print('\n=== single configs'); print(R.round(4).to_string(index=False))
    G = pd.DataFrame(grid); G.to_csv(D + 'val_grid.csv', index=False)
    print('\n=== grid (cand_mean <= 160), top by pair_R'); print(G[G.cand_mean <= 160].sort_values('pair_R', ascending=False).head(topn).round(4).to_string(index=False))
    print('\n=== Pareto: best pair_R at each mean-candidate budget'); 
    for b in (60, 80, 100, 125, 150): 
        g = pd.concat([R, G]); g = g[g.cand_mean <= b]; print(b, g.sort_values('pair_R', ascending=False).head(1).round(4).to_string(header=False, index=False))
    # ---- breakdown for headline configs
    def breakdown(cand, tag):
        s, t, sz = summary(cand, tag); print(f'\n=== breakdown {tag}: cand mean {s["cand_mean"]:.1f} (matched {s["cand_matched"]:.1f}, singleton {s["cand_single"]:.1f})')
        def rep(g): return pd.Series({'pairs': len(g), 'pair_R': g.hit.mean(), 'cluster_full': g.groupby('a').hit.min().mean(), 'any_hit': g.groupby('a').hit.max().mean()})
        o = {'overall': rep(t)}
        for c, g in t.groupby('country'): o['country=' + c] = rep(g)
        for c, g in t.groupby('script'): o['script=' + c] = rep(g)
        for c, g in t.groupby('src'): o['src=' + c] = rep(g)
        for c, g in t[t.country == 'India'].groupby('script'): o['India,script=' + c] = rep(g)
        for c, g in t[t.country == 'US'].groupby('script'): o['US,script=' + c] = rep(g)
        print(pd.DataFrame(o).T.round(4).to_string()); return sz
    sz50 = breakdown(build(50, 0, 0), 'dense@50 (TRUE full pool)')
    bt = G[G.cand_mean <= 150].sort_values('pair_R', ascending=False).iloc[0].tag; print('best <=150 grid config:', bt)
    import re as _re; m_ = _re.match(r'D(\d+)\+E(\d+)\+L(\d+)\[(\w+)\]', bt)
    bc = build(int(m_[1]), int(m_[2]), int(m_[3]), 'all', m_[4]); szb = breakdown(bc, bt)
    # ---- singletons
    ok = np.isfinite(cs[:, 0]); s = ~an.has_match.values; top1 = cs[:, 0]
    print('\n=== singleton anchors (must get empty predictions later)')
    for nm_, mask in [('singletons', s & ok), ('matched', ~s & ok)]:
        print(nm_, 'n', mask.sum(), 'dense top1 cos pctl 5/25/50/75/95:', np.round(np.quantile(top1[mask], [.05, .25, .5, .75, .95]), 4))
    for c in np.unique(C):
        m1 = s & ok & (C == c); print(f'  {c}: singleton n {m1.sum()} cand@dense50 {sz50[m1].mean():.1f} cand@best {szb[m1].mean():.1f}; top1 median singleton {np.median(top1[m1]):.3f} vs matched {np.median(top1[(~s) & ok & (C == c)]):.3f}')
    thr = np.quantile(top1[~s & ok], 0.10); print(f'a top1-cos threshold keeping 90% of matched anchors abstains on {(top1[s & ok] < thr).mean():.1%} of singletons (thr {thr:.3f}); note top1 alone is a weak abstain signal, matcher decides')
    print('REPORT DONE', time.time() - t0, 'rss', rss())

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('--chunk', type=int, default=100_000)
    ap.add_argument('--limit', type=int, default=None); ap.add_argument('--n', type=int, default=5000); a = ap.parse_args()
    if a.stage == 'encode': encode(a.chunk, a.limit)
    elif a.stage == 'search': search(a.n)
    elif a.stage == 'lex': lex()
    elif a.stage == 'report': report()
