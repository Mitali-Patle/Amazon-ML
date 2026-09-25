"""Unified candidate generation (frozen ADR-002 config D70+E20+L60[votes], cap 150), identical for train / val / test anchors.
Inference-only: per-record transforms, embedding, search. No statistics/tuning on the data it is given.

  encode_queries(raw, outdir, name)        # anchor 'query:' embeddings, checkpointed/resumable (same text builder as train run)
  candidates(anchor_df, pool_df, emb_dir, outdir, Q=...)  -> DataFrame(a, p, src, dense_rank, dense_score, votes, rank)
  write_candidate_pairs_tsv(cands, anchor_ids, pool_ids, path)

anchor_df / pool_df: prepped frames (norm.prep: id,country,nn,nums,script), row order == embedding order:
  pool_df row i  <-> pool position i in emb_dir (pos_<slug>.npy holds positions per country)
  anchor_df row i <-> Q[i]
Output 'a' = anchor row index, 'p' = pool row position. Country-partitioned: an anchor only ever sees pool rows of its own
country (ckey); anchors whose country has no pool rows get NO candidates (logged count).
CLI:  python -m src.blocking.candidates encode-test | prep-test | cands-test | cands-val | rehearse"""
import argparse, hashlib, json, os, re, time, resource, sys
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from . import norm
from . import full_pool as F
from . import blockers as B

DIM = F.DIM
CFG = dict(kd=70, seeds=3, nb=11, me=20, ml=60, cap=150)   # FROZEN
SRC_DENSE, SRC_EXP, SRC_LEX = 1, 2, 4
LEX = [('addrnum2', lambda: B.AddrNum(3000, 2)), ('addrnum0', lambda: B.AddrNum(3000, 0)),
       ('prefix4c', lambda: B.Prefix(4, cap=2000)), ('tokc300', lambda: B.Token(300))]

def rss_gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
def cur_rss_gb():
    try: return int(open('/proc/self/statm').read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 1e9
    except Exception: return float('nan')
def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)
def _hash(x): return hashlib.md5('\x00'.join(map(str, x)).encode()).hexdigest()
def _atomic_json(o, f): json.dump(o, open(f + '.tmp', 'w')); os.replace(f + '.tmp', f)

# ----------------------------------------------------------------------------- encoders
def make_encoder(bs=512, max_len=64):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(F.MODEL, device='cuda', model_kwargs={'torch_dtype': 'float16'}); m.max_seq_length = max_len
    return lambda t: m.encode(t, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

def encode_queries(raw, outdir, name='anchors', encoder=None, chunk=100_000):
    """raw: pyarrow Table / DataFrame with entity_id, business_name, business_address. Writes outdir/Q_<name>.f16 (n,DIM) fp16
    (NORMALIZED 'query: ' + norm.embed_text) + progress json with n and id hash; resumable. Returns the read-only memmap."""
    assert outdir, 'outdir required'
    D = outdir if outdir.endswith('/') else outdir + '/'
    assert os.path.abspath(D) != os.path.abspath(F.D), 'never write into the train embeddings dir'
    os.makedirs(D, exist_ok=True)
    tbl = raw if isinstance(raw, pa.Table) else pa.Table.from_pandas(raw, preserve_index=False)
    n = tbl.num_rows; ids = tbl.column('entity_id').to_numpy(zero_copy_only=False); h = _hash(ids)
    fn, pf = D + f'Q_{name}.f16', D + f'Qprogress_{name}.json'; done = 0
    if os.path.exists(pf):
        pr = json.load(open(pf)); assert pr['n'] == n and pr['ids_hash'] == h, f'{pf} belongs to different anchors: refusing to resume'; done = pr['done']
    if os.path.exists(fn): assert os.path.getsize(fn) == n * DIM * 2
    mm = np.memmap(fn, dtype=np.float16, mode='r+' if os.path.exists(fn) else 'w+', shape=(n, DIM))
    encoder = encoder or make_encoder(); t0 = time.time(); d0 = done
    for s in range(done, n, chunk):
        sub = tbl.slice(s, chunk)
        nm = pd.Series(sub.column('business_name').to_numpy(zero_copy_only=False)); ad = pd.Series(sub.column('business_address').to_numpy(zero_copy_only=False))
        txt = np.asarray(['query: ' + t for t in norm.embed_text(nm, ad)], dtype=object)
        order = np.argsort([len(t) for t in txt], kind='stable'); e = encoder(list(txt[order]))
        out = np.empty((len(txt), DIM), np.float16); out[order] = e.astype(np.float16); assert np.isfinite(out).all()
        mm[s:s + len(txt)] = out; mm.flush(); done = s + len(txt)
        _atomic_json({'done': done, 'n': n, 'ids_hash': h}, pf)
        el = time.time() - t0; log(f'[Q {name}] {done}/{n} rate {(done-d0)/el:.0f}/s ETA {(n-done)/((done-d0)/el)/60:.1f}m rss {cur_rss_gb():.1f}GB')
    assert json.load(open(pf))['done'] == n
    return np.memmap(fn, dtype=np.float16, mode='r', shape=(n, DIM))

def open_queries(outdir, name='anchors', n=None):
    D = outdir if outdir.endswith('/') else outdir + '/'
    pr = json.load(open(D + f'Qprogress_{name}.json')); assert pr['done'] == pr['n'] and (n is None or n == pr['n']), 'query encode incomplete / wrong size'
    return np.memmap(D + f'Q_{name}.f16', dtype=np.float16, mode='r', shape=(pr['n'], DIM))

# ----------------------------------------------------------------------------- dense search (bounded VRAM)
class PoolEmb:
    """One country's pool embeddings. If CUDA and the country fits (<= gpu_limit_gb fp16) it is made GPU-resident (no per-query PCIe
    streaming, fast pair scoring); otherwise blocks are streamed from the memmap. CPU path (tests) streams as fp32."""
    def __init__(self, mm, gpu_limit_gb=5.0):
        import torch
        self.mm, self.N, self.dev, self.T = mm, mm.shape[0], F._dev(), None
        if self.dev == 'cuda' and self.N * DIM * 2 / 1e9 <= gpu_limit_gb:
            self.T = torch.empty((self.N, DIM), dtype=torch.float16, device='cuda')
            fn = mm.filename   # plain file reads (not the memmap) so the 3-5GB of pool pages are not charged to this process's RSS
            with open(fn, 'rb') as f:
                for s in range(0, self.N, 500_000):
                    n = min(500_000, self.N - s); b = np.fromfile(f, dtype=np.float16, count=n * DIM).reshape(n, DIM); self.T[s:s + n] = torch.from_numpy(b).to('cuda')
    def close(self):
        import torch
        self.T = None; torch.cuda.empty_cache() if self.dev == 'cuda' else None
    def block(self, s, e):
        import torch
        if self.T is not None: return self.T[s:e]
        b = torch.from_numpy(np.ascontiguousarray(self.mm[s:e])).to(self.dev); return b.float() if self.dev == 'cpu' else b
    def topk(self, Q, k, nq_x_p=200_000_000):
        """exact top-k inner product of Q (nq,DIM torch, same device) over the country pool; pool block size adapts so nq*pblock <= nq_x_p.
        returns (rows int64 (nq,kk), scores fp32)."""
        import torch
        nq = Q.shape[0]; pchunk = max(4096, nq_x_p // max(nq, 1))
        if self.dev == 'cpu': Q = Q.float()
        bv = bi = None
        for s in range(0, self.N, pchunk):
            blk = self.block(s, s + pchunk); sc = (Q @ blk.T).float(); v, i = sc.topk(min(k, blk.shape[0]), dim=1); i = i + s
            if bv is None: bv, bi = v, i
            else: v2, j = torch.cat([bv, v], 1).topk(min(k, bv.shape[1] + v.shape[1]), dim=1); bi = torch.cat([bi, i], 1).gather(1, j); bv = v2
            del sc, blk
        return bi.cpu().numpy(), bv.cpu().numpy()
    def pair_scores(self, Qc, a, p):
        """exact fp32 q.e for (local anchor idx a, local pool row p)."""
        import torch
        out = np.empty(len(a), np.float32)
        if self.T is not None:
            Qt = torch.from_numpy(np.ascontiguousarray(Qc)).to('cuda')
            for s in range(0, len(a), 250_000):
                at = torch.from_numpy(a[s:s + 250_000]).to('cuda'); pt = torch.from_numpy(p[s:s + 250_000]).to('cuda')
                out[s:s + 250_000] = (Qt[at].float() * self.T[pt].float()).sum(1).cpu().numpy()
            return out
        for s in range(0, len(a), 400_000):
            out[s:s + 400_000] = np.einsum('ij,ij->i', Qc[a[s:s + 400_000]].astype(np.float32), self.mm[p[s:s + 400_000]].astype(np.float32))
        return out

def dense_expand(Qc, pe, kd, seeds, nb, qchunk=10_000):
    """Qc (n,DIM fp16 numpy) vs PoolEmb. returns rows (n,kk) int64 / scores fp32 / erows (n, seeds*(nb-1)) int64 (-1 padded).
    Expansion: neighbours (top-nb in pool, minus the seed itself, first nb-1) of the top-`seeds` dense hits. Neighbour search is
    done once per UNIQUE seed row (pool-side quantity, independent of the anchor)."""
    import torch
    dev = pe.dev; n = len(Qc); N = pe.N; kk = min(kd, N); se = min(seeds, kk); nk = min(nb, N)
    R = np.empty((n, kk), np.int64); V = np.empty((n, kk), np.float32)
    for s in range(0, n, qchunk):
        R[s:s + qchunk], V[s:s + qchunk] = pe.topk(torch.from_numpy(np.ascontiguousarray(Qc[s:s + qchunk])).to(dev), kk)
    seed_rows = R[:, :se]; u, inv = np.unique(seed_rows.ravel(), return_inverse=True)
    UR = np.empty((len(u), nk), np.int64)
    for s in range(0, len(u), qchunk):
        ub = u[s:s + qchunk]
        Sq = pe.T[torch.from_numpy(ub).to(dev)] if pe.T is not None else torch.from_numpy(np.ascontiguousarray(pe.mm[ub])).to(dev)
        UR[s:s + qchunk], _ = pe.topk(Sq, nk)
    nr = UR[inv.reshape(-1)].reshape(n, se, nk); keep = nr != seed_rows[:, :, None]
    o = np.argsort(~keep, axis=2, kind='stable')[:, :, :nb - 1]     # kept entries first, original order
    g = np.take_along_axis(nr, o, 2); gk = np.take_along_axis(keep, o, 2)
    E = np.full((n, seeds, nb - 1), -1, np.int64); E[:, :se, :g.shape[2]] = np.where(gk, g, -1)
    return R, V, E.reshape(n, -1)

# ----------------------------------------------------------------------------- selection helpers
def _keys(a, p): return a.astype(np.int64) * (1 << 32) + p.astype(np.int64)

def _topm_excl(df, M, excl, by, tie='p'):
    """per anchor keep top-M rows by `by` desc (ties: smaller p first, deterministic) among rows whose key is not in excl."""
    if M <= 0 or len(df) == 0: return df.iloc[:0]
    d = df[~np.isin(_keys(df.a.values, df.p.values), excl)]
    d = d.sort_values(['a', by, tie], ascending=[True, False, True], kind='stable')
    return d[d.groupby('a').cumcount().values < M]

def _lex_votes(anc_c, blockers, nsub=25_000):
    """anc_c: prepped anchors (local index 0..n-1) of one country; blockers: fitted on the country's pool (local rows).
    Per sub-chunk: raw pairs from each blocker -> votes (# blockers that fired) -> vote table. Raw pairs never accumulate beyond a sub-chunk."""
    outs = []
    for s in range(0, len(anc_c), nsub):
        sub = anc_c.iloc[s:s + nsub].reset_index(drop=True); parts = []
        for b in blockers:
            a, p = b.query(sub)
            x = pd.DataFrame({'a': a.astype(np.int64) + s, 'p': p.astype(np.int64)}).drop_duplicates(); x['v'] = np.int8(1); parts.append(x)
        outs.append(pd.concat(parts).groupby(['a', 'p'], as_index=False).v.sum().rename(columns={'v': 'votes'}))
        del parts
    return pd.concat(outs, ignore_index=True) if outs else pd.DataFrame({'a': [], 'p': [], 'votes': []})

def _select(R, V, E, Qc, pe, lex, cfg):
    """one anchor block of one country -> DataFrame(a,p,src,dense_rank,dense_score,votes) with LOCAL indices. Enforces per-anchor cap.
    Lexical rank = votes + cosine (cosine in [0,1) so votes dominate): only pairs in the vote tiers that can reach the top-ml are scored (exact)."""
    n = len(R); kd = min(cfg['kd'], R.shape[1])
    D = pd.DataFrame({'a': np.repeat(np.arange(n), kd), 'p': R[:, :kd].ravel(), 'score': V[:, :kd].ravel(), 'rank': np.tile(np.arange(kd), n)})
    ex = _keys(D.a.values, D.p.values)
    Ex = pd.DataFrame({'a': np.repeat(np.arange(n), E.shape[1]), 'p': E.ravel()}); Ex = Ex[Ex.p >= 0].drop_duplicates(['a', 'p'])
    Ex['score'] = pe.pair_scores(Qc, Ex.a.values, Ex.p.values) if len(Ex) else np.empty(0, np.float32)
    X = _topm_excl(Ex, cfg['me'], ex, 'score')
    ex2 = np.concatenate([ex, _keys(X.a.values, X.p.values)])
    L = lex[~np.isin(_keys(lex.a.values, lex.p.values), ex2)]
    if len(L):
        C = np.bincount(L.a.values * 8 + L.votes.values, minlength=n * 8).reshape(n, 8)          # (anchor, votes) counts
        ge = np.cumsum(C[:, ::-1], 1)[:, ::-1]                                                    # ge[a,t] = #pairs with votes >= t
        thr = np.where(ge >= cfg['ml'], np.arange(8)[None, :], 0).max(1); thr = np.maximum(thr, 1)  # highest tier still holding >= ml pairs
        L = L[L.votes.values >= thr[L.a.values]].copy()
        L['score'] = pe.pair_scores(Qc, L.a.values, L.p.values); L['rk'] = L.votes.astype(np.float32) + L.score
    else: L = L.assign(score=np.empty(0, np.float32), rk=np.empty(0, np.float32))
    Y = _topm_excl(L, cfg['ml'], np.empty(0, np.int64), 'rk')
    cols = ['a', 'p', 'src', 'dense_rank', 'dense_score', 'votes']
    o = pd.concat([D.assign(src=SRC_DENSE, dense_rank=D['rank'], dense_score=D.score, votes=0)[cols],
                   X.assign(src=SRC_EXP, dense_rank=-1, dense_score=X.score, votes=0)[cols],
                   Y.assign(src=SRC_LEX, dense_rank=-1, dense_score=Y.score, votes=Y.votes)[cols]], ignore_index=True)
    assert not o.duplicated(['a', 'p']).any(), 'duplicate (a,p)'
    assert (np.bincount(o.a.values, minlength=n) <= cfg['cap']).all(), 'per-anchor cap exceeded'
    o['rank'] = o.groupby('a').cumcount().astype(np.int16)   # concat order: dense (by score), expansion (by score), lexical (by votes+score)
    return o.astype({'a': np.int32, 'p': np.int32, 'src': np.int8, 'dense_rank': np.int16, 'dense_score': np.float32, 'votes': np.int8})

# ----------------------------------------------------------------------------- main entry
def candidates(anchor_df, pool_df, emb_dir, outdir, Q, chunk=50_000, qchunk=10_000, sub=10_000, cfg=None, only_anchors=None, nice_log=None):
    """anchor_df: prepped anchors (id,country,nn,nums,script); pool_df: prepped pool (same cols); Q: (nA,DIM) fp16 array/memmap of
    'query:' embeddings (encode_queries); emb_dir: encoded pool dir (full_pool.encode). outdir: resumable per-chunk parquet
    (cands_<slug>_<i>.parquet + meta.json). Returns concatenated DataFrame with GLOBAL a (anchor row) and p (pool row) indices.
    Country-partitioned; countries are derived from the data. Anchors whose country has no encoded pool rows: no candidates (logged)."""
    cfg = {**CFG, **(cfg or {})}; emb_dir = emb_dir if emb_dir.endswith('/') else emb_dir + '/'
    os.makedirs(outdir, exist_ok=True); outdir = outdir if outdir.endswith('/') else outdir + '/'
    nA = len(anchor_df); assert len(Q) == nA, 'Q rows != anchors'
    assert os.path.abspath(emb_dir) != os.path.abspath(outdir)
    meta = dict(cfg=cfg, chunk=chunk, nA=nA, nP=len(pool_df), aid=_hash(anchor_df.id.values), pid=_hash(pool_df.id.values[::97]) + str(len(pool_df)))
    mf = outdir + 'meta.json'
    if os.path.exists(mf): assert json.load(open(mf)) == meta, f'{mf}: outdir belongs to a different run (anchors/pool/config); refusing to resume'
    else: _atomic_json(meta, mf)
    ak = anchor_df.country.map(F.ckey).values; pk = pool_df.country.map(F.ckey).values
    sel_all = np.ones(nA, bool) if only_anchors is None else np.isin(np.arange(nA), only_anchors)
    t0 = time.time(); done_a = 0; tot_a = int(sel_all.sum()); skipped = {}
    for c in sorted(set(ak)):
        ai = np.flatnonzero((ak == c) & sel_all)
        if len(ai) == 0: continue
        pos, mm = F.open_country(c, outdir=emb_dir, missing_ok=True)
        if pos is None:
            skipped[c] = len(ai); log(f'WARNING [{c}] no pool records: {len(ai)} anchors get empty candidates'); done_a += len(ai); continue
        slug = F._slug_for(c, emb_dir); nch = (len(ai) + chunk - 1) // chunk
        todo = [i for i in range(nch) if not os.path.exists(outdir + f'cands_{slug}_{i}.parquet')]
        if not todo: done_a += len(ai); log(f'[{c}] all {nch} chunks present'); continue
        pool_c = pool_df.iloc[pos].reset_index(drop=True); pool_c['country'] = pool_c['country'].astype(object)   # local row r <-> pos[r]
        assert (pk[pos] == c).all()
        blockers = [f().fit(pool_c) for _, f in LEX]; log(f'[{c}] pool {len(pos)} blockers fit, rss {cur_rss_gb():.1f}GB peak {rss_gb():.1f}GB')
        pe = PoolEmb(mm); log(f'[{c}] pool embeddings {"GPU-resident" if pe.T is not None else "streamed"}')
        for i in range(nch):
            fn = outdir + f'cands_{slug}_{i}.parquet'; loc = ai[i * chunk:(i + 1) * chunk]
            if os.path.exists(fn): done_a += len(loc); continue
            tt = np.zeros(3); outs = []
            for s0 in range(0, len(loc), sub):          # blocks of `sub` anchors bound the RAM of raw lexical pairs / scored tables
                bl = loc[s0:s0 + sub]; Qb = np.ascontiguousarray(Q[bl]); t = time.time()
                R, V, E = dense_expand(Qb, pe, cfg['kd'], cfg['seeds'], cfg['nb'], qchunk); tt[0] += time.time() - t; t = time.time()
                anc_c = anchor_df.iloc[bl].reset_index(drop=True)
                if 'country' in anc_c: anc_c['country'] = anc_c['country'].astype(object)
                lex = _lex_votes(anc_c, blockers); tt[1] += time.time() - t; t = time.time()
                o = _select(R, V, E, Qb, pe, lex, cfg); tt[2] += time.time() - t
                o['a'] = bl[o.a.values].astype(np.int32); o['p'] = pos[o.p.values].astype(np.int32)   # -> global anchor row / pool position
                outs.append(o); del R, V, E, lex
            o = pd.concat(outs, ignore_index=True); del outs
            o.to_parquet(fn + '.tmp', index=False); os.replace(fn + '.tmp', fn)
            done_a += len(loc); el = time.time() - t0
            log(f'[{c}] chunk {i+1}/{nch} anchors {done_a}/{tot_a} mean cands {len(o)/len(loc):.1f} [dense {tt[0]:.0f}s lex {tt[1]:.0f}s select {tt[2]:.0f}s per {len(loc)} anchors] elapsed {el/60:.1f}m ETA {(tot_a-done_a)/max(done_a,1)*el/60:.1f}m rss {cur_rss_gb():.1f}GB peak {rss_gb():.1f}GB')
            del o
        pe.close(); del pe
        del blockers, pool_c; import gc; gc.collect()
        try: import ctypes; ctypes.CDLL('libc.so.6').malloc_trim(0)   # hand freed heap back to the OS between countries
        except Exception: pass
    if skipped: log(f'anchors with EMPTY candidates (country absent from pool): {skipped}, total {sum(skipped.values())}')
    _atomic_json({'skipped_countries': skipped}, outdir + 'skipped.json')
    return load_candidates(outdir)

def load_candidates(outdir):
    fs = sorted(f for f in os.listdir(outdir) if f.startswith('cands_') and f.endswith('.parquet'))
    return pd.concat([pd.read_parquet(outdir + f) for f in fs], ignore_index=True) if fs else pd.DataFrame(columns=['a', 'p', 'src', 'dense_rank', 'dense_score', 'votes', 'rank'])

# ----------------------------------------------------------------------------- candidate_pairs.tsv
def write_candidate_pairs_tsv(cands, anchor_ids, pool_ids, path):
    """official schema: source1_entity_id TAB comma-separated candidate ids; one row per anchor (in anchor_ids order), empty when none.
    Only S2-/S3- ids, no duplicates, no header rows beyond the schema header (source1_entity_id, candidate_entity_ids)."""
    anchor_ids = np.asarray(anchor_ids); pool_ids = np.asarray(pool_ids)
    assert len(set(anchor_ids.tolist())) == len(anchor_ids), 'duplicate anchor ids'
    c = cands.sort_values(['a', 'rank'], kind='stable'); assert not c.duplicated(['a', 'p']).any()
    pid = pool_ids[c.p.values]; assert all(x[:3] in ('S2-', 'S3-') for x in pd.unique(pid)), 'non S2-/S3- candidate id'
    starts = np.searchsorted(c.a.values, np.arange(len(anchor_ids) + 1))
    with open(path + '.tmp', 'w') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for i in range(len(anchor_ids)): f.write(f'{anchor_ids[i]}\t{",".join(pid[starts[i]:starts[i+1]])}\n')
    os.replace(path + '.tmp', path)

# ----------------------------------------------------------------------------- test-data plumbing (per-record transforms only)
T = norm.I
TEST_EMB = T + 'test_pool/'
def test_raw_pool():
    return pa.concat_tables([pq.read_table(T + f'test_source{i}.parquet') for i in (2, 3)])
def prep_test_pool(out=T + 'test_pool_norm.parquet'):
    """per-record norm.prep of test S2+S3 (same code as train pool_norm.parquet), streamed in row groups to bound RAM."""
    if os.path.exists(out): return
    w = None
    for i in (2, 3):
        pf = pq.ParquetFile(T + f'test_source{i}.parquet')
        for b in pf.iter_batches(batch_size=500_000):
            t = pa.Table.from_pandas(norm.prep(b.to_pandas()), preserve_index=False)
            w = w or pq.ParquetWriter(out + '.tmp', t.schema); w.write_table(t)
    w.close(); os.replace(out + '.tmp', out)

def _load_norm(path, cols=None): return pd.read_parquet(path, columns=cols)

def encode_test():
    """pool first (long pole), then anchor queries. Resumable. Same dir data/interim/test_pool/."""
    enc = make_encoder(); pool = test_raw_pool(); pdf = pd.DataFrame({'id': pool.column('entity_id').to_numpy(zero_copy_only=False), 'country': pool.column('country').to_numpy(zero_copy_only=False)})
    F.encode(pool=pdf, raw=pool, outdir=TEST_EMB, encoder=enc); del pool, pdf
    encode_queries(pq.read_table(T + 'test_source1.parquet'), TEST_EMB, 'anchors', encoder=enc)
    log('TEST ENCODE DONE')


# ----------------------------------------------------------------------------- val rehearsal / val tables (train data only)
def sample_val(n, seed):
    """n=0 -> all is_val anchors. else proportional stratified (country x has_match) sample, seed-deterministic."""
    sp = pd.read_parquet(norm.I + 'split.parquet'); v = sp[sp.is_val].reset_index(drop=True)
    if n and n < len(v):
        parts = [g.sample(int(round(n * len(g) / len(v))), random_state=seed) for _, g in v.groupby(['country', 'has_match'])]
        v = pd.concat(parts)
    return v.sample(frac=1, random_state=seed).reset_index(drop=True)

def run_val(n, seed, outdir, qdir, chunk=50_000):
    """val anchors vs the FULL train pool with the train embeddings (read-only). Returns (anchors, cands, metrics)."""
    t0 = time.time(); pick = sample_val(n, seed); log(f'val anchors {len(pick)} {pick.groupby(["country","has_match"]).size().to_dict()}')
    s1 = pq.read_table(norm.I + 'source1.parquet').to_pandas().set_index('entity_id').loc[pick.source1_entity_id].reset_index()
    assert (s1.entity_id.values == pick.source1_entity_id.values).all() and (s1.country.values == pick.country.values).all()
    an = norm.prep(s1); an['has_match'] = pick.has_match.values
    Q = encode_queries(s1[['entity_id', 'business_name', 'business_address']], qdir, 'val'); del s1
    pool = norm.load_pool(); log(f'pool loaded rss {cur_rss_gb():.1f}GB')
    c = candidates(an, pool, F.D, outdir, Q, chunk=chunk)
    m = eval_val(an, c, pool); m['wall_s'] = time.time() - t0; m['peak_rss_gb'] = rss_gb(); log('METRICS', json.dumps(m)); return an, c, m

def eval_val(an, c, pool):
    nA = len(an); gt = pd.read_parquet(norm.I + 'gt.parquet').set_index('source1_entity_id').matched_entity_ids.loc[an.id.values]
    e = pd.DataFrame({'a': np.arange(nA), 't': gt.str.split(',').values}).explode('t'); e = e[e.t.notna() & (e.t != '')]
    e['p'] = pd.Index(pool.id.values).get_indexer(e.t.values); assert (e.p >= 0).all()
    hit = np.isin(_keys(e.a.values, e.p.values), _keys(c.a.values, c.p.values)); e['hit'] = hit.astype(float)
    sz = np.bincount(c.a.values, minlength=nA); g = e.groupby('a').hit
    return dict(anchors=nA, true_pairs=len(e), pair_R=100 * hit.mean(), any_hit=100 * g.max().mean(), cluster_full=100 * g.min().mean(),
                cand_mean=float(sz.mean()), cand_max=int(sz.max()), cand_singleton_mean=float(sz[~an.has_match.values].mean()))

def run_test(outdir='data/processed/cands/test/', chunk=50_000):
    """FULL test candidate generation (inference only; no statistics on test)."""
    for f in ('test_pool_norm.parquet',): assert os.path.exists(T + f), 'run prep-test first'
    d = json.load(open(TEST_EMB + 'countries.json')); log('test countries in manifest:', d)
    an = norm.prep(pq.read_table(T + 'test_source1.parquet').to_pandas()); pool = _load_norm(T + 'test_pool_norm.parquet')
    Q = open_queries(TEST_EMB, 'anchors', len(an)); log(f'anchors {len(an)} pool {len(pool)} rss {cur_rss_gb():.1f}GB')
    c = candidates(an, pool, TEST_EMB, outdir, Q, chunk=chunk)
    write_candidate_pairs_tsv(c, an.id.values, pool.id.values, outdir + 'candidate_pairs.tsv'); log('TEST CANDS DONE', len(c))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('--n', type=int, default=50_000); ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--chunk', type=int, default=50_000); ap.add_argument('--tag', default=''); a = ap.parse_args()
    if a.stage == 'encode-test': encode_test()
    elif a.stage == 'prep-test': prep_test_pool()
    elif a.stage == 'rehearse': run_val(a.n, a.seed, f'data/processed/cands/rehearsal_{a.n}{a.tag}/', f'data/interim/val_q_{a.n}_{a.seed}/', a.chunk)
    elif a.stage == 'cands-val': run_val(a.n, a.seed, 'data/processed/cands/val/', f'data/interim/val_q_{a.n}_{a.seed}/', a.chunk)
    elif a.stage == 'cands-test': run_test(chunk=a.chunk)
