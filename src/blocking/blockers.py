"""Blocker interface: fit(pool: DataFrame) once; query(anchors: DataFrame) -> (a_pos, p_pos) int arrays
(positions into anchors / pool; pairs may repeat across blockers -> dedup in union).
All blockers apply the country hard-filter (true pairs agree 100%)."""
import numpy as np, pandas as pd

class Blocker:
    name = 'base'
    def fit(self, pool): self.pool = pool; return self
    def query(self, anchors): raise NotImplementedError

def _join(a_keys, p_keys, cap):
    """inner join on key columns of two Series; drop keys whose pool block > cap. returns pos arrays."""
    a = pd.DataFrame({'k': a_keys.values, 'a': np.arange(len(a_keys))})
    p = pd.DataFrame({'k': p_keys.values, 'p': np.arange(len(p_keys))})
    sz = p.k.map(p.k.value_counts())
    p = p[sz <= cap] if cap else p
    m = a.merge(p, on='k')
    return m.a.values, m.p.values

class Prefix(Blocker):
    def __init__(self, n=4, cap=None): self.n, self.cap, self.name = n, cap, f'prefix{n}'
    def fit(self, pool):
        self.pool = pool; self.pk = pool.country + '|' + pool.nn.str[:self.n]; return self
    def query(self, anchors):
        ak = anchors.country + '|' + anchors.nn.str[:self.n]
        return _join(ak, self.pk.reset_index(drop=True), self.cap)

class Token(Blocker):
    """any shared normalized name token (len>=2), tokens with pool df > cap ignored."""
    def __init__(self, cap=5000): self.cap, self.name = cap, f'token_cap{cap}'
    def fit(self, pool):
        self.pool = pool
        t = pool.nn.str.split().explode()
        t = t[t.str.len() >= 2]
        self.p = pd.DataFrame({'k': pool.country.values[t.index.values] + '|' + t.values, 'p': t.index.values})
        vc = self.p.k.value_counts(); self.p = self.p[self.p.k.map(vc) <= self.cap]; return self
    def query(self, anchors):
        t = anchors.nn.str.split().explode(); t = t[t.str.len() >= 2]
        pos = anchors.reset_index(drop=True).index
        a = pd.DataFrame({'k': anchors.country.values[t.index.values] + '|' + t.values, 'a': t.index.values})
        m = a.merge(self.p, on='k').drop_duplicates(['a', 'p'])
        return m.a.values, m.p.values

class AddrNum(Blocker):
    """(country, address number token, first 2 chars of name): cheap key for address-number overlap."""
    def __init__(self, cap=3000, name_chars=2): self.cap, self.nc, self.name = cap, name_chars, f'addrnum_c{name_chars}_cap{cap}'
    def _keys(self, d):
        t = d.nums.str.split().explode().dropna(); t = t[t != '']
        k = d.country.values[t.index.values] + '|' + t.values
        if self.nc: k = k + '|' + d.nn.str[:self.nc].values[t.index.values]
        return pd.DataFrame({'k': k, 'i': t.index.values})
    def fit(self, pool):
        self.pool = pool; p = self._keys(pool); vc = p.k.value_counts(); self.p = p[p.k.map(vc) <= self.cap]; return self
    def query(self, anchors):
        m = self._keys(anchors).merge(self.p, on='k', suffixes=('_a', '_p')).drop_duplicates(['i_a', 'i_p'])
        return m.i_a.values, m.i_p.values

class Union(Blocker):
    def __init__(self, *bs): self.bs, self.name = bs, '+'.join(b.name for b in bs)
    def fit(self, pool):
        for b in self.bs: b.fit(pool)
        return self
    def query(self, anchors):
        r = [b.query(anchors) for b in self.bs]
        a = np.concatenate([x[0] for x in r]); p = np.concatenate([x[1] for x in r])
        u = np.unique(a.astype(np.int64) * (1 << 32) + p)
        return u >> 32, u & ((1 << 32) - 1)

class EmbANN(Blocker):
    """Dense blocker: sentence-transformers e5 model, per-country exact top-k inner-product search (GPU matmul, chunked).
    Requires a `text` column in pool and anchors (norm.embed_text). Countries are derived from the data.
    e5 prefixes: pool -> 'passage: ', anchors -> 'query: '. fp16, embeddings L2-normalized, kept as fp16 on CPU."""
    def __init__(self, model='intfloat/multilingual-e5-small', topk=50, bs=512, max_len=64, cache=None, device='cuda'):
        self.model_name, self.k, self.bs, self.max_len, self.cache, self.device = model, topk, bs, max_len, cache, device
        self.name = f"emb_{model.split('/')[-1]}_top{topk}"; self.stats = {}
    def _model(self):
        if not hasattr(self, 'm'):
            from sentence_transformers import SentenceTransformer
            self.m = SentenceTransformer(self.model_name, device=self.device, model_kwargs={'torch_dtype': 'float16'})
            self.m.max_seq_length = self.max_len
        return self.m
    def encode(self, texts, prefix):
        import time, torch
        txt = np.asarray([prefix + t for t in texts], dtype=object)
        order = np.argsort([len(t) for t in txt], kind='stable')  # length-sorted batches
        m = self._model(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        e = m.encode(list(txt[order]), batch_size=self.bs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        dt = time.time() - t0
        out = np.empty_like(e, dtype=np.float16); out[order] = e.astype(np.float16)
        self.stats[prefix.strip()] = dict(n=len(txt), sec=dt, rec_per_s=len(txt) / dt, peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9)
        return out
    def fit(self, pool):
        self.pool = pool.reset_index(drop=True)
        self.E = self.encode(self.pool.text.values, 'passage: ')
        self.cidx = {c: np.flatnonzero(self.pool.country.values == c) for c in self.pool.country.unique()}
        return self
    def query(self, anchors, chunk=4096, pchunk=1_000_000):
        import torch
        anchors = anchors.reset_index(drop=True)
        Q = self.encode(anchors.text.values, 'query: ')
        A, P = [], []
        for c, ai in ((c, np.flatnonzero(anchors.country.values == c)) for c in anchors.country.unique()):
            pi = self.cidx.get(c)
            if pi is None or len(pi) == 0: continue
            k = min(self.k, len(pi))
            for s in range(0, len(ai), chunk):
                a = ai[s:s + chunk]; q = torch.from_numpy(Q[a]).to(self.device)
                bv = bi = None
                for ps in range(0, len(pi), pchunk):
                    sub = pi[ps:ps + pchunk]
                    sc = q @ torch.from_numpy(self.E[sub]).to(self.device).T
                    v, i = sc.topk(min(k, len(sub)), dim=1); i = i + ps
                    if bv is None: bv, bi = v, i
                    else:
                        v2, j = torch.cat([bv, v], 1).topk(k, dim=1); bi = torch.cat([bi, i], 1).gather(1, j); bv = v2
                bi = bi.cpu().numpy()
                A.append(np.repeat(a, bi.shape[1])); P.append(pi[bi.ravel()])
        return np.concatenate(A), np.concatenate(P)
