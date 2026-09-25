import sys, pathlib, zlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, pyarrow as pa, pytest
from src.blocking import candidates as C, full_pool as F, norm


def fake_encoder(texts):   # deterministic char-trigram hashing bag, L2-normalised (similar strings -> similar vectors)
    out = np.zeros((len(texts), F.DIM), np.float32)
    for i, t in enumerate(texts):
        for j in range(len(t) - 2): out[i, zlib.crc32(t[j:j + 3].encode()) % F.DIM] += 1
        out[i, 0] += 1e-3
    return out / np.linalg.norm(out, axis=1, keepdims=True)


@pytest.fixture(autouse=True)
def cpu(monkeypatch): monkeypatch.setattr(F, '_dev', lambda: 'cpu')


def raw_df(prefix, rows):
    return pd.DataFrame({'entity_id': [f'{prefix}-{i}' for i in range(len(rows))], 'business_name': [r[0] for r in rows],
                         'business_address': [r[1] for r in rows], 'country': [r[2] for r in rows]})


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('w'); rng = np.random.default_rng(0)
    rows = []
    for i in range(300): rows.append((f'cafe {i % 40} corner{rng.integers(50)}', f'{i} main street {i % 17}', 'B'))       # big country: lex blocks saturate
    for i in range(60): rows.append((f'shop {i} alpha', f'{i + 1} road', 'A'))
    for i in range(10): rows.append((f'tiny {i}', f'{i} lane', 'C'))
    pool_raw = raw_df('S2', rows); pool_raw['entity_id'] = [('S2-' if i % 2 else 'S3-') + str(i) for i in range(len(pool_raw))]
    pool = norm.prep(pool_raw)
    an_rows = [(f'cafe {i} corner{i}', f'{i} main street', 'B') for i in range(0, 40)] + [(f'shop {i} alpha', f'{i + 1} road', 'A') for i in range(15)] + \
              [(f'tiny {i}', f'{i} lane', 'C') for i in range(3)] + [('lost anchor', '1 nowhere', 'Z'), ('lost two', '2 nowhere', 'Z')]
    an_raw = raw_df('S1', an_rows); an = norm.prep(an_raw)
    emb = str(tmp / 'emb') + '/'
    F.encode(chunk=64, pool=pool, raw=pa.Table.from_pandas(pool_raw[['entity_id', 'business_name', 'business_address']], preserve_index=False), encoder=fake_encoder, outdir=emb)
    qdir = str(tmp / 'q') + '/'; Q = C.encode_queries(an_raw, qdir, 'a', encoder=fake_encoder, chunk=25)
    return dict(tmp=tmp, pool=pool, an=an, emb=emb, Q=np.asarray(Q))


def run(w, name, chunk=8, **kw):
    return C.candidates(w['an'], w['pool'], w['emb'], str(w['tmp'] / name) + '/', w['Q'], chunk=chunk, qchunk=16, **kw)


def canon(c): return c.sort_values(['a', 'p']).reset_index(drop=True)


def test_cap_dupes_country(world):
    c = run(world, 'r1'); w = world
    assert len(c) and not c.duplicated(['a', 'p']).any()
    sz = np.bincount(c.a.values, minlength=len(w['an']))
    assert sz.max() <= 150 and sz[:40].max() > 100                       # cap binding-ish for the big country
    assert (w['pool'].country.values[c.p.values] == w['an'].country.values[c.a.values]).all()   # no cross-country
    assert (sz[-2:] == 0).all() and sz[-2:].sum() == 0                    # absent-country anchors: empty, no crash
    assert sz[40:55].max() <= 60 and sz[55:58].max() <= 10                # bounded by the country's pool size
    assert np.isfinite(c.dense_score).all() and set(c.src.unique()) <= {1, 2, 4}
    assert (c.groupby('a')['rank'].apply(lambda r: (np.sort(r.values) == np.arange(len(r))).all())).all()


def test_resume_identical(world, monkeypatch):
    clean = canon(run(world, 'clean')); calls = {'n': 0}; orig = C._select
    def boom(*a, **k):
        calls['n'] += 1
        if calls['n'] == 3: raise RuntimeError('simulated crash')
        return orig(*a, **k)
    monkeypatch.setattr(C, '_select', boom)
    with pytest.raises(RuntimeError): run(world, 'crash')
    monkeypatch.setattr(C, '_select', orig)
    import os; done = [f for f in os.listdir(str(world['tmp'] / 'crash')) if f.startswith('cands_')]; assert 0 < len(done) < 8
    pd.testing.assert_frame_equal(canon(run(world, 'crash')), clean)


def test_resume_refuses_other_run(world):
    run(world, 'r2')
    an2 = world['an'].iloc[:-1]
    with pytest.raises(AssertionError): C.candidates(an2, world['pool'], world['emb'], str(world['tmp'] / 'r2') + '/', world['Q'][:-1], chunk=8, qchunk=16)


def test_tsv_format(world):
    c = run(world, 'r3'); an, pool = world['an'], world['pool']; p = str(world['tmp'] / 'cp.tsv')
    C.write_candidate_pairs_tsv(c, an.id.values, pool.id.values, p)
    lines = open(p).read().split('\n'); assert lines[-1] == '' and lines[0] == 'source1_entity_id\tcandidate_entity_ids'; lines = lines[1:-1]
    assert len(lines) == len(an)
    for ln, aid in zip(lines, an.id.values):
        f = ln.split('\t'); assert len(f) == 2 and f[0] == aid
        ids = f[1].split(',') if f[1] else []; assert len(ids) == len(set(ids)) and all(i.startswith(('S2-', 'S3-')) for i in ids) and len(ids) <= 150
    assert lines[-1].endswith('\t') and lines[-2].endswith('\t')          # empty when no candidates


def test_gpu_resident_matches_cpu_and_block_size_invariant(world, monkeypatch):
    cpu_c = canon(run(world, 'g_cpu'))
    same_blocks = canon(run(world, 'g_cpu2', sub=5))
    k = lambda d: set(zip(d.a.values.tolist(), d.p.values.tolist()))
    assert len(k(same_blocks) & k(cpu_c)) / len(k(cpu_c)) > 0.99      # block size only changes tie order at the k-th dense boundary (synthetic data has exact duplicate vectors)
    import torch
    if not torch.cuda.is_available(): pytest.skip('no cuda')
    monkeypatch.setattr(F, '_dev', lambda: 'cuda')
    gpu_c = canon(run(world, 'g_gpu'))
    assert len(k(gpu_c) & k(cpu_c)) / len(k(cpu_c)) > 0.98                                             # fp16 vs fp32 matmul only perturbs near-ties
    assert (np.bincount(gpu_c.a.values) <= 150).all()
