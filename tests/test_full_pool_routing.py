import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, pyarrow as pa, pytest
from src.blocking import full_pool as fp


def fake_encoder(texts):
    import zlib
    e = np.stack([np.random.default_rng(zlib.crc32(t.encode())).normal(size=fp.DIM) for t in texts]).astype(np.float32)   # deterministic per text
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def mk(countries, prefix="S2"):
    n = len(countries)
    pool = pd.DataFrame({"id": [f"{prefix}-{i}" for i in range(n)], "country": countries})
    raw = pa.table({"entity_id": pool.id.values, "business_name": [f"n{i}" for i in range(n)], "business_address": [f"a{i}" for i in range(n)]})
    return pool, raw


def enc(tmp_path, countries, sub="o", **kw):
    out = str(tmp_path / sub) + "/"; pool, raw = mk(countries)
    fp.encode(chunk=2, pool=pool, raw=raw, encoder=fake_encoder, outdir=out, **kw)
    return out, pool


def emb_of(out, c):
    pos, mm = fp.open_country(c, outdir=out); return pos, np.asarray(mm, dtype=np.float32)


def test_encode_arbitrary_pool_and_missing_country(tmp_path):
    out, _ = enc(tmp_path, ["India", "US", "Japan", "India", "Japan", "US"])
    for c in ("India", "US", "Japan"):
        pos, mm = fp.open_country(c, outdir=out); assert len(pos) == 2 and mm.shape == (2, fp.DIM)
    assert len(fp.open_country("  japan ", outdir=out)[0]) == 2          # case/whitespace normalisation
    assert fp.open_country("France", outdir=out, missing_ok=True) == (None, None)
    with pytest.raises(FileNotFoundError):
        fp.open_country("France", outdir=out)


def test_label_variants_route_together_and_distinct_stay_apart(tmp_path):
    out, pool = enc(tmp_path, ["India", "india ", "INDIA", "New-York", "New York", "US", "U.S.", "US"])
    assert len(fp.open_country("India", outdir=out)[0]) == 3            # one file, not three
    assert len(fp.open_country("  iNdIa", outdir=out)[0]) == 3
    a, b = fp.open_country("New-York", outdir=out)[0], fp.open_country("New York", outdir=out)[0]
    assert len(a) == 1 and len(b) == 1 and a[0] != b[0]                 # same cslug, still separate countries
    assert len(fp.open_country("US", outdir=out)[0]) == 2 and len(fp.open_country("U.S.", outdir=out)[0]) == 1
    slugs = fp._assign_slugs(pool.country.values); assert len({v.casefold() for v in slugs.values()}) == len(slugs) == 5


def test_encode_refuses_train_dir_and_missing_outdir(tmp_path):
    pool, raw = mk(["India", "US"])
    with pytest.raises(AssertionError):
        fp.encode(pool=pool, raw=raw, encoder=fake_encoder)                       # no outdir
    with pytest.raises(AssertionError):
        fp.encode(pool=pool, raw=raw, encoder=fake_encoder, outdir=fp.D)          # train dir
    with pytest.raises(AssertionError):
        fp.encode(pool=pool, raw=raw, encoder=fake_encoder, outdir=fp.D.rstrip("/") + "/../full_pool/")


def test_progress_tied_to_pool(tmp_path):
    out, _ = enc(tmp_path, ["India", "US", "India", "US"])
    pool2, raw2 = mk(["India", "US", "India", "US"], prefix="T2")               # different ids, same shape
    with pytest.raises(AssertionError):
        fp.encode(chunk=2, pool=pool2, raw=raw2, encoder=fake_encoder, outdir=out)


def _queries(out, c, n=2):
    pos, e = emb_of(out, c); return e[:n].astype(np.float16), pos


def test_search_absent_country_all_minus1_no_cross_country(tmp_path):
    out, pool = enc(tmp_path, ["India"] * 5 + ["US"] * 5 + ["India"])
    Qi, _ = _queries(out, "India"); Qu, _ = _queries(out, "US")
    Q = np.concatenate([Qi, Qu, Qi[:1]])
    ctry = np.array(["India", "India", "US", "US", "France"], dtype=object)
    cp, cs, ep = fp.dense_search(Q, ctry, outdir=out)
    assert (cp[4] == -1).all() and np.isneginf(cs[4]).all() and (ep[4] == -1).all()
    assert ((cp >= 0) == np.isfinite(cs)).all()
    pc = pool.country.values
    for i in range(4):
        v = cp[i][cp[i] >= 0]; e = ep[i][ep[i] >= 0]
        assert len(v) > 0 and (pc[v] == ctry[i]).all() and (pc[e] == ctry[i]).all() and len(set(v)) == len(v)
    t = fp.dense_table(cp, cs); assert (t.p >= 0).all() and not t.duplicated(["a", "p"]).any() and 4 not in set(t.a)


def test_search_tiny_country_below_k(tmp_path):
    out, pool = enc(tmp_path, ["India"] * 3 + ["US"] * 300)                     # India 3 rows < K=100, < nb
    Q, _ = _queries(out, "India", 2)
    cp, cs, ep = fp.dense_search(Q, np.array(["India", "india "], dtype=object), outdir=out)
    assert cp.shape == (2, fp.K)
    for i in range(2):
        v = cp[i][cp[i] >= 0]; assert sorted(v) == [0, 1, 2] and (cp[i][3:] == -1).all() and np.isneginf(cs[i][3:]).all()
        assert cp[i, 0] == i                                                      # self is nearest
    Q1, _ = _queries(out, "India", 1)
    one, _ = enc(tmp_path, ["India"], sub="one"); Qo, _ = _queries(one, "India", 1)  # 1-row pool
    cp, cs, ep = fp.dense_search(Qo, np.array(["India"], dtype=object), outdir=one)
    assert cp[0, 0] == 0 and (cp[0, 1:] == -1).all() and (ep == -1).all()


def test_dense_scores_absent_country_no_raise(tmp_path):
    out, _ = enc(tmp_path, ["India", "India", "US", "US"])
    Q = np.zeros((2, fp.DIM), np.float16); Q[:, 0] = 1
    old = fp.D; fp.D = out
    try:
        s = fp._dense_scores(np.array([0, 1]), np.array([0, 1]), Q, None, np.array(["India", "France"], dtype=object), {})
    finally: fp.D = old
    assert np.isfinite(s).all() and s[1] == -1.0


def test_search_matches_bruteforce_multi_chunk(tmp_path):
    """equivalence with exact brute force (the path validated on US/India) incl. multiple pchunk blocks."""
    out, pool = enc(tmp_path, ["India"] * 40 + ["US"] * 60)
    pos, e = emb_of(out, "US"); Q = e[:5].astype(np.float16)
    import torch
    r, v = fp.topk_stream(torch.from_numpy(Q), np.asarray(fp.open_country("US", outdir=out)[1]), k=10, pchunk=7)
    ref = np.argsort(-(Q.astype(np.float32) @ e.T), axis=1, kind="stable")[:, :10]
    assert (np.sort(r, 1) == np.sort(ref, 1)).all()
