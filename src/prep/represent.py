"""Phase 3 representation build. Per-record, fits nothing; runs on train and test alike.
Usage: python src/prep/represent.py [--debug] [--split train|test|both]
Output: data/processed/prep/{split}_source{1,2,3}.parquet + manifest.json
name_char3grams: NOT materialised (12M+ lists of strings = several GB, pair-feature-only); use text.char3grams(name_norm) lazily.
"""
import argparse, hashlib, json, sys, time, multiprocessing as mp, importlib.metadata as md
import pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, '/home/kr95/Projects/Amazon-ML')
from src.prep import text as T

ROOT = '/home/kr95/Projects/Amazon-ML/'
CFG_PATH = ROOT + 'config/legal_suffixes.json'
OUTD = ROOT + 'data/processed/prep/'
CHUNK = 100_000
LS = pa.list_(pa.string())
SCHEMA = pa.schema([('entity_id', pa.string()), ('country', pa.string()), ('business_name', pa.string()),
                    ('business_address', pa.string()), ('name_norm', pa.string()), ('addr_norm', pa.string()),
                    ('name_script', pa.string()), ('addr_script', pa.string()), ('legal_suffix', LS),
                    ('name_core', pa.string()), ('name_tokens', LS), ('addr_tokens', LS),
                    ('name_translit', pa.string()), ('addr_translit', pa.string()), ('addr_pin', pa.string()),
                    ('addr_numbers', LS), ('addr_state_tok', pa.string())])
_TAB, _CFG = None, None


def process(rows):
    eid, name, addr, ctry = rows
    cols = {k: [] for k in SCHEMA.names}
    for i in range(len(eid)):
        nn, ns, nseq, ntok, ntr = T.field_tokens(name[i])
        an, as_, aseq, atok, atr = T.field_tokens(addr[i])
        leg, core = T.extract_suffix(nseq, _CFG, ctry[i])
        if not core: core = nn                       # never empty (fallback: whole normalised name)
        atok = [t for t in atok if t != 'null']      # NULL placeholder kept in addr_norm, dropped from token set
        pin, nums, st = T.addr_extras(addr[i], aseq, ctry[i])
        for k, v in (('entity_id', eid[i]), ('country', ctry[i]), ('business_name', name[i]), ('business_address', addr[i]),
                     ('name_norm', nn), ('addr_norm', an), ('name_script', ns), ('addr_script', as_), ('legal_suffix', leg),
                     ('name_core', core), ('name_tokens', ntok), ('addr_tokens', atok), ('name_translit', ntr),
                     ('addr_translit', atr), ('addr_pin', pin), ('addr_numbers', nums), ('addr_state_tok', st)):
            cols[k].append(v)
    return pa.table(cols, schema=SCHEMA)


def _chunk(rng):
    s, n = rng
    t = _TAB.slice(s, n)
    return process(tuple(t[c].to_pylist() for c in ('entity_id', 'business_name', 'business_address', 'country')))


def run(src, dst, nrows=None, workers=14):
    global _TAB
    t = pq.read_table(src, columns=['entity_id', 'business_name', 'business_address', 'country'])
    if nrows: t = t.slice(0, nrows)
    _TAB = t
    n = len(t)
    t0 = time.time()
    with mp.get_context('fork').Pool(workers) as p, pq.ParquetWriter(dst, SCHEMA, compression='zstd') as w:
        done = 0
        for i, tb in enumerate(p.imap(_chunk, [(s, CHUNK) for s in range(0, n, CHUNK)])):
            w.write_table(tb); done += len(tb)
            if i % 10 == 0:
                el = time.time() - t0
                print(f'  {dst.split("/")[-1]} {done}/{n} eta {el / done * (n - done):.0f}s', flush=True)
    assert done == n
    return n


def main():
    global _CFG
    ap = argparse.ArgumentParser(); ap.add_argument('--debug', action='store_true'); ap.add_argument('--split', default='both')
    a = ap.parse_args()
    _CFG = T.load_suffix_cfg(CFG_PATH)
    exp = {'train': (2206821, 5034616, 5285603), 'test': (1732544, 4887273, 5082316)}
    src = {'train': lambda k: ROOT + f'data/interim/source{k}.parquet', 'test': lambda k: ROOT + f'data/interim/test_source{k}.parquet'}
    splits = ['train', 'test'] if a.split == 'both' else [a.split]
    man = dict(versions={p: md.version(p) for p in ('polars', 'pyarrow', 'numpy', 'pandas')}, python=sys.version.split()[0],
               config_hash=hashlib.sha256(open(CFG_PATH, 'rb').read() + open(ROOT + 'src/prep/text.py', 'rb').read()).hexdigest()[:16],
               debug=a.debug, rows={})
    for sp in splits:
        for k in (1, 2, 3):
            t0 = time.time()
            dst = OUTD + (f'debug_{sp}_source{k}.parquet' if a.debug else f'{sp}_source{k}.parquet')
            n = run(src[sp](k), dst, 20000 if a.debug else None)
            if not a.debug: assert n == exp[sp][k - 1], (sp, k, n)
            man['rows'][f'{sp}_source{k}'] = n
            print(f'{sp} source{k}: {n} rows {time.time() - t0:.0f}s', flush=True)
    json.dump(man, open(OUTD + ('manifest_debug.json' if a.debug else 'manifest.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
