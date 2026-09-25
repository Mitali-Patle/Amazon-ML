"""Check 5 (TRAIN only): does the transliterated token space raise name-token overlap recall for true pairs?
Val anchors (split.parquet is_val & has_match), 3000 sampled (seed 0), all their true targets.
recall_any = share >=1 name token; recall_rare = share >=1 name token with df<=200 in the same-country S2+S3 pool.
before = frozen norm.nname tokens (no translit); after = prep name_tokens (translit unioned). Report only."""
import sys, numpy as np, pandas as pd, polars as pl
sys.path.insert(0, '/home/kr95/Projects/Amazon-ML')
from src.blocking.norm import nname
R = '/home/kr95/Projects/Amazon-ML/data/'
RARES = (200, 3000)
sp = pd.read_parquet(R + 'interim/split.parquet')
a = sp[sp.is_val & sp.has_match].sample(3000, random_state=0)
gt = pd.read_parquet(R + 'interim/gt.parquet').merge(a[['source1_entity_id']], on='source1_entity_id')
pairs = gt.assign(t=gt.matched_entity_ids.str.split(',')).explode('t')[['source1_entity_id', 't']]
pairs['t'] = pairs.t.str.strip(); print('anchors', len(a), 'pairs', len(pairs))

s1 = pl.read_parquet(R + 'processed/prep/train_source1.parquet', columns=['entity_id', 'country', 'business_name', 'name_tokens', 'name_script'])
s1 = s1.filter(pl.col('entity_id').is_in(a.source1_entity_id.tolist())).to_pandas()
pool = pl.concat([pl.read_parquet(R + f'processed/prep/train_source{k}.parquet', columns=['entity_id', 'country', 'name_tokens', 'name_script']) for k in (2, 3)])
pn = pl.read_parquet(R + 'interim/pool_norm.parquet', columns=['id', 'country', 'nn'])
assert pn.height == pool.height and (pn['id'] == pool['entity_id']).all(), 'pool row alignment'
pool = pool.with_columns(pn['nn'].str.split(' ').alias('old_tokens'))
tg = pool.filter(pl.col('entity_id').is_in(pairs.t.unique().tolist())).to_pandas()

s1['old_tokens'] = [x.split() for x in nname(s1.business_name)]
A = s1.set_index('entity_id'); Tt = tg.set_index('entity_id')
pairs['country'] = pairs.source1_entity_id.map(A.country)
assert (pairs.country == pairs.t.map(Tt.country)).all()

def df_of(col, toks):
    q = pl.DataFrame({'tok': sorted(toks)})
    d = (pool.select('country', pl.col(col).alias('tok')).explode('tok').drop_nulls()
         .filter(pl.col('tok').is_in(q['tok'].implode())).group_by('country', 'tok').len().to_pandas())
    return {(c, t): n for c, t, n in zip(d.country, d.tok, d.len)}

res = {}
for col in ('old_tokens', 'name_tokens'):
    toks = set(t for x in A[col] for t in x) | set(t for x in Tt[col] for t in x)
    dfm = df_of(col, toks)
    any_, rare = [], {r: [] for r in RARES}
    for s, t, c in zip(pairs.source1_entity_id, pairs.t, pairs.country):
        sh = set(A.at[s, col]) & set(Tt.at[t, col])
        any_.append(bool(sh))
        for r in RARES: rare[r].append(any(dfm.get((c, x), 0) <= r for x in sh))
    res[col] = (np.array(any_), {r: np.array(v) for r, v in rare.items()})
pairs['tscript'] = pairs.t.map(Tt.name_script); pairs['ascript'] = pairs.source1_entity_id.map(A.name_script)
pairs['grp'] = np.where((pairs.tscript == 'latin') & (pairs.ascript == 'latin'), 'latin-latin', 'non-latin(either side)')
out = []
for name, mask in [('ALL', np.ones(len(pairs), bool))] + [(g, (pairs.grp == g).values) for g in pairs.grp.unique()] + \
        [('target=' + s, (pairs.tscript == s).values) for s in sorted(pairs.tscript.unique())]:
    r = dict(group=name, n_pairs=int(mask.sum()))
    for col, lab in (('old_tokens', 'before'), ('name_tokens', 'after')):
        r[f'any_{lab}'] = round(res[col][0][mask].mean() * 100, 2)
        for q in RARES: r[f'rare{q}_{lab}'] = round(res[col][1][q][mask].mean() * 100, 2)
    out.append(r)
print(pd.DataFrame(out).to_string(index=False))
